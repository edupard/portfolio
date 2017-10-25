import threading
import queue
import csv
from enum import Enum
import os
import pandas as pd
import numpy as np

from tiingo.tiingo import get_historical_data
from stock_data.config import get_config
from utils.folders import create_dir, remove_dir


class MessageType(Enum):
    STOP = 0
    TASK = 1


class Message:
    def __init__(self, message_type, payload=None):
        self.message_type = message_type
        self.payload = payload


def download_data(tickers, num_workers=1):
    manager = DownloadManager(tickers, num_workers)
    manager.start()
    manager.wait()


class DownloadManager:
    def __init__(self, tickers, num_workers):
        self.tickers = tickers
        self.num_workers = num_workers
        self.queue = queue.Queue()
        thread_func = lambda: self._thread_func()
        self.thread = threading.Thread(target=(thread_func))

    def start(self):
        self.thread.start()

    def _thread_func(self):
        # copy tickers list
        to_process = []
        for item in self.tickers: to_process.append(item)

        # create workers pool
        avail_workers = []
        for _ in range(self.num_workers):
            worker = DownloadWorker(self)
            worker.start()
            avail_workers.append(worker)

        while len(to_process) > 0:
            ticker = to_process.pop(0)
            if len(avail_workers) == 0:
                # wait for some worker to became free
                worker = self.queue.get()
                avail_workers.append(worker)
            worker = avail_workers.pop(0)
            worker.download_data(ticker)

        # wait for all workers to complete tasks
        while len(avail_workers) != self.num_workers:
            worker = self.queue.get()
            avail_workers.append(worker)

        # stop all workers
        for worker in avail_workers:
            worker.stop()

    def notify(self, worker):
        self.queue.put(worker)

    def wait(self):
        self.thread.join()


class DownloadWorker:
    def __init__(self, manager):
        self.queue = queue.Queue()
        self.manager = manager
        thread_func = lambda: self._thread_func()
        self.thread = threading.Thread(target=(thread_func))

    def start(self):
        self.thread.start()

    def download_data(self, ticker):
        self.queue.put(Message(MessageType.TASK, ticker))

    def stop(self):
        self.queue.put(Message(MessageType.STOP))

    def _thread_func(self):
        while True:
            m = self.queue.get()
            if m.message_type == MessageType.STOP:
                break
            elif m.message_type == MessageType.TASK:
                ticker = m.payload
                try:
                    dir_name = get_config().get_dir_name(ticker)
                    file_name = get_config().get_data_file_name(ticker)
                    dump_file_name = get_config().get_dump_file_name(ticker)

                    print('dowloading prices for %s' % ticker)
                    json = get_historical_data(ticker, get_config().DATA_BEG, get_config().DATA_END)
                    if json is None or len(json) == 0:
                        print('no data for %s' % ticker)
                        continue
                    else:
                        print('writing prices for %s' % ticker)
                        create_dir(dir_name)

                        with open(file_name, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                ('date', 'close', 'high', 'low', 'open', 'volume', 'adjClose', 'adjHigh', 'adjLow',
                                 'adjOpen', 'adjVolume', 'divCash', 'splitFactor'))
                            for d in json:
                                o = d['open']
                                c = d['close']
                                h = d['high']
                                l = d['low']
                                v = d['volume']
                                adj_o = d['adjOpen']
                                adj_c = d['adjClose']
                                adj_h = d['adjHigh']
                                adj_l = d['adjLow']
                                adj_v = d['adjVolume']
                                div_cash = d['divCash']
                                split_factor = d['splitFactor']
                                date = d['date'].split("T")[0]
                                writer.writerow(
                                    (date, c, h, l, o, v, adj_c, adj_h, adj_l, adj_o, adj_v, div_cash, split_factor))
                        print('prices for %s stored' % ticker)




                    print('preprocessing prices for %s' % ticker)
                    df = pd.read_csv(file_name)
                    df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
                    df.date = df.date.apply(lambda x: x.timestamp()).astype(float)
                    df = df.sort_values(['date'], ascending=[True])

                    np.savez(dump_file_name,
                             ts=df.date.values,
                             o=df.open.values,
                             h=df.high.values,
                             l=df.low.values,
                             c=df.close.values,
                             v=df.volume.values,
                             a_o=df.adjOpen.values,
                             a_h=df.adjHigh.values,
                             a_l=df.adjLow.values,
                             a_c=df.adjClose.values,
                             a_v=df.adjVolume.values,
                             )
                except:
                    print('download failed for %s' % ticker)
                    remove_dir(dir_name)
                    continue
                finally:
                    self.manager.notify(self)
