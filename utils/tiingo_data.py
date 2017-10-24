import csv
import datetime
import threading
import queue
from enum import Enum
import numpy as np

from tiingo.tiingo import get_historical_data

class PayloadType(Enum):
    DATA = 0
    TASK_COMPLETED = 1


class Payload:
    def __init__(self, payload_type, ticker, payload):
        self.payload_type = payload_type
        self.ticker = ticker
        self.payload = payload


class Writer:
    def __init__(self, FILE_NAME, NUM_WORKERS):
        thread_func = lambda: self.task()
        self.thread = threading.Thread(target=(thread_func))
        self.queue = queue.Queue()
        self.tasks_completed = 0
        self.FILE_NAME = FILE_NAME
        self.NUM_WORKERS = NUM_WORKERS

    def task(self):
        print('downloading data...')
        with open(self.FILE_NAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('ticker', 'date', 'o', 'c', 'h', 'l', 'v', 'adj_o', 'adj_c', 'adj_h', 'adj_l', 'adj_v',
                             'div', 'split'))
            while True:
                p = self.queue.get()
                if p.payload_type == PayloadType.TASK_COMPLETED:
                    self.tasks_completed += 1
                    if self.tasks_completed == self.NUM_WORKERS:
                        break
                elif p.payload_type == PayloadType.DATA:
                    try:
                        for d in p.payload:
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
                            dt = d['date'].split("T")[0]
                            t = p.ticker
                            writer.writerow(
                                (t, dt, o, c, h, l, v, adj_o, adj_c, adj_h, adj_l, adj_v, div_cash, split_factor))
                    except:
                        pass
        print('download completed!')

    def start(self):
        self.thread.start()

    def join(self):
        self.thread.join()

    def write_data(self, ticker, data):
        self.queue.put(Payload(PayloadType.DATA, ticker, data))

    def task_completed(self):
        self.queue.put(Payload(PayloadType.TASK_COMPLETED, None, None))


class Worker:
    def __init__(self, writer, tickers, idx, START_DATE, END_DATE):
        self.idx = idx
        self.writer = writer
        self.tickers = tickers
        thread_func = lambda: self.task()
        self.thread = threading.Thread(target=(thread_func))
        self.START_DATE = START_DATE
        self.END_DATE = END_DATE

    def start(self):
        self.thread.start()

    def task(self):
        for ticker in self.tickers:
            data = get_historical_data(ticker, self.START_DATE, self.END_DATE)
            if data is not None:
                self.writer.write_data(ticker, data)
        self.writer.task_completed()


def download_data(tickers, FILE_NAME, START_DATE, END_DATE, NUM_WORKERS=20):
    writer = Writer(FILE_NAME, NUM_WORKERS)
    writer.start()

    tickers_per_worker = len(tickers) // NUM_WORKERS + 1
    for idx in range(NUM_WORKERS):
        tickers_slice = tickers[idx * tickers_per_worker: min((idx + 1) * tickers_per_worker, len(tickers))]
        worker = Worker(writer, tickers_slice, idx, START_DATE, END_DATE)
        worker.start()

    writer.join()


def preprocess_data(tickers, FILE_NAME, START_DATE, END_DATE, DUMP_FILE_NAME, features):
    print('preprocessing data...')

    ticker_to_idx = {}
    idx = 0
    for ticker in tickers:
        ticker_to_idx[ticker] = idx
        idx += 1

    num_tickers = len(tickers)
    days = (END_DATE - START_DATE).days
    data_points = days + 1

    idx_map = {
        'o': 0,
        'c': 1,
        'h': 2,
        'l': 3,
        'v': 4,
        'a_o': 5,
        'a_c': 6,
        'a_h': 7,
        'a_l': 8,
        'a_v': 9,
        'to': 10
    }

    features_len = len(features)
    raw_data = np.zeros((num_tickers, data_points, features_len))
    idx_arr = np.zeros((features_len), dtype=np.int)
    for i in range(features_len):
        idx_arr[i] = idx_map.get(features[i])
    data_arr = np.zeros(len(idx_map))

    raw_dt = np.zeros((data_points))
    for idx in range(data_points):
        date = START_DATE + datetime.timedelta(days=idx)
        # convert date to datetime
        dt = datetime.datetime.combine(date, datetime.time.min)
        raw_dt[idx] = dt.timestamp()

    num_lines = sum(1 for line in open(FILE_NAME))
    line = 0
    curr_progress = 0
    # with open(FILE_NAME, 'r') as csv_file, open('data/snp/errors.csv', 'w') as csv_error_file:
    with open(FILE_NAME, 'r') as csv_file:
        reader = csv.reader(csv_file)
        # writer = csv.writer(csv_error_file)
        for row in reader:
            line += 1
            if line == 1:
                continue

            progress = line // (num_lines // 10)
            if progress != curr_progress:
                print('.', sep=' ', end='', flush=True)
                curr_progress = progress

            ticker = row[0]
            if ticker not in ticker_to_idx:
                continue
            ticker_idx = ticker_to_idx[ticker]
            dt = datetime.datetime.strptime(row[1], '%Y-%m-%d').date()
            if dt < START_DATE or dt > END_DATE:
                continue
            dt_idx = (dt - START_DATE).days
            try:
                o = float(row[2])
                c = float(row[3])
                h = float(row[4])
                l = float(row[5])
                v = float(row[6])
                a_o = float(row[7])
                a_c = float(row[8])
                a_h = float(row[9])
                a_l = float(row[10])
                a_v = float(row[11])
                # d_c = float(row[12])
                # s_f = float(row[13])
                to = v * c

                data_arr[0] = o
                data_arr[1] = c
                data_arr[2] = h
                data_arr[3] = l
                data_arr[4] = v
                data_arr[5] = a_o
                data_arr[6] = a_c
                data_arr[7] = a_h
                data_arr[8] = a_l
                data_arr[9] = a_v
                data_arr[10] = to

                for i in range(features_len):
                    raw_data[ticker_idx, dt_idx, i] = data_arr[idx_arr[i]]
            except:
                # writer.writerow(row)
                pass

    np_tickers = np.array(tickers, dtype=np.object)
    print('')
    print('saving file...')
    np.savez(DUMP_FILE_NAME, raw_tickers=np_tickers, raw_dt=raw_dt, raw_data=raw_data)
    print('preprocessing completed!')


def load_npz_data(DUMP_FILE_NAME):
    input = np.load(DUMP_FILE_NAME)
    raw_tickers = input['raw_tickers']
    raw_dt = input['raw_dt']
    raw_data = input['raw_data']
    return raw_tickers, raw_dt, raw_data
