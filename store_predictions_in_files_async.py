import snp.snp as snp
import net.train_config as train_config
import net.eval_config as eval_config
import utils.folders as folders
import os
import pandas as pd
import csv
import timeit
import threading
import queue

train_config = train_config.get_train_config_petri()
eval_config = eval_config.get_eval_config_petri_whole_set()

BASE_FOLDER = train_config.DATA_FOLDER

EPOCH = 600

mutex = threading.Lock()


class FileWrapper():
    def __init__(self, file_path):
        self.file = open(file_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(("ticker", "weak_predictor", "prediction"))

    def batch_write(self, rows_arr):
        self.writer.writerows(rows_arr)

    def close(self):
        self.file.close()


class Reader():
    def __init__(self, writer):
        self.writer = writer
        self.queue = queue.Queue()
        _thread_func = lambda: self.thread_func()
        self.thread = threading.Thread(target=(_thread_func))
        self.thread.start()

    def read(self, weak_predictor):
        self.queue.put(weak_predictor)

    def stop(self):
        self.queue.put(None)

    def thread_func(self):
        while True:
            weak_predictor = self.queue.get()
            rows_dict = {}

            if weak_predictor is None:
                self.writer.write_data(self, rows_dict, weak_predictor)
                return

            start = timeit.default_timer()
            for ticker in snp.get_snp_hitorical_components_tickers():
                mutex.acquire()
                try:
                    train_config.DATA_FOLDER = "%s/%s" % (BASE_FOLDER, weak_predictor)
                    _, prediction_path = folders.get_prediction_path(train_config, eval_config, ticker, EPOCH)
                finally:
                    mutex.release()

                if os.path.exists(prediction_path):
                    pred_df = pd.read_csv(prediction_path)

                    for index, row in pred_df.iterrows():
                        s_date = row.date
                        prediction = row.prediction
                        if s_date in rows_dict:
                            rows_arr = rows_dict[s_date]
                        else:
                            rows_arr = []
                            rows_dict[s_date] = rows_arr
                        rows_arr.append((ticker, weak_predictor, prediction))
                # # debug
                # if ticker != "TEG":
                #     break
            stop = timeit.default_timer()
            print("Read %s predictions for %.2fs" % (weak_predictor, stop - start))
            self.writer.write_data(self, rows_dict, weak_predictor)


class WriterMessage():

    def __init__(self, reader, weak_predictor, rows_dict):
        self.reader = reader
        self.weak_predictor = weak_predictor
        self.rows_dict = rows_dict


class Writer():
    def __init__(self, num_readers = 5):
        self.num_readers = num_readers
        self.files_map = {}
        self.queue = queue.Queue()
        self.avail_readers = []
        for i in range(num_readers):
            self.avail_readers.append(Reader(self))

    def write_data(self, reader, rows_dict, weak_predictor):
        self.queue.put(WriterMessage(reader, weak_predictor, rows_dict))

    def write_reader_data(self, msg):
        start = timeit.default_timer()
        for s_date, rows_arr in msg.rows_dict.items():
            if s_date in self.files_map:
                fw = self.files_map[s_date]

            else:
                fw = FileWrapper("data/eval/dates_test/petri/%s.csv" % s_date)
                self.files_map[s_date] = fw

            fw.batch_write(rows_arr)
        stop = timeit.default_timer()
        print("%s weak predictions saved in %.2f seconds" % (msg.weak_predictor, stop - start))

    def process(self):
        for weak_predictor in snp.get_snp_hitorical_components_tickers():
            if len(self.avail_readers) == 0:
                # wait for next free worker
                msg = self.queue.get()
                reader = msg.reader
                self.avail_readers.append(reader)
                self.write_reader_data(msg)
            reader = self.avail_readers.pop(0)
            reader.read(weak_predictor)
            # # debug
            # if weak_predictor != "TEG":
            #     break

        while len(self.avail_readers) != self.num_readers:
            msg = self.queue.get()
            reader = msg.reader
            self.avail_readers.append(reader)
            if msg.weak_predictor is None:
                continue
            self.write_reader_data(msg)

        for reader in self.avail_readers:
            reader.stop()

        print("Closing file handles...")
        for _, fw in self.files_map.items():
            fw.close()


writer = Writer(num_readers=10)
writer.process()