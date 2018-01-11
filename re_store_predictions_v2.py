import pymongo
import csv
import timeit

client = pymongo.MongoClient('localhost',27017)
db = client['predictions']
collection = db['predictions']

class FileWrapper():
    def __init__(self, file_path):
        self.file = open(file_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(("ticker", "weak_predictor", "prediction"))

    def write(self, ticker, weak_predictor, prediction):
        self.writer.writerow((ticker, weak_predictor, prediction))

    def batch_write(self, rows_arr):
        self.writer.writerows(rows_arr)

    def close(self):
        self.file.close()

files_map = {}
rows_dict = {}

BATCH_SIZE = 1000000

def write_rows(i, sec, rows_dict):
    print("Read %.0f mln records in %.2f seconds" % (i / 1000000, sec))
    start = timeit.default_timer()
    for s_date, rows_arr in rows_dict.items():
        if s_date in files_map:
            fw = files_map[s_date]

        else:
            fw = FileWrapper("data/eval/dates/petri/%s.csv" % s_date)
            files_map[s_date] = fw

        fw.batch_write(rows_arr)
    stop = timeit.default_timer()
    print("Write complete in %.2f seconds" % (stop - start))


it = collection.find({})
i = 0
global_start = timeit.default_timer()
start = global_start
for doc in it:
    if i % BATCH_SIZE == 0:
        write_rows(i, timeit.default_timer() - start, rows_dict)
        start = timeit.default_timer()
        print("Passed %.2f minutes" % ((start - global_start) / 60))
        rows_dict = {}

    s_date = doc["date"]
    prediction = doc["prediction"]
    weak_predictor = doc["weak_predictor"]
    ticker = doc["ticker"]

    if s_date in rows_dict:
        rows_arr = rows_dict[s_date]
    else:
        rows_arr = []
        rows_dict[s_date] = rows_arr
    rows_arr.append((ticker, weak_predictor, prediction))

    i += 1

write_rows(i, timeit.default_timer() - start, rows_dict)

print("Closing file handles...")
for _, fw in files_map.items():
    fw.close()