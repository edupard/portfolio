import snp.snp as snp
import net.train_config as train_config
import net.eval_config as eval_config
import utils.folders as folders
import os
import pandas as pd
import csv
import timeit

train_config = train_config.get_train_config_petri()
eval_config = eval_config.get_eval_config_petri_whole_set()

BASE_FOLDER = train_config.DATA_FOLDER

EPOCH = 600

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

try:
    i = 0
    w_p_i = 0
    for weak_predictor in snp.get_snp_hitorical_components_tickers():
        w_p_start_time = timeit.default_timer()
        try:
            print("Processing %d %s weak predictor..." % (w_p_i, weak_predictor))
            train_config.DATA_FOLDER = "%s/%s" % (BASE_FOLDER, weak_predictor)

            rows_dict = {}
            p_i = 0
            for ticker in snp.get_snp_hitorical_components_tickers():
                _, prediction_path = folders.get_prediction_path(train_config, eval_config, ticker, EPOCH)
                if os.path.exists(prediction_path):
                    print("Found %s prediction for %s" % (p_i, ticker))
                    p_i += 1
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
                        i += 1
                        if i % 1000000 == 0:
                            print("%.2f" % (i / 1000000))
            w_p_store_start_time = timeit.default_timer()
            print(w_p_store_start_time - w_p_start_time)

            print("Storing %s weak predictions..." % weak_predictor)
            for s_date, rows_arr in rows_dict.items():
                if s_date in files_map:
                    fw = files_map[s_date]

                else:
                    fw = FileWrapper("data/eval/dates/petri/%s.csv" % s_date)
                    files_map[s_date] = fw

                fw.batch_write(rows_arr)
            w_p_end_time = timeit.default_timer()
            print(w_p_end_time - w_p_store_start_time)
        except:
            print("Can not process %s weak predictor" % weak_predictor)
            pass
        w_p_i += 1
finally:
    print("Closing file handles...")
    for _, fw in files_map.items():
        fw.close()