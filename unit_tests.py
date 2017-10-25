import unittest
import numpy as np

from net.net_shiva import NetShiva
import net.train_config as net_config
from stock_data.download import download_data
from stock_data.datasource import DataSource
from net.train_config import get_train_config_petri
from net.eval_config import get_eval_config_petri_train_set
from net.train import train_epoch
from net.predict import predict
import pandas as pd
import utils.folders as folders
import snp.snp as snp
from utils.utils import date_from_timestamp
from utils.csv import create_csv, append_csv


class NetTest(unittest.TestCase):
    def test_download_data(self):
        tickers = snp.get_snp_hitorical_components_tickers()
        download_data(tickers, num_workers=20)

    def test_datasource(self):
        tickers = snp.get_snp_hitorical_components_tickers()
        dss = []
        for ticker in tickers:
            try:
                dss.append(DataSource(ticker))
            except:
                pass

    def test_variance(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7])
        cov = np.cov(arr)
        mean = np.mean(arr)
        _debug = 0

    def test_multi_stock_train(self):
        train_config = get_train_config_petri()
        train_config.DATA_FOLDER = "%s/%s" % (train_config.DATA_FOLDER, 'UNIVERSAL_NET')

        net = NetShiva(train_config)

        dss = []
        for ticker in snp.get_snp_hitorical_components_tickers():
            try:
                dss.append(DataSource(ticker))
            except:
                continue

        net.init_weights(0)
        for e in range(600):
            net.save_weights(e)
            print("Training %d epoch..." % e)
            train_epoch(net, dss, train_config)
        net.save_weights(600)

    def test_train(self):
        ticker = 'AAPL'

        train_config = get_train_config_petri()
        train_config.DATA_FOLDER = "%s/%s" % (train_config.DATA_FOLDER, ticker)

        dss = [DataSource(ticker)]

        net = NetShiva(train_config)

        eval_config = get_eval_config_petri_train_set()

        net.init_weights()
        for e in range(2000):
            net.save_weights(e)

            print("Evaluating %d epoch..." % e)
            avg_loss, predictions_history = predict(net, dss, train_config, eval_config)
            print("Avg loss: %.4f%% " % (avg_loss * 100))

            for ds, p_h in zip(dss, predictions_history):
                if p_h is None:
                    continue
                folder_path, file_path = folders.get_prediction_path(train_config, eval_config, ds.ticker, e)
                folders.create_dir(folder_path)

                df = pd.DataFrame({'ts': p_h[:, 0], 'prediction': p_h[:, 1], 'observation': p_h[:, 2]})
                df.to_csv(file_path, index=False)

            print("Training %d epoch..." % e)
            avg_loss = train_epoch(net, dss, train_config)
            print("Avg loss: %.4f%% " % (avg_loss * 100))

        net.save_weights(600)

    def test_train_multiple_stocks(self):
        train_config = get_train_config_petri()

        snp_tickers = snp.get_snp_hitorical_components_tickers()
        weak_predictors = []
        dss = []
        for ticker in snp_tickers:
            try:
                ds = DataSource(ticker)
            except:
                continue
            beg_idx, end_idx = ds.get_data_range(train_config.BEG, train_config.END)
            if beg_idx is None or end_idx is None:
                continue
            ts = ds.get_ts(beg_idx, end_idx)

            if (date_from_timestamp(ts[-1]) - date_from_timestamp(ts[0])).days < 365 * 3:
                continue

            weak_predictors.append(ticker)
            dss.append(dss)

        for ticker, ds in zip(weak_predictors, dss):
            print("Fitting %s" % ticker)

            train_config.DATA_FOLDER = "%s/%s" % (train_config.DATA_FOLDER, ticker)

            train_dss = [DataSource(ticker)]

            net = NetShiva(train_config)

            # eval_config = get_eval_config_petri_train_set()

            folder_path, file_path = folders.get_train_progress_path(train_config)
            folders.create_dir(folder_path)
            create_csv(file_path, ['loss'])


            net.init_weights()
            for e in range(600):
                net.save_weights(e)

                # print("Evaluating %d epoch..." % e)
                # avg_loss, predictions_history = predict(net, train_dss, train_config, eval_config)
                # print("Avg loss: %.4f%% " % (avg_loss * 100))
                #
                # for ds, p_h in zip(train_dss, predictions_history):
                #     if p_h is None:
                #         continue
                #     folder_path, file_path = folders.get_prediction_path(train_config, eval_config, ds.ticker, e)
                #     folders.create_dir(folder_path)
                #
                #     df = pd.DataFrame({'ts': p_h[:, 0], 'prediction': p_h[:, 1], 'observation': p_h[:, 2]})
                #     df.to_csv(file_path, index=False)

                print("Training %d epoch..." % e)
                avg_loss = train_epoch(net, train_dss, train_config)
                print("Avg loss: %.4f%% " % (avg_loss * 100))

                append_csv(file_path, [avg_loss])


            net.save_weights(600)

    def test_eval(self):
        ticker = 'AAPL'

        train_config = get_train_config_petri()
        train_config.DATA_FOLDER = "%s/%s" % (train_config.DATA_FOLDER, ticker)

        dss = []
        for ticker in snp.get_snp_hitorical_components_tickers():
            try:
                dss.append(DataSource(ticker))
            except:
                continue

        net = NetShiva(train_config)
        net.load_weights(600)

        eval_config = get_eval_config_petri_train_set()

        print("Evaluating %d epoch..." % 600)
        avg_loss, predictions_history = predict(net, dss, train_config, eval_config)
        print("Avg loss: %.4f%% " % (avg_loss * 100))

        for ds, p_h in zip(dss, predictions_history):
            if p_h is None:
                continue
            folder_path, file_path = folders.get_prediction_path(train_config, eval_config, ds.ticker, 600)
            folders.create_dir(folder_path)

            df = pd.DataFrame({'ts': p_h[:, 0], 'prediction': p_h[:, 1], 'observation': p_h[:, 2]})
            df.to_csv(file_path, index=False)
