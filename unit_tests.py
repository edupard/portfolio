import unittest
import numpy as np
import os
import math

from net.net_shiva import NetShiva
import net.train_config as net_config
from stock_data.download import download_data
from stock_data.datasource import DataSource
from net.train_config import get_train_config_petri
from net.eval_config import get_eval_config_petri_train_set, get_eval_config_petri_test_set, get_eval_config_petri_whole_set
from net.train import train_epoch
from net.predict import predict
import pandas as pd
import utils.folders as folders
import snp.snp as snp
from utils.utils import date_from_timestamp
from utils.csv import create_csv, append_csv


def save_predictions(predictions_history, dss, train_config, eval_config, epoch):
    for ds, p_h in zip(dss, predictions_history):
        if p_h is None:
            continue
        folder_path, file_path = folders.get_prediction_path(train_config, eval_config, ds.ticker, epoch)
        folders.create_dir(folder_path)

        df = pd.DataFrame({'ts': p_h[:, 0], 'prediction': p_h[:, 1], 'observation': p_h[:, 2]})
        df['ticker'] = ds.ticker
        df['date'] = df['ts'].apply(date_from_timestamp)
        df.to_csv(file_path, index=False)

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

            save_predictions(predictions_history, dss, train_config, eval_config, e)

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

        BASE_FOLDER = train_config.DATA_FOLDER

        for ticker, ds in zip(weak_predictors, dss):
            print("Fitting %s" % ticker)

            train_config.DATA_FOLDER = "%s/%s" % (BASE_FOLDER, ticker)

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

                # save_predictions(predictions_history, dss, train_config, eval_config, e)

                print("Training %d epoch..." % e)
                avg_loss = train_epoch(net, train_dss, train_config)
                print("Avg loss: %.4f%% " % (avg_loss * 100))

                append_csv(file_path, [avg_loss])

            net.save_weights(600)


    def test_eval(self):
        ticker = 'AAPL'
        EPOCH = 600

        train_config = get_train_config_petri()
        train_config.DATA_FOLDER = "%s/%s" % (train_config.DATA_FOLDER, ticker)

        dss = []
        for ticker in snp.get_snp_hitorical_components_tickers():
            try:
                dss.append(DataSource(ticker))
            except:
                continue

        net = NetShiva(train_config)
        net.load_weights(EPOCH)

        eval_config = get_eval_config_petri_train_set()

        print("Evaluating %d epoch..." % EPOCH)
        avg_loss, predictions_history = predict(net, dss, train_config, eval_config)
        print("Avg loss: %.4f%% " % (avg_loss * 100))
        save_predictions(predictions_history, dss, train_config, eval_config, EPOCH)

    def test_eval_universal_strategy(self):
        EPOCH = 585
        ticker = 'UNIVERSAL_NET'

        train_config = get_train_config_petri()
        train_config.DATA_FOLDER = "%s/%s" % (train_config.DATA_FOLDER, ticker)

        dss = []
        for ticker in snp.get_snp_hitorical_components_tickers():
            try:
                dss.append(DataSource(ticker))
            except:
                continue

        net = NetShiva(train_config)
        net.load_weights(EPOCH)

        eval_config = get_eval_config_petri_test_set()


        print("Evaluating %d epoch..." % EPOCH)
        avg_loss, predictions_history = predict(net, dss, train_config, eval_config)
        print("Avg loss: %.4f%% " % (avg_loss * 100))

        save_predictions(predictions_history, dss, train_config, eval_config, EPOCH)

    def test_eval_weak_predictors(self):
        EPOCH = 600

        tickers = snp.get_snp_hitorical_components_tickers()

        tickers_to_exclude = ['CPWR', 'BTU', 'MIL', 'PGN']

        dss = []
        for ticker in tickers:
            if ticker in tickers_to_exclude:
                continue
            try:
                dss.append(DataSource(ticker))
            except:
                continue

        train_config = get_train_config_petri()
        eval_config = get_eval_config_petri_whole_set()
        net = NetShiva(train_config)

        BASE_FOLDER = train_config.DATA_FOLDER

        for ticker in tickers:
            try:
                print("Evaluating %s weak predictor..." % ticker)

                train_config.DATA_FOLDER = "%s/%s" % (BASE_FOLDER, ticker)
                net.update_config(train_config)

                net.load_weights(EPOCH)

                avg_loss, predictions_history = predict(net, dss, train_config, eval_config)
                print("Avg loss: %.4f%% " % (avg_loss * 100))

                save_predictions(predictions_history, dss, train_config, eval_config, EPOCH)
            except:
                pass


    def test_adaptive_prediction(self):
        EPOCH = 600
        PRED_HORIZON = 5

        LR = 1000.0
        SMOOTH_FACTOR = 0.05

        WEAK_PREDICTOR_TO_AVERAGE = 5

        tickers = snp.get_snp_hitorical_components_tickers()

        train_config = get_train_config_petri()
        eval_config = get_eval_config_petri_whole_set()

        BASE_FOLDER = train_config.DATA_FOLDER

        for ticker in tickers:
            print("Generating adaptive predictions for %s" % ticker)

            weak_predictions = []
            for weak_predictor in tickers:
                train_config.DATA_FOLDER = "%s/%s" % (BASE_FOLDER, weak_predictor)
                _, prediction_path = folders.get_prediction_path(train_config, eval_config, ticker, EPOCH)
                if os.path.exists(prediction_path):
                    weak_predictions.append((weak_predictor, prediction_path))

            weak_predictors_num = len(weak_predictions)

            dates = None
            ts = None
            weak_predictions_pred = []
            obs = None

            if weak_predictors_num > 0:
                print("Parsing %d weak predictions" % weak_predictors_num)
                for weak_predictor, prediction_path in weak_predictions:
                    df = pd.read_csv(prediction_path)
                    if dates is None:
                        dates = df.date.values
                    if ts is None:
                        ts = df.ts.values
                    if obs is None:
                        obs = df.observation.values
                    weak_predictions_pred.append(df.prediction.values)
            else:
                print("No predictions for stock")
                continue



            weigths = np.full((weak_predictors_num), 1 / weak_predictors_num)
            errors = np.zeros((weak_predictors_num))
            predictions = np.zeros((weak_predictors_num))
            data_points = len(dates)
            strong_predictions = np.zeros((data_points))
            predictors_num = np.zeros((data_points))

            for i in range(data_points):
                for j in range(weak_predictors_num):
                    predictions[j] = weak_predictions_pred[j][i]
                    if i >= PRED_HORIZON:
                        known_moment = i - PRED_HORIZON
                        errors[j] = SMOOTH_FACTOR * abs(weak_predictions_pred[j][known_moment]- obs[known_moment]) + (1 - SMOOTH_FACTOR) * errors[j]

                # sorted_errors = np.sort(errors)
                # print("%s top 10 weak predictors: %s" % (dates[i], sorted_errors[:10]))


                # update weights
                err_mean = np.mean(errors)
                err_std = np.std(errors)
                dw = -LR * (errors - err_mean)
                updated_weights = weigths + dw
                negative_mask = updated_weights <=0
                updated_weights[negative_mask] = 0
                z = np.sum(updated_weights)
                updated_weights = updated_weights / z
                weigths = updated_weights

                # second variant
                sorted_err_idx = np.argsort(errors)
                weigths[:] = 0
                weigths[sorted_err_idx[:WEAK_PREDICTOR_TO_AVERAGE]] = 1/WEAK_PREDICTOR_TO_AVERAGE


                remaining_weak = np.nonzero(weigths)[0].shape[0]
                predictors_num[i] = remaining_weak
                # print("%s remaining weak predictors: %d" % (dates[i], remaining_weak))



                # make prediction
                strong_prediction = np.sum(weigths * predictions)
                strong_predictions[i] = strong_prediction

            print("%s avg num predictors: %.2f" % (ticker, np.mean(predictors_num)))
            e = strong_predictions - obs
            se = e * e
            mse = np.mean(se)
            avg_error = math.sqrt(mse)
            print("%s MSE: %.4f%% !!!" % (ticker, avg_error * 100))


            df = pd.DataFrame({'ts': ts, 'prediction': strong_predictions, 'observation': obs, 'date': dates})
            df['ticker'] = ticker

            folder_path, file_path = folders.get_adaptive_prediction_path("ADAPTIVE_1000_0", eval_config, ticker, EPOCH)
            folders.create_dir(folder_path)
            df.to_csv(file_path, index=False)








