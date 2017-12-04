import unittest
import numpy as np
import os
import math
import datetime

from net.net_shiva import NetShiva
from net.net_softmax import NetSoftmax
import net.train_config as net_config
from stock_data.download import download_data
from stock_data.datasource import DataSource
from net.train_config import get_train_config_petri
from net.eval_config import get_eval_config_petri_train_set, get_eval_config_petri_test_set, \
    get_eval_config_petri_whole_set
from net.train_config import get_train_config_stacking
from net.train_config import get_train_config_bagging
from net.train import train_epoch, train_stack_epoch
from net.predict import predict
from net.predict import stacking_net_predict
import pandas as pd
import utils.folders as folders
import snp.snp as snp
from utils.utils import date_from_timestamp, get_date_timestamp
from utils.csv import create_csv, append_csv
from net.equity_curve import build_eq
import plots.plots as plots
from utils.metrics import get_eq_params
import utils.dates as known_dates
from net.equity_curve import PosStrategy, HoldFriFriPosStrategy, HoldMonFriPosStrategy, PeriodicPosStrategy, \
    HoldMonFriPredictMonPosStrategy, HoldAlwaysAndRebalance, HoldMonMonPosStrategy
from utils.utils import is_same_week
from bma.bma import Bma
from bma.bma_analytical import BmaAnalytical


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
            avg_loss, predictions_history, last_states = predict(net, dss, train_config, eval_config)
            print("Avg loss: %.4f%% " % (avg_loss * 100))

            save_predictions(predictions_history, dss, train_config, eval_config, e)

            print("Training %d epoch..." % e)
            avg_loss = train_epoch(net, dss, train_config)
            print("Avg loss: %.4f%% " % (avg_loss * 100))

        net.save_weights(600)

    def test_train_stacking_net(self):
        train_config = get_train_config_petri()
        # weak_train_config = get_train_config_petri()
        # train_config = get_train_config_stacking()

        eval_config = get_eval_config_petri_whole_set()

        snp_tickers = snp.get_snp_hitorical_components_tickers()

        weak_predictors = []
        dss = []
        for ticker in snp_tickers:
            try:
                ds = DataSource(ticker)
            except:
                continue
            dss.append(ds)
            beg_idx, end_idx = ds.get_data_range(train_config.BEG, train_config.END)
            if beg_idx is None or end_idx is None:
                continue
            ts = ds.get_ts(beg_idx, end_idx)

            if (date_from_timestamp(ts[-1]) - date_from_timestamp(ts[0])).days < 365 * 3:
                continue

            weak_predictors.append(ticker)

        print("Fitting stacking predictor")

        BASE_FOLDER = train_config.DATA_FOLDER

        train_config.DATA_FOLDER = "%s/%s" % (BASE_FOLDER, "AVERAGE")
        train_config.FEATURE_FUNCTIONS = [0] * len(weak_predictors)

        net = NetSoftmax(train_config)
        # net = NetShiva(train_config)

        folder_path, file_path = folders.get_train_progress_path(train_config)
        folders.create_dir(folder_path)
        # create_csv(file_path, ['loss'])
        # net.init_weights()
        net.load_weights(600)
        # for e in range(600):
        for e in range(600,601):
            # net.save_weights(e)

            print("Evaluating %d epoch..." % e)
            avg_loss, predictions_history = stacking_net_predict(net, dss, train_config, eval_config, weak_predictors)
            print("Avg loss: %.4f%% " % (avg_loss * 100))

            save_predictions(predictions_history, dss, train_config, eval_config, e)
            break

            # print("Training %d epoch..." % e)
            # avg_loss = train_stack_epoch(net, dss, train_config, weak_predictors, eval_config)
            # print("Avg loss: %.4f%% " % (avg_loss * 100))
            #
            # append_csv(file_path, [avg_loss])

        # net.save_weights(600)

    def test_train_multiple_stocks(self):
        train_config = get_train_config_bagging()

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

            if (date_from_timestamp(ts[-1]) - date_from_timestamp(ts[0])).days < 365 * 5:
                continue

            weak_predictors.append(ticker)
            dss.append(dss)

        BASE_FOLDER = train_config.DATA_FOLDER

        for ticker, ds in zip(weak_predictors, dss):
            print("Fitting %s" % ticker)

            train_config.DATA_FOLDER = "%s/%s" % (BASE_FOLDER, ticker)

            train_dss = [DataSource(ticker)]

            # eval_config = get_eval_config_petri_train_set()
            folder_path, file_path = folders.get_train_progress_path(train_config)
            folders.create_dir(folder_path)
            if os.path.exists(file_path):
                print("Already processed")
                continue

            create_csv(file_path, ['loss'])

            net = NetShiva(train_config)
            net.init_weights()
            for e in range(600):
                if e % 100 == 0:
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
        avg_loss, predictions_history, last_states = predict(net, dss, train_config, eval_config)
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
        avg_loss, predictions_history, last_states = predict(net, dss, train_config, eval_config)
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

                avg_loss, predictions_history, last_states = predict(net, dss, train_config, eval_config)
                print("Avg loss: %.4f%% " % (avg_loss * 100))

                save_predictions(predictions_history, dss, train_config, eval_config, EPOCH)
            except:
                pass

    def test_eq(self):
        EPOCH = 600
        ticker = "MRO"
        eval_config = get_eval_config_petri_whole_set()
        folder_path, file_path = folders.get_adaptive_prediction_path("ADAPTIVE_1000_0", eval_config, ticker, EPOCH)
        file_path = 'data/eval/petri/%s/eval/%s/prediction/%s_600.csv' % (ticker, ticker, ticker)
        df = pd.read_csv(file_path)
        ds = DataSource(ticker)

        predictions = df.prediction.values
        eq, ts = build_eq(ds, predictions, eval_config)
        dd, sharpe, y_avg = get_eq_params(eq, ts)
        plots.plot_eq("test", eq, ts)
        plots.show_plots()

    def test_ma_curve(self):
        # plots.iof()
        # EPOCH = 0
        EPOCH = 600
        PRED_HORIZON = 5

        eval_config = get_eval_config_petri_whole_set()

        total_days = (eval_config.END - eval_config.BEG).days + 1
        tickers = snp.get_snp_hitorical_components_tickers()
        total_tickers = len(tickers)

        # ALGO_NAME = "BB_BMA_20"
        # ALGO_NAME = "RANKING"
        # ALGO_NAME = "SOFTMAX_10"
        # ALGO_NAME = "SOFTMAX_95"
        # ALGO_NAME = "SOFTMAX_164"
        # ALGO_NAME = "AVERAGE"
        # ALGO_NAME = "STACK"
        # ALGO_NAME = "COMBINE"
        ALGO_NAME = "MODEL_AVERAGING"

        # ALGO_NAME = "TOP_100_ed_0_05"
        # ALGO_NAME = "TOP_100_ed_0_15"
        # ALGO_NAME = "TOP_150_ed_0_05"
        # ALGO_NAME = "TOP_150_ed_0_15"
        # ALGO_NAME = "TOP_200_ed_0_05"
        # ALGO_NAME = "TOP_200_ed_0_15"
        # ALGO_NAME = "TOP_25_ed_0_05"
        # ALGO_NAME = "TOP_25_ed_0_15"
        # ALGO_NAME = "TOP_25_ed_0_25"
        # ALGO_NAME = "TOP_5_ed_0_05"
        # ALGO_NAME = "TOP_5_ed_0_15"
        # ALGO_NAME = "TOP_5_ed_0_25"
        # ALGO_NAME = "TOP_50_ed_0_05"
        # ALGO_NAME = "TOP_50_ed_0_15"
        # ALGO_NAME = "TOP_50_ed_0_25"

        # ALGO_NAME = "AA_1_ed_0_05"
        # ALGO_NAME = "AA_1_ed_0_5"
        # ALGO_NAME = "AA_1_ed_1_0"
        # ALGO_NAME = "AA_0_1_ed_0_5"
        # ALGO_NAME = "AA_0_1_ed_1_0"
        # ALGO_NAME = "AA_0_01_ed_0_05"
        # ALGO_NAME = "AA_0_01_ed_0_7"

        # ALGO_NAME = "AA_0_01_ed_1_0"


        snp_history = snp.get_snp_history()

        prices = np.zeros((total_tickers, total_days))
        trading_mask = np.full((total_tickers, total_days), False)
        snp_mask = np.full((total_tickers, total_days), False)

        if os.path.exists("data/snp.npz"):
            data = np.load("data/snp.npz")
            prices = data['prices']
            trading_mask = data['trading_mask']
            snp_mask = data['snp_mask']
        else:
            for ticker, ticker_idx in zip(tickers, range(total_tickers)):
                try:
                    ds = DataSource(ticker)
                    b, e = ds.get_data_range(eval_config.BEG, eval_config.END)
                    px = ds.get_a_c(b, e)
                    ts = ds.get_ts(b, e)
                    _prev_idx = 0
                    for j in range(ts.shape[0]):
                        price = px[j]
                        date = date_from_timestamp(ts[j])
                        _idx = (date - eval_config.BEG).days

                        prices[ticker_idx, _prev_idx: _idx + 1] = price
                        trading_mask[ticker_idx, _idx] = True
                        if snp_history.check_if_belongs(ticker, date):
                            snp_mask[ticker_idx, _idx] = True
                        _prev_idx = _idx + 1
                    prices[ticker_idx, _prev_idx:] = price
                except:
                    print("No data")
                    pass

            np.savez("data/snp.npz",
                     prices=prices,
                     trading_mask=trading_mask,
                     snp_mask=snp_mask,
                     )

        predictions = np.zeros((total_tickers, total_days))

        npz_folder, npz_file = folders.get_adaptive_compressed_predictions_path(ALGO_NAME)
        folders.create_dir(npz_folder)
        if os.path.exists(npz_file):
            data = np.load(npz_file)
            predictions = data['predictions']
        else:
            for ticker, ticker_idx in zip(tickers, range(total_tickers)):
                print("Loading predictions for %s" % ticker)
                try:
                    # file_path = "data/eval/petri/AVERAGE/eval/%s/prediction/%s_164.csv" % (ticker, ticker)
                    folder_path, file_path = folders.get_adaptive_prediction_path(ALGO_NAME, eval_config, ticker, EPOCH)
                    df = pd.read_csv(file_path)
                    p = df.prediction.values
                    ts = df.ts.values

                    for j in range(p.shape[0]):
                        date = date_from_timestamp(ts[j])
                        _idx = (date - eval_config.BEG).days
                        _prediction = p[j]
                        predictions[ticker_idx, _idx] = _prediction
                except:
                    print("No data")
                    pass
                np.savez(npz_file,
                         predictions=predictions
                         )

        traded_stocks_per_day = trading_mask[:, :].sum(0)
        trading_day_mask = traded_stocks_per_day >= 30
        trading_day_idxs = np.nonzero(trading_day_mask)[0]

        cash = 0
        pos = np.zeros((total_tickers))
        index_pos = np.zeros((total_tickers))
        pos_px = np.zeros((total_tickers))
        eq = np.zeros(total_days)
        ts = np.zeros(total_days)

        def calc_pl(pos, curr_px, pos_px, slippage):
            return np.sum(pos * (curr_px * (1 - np.sign(pos) * slippage) - pos_px * (1 + np.sign(pos) * slippage)))

        if eval_config.POS_STRATEGY == PosStrategy.MON_FRI:
            pos_strategy = HoldMonFriPosStrategy()
        elif eval_config.POS_STRATEGY == PosStrategy.FRI_FRI:
            pos_strategy = HoldFriFriPosStrategy()
        elif eval_config.POS_STRATEGY == PosStrategy.PERIODIC:
            pos_strategy = PeriodicPosStrategy(eval_config.TRADES_FREQ)

        # pos_strategy = HoldMonFriPredictMonPosStrategyV1()
        # pos_strategy = HoldFriFriPosStrategyV1()
        # pos_strategy = HoldMonFriPosStrategyV1()
        # pos_strategy = HoldAlwaysAndRebalance()
        pos_strategy = HoldMonMonPosStrategy()

        SELECTION = 5

        td_idx = 0
        for i in range(total_days):
            date = eval_config.BEG + datetime.timedelta(days=i)
            curr_px = prices[:, i]
            tradeable = trading_mask[:, i]
            in_snp = snp_mask[:, i]
            trading_day = trading_day_mask[i]

            if trading_day:
                next_trading_date = None
                if td_idx + 1 < trading_day_idxs.shape[0]:
                    next_trading_date = eval_config.BEG + datetime.timedelta(days=trading_day_idxs[td_idx + 1].item())

                predict, close_pos, open_pos = pos_strategy.decide(date, next_trading_date)

                if predict:
                    prediction = predictions[:, i]

                if close_pos:
                    rpl = calc_pl(pos, curr_px, pos_px, eval_config.SLIPPAGE)
                    rpl += calc_pl(index_pos, curr_px, pos_px, 0)
                    cash += rpl
                    pos[:] = 0
                    index_pos[:] = 0
                if open_pos:

                    # long_pos_mask = tradeable & in_snp
                    #
                    # prediction_sorted = np.sort(prediction[long_pos_mask])
                    # long_bound_idx = max(-SELECTION // 2, -prediction_sorted.shape[0])
                    # long_bound = prediction_sorted[long_bound_idx]
                    # prediction_pos_mask = prediction >= long_bound
                    # long_pos_mask &= prediction_pos_mask
                    #
                    # short_pos_mask = tradeable & in_snp
                    #
                    # prediction_sorted = np.sort(prediction[short_pos_mask])
                    # short_bound_idx = min(SELECTION // 2, prediction_sorted.shape[0])
                    # short_bound = prediction_sorted[short_bound_idx]
                    # prediction_pos_mask = prediction <= short_bound
                    # short_pos_mask &= prediction_pos_mask
                    #
                    # long_num_stks = np.sum(long_pos_mask)
                    # pos[long_pos_mask] = 0.5 / long_num_stks / curr_px[long_pos_mask]
                    #
                    # short_num_stks = np.sum(short_pos_mask)
                    # pos[short_pos_mask] = 0.5 / short_num_stks / curr_px[short_pos_mask]



                    pos_mask = tradeable & in_snp

                    prediction_sorted = np.sort(np.abs(prediction[pos_mask]))
                    bound_idx = max(-SELECTION, -prediction_sorted.shape[0])
                    bound = prediction_sorted[bound_idx]
                    prediction_pos_mask = prediction >= bound
                    prediction_pos_mask |= prediction <= -bound
                    pos_mask &= prediction_pos_mask

                    num_stks = np.sum(pos_mask)
                    pos[pos_mask] = 1 / num_stks / curr_px[pos_mask] * np.sign(prediction[pos_mask])


                    # prediction_sorted = np.sort(prediction[pos_mask])
                    # bound_idx = min(SELECTION -1, prediction_sorted.shape[0] - 1)
                    # bound = prediction_sorted[bound_idx]
                    # prediction_pos_mask = prediction <= min(bound, 0)
                    # pos_mask &= prediction_pos_mask

                    # prediction_sorted = np.sort(prediction[pos_mask])
                    # bound_idx = max(-SELECTION, -prediction_sorted.shape[0])
                    # bound = prediction_sorted[bound_idx]
                    # prediction_pos_mask = prediction >= bound
                    # pos_mask &= prediction_pos_mask



                    # num_stks = np.sum(pos_mask)
                    # pos[pos_mask] = 0.5 / num_stks / curr_px[pos_mask] * np.sign(prediction[pos_mask])
                    #
                    # index_pos_mask = tradeable & in_snp
                    #
                    # num_stks = np.sum(index_pos_mask)
                    # index_pos[index_pos_mask] = 1 / num_stks / curr_px[index_pos_mask]

                    # num_stks = np.sum(pos_mask)
                    # pos[pos_mask] = 0.5 / num_stks / curr_px[pos_mask] * np.sign(prediction[pos_mask])
                    #
                    # num_stks = np.sum(index_pos_mask)
                    # pos[index_pos_mask] += -0.5 / num_stks / curr_px[index_pos_mask]

                    pos_px = curr_px
                td_idx += 1

            urpl = calc_pl(pos, curr_px, pos_px, eval_config.SLIPPAGE)
            urpl += calc_pl(index_pos, curr_px, pos_px, 0)
            nlv = cash + urpl

            eq[i] = nlv
            ts[i] = get_date_timestamp(date)

        # plot eq
        folder_path, file_path = folders.get_adaptive_plot_path(ALGO_NAME, "0", EPOCH)
        folders.create_dir(folder_path)

        fig = plots.plot_eq("total", eq, ts)
        fig.savefig(file_path)
        # plots.close_fig(fig)

        CV_TS = get_date_timestamp(known_dates.YR_07)

        test = False
        train = False

        test_idxs = np.nonzero(ts > CV_TS)[0]
        if test_idxs.shape[0] > 0:
            test = True
        train_idxs = np.nonzero(ts <= CV_TS)[0]
        if train_idxs.shape[0] > 0:
            train = True

        test_dd = 0
        test_sharpe = 0
        test_y_avg = 0
        train_dd = 0
        train_sharpe = 0
        train_y_avg = 0

        # save stat
        if test:
            test_dd, test_sharpe, test_y_avg = get_eq_params(eq[test_idxs], ts[test_idxs])
        if train:
            train_dd, train_sharpe, train_y_avg = get_eq_params(eq[train_idxs], ts[train_idxs])
        print("Train dd: %.2f%% y avg: %.2f%% sharpe: %.2f" % (train_dd * 100, train_y_avg * 100, train_sharpe))
        print("Test dd: %.2f%% y avg: %.2f%% sharpe: %.2f" % (test_dd * 100, test_y_avg * 100, test_sharpe))
        plots.show_plots()

    def test_adaptive_prediction(self):
        plots.iof()
        EPOCH = 600
        PRED_HORIZON = 5

        GS = [
            # ("MODEL_AVERAGING", 0.05, 0.0, None),

            # ("TOP_5_ed_0_05", 0.05, 0.0, 5),
            # ("TOP_25_ed_0_05", 0.05, 0.0, 25),
            # ("TOP_50_ed_0_05", 0.05, 0.0, 50),
            # ("TOP_100_ed_0_05", 0.05, 0.0, 100),
            # ("TOP_150_ed_0_05", 0.05, 0.0, 150),
            # ("TOP_200_ed_0_05", 0.05, 0.0, 200),
            #
            # ("TOP_5_ed_0_15", 0.15, 0.0, 5),
            # ("TOP_25_ed_0_15", 0.15, 0.0, 25),
            # ("TOP_50_ed_0_15", 0.15, 0.0, 50),
            # ("TOP_100_ed_0_15", 0.15, 0.0, 100),
            # ("TOP_150_ed_0_15", 0.15, 0.0, 150),
            # ("TOP_200_ed_0_15", 0.15, 0.0, 200),
            #
            # ("TOP_5_ed_0_25", 0.25, 0.0, 5),
            # ("TOP_25_ed_0_25", 0.25, 0.0, 25),


            # ("TOP_50_ed_0_25", 0.25, 0.0, 50),
            # ("TOP_100_ed_0_25", 0.25, 0.0, 100),
            # ("TOP_150_ed_0_25", 0.25, 0.0, 150),
            # ("TOP_200_ed_0_25", 0.25, 0.0, 200),
            #
            # ("TOP_5_ed_0_5", 0.5, 0.0, 5),
            # ("TOP_25_ed_0_5", 0.5, 0.0, 25),
            # ("TOP_50_ed_0_5", 0.5, 0.0, 50),
            # ("TOP_100_ed_0_5", 0.5, 0.0, 100),
            # ("TOP_150_ed_0_5", 0.5, 0.0, 150),
            # ("TOP_200_ed_0_5", 0.5, 0.0, 200),
            #
            # ("TOP_5_ed_0_7", 0.7, 0.0, 5),
            # ("TOP_25_ed_0_7", 0.7, 0.0, 25),
            # ("TOP_50_ed_0_7", 0.7, 0.0, 50),
            # ("TOP_100_ed_0_7", 0.7, 0.0, 100),
            # ("TOP_150_ed_0_7", 0.7, 0.0, 150),
            # ("TOP_200_ed_0_7", 0.7, 0.0, 200),
            #
            # ("TOP_5_ed_0_95", 0.95, 0.0, 5),
            # ("TOP_25_ed_0_95", 0.95, 0.0, 25),
            # ("TOP_50_ed_0_95", 0.95, 0.0, 50),
            # ("TOP_100_ed_0_95", 0.95, 0.0, 100),
            # ("TOP_150_ed_0_95", 0.95, 0.0, 150),
            # ("TOP_200_ed_0_95", 0.95, 0.0, 200),
            #
            # ("TOP_5_ed_1_0", 1.0, 0.0, 5),
            # ("TOP_25_ed_1_0", 1.0, 0.0, 25),
            # ("TOP_50_ed_1_0", 1.0, 0.0, 50),
            # ("TOP_100_ed_1_0", 1.0, 0.0, 100),
            # ("TOP_150_ed_1_0", 1.0, 0.0, 150),
            # ("TOP_200_ed_1_0", 1.0, 0.0, 200),

            ("AA_1_ed_0_05", 0.05, 1.0, None),
            # ("AA_1_ed_0_15", 0.15, 1.0, None),
            # ("AA_1_ed_0_25", 0.25, 1.0, None),
            ("AA_1_ed_0_5", 0.5, 1.0, None),
            # ("AA_1_ed_0_7", 0.7, 1.0, None),
            # ("AA_1_ed_0_95", 0.95, 1.0, None),
            ("AA_1_ed_1_0", 1.0, 1.0, None),

            # ("AA_10_ed_0_05", 0.05, 10.0, None),
            # ("AA_10_ed_0_15", 0.15, 10.0, None),
            # ("AA_10_ed_0_25", 0.25, 10.0, None),
            # ("AA_10_ed_0_5", 0.5, 10.0, None),
            # ("AA_10_ed_0_7", 0.7, 10.0, None),
            # ("AA_10_ed_0_95", 0.95, 10.0, None),
            # ("AA_10_ed_1_0", 1.0, 10.0, None),

            # ("AA_0_1_ed_0_05", 0.05, 0.1, None),
            # ("AA_0_1_ed_0_15", 0.15, 0.1, None),
            # ("AA_0_1_ed_0_25", 0.25, 0.1, None),
            ("AA_0_1_ed_0_5", 0.5, 0.1, None),
            # ("AA_0_1_ed_0_7", 0.7, 0.1, None),
            # ("AA_0_1_ed_0_95", 0.95, 0.1, None),
            ("AA_0_1_ed_1_0", 1.0, 0.1, None),

            ("AA_0_01_ed_0_05", 0.05, 0.01, None),
            # ("AA_0_01_ed_0_15", 0.15, 0.01, None),
            # ("AA_0_01_ed_0_25", 0.25, 0.01, None),
            # ("AA_0_01_ed_0_5", 0.5, 0.01, None),
            ("AA_0_01_ed_0_7", 0.7, 0.01, None),
            # ("AA_0_01_ed_0_95", 0.95, 0.01, None),
            ("AA_0_01_ed_1_0", 1.0, 0.01, None),

            # ("AA_0_001_ed_0_05", 0.05, 0.001, None),
            # ("AA_0_001_ed_0_15", 0.15, 0.001, None),
            # ("AA_0_001_ed_0_25", 0.25, 0.001, None),
            # ("AA_0_001_ed_0_5", 0.5, 0.001, None),
            # ("AA_0_001_ed_0_7", 0.7, 0.001, None),
            # ("AA_0_001_ed_0_95", 0.95, 0.001, None),
            # ("AA_0_001_ed_1_0", 1.0, 0.001, None),

        ]

        # # +?
        # ALGO_NAME = "MODEL_AVERAGING"
        # SMOOTH_FACTOR = 0.05
        # LR = 0.0
        # WEAK_PREDICTOR_TO_AVERAGE = None

        # # -
        # ALGO_NAME = "TOP_5_0_05"
        # SMOOTH_FACTOR = 0.05
        # LR = 0.0
        # WEAK_PREDICTOR_TO_AVERAGE = 5

        # # - over zero exist good stocks but also bad
        # ALGO_NAME = "TOP_5_1_0"
        # SMOOTH_FACTOR = 1.0
        # LR = 0.0
        # WEAK_PREDICTOR_TO_AVERAGE = 5

        # # - over zero exist good stocks but also bad
        # ALGO_NAME = "TOP_5_0_5"
        # SMOOTH_FACTOR = 0.5
        # LR = 0.0
        # WEAK_PREDICTOR_TO_AVERAGE = 5

        # # -+ worst than model averaging
        # ALGO_NAME = "TOP_50_0_5"
        # SMOOTH_FACTOR = 0.5
        # LR = 0.0
        # WEAK_PREDICTOR_TO_AVERAGE = 50

        # #
        # ALGO_NAME = "TOP_250_0_5"
        # SMOOTH_FACTOR = 1.0
        # LR = 0.0
        # WEAK_PREDICTOR_TO_AVERAGE = 250

        for ALGO_NAME, SMOOTH_FACTOR, LR, WEAK_PREDICTOR_TO_AVERAGE in GS:

            CV_TS = get_date_timestamp(known_dates.YR_07)

            tickers = snp.get_snp_hitorical_components_tickers()

            train_config = get_train_config_petri()
            eval_config = get_eval_config_petri_whole_set()

            BASE_FOLDER = train_config.DATA_FOLDER

            stat_folder_path, stat_file_path = folders.get_adaptive_stat_path(ALGO_NAME, EPOCH)
            folders.create_dir(stat_folder_path)
            create_csv(stat_file_path, [
                'ticker',
                'tst n p',
                'tst me',
                'tst dd',
                'tst sharpe',
                'tst y_avg',
                'tr n p',
                'tr me',
                'tr dd',
                'tr sharpe',
                'tr y_avg',
            ])

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
                            errors[j] = SMOOTH_FACTOR * abs(
                                weak_predictions_pred[j][known_moment] - obs[known_moment]) + (1 - SMOOTH_FACTOR) * \
                                                                                              errors[j]

                    # sorted_errors = np.sort(errors)
                    # print("%s top 10 weak predictors: %s" % (dates[i], sorted_errors[:10]))


                    if WEAK_PREDICTOR_TO_AVERAGE is None:
                        # update weights
                        err_mean = np.mean(errors)
                        err_std = np.std(errors)
                        dw = -LR * (errors - err_mean)
                        updated_weights = weigths + dw
                        negative_mask = updated_weights <= 0
                        updated_weights[negative_mask] = 0
                        z = np.sum(updated_weights)
                        updated_weights = updated_weights / z
                        weigths = updated_weights
                    else:
                        # second variant
                        sorted_err_idx = np.argsort(errors)
                        weigths[:] = 0
                        weigths[sorted_err_idx[:WEAK_PREDICTOR_TO_AVERAGE]] = 1 / WEAK_PREDICTOR_TO_AVERAGE

                        remaining_weak = np.nonzero(weigths)[0].shape[0]
                        predictors_num[i] = remaining_weak
                        # print("%s remaining weak predictors: %d" % (dates[i], remaining_weak))

                    # make prediction
                    strong_prediction = np.sum(weigths * predictions)
                    strong_predictions[i] = strong_prediction

                test = False
                train = False

                test_idxs = np.nonzero(ts > CV_TS)[0]
                if test_idxs.shape[0] > 0:
                    test = True
                train_idxs = np.nonzero(ts <= CV_TS)[0]
                if train_idxs.shape[0] > 0:
                    train = True

                # save strong predictions
                df = pd.DataFrame({'ts': ts, 'prediction': strong_predictions, 'observation': obs, 'date': dates})
                df['ticker'] = ticker

                folder_path, file_path = folders.get_adaptive_prediction_path(ALGO_NAME, eval_config, ticker, EPOCH)
                folders.create_dir(folder_path)
                df.to_csv(file_path, index=False)

                # plot eq
                folder_path, file_path = folders.get_adaptive_plot_path(ALGO_NAME, ticker, EPOCH)
                folders.create_dir(folder_path)
                ds = DataSource(ticker)
                eq, ts = build_eq(ds, strong_predictions, eval_config)

                fig = plots.plot_eq(ticker, eq, ts)
                fig.savefig(file_path)
                plots.close_fig(fig)

                test_predictors = 0
                test_avg_error = 0
                test_dd = 0
                test_sharpe = 0
                test_y_avg = 0
                train_predictors = 0
                train_avg_error = 0
                train_dd = 0
                train_sharpe = 0
                train_y_avg = 0

                # save stat
                if test:
                    test_predictors = np.mean(predictors_num[test_idxs])
                    e = strong_predictions[test_idxs] - obs[test_idxs]
                    se = e * e
                    mse = np.mean(se)
                    test_avg_error = math.sqrt(mse)
                    test_dd, test_sharpe, test_y_avg = get_eq_params(eq[test_idxs], ts[test_idxs])
                if train:
                    train_predictors = np.mean(predictors_num[train_idxs])
                    e = strong_predictions[train_idxs] - obs[train_idxs]
                    se = e * e
                    mse = np.mean(se)
                    train_avg_error = math.sqrt(mse)
                    train_dd, train_sharpe, train_y_avg = get_eq_params(eq[train_idxs], ts[train_idxs])
                append_csv(stat_file_path, [
                    ticker,
                    test_predictors,
                    test_avg_error,
                    test_dd,
                    test_sharpe,
                    test_y_avg,
                    train_predictors,
                    train_avg_error,
                    train_dd,
                    train_sharpe,
                    train_y_avg,
                ])

    def test_bma_prediction(self):
        plots.iof()
        EPOCH = 600
        PRED_HORIZON = 5
        ALGO_NAME = "BB_BMA_20"
        LOOKBACK = 20


        CV_TS = get_date_timestamp(known_dates.YR_07)

        tickers = snp.get_snp_hitorical_components_tickers()

        train_config = get_train_config_petri()
        eval_config = get_eval_config_petri_whole_set()

        BASE_FOLDER = train_config.DATA_FOLDER

        stat_folder_path, stat_file_path = folders.get_adaptive_stat_path(ALGO_NAME, EPOCH)
        folders.create_dir(stat_folder_path)
        create_csv(stat_file_path, [
            'ticker',
            'tst n p',
            'tst me',
            'tst dd',
            'tst sharpe',
            'tst y_avg',
            'tr n p',
            'tr me',
            'tr dd',
            'tr sharpe',
            'tr y_avg',
        ])

        processed = [
            'DVN',
            'ETR',
            'CXO',
            'UPS',
            'HON',
            'SHLD',
            'ANDV',
            'PEG',
            'DLR',
            'PRU',
            'HBI',
            'ROK',
            'INTC',
            'ETN',
            'BBT',
            'PXD',
            'WYNN',
            'PCAR',
            'LYB',
            'FBHS',
            'ADP',
            'HNZ',
            'MTD',
            'FAST',
            'NLSN',
            'BWA',
            'DTE',
            'CCE',
            'LB',
            'PAYX',
            'AYE',
            'CPB',
            'NFLX',
            'FSLR',
            'TER',
            'DLTR',
            'AVP',
            'NBR',
            'ALTR',
            'NTRS',
            'CMCSK',
            'ILMN',
            'LSI',
            'GRA',
            'CVS',
            'DTV',
            'ESRX',
            'GAS',
            'CME',
            'FRT',
            'KORS',
            'GRMN',
            'GILD',
            'BRK-B',
            'PNC',
            'BXP',
            'PX',
            'BXLT',
            'GMCR',
            'LNC',
            'DNB',
            'MMC',
            'O',
            'TRIP',
            'EOG',
            'LM',
            'ABC',
            'GS',
            'MNST',
            'VTR',
            'HST',
            'WLTW',
            'CA',
            'NI',
            'UAL',
            'JNPR',
            'MCO',
            'DLPH',
            'ALLE',
            'ESV',
            'CTSH',
            'HRL',
            'MA',
            'EQT',
            'GWW',
            'CSRA',
            'RSH',
            'LKQ',
            'DRI',
            'XL',
            'NOC',
            'BLL',
            'EA',
            'CAH',
            'RRC',
            'WFM',
            'LLL',
            'IP',
            'KMI',
            'ATVI',
            'OMC',
            'CSX',
            'AGN',
            'CERN',
            'MKC',
            'BSX',
            'XRAY',
            'EMR',
            'TWC',
            'MUR',
            'FFIV',
            'FOX',
            'SNDK',
            'BMY',
            'D',
            'DPS',
            'FIS',
            'ADM',
            'UDR',
            'STJ',
            'NTAP',
            'XEL',
            'T',
            'FLS',
            'MDLZ',
            'IGT',
            'EXPE',
            'AMP',
            'CB',
            'WPX',
            'DISH',
            'L',
            'VRTX',
            'MSI',
            'DO',
            'SUN',
            'MCK',
            'PBI',
            'EQIX',
            'LUK',
            'WHR',
            'KODK',
            'SNPS',
            'MS',
            'CLF',
            'COST',
            'JCP',
            'KHC',
            'JNS',
            'JBHT',
            'KRFT',
            'JOY',
            'AYI',
            'MAA',
            'CVX',
            'ADI',
            'AVGO',
            'DXC',
            'MHK',
            'TROW',
            'GM',
            'RJF',
            'HOLX',
            'EXR',
            'NE',
            'HSY',
            'HRB',
            'X',
            'URI',
            'WM',
            'MHS',
            'HAR',
            'USB',
            'BA',
            'FMC',
            'CAT',
            'HOG',
            'APA',
            'SRCL',
            'UNH',
            'AWK',
            'AZO',
            'IFF',
            'SPGI',
            'ADT',
            'ICE',
            'PFE',
            'CF',
            'MLM',
            'ROP',
            'EMN',
            'AES',
            'AOS',
            'IDXX',
            'SNI',
            'DELL',
            'LLY',
            'SPLS',
            'AVB',
            'EVHC',
            'HOT',
            'BLK',
            'MSFT',
            'AJG',
            'YUM',
            'SCHW',
            'PETM',
            'AEE',
            'MOS',
            'TGT',
            'DVA',
            'NFX',
            'VZ',
            'DIS',
            'MMM',
            'DNR',
            'SIAL',
            'PYPL',
            'AN',
            'CFN',
            'GE',
            'REGN',
            'SLG',
            'CMS',
            'PKG',
            'PG',
            'DFS',
            'GNW',
            'MAT',
            'PPL',
            'EQR',
            'XOM',
            'PNR',
            'LUV',
            'FL',
            'BJS',
            'AIV',
            'EXPD',
            'PGR',
            'ALB',
            'VRSN',
            'LEG',
            'GPS',
            'FTV',
            'FE',
            'OXY',
            'MCHP',
            'CVC',
            'FII',
            'NWS',
            'PBCT',
            'ITW',
            'NDAQ',
            'EIX',
            'GHC',
            'DV',
            'ANF',
            'RAI',
            'SPG',
            'IR',
            'ZTS',
            'JDSU',
            'ETFC',
            'CINF',
            'BEAM',
            'BCR',
            'EXC',
            'GOOGL',
            'MMI',
            'DAL',
            'PDCO',
            'INCY',
            'BK',
            'MGM',
            'EFX',
            'PCP',
            'PM',
            'AME',
            'OI',
            'ED',
            'CFG',
            'BMS',
            'ARE',
            'SWN',
            'UNM',
            'BBY',
            'PHM',
            'QCOM',
            'KSS',
            'SE',
            'HAS',
            'TMK',
            'QEP',
            'LO',
            'DUK',
            'NKE',
            'AAL',
            'LVLT',
            'MAS',
            'ALXN',
            'JBL',
            'VMC',
            'DOV',
            'VIAB',
            'ARNC',
            'PRGO',
            'C',
            'ODP',
            'PPG',
            'TXT',
            'UHS',
            'BF-B',
            'DRE',
            'SYK',
            'PNW',
            'URBN',
            'TWX',
            'AAPL',
            'SBR',
            'ORCL',
            'ALK',
            'GD',
            'NWL',
            'DGX',
            'STI',
            'FOSL',
            'Q',
            'SWY',
            'SJM',
            'KLAC',
            'TRV',
            'FCX',
            'BAC',
            'CHK',
            'UA',
            'CI',
            'WBA',
            'DISCK',
            'NEM',
            'MAR',
            'VLO',
            'ABBV',
            'DHR',
            'GGP',
            'HLT',
            'A',
            'KMX',
            'ECL',
            'MRK',
            'VRSK',
            'NYT',
            'AEP',
            'MPC',
            'APD',
            'BDX',
            'HCA',
            'RDC',
            'PKI',
            'HP',
            'TEL',
            'HCP',
            'CAG',
            'IBM',
            'ALL',
            'CHRW',
            'CMI',
            'MTB',
            'DD',
            'GT',
            'ZBH',
            'INFO',
            'BBBY',
            'KEY',
            'GME',
            'GLW',
            'JNJ',
            'HCBK',
            'CMCSA',
            'IRM',
            'CCI',
            'ZION',
            'CMG',
            'EBAY',
            'ITT',
            'CNX',
            'ORLY',
            'WIN',
            'MON',
            'TSN',
            'OKE',
            'HES',
            'CBG',
            'CTXS',
            'AKS',
            'LRCX',
            'MNK',
            'PCLN',
            'FTR',
            'FLR',
            'CRM',
            'CMA',
            'BHGE',
            'CHD',
            'RSG',
            'HSP',
            'COV',
            'KIM',
            'WEC',
            'AMD',
            'LEN',
            'WRK',
            'GPC',
            'IPG',
            'ARG',
            'NSM',
            'PFG',
            'MYL',
            'TAP',
            'WU',
            'CEG',
            'AET',
            'STR',
            'DF',
            'JEC',
            'VNO',
            'SO',
            'SNA',
            'CNP',
            'AAP',
            'GPN',
            'FLIR',
            'SWKS',
            'AMG',
            'PSX',
            'XEC',
            'WMT',
            'IT',
            'PH',
            'HUM',
            'JPM',
            'APOL',
            'RMD',
            'DG',
            'RTN',
            'COO',
            'WAT',
            'SYMC',
            'AFL',
            'WY',
            'CCL',
            'CAM',
            'FITB',
            'CSC',
            'RIG',
            'MAC',
            'ANR',
            'FB',
            'PSA',
            'KG',
            'WDC',
            'SIG',
            'BHF',
            'MCD',
            'SRE',
            'BRCM',
            'DE',
            'JCI',
            'REG',
            'KO',
            'NUE',
            'PCG',
            'AMGN',
            'BEN',
            'NSC',
            'ENDP',
            'KMB',
            'RHI',
            'JWN',
            'FRX',
            'YHOO',
            'SBUX',
            'HIG',
            'PLD',
            'MFE',
            'BAX',
            'UAA',
            'ROST',
            'SHW',
            'NEE',
            'CLX',
            'THC',
            'XRX',
            'QRVO',
            'NAVI',
            'ANSS',
            'S',
            'AKAM',
            'MDT',
            'EL',
            'ADS',
            'ACN',
            'IVZ',
            'FDO',
            'PWR',
            'COL',
            'ESS',
            'SLB',
            'ALGN',
            'SYF',
            'FOXA',
            'LXK',
            'CNC',
            'BIG',
            'SLM',
            'ATI',
            'COP',
            'NVLS',
            'TIF',
            'LIFE',
            'SEE',
            'CELG',
            'NVDA',
            'UNP',
            'TDG',
            'TMO',
            'HPQ',
            'STT',
            'ADBE',
            'MRO',
            'TEG',
        ]

        for ticker in tickers:
            if ticker in processed:
                continue
            print("Generating bma predictions for %s" % ticker)

            weak_predictions = []
            for weak_predictor in tickers:
                train_config.DATA_FOLDER = "%s/%s" % (BASE_FOLDER, weak_predictor)
                _, prediction_path = folders.get_prediction_path(train_config, eval_config, ticker, EPOCH)
                if os.path.exists(prediction_path):
                    weak_predictions.append((weak_predictor, prediction_path))

            weak_predictors_num = len(weak_predictions)

            dates = None
            data_points = None
            ts = None
            weak_predictions_pred = None
            obs = None

            if weak_predictors_num > 0:
                print("Parsing %d weak predictions" % weak_predictors_num)
                idx  = 0
                for weak_predictor, prediction_path in weak_predictions:
                    df = pd.read_csv(prediction_path)
                    if dates is None:
                        dates = df.date.values
                        data_points = len(dates)
                    if ts is None:
                        ts = df.ts.values
                    if obs is None:
                        obs = df.observation.values
                    if weak_predictions_pred is None:
                        weak_predictions_pred = np.zeros((weak_predictors_num, data_points))
                    weak_predictions_pred[idx, :] = df.prediction.values
                    idx += 1
            else:
                print("No predictions for stock")
                continue

            observations = np.stack([obs] * weak_predictors_num)

            strong_predictions = np.zeros((data_points))
            predictors_num = np.zeros((data_points))

            for i in range(data_points):

                known_idx = max(0, i - PRED_HORIZON)
                window_beg_idx = max(0, i - PRED_HORIZON - LOOKBACK)

                bma_predictions_window = weak_predictions_pred[:,window_beg_idx:known_idx]
                bma_obs_window = observations[:,window_beg_idx:known_idx]
                # bma_obs_window = obs[window_beg_idx:known_idx]

                # bma = Bma()
                # # 'BB-pseudo-BMA'
                # # 'pseudo-BMA'
                # # 'stacking'
                # weigths = bma.get_prediction_weights(bma_predictions_window, bma_obs_window, method='BB-pseudo-BMA')

                bma = BmaAnalytical(LOOKBACK)
                weigths = bma.get_prediction_weights(bma_predictions_window, bma_obs_window)

                # make prediction
                strong_prediction = np.sum(weigths * weak_predictions_pred[:,i])
                strong_predictions[i] = strong_prediction

            test = False
            train = False

            test_idxs = np.nonzero(ts > CV_TS)[0]
            if test_idxs.shape[0] > 0:
                test = True
            train_idxs = np.nonzero(ts <= CV_TS)[0]
            if train_idxs.shape[0] > 0:
                train = True

            # save strong predictions
            df = pd.DataFrame({'ts': ts, 'prediction': strong_predictions, 'observation': obs, 'date': dates})
            df['ticker'] = ticker

            folder_path, file_path = folders.get_adaptive_prediction_path(ALGO_NAME, eval_config, ticker, EPOCH)
            folders.create_dir(folder_path)
            df.to_csv(file_path, index=False)

            # plot eq
            folder_path, file_path = folders.get_adaptive_plot_path(ALGO_NAME, ticker, EPOCH)
            folders.create_dir(folder_path)
            ds = DataSource(ticker)
            eq, ts = build_eq(ds, strong_predictions, eval_config)

            fig = plots.plot_eq(ticker, eq, ts)
            fig.savefig(file_path)
            plots.close_fig(fig)

            test_predictors = 0
            test_avg_error = 0
            test_dd = 0
            test_sharpe = 0
            test_y_avg = 0
            train_predictors = 0
            train_avg_error = 0
            train_dd = 0
            train_sharpe = 0
            train_y_avg = 0

            # save stat
            if test:
                test_predictors = np.mean(predictors_num[test_idxs])
                e = strong_predictions[test_idxs] - obs[test_idxs]
                se = e * e
                mse = np.mean(se)
                test_avg_error = math.sqrt(mse)
                test_dd, test_sharpe, test_y_avg = get_eq_params(eq[test_idxs], ts[test_idxs])
            if train:
                train_predictors = np.mean(predictors_num[train_idxs])
                e = strong_predictions[train_idxs] - obs[train_idxs]
                se = e * e
                mse = np.mean(se)
                train_avg_error = math.sqrt(mse)
                train_dd, train_sharpe, train_y_avg = get_eq_params(eq[train_idxs], ts[train_idxs])
            append_csv(stat_file_path, [
                ticker,
                test_predictors,
                test_avg_error,
                test_dd,
                test_sharpe,
                test_y_avg,
                train_predictors,
                train_avg_error,
                train_dd,
                train_sharpe,
                train_y_avg,
            ])

    def test_rank_prediction(self):
        plots.iof()
        EPOCH = 600
        PRED_HORIZON = 5
        ALGO_NAME = "RANKING"

        tickers = snp.get_snp_hitorical_components_tickers()

        train_config = get_train_config_petri()
        eval_config = get_eval_config_petri_whole_set()

        BASE_FOLDER = train_config.DATA_FOLDER

        tickers_to_exclude = ['CPWR', 'BTU', 'MIL', 'PGN']

        # gather weak predictors
        weak_predictors = []
        for weak_predictor in tickers:
            if weak_predictor in tickers_to_exclude:
                continue
            train_config.DATA_FOLDER = "%s/%s" % (BASE_FOLDER, weak_predictor)
            _, prediction_path = folders.get_prediction_path(train_config, eval_config, "AAPL", EPOCH)
            if os.path.exists(prediction_path):
                weak_predictors.append(weak_predictor)

        weak_predictors_num = len(weak_predictors)

        weak_predictors_hist = []
        max_ranks = np.zeros((weak_predictors_num))
        # sort predictions for each weak predictor
        for weak_predictor, idx in zip(weak_predictors, range(weak_predictors_num)):
            train_config.DATA_FOLDER = "%s/%s" % (BASE_FOLDER, weak_predictor)
            # get self prediction path
            _, prediction_path = folders.get_prediction_path(train_config, eval_config, weak_predictor, EPOCH)
            df = pd.read_csv(prediction_path)
            sorted_predictions = np.sort(df.prediction.values)
            weak_predictors_hist.append(sorted_predictions)
            max_ranks[idx] = sorted_predictions.shape[0]


        for ticker in tickers:
            print("Generating rank predictions for %s" % ticker)

            weak_predictions = []
            for weak_predictor in tickers:
                if weak_predictor in tickers_to_exclude:
                    continue
                train_config.DATA_FOLDER = "%s/%s" % (BASE_FOLDER, weak_predictor)
                _, prediction_path = folders.get_prediction_path(train_config, eval_config, ticker, EPOCH)
                if os.path.exists(prediction_path):
                    weak_predictions.append((weak_predictor, prediction_path))

            if weak_predictors_num != len(weak_predictions):
                print("Weakpredictors mismatch for %s!" % ticker)
                continue

            dates = None
            data_points = None
            ts = None
            weak_predictions_pred = None
            obs = None

            if weak_predictors_num > 0:
                print("Parsing %d weak predictions" % weak_predictors_num)
                idx  = 0
                for weak_predictor, prediction_path in weak_predictions:
                    df = pd.read_csv(prediction_path)
                    if dates is None:
                        dates = df.date.values
                        data_points = len(dates)
                    if ts is None:
                        ts = df.ts.values
                    if obs is None:
                        obs = df.observation.values
                    if weak_predictions_pred is None:
                        weak_predictions_pred = np.zeros((weak_predictors_num, data_points))
                    weak_predictions_pred[idx, :] = df.prediction.values
                    idx += 1
            else:
                print("No predictions for stock")
                continue

            ranks = np.zeros(weak_predictions_pred.shape)

            for i in range(weak_predictors_num):
                pred_hist = weak_predictors_hist[i]
                predictions = weak_predictions_pred[i, :]
                ranks[i,:] = np.searchsorted(pred_hist, predictions) / max_ranks[i]

            strong_predictions = np.mean(ranks, axis=0)

            # save strong predictions
            df = pd.DataFrame({'ts': ts, 'prediction': strong_predictions, 'observation': obs, 'date': dates})
            df['ticker'] = ticker

            folder_path, file_path = folders.get_adaptive_prediction_path(ALGO_NAME, eval_config, ticker, EPOCH)
            folders.create_dir(folder_path)
            df.to_csv(file_path, index=False)
