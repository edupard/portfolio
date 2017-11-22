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
from net.equity_curve import PosStrategy, HoldFriFriPosStrategyV1, HoldMonFriPosStrategyV1, PeriodicPosStrategyV1, \
    HoldMonFriPredictMonPosStrategyV1, HoldAlwaysAndRebalance, HoldMonMonPosStrategyV1
from utils.utils import is_same_week
from bma.bma import Bma
from bma.bma_analytical import BmaAnalytical


class PortfolioTest(unittest.TestCase):

    def test_ma_curve(self):
        EPOCH = 600
        PRED_HORIZON = 5

        eval_config = get_eval_config_petri_whole_set()

        total_days = (eval_config.END - eval_config.BEG).days + 1
        tickers = snp.get_snp_hitorical_components_tickers()
        total_tickers = len(tickers)


        ALGO_NAME = "MODEL_AVERAGING"

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
            pos_strategy = HoldMonFriPosStrategyV1()
        elif eval_config.POS_STRATEGY == PosStrategy.FRI_FRI:
            pos_strategy = HoldFriFriPosStrategyV1()
        elif eval_config.POS_STRATEGY == PosStrategy.PERIODIC:
            pos_strategy = PeriodicPosStrategyV1(eval_config.TRADES_FREQ)

        pos_strategy = HoldMonFriPredictMonPosStrategyV1()
        # pos_strategy = HoldFriFriPosStrategyV1()
        # pos_strategy = HoldMonFriPosStrategyV1()
        # pos_strategy = HoldAlwaysAndRebalance()
        # pos_strategy = HoldMonMonPosStrategyV1()

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
