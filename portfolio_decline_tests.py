import unittest
import numpy as np
import os
import datetime

from stock_data.datasource import DataSource
from net.eval_config import get_eval_config_petri_train_set, get_eval_config_petri_test_set, \
    get_eval_config_petri_whole_set
import pandas as pd
import utils.folders as folders
import snp.snp as snp
from utils.utils import date_from_timestamp, get_date_timestamp
import plots.plots as plots
from utils.metrics import get_eq_params
import utils.dates as known_dates
from net.equity_curve import PosStrategy, HoldFriFriPosStrategy, HoldMonFriPosStrategy, PeriodicPosStrategy, \
    HoldMonFriPredictMonPosStrategy, HoldAlwaysAndRebalance, HoldMonMonPosStrategy

from net.equity_curve import HoldFriFriRebalanceMonPosStrategy

from net.equity_curve import TrackMonFriStrategy
from net.equity_curve import TrackFriFriStrategy
import quandl_data.quandl_data as quandl_data


class PortfolioTest(unittest.TestCase):
    def test_ma_curve(self):
        EPOCH = 600

        eval_config = get_eval_config_petri_whole_set()

        total_days = (eval_config.END - eval_config.BEG).days + 1
        tickers = snp.get_snp_hitorical_components_tickers()
        total_tickers = len(tickers)

        ALGO_NAME = "MODEL_AVERAGING"

        es_tickers = quandl_data.get_es_tickers()
        if os.path.exists("data/es.npz"):
            data = np.load("data/es.npz")
            es_prices = data['es_prices']
            es_trading_mask = data['es_trading_mask']
        else:
            total_es_tickers = len(es_tickers)
            es_prices = np.zeros((total_es_tickers, total_days))
            es_trading_mask = np.full((total_es_tickers, total_days), False)
            es_ds = []
            for (t, _), ticker_idx in zip(es_tickers, range(total_es_tickers)):
                ds = quandl_data.EsDataSouce(t)
                es_ds.append(ds)
                close_pxs = ds.get_close_px()
                dates = ds.get_dates()
                if close_pxs is None:
                    continue
                data_len = close_pxs.shape[0]
                _prev_idx = 0
                price = None
                for i in range(data_len):
                    date = dates[dates.index[i]]
                    if eval_config.BEG <= date <= eval_config.END:
                        _idx = (date - eval_config.BEG).days
                        price = close_pxs[i]
                        # if _prev_idx == 0:
                        #     _prev_idx = _idx
                        es_prices[ticker_idx, _prev_idx: _idx + 1] = price
                        es_trading_mask[ticker_idx, _idx] = True
                        _prev_idx = _idx + 1
                if price is not None:
                    es_prices[ticker_idx, _prev_idx:] = price
            np.savez("data/es.npz",
                     es_prices=es_prices,
                     es_trading_mask=es_trading_mask,
                     )

        vx_tickers = quandl_data.get_vx_tickers()
        if os.path.exists("data/vx.npz"):
            data = np.load("data/vx.npz")
            vx_prices = data['vx_prices']
            vx_trading_mask = data['vx_trading_mask']
        else:
            total_vx_tickers = len(vx_tickers)
            vx_prices = np.zeros((total_vx_tickers, total_days))
            vx_trading_mask = np.full((total_vx_tickers, total_days), False)
            vx_ds = []
            for (t, _), ticker_idx in zip(vx_tickers, range(total_vx_tickers)):
                ds = quandl_data.VxDataSouce(t)
                vx_ds.append(ds)
                close_pxs = ds.get_close_px()
                dates = ds.get_dates()
                if close_pxs is None:
                    continue
                data_len = close_pxs.shape[0]
                _prev_idx = 0
                price = None
                for i in range(data_len):
                    date = dates[dates.index[i]]
                    if eval_config.BEG <= date <= eval_config.END:
                        _idx = (date - eval_config.BEG).days
                        price = close_pxs[i]
                        # if _prev_idx == 0:
                        #     _prev_idx = _idx
                        vx_prices[ticker_idx, _prev_idx: _idx + 1] = price
                        vx_trading_mask[ticker_idx, _idx] = True
                        _prev_idx = _idx + 1
                if price is not None:
                    vx_prices[ticker_idx, _prev_idx:] = price
            np.savez("data/vx.npz",
                     vx_prices=vx_prices,
                     vx_trading_mask=vx_trading_mask,
                     )

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

        CV_TS = get_date_timestamp(known_dates.YR_07)
        ts = np.zeros(total_days)

        for i in range(total_days):
            date = eval_config.BEG + datetime.timedelta(days=i)
            ts[i] = get_date_timestamp(date)

        after_2007_mask = ts >= CV_TS

        traded_stocks_per_day = trading_mask[:, :].sum(0)
        trading_day_mask = traded_stocks_per_day >= 30
        trading_day_mask &= after_2007_mask
        trading_day_idxs = np.nonzero(trading_day_mask)[0]

        es_trader_futures_per_day = es_trading_mask[:, :].sum(0)
        es_trading_day_mask = es_trader_futures_per_day >= 1
        es_trading_day_mask &= after_2007_mask
        es_trading_day_idxs = np.nonzero(es_trading_day_mask)[0]
        total_es_trading_days = es_trading_day_idxs.shape[0]

        vx_trader_futures_per_day = vx_trading_mask[:, :].sum(0)
        vx_trading_day_mask = vx_trader_futures_per_day >= 1
        vx_trading_day_mask &= after_2007_mask
        vx_trading_day_idxs = np.nonzero(vx_trading_day_mask)[0]
        total_vx_trading_days = vx_trading_day_idxs.shape[0]

        es_active_contract_idx = np.full((total_days), -1, dtype=np.int)
        es_close_pos_contract_idx = np.full((total_days), -1, dtype=np.int)
        es_open_pos_contract_idx = np.full((total_days), -1, dtype=np.int)
        vx_active_contract_idx = np.full((total_days), -1, dtype=np.int)
        vx_close_pos_contract_idx = np.full((total_days), -1, dtype=np.int)
        vx_open_pos_contract_idx = np.full((total_days), -1, dtype=np.int)
        es_active_idx = -1
        vx_active_idx = -1
        for i in range(total_days):
            es_mask = es_trading_mask[:, i]
            es_idxs = np.nonzero(es_mask)[0]
            if es_idxs.shape[0] > 0:
                new_es_active_idx = es_idxs[0]
                if new_es_active_idx < es_active_idx:
                    _debug = 0
                es_active_idx = new_es_active_idx
            es_active_contract_idx[i] = es_active_idx

            vx_mask = vx_trading_mask[:, i]
            vx_idxs = np.nonzero(vx_mask)[0]
            if vx_idxs.shape[0] > 0:
                new_vx_active_idx = vx_idxs[0]
                if new_vx_active_idx < vx_active_idx:
                    _debug = 0
                vx_active_idx = new_vx_active_idx
            vx_active_contract_idx[i] = vx_active_idx

        es_close_pos_contract_idx[:] = es_active_contract_idx[:]
        es_open_pos_contract_idx[:] = es_active_contract_idx[:]

        vx_close_pos_contract_idx[:] = vx_active_contract_idx[:]
        vx_open_pos_contract_idx[:] = vx_active_contract_idx[:]

        ES_ROLL_DAYS = 1
        VX_ROLL_DAYS = 1

        prev_day_idx = None
        for i in range(total_es_trading_days - ES_ROLL_DAYS):
            day_idx = es_trading_day_idxs[i]
            if prev_day_idx is None:
                prev_day_idx = day_idx
            roll_day_idx = es_trading_day_idxs[i + ES_ROLL_DAYS]
            es_close_pos_contract_idx[prev_day_idx:day_idx + 1] = es_active_contract_idx[roll_day_idx]
            prev_day_idx = day_idx + 1

        prev_day_idx = None
        for i in range(total_vx_trading_days - VX_ROLL_DAYS):
            day_idx = vx_trading_day_idxs[i]
            if prev_day_idx is None:
                prev_day_idx = day_idx
            roll_day_idx = vx_trading_day_idxs[i + VX_ROLL_DAYS]
            vx_close_pos_contract_idx[prev_day_idx:day_idx + 1] = vx_active_contract_idx[roll_day_idx]
            prev_day_idx = day_idx + 1

        ES_ROLL_DAYS += 1
        VX_ROLL_DAYS += 1

        prev_day_idx = None
        for i in range(total_es_trading_days - ES_ROLL_DAYS):
            day_idx = es_trading_day_idxs[i]
            if prev_day_idx is None:
                prev_day_idx = day_idx
            roll_day_idx = es_trading_day_idxs[i + ES_ROLL_DAYS]
            es_open_pos_contract_idx[prev_day_idx:day_idx + 1] = es_active_contract_idx[roll_day_idx]
            prev_day_idx = day_idx + 1

        prev_day_idx = None
        for i in range(total_vx_trading_days - VX_ROLL_DAYS):
            day_idx = vx_trading_day_idxs[i]
            if prev_day_idx is None:
                prev_day_idx = day_idx
            roll_day_idx = vx_trading_day_idxs[i + VX_ROLL_DAYS]
            vx_open_pos_contract_idx[prev_day_idx:day_idx + 1] = vx_active_contract_idx[roll_day_idx]
            prev_day_idx = day_idx + 1

        def calc_pl_simple(pos, curr_px, pos_px, price_step):
            return pos * ((curr_px - np.sign(pos) * price_step) - (pos_px + np.sign(pos) * price_step))

        def calc_pl(pos, curr_px, pos_px, slippage):
            return np.sum(pos * (curr_px * (1 - np.sign(pos) * slippage) - pos_px * (1 + np.sign(pos) * slippage)))

        if eval_config.POS_STRATEGY == PosStrategy.MON_FRI:
            pos_strategy = HoldMonFriPosStrategy()
        elif eval_config.POS_STRATEGY == PosStrategy.FRI_FRI:
            pos_strategy = HoldFriFriPosStrategy()
        elif eval_config.POS_STRATEGY == PosStrategy.PERIODIC:
            pos_strategy = PeriodicPosStrategy(eval_config.TRADES_FREQ)

        pos_strategy = HoldMonFriPredictMonPosStrategy()

        # pos_strategy = HoldFriFriPosStrategy()
        # pos_strategy = HoldMonFriPosStrategy()
        # pos_strategy = HoldAlwaysAndRebalance()
        # pos_strategy = HoldMonMonPosStrategy()

        class RollPosStrategy(object):

            def __init__(self):
                self.in_pos = False

            def decide(self, close_contract_idx, open_contract_idx):
                # to open initial position
                if not self.in_pos:
                    self.in_pos = True
                    return True, True
                if close_contract_idx != open_contract_idx:
                    return True, True
                return False, False

        es_pos_strategy = RollPosStrategy()
        vx_pos_strategy = RollPosStrategy()
        VX_LAST_TRADING_DATE_BEFORE_RESCALE = datetime.datetime.strptime('2007-03-23', '%Y-%m-%d').date()

        SELECTION = 5
        RECAP = False

        NET_BET = 0
        ES_BET = 1
        VX_BET = 0

        # GS = [(0,-10,-10),(0,-100,-100)]
        # GS = [(0, -100, -100), (0, -10, -10)]
        # GS = [(1, 0, 0),(0,-1,0),(0,0,-1)]
        # GS = [(1, 0, -1)]

        # GS = []
        # for i in np.arange(0, 101, 10):
        #     for j in np.arange(-100,101,10):
        #         for k in np.arange(-100,101,10):
        #             if i == 0 and j == 0 and k == 0:
        #                 continue
        #             NET_BET = i
        #             ES_BET = j
        #             VX_BET = k
        #             GS.append((NET_BET, ES_BET, VX_BET))

        # GS = []
        # for j in np.arange(-400, 401, 10):
        #     for k in np.arange(-150, 101, 10):
        #         NET_BET = 100
        #         ES_BET = j
        #         VX_BET = k
        #         if NET_BET == 0 and ES_BET == 0 and VX_BET == 0:
        #             continue
        #         GS.append((NET_BET, ES_BET, VX_BET))

        GS = []
        for k in np.arange(-100, 101, 10):
            NET_BET = 0
            ES_BET = -100
            VX_BET = k
            if NET_BET == 0 and ES_BET == 0 and VX_BET == 0:
                continue
            GS.append((NET_BET, ES_BET, VX_BET))

        # GS = [(1, 0, -1), (0, -1, 0)]

        # GS = [(1, 0, 0)]
        GS = [
            (100, -120, -40),
            (100, -120, -70),
            (100, -120, -50),
        ]

        GS = [
            (1, 0, 0)
        ]

        DECLINE_LONG_FACTOR = 0.5
        DECLINE_SHORT_FACTOR = 0.5
        LONG_STOCKS = 5
        SHORT_STOCKS = 5
        LONG_PCT = 0.5
        SHORT_PCT = 1 - LONG_PCT

        # DECLINE_GS = [
        #     (0.5, 0.5, 5, 5),
        #     (0.5, 0.5, 5, 5),
        # ]

        DECLINE_GS = []


        for SHORT_STOCKS in range(5, 51, 5):
            for LONG_STOCKS in range(SHORT_STOCKS, 51, 5):
                for DECLINE_LONG_FACTOR in np.arange(0, 1.01, 0.25):
                    for DECLINE_SHORT_FACTOR in np.arange(0, 1.01, 0.25):
                        DECLINE_GS.append((DECLINE_LONG_FACTOR, DECLINE_SHORT_FACTOR, LONG_STOCKS, SHORT_STOCKS))


        PRINT_DECLINE_GS = True

        HIDE_PLOT = True

        if HIDE_PLOT:
            plots.iof()

        fig = None
        ax = None

        PRINT_TRADES = False
        trades = []

        ES_PRICE_STEP = 5
        VX_PRICE_STEP = 0.05

        INIT_NLV = 1000000
        PRINT_PERIODIC_STAT = False
        BY_QUARTER = False
        if PRINT_PERIODIC_STAT:
            RECAP = True

        PRINT_WEEKLY_PL_CHANGES = False
        if PRINT_WEEKLY_PL_CHANGES:
            weekly_balance = []
            open_nlv = None
            open_date = None

            # # NetChange_MonFri
            # weekly_balance_file_name = "data/net_change_mon_fri.csv"
            # CH_COL_NAME = "NetChange_MonFri"
            # OPEN_DATE_COL_NAME = "w beg"
            # CLOSE_DATE_COL_NAME = "w end"
            # track_strategy = TrackMonFriStrategy()
            # pos_strategy = HoldMonFriPredictMonPosStrategy()
            # GS = [(1, 0, 0)]

            # # NetChange_FriFri
            # weekly_balance_file_name = "data/net_change_fri_fri.csv"
            # CH_COL_NAME = "NetChange_FriFri"
            # OPEN_DATE_COL_NAME = "prev w end"
            # CLOSE_DATE_COL_NAME = "w end"
            # track_strategy = TrackFriFriStrategy()
            # pos_strategy = HoldFriFriPosStrategy()
            # GS = [(1, 0, 0)]

            # NetChange_FriFri rebalance Mon
            weekly_balance_file_name = "data/net_change_fri_fri_rebalance_mon.csv"
            CH_COL_NAME = "NetChange_FriFri_Rebalance_Mon"
            OPEN_DATE_COL_NAME = "prev w end"
            CLOSE_DATE_COL_NAME = "w end"
            track_strategy = TrackFriFriStrategy()
            pos_strategy = HoldFriFriRebalanceMonPosStrategy()
            GS = [(1, 0, 0)]

            # # EsChange_MonFri
            # weekly_balance_file_name = "data/es_change_mon_fri.csv"
            # CH_COL_NAME = "EsChange_MonFri"
            # OPEN_DATE_COL_NAME = "w beg"
            # CLOSE_DATE_COL_NAME = "w end"
            # track_strategy = TrackMonFriStrategy()
            # GS = [(0, 1, 0)]
            # ES_PRICE_STEP = 0

            # # EsChange_FriFri
            # weekly_balance_file_name = "data/es_change_fri_fri.csv"
            # CH_COL_NAME = "EsChange_FriFri"
            # OPEN_DATE_COL_NAME = "prev w end"
            # CLOSE_DATE_COL_NAME = "w end"
            # track_strategy = TrackFriFriStrategy()
            # GS = [(0, 1, 0)]
            # ES_PRICE_STEP = 0

        for NET_BET, ES_BET, VX_BET in GS:
            for DECLINE_LONG_FACTOR, DECLINE_SHORT_FACTOR, LONG_STOCKS, SHORT_STOCKS in DECLINE_GS:
                LONG_PCT = LONG_STOCKS / (LONG_STOCKS + SHORT_STOCKS)
                SHORT_PCT = SHORT_STOCKS / (LONG_STOCKS + SHORT_STOCKS)
                decline_params = "DLF %03.0f%% DSF %03.0f%% LS %02d SS %02d" % (
                    DECLINE_LONG_FACTOR * 100, DECLINE_SHORT_FACTOR * 100, LONG_STOCKS, SHORT_STOCKS)

                prev_pos_mask = None
                prev_px = None

                linear_combination = "NET_%d_ES_%d_VX_%d" % (NET_BET, ES_BET, VX_BET)

                Z = abs(NET_BET) + abs(ES_BET) + abs(VX_BET)
                NET_BET /= Z
                ES_BET /= Z
                VX_BET /= Z

                linear_combination_normalizer = "NET_%.2f_ES_%.2f_VX_%.2f" % (NET_BET, ES_BET, VX_BET)

                cash = 1
                pos = np.zeros((total_tickers))
                pos_px = np.zeros((total_tickers))
                eq = np.zeros(total_days)
                es_pos = 0
                es_pos_px = 0
                vx_pos = 0
                vx_pos_px = 0

                td_idx = 0
                for i in range(total_days):
                    date = eval_config.BEG + datetime.timedelta(days=i)

                    es_trading_day = es_trading_day_mask[i]

                    es_close_contract_idx = es_close_pos_contract_idx[i]
                    es_open_contract_idx = es_open_pos_contract_idx[i]

                    es_close_px = es_prices[es_close_contract_idx, i]
                    es_open_px = es_prices[es_open_contract_idx, i]

                    es_close_pos = False
                    es_open_pos = False

                    if es_trading_day:
                        es_close_pos, es_open_pos = es_pos_strategy.decide(es_close_contract_idx, es_open_contract_idx)

                    if es_close_pos:
                        rpl = calc_pl_simple(es_pos, es_close_px, es_pos_px, ES_PRICE_STEP)
                        cash += rpl
                        es_pos = 0
                    if es_open_pos:
                        if RECAP:
                            es_pos = abs(ES_BET) * eq[i - 1] / es_open_px * np.sign(ES_BET)
                            # es_pos = abs(ES_BET) * cash / es_open_px * np.sign(ES_BET)
                        else:
                            es_pos = abs(ES_BET) / es_open_px * np.sign(ES_BET)
                        es_pos_px = es_open_px
                        es_close_px = es_open_px

                    if date == datetime.datetime.strptime('2007-04-13', '%Y-%m-%d').date():
                        _debug = 0

                    vx_trading_day = vx_trading_day_mask[i]

                    vx_close_contract_idx = vx_close_pos_contract_idx[i]
                    vx_open_contract_idx = vx_open_pos_contract_idx[i]

                    vx_close_px = vx_prices[vx_close_contract_idx, i]
                    if vx_tickers[vx_close_contract_idx][0] == 'VXJ2007' and date > VX_LAST_TRADING_DATE_BEFORE_RESCALE:
                        vx_close_px *= 10

                    vx_open_px = vx_prices[vx_open_contract_idx, i]
                    vx_close_pos = False
                    vx_open_pos = False

                    if vx_trading_day:
                        vx_close_pos, vx_open_pos = vx_pos_strategy.decide(vx_close_contract_idx, vx_open_contract_idx)

                    if vx_close_pos:
                        rpl = calc_pl_simple(vx_pos, vx_close_px, vx_pos_px, VX_PRICE_STEP)
                        cash += rpl
                        vx_pos = 0
                    if vx_open_pos:
                        if RECAP:
                            vx_pos = abs(VX_BET) * eq[i - 1] / vx_open_px * np.sign(VX_BET)
                            # vx_pos = abs(VX_BET) * cash / vx_open_px * np.sign(VX_BET)
                        else:
                            vx_pos = abs(VX_BET) / vx_open_px * np.sign(VX_BET)
                        vx_pos_px = vx_open_px
                        vx_close_px = vx_open_px

                    curr_px = prices[:, i]
                    tradeable = trading_mask[:, i]
                    in_snp = snp_mask[:, i]
                    trading_day = trading_day_mask[i]

                    if trading_day:
                        next_trading_date = None
                        if td_idx + 1 < trading_day_idxs.shape[0]:
                            next_trading_date = eval_config.BEG + datetime.timedelta(
                                days=trading_day_idxs[td_idx + 1].item())

                    if trading_day:
                        predict, close_pos, open_pos = pos_strategy.decide(date, next_trading_date)

                        if predict:
                            prediction = predictions[:, i]

                        if close_pos:
                            rpl = calc_pl(pos, curr_px, pos_px, eval_config.SLIPPAGE)
                            cash += rpl

                            if PRINT_TRADES:
                                stk_idxs = np.nonzero(pos)[0]
                                for l in range(stk_idxs.shape[0]):
                                    stk_idx = stk_idxs[l]
                                    trade_ticker = tickers[stk_idx]
                                    trade_px = curr_px[stk_idx]
                                    trade_pos = -pos[stk_idx]
                                    trades.append({
                                        'date': date,
                                        'ticker': trade_ticker,
                                        'pos': trade_pos,
                                        'px': trade_px,
                                        'action': 'close',
                                        'cash flow': -trade_pos * trade_px,
                                        'capital': cash if l == stk_idxs.shape[0] - 1 else 0
                                    })

                            pos[:] = 0

                        if open_pos:
                            pos_mask = tradeable & in_snp

                            if prev_pos_mask is None:
                                prev_pos_mask = pos_mask
                            if prev_px is None:
                                prev_px = curr_px

                            pos_mask &= prev_pos_mask

                            decline = np.zeros(pos_mask.shape)
                            decline[pos_mask] = (curr_px[pos_mask] - prev_px[pos_mask]) / prev_px[pos_mask]

                            if DECLINE_LONG_FACTOR != 1:
                                net_long_mask = np.full(pos_mask.shape, False, dtype=np.bool)
                                net_long_mask[pos_mask] = prediction[pos_mask] > 0
                            else:
                                net_long_mask = np.full(pos_mask.shape, True, dtype=np.bool)

                            if DECLINE_SHORT_FACTOR != 1:
                                net_short_mask = np.full(pos_mask.shape, False, dtype=np.bool)
                                net_short_mask[pos_mask] = prediction[pos_mask] < 0
                            else:
                                net_short_mask = np.full(pos_mask.shape, True, dtype=np.bool)

                            if DECLINE_LONG_FACTOR != 0:
                                decline_long_mask = np.full(pos_mask.shape, False, dtype=np.bool)
                                decline_long_mask[pos_mask] = decline[pos_mask] < 0
                            else:
                                decline_long_mask = np.full(pos_mask.shape, True, dtype=np.bool)

                            if DECLINE_SHORT_FACTOR != 0:
                                decline_short_mask = np.full(pos_mask.shape, False, dtype=np.bool)
                                decline_short_mask[pos_mask] = decline[pos_mask] > 0
                            else:
                                decline_short_mask = np.full(pos_mask.shape, True, dtype=np.bool)

                            long_pos_mask = net_long_mask & decline_long_mask
                            short_pos_mask = net_short_mask & decline_short_mask

                            long_metric = np.zeros(pos_mask.shape)
                            short_metric = np.zeros(pos_mask.shape)

                            long_metric[long_pos_mask] = DECLINE_LONG_FACTOR * -decline[long_pos_mask] + (
                                                                                                             1 - DECLINE_LONG_FACTOR) * \
                                                                                                         prediction[
                                                                                                             long_pos_mask]
                            short_metric[short_pos_mask] = DECLINE_SHORT_FACTOR * -decline[short_pos_mask] + (
                                                                                                                 1 - DECLINE_SHORT_FACTOR) * \
                                                                                                             prediction[
                                                                                                                 short_pos_mask]

                            long_candidates = np.sum(long_pos_mask)
                            long_metric_sorted = np.sort(long_metric)
                            long_bound_idx = max(-LONG_STOCKS, -long_candidates)
                            long_bound = long_metric_sorted[long_bound_idx]
                            if long_candidates == 0:
                                long_bound = np.iinfo(np.int32).max

                            short_candidates = np.sum(short_pos_mask)
                            short_metric_sorted = np.sort(short_metric)
                            short_bound_idx = min(SHORT_STOCKS - 1, short_candidates - 1)
                            short_bound = short_metric_sorted[short_bound_idx]
                            if short_candidates == 0:
                                short_bound = np.iinfo(np.int32).max

                            long_pos_mask &= long_metric >= long_bound
                            short_pos_mask &= short_metric <= short_bound

                            long_stocks = np.sum(long_pos_mask)
                            short_stocks = np.sum(short_pos_mask)
                            num_stks = long_stocks + short_stocks

                            if RECAP:
                                if long_stocks != 0:
                                    pos[long_pos_mask] = LONG_PCT * NET_BET * eq[i - 1] / long_stocks / curr_px[
                                        long_pos_mask]
                                if short_stocks != 0:
                                    pos[short_pos_mask] = -SHORT_PCT * NET_BET * eq[i - 1] / short_stocks / curr_px[
                                        short_pos_mask]
                            else:
                                if long_stocks != 0:
                                    pos[long_pos_mask] = LONG_PCT * NET_BET / long_stocks / curr_px[long_pos_mask]
                                if short_stocks != 0:
                                    pos[short_pos_mask] = -SHORT_PCT * NET_BET / short_stocks / curr_px[short_pos_mask]

                            # prediction_sorted = np.sort(np.abs(prediction[pos_mask]))
                            # bound_idx = max(-SELECTION, -prediction_sorted.shape[0])
                            # bound = prediction_sorted[bound_idx]
                            # prediction_pos_mask = prediction >= bound
                            # prediction_pos_mask |= prediction <= -bound
                            # pos_mask &= prediction_pos_mask
                            #
                            # num_stks = np.sum(pos_mask)
                            # if RECAP:
                            #     pos[pos_mask] = NET_BET * eq[i - 1] / num_stks / curr_px[pos_mask] * np.sign(
                            #         prediction[pos_mask])
                            # else:
                            #     pos[pos_mask] = NET_BET / num_stks / curr_px[pos_mask] * np.sign(prediction[pos_mask])

                            pos_px = curr_px

                            if PRINT_TRADES:
                                stk_idxs = np.nonzero(pos)[0]
                                for l in range(stk_idxs.shape[0]):
                                    stk_idx = stk_idxs[l]
                                    trade_ticker = tickers[stk_idx]
                                    trade_px = curr_px[stk_idx]
                                    trade_pos = pos[stk_idx]
                                    trades.append({
                                        'date': date,
                                        'ticker': trade_ticker,
                                        'pos': trade_pos,
                                        'px': trade_px,
                                        'action': 'open',
                                        'cash flow': -trade_pos * trade_px,
                                        'capital': 0
                                    })

                        td_idx += 1

                        prev_px = curr_px
                        prev_pos_mask = pos_mask

                    urpl = calc_pl(pos, curr_px, pos_px, eval_config.SLIPPAGE)
                    es_urpl = calc_pl_simple(es_pos, es_close_px, es_pos_px, ES_PRICE_STEP)
                    vx_urpl = calc_pl_simple(vx_pos, vx_close_px, vx_pos_px, VX_PRICE_STEP)
                    nlv = cash + urpl + es_urpl + vx_urpl

                    eq[i] = nlv
                    ts[i] = get_date_timestamp(date)

                    if PRINT_WEEKLY_PL_CHANGES:
                        if trading_day:
                            track_close_pos, track_open_pos = track_strategy.decide(date, next_trading_date)
                            if track_close_pos:
                                close_nlv = nlv
                                close_date = date
                                weekly_balance.append((
                                    close_nlv - open_nlv,
                                    open_date,
                                    close_date))
                            if track_open_pos:
                                open_nlv = nlv
                                open_date = date

                if PRINT_PERIODIC_STAT or PRINT_DECLINE_GS:
                    test_eq = eq[after_2007_mask]
                    test_ts = ts[after_2007_mask]
                    period_beg_idx = 0
                    period_start_nlv_idx = 0
                    stat = []
                    gs_ch = []
                    gs_dd = []
                    for i in range(test_ts.shape[0]):
                        date = date_from_timestamp(test_ts[i])

                        period_beg_date = date_from_timestamp(test_ts[period_beg_idx])
                        period_start_nlv_date = date_from_timestamp(test_ts[period_start_nlv_idx])

                        if BY_QUARTER:
                            def check_new_period(period_beg_date, date):
                                quarter_beg_date = (period_beg_date.month - 1) // 3
                                quarter_date = (date.month - 1) // 3
                                if quarter_beg_date != quarter_date:
                                    return True
                                return False
                        else:
                            def check_new_period(period_beg_date, date):
                                return period_beg_date.month != date.month

                        if check_new_period(period_beg_date, date):
                            min_equity = INIT_NLV * np.min(test_eq[period_start_nlv_idx:i])
                            nlv_beg = INIT_NLV * test_eq[period_start_nlv_idx]
                            nlv_end = INIT_NLV * test_eq[i - 1]
                            pct_ch = (nlv_end - nlv_beg) / nlv_beg
                            dd = (min_equity - nlv_beg) / nlv_beg
                            period_end_date = date_from_timestamp(test_ts[i - 1])
                            period_beg_idx = i
                            period_start_nlv_idx = i - 1
                            if PRINT_DECLINE_GS:
                                gs_ch.append(pct_ch)
                                gs_dd.append(dd)
                            if PRINT_PERIODIC_STAT:
                                stat.append({
                                    "date beg": period_start_nlv_date,
                                    "date end": period_end_date,
                                    "nlv beg": nlv_beg,
                                    "nlv end": nlv_end,
                                    "dd": dd,
                                    "change pct": pct_ch,
                                })
                    if PRINT_DECLINE_GS:
                        avg_gs_ch = np.mean(np.array(gs_ch))
                        avg_gs_dd = np.mean(np.array(gs_dd))

                    if PRINT_PERIODIC_STAT:
                        stat_df = pd.DataFrame(stat)
                        stat_df = stat_df[['date beg', 'date end', 'nlv beg', 'nlv end', 'dd', 'change pct']]
                        stat_df.to_csv(
                            "data/%s_%s_stat.csv" % (
                                linear_combination_normalizer, "quaterly" if BY_QUARTER else "monthly"),
                            index=False)

                # plot eq
                folder_path, file_path = folders.get_adaptive_plot_path("DECLINE", decline_params, EPOCH)
                folders.create_dir(folder_path)

                fig = plots.plot_eq("total", eq[after_2007_mask], ts[after_2007_mask])
                fig.savefig(file_path)

                # if fig is None:
                #     fig, ax = plots.create_eq_ax("equity curve")
                #
                # plots.plot_serie(ax, eq[after_2007_mask], ts[after_2007_mask])

                if HIDE_PLOT:
                    plots.close_fig(fig)

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

                train = False

                # save stat
                if test:
                    test_dd, test_sharpe, test_y_avg = get_eq_params(eq[test_idxs], ts[test_idxs], RECAP)
                    if PRINT_DECLINE_GS:
                        print("%s dd %.2f%% y_avg %.2f%% sharpe %.2f avg_m_dd %.2f%% avg_m_ret %.2f%%" % (
                            decline_params, test_dd * 100, test_y_avg * 100, test_sharpe, avg_gs_dd * 100,
                            avg_gs_ch * 100))
                    else:
                        print("%s Test dd: %.2f%% y avg: %.2f%% sharpe: %.2f" % (
                            linear_combination_normalizer, test_dd * 100, test_y_avg * 100, test_sharpe))
                if train:
                    train_dd, train_sharpe, train_y_avg = get_eq_params(eq[train_idxs], ts[train_idxs], RECAP)
                    print("%s Train dd: %.2f%% y avg: %.2f%% sharpe: %.2f" % (
                        linear_combination_normalizer, train_dd * 100, train_y_avg * 100, train_sharpe))

        if PRINT_TRADES:
            pd.DataFrame(trades).to_csv("data/trades.csv", index=False)

        if PRINT_WEEKLY_PL_CHANGES:
            components = [[*x] for x in zip(*weekly_balance)]
            change = components[0]
            open_dates = components[1]
            close_dates = components[2]
            pd.DataFrame({
                CH_COL_NAME: change,
                OPEN_DATE_COL_NAME: open_dates,
                CLOSE_DATE_COL_NAME: close_dates
            }).to_csv(weekly_balance_file_name, index=False)

        if not HIDE_PLOT:
            plots.show_plots()
