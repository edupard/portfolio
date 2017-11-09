from stock_data.datasource import DataSource
from net.train_config import TrainConfig
from net.eval_config import EvalConfig, PosStrategy
from typing import List
import numpy as np
from utils.utils import date_from_timestamp
import datetime
import math
from utils.utils import is_same_week

class HoldMonFriPredictMonPosStrategyV1():
    def __init__(self):
        self.in_pos = False
        self.has_prediction = False

    def decide(self, date: datetime.date, next_trading_date: datetime.datetime):
        predict = False
        close_pos = False
        open_pos = False
        # check if we need to close position
        if self.in_pos:
            if next_trading_date is None or not is_same_week(date, next_trading_date):
                close_pos = True
                self.in_pos = False
        if not self.in_pos:
            if next_trading_date is not None:
                if is_same_week(date, next_trading_date):
                    predict = True
                    self.has_prediction = True
                    if self.has_prediction:
                        open_pos = True
                        self.in_pos = True

        return predict, close_pos, open_pos


class HoldMonFriPosStrategyV1():
    def __init__(self):
        self.in_pos = False
        self.has_prediction = False

    def decide(self, date: datetime.date, next_trading_date: datetime.datetime):
        predict = False
        close_pos = False
        open_pos = False
        # check if we predict
        if next_trading_date is not None and not is_same_week(date, next_trading_date):
            predict = True
            self.has_prediction = True
        # check if we need to close position
        if self.in_pos:
            if next_trading_date is None or not is_same_week(date, next_trading_date):
                close_pos = True
                self.in_pos = False
        if not self.in_pos:
            if next_trading_date is not None:
                if is_same_week(date, next_trading_date):
                    if self.has_prediction:
                        open_pos = True
                        self.in_pos = True

        return predict, close_pos, open_pos

class HoldFriFriPosStrategyV1():
    def __init__(self):
        self.in_pos = False
        self.has_prediction = False

    def decide(self, date: datetime.date, next_trading_date: datetime.datetime):
        predict = False
        close_pos = False
        open_pos = False
        # check if we predict
        if next_trading_date is not None and not is_same_week(date, next_trading_date):
            predict = True
            self.has_prediction = True
        # check if we need to close position
        if self.in_pos:
            if next_trading_date is None or not is_same_week(date, next_trading_date):
                close_pos = True
                self.in_pos = False
        if not self.in_pos:
            if next_trading_date is not None:
                if not is_same_week(date, next_trading_date):
                    if self.has_prediction:
                        open_pos = True
                        self.in_pos = True

        return predict, close_pos, open_pos

class PeriodicPosStrategyV1():
    def __init__(self, TRADES_FREQ):
        self.data_idx = 0
        self.TRADES_FREQ = TRADES_FREQ

    def decide(self, date: datetime.date, next_trading_date: datetime.datetime):
        predict = False
        open_pos = False
        close_pos = False

        if self.data_idx % self.TRADES_FREQ == 0:
            predict = True
            close_pos = True
            open_pos = True

        self.data_idx += 1

        return predict, open_pos, close_pos







class HoldMonFriPosStrategy():
    def __init__(self):
        self.in_pos = False

    def decide(self, date: datetime.date, next_trading_date: datetime.datetime):
        close_pos = False
        open_pos = False
        # check if we need to close position
        if self.in_pos:
            if next_trading_date is None or not is_same_week(date, next_trading_date):
                close_pos = True
                self.in_pos = False
        if not self.in_pos:
            if next_trading_date is not None:
                if is_same_week(date, next_trading_date):
                    open_pos = True
                    self.in_pos = True

        return close_pos, open_pos

class HoldFriFriPosStrategy():
    def __init__(self):
        self.in_pos = False

    def decide(self, date: datetime.date, next_trading_date: datetime.datetime):
        close_pos = False
        open_pos = False
        # check if we need to close position
        if self.in_pos:
            if next_trading_date is None or not is_same_week(date, next_trading_date):
                close_pos = True
                self.in_pos = False
        if not self.in_pos:
            if next_trading_date is not None:
                if not is_same_week(date, next_trading_date):
                    open_pos = True
                    self.in_pos = True

        return close_pos, open_pos

class PeriodicPosStrategy():
    def __init__(self, TRADES_FREQ):
        self.data_idx = 0
        self.TRADES_FREQ = TRADES_FREQ

    def decide(self, date: datetime.date, next_trading_date: datetime.datetime):
        open_pos = False
        close_pos = False

        if self.data_idx % self.TRADES_FREQ == 0:
            close_pos = True
            open_pos = True

        self.data_idx += 1

        return open_pos, close_pos


def calc_pl(pos,curr_px,pos_px, slippage):
    return pos * (curr_px * (1 - math.copysign(1, pos) * slippage) - pos_px * (1 + math.copysign(1, pos) * slippage))


def build_eq(ds: DataSource, predictions, eval_config: EvalConfig):
    beg_data_idx, end_data_idx = ds.get_data_range(eval_config.BEG, eval_config.END)

    if beg_data_idx is None or end_data_idx is None:
        return None

    data_points = end_data_idx - beg_data_idx

    cash = 0
    pos = 0
    pos_px = 0
    if eval_config.POS_STRATEGY == PosStrategy.MON_FRI:
        pos_strategy = HoldMonFriPosStrategy()
    elif eval_config.POS_STRATEGY == PosStrategy.FRI_FRI:
        pos_strategy = HoldFriFriPosStrategy()
    elif eval_config.POS_STRATEGY == PosStrategy.PERIODIC:
        pos_strategy = PeriodicPosStrategy(eval_config.TRADES_FREQ)


    eq = np.zeros(data_points)

    for idx in range(data_points):
        data_idx = beg_data_idx + idx
        p = predictions[idx]

        curr_px = ds.get_a_c(data_idx)
        date = date_from_timestamp(ds.get_ts(data_idx))
        next_trading_date = None
        if idx < data_points - 1:
            next_trading_date = date_from_timestamp(ds.get_ts(data_idx + 1))

        close_pos, open_pos = pos_strategy.decide(date, next_trading_date)

        if close_pos:
            rpl = calc_pl(pos,curr_px,pos_px, eval_config.SLIPPAGE)
            cash += rpl
            pos = 0
        if open_pos:
            pos_px = curr_px
            pos = 1 / curr_px * np.sign(p)

        urpl = calc_pl(pos,curr_px,pos_px, eval_config.SLIPPAGE)
        nlv = cash + urpl

        eq[idx] = nlv

    return eq, ds.get_ts(beg_data_idx, end_data_idx)


# def build_eq(dss: List[DataSource], train_config: TrainConfig, eval_config: EvalConfig):
#     data_ranges = []
#     total_length = 0
#     for ds in dss:
#         r = ds.get_data_range(train_config.BEG, train_config.END)
#         b, e = r
#         data_ranges.append(r)
#         if b is not None and e is not None:
#             total_length += e - b
#
#     batch_size = 1
#     # reset state
#     curr_progress = 0
#     processed = 0
#
#     for ds, r in zip(dss, data_ranges):
#         beg_data_idx, end_data_idx = r
#         if beg_data_idx is None or end_data_idx is None:
#             continue
#         data_points = end_data_idx - beg_data_idx
#
#         pos_date = None
#         pos = 0
#         pos_px = 0
#         pos_strategy = PosStrategy(eval_config)
#
#
#         # load predictions
#         predictions = np.zeros(data_points)
#         eq = np.zeros(data_points)
#
#         for idx in range(data_points):
#             data_idx = beg_data_idx + idx
#             p = predictions[idx]
#
#             curr_px = ds.get_a_c(data_idx)
#             date = date_from_timestamp(ds.get_ts(data_idx))
#
#             close_pos, open_pos = pos_strategy.decide(date)
#
#             if close_pos:
#                 rpl = np.sum(pos * (curr_px - pos_px))
#                 cash += rpl
#                 pos[:] = 0
#             if open_pos:
#                 pos_px = curr_px
#                 pos_mask = port_mask[:, data_idx]
#                 num_stks = np.sum(pos_mask)
#                 if get_config().CAPM:
#                     exp, cov = env.get_exp_and_cov(pos_mask,
#                                                    global_data_idx - get_config().COVARIANCE_LENGTH + 1,
#                                                    global_data_idx)
#                     exp = get_config().REBALANCE_FREQ * exp
#                     cov = get_config().REBALANCE_FREQ * get_config().REBALANCE_FREQ * cov
#                     if get_config().CAPM_USE_NET_PREDICTIONS:
#                         exp = predictions[:, i, 0][pos_mask]
#
#                     capm = Capm(num_stks)
#                     capm.init()
#
#                     best_sharpe = None
#                     best_weights = None
#                     best_constriant = None
#                     while i <= 10000:
#                         w, sharpe, constraint = capm.get_params(exp, cov)
#                         # print("Iteration: %d Sharpe: %.2f Constraint: %.6f" % (i, sharpe, constraint))
#                         if w is None:
#                             break
#                         if best_sharpe is None or sharpe >= best_sharpe:
#                             best_weights = w
#                             best_sharpe = sharpe
#                             best_constriant = constraint
#                         capm.fit(exp, cov)
#                         capm.rescale_weights()
#
#                         i += 1
#                     date = datetime.datetime.fromtimestamp(raw_dates[data_idx]).date()
#                     print("Date: %s sharpe: %.2f constraint: %.6f" %
#                           (date.strftime('%Y-%m-%d'),
#                            best_sharpe,
#                            best_constriant)
#                           )
#
#                     pos[pos_mask] = best_weights / curr_px[pos_mask]
#                 else:
#                     pos[pos_mask] = 1 / num_stks / curr_px[pos_mask] * np.sign(predictions[pos_mask, i, 0])
#
#             urpl = np.sum(pos * (curr_px - pos_px))
#             nlv = cash + urpl
#
#             eq[data_idx] = nlv
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#         batches = data_points // train_config.BPTT_STEPS if data_points % train_config.BPTT_STEPS == 0 else data_points // train_config.BPTT_STEPS + 1
#         for b in range(batches):
#             b_d_i = beg_data_idx + b * train_config.BPTT_STEPS
#             e_d_i = beg_data_idx + (b + 1) * train_config.BPTT_STEPS
#             e_d_i = min(e_d_i, end_data_idx)
#
#             seq_len = e_d_i - b_d_i
#
#             for f in range(num_features):
#                 input[0, :seq_len, f] = train_config.FEATURE_FUNCTIONS[f](ds, b_d_i, e_d_i)
#
#             labels[0, :seq_len] = train_config.LABEL_FUNCTION(ds, b_d_i, e_d_i)
#
#             if seq_len < train_config.BPTT_STEPS:
#                 _input = input[:, :seq_len, :]
#                 _labels = labels[:, :seq_len]
#                 _mask = mask[:, :seq_len]
#
#             else:
#                 _input = input
#                 _labels = labels
#                 _mask = mask
#
#             state, sse, predictions = net.fit(state, _input, _labels, _mask.astype(np.float32))
#             if math.isnan(sse):
#                 raise "Nan"
#             total_sse += sse
#             total_sse_members += np.sum(_mask)
#             processed += seq_len
#             curr_progress = progress.print_progress(curr_progress, processed, total_length)
#
#     progress.print_progess_end()
#     avg_loss = math.sqrt(total_sse / total_sse_members)
#     return avg_loss
