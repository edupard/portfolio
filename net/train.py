import numpy as np
import math
from typing import List

from net.net_shiva import NetShiva
from stock_data.datasource import DataSource
import utils.progress as progress
from net.train_config import TrainConfig
from net.eval_config import EvalConfig



def train_epoch(net: NetShiva, dss: List[DataSource], train_config: TrainConfig):
    # here we use online train strategy, i.e. iterate over all datasources and train sequentially
    # it perfectly suit when you train net on a single datasource and reasonable when you have several datasources
    # TODO: use another batch strategy for multiple datasources
    # 1. all datasources in one batch, spread data by time
    # 2. all datasources in one batch, spread data approximately uniformelly: calc max datasource len and use random start for individual stocks
    # 3. Specify bacth size, track hidden state for each datasource, select datasources into minibatch by probabiliy proportional to unprocessed data
    # https://stackoverflow.com/questions/2140787/select-k-random-elements-from-a-list-whose-elements-have-weights

    data_ranges = []
    total_length = 0
    for ds in dss:
        r = ds.get_data_range(train_config.BEG, train_config.END)
        b, e = r
        data_ranges.append(r)
        if b is not None and e is not None:
            total_length += e - b

    batch_size = 1
    # reset state
    curr_progress = 0
    processed = 0
    total_sse = 0
    total_sse_members = 0

    num_features = len(train_config.FEATURE_FUNCTIONS)
    input = np.zeros([batch_size, train_config.BPTT_STEPS, num_features])
    mask = np.ones([batch_size, train_config.BPTT_STEPS])
    labels = np.zeros([batch_size, train_config.BPTT_STEPS])

    for ds_idx in range(len(dss)):
        ds = dss[ds_idx]
        state = net.zero_state(batch_size=batch_size)
        beg_data_idx, end_data_idx = data_ranges[ds_idx]
        if beg_data_idx is None or end_data_idx is None:
            continue
        data_points = end_data_idx - beg_data_idx
        batches = data_points // train_config.BPTT_STEPS if data_points % train_config.BPTT_STEPS == 0 else data_points // train_config.BPTT_STEPS + 1
        for b in range(batches):
            b_d_i = beg_data_idx + b * train_config.BPTT_STEPS
            e_d_i = beg_data_idx + (b + 1) * train_config.BPTT_STEPS
            e_d_i = min(e_d_i, end_data_idx)

            seq_len = e_d_i - b_d_i

            for f in range(num_features):
                input[0, :seq_len, f] = train_config.FEATURE_FUNCTIONS[f](ds, b_d_i, e_d_i)

            labels[0, :seq_len] = train_config.LABEL_FUNCTION(ds, b_d_i, e_d_i)

            if seq_len < train_config.BPTT_STEPS:
                _input = input[:, :seq_len, :]
                _labels = labels[:, :seq_len]
                _mask = mask[:, :seq_len]

            else:
                _input = input
                _labels = labels
                _mask = mask

            state, sse, predictions = net.fit(state, _input, _labels, _mask.astype(np.float32))
            if math.isnan(sse):
                raise "Nan"
            total_sse += sse
            total_sse_members += np.sum(_mask)
            processed += seq_len
            curr_progress = progress.print_progress(curr_progress, processed, total_length)

    progress.print_progess_end()
    avg_loss = math.sqrt(total_sse / total_sse_members)
    return avg_loss

def train_stack_epoch(net: NetShiva, dss: List[DataSource], train_config: TrainConfig, weak_predictors, eval_config: EvalConfig):
    data_ranges = []
    predictions_data_ranges = []
    total_length = 0
    for ds in dss:
        r = ds.get_data_range(train_config.BEG, train_config.END)
        b, e = r
        data_ranges.append(r)
        r = ds.get_data_range(eval_config.BEG, eval_config.END)
        predictions_data_ranges.append(r)

        if b is not None and e is not None:
            total_length += e - b

    batch_size = 1
    # reset state
    curr_progress = 0
    processed = 0
    total_sse = 0
    total_sse_members = 0

    num_features = len(train_config.FEATURE_FUNCTIONS)
    input = np.zeros([batch_size, train_config.BPTT_STEPS, num_features])
    mask = np.ones([batch_size, train_config.BPTT_STEPS])
    labels = np.zeros([batch_size, train_config.BPTT_STEPS])

    for ds_idx in range(len(dss)):
        ds = dss[ds_idx]
        state = net.zero_state(batch_size=batch_size)
        beg_data_idx, end_data_idx = data_ranges[ds_idx]
        beg_pred_idx, end_pred_idx = predictions_data_ranges[ds_idx]
        ds.load_weak_predictions(weak_predictors, beg_pred_idx, end_pred_idx)

        if beg_data_idx is None or end_data_idx is None:
            continue
        data_points = end_data_idx - beg_data_idx
        batches = data_points // train_config.BPTT_STEPS if data_points % train_config.BPTT_STEPS == 0 else data_points // train_config.BPTT_STEPS + 1
        for b in range(batches):
            b_d_i = beg_data_idx + b * train_config.BPTT_STEPS
            e_d_i = beg_data_idx + (b + 1) * train_config.BPTT_STEPS
            e_d_i = min(e_d_i, end_data_idx)

            seq_len = e_d_i - b_d_i

            # for f in range(num_features):
            #     input[0, :seq_len, f] = train_config.FEATURE_FUNCTIONS[f](ds, b_d_i, e_d_i)

            input[0, :seq_len, :] = np.transpose(ds.get_weak_predictions(b_d_i, e_d_i))
            try:
                w_p_idx = weak_predictors.index(ds.ticker)
                input[0, :, w_p_idx] = 0
            except:
                pass


            labels[0, :seq_len] = train_config.LABEL_FUNCTION(ds, b_d_i, e_d_i)

            if seq_len < train_config.BPTT_STEPS:
                _input = input[:, :seq_len, :]
                _labels = labels[:, :seq_len]
                _mask = mask[:, :seq_len]

            else:
                _input = input
                _labels = labels
                _mask = mask

            state, sse, predictions = net.fit(state, _input, _labels, _mask.astype(np.float32))
            if math.isnan(sse):
                raise "Nan"
            total_sse += sse
            total_sse_members += np.sum(_mask)
            processed += seq_len
            curr_progress = progress.print_progress(curr_progress, processed, total_length)
        ds.unload_weak_predictions()

    progress.print_progess_end()
    avg_loss = math.sqrt(total_sse / total_sse_members)
    return avg_loss