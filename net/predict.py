import numpy as np
import math
from typing import List

from net.net_shiva import NetShiva
from stock_data.datasource import DataSource
import utils.progress as progress
from net.train_config import TrainConfig
from net.eval_config import EvalConfig


def stacking_net_predict(net: NetShiva, dss: List[DataSource], train_config: TrainConfig, eval_config: EvalConfig,
                         weak_predictors):
    data_ranges = []
    total_length = 0
    for ds in dss:
        r = ds.get_data_range(eval_config.BEG, eval_config.END)
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
    ts = np.zeros([batch_size, train_config.BPTT_STEPS])

    predictions_history = []

    for ds_idx in range(len(dss)):
        ds = dss[ds_idx]
        state = net.zero_state(batch_size=batch_size)
        beg_data_idx, end_data_idx = data_ranges[ds_idx]
        if beg_data_idx is None or end_data_idx is None:
            predictions_history.append(None)
            continue

        ds.load_weak_predictions(weak_predictors, beg_data_idx, end_data_idx)

        data_points = end_data_idx - beg_data_idx

        p_h = np.zeros([data_points, 3])

        batches = data_points // eval_config.BPTT_STEPS if data_points % eval_config.BPTT_STEPS == 0 else data_points // eval_config.BPTT_STEPS + 1
        for b in range(batches):
            b_d_i = beg_data_idx + b * eval_config.BPTT_STEPS
            e_d_i = beg_data_idx + (b + 1) * eval_config.BPTT_STEPS
            e_d_i = min(e_d_i, end_data_idx)

            seq_len = e_d_i - b_d_i

            # for f in range(num_features):
            #     input[0, :seq_len, f] = train_config.FEATURE_FUNCTIONS[f](ds, b_d_i, e_d_i)
            input[0, :seq_len, :] = np.transpose(ds.get_weak_predictions(b_d_i, e_d_i))

            labels[0, :seq_len] = train_config.LABEL_FUNCTION(ds, b_d_i, e_d_i)

            ts[0, :seq_len] = ds.get_ts(b_d_i, e_d_i)

            if seq_len < eval_config.BPTT_STEPS:
                _input = input[:, :seq_len, :]
                _labels = labels[:, :seq_len]
                _mask = mask[:, :seq_len]
                _ts = ts[:, :seq_len]

            else:
                _input = input
                _labels = labels
                _mask = mask
                _ts = ts

            state, sse, predictions = net.eval(state, _input, _labels, _mask.astype(np.float32))

            b_i = b_d_i - beg_data_idx
            e_i = e_d_i - beg_data_idx

            p_h[b_i:e_i, 0] = _ts
            p_h[b_i:e_i, 1] = predictions.reshape([-1])
            p_h[b_i:e_i, 2] = _labels

            if math.isnan(sse):
                raise "Nan"
            total_sse += sse
            total_sse_members += np.sum(_mask)
            processed += seq_len
            curr_progress = progress.print_progress(curr_progress, processed, total_length)

        ds.unload_weak_predictions()
        predictions_history.append(p_h)

    progress.print_progess_end()
    avg_loss = math.sqrt(total_sse / total_sse_members)
    return avg_loss, predictions_history


def predict(net: NetShiva, dss: List[DataSource], train_config: TrainConfig, eval_config: EvalConfig):
    data_ranges = []
    total_length = 0
    for ds in dss:
        r = ds.get_data_range(eval_config.BEG, eval_config.END)
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
    ts = np.zeros([batch_size, train_config.BPTT_STEPS])

    predictions_history = []
    states = []

    for ds_idx in range(len(dss)):
        ds = dss[ds_idx]
        state = net.zero_state(batch_size=batch_size)
        beg_data_idx, end_data_idx = data_ranges[ds_idx]
        if beg_data_idx is None or end_data_idx is None:
            predictions_history.append(None)
            continue
        data_points = end_data_idx - beg_data_idx

        p_h = np.zeros([data_points, 3])

        batches = data_points // eval_config.BPTT_STEPS if data_points % eval_config.BPTT_STEPS == 0 else data_points // eval_config.BPTT_STEPS + 1
        for b in range(batches):
            b_d_i = beg_data_idx + b * eval_config.BPTT_STEPS
            e_d_i = beg_data_idx + (b + 1) * eval_config.BPTT_STEPS
            e_d_i = min(e_d_i, end_data_idx)

            seq_len = e_d_i - b_d_i

            for f in range(num_features):
                input[0, :seq_len, f] = train_config.FEATURE_FUNCTIONS[f](ds, b_d_i, e_d_i)

            labels[0, :seq_len] = train_config.LABEL_FUNCTION(ds, b_d_i, e_d_i)

            ts[0, :seq_len] = ds.get_ts(b_d_i, e_d_i)

            if seq_len < eval_config.BPTT_STEPS:
                _input = input[:, :seq_len, :]
                _labels = labels[:, :seq_len]
                _mask = mask[:, :seq_len]
                _ts = ts[:, :seq_len]

            else:
                _input = input
                _labels = labels
                _mask = mask
                _ts = ts

            state, sse, predictions = net.eval(state, _input, _labels, _mask.astype(np.float32))

            b_i = b_d_i - beg_data_idx
            e_i = e_d_i - beg_data_idx

            p_h[b_i:e_i, 0] = _ts
            p_h[b_i:e_i, 1] = predictions.reshape([-1])
            p_h[b_i:e_i, 2] = _labels

            if math.isnan(sse):
                raise "Nan"
            total_sse += sse
            total_sse_members += np.sum(_mask)
            processed += seq_len
            curr_progress = progress.print_progress(curr_progress, processed, total_length)

        predictions_history.append(p_h)
        states.append(state)

    progress.print_progess_end()
    avg_loss = math.sqrt(total_sse / total_sse_members)
    return avg_loss, predictions_history, states


def batch_predict(net: NetShiva, dss: List[DataSource], train_config: TrainConfig, eval_config: EvalConfig):
    data_ranges = []
    data_points = 0
    for ds in dss:
        r = ds.get_data_range(eval_config.BEG, eval_config.END)
        b, e = r
        data_ranges.append(r)
        if b is not None and e is not None:
            if (e - b) > data_points:
                data_points = e - b

    batch_size = len(dss)
    # reset state
    curr_progress = 0
    processed = 0

    num_features = len(train_config.FEATURE_FUNCTIONS)
    input = np.zeros([batch_size, train_config.BPTT_STEPS, num_features])
    mask = np.ones([batch_size, train_config.BPTT_STEPS])
    labels = np.zeros([batch_size, train_config.BPTT_STEPS])
    seq_len = np.zeros([batch_size], dtype=np.int32)

    batches = data_points // eval_config.BPTT_STEPS if data_points % eval_config.BPTT_STEPS == 0 else data_points // eval_config.BPTT_STEPS + 1

    state = net.zero_state(batch_size=batch_size)

    predictions_history = np.zeros([batch_size, batches * eval_config.BPTT_STEPS])

    total_seq_len = np.zeros([batch_size], dtype=np.int32)
    for ds_idx in range(len(dss)):
        beg_data_idx, end_data_idx = data_ranges[ds_idx]
        if beg_data_idx is None or end_data_idx is None:
            continue
        t_s_l = end_data_idx - beg_data_idx
        total_seq_len[ds_idx] = t_s_l


    for b in range(batches):

        for ds_idx in range(len(dss)):
            ds = dss[ds_idx]
            beg_data_idx, end_data_idx = data_ranges[ds_idx]
            if beg_data_idx is None or end_data_idx is None:
                continue

            b_d_i = beg_data_idx + b * eval_config.BPTT_STEPS
            e_d_i = beg_data_idx + (b + 1) * eval_config.BPTT_STEPS
            b_d_i = min(b_d_i, end_data_idx)
            e_d_i = min(e_d_i, end_data_idx)

            s_l = e_d_i - b_d_i
            seq_len[ds_idx] = s_l

            for f in range(num_features):
                input[ds_idx, :s_l, f] = train_config.FEATURE_FUNCTIONS[f](ds, b_d_i, e_d_i)

            labels[ds_idx, :s_l] = train_config.LABEL_FUNCTION(ds, b_d_i, e_d_i)

        state, sse, predictions = net.eval(state, input, labels, mask.astype(np.float32), seq_len)

        predictions_history[:, b * eval_config.BPTT_STEPS: (b + 1) * eval_config.BPTT_STEPS] = predictions[:,:,0]

        if math.isnan(sse):
            raise "Nan"

        # TODO: not absolutelly correct
        processed += eval_config.BPTT_STEPS
        curr_progress = progress.print_progress(curr_progress, processed, data_points)

    weak_predictions = np.zeros([batch_size])

    for j in range(batch_size):
        weak_predictions[j] = predictions_history[j, total_seq_len[j] - 1]

    progress.print_progess_end()
    return weak_predictions
