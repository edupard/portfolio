import numpy as np
import math
from typing import List

from net.net_shiva import NetShiva
from stock_data.datasource import DataSource
import utils.progress as progress
from net.train_config import TrainConfig
from net.eval_config import EvalConfig


def predict_v2(net: NetShiva, dss: List[DataSource], train_config: TrainConfig, eval_config: EvalConfig):
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

    progress.print_progess_end()
    avg_loss = math.sqrt(total_sse / total_sse_members)
    return avg_loss, predictions_history

def predict(net: NetShiva, ds: DataSource, train_config: TrainConfig, eval_config: EvalConfig):
    r = ds.get_data_range(eval_config.BEG, eval_config.END)
    beg_data_idx, end_data_idx = r
    if beg_data_idx is None or end_data_idx is None:
        return
    total_length = end_data_idx - beg_data_idx

    predictions_history = np.zeros([total_length, 2])

    batch_size = 1
    # reset state
    curr_progress = 0
    processed = 0
    total_sse = 0
    total_sse_members = 0

    num_features = len(train_config.FEATURE_FUNCTIONS)
    input = np.zeros([batch_size, eval_config.BPTT_STEPS, num_features])
    mask = np.ones([batch_size, eval_config.BPTT_STEPS])
    labels = np.zeros([batch_size, eval_config.BPTT_STEPS])

    state = net.zero_state(batch_size=batch_size)
    batches = total_length // eval_config.BPTT_STEPS if total_length % eval_config.BPTT_STEPS == 0 else total_length // eval_config.BPTT_STEPS + 1
    for b in range(batches):
        b_d_i = beg_data_idx + b * eval_config.BPTT_STEPS
        e_d_i = beg_data_idx + (b + 1) * eval_config.BPTT_STEPS
        e_d_i = min(e_d_i, end_data_idx)

        seq_len = e_d_i - b_d_i

        for f in range(num_features):
            input[0, :seq_len, f] = train_config.FEATURE_FUNCTIONS[f](ds, b_d_i, e_d_i)

        labels[0, :seq_len] = train_config.LABEL_FUNCTION(ds, b_d_i, e_d_i)

        if seq_len < eval_config.BPTT_STEPS:
            _input = input[:, :seq_len, :]
            _labels = labels[:, :seq_len]
            _mask = mask[:, :seq_len]

        else:
            _input = input
            _labels = labels
            _mask = mask

        state, sse, predictions = net.eval(state, _input, _labels, _mask.astype(np.float32))
        total_sse += sse
        total_sse_members += np.sum(_mask)
        b_i = b_d_i - beg_data_idx
        e_i = e_d_i - beg_data_idx
        predictions_history[b_i:e_i, 0] = _labels
        predictions_history[b_i:e_i, 1] = predictions.reshape([-1])
        processed += seq_len
        curr_progress = progress.print_progress(curr_progress, processed, total_length)

    progress.print_progess_end()

    avg_loss = math.sqrt(total_sse / total_sse_members)

    return avg_loss, predictions_history