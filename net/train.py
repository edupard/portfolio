import numpy as np

from net.net_shiva import NetShiva
from stock_data.datasource import DataSource
import utils.progress as progress
from net.train_config import TrainConfig
from net.eval_config import EvalConfig


def train(net: NetShiva, train_config: TrainConfig, eval_config: EvalConfig, dss, epoch_from, num_epochs):
    net.init_weights(epoch=epoch_from)
    for e in range(num_epochs):
        epoch = epoch_from + e
        print("Evaluating %d epoch..." % epoch)
        eval(net, dss[0], train_config, eval_config)
        epoch += 1
        print("Training %d epoch..." % epoch)
        train_epoch(net, dss, train_config)
        net.save_weights(epoch)


def train_epoch(net: NetShiva, dss, train_config: TrainConfig):
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
            b_i = b * train_config.BPTT_STEPS
            e_i = (b + 1) * train_config.BPTT_STEPS
            e_i = min(e_i, end_data_idx)

            seq_len = e_i - b_i

            for f in range(num_features):
                input[0, :seq_len, f] = train_config.FEATURE_FUNCTIONS[f](ds, b_i, e_i)

            labels[0, :seq_len] = train_config.LABEL_FUNCTION(ds, b_i, e_i)

            if seq_len < train_config.BPTT_STEPS:
                _input = input[:, :seq_len, :]
                _labels = labels[:, :seq_len]
                _mask = mask[:, :seq_len]

            else:
                _input = input
                _labels = labels
                _mask = mask

            state, loss, predictions = net.fit(state, _input, _labels, _mask.astype(np.float32))
            processed += seq_len
            curr_progress = progress.print_progress(curr_progress, processed, total_length)

    progress.print_progess_end()


def eval(net: NetShiva, ds, train_config: TrainConfig, eval_config: EvalConfig):
    r = ds.get_data_range(eval_config.BEG, eval_config.END)
    beg_data_idx, end_data_idx = r
    if beg_data_idx is None or end_data_idx is None:
        return
    total_length = end_data_idx - beg_data_idx

    batch_size = 1
    # reset state
    curr_progress = 0
    processed = 0

    num_features = len(train_config.FEATURE_FUNCTIONS)
    input = np.zeros([batch_size, eval_config.TIME_STEPS, num_features])
    mask = np.ones([batch_size, eval_config.TIME_STEPS])
    labels = np.zeros([batch_size, eval_config.TIME_STEPS])

    state = net.zero_state(batch_size=batch_size)
    data_points = end_data_idx - beg_data_idx
    batches = data_points // eval_config.TIME_STEPS if data_points % eval_config.TIME_STEPS == 0 else data_points // eval_config.TIME_STEPS + 1
    for b in range(batches):
        b_i = b * eval_config.TIME_STEPS
        e_i = (b + 1) * eval_config.TIME_STEPS
        e_i = min(e_i, end_data_idx)

        seq_len = e_i - b_i

        for f in range(num_features):
            input[0, :seq_len, f] = train_config.FEATURE_FUNCTIONS[f](ds, b_i, e_i)

        labels[0, :seq_len] = train_config.LABEL_FUNCTION(ds, b_i, e_i)

        if seq_len < eval_config.TIME_STEPS:
            _input = input[:, :seq_len, :]
            _labels = labels[:, :seq_len]
            _mask = mask[:, :seq_len]

        else:
            _input = input
            _labels = labels
            _mask = mask

        state, loss, predictions = net.eval(state, _input, _labels, _mask.astype(np.float32))
        processed += seq_len
        curr_progress = progress.print_progress(curr_progress, processed, total_length)

    progress.print_progess_end()


