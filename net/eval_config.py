import copy

import utils.dates as dates
from stock_data.datasource import DataSource

class EvalConfig(object):
    DATA_FOLDER = ''
    BEG = dates.YR_00
    END = dates.YR_07
    BPTT_STEPS = 100
    OPEN_POS_DOW = 1
    HOLD_POS_DAYS = 4

_config_proto = EvalConfig()

def get_config_instance():
    return copy.deepcopy(_config_proto)

def get_eval_config_petri_train_set():
    config = get_config_instance()
    config.DATA_FOLDER = 'train'
    config.BEG = dates.YR_00
    config.END = dates.YR_07
    config.OPEN_POS_DOW = 1
    config.HOLD_POS_DAYS = 4
    config.REBALANCE_FREQ = None

    return config

def get_eval_config_petri_test_set():
    config = get_config_instance()
    config.DATA_FOLDER = 'test'
    config.BEG = dates.YR_07
    config.END = dates.LAST_DATE
    config.OPEN_POS_DOW = 1
    config.HOLD_POS_DAYS = 4
    config.REBALANCE_FREQ = None

    return config

def get_eval_config_petri_whole_set():
    config = get_config_instance()
    config.DATA_FOLDER = 'eval'
    config.BEG = dates.YR_00
    config.END = dates.LAST_DATE
    config.OPEN_POS_DOW = 1
    config.HOLD_POS_DAYS = 4
    config.REBALANCE_FREQ = None

    return config



