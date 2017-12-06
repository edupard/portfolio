import copy

import utils.dates as dates
from stock_data.datasource import DataSource
from enum import Enum

class PosStrategy(Enum):
    MON_FRI = 1
    FRI_FRI = 2
    PERIODIC = 3

class EvalConfig(object):
    DATA_FOLDER = ''
    BEG = dates.YR_00
    END = dates.YR_07
    BPTT_STEPS = 100
    POS_STRATEGY = PosStrategy.PERIODIC
    TRADES_FREQ = None
    SLIPPAGE = 0.0005

_config_proto = EvalConfig()

def get_config_instance():
    return copy.deepcopy(_config_proto)

def get_eval_config_petri_train_set():
    config = get_config_instance()
    config.DATA_FOLDER = 'train'
    config.BEG = dates.YR_00
    config.END = dates.YR_07
    config.POS_STRATEGY = PosStrategy.MON_FRI
    config.TRADES_FREQ = None
    config.SLIPPAGE = 0.0005

    return config

def get_eval_config_petri_test_set():
    config = get_config_instance()
    config.DATA_FOLDER = 'test'
    config.BEG = dates.YR_07
    config.END = dates.LAST_DATE
    config.POS_STRATEGY = PosStrategy.MON_FRI
    config.TRADES_FREQ = None
    config.SLIPPAGE = 0.0005

    return config

def get_eval_config_petri_whole_set():
    config = get_config_instance()
    config.DATA_FOLDER = 'eval'
    config.BEG = dates.YR_00
    config.END = dates.LAST_DATE
    config.POS_STRATEGY = PosStrategy.MON_FRI
    config.TRADES_FREQ = None
    # config.SLIPPAGE = 0
    config.SLIPPAGE = 0.001
    # config.SLIPPAGE = 0.0005

    return config

def get_current_eval_config(beg, end):
    config = get_config_instance()
    config.DATA_FOLDER = 'eval'
    config.BEG = beg
    config.END = end
    config.POS_STRATEGY = PosStrategy.MON_FRI
    config.TRADES_FREQ = None
    config.SLIPPAGE = 0.001

    return config


