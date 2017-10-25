import copy

import utils.dates as dates
from stock_data.datasource import DataSource

class EvalConfig(object):
    DATA_FOLDER = ''
    BEG = dates.YR_00
    END = dates.YR_07
    BPTT_STEPS = 100

_config_proto = EvalConfig()

def get_config_instance():
    return copy.deepcopy(_config_proto)

def get_eval_config_petri_train_set():
    config = get_config_instance()
    config.DATA_FOLDER = 'train'
    config.BEG = dates.YR_00
    config.END = dates.YR_07
    return config

