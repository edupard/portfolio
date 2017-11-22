import copy

import utils.dates as dates
from stock_data.datasource import DataSource

class TrainConfig(object):
    DATA_FOLDER = ''
    DROPOUT_KEEP_RATE = 0.5
    LSTM_LAYERS = 3
    LSTM_LAYER_SIZE = 5
    FC_LAYERS = [30]
    BPTT_STEPS = 100
    BEG = dates.YR_00
    END = dates.YR_07
    FEATURE_FUNCTIONS = [
        DataSource.get_o_f,
        DataSource.get_c_f,
        DataSource.get_h_f,
        DataSource.get_l_f,
        DataSource.get_v_f,
        DataSource.get_dow_f,
        DataSource.get_moy_f,
    ]
    LABEL_FUNCTION = DataSource.get_5dy_f


_config_proto = TrainConfig()

def get_config_instance():
    return copy.deepcopy(_config_proto)

def get_train_config_bagging():
    config = get_config_instance()
    config.DATA_FOLDER = 'bagging'
    config.DROPOUT_KEEP_RATE = 1.0
    config.LSTM_LAYERS = 3
    config.LSTM_LAYER_SIZE = 30
    config.FC_LAYERS = [100]
    config.BPTT_STEPS = 100
    config.BEG = dates.YR_00
    config.END = dates.LAST_DATE
    config.FEATURE_FUNCTIONS = [
        DataSource.get_o_f,
        DataSource.get_c_f,
        DataSource.get_h_f,
        DataSource.get_l_f,
        DataSource.get_v_f,
        DataSource.get_dow_f,
        DataSource.get_moy_f,
    ]
    config.LABEL_FUNCTION = DataSource.get_5dy_f
    return config


def get_train_config_petri():
    config = get_config_instance()
    config.DATA_FOLDER = 'petri'
    config.DROPOUT_KEEP_RATE = 1.0
    config.LSTM_LAYERS = 3
    config.LSTM_LAYER_SIZE = 30
    config.FC_LAYERS = [100]
    config.BPTT_STEPS = 100
    config.BEG = dates.YR_00
    config.END = dates.YR_07
    config.FEATURE_FUNCTIONS = [
        DataSource.get_o_f,
        DataSource.get_c_f,
        DataSource.get_h_f,
        DataSource.get_l_f,
        DataSource.get_v_f,
        DataSource.get_dow_f,
        DataSource.get_moy_f,
    ]
    config.LABEL_FUNCTION = DataSource.get_5dy_f
    return config

def get_train_config_stacking():
    config = get_config_instance()
    config.DATA_FOLDER = 'petri'
    config.DROPOUT_KEEP_RATE = 1.0
    config.LSTM_LAYERS = 3
    config.LSTM_LAYER_SIZE = 30
    config.FC_LAYERS = [100]
    config.BPTT_STEPS = 100
    config.BEG = dates.YR_07
    config.END = dates.YR_08
    config.FEATURE_FUNCTIONS = [
        DataSource.get_o_f,
        DataSource.get_c_f,
        DataSource.get_h_f,
        DataSource.get_l_f,
        DataSource.get_v_f,
        DataSource.get_dow_f,
        DataSource.get_moy_f,
    ]
    config.LABEL_FUNCTION = DataSource.get_5dy_f
    return config
