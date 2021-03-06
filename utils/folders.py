import os
from net.train_config import TrainConfig
from net.eval_config import EvalConfig
import shutil

def get_weights_path(train_config: TrainConfig):
    folder_path = 'data/nets/%s/weights' % train_config.DATA_FOLDER
    weights_path_prefix = '%s/%s' % (folder_path, 'weights')
    return folder_path, weights_path_prefix


def get_state_file_path(train_config: TrainConfig):
    folder_path = 'data/state/%s' % train_config.DATA_FOLDER
    file_path = '%s/state.pickle' % (folder_path)
    return folder_path, file_path

def get_train_progress_path(train_config: TrainConfig):
    folder_path = 'data/nets/%s' % train_config.DATA_FOLDER
    file_path = '%s/train_progress.csv' % (folder_path)
    return folder_path, file_path

def get_adaptive_compressed_predictions_path(ADAPTIVE_ALGO_NAME):
    folder_path = 'data/eval/%s' % ADAPTIVE_ALGO_NAME
    file_path = '%s/predictions.npz' % folder_path
    return folder_path, file_path

def get_adaptive_prediction_path(ADAPTIVE_ALGO_NAME, eval_config, ticker, epoch):
    folder_path = 'data/eval/%s/%s/%s/prediction' % (ADAPTIVE_ALGO_NAME, eval_config.DATA_FOLDER, ticker)
    file_path = '%s/%s_%s.csv' % (folder_path, ticker, epoch)
    return folder_path, file_path

def get_adaptive_plot_path(ADAPTIVE_ALGO_NAME, ticker, epoch):
    folder_path = 'data/eval/%s' % (ADAPTIVE_ALGO_NAME)
    file_path = '%s/%s_%d.png' % (folder_path, ticker, epoch)
    return folder_path, file_path

def get_adaptive_stat_path(ADAPTIVE_ALGO_NAME, epoch):
    folder_path = 'data/eval/%s' % (ADAPTIVE_ALGO_NAME)
    file_path = '%s/%d_stat.csv' % (folder_path, epoch)
    return folder_path, file_path

def get_prediction_path(train_config: TrainConfig, eval_config: EvalConfig, ticker, epoch):
    folder_path = 'data/eval/%s/%s/%s/prediction' % (train_config.DATA_FOLDER, eval_config.DATA_FOLDER, ticker)
    file_path = '%s/%s_%s.csv' % (folder_path, ticker, epoch)
    return folder_path, file_path


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def remove_dir(dir_name):
    try:
        shutil.rmtree(dir_name)
    except:
        pass
