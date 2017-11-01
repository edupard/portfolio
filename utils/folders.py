import os
from net.train_config import TrainConfig
from net.eval_config import EvalConfig
import shutil

def get_weights_path(train_config: TrainConfig):
    folder_path = 'data/nets/%s/weights' % train_config.DATA_FOLDER
    weights_path_prefix = '%s/%s' % (folder_path, 'weights')
    return folder_path, weights_path_prefix

def get_train_progress_path(train_config: TrainConfig):
    folder_path = 'data/nets/%s' % train_config.DATA_FOLDER
    file_path = '%s/train_progress.csv' % (folder_path)
    return folder_path, file_path

def get_adaptive_prediction_path(ADAPTIVE_ALGO_NAME, eval_config, ticker, epoch):
    folder_path = 'data/eval/%s/%s/%s/prediction' % (ADAPTIVE_ALGO_NAME, eval_config.DATA_FOLDER, ticker)
    file_path = '%s/%s_%s.csv' % (folder_path, ticker, epoch)

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
