import os

TEMP_FOLDER = 'temp'
DATA_FOLDER = 'data'


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def remove_dir(dir_name):
    os.removedirs(dir_name)