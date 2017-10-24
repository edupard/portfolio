import utils.dates as dates
import utils.folders as folders

class Config(object):
    DATA_BEG = dates.YR_80
    DATA_END = dates.LAST_DATE
    FOLDER = folders.DATA_FOLDER

    def get_dir_name(self, ticker):
        return '%s/stocks/%s' % (self.FOLDER, ticker)

    def get_data_file_name(self, ticker):
        return '%s/%s.csv' % (self.get_dir_name(ticker), ticker)

    def get_dump_file_name(self, ticker):
        return '%s/%s.npz' % (self.get_dir_name(ticker), ticker)

_config = Config()

def get_config():
    return _config