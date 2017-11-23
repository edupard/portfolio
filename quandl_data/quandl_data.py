import quandl
import pandas as pd

from quandl_data.config import AUTH_TOKEN
from utils.folders import create_dir

MONTH_TICKERS = [
    'F',
    'G',
    'H',
    'J',
    'K',
    'M',
    'N',
    'Q',
    'U',
    'V',
    'X',
    'Z',
]

QUARTER_TICKERS = [
    'H',
    'M',
    'U',
    'Z',
]

def get_es_tickers():
    result = []
    for y in range(1997, 2018):
        for qt in QUARTER_TICKERS:
            ticker = "ES%s%d" % (qt, y)
            quandle_id = "CME/%s" % ticker
            result.append((ticker, quandle_id))
    return result


def get_vx_tickers():
    result = []
    for y in range(2004, 2019):
        for mt in MONTH_TICKERS:
            ticker = "VX%s%d" % (mt, y)
            quandle_id = "CBOE/%s" % ticker
            result.append((ticker, quandle_id))
    return result


def get_historical_data(ticker):
    return quandl.get(ticker, authtoken=AUTH_TOKEN)

def get_vx_data_path(ticker):
    folder_name = "data/futures/VX"
    file_name = "%s/%s.csv" % (folder_name, ticker)
    return file_name, folder_name

def download_vx_data():
    for t, q_id in get_vx_tickers():
        print("Downloading %s" % t)
        try:
            ds = get_historical_data(q_id)
            file_name, folder_name = get_vx_data_path(t)
            create_dir(folder_name)
            ds.to_csv(file_name)
        except:
            print("No data")
            continue

def get_es_data_path(ticker):
    folder_name = "data/futures/ES"
    file_name = "%s/%s.csv" % (folder_name, ticker)
    return file_name, folder_name


def download_es_data():
    for t, q_id in get_es_tickers():
        print("Downloading %s" % t)
        try:
            ds = get_historical_data(q_id)
            file_name, folder_name = get_es_data_path(t)
            create_dir(folder_name)
            ds.to_csv(file_name)
        except:
            print("No data")
            continue


class VxDataSouce(object):

    def __init__(self, ticker):
        self.ticker = ticker
        file_name, folder_name = get_vx_data_path(ticker)
        self.df = None
        try:
            self.df = df = pd.read_csv(file_name)
            self.volume = df['Total Volume'].values
            self.close_px = df.Close.values
            df.Date = pd.to_datetime(df['Trade Date'], format='%Y-%m-%d').dt.date
            self.date = df.Date
            trading_day_mask = self.volume != 0
            self.volume = self.volume[trading_day_mask]
            self.close_px = self.close_px[trading_day_mask]
            self.date = self.date[trading_day_mask]
        except:
            pass

    def get_close_px(self):
        if self.df is not None:
            return self.close_px
        return None

    def get_dates(self):
        if self.df is not None:
            return self.date
        return None


class EsDataSouce(object):

    def __init__(self, ticker):
        self.ticker = ticker
        file_name, folder_name = get_es_data_path(ticker)
        self.df = None
        try:
            self.df = df = pd.read_csv(file_name)
            self.volume = df.Volume.values
            self.close_px = df.Settle.values
            df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d').dt.date
            self.date = df.Date
            trading_day_mask = self.volume != 0
            self.volume = self.volume[trading_day_mask]
            self.close_px = self.close_px[trading_day_mask]
            self.date = self.date[trading_day_mask]
        except:
            pass

    def get_close_px(self):
        if self.df is not None:
            return self.close_px
        return None

    def get_dates(self):
        if self.df is not None:
            return self.date
        return None
