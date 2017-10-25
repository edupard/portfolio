import unittest
import numpy as np

from net.net_shiva import NetShiva
import net.train_config as net_config
from stock_data.download import download_data
from stock_data.datasource import DataSource
from net.train_config import get_train_config_petri
from net.eval_config import get_eval_config_petri_train_set
from net.train import train_epoch
from net.predict import predict, predict_v2
import pandas as pd
import utils.folders as folders

TICKERS = [
    'A',
    'AAPL',
    'ABBV',
    'ABC',
    'ABT',
    'ACN',
    'ADBE',
    'ADI',
    'ADM',
    'ADP',
    'ADS',
    'ADSK',
    'AEE',
    'AEP',
    'AES',
    'AET',
    'AFL',
    'AGN',
    'AIG',
    'AIV',
    'AIZ',
    'AKAM',
    'ALL',
    'ALLE',
    'ALXN',
    'AMAT',
    'AME',
    'AMG',
    'AMGN',
    'AMP',
    'AMT',
    'AMZN',
    'ANDV',
    'ANTM',
    'AON',
    'APA',
    'APC',
    'APD',
    'APH',
    'ARNC',
    'AVB',
    'AVGO',
    'AVY',
    'AXP',
    'AZO',
    'BA',
    'BAC',
    'BAX',
    'BBT',
    'BBY',
    'BCR',
    'BDX',
    'BEN',
    'BF-B',
    'BHGE',
    'BIIB',
    'BK',
    'BLK',
    'BLL',
    'BMY',
    'BRK-B',
    'BSX',
    'BWA',
    'BXP',
    'C',
    'CA',
    'CAG',
    'CAH',
    'CAT',
    'CB',
    'CBG',
    'CBS',
    'CCI',
    'CCL',
    'CELG',
    'CERN',
    'CF',
    'CHK',
    'CHRW',
    'CI',
    'CINF',
    'CL',
    'CLX',
    'CMA',
    'CME',
    'CMG',
    'CMI',
    'CMS',
    'CNP',
    'COF',
    'COG',
    'COH',
    'COL',
    'COP',
    'COST',
    'CPB',
    'CRM',
    'CSCO',
    'CSX',
    'CTAS',
    'CTL',
    'CTSH',
    'CTXS',
    'CVS',
    'CVX',
    'D',
    'DAL',
    'DE',
    'DFS',
    'DG',
    'DGX',
    'DHI',
    'DHR',
    'DIS',
    'DISCA',
    'DISCK',
    'DLPH',
    'DLTR',
    'DOV',
    'DOW',
    'DPS',
    'DRI',
    'DTE',
    'DUK',
    'DVA',
    'DVN',
    'EA',
    'EBAY',
    'ECL',
    'ED',
    'EFX',
    'EIX',
    'EL',
    'EMN',
    'EMR',
    'EOG',
    'EQR',
    'EQT',
    'ES',
    'ESRX',
    'ESS',
    'ETFC',
    'ETN',
    'ETR',
    'EW',
    'EXC',
    'EXPD',
    'EXPE',
    'F',
    'FAST',
    'FB',
    'FCX',
    'FDX',
    'FE',
    'FFIV',
    'FIS',
    'FISV',
    'FITB',
    'FLIR',
    'FLR',
    'FLS',
    'FMC',
    'FOXA',
    'FTI',
    'GD',
    'GE',
    'GGP',
    'GILD',
    'GIS',
    'GLW',
    'GM',
    'GOOG',
    'GOOGL',
    'GPC',
    'GPS',
    'GRMN',
    'GS',
    'GT',
    'GWW',
    'HAL',
    'HAS',
    'HBAN',
    'HCA',
    'HCN',
    'HCP',
    'HD',
    'HES',
    'HIG',
    'HOG',
    'HON',
    'HP',
    'HPQ',
    'HRB',
    'HRL',
    'HRS',
    'HST',
    'HSY',
    'HUM',
    'IBM',
    'ICE',
    'IFF',
    'INTC',
    'INTU',
    'IP',
    'IPG',
    'IR',
    'IRM',
    'ISRG',
    'ITW',
    'IVZ',
    'JCI',
    'JEC',
    'JNJ',
    'JNPR',
    'JPM',
    'JWN',
    'K',
    'KEY',
    'KIM',
    'KLAC',
    'KMB',
    'KMI',
    'KMX',
    'KO',
    'KORS',
    'KR',
    'KSS',
    'KSU',
    'L',
    'LB',
    'LEG',
    'LEN',
    'LH',
    'LLL',
    'LLY',
    'LMT',
    'LNC',
    'LOW',
    'LRCX',
    'LUK',
    'LUV',
    'LVLT',
    'LYB',
    'M',
    'MA',
    'MAC',
    'MAR',
    'MAS',
    'MAT',
    'MCD',
    'MCHP',
    'MCK',
    'MCO',
    'MDLZ',
    'MDT',
    'MET',
    'MHK',
    'MKC',
    'MLM',
    'MMC',
    'MMM',
    'MNST',
    'MO',
    'MON',
    'MOS',
    'MPC',
    'MRK',
    'MRO',
    'MS',
    'MSFT',
    'MSI',
    'MTB',
    'MU',
    'MYL',
    'NAVI',
    'NBL',
    'NDAQ',
    'NEE',
    'NEM',
    'NFLX',
    'NFX',
    'NI',
    'NKE',
    'NLSN',
    'NOC',
    'NOV',
    'NRG',
    'NSC',
    'NTAP',
    'NTRS',
    'NUE',
    'NVDA',
    'NWL',
    'NWSA',
    'OKE',
    'OMC',
    'ORCL',
    'ORLY',
    'OXY',
    'PAYX',
    'PBCT',
    'PCAR',
    'PCG',
    'PCLN',
    'PDCO',
    'PEG',
    'PEP',
    'PFE',
    'PFG',
    'PG',
    'PGR',
    'PH',
    'PHM',
    'PKI',
    'PLD',
    'PM',
    'PNC',
    'PNR',
    'PNW',
    'PPG',
    'PPL',
    'PRGO',
    'PRU',
    'PSA',
    'PSX',
    'PVH',
    'PWR',
    'PX',
    'PXD',
    'QCOM',
    'RCL',
    'REGN',
    'RF',
    'RHI',
    'RHT',
    'RL',
    'ROK',
    'ROP',
    'ROST',
    'RRC',
    'RSG',
    'RTN',
    'SBUX',
    'SCG',
    'SCHW',
    'SEE',
    'SHW',
    'SJM',
    'SLB',
    'SNA',
    'SNI',
    'SO',
    'SPG',
    'SPGI',
    'SPLS',
    'SRCL',
    'SRE',
    'STI',
    'STT',
    'STX',
    'STZ',
    'SWK',
    'SYK',
    'SYMC',
    'SYY',
    'T',
    'TAP',
    'TEL',
    'TGT',
    'TIF',
    'TJX',
    'TMK',
    'TMO',
    'TRIP',
    'TROW',
    'TRV',
    'TSCO',
    'TSN',
    'TSS',
    'TWX',
    'TXN',
    'TXT',
    'UA',
    'UHS',
    'UNH',
    'UNM',
    'UNP',
    'UPS',
    'URI',
    'USB',
    'UTX',
    'V',
    'VAR',
    'VFC',
    'VIAB',
    'VLO',
    'VMC',
    'VNO',
    'VRSN',
    'VRTX',
    'VTR',
    'VZ',
    'WAT',
    'WBA',
    'WDC',
    'WEC',
    'WFC',
    'WHR',
    'WM',
    'WMB',
    'WMT',
    'WRK',
    'WU',
    'WY',
    'WYN',
    'WYNN',
    'XEC',
    'XEL',
    'XL',
    'XLNX',
    'XOM',
    'XRAY',
    'XRX',
    'XYL',
    'YUM',
    'ZBH',
    'ZION',
    'ZTS',
    'WLTW',
    'CHD',
    'CSRA',
    'ILMN',
    'SYF',
    'HPE',
    'VRSK',
    'FOX',
    'NWS',
    'CMCSA',
    'UAL',
    'ATVI',
    'SIG',
    'PYPL',
    'AAP',
    'KHC',
    'JBHT',
    'QRVO',
    'O',
    'AAL',
    'SLG',
    'HBI',
    'EQIX',
    'HSIC',
    'SWKS',
]


class NetTest(unittest.TestCase):

    def test_download_data(self):
        tickers = ['FOX']
        download_data(tickers, num_workers=20)

    def test_datasource(self):
        tickers = TICKERS
        dss = []
        for ticker in tickers:
            dss.append(DataSource(ticker))
        _debug = 0

    def test_variance(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7])
        cov = np.cov(arr)
        mean = np.mean(arr)
        _debug = 0

    def test_multi_stock_train(self):
        train_config = get_train_config_petri()
        train_config.DATA_FOLDER = "%s/%s" % (train_config.DATA_FOLDER, 'UNIVERSAL_NET')

        net = NetShiva(train_config)

        dss = []
        for ticker in TICKERS:
            dss.append(DataSource(ticker))

        net.init_weights(0)
        for e in range(600):
            net.save_weights(e)
            print("Training %d epoch..." % e)
            train_epoch(net, dss, train_config)
        net.save_weights(600)

    def test_train(self):
        ticker = 'AAPL'

        train_config = get_train_config_petri()
        train_config.DATA_FOLDER = "%s/%s" % (train_config.DATA_FOLDER, ticker)

        dss = [DataSource(ticker)]

        net = NetShiva(train_config)

        eval_config = get_eval_config_petri_train_set()

        net.init_weights()
        for e in range(2000):
            net.save_weights(e)

            print("Evaluating %d epoch..." % e)
            avg_loss, predictions_history = predict_v2(net, dss, train_config, eval_config)
            print("Avg loss: %.4f%% " % (avg_loss * 100))

            for ds, p_h in zip(dss, predictions_history):
                if p_h is None:
                    continue
                folder_path, file_path = folders.get_prediction_path(train_config, eval_config, ds.ticker, e)
                folders.create_dir(folder_path)

                df = pd.DataFrame({'ts': p_h[:, 0],'prediction': p_h[:, 1], 'observation': p_h[:, 2]})
                df.to_csv(file_path, index=False)


            print("Training %d epoch..." % e)
            avg_loss = train_epoch(net, dss, train_config)
            print("Avg loss: %.4f%% " % (avg_loss * 100))


        net.save_weights(600)


    def test_eval(self):
        ticker = 'A'

        train_config = get_train_config_petri()
        train_config.DATA_FOLDER = "%s/%s" % (train_config.DATA_FOLDER, ticker)

        dss = []
        for ticker in TICKERS:
            dss.append(DataSource(ticker))

        net = NetShiva(train_config)
        net.load_weights(600)

        eval_config = get_eval_config_petri_train_set()

        print("Evaluating %d epoch..." % 600)
        avg_loss, predictions_history = predict_v2(net, dss, train_config, eval_config)
        print("Avg loss: %.4f%% " % (avg_loss * 100))

        for ds, p_h in zip(dss, predictions_history):
            if p_h is None:
                continue
            folder_path, file_path = folders.get_prediction_path(train_config, eval_config, ds.ticker, 600)
            folders.create_dir(folder_path)

            df = pd.DataFrame({'ts': p_h[:, 0], 'prediction': p_h[:, 1], 'observation': p_h[:, 2]})
            df.to_csv(file_path, index=False)