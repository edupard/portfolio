import quandl

from quandl_data.config import AUTH_TOKEN

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

def get_snp_tickers():
    result = []
    for y in range(1997, 2018):
        for qt in QUARTER_TICKERS:
            ticker = "ES%s%d" % (qt, y)
            quandle_id = "CME/%s" % ticker
            result.append((ticker, quandle_id))
    return result


def get_vix_tickers():
    result = []
    for y in range(2004, 2018):
        for mt in MONTH_TICKERS:
            ticker = "VX%s%d" % (mt, y)
            quandle_id = "CBOE/%s" % ticker
            result.append((ticker, quandle_id))
    return result

def get_historical_data(ticker):
    return quandl.get(ticker, authtoken=AUTH_TOKEN)