import requests

from tiingo.config import TIINGO_API_TOKEN

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Token %s' % TIINGO_API_TOKEN
}


def get_historical_data(ticker, start_date, end_date):
    req_uri = "https://api.tiingo.com/tiingo/daily/{}/prices?startDate={}&endDate={}".format(ticker,
                                                                                             start_date.strftime(
                                                                                                 '%Y-%m-%d'),
                                                                                             end_date.strftime(
                                                                                                 '%Y-%m-%d'))
    # print(req_uri)
    response = requests.get(req_uri,
                            headers=headers)

    if response.ok:
        return response.json()
    else:
        print("{} response status code: {}".format(ticker, response.status_code))
    return None
