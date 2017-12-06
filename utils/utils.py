import datetime

def date_from_timestamp(ts):
    return datetime.datetime.fromtimestamp(ts).date()

def get_date_timestamp(date):
    dt = datetime.datetime.combine(date, datetime.time.min)
    return dt.timestamp()


def is_same_week(date1, date2):
    _min = min(date1, date2)
    _max = max(date1, date2)
    if (_max - _min).days < (7 - _min.isoweekday()):
        return True
    return False

def convert_to_ib(ticker):
    return ticker.replace('-', ' ')