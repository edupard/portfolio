import datetime

def date_from_timestamp(ts):
    return datetime.datetime.fromtimestamp(ts).date()

def get_date_timestamp(date):
    dt = datetime.datetime.combine(date, datetime.time.min)
    return dt.timestamp()
