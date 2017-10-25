import pandas as pd
import datetime
import math

WIKI_TIINGO_TICKERS_MAP = {
    'EK' : 'KODK',
    'SLE' : 'HSH'
}

def rename_wiki_ticker_to_tiingo_ticker(wiki_ticker):
    tiingo_ticker = WIKI_TIINGO_TICKERS_MAP.get(wiki_ticker, wiki_ticker)
    return tiingo_ticker.replace('.', '-')

def get_snp_hitorical_components_tickers():
    snp_mask_df = pd.read_csv('data/snp/snp_mask.csv')
    return snp_mask_df.ticker.unique()

def get_snp_tickers():
    snp_df = pd.read_csv('data/snp/snp.csv')
    return snp_df.ticker.unique()

def generate_snp_mask():
    snp_df = pd.read_csv('data/snp/snp.csv')
    snp_df.added = pd.to_datetime(snp_df.added, format='%d.%m.%Y').dt.date

    snp_changes_df = pd.read_csv('data/snp/snp_changes.csv')
    snp_changes_df = snp_changes_df.assign(Date=pd.to_datetime(snp_changes_df.Date, format='%B %d, %Y').dt.date)
    snp_changes_df = snp_changes_df.rename(index=str, columns={"Date": "date"})

    changes = {}

    def add_change(ticker, date, added):
        if ticker not in changes:
            changes[ticker] = {}
        changes[ticker][date] = added

    for index, row in snp_df.iterrows():
        add_change(row.ticker,
                   row.added if type(row.added) is datetime.date else datetime.date.min, True)
        add_change(row.ticker, datetime.date.max, False)

    for index, row in snp_changes_df.iterrows():
        if type(row.added) is str and row.added != "":
            add_change(row.added, row.date, True)
        if type(row.removed) is str and row.removed != "":
            add_change(row.removed, row.date, False)

    def get_ticker_periods(chgs):
        periods = []

        # if ticker present only in changes file and first record was removal
        if chgs[min(chgs.keys())] == False:
            chgs[datetime.date.min] = True
        dates = sorted(chgs.keys())

        in_index = False
        prev_date = None
        for date in dates:
            added = chgs[date]
            if in_index == added:
                # missmatch in dates exists in snp.csv and snp_changes.csv
                # just allow records to missmatch and skip latest
                if (date - prev_date) <= datetime.timedelta(days=31):
                    continue
                return None
            if not added:
                # subtract one day because ticker already absent on index at removal date
                periods.append((prev_date, date - datetime.timedelta(days=1)))
            in_index = added
            prev_date = date
        return periods

    snp_mask_df = pd.DataFrame(columns=('ticker', 'from', 'to'))
    i = 0

    for ticker in changes.keys():
        periods = get_ticker_periods(changes[ticker])
        if periods is None:
            print(ticker)
        else:
            for _from, _to in periods:
                row = [rename_wiki_ticker_to_tiingo_ticker(ticker), _from, _to]
                snp_mask_df.loc[i] = row
                i += 1

    snp_mask_df.to_csv('data/snp/snp_mask.csv', index=False)
