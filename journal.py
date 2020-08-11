import pandas as pd 
import numpy as np
from datetime import date
import matplotlib.pyplot as plt

import os

DIRNAME = os.path.dirname(os.path.abspath("__file__"))
PATH_TO_CSV = os.path.join(DIRNAME, 'daylio_export.csv')


def gen_baseline_metrics(path_to_csv = PATH_TO_CSV)

    ugly_df = pd.read_csv(path_to_csv, index_col=False)
    df = clean_df(ugly_df)

    entries_over_time = gen_entries_over_time_hist(df)

    pass

def clean_df(df)

    df.full_date = pd.to_datetime(df.full_date)

    return(df)


def gen_entries_over_time_hist(df):

    df._month = pd.to_datetime(df.full_date).dt.to_period('M').dt.to_timestamp()

    earliest_entry = min(df._month)
    start_year = earliest_entry.year
    start_month = earliest_entry.month

    latest_entry = max(df._month)
    end_year = latest_entry.year
    end_month = latest_entry.month

    all_months = [date(m//12, m%12+1, 1) for m in range(start_year*12+start_month-1, end_year*12+end_month)]
    num_entries = []

    for month in all_months:
        num_entries.append(len(df[df._month == month]))

    ax = plt.subplot(111)
    ax.bar(all_months, num_entries, width = 25, color = "darkorange")
    ax.xaxis_date()
    plt.title("# journal entries written, by month")

return(plt)



