import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils.utils import date_from_timestamp

DDMMMYY_FMT = matplotlib.dates.DateFormatter('%y %b %d')
YYYY_FMT = matplotlib.dates.DateFormatter('%Y')


def format_time_labels(ax, fmt):
    ax.xaxis.set_major_formatter(fmt)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)


def draw_grid(ax):
    ax.grid(True, linestyle='-', color='0.75')


def build_time_axis(ts):
    dt = []
    for raw_dt in np.nditer(ts):
        dt.append(date_from_timestamp(raw_dt))
    return dt


def plot_eq(caption, eq, ts):
    dt = build_time_axis(ts)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    draw_grid(ax)
    format_time_labels(ax, fmt=DDMMMYY_FMT)
    ax.set_title(caption)
    ax.plot_date(dt, eq, color='b', fmt='-')
    return fig