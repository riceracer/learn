import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

"""
USAGE:

python3 process_behavior_chart.py AMZN.csv Date Close

python3 process_behavior_chart.py trade_def.csv Date Deficit

PENDING / TODO:
* Add an argument flag to plot the metric as a growing line instead of horizontal (e.g. for stocks)
"""

RANGE_MULTIPLIER = 3.27
NATURAL_PROCESS_MULTIPLIER = 2.66

def process_behavior_plot(x, y_metric, y_diff, metric_label, min_idx=None, max_idx=None):

    avg_metric = y_metric[min_idx:max_idx].mean()
    avg_diff = y_diff[min_idx:max_idx].mean()
    upper_range_limit = RANGE_MULTIPLIER  * avg_diff
    upper_metric_limit = avg_metric + NATURAL_PROCESS_MULTIPLIER * avg_diff
    lower_metric_limit = avg_metric - NATURAL_PROCESS_MULTIPLIER * avg_diff

    fig, axs = plt.subplots(2, figsize=(12, 10))
    fig.suptitle(f'Process Behavior: {metric_label}')
    idx = np.arange(len(x))    
    print(avg_metric)
    print(avg_diff)
    
    # metric plot
    axs[0].plot(idx, y_metric, color='green', marker='o')
    axs[0].hlines(avg_metric, min(idx), max(idx), linestyles='solid')
    axs[0].hlines([lower_metric_limit, upper_metric_limit], min(idx), max(idx), linestyles='dashed')
    
    # range plot
    axs[1].plot(idx, y_diff, color='green', marker='o')
    axs[1].hlines(avg_diff, min(idx), max(idx), linestyles='solid')
    axs[1].hlines(upper_range_limit, min(idx), max(idx), linestyles='dashed')

    # TODO - need to fix xticks for the subplot
    w = 0.4
    plt.xticks(idx + w / 2, x, rotation="vertical")
    plt.setp(axs[0], ylabel=metric_label)
    plt.setp(axs[1], ylabel='Moving Range')
    # range plot
    plt.show()


def main(input_file: str, date_col: str, metric_col: str, min_idx, max_idx):
    df = pd.read_csv(input_file)
    metric_diff = 'metric_diff'
    df[metric_diff] = df[metric_col].diff().abs()
    process_behavior_plot(df[date_col], df[metric_col], df[metric_diff], metric_col, min_idx=min_idx, max_idx=max_idx)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("date_col")
    parser.add_argument("metric_col")
    parser.add_argument("--min_idx", default=None, type=int)
    parser.add_argument("--max_idx", default=None, type=int)

    args = parser.parse_args()
    print(f"Args received: {args}")
    main(args.input_file, args.date_col, args.metric_col, args.min_idx, args.max_idx)
