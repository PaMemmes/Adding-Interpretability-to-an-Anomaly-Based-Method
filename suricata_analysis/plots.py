import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import glob

import re
from collections import defaultdict
import seaborn
import orjson


def make_xlabels(data, chars=None):
    x_ticks = [i for i in data.keys()]
    xlabels_new = [re.sub("(.{6})", "\\1\n", label, 0, re.DOTALL) for label in x_ticks]
    return xlabels_new

def plot_comparison_severity_distribution(dist1, dist2, save=None):
    # highest severity: 1, lowest severity: 4
    # Theoretically until 255 when creating rules manually
    width = 0.33

    severity_range = np.arange(4)

    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Severity Distribution')
    ax.set_xlabel('Severity Level (1: Highest, 4: Lowest)')
    ax.set_ylabel('Number of Alerts')
    ax.set_xticks(severity_range + width / 2)
    ax.set_xticklabels(('1', '2', '3', '4'))
    rects1 = ax.bar(severity_range, dist1.values(), width=width, align='center')
    rects2 = ax.bar(severity_range+width, dist2.values(), width=width, align='center')
    ax.bar_label(rects1)
    ax.bar_label(rects2)
    ax.legend((rects1[0], rects2[0]), ('Normal', 'Fragmented'))

    if save is not None:
        fig.savefig(save)
    plt.close('all')

def plot_comparison_packet_alerts(packets_sum, frag_packets_sum, sigs_sum, frag_sigs_sum, save=None):   
    width = 0.33
    
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Total Alerts vs. Total Packets')
    ax.set_xlabel('Category')
    ax.set_ylabel('Number Alerts (log)')

    ax.set_xticks(np.arange(2) + width / 2)
    ax.set_xticklabels(('Total Alerts', 'Total Packets'))

    bar = ax.bar(np.arange(2), [sigs_sum, packets_sum], width=width)
    bar_fragmented = ax.bar(np.arange(2)+width, [frag_sigs_sum, frag_packets_sum], width=width)

    ax.legend((bar[0], bar_fragmented[0]), ('Normal', 'Fragmented'))
    ax.bar_label(bar)
    ax.bar_label(bar_fragmented)
    ax.set_yscale('log')

    if save is not None:
        fig.savefig(save)
    plt.close('all')

def plot_packet_alerts(packets_sum, sigs_sum, save=None):    
    print('Total signatures:', sigs_sum)
    print('Total packets:', packets_sum)
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Total Alerts vs. Total Packets')
    ax.set_xlabel('Category')
    ax.set_ylabel('Number')
    ax.set_xticks(range(2),['Total Alerts', 'Total Packets'])
    bar = ax.bar(np.arange(2), [sigs_sum, packets_sum])
    ax.bar_label(bar)
    
    if save is not None:
        fig.savefig(save)
    plt.close('all')

def plot_alerts(nmbr_signatures, file, save=None):    
    xlabels_new = make_xlabels(nmbr_signatures)
    fig, ax = plt.subplots()
    file = file.removesuffix('/eve.json')
    file = file.removeprefix('suricata_logs/')
    
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.set_xticks(np.arange(len(nmbr_signatures)), xlabels_new)
    ax.set_ylabel('Number of Alerts')
    ax.set_xlabel('Alert Name')
    ax.set_title(f'Number of Alerts in File {file}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.set_size_inches(8,4)
    ax.bar(np.arange(len(nmbr_signatures)), nmbr_signatures.values())
    
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    plt.close('all')

def plot_alert_distribution(sigs_dist, save=None):
    xlabels_new = make_xlabels(sigs_dist)

    fig, ax = plt.subplots()
    fig.set_size_inches(24,8)
    ax.set_xticks(np.arange(len(sigs_dist)), xlabels_new)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Alert Distribution')
    ax.set_xlabel('Alert')
    ax.set_ylabel('Number of Alerts')
    ax.bar(np.arange(len(sigs_dist)), sigs_dist.values())
    
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    plt.close('all')

def plot_categories(categories, save=None):
    total_cats = defaultdict(int)
    for cat in categories:
        for key, value in cat.items():
            total_cats[key] += 1

    fig, ax = plt.subplots()
    fig.set_size_inches(8,4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('ET Category Distribution')
    ax.set_xlabel('ET Category')
    ax.set_ylabel('Number of alerts')
    xlabels_new = make_xlabels(total_cats)
    ax.bar(np.arange(len(total_cats)), total_cats.values())
    ax.set_xticks(range(0, len(total_cats.keys())), xlabels_new)

    if save is not None:
        fig.savefig(save)
    plt.close('all')

def plot_severity_distribution(dist, save=None):
    # highest severity: 1, lowest severity: 4
    # Theoretically until 255 when creating rules manually
    severity_range = [1,2,3,4]
    total_sigs = defaultdict(int)
    for signatures in dist:
        for key, value in signatures.items():
            total_sigs[value] += 1

    for key in severity_range:
        if key not in total_sigs.keys():
            total_sigs[key] = 0
    fig, ax = plt.subplots()
    fig.set_size_inches(8,4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Severity Distribution')
    ax.set_xlabel('Severity Level (1: Highest, 4: Lowest)')
    ax.set_ylabel('Number of alerts')
    ax.bar(severity_range, total_sigs.values())
    
    if save is not None:
        fig.savefig(save)

    plt.close('all')
