import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import glob
import scienceplots
from textwrap import wrap

import re
from collections import defaultdict
import orjson

plt.style.use(['ieee', 'science'])
plt.style.use('seaborn-colorblind')

def make_xlabels(data, chars=None):
    # Formats x labels by linebreaks
    x_ticks = data
    labels = ['\n'.join(wrap(l, chars)) for l in x_ticks]
    return labels

def plot_comparison_severity_distribution(dist1, dist2, dist3, save=None):
    # highest severity: 1, lowest severity: 4
    # Theoretically until 255 when creating rules manually
    width = np.min(np.diff(np.arange(2))) / 4

    severity_range = np.arange(4) - 1
    dist1 = [x[1] for x in sorted(dist1.items())]
    dist2 = [x[1] for x in sorted(dist2.items())]
    dist3 = [x[1] for x in sorted(dist3.items())]

    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Severity Level (1: Highest, 4: Lowest)')
    ax.set_ylabel('Number of Alerts')
    ax.set_xticks(severity_range + width / 2)
    ax.set_xticklabels(('1', '2', '3', '4'))
    rects1 = ax.bar(severity_range-width/2, dist1, width=width, align='center')
    rects2 = ax.bar(severity_range+width/2, dist2, width=width, align='center')
    rects3 = ax.bar(severity_range+width*1.5, dist3, width=width, align='center')
    ax.bar_label(rects1)
    ax.bar_label(rects2)
    ax.bar_label(rects3)
    ax.legend((rects1[0], rects2[0], rects3[0]), ('Normal', 'Fragmented', 'Fragmented Randomly'))

    if save is not None:
        fig.savefig(save)
    plt.close('all')

def plot_comparison_packet_alerts(packets_sum, frag_packets_sum, rnd_frag_packets_sum, sigs_sum, frag_sigs_sum, rnd_frag_sigs_sum, save=None):   
    # Plots bar chart of alerts for 3 experiments
    width = np.min(np.diff(np.arange(2))) / 4
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Category')
    ax.set_ylabel('Number Alerts (log)')

    ax.set_xticks(np.arange(2) + width / 2)
    ax.set_xticklabels(('Total Alerts', 'Total Packets'))

    bar = ax.bar(np.arange(2)-width/2, [sigs_sum, packets_sum], width=width)
    bar_fragmented = ax.bar(np.arange(2)+width/2, [frag_sigs_sum, frag_packets_sum], width=width)
    bar_rnd_fragmented = ax.bar(np.arange(2)+width*1.5, [rnd_frag_sigs_sum, rnd_frag_packets_sum], width=width)

    ax.legend((bar[0], bar_fragmented[0], bar_rnd_fragmented[0]), ('Normal', 'Fragmented', 'Fragmented Randomly'))
    ax.bar_label(bar)
    ax.bar_label(bar_fragmented)
    ax.bar_label(bar_rnd_fragmented)
    ax.set_yscale('log')

    if save is not None:
        fig.savefig(save)
    plt.close('all')

def plot_comparison_categories(categories, frag_categories, frag_rnd_categories, save=None):   
    # Plots bar chart of alert categories of 3 experiments
    labels = [x[0] for x in sorted(categories.items())]
    new_labels = make_xlabels(labels, chars=10)

    categories = [x[1] for x in sorted(categories.items())]
    frag_categories = [x[1] for x in sorted(frag_categories.items())]
    frag_rnd_categories = [x[1] for x in sorted(frag_rnd_categories.items())]
    
    width = np.min(np.diff(np.arange(2))) / 4

    severity_range = np.arange(6)

    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Category')
    ax.set_ylabel('Number of Alerts')
    ax.set_xticks(severity_range + width / 2)
    ax.set_xticklabels(new_labels)
    rects1 = ax.bar(severity_range-width/2, categories, width=width, align='center')
    rects2 = ax.bar(severity_range+width/2, frag_categories, width=width, align='center')
    rects3 = ax.bar(severity_range+width*1.5, frag_rnd_categories, width=width, align='center')
    ax.bar_label(rects1)
    ax.bar_label(rects2)
    ax.bar_label(rects3)
    ax.legend((rects1[0], rects2[0], rects3[0]), ('Normal', 'Fragmented', 'Fragmented Randomly'))

    if save is not None:
        fig.savefig(save)
    plt.close('all')

def plot_packet_alerts(packets_sum, sigs_sum, save=None):
    # Plots total packets and total alerts
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
    # Plots alerts of an individual malware 
    xlabels_new = make_xlabels(nmbr_signatures, chars=10)
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
    # Plots the alert distribution of one experiment
    xlabels_new = make_xlabels(sigs_dist, chars=8)

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

def plot_categories(category_dist, save=None):
    # Plots category distribution of an experiment
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
    xlabels_new = make_xlabels(category_dist, chars=10)
    ax.bar(np.arange(len(category_dist)), category_dist.values())
    ax.set_xticks(range(0, len(category_dist.keys())), xlabels_new)

    if save is not None:
        fig.savefig(save)
    plt.close('all')

def plot_severity_distribution(dist, save=None):
    # highest severity: 1, lowest severity: 4
    # Theoretically until 255 when creating rules manually
    severity_range = [1,2,3,4]
    fig, ax = plt.subplots()
    fig.set_size_inches(8,4)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Severity Distribution')
    ax.set_xticks(severity_range)
    ax.set_xticklabels(('1', '2', '3', '4'))
    ax.set_xlabel('Severity Level (1: Highest, 4: Lowest)')
    ax.set_ylabel('Number of alerts')
    ax.bar(severity_range, dist.values())
    
    if save is not None:
        fig.savefig(save)

    plt.close('all')
