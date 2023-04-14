import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import itertools
import os
import glob

import re
from collections import defaultdict
import seaborn
import orjson

from plots import plot_alerts, get_json_data, get_alerts_and_packets, get_signatures

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

plt.style.use('science')

def make_xlabels(data):
    x_ticks = [i for i in data.keys()]
    xlabels_new = [re.sub("(.{5})", "\\1\n", label, 0, re.DOTALL) for label in x_ticks]
    return xlabels_new

def plot_severity_distribution(dist1, dist2, save=None):
    # highest severity: 1, lowest severity: 4
    # Theoretically until 255 when creating rules manually
    width = 0.33

    severity_range = np.arange(4)
    severity_range2 = np.arange(4)
    total_sigs_dist1 = defaultdict(int)
    for signatures in dist1:
        for key, value in signatures.items():
            total_sigs_dist1[value] += 1

    for key in severity_range:
        if key not in total_sigs_dist1.keys():
            total_sigs_dist1[key] = 0

    total_sigs_dist2 = defaultdict(int)
    for signatures in dist2:
        for key, value in signatures.items():
            total_sigs_dist2[value] += 1

    for key in severity_range:
        if key not in total_sigs_dist2.keys():
            total_sigs_dist2[key] = 0

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
    rects1 = ax.bar(severity_range, total_sigs_dist1.values(), width=width, align='center')
    rects2 = ax.bar(severity_range2+width, total_sigs_dist2.values(), width=width, align='center')
    ax.bar_label(rects1)
    ax.bar_label(rects2)
    ax.legend((rects1[0], rects2[0]), ('Normal', 'Fragmented'))

    if save is not None:
        fig.savefig(save)

def plot_packet_alerts(packets, packets_fragmented, total_signatures, total_signatures_fragmented, save=None):   
    width = 0.33

    total_sigs = 0
    for elem in total_signatures:
        for value in elem.values():
            total_sigs += value
    
    total_packets = 0
    for elem in packets:
        total_packets += elem
    
    total_sigs_fragmented = 0
    for elem in total_signatures_fragmented:
        for value in elem.values():
            total_sigs_fragmented += value
    
    total_packets_fragmented = 0
    for elem in packets_fragmented:
        total_packets_fragmented += elem
    
    print('Total Signatures:', total_sigs)
    print('Total Packets:', total_packets)

    print('Total signatures fragmented:', total_sigs)
    print('Total packets fragmented:', total_packets)

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

    bar = ax.bar(np.arange(2), [total_sigs, total_packets], width=width)
    bar_fragmented = ax.bar(np.arange(2)+width, [total_sigs_fragmented, total_packets_fragmented], width=width)

    ax.legend((bar[0], bar_fragmented[0]), ('Normal', 'Fragmented'))
    ax.bar_label(bar)
    ax.bar_label(bar_fragmented)
    ax.set_yscale('log')

    if save is not None:
        fig.savefig(save)

def plot_alert_distribution(dist, severity, save=None):
    # TODO: Add severity to xticks
    total_sigs = defaultdict(int)
    for signatures in dist:
        for key, value in signatures.items():
            total_sigs[key] += value
    xlabels_new = make_xlabels(total_sigs)

    fig, ax = plt.subplots()
    fig.set_size_inches(24,8)
    ax.set_xticks(np.arange(len(total_sigs)), xlabels_new)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='white', linestyle='solid')
    ax.xaxis.grid(color='white', linestyle='solid')
    ax.set_facecolor("lightgrey")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Alert Distribution')
    ax.set_xlabel('Alert')
    ax.set_ylabel('Number of Alerts')
    bar = ax.bar(np.arange(len(total_sigs)), total_sigs.values())
    ax.bar_label(bar)
    if save is not None:
        fig.savefig(save, bbox_inches="tight")

if __name__ == '__main__':
    category_unfragmented = '_unfragmented'
    category_fragmented = '_fragmented'
    # color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
    #             CB91_Purple, CB91_Violet]
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

    total_packets_fragmented = []
    total_signatures_fragmented = []
    total_severity_fragmented = []

    severity_dict_fragmented = defaultdict(int)
    for filepath in sorted(glob.glob('theZoo' + category_fragmented + '/**/*.json')):
        json_data_fragmented = get_json_data(filepath)
        alerts_fragmented, packets_fragmented = get_alerts_and_packets(json_data_fragmented)
        nmbr_signatures_fragmented, severity_fragmented = get_signatures(alerts_fragmented)
        
        for key, value in severity_fragmented.items():
            severity_dict_fragmented[key] = value
        total_packets_fragmented.append(packets_fragmented)
        total_signatures_fragmented.append(nmbr_signatures_fragmented)
        total_severity_fragmented.append(severity_fragmented)

    total_packets_unfragmented = []
    total_signatures_unfragmented = []
    total_severity_unfragmented = []

    severity_dict_unfragmented = defaultdict(int)
    for filepath in sorted(glob.glob('theZoo' + category_unfragmented + '/**/*.json')):
        json_data_unfragmented = get_json_data(filepath)
        alerts_unfragmented, packets_unfragmented = get_alerts_and_packets(json_data_unfragmented)
        nmbr_signatures_unfragmented, severity_unfragmented = get_signatures(alerts_unfragmented)
        
        for key, value in severity_unfragmented.items():
            severity_dict_unfragmented[key] = value
        total_packets_unfragmented.append(packets_unfragmented)
        total_signatures_unfragmented.append(nmbr_signatures_unfragmented)
        total_severity_unfragmented.append(severity_unfragmented)

    plot_severity_distribution(total_severity_unfragmented, total_severity_fragmented, save='plots/severity_dist_both.pdf')
    plot_packet_alerts(total_packets_unfragmented, total_packets_fragmented, total_signatures_unfragmented, total_signatures_fragmented, save='plots/packet_alerts_both.pdf')
    
    plot_alert_distribution(total_signatures_fragmented, severity_dict_fragmented, save='plots/alert_dist_' + 'fragmented.pdf')
    plot_alert_distribution(total_signatures_unfragmented, severity_dict_unfragmented, save='plots/alert_dist_' + 'unfragmented.pdf')

    file = 'theZoo_unfragmented/All.ElectroRAT/eve.json'
    json_data = get_json_data(file)
    alerts, packets = get_alerts_and_packets(json_data)
    nmbr_signatures, severity = get_signatures(alerts)

    plot_alerts(nmbr_signatures, file, save=f'plots/All.ElectroRAT_unfragmented.pdf')

    file = 'theZoo_fragmented/All.ElectroRAT/eve.json'
    json_data = get_json_data(file)
    alerts, packets = get_alerts_and_packets(json_data)
    nmbr_signatures, severity = get_signatures(alerts)

    plot_alerts(nmbr_signatures, file, save=f'plots/All.ElectroRAT_fragmented.pdf')