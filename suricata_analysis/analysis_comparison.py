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

from plots import plot_categories, plot_alert_distribution, plot_comparison_severity_distribution, plot_comparison_packet_alerts, plot_alerts
from utils import get_alerts_and_packets, get_json_data, get_signatures, get_categories

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

plt.style.use('science')

if __name__ == '__main__':
    category_unfragmented = '_unfragmented'
    category_fragmented = '_fragmented'
    # color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
    #             CB91_Purple, CB91_Violet]
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

    total_packets_fragmented = []
    total_signatures_fragmented = []
    total_severity_fragmented = []
    total_categories_fragmented = []

    severity_dict_fragmented = defaultdict(int)
    for filepath in sorted(glob.glob('theZoo' + category_fragmented + '/**/*.json')):
        json_data_fragmented = get_json_data(filepath)
        alerts_fragmented, packets_fragmented = get_alerts_and_packets(json_data_fragmented)
        nmbr_signatures_fragmented, severity_fragmented = get_signatures(alerts_fragmented)
        cats_frag = get_categories(alerts_fragmented)

        for key, value in severity_fragmented.items():
            severity_dict_fragmented[key] = value
        total_packets_fragmented.append(packets_fragmented)
        total_signatures_fragmented.append(nmbr_signatures_fragmented)
        total_severity_fragmented.append(severity_fragmented)
        total_categories_fragmented.append(cats_frag)

    total_packets_unfragmented = []
    total_signatures_unfragmented = []
    total_severity_unfragmented = []
    total_categories_unfragmented = []

    severity_dict_unfragmented = defaultdict(int)
    for filepath in sorted(glob.glob('theZoo' + category_unfragmented + '/**/*.json')):
        json_data_unfragmented = get_json_data(filepath)
        alerts_unfragmented, packets_unfragmented = get_alerts_and_packets(json_data_unfragmented)
        nmbr_signatures_unfragmented, severity_unfragmented = get_signatures(alerts_unfragmented)
        cats_unfrag = get_categories(alerts_unfragmented)

        for key, value in severity_unfragmented.items():
            severity_dict_unfragmented[key] = value
        total_packets_unfragmented.append(packets_unfragmented)
        total_signatures_unfragmented.append(nmbr_signatures_unfragmented)
        total_severity_unfragmented.append(severity_unfragmented)
        total_categories_unfragmented.append(cats_unfrag)
    
    plot_categories(total_categories_fragmented, save='plots/cat_dist_frag.pdf')
    plot_categories(total_categories_unfragmented, save='plots/cat_dist_unfrag.pdf')

    plot_comparison_severity_distribution(total_severity_unfragmented, total_severity_fragmented, save='plots/severity_dist_both.pdf')
    plot_comparison_packet_alerts(total_packets_unfragmented, total_packets_fragmented, total_signatures_unfragmented, total_signatures_fragmented, save='plots/packet_alerts_both.pdf')
    
    plot_alert_distribution(total_signatures_fragmented, save='plots/alert_dist_' + 'fragmented.pdf')
    plot_alert_distribution(total_signatures_unfragmented, save='plots/alert_dist_' + 'unfragmented.pdf')

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