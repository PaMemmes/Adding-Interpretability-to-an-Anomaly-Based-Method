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
from plots import plot_severity_distribution, plot_packet_alerts
from utils import get_alerts_and_packets, get_json_data, get_signatures, get_categories

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

plt.style.use('science')

class AllDataHolder():
    def __init__(self, original, frag, frag_random):
        self.original = DataHolder('original')
        self.frag = DataHolder('fragmented')
        self.frag_random = DataHolder('fragmented_random')
        self.original.preprocess_data()
        self.frag.preprocess_data()
        self.frag_random.preprocess_data()

    def make_comp_plots(self):
        plot_comparison_severity_distribution(self.original.total_severity, self.frag.total_severity, save='plots/severity_dist_both.pdf')
        plot_comparison_packet_alerts(self.original.total_packets, self.frag.total_packets, self.original.total_signatures, self.frag.total_signatures, save='plots/packet_alerts_both.pdf')

    def make_individual_plots(self):
        self.original.make_plots()
        self.frag.make_plots()
        self.frag_random.make_plots()

class DataHolder():
    def __init__(self, kind):
        self.kind = kind
        self.total_packets = []
        self.total_signatures = []
        self.total_severity = []
        self.total_categories = []
    
    def preprocess_data(self):
        for filepath in sorted(glob.glob('theZoo_' + self.kind + '/**/*.json')):
            json_file = get_json_data(filepath)
            alerts, packets = get_alerts_and_packets(json_file)
            nmbr_signatures, severity= get_signatures(alerts)
            cats_frag = get_categories(alerts)

            self.total_packets.append(packets)
            self.total_signatures.append(nmbr_signatures)
            self.total_severity.append(severity)
            self.total_categories.append(cats_frag)

    def plot_file(self, file):
        file = 'theZoo_original/' + file + '/eve.json'
        json_data = get_json_data(file)
        alerts, _ = get_alerts_and_packets(json_data)
        nmbr_signatures, _ = get_signatures(alerts)

        plot_alerts(nmbr_signatures, file, save=f'plots/' + file + '_' + self.kind + '_original.pdf')

    def make_plots(self):
        plot_alert_distribution(self.total_signatures, save='plots/alert_dist_' + self.kind + '.pdf')
        plot_severity_distribution(self.total_severity, save='plots/severity_dist_' + self.kind + '.pdf')
        plot_packet_alerts(self.total_packets, self.total_signatures, save='plots/packet_alerts_' + self.kind + '.pdf')
        plot_categories(self.total_categories, save='plots/cat_dist_' + self.kind + '.pdf')
        self.plot_file('All.ElectroRAT')

if __name__ == '__main__':
    category_original= '_original'
    category_fragmented = '_fragmented'
    # color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
    #             CB91_Purple, CB91_Violet]
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

    all_data = AllDataHolder(DataHolder('original'), DataHolder('fragmented'), DataHolder('fragmented_random'))
    all_data.make_comp_plots()
    all_data.make_individual_plots()
