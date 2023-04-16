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
        
        self.original.process_data()
        self.frag.process_data()
        self.frag_random.process_data()

    def make_comp_plots(self):
        plot_comparison_packet_alerts(self.original.packets_sum, self.frag.packets_sum, self.frag_random.packets_sum, self.original.sigs_sum, self.frag.sigs_sum, self.frag_random.sigs_sum, save='plots/packet_alerts_both.pdf')
        plot_comparison_severity_distribution(self.original.sev_dist, self.frag.sev_dist, self.frag_random.sev_dist, save='plots/severity_dist_both.pdf')

    def make_individual_plots(self):
        self.original.make_plots()
        self.frag.make_plots()
        self.frag_random.make_plots()

class DataHolder():
    def __init__(self, kind):
        self.kind = kind

        self.packets = []
        self.signatures = []
        self.severity = []
        self.categories = []

        self.sigs_dist = defaultdict(int)
        self.sev_dist = defaultdict(int)

        self.sigs_sum = 0
        self.packets_sum = 0

    def process_data(self):
        for filepath in sorted(glob.glob('theZoo_' + self.kind + '/**/*.json')):
            json_file = get_json_data(filepath)
            alerts, packets = get_alerts_and_packets(json_file)
            nmbr_signatures, severity= get_signatures(alerts)
            cats_frag = get_categories(alerts)

            self.packets.append(packets)
            self.signatures.append(nmbr_signatures)
            self.severity.append(severity)
            self.categories.append(cats_frag)

        for elem in self.signatures:
            for value in elem.values():
                self.sigs_sum += value

        for elem in self.packets:
            self.packets_sum += elem
        
        for signatures in self.signatures:
            for key, value in signatures.items():
                self.sigs_dist[key] += value
        
        for signatures in self.severity:
            for key, value in signatures.items():
                self.sev_dist[value] += 1
        
        # Set value "0" for every key that is existent
        for key in np.arange(len(self.sev_dist)):
            if key not in self.sev_dist.keys():
                self.sev_dist[key] = 0
    
        print(f'Total signatures {self.kind}: {self.sigs_sum}')
        print(f'Total packets of {self.kind}: {self.packets_sum}')

    def plot_file(self, file):
        filepath = 'theZoo_original/' + file + '/eve.json'
        json_data = get_json_data(filepath)
        alerts, _ = get_alerts_and_packets(json_data)
        nmbr_signatures, _ = get_signatures(alerts)

        plot_alerts(nmbr_signatures, file, save=f'plots/' + file + '_' + self.kind + '.pdf')

    def make_plots(self):
        plot_alert_distribution(self.sigs_dist, save='plots/alert_dist_' + self.kind + '.pdf')
        plot_severity_distribution(self.severity, save='plots/severity_dist_' + self.kind + '.pdf')
        plot_packet_alerts(self.packets_sum, self.sigs_sum, save='plots/packet_alerts_' + self.kind + '.pdf')
        plot_categories(self.categories, save='plots/cat_dist_' + self.kind + '.pdf')
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
