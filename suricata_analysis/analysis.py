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

from plots import plot_packet_alerts, plot_alerts, plot_alert_distribution, plot_severity_distribution, get_json_data, get_alerts_and_packets, get_signatures

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'



if __name__ == '__main__':
    category = '_unfragmented'
   
    color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
                CB91_Purple, CB91_Violet]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

    total_packets = []
    total_signatures = []
    total_severity = []

    severity_dict = defaultdict(int)
    for filepath in sorted(glob.glob('theZoo' + category + '/**/*.json')):
        json_data = get_json_data(filepath)
        alerts, packets = get_alerts_and_packets(json_data)
        nmbr_signatures, severity = get_signatures(alerts)
        
        for key, value in severity.items():
            severity_dict[key] = value
        total_packets.append(packets)
        total_signatures.append(nmbr_signatures)
        total_severity.append(severity)

    plot_alert_distribution(total_signatures, severity_dict, save='plots/alert_dist' + category)
    plot_severity_distribution(total_severity, save='plots/severity_dist' + category)
    plot_packet_alerts(total_packets, total_signatures, save='plots/packet_alerts' + category)

    file = 'theZoo' + category + '/All.ElectroRAT/eve.json'
    json_data = get_json_data(file)
    alerts, packets = get_alerts_and_packets(json_data)
    nmbr_signatures, severity = get_signatures(alerts)

    plot_alerts(nmbr_signatures, file, save=f'plots/All.ElectroRAT' + category + '.png')