import orjson
from collections import defaultdict

def get_json_data(file):
    # Loads json data
    return [orjson.loads(line) for line in open(file, "rb")]

def get_alerts_and_packets(json_data):
    # Gets a list of alerts and the total amount of packets
    alerts = [line for line in json_data if "alert" in line]
    packets = len(json_data)
    return alerts, packets

def get_signatures(alerts):
    # Calculates the total number of a specific alert
    # and the category and a severity of an alert
    nmbr_signatures = defaultdict(int)
    severity = defaultdict(int)
    category = defaultdict(int)
    #nmbr_severity = defaultdict(int)
    packets = 0
    for entry in alerts:
        nmbr_signatures[entry['alert']['signature']] += 1
        category[entry['alert']['category']] += 1
        severity[entry['alert']['severity']] += 1
        
    return nmbr_signatures, severity, category
