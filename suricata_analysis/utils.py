import orjson
from collections import defaultdict

def get_json_data(file):
    return [orjson.loads(line) for line in open(file, "rb")]

def get_alerts_and_packets(json_data):
    alerts = [line for line in json_data if "alert" in line]
    packets = len(json_data)
    return alerts, packets

def get_signatures(alerts):
    nmbr_signatures = defaultdict(int)
    severity = defaultdict(int)
    packets = 0
    for entry in alerts:
        nmbr_signatures[entry['alert']['signature']] += 1
        severity[entry['alert']['signature']] = entry['alert']['severity']
    return nmbr_signatures, severity

def get_categories(alerts):
    category = defaultdict(int)
    for entry in alerts:
        category[entry['alert']['category']] += 1
    return category