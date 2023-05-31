import orjson
from collections import defaultdict

def get_json_data(file):
    # return [orjson.loads(line) for line in open(file, "rb")]
    index = 1
    result = []
    for line in open(file, "rb"):
        l = orjson.loads(line)
        result.append(l)
    return result
def get_alerts_and_packets(json_data):
    alerts = [line for line in json_data if "alert" in line]
    packets = len(json_data)
    return alerts, packets

def get_signatures(alerts):
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
