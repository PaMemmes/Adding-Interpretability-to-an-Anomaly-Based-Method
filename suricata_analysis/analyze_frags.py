import os
import pandas as pd
import glob
import numpy as np
from ydata_profiling import ProfileReport

def analyze_frags():
    # Creates a ydata Profile html Report 
    all_files = glob.glob(os.path.join('../data/csv_fragmentedV3', "*.csv"))
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    df = df.drop(['Dst IP', 'Flow ID', 'Src IP', 'Src Port', 'Timestamp'], axis=1)
    df['Label'] = 'Fragmented Malware'
    df['Dst Port'] = np.random.randint(0, df.max(axis=0)['Dst Port'],size=(len(df)))

    print(df.shape)
    data_profile = ProfileReport(df, minimal=True)
    data_profile.to_file(f'plots/frag_data.html')
    
if __name__ == '__main__':
    analyze_frags()
    