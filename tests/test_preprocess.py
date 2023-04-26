import pytest
import pandas as pd
import numpy as np
from src.utils.utils import remove_infs, encode

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

@pytest.fixture
def example_df():
    return pd.DataFrame({'Feature1': [1, 3, 5, float('Nan'), 100000, 4, -10, 4, 5, float('inf'), float('-inf'), float('nan')],
                         'Feature2': [3, 12, 5, 3, 3, 1, 4, 10, 51, 5, 10000, -1053445],
                         'Label': ['BENIGN', 'BENIGN', 'BOT', 'DDoS', 'Trojan', 'Worm', 'Scan', 'BENIGN', 'BENIGN', 'Trojan', 'Worm', 'Scan']})


def test_remove_infs(example_df):
    df, labels = remove_infs(example_df)
    assert df.isnull().sum().sum() == 0
    assert np.isinf(df).sum().sum() == 0


def test_encode(example_df):
    le = LabelEncoder()
    le.fit(example_df['Label'])
    labels = encode(le, example_df['Label'])
    for elem in labels:
        assert isinstance(elem, np.int64)
    assert len(np.unique(labels)) == 6