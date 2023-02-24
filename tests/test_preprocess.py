import pytest
import pandas as pd
from src.preprocess import remove_infs

def test_remove_infs():
    df = pd.DataFrame(data={'col1': [float('inf'), 1, 5, 4.3]})
    labels = pd.DataFrame(data={'Labels': [5, 2, 4, 5]})
    df, labels = remove_infs(df, labels)
    assert df['col1'].iloc[0] == 1
