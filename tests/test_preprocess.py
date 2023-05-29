import pytest
import pandas as pd
import numpy as np
from src.preprocess import DataFrame

@pytest.fixture
def df():
    return DataFrame

def test_create_df(df):
    df.create_df('data/cicids2018/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv')