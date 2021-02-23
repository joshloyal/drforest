import os
import pandas as pd


__all__ = ['load_beijing']


def load_beijing():
    """Load and return the Beijing air quality dataset."""

    module_path = os.path.dirname(__file__)

    data = pd.read_csv(
        os.path.join(module_path, 'data', 'beijing_air_quality.csv'))

    return data
