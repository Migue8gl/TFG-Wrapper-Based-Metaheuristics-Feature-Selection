import numpy as np

from src import data_utils
from src.constants import SAMPLE, LABELS


def test_load_data():
    dataset = data_utils.load_data('./data/iris.arff')
    dataset2 = data_utils.load_data('./results/binary/analysis_results.csv')
    dataset3 = data_utils.load_data(8282)

    assert dataset is not None and dataset2 is not None
    assert dataset3 is None


def test_split_data_to_dict():
    dataset = data_utils.load_data('./data/iris.arff')
    dataset = data_utils.split_data_to_dict(dataset)

    assert dataset is not None and isinstance(dataset, dict)
    assert SAMPLE in dataset and LABELS in dataset
    assert isinstance(dataset[SAMPLE], np.ndarray) and isinstance(
        dataset[LABELS], np.ndarray)
    assert dataset[SAMPLE].dtype == np.float64 and dataset[
        LABELS].dtype == np.int8
