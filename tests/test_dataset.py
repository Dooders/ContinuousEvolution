import numpy as np
import pytest

from evolution.data.data import Dataset


@pytest.fixture
def sample_dataset():
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    indices = np.array([0, 1, 2])
    metadata = {"name": "test_dataset"}
    target_index = 3
    return Dataset(data, indices, metadata, target_index)


def test_dataset_initialization(sample_dataset):
    assert sample_dataset.data.shape == (3, 4)
    assert np.array_equal(sample_dataset.indices, np.array([0, 1, 2]))
    assert sample_dataset.metadata == {"name": "test_dataset"}
    assert sample_dataset.target_index == 3


def test_dataset_len(sample_dataset):
    assert len(sample_dataset) == 3


def test_dataset_getitem(sample_dataset):
    data, index, metadata = sample_dataset[1]
    assert np.array_equal(data, np.array([5, 6, 7, 8]))
    assert index == 1
    assert metadata == {"name": "test_dataset"}


def test_get_target(sample_dataset):
    assert sample_dataset.get_target(0) == 4
    assert sample_dataset.get_target(1) == 8
    assert sample_dataset.get_target(2) == 12


def test_get_features(sample_dataset):
    assert np.array_equal(sample_dataset.get_features(0), np.array([1, 2, 3]))
    assert np.array_equal(sample_dataset.get_features(1), np.array([5, 6, 7]))
    assert np.array_equal(sample_dataset.get_features(2), np.array([9, 10, 11]))


def test_dataset_iter(sample_dataset):
    data_list = list(sample_dataset)
    assert len(data_list) == 3
    assert np.array_equal(data_list[0], np.array([1, 2, 3, 4]))
    assert np.array_equal(data_list[1], np.array([5, 6, 7, 8]))
    assert np.array_equal(data_list[2], np.array([9, 10, 11, 12]))


def test_dataset_next(sample_dataset):
    iterator = iter(sample_dataset)
    assert np.array_equal(next(iterator), np.array([1, 2, 3, 4]))
    assert np.array_equal(next(iterator), np.array([5, 6, 7, 8]))
    assert np.array_equal(next(iterator), np.array([9, 10, 11, 12]))
    with pytest.raises(StopIteration):
        next(iterator)


def test_dataset_empty():
    empty_dataset = Dataset(np.array([]), np.array([]), {}, 0)
    assert len(empty_dataset) == 0
    assert list(empty_dataset) == []


def test_dataset_single_item():
    single_item_dataset = Dataset(
        np.array([[1, 2]]), np.array([0]), {"name": "single"}, 1
    )
    assert len(single_item_dataset) == 1
    assert np.array_equal(single_item_dataset.get_features(0), np.array([1]))
    assert single_item_dataset.get_target(0) == 2


def test_dataset_invalid_target_index():
    with pytest.raises(IndexError):
        Dataset(np.array([[1, 2], [3, 4]]), np.array([0, 1]), {}, 2)


def test_dataset_out_of_bounds_index():
    dataset = Dataset(np.array([[1, 2], [3, 4]]), np.array([0, 1]), {}, 1)
    with pytest.raises(IndexError):
        dataset[2]


def test_dataset_metadata_access():
    metadata = {"name": "test", "description": "A test dataset"}
    dataset = Dataset(np.array([[1, 2], [3, 4]]), np.array([0, 1]), metadata, 1)
    assert dataset.metadata["name"] == "test"
    assert dataset.metadata["description"] == "A test dataset"


def test_dataset_with_string_data():
    data = np.array([["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]])
    dataset = Dataset(data, np.array([0, 1, 2]), {}, 2)
    assert dataset.get_target(1) == "f"
    assert np.array_equal(dataset.get_features(0), np.array(["a", "b"]))


def test_dataset_with_mixed_data_types():
    data = np.array([[1, "a", 3.14], [2, "b", 2.718], [3, "c", 1.414]])
    dataset = Dataset(data, np.array([0, 1, 2]), {}, 2)
    assert dataset.get_target(0) == 3.14
    assert np.array_equal(dataset.get_features(1), np.array([2, "b"]))


def test_dataset_slice():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    dataset = Dataset(data, np.array([0, 1, 2, 3]), {}, 2)
    sliced_data = dataset[1:3]
    assert len(sliced_data) == 2
    assert np.array_equal(sliced_data[0][0], np.array([4, 5, 6]))
    assert np.array_equal(sliced_data[1][0], np.array([7, 8, 9]))
