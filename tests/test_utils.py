import numpy as np

from rlll.utils import StackDataManager


def test_stack_data():
    sdm = StackDataManager(4)
    sdm.add(np.array([1, 2, 3, 4]))

    assert np.all(np.equal(sdm.stacked_data,
                    np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])))

    sdm.add(np.array([2, 3, 4, 5]))

    assert np.all(np.equal(sdm.stacked_data,
                    np.array([
                        [2, 3, 4, 5],
                        [1, 2, 3, 4],
                        [1, 2, 3, 4],
                        [1, 2, 3, 4],
                    ])))

    sdm.add(np.array([3, 4, 5, 6]))

    assert np.all(np.equal(sdm.stacked_data,
                    np.array([
                        [3, 4, 5, 6],
                        [2, 3, 4, 5],
                        [1, 2, 3, 4],
                        [1, 2, 3, 4],
                    ])))
