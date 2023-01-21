import numpy as np

from rlll.utils import StackDataManager


def test_stack_data():
    sdm = StackDataManager(3)
    sdm.add(np.array([1]*8))

    print(sdm.stacked_data)

    assert np.all(np.equal(sdm.stacked_data,
                           np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])))

    sdm.add(np.array([2]*8))

    print(list(sdm.stacked_data))

    assert np.all(np.equal(sdm.stacked_data,
                           np.array(
                               [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))

    sdm.add(np.array([4]*8))

    print(list(sdm.stacked_data))
