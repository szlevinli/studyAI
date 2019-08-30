import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
import logistic_regression_ass as lra


class Test_logistic_regression_ass(unittest.TestCase):
    def test_get_data(self):
        # 0 1 2
        # 3 4 5
        df = pd.DataFrame(
            np.reshape(range(6), (2, 3)),
            columns=[0, 1, 2]
        )
        # mock
        pd.read_csv = MagicMock(return_value=df)
        # expect
        expect_theta = np.array([0, 0, 0], dtype='float64')
        expect_X = np.array([
            [1, 0, 1],
            [1, 3, 4]
        ], dtype='float64')
        expect_y = np.array([2, 5], dtype='float64')
        # call function
        theta, X, y = lra.get_data(
            './data/andrew/ex2data1.csv',
            [0, 1],
            2,
            header=None
        )
        # assert
        pd.read_csv.assert_called_with(
            './data/andrew/ex2data1.csv',
            header=None
        )
        self.assertTrue(np.array_equal(expect_theta, theta))
        self.assertTrue(np.array_equal(expect_X, X))
        self.assertTrue(np.array_equal(expect_y, y))


if __name__ == '__main__':
    unittest.main()
