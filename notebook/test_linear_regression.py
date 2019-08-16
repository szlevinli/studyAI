import re
import unittest
from unittest.mock import MagicMock, Mock
import pandas as pd
import numpy as np
import linear_regression as LR


class Test_linear_regression(unittest.TestCase):

    def test_normalization(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        expect = pd.Series([-0.5, -0.25, 0, 0.25, 0.5])
        result = LR.normalization(s)

        self.assertTrue(expect.equals(result))

    def test_normalization_diff_is_zero(self):
        s = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
        expect = pd.Series([0, 0, 0, 0, 0], dtype='int32')
        result = LR.normalization(s)

        self.assertTrue(expect.equals(result))

    def test_calc_prediction_price_with_exception(self):
        thetas = np.array([1, 2, 3])
        features = np.reshape([1, 2, 3, 4], (2, 2))

        with self.assertRaises(Exception) as context:
            LR.calc_prediction_price(thetas, features)
        self.assertTrue(
            'thetas rows is 3 not equals' in context.exception.args[0])

    def test_calc_prediction_price(self):
        thetas = np.array([1, 0.8, -1.35])
        features = np.array([
            [-0.88, 1.35, -0.23],
            [-0.86, 2.15, -0.22],
            [-0.84, 2.95, -0.21],
            [-0.82, 3.75, -0.2],
            [-0.8, 4.55, -0.19]
        ])
        expect = 9.0175
        result = round(LR.calc_prediction_price(thetas, features), 4)

        self.assertEqual(expect, result)

    def test_calc_J_value(self):
        thetas = np.array([1, 0.8, -1.35])
        features = np.array([
            [-0.88, 1.35, -0.23],
            [-0.86, 2.15, -0.22],
            [-0.84, 2.95, -0.21],
            [-0.82, 3.75, -0.2],
            [-0.8, 4.55, -0.19]
        ])
        targets = np.array([
            -0.887,
            -0.256,
            0.3567,
            0.6897,
            0.8876
        ])
        expect = 1.40207
        result = round(LR.calc_J_value(thetas, features, targets), 5)

        self.assertEqual(expect, result)

    def test_calc_J_value_with_exception(self):
        thetas = np.array([1, 2, 3])
        features = np.reshape([1, 2, 3, 4], (2, 2))
        targets = np.reshape([1, 2, 3, 4, 5, 6], (3, 2))

        with self.assertRaises(Exception) as context:
            LR.calc_J_value(thetas, features, targets)
        self.assertTrue(
            'thetas rows is 3' in context.exception.args[0])
        self.assertTrue(
            'features rows 2' in context.exception.args[0])

if __name__ == "__main__":
    unittest.main()
