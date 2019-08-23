import unittest
import pandas as pd
import numpy as np
import linear_regression as LR


class Test_linear_regression(unittest.TestCase):

    def test_normalization_2D(self):
        arr = np.array([[5650, 221900],
                        [7242, 538000],
                        [10000, 180000],
                        [5000, 604000],
                        [8080, 510000]])
        expect = np.array([[-0.86719, -1.08224],
                           [0.02673, 0.72894],
                           [1.57536, -1.32232],
                           [-1.23217, 1.10711],
                           [0.49727, 0.56851]]
                          )
        result = np.round(LR.normalization(arr), 5)

        self.assertTrue(np.array_equal(expect, result))

    def test_normalization_1D(self):
        arr = np.array([5650,
                        7242,
                        10000,
                        5000,
                        8080])
        expect = np.array([-0.86719,
                           0.02673,
                           1.57536,
                           -1.23217,
                           0.49727])
        result = np.round(LR.normalization(arr), 5)

        self.assertTrue(np.array_equal(expect, result))

    def test_normalization_1D_with_div_0(self):
        arr = np.array([1, 1, 1, 1])
        expect = np.array([0, 0, 0, 0])
        result = LR.normalization(arr)

        self.assertTrue(np.array_equal(expect, result))

    def test_verify_parameters_no_raise_exception(self):
        thetas = np.array([[1], [2], [3]])
        features = np.reshape([1, 2, 3, 4, 5, 6], (2, 3))
        targets = np.reshape([1, 2, 3, 4, 5, 6], (2, 3))

        try:
            LR.verify_parameters(thetas, features)
            LR.verify_parameters(thetas, features, targets)
        except Exception:
            self.fail('verify_parameters() raised Exception')

    def test_verify_parameters_raise_exception(self):
        thetas = np.array([[1], [2], [3]])
        features = np.array([1, 2, 3, 4, 5, 6])
        targets = np.array([1, 2, 3, 4, 5, 6])

        with self.assertRaises(Exception) as context:
            LR.verify_parameters(
                thetas,
                np.reshape(features, (3, 2)))
        self.assertTrue(
            'thetas rows is 3 not equals' in context.exception.args[0])

        with self.assertRaises(Exception) as context:
            LR.verify_parameters(
                thetas,
                np.reshape(features, (2, 3)),
                np.reshape(targets, (3, 2)))
        self.assertTrue(
            'features rows 2 not equals' in context.exception.args[0])

    def test_calc_J_value(self):
        thetas = np.array([[1], [0.8], [-1.35]])
        features = np.array([
            [-0.88, 1.35, -0.23],
            [-0.86, 2.15, -0.22],
            [-0.84, 2.95, -0.21],
            [-0.82, 3.75, -0.2],
            [-0.8, 4.55, -0.19]
        ])
        targets = np.array([
            [-0.887],
            [-0.256],
            [0.3567],
            [0.6897],
            [0.8876]
        ])
        expect = 1.40207
        result = round(LR.calc_J_value(thetas, features, targets), 5)

        self.assertEqual(expect, result)

    def test_calc_J_partial_derivative(self):
        thetas = np.reshape(np.array([1, 0.8, -1.35]), (3, 1))
        features = np.array([
            [-0.88, 1.35, -0.23],
            [-0.86, 2.15, -0.22],
            [-0.84, 2.95, -0.21],
            [-0.82, 3.75, -0.2],
            [-0.8, 4.55, -0.19]
        ])
        targets = np.reshape(np.array([
            -0.887,
            -0.256,
            0.3567,
            0.6897,
            0.8876
        ]), (5, 1))
        expect = np.array([[-1.37417], [5.16885], [-0.34157]])
        result = np.round(LR.calc_J_partial_derivative(
            thetas, features, targets), 5)

        self.assertTrue(np.array_equal(expect, result))

    def test_calc_new_thetas(self):
        thetas = np.array([[1], [0.8], [-1.35]])
        features = np.array([
            [-0.88, 1.35, -0.23],
            [-0.86, 2.15, -0.22],
            [-0.84, 2.95, -0.21],
            [-0.82, 3.75, -0.2],
            [-0.8, 4.55, -0.19]
        ])
        targets = np.array([
            [-0.887],
            [-0.256],
            [0.3567],
            [0.6897],
            [0.8876]
        ])
        learning_rate = 0.001
        expect = np.array([[1.00137417], [0.79483115], [-1.34965843]])
        result = np.round(LR.calc_new_thetas(
            thetas, features, targets, learning_rate), 8)

        self.assertTrue(np.array_equal(expect, result))

    def test_normal_equation_raise_exception(self):
        features = np.reshape(range(10), (2, 5))
        targets = np.reshape(range(10), (5, 2))

        with self.assertRaises(Exception) as context:
            LR.normal_equation(features, targets)
        self.assertTrue(
            'features rows 2 not equals' in context.exception.args[0])


if __name__ == "__main__":
    unittest.main()
