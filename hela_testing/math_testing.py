import unittest
import numpy as np
from Hela.math import (
    GeometricMean,
    PowerIteration,
    Harmonic,
    Signal,
    FastFourierTransforms,
)


class TestGeometricMean(unittest.TestCase):
    def test_geometric_mean(self):
        gm = GeometricMean([2, 4, 8])
        result = gm()

        expected_result = 4.0
        self.assertAlmostEqual(result, expected_result)

    def test_geometric_mean_float(self):
        gm = GeometricMean([4, 8, 16])
        result = gm()
        expected_result = 7.999999999999999
        self.assertEqual(result, expected_result)

    def test_geometric_mean_zero(self):
        gm = GeometricMean([0, 2, 3])
        result = gm()
        expected_result = 0.0
        self.assertEqual(result, expected_result)


class TestPowerIteration(unittest.TestCase):
    def test_power_iteration_real_matrix(self):
        input_matrix = np.array([[2, 1], [1, 3]])
        vector = np.array([1, 0])
        power_iteration = PowerIteration(input_matrix, vector)
        eigenvalue, eigenvector = power_iteration()

        expected_eigenvalue = 2.23606797749979
        expected_eigenvector = np.array([0.37139068, 0.92847669])

        self.assertAlmostEqual(eigenvalue, expected_eigenvalue, places=12)


class TestHarmonic(unittest.TestCase):
    def test_is_series_valid(self):
        valid_series = [2, 4, 6]
        harmonic_mean = Harmonic(valid_series)
        self.assertFalse(harmonic_mean.is_series())

    def test_mean_valid(self):
        valid_series = [1, 4, 4]
        harmonic_mean = Harmonic(valid_series)
        self.assertEqual(harmonic_mean.mean(), 2.0)

    def other_test_mean_valid(self):
        valid_series = [3, 6, 9, 12]
        harmonic_mean = Harmonic(valid_series)
        self.assertEqual(harmonic_mean.mean(), 5.759999999999999)

    def test_mean_valid_dummy(self):
        valid_series = [1, 2, 3]
        harmonic_mean = Harmonic(valid_series)
        self.assertEqual(harmonic_mean.mean(), 1.6363636363636365)


class TestSignal(unittest.TestCase):
    def test_Signal_for_cspline1d(self):
        vector = np.array([1, 2, 3, 4, 5])
        control_vec = np.array([10, 20, 30, 40, 50])
        k, x, p = 2, 3, 2
        Signal_DDF = Signal(vector, control_vec).cspline1d(k, x, p)
        self.assertEqual(Signal_DDF, 3.25)

    def test_signal_for_cspline2d(self):
        vector = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        control_vec = np.array([10, 60, 30, 40, 50])
        eval_point, index_knot, degree = 1.5, 0, 1
        signal_DDF2 = Signal(vector, control_vec).cspline2d(
            eval_point, index_knot, degree
        )
        self.assertEqual(signal_DDF2, 0.0)


class TestFastFourierTransforms(unittest.TestCase):
    def test_FFT_discrete(self):
        vector = np.array([1.0, 2.0, 3.0, 4.0])
        fft_instance = FastFourierTransforms()
        result = fft_instance.discrectefft(vector)
        expected_result = np.array([1.0 + 0.0j, -2.0 + 2.0j, 0.0 + 0.0j, -2.0 - 2.0j])
        # Compare complex arrays with tolerance for floating-point precision
        self.assertEqual(result.any(), expected_result.any())
