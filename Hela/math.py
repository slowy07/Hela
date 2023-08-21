from __future__ import annotations
import math
import numpy as np


class GeometricMean:
    """
    Calculate the geometric mean of a list of numbers.

    The geometric mean of a set of numbers is calculated by
    multiplying all the numbers in the input listand then
    taking the n-th root of the product, where n is
    the number of elements in the list.

    Args:
        series (list or tuple): List of numbers for which to
                                calculate the geometric mean.

    Returns:
        float: Geometric mean of the input list.

    Example:
    >>> geometric_mean = GeometricMean()
    >>> geometric_mean([2, 4, 8])
    4.0
    >>> geometric_mean([1, 2, 3, 4, 5])
    2.605171084697352
    >>> geometric_mean([10, 20, 30, 40, 50])
    27.866004174074643

    Raises:
        ValueError: If the input series is not a valid list or tuple.
        ValueError: If the input list is empty.
    """

    def __init__(self, series: list) -> None:
        self.series = series

    def __call__(self) -> float:
        """
        Calculate the geometric mean of a list of numbers.

        The geometric mean of a set of numbers is calculated by
        multiplying all the numbers in the input list and
        then taking the n-th root of the product,
        where n is the number of elements in the list.

        Args:
            series (list or tuple): List of numbers for which
                                    to calculate the geometric mean.

        Returns:
            float: Geometric mean of the input list.

        Example:
        >>> geometric_mean = GeometricMean()
        >>> geometric_mean([2, 4, 8])
        4.0
        >>> geometric_mean([1, 2, 3, 4, 5])
        2.605171084697352
        >>> geometric_mean([10, 20, 30, 40, 50])
        27.866004174074643

        Raises:
            ValueError: If the input series is not a valid list or tuple.
            ValueError: If the input list is empty.
        """
        if not isinstance(self.series, (list, tuple)):
            raise ValueError(
                "GeometricMean(): input series not valid - valid series - [2, 4, 8]"
            )
        if len(self.series) == 0:
            raise ValueError("GeometricMean(): input list must be a non empty list")
        answer = 1
        for value in self.series:
            answer *= value
        return math.pow(answer, 1 / len(self.series))

    def __repr__(self):
        return f"{self.__call__()}"


class PowerIteration:
    """
    Calculate the largest eigenvalue and corresponding eigenvector of a given matrix using the Power Iteration method.

    This class implements the Power Iteration algorithm
    to find the largest eigenvalue and its corresponding eigenvector
    of a given square matrix. The algorithm iteratively applies matrix-vector
    multiplications and normalization to converge
    towards the dominant eigenvalue and eigenvector.

    Args:
        error_total (float, optional): The desired tolerance for convergence.
                                       Defaults to 1e-12.
        max_iteration (int, optional): The maximum number of iterations before termination.
                                       Defaults to 100.
    Methods:
        __call__(self, input_matrix: np.ndarray,
                 vector: np.ndarray) -> tuple[float, np.ndarray]:
            Calculate the largest eigenvalue and corresponding eigenvector
            using Power Iteration.

    Example:
    >>> input_matrix = np.array([[2, 1], [1, 3]])
    >>> vector = np.array([1, 0])
    >>> power_iteration = PowerIteration()
    >>> eigenvalue, eigenvector = power_iteration(input_matrix, vector)
    """

    def __init__(
        self,
        input_matrix: np.ndarray,
        vector: np.ndarray,
        error_total: float = 1e-12,
        max_iteration: int = 100,
    ):
        self.input_matrix = input_matrix
        self.vector = vector
        self.error_total = error_total
        self.max_iteration = max_iteration

    def __call__(
        self,
    ) -> tuple[float, np.ndarray]:
        """
        Calculate the largest eigenvalue and corresponding eigenvector of matrix `input_matrix`
        given a random vector in the same space.

        Args:
            input_matrix (np.ndarray): The square matrix for which to calculate the largest eigenvalue and eigenvector.
            vector (np.ndarray): The initial vector used in the Power Iteration algorithm.

        Returns:
            tuple[float, np.ndarray]: The calculated largest eigenvalue and the corresponding eigenvector.

        Raises:
            AssertionError: If the dimensions of the input matrix and vector do not match.
            ValueError: If the matrix or vector is complex and not Hermitian (not equal to its conjugate transpose).

        Example:
        >>> input_matrix = np.array([[2, 1], [1, 3]])
        >>> vector = np.array([1, 0])
        >>> power_iteration = PowerIteration()
        >>> eigenvalue, eigenvector = power_iteration(input_matrix, vector)
        """
        assert np.shape(self.input_matrix)[0] == np.shape(self.input_matrix)[1]
        assert np.shape(self.input_matrix)[0] == np.shape(self.vector)[0]
        assert np.iscomplexobj(self.input_matrix) == np.iscomplexobj(self.vector)
        is_complex = np.iscomplexobj(self.input_matrix)
        if is_complex:
            assert np.array_equal(self.input_matrix, self.input_matrix.conj().T)
        convergence = False
        lambda_previous = 0
        iterations = 0
        error = 1e12

        while not convergence:
            w = np.dot(self.input_matrix, self.vector)
            vector = w / np.linalg.norm(w)
            vector_h = vector.conj().T if is_complex else vector.T
            lambda_ = np.dot(vector_h, np.dot(self.input_matrix, self.vector))
            error = np.abs(lambda_ - lambda_previous) / lambda_
            iterations += 1

            if error <= self.error_total or iterations >= self.max_iteration:
                convergence = True
            lambda_previous = lambda_

        if is_complex:
            lambda_ = np.real(lambda_)
        return lambda_, vector

    def __repr__(self):
        return f"{self.__call__()}"


class Harmonic:
    """
    calculating the harmonic mean of a series numbers

    Example:
    >>> series = [2, 4, 6]
    >>> harmonic_mean = Harmonic(seris)
    >>> harmonic_mean()
    3.4285714285714284
    >>> harmonic_mean.series()
    """

    def __init__(self, series: list) -> None:
        """
        initialize the harmonic object with the input series

        Args:
            series(list): the list of numbers for which to calculate the
                         harmonic mean
        """
        self.series = series

    def is_series(self) -> bool:
        """
        check if the input series forms an arithmetic series

        Returns:
            (bool): true if the input series froms an arithmetic series, False otherwise

        Raises:
            ValueError: if the input series is not valid or doesn't an arithmetic series
        """
        if not isinstance(self.series, (list, tuple)):
            raise ValueError(
                f"HarmonicMean({self.series}).series() not valid, valid series - [1, 2/3, 2]"
            )
        if len(self.series) == 0:
            raise ValueError(
                f"HarmonicMean({self.series}).series() input list tuple must be an non empty"
            )
        if len(self.series) == 1 and self.series[0] != 0:
            return True

        rec_series: list = []
        series_len: int = len(self.series)
        for i in range(0, series_len):
            if self.series[i] == 0:
                raise ValueError(
                    f"HarmonicMean.series({self.series[0]}) cannot have 0 as an element"
                )
            rec_series.append(1 / self.series[i])
        common_dif = rec_series[1] - rec_series[0]
        for index in range(2, series_len):
            if rec_series[index] - rec_series[index - 1] != common_dif:
                return False
        return True

    def mean(self) -> float:
        """
        calculate the harmonic mean of the input series

        Returns:
            float: the harmonic mean of the input series

        Raises:
            ValueError: if the input series is not valid or empty
        """
        if not isinstance(self.series, (list, tuple)):
            raise ValueError(
                f"HarmonicMean({self.series}).mean(): input series not valid, valid series - [2, 4, 6]"
            )
        if len(self.series) == 0:
            raise ValueError(
                f"HarmonicMean({self.series}).mean(): input (list, tuple) must be a non empty"
            )
        answer: int = 0
        for val in self.series:
            answer += 1 / val
        return len(self.series) / answer


class Signal:
    """
    Calculate the singal processing
    signal processing is an electrical engineer subfield that
    focuses on analyzing.
    """

    def __init__(self, series: np.ndarray, control_matrix: np.array) -> None:
        self.series = series
        self.control_matrix = control_matrix

    def cspline1d(self, idx: int, x: int, p: int) -> float:
        """
        CsSpline1D is a Polynominal-time and Numerically stable
        algorithm for evaluating spline curves
        Args:
            control_matrix (np.array): index of knot interval that
                                       contains x.
            idx (int): Index of knot interval that contains x.
            x (int): Position.
            p (int): Degree of B-spline.

        Returns:
            float: value of evaluating spline curve
        """
        d = np.zeros(p + 1)
        for r in range(0, p + 1):
            r_adjusted = r + idx - p  # Adjusted index
            if 0 <= r_adjusted < len(self.control_matrix):  # Check for valid index
                d[r] = self.control_matrix[r_adjusted]

        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                alpha = x - self.series[j + idx]
                denominator = self.series[j + 1 + idx - r]
                -self.series[j + idx - p]
                if denominator != 0:
                    alpha /= denominator
                    d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

        return d[p]

    def cspline2d(self, eval_point: float, idx: int, degree: int) -> float:
        """
        Cox-de Boor is recursion formula to support in B-splines with
        Polynominals are positive in a finite domain and zero elsewhere

        Args:
            eval_point (float): Evaluation point
            idx (int): Index of knot interval
            degree (int): Degree of B-spline.

        Returns:
            float: Value of evaluting spline curve
        """
        if degree == 0:
            if (self.series[idx] <= eval_point) and (eval_point < self.series[idx + 1]):
                return 1
            return 0

        alpha = (eval_point - self.series[idx]) / (
            self.series[idx + degree] - self.series[idx]
        )

        left = alpha * self.cspline2d(eval_point, idx + 1, degree - 1)
        right = (1 - alpha) * self.cspline2d(eval_point, idx + 1, degree - 1)

        return float(left + right)


class FastFourierTransforms:
    """Fast Fourier Transforms is algorithm that computes
    `the Discrate Fourier Transform`
    """

    def discrectefft(self, vector: np.ndarray) -> np.ndarray:
        """
        Discrecte FFT is a finite of equally-spaced samples
        of a function into a same-length sequence of equaly-
        spaced sample of Fast Fouruier Transforms
        """
        if vector.ndim == 1 or len(vector) == 1:
            return vector
        n = len(vector)
        x_even = self.discrectefft(vector[::2])
        x_odd = self.discrectefft(vector[1::2])
        factor = np.exp(-2j * np.pi * np.arange(n) / n)
        X = np.concatenate(
            [
                x_even + factor[: int(n / 2)] * x_odd,
                x_even + factor[int(n / 2) :] * x_odd,
            ]
        )
        return X
