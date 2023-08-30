from __future__ import annotations
import math
import random
import numpy as np


class NormalDistribution:
    @staticmethod
    def normal_pdf(x: float, mean: float, sigma: float) -> float:
        """
        calculate the probability density function (PDF) of the normal distribution

        Args:
            x (float): the input value at which to calculate the PDF
            mean (float): the mean (average) of the normal distribution
            sigma (float): the standard deviation of the normal distribution

        Returns:
            (float): the value of the PDF at the given input.
        """
        coefficient = 1 / (sigma * math.sqrt(2 * math.pi))
        exponent = -((x - mean) ** 2) / (2 * sigma**2)
        pdf_value = coefficient * math.exp(exponent)
        return pdf_value

    @staticmethod
    def standard_normal_pdf(x: float) -> float:
        """
        calculate the probability density function (PDF) of the standard
        normal distribution

        Args:
            x (float): the input value at which to calculate the pdf
        """
        return NormalDistribution.normal_pdf(x, mean=0, sigma=1)

    @staticmethod
    def general_normal_pdf(x: float, mean: float, sigma: float) -> float:
        """
        calculate the probability density function (PDF) of the general
        normal distribution

        Args:
            x (float): the input value at which to calculate the PDF
            mean (float): the mean (average) of the normal distribution
            sigma (float): the standard deviation of the normal distribution

        Return:
            (float): the value of the PDF at the given input
        """
        return NormalDistribution.normal_pdf(x, mean, sigma)


class ContinousDistribution:
    @staticmethod
    def continous_uniform_pdf(x: float, a: float, b: float) -> float:
        """
        calculate the probability density function (PDF) of the continous
        distribution

        Args:
            x (float): the input value at which to calculate the PDF
            a (float): the lower bound of the uniform distribution
            b (float): the upper bound of the uniform distribution

        Returns:
            (float): the value of the PDF at the given input

        Example
        >>> pdf = ContinuousDistribution.continous_uniform_pdf(3.0, 2.0, 5.0)
        >>> print(f"PDF value at x = 3.0: {pdf:.6f}")
        PDF value at x = 3.0: 0.333333
        """
        if x < a or x > b:
            return 0.0
        return 1 / (b - a)

    @staticmethod
    def continous_uniform_cdf(x: float, a: float, b: float) -> float:
        """
        calculate the cumulative distribution function (CDF) of the continous
        distribution

        Args:
            x (float): the input value at which to calculate the CDF
            a (float): the lower bound of the uniform distribution
            b (float): the upper bound of the uniform distribution

        Returns:
            (float): the value of the CDDF the given output

        Example
        >>> cdf = ContinuousDistribution.continuous_uniform_cdf(3.0, 2.0, 5.0)
        >>> print(f"CDF value at x = 3.0: {cdf:.6f}")
        CDF value at x = 3.0: 0.333333
        """
        if x < a:
            return 0.0
        elif x >= b:
            return 1.0
        return (x - a) / (b - a)

    @staticmethod
    def generate_random_sample(a: float, b: float) -> float:
        """
        generate random sample from the continous uniform distribution

        Args:
            a (float): the lower bound of the uniform distribution
            b (float): the upper bound of the uniform distribution

        Returns:
            (float): random sample from the uniform distribution

        Example
        >>> sample = ContinuousDistribution.generate_random_sample(2.0, 5.0)
        >>> print(f"Generated random sample: {sample:.6f}")
        Generated random sample: 3.491482
        """
        return random.uniform(a, b)


class BetaDistribution:
    @staticmethod
    def beta_pdf(x: float, alpha: float, beta: float) -> float:
        """
        calculate the probability density function (PDF) of the beta
        distribution

        Args:
            x (float): the input value at which calculate the PDF
            alpha (float): the shape parameter alpha of the beta distribution
            beta (float): the shape parameter of the beta distribution

        Returns:
            (float): the value of the PDF at the given input

        Example
        >>> pdf = beta_pdf(0.3, alpha=2, beta=5)
        >>> print(f"PDF value at x = 0.3: {pdf:.6f}")
        PDF value at x = 0.3: 1.170225
        """
        if x < 0 or x > 1:
            return 0.0
        coefficient = math.gamma(alpha + beta) / (math.gamma(alpha) * math.gamma(beta))
        pdf_value = coefficient * (x ** (alpha - 1)) * ((1 - x) ** (beta - 1))
        return pdf_value

    @staticmethod
    def beta_cdf(x: float, alpha: float, beta: float) -> float:
        """
        calculate the cumulative distribution function (CDF) of beta distribution

        Args:
            x (float): the input value at which to calculate the CDF
            alpha (float): the shape parameter alpha of the beta distribution
            beta (float): the shape parameter beta of the beta distribution

        Returns:
            (float): the value of the CDF at the given input

        Example
        >>> cdf = beta_cdf(0.3, alpha=2, beta=5)
        >>> print(f"CDF value at x = 0.3: {cdf:.6f}")
        CDF value at x = 0.3: 0.462543
        """

        def frange(start, stop, step):
            while start < stop:
                yield start
                start += step

        if x < 0:
            return 0.0
        elif x >= 1:
            return 1.0
        return (
            sum(BetaDistribution.beta_pdf(t, alpha, beta) for t in frange(0, x, 0.001))
            * 0.001
        )


class ExponetialDistribution:
    @staticmethod
    def exponential_pdf(x: float, lambd: float) -> float:
        """
        calculate the probability density function (PDF) of the exponential distribution

        Args:
            x (float): the input value at which
            lambd (float): the rate parameter (inverse of mean) of the exponential distribution

        Returns:
            (float): the value of the PDF at given input

        Example
        >>> pdf = exponential_pdf(2.5, lambda=0.8)
        >>> print(f"PDF value at x = 2.5: {pdf:.6f}")
        print(f"PDF value at x = 2.5: {pdf:.6f}")
        """
        if x < 0:
            return 0.0
        pdf_value = lambd * math.exp(-lambd * x)
        return pdf_value

    @staticmethod
    def exponential_cdf(x: float, lambd: float) -> float:
        """
        calculate the cumulative distribution function (CDF) of the exponential distribution

        Args:
            x (float): the input value at which to calculate the CDF
            lambd (float): the rate parameter (inverse of mean) of exponential distribution

        Returns:
            (float): the value the CDF at the given input

        Example
        >>> cdf = exponential_cdf(2.5, lambd=0.8)
        >>> print(f"CDF value at x = 2.5: {cdf:.6f}")
        CDF value at x = 2.5: 0.932332
        """
        if x < 0:
            return 0.0
        cdf_value = 1 - math.exp(-lambd * x)
        return cdf_value


class DirichletDistribution:
    @staticmethod
    def dirichlet_pdf(x: list, alpha: list) -> float:
        """
        calculate the probability density function (PDF) of the dirichlet distribution

        Args:
            x (list): the input list probabilities (values should sum to 1)
            alpha (list): the concentration parameters of the dirichlet distribution

        Returns:
            (float): the value of the PDF ath the given input

        NOTE: the length of the x and alpha list should be same

        Example
        >>> pdf = dirichlet_pdf([0.3, 0.4, 0.3], alpha=[2, 3, 2])
        >>> print(f"PDF value: {pdf:.6f}")
        PDF value: 2.012448

        >>> pdf = dirichlet_pdf([0.25, 0.25, 0.5], alpha=[0.5, 0.5, 0.5])
        >>> print(f"PDF value: {pdf:.6f}")
        PDF value: 1.366025
        """
        if len(x) != len(alpha):
            raise ValueError(
                "DirichletDistribution.pdf(): length of x and alpha must same"
            )
        normalization = math.gamma(sum(alpha)) / math.prod(
            [math.gamma(a) for a in alpha]
        )
        pdf_value = normalization * math.prod(
            [xi ** (ai - 1) for xi, ai in zip(x, alpha)]
        )
        return pdf_value


class ChiSquareDistribution:
    @staticmethod
    def chi_squared_pdf(x: float, k: int) -> float:
        """
        calculate the probability density function (PDF) of the chi-squared distribution

        Args:
            x (float): the input value at which to calculate the PDF
            k (int): the degrees of freedom parameter of the chi-squared distribution

        Returns:
            (float): the value of the PDF at the given input

        Example
        >>> pdf = chi_squared_pdf(2.5, k=3)
        >>> print(f"PDF value at x = 2.5: {pdf:.6f}")
        PDF value at x = 2.5: 0.180722
        """
        if x < 0:
            return 0.0
        coefficient = 1 / (2 ** (k / 2) * math.gamma(k / 2))
        pdf_value = coefficient * (x ** (k / 2 - 1)) * math.exp(-x / 2)
        return pdf_value

    @staticmethod
    def chi_squared_cdf(x: float, df: int) -> float | list[float]:
        """
        calculate the Cumulative Distribution Function (CDF) of the Chi-Square Distribution.

        Args:
            x (float or array-like): value(s) at which to evaluate the CDF.
            df (int): degrees of freedom for the Chi-Square Distribution.

        Returns:
            (float or array-like): The calculated CDF value(s) at the given value(s) of x.

        Example
        >>> chi_square_cdf(2.5, df=2)
        0.48982412914811513

        >>> x_values = [1.0, 3.0, 5.0]
        >>> degrees_of_freedom = 4
        >>> cdf_results = chi_square_cdf(x_values, degrees_of_freedom)
        >>> cdf_results
        [0.19914827347145588, 0.7733726476231319, 0.9453031234212763]
        """
        if x < 0 or df <= 0:
            raise ValueError("x must be non-negative and df must be positive.")

        # Regularized incomplete gamma function
        def gamma_incomplete(a, x):
            term = t = 1.0
            s = term
            for i in range(1, 1000):
                t *= x / (a + i)
                term = t / (a + i)
                s += term
                if abs(term) < 1e-8:
                    break
            return s * math.exp(-x) * (x**a) / math.gamma(a)

        cdf_values = 1.0 - gamma_incomplete(df / 2.0, x / 2.0)
        return cdf_values


class Poisson:
    """
    # Description
    Poisson Distribution is a discrete probability distribution
    that expresses the probabilty of a given number of even
    occurring in fixed interval of time or space if these
    events occur with a known constant mean rate and independently of
    the time since the last event.
    """

    @staticmethod
    def poisson_pmf(x: int, alpha: float) -> float:
        """
        calculate with pmf of poisson distributions algorithm
        Args:
            x (int): input value for calculate
            alpha (float,optional): this is a degree
                                    freedom for calculate
                                    that.
        Returns:
            float: result from that calculate
        Example:
        >>> from Hela.common.distribution import Poisson
        >>> Poisson.poisson_pmf(5,2)
        0.03608940886309672
        """
        result = pow(alpha, x) * np.exp(-alpha)
        result /= math.factorial(x)
        return result

    @staticmethod
    def poisson_cdf(x: int, alpha: float):
        """
        calculate with cdf of poisson distribution algorithm
        Args:
            x (int): input value for calculate
            alpha (float,optional): this is a degree
                                    freedom for calculate
                                    that.
        Returns:
            float: resukt from that calculate
        Example:
        >>> from Hela.common.distribution import Poisson
        >>> Poisson.poisson_cdf(5,2)
        array([0.13533528, 0.27067057, 0.27067057, 0.18044704, 0.09022352])
        """
        par_range = np.arange(x)
        result = np.array([pow(alpha, i) / math.factorial(i) for i in par_range])
        result *= np.exp(-alpha)
        return result


class student_distribution:
    """
    Studentn Distribution (familiar with name t-distribution)
    is a continous probabilty distribution that  generalizes
    the standard normal distribution. Like the latter,
    it is symmetric around zero and bell-shaped.
    """

    @staticmethod
    def t_distribution_pdf(vector: np.ndarray, degree: int = 10) -> float:
        """
        Calculate with Student Distribution with pdf
        Args:
        vector (np.ndarray) : this paramater represents input values
        degree (int,optional) : value for represents particular node

        Returns:
            float: result from that
        Example:
        >>> from Hela.common.distribution import student_distribution
        >>> import numpy as np
        >>> a = np.array([1,2,3,4])
        >>> student_distribution.t_distribution_pdf(a,10)
        array([0.61917584, 8.00552478, 8.00552478, 0.61917584])
        """
        t = vector - np.mean(vector)
        t /= np.std(vector) / np.sqrt(vector.shape[0])

        result = (
            math.gamma((degree + 1) / 2)
            / math.gamma(degree / 2)
            * np.sqrt(degree * np.pi)
        )
        result *= (t**2 / degree + 1) ** -((degree + 1) / 2)
        return result
