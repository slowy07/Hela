from __future__ import annotations
from typing import Callable, Union

import numpy as np
from Hela.common.differential import Differential


class AntiDifferential:
    @staticmethod
    def general_antiderivative(
        f: Callable[[float], float],
        a: float,
        b: float,
        num_interval: int = 4,
        method: str = "riemann",
    ) -> float:
        """
        Calculate the general antiderivative of a function.

        Args:
            f (Callable[[float], float]): The function to integrate.
            a (float): The lower limit of integration.
            b (float): The upper limit of integration.
            num_interval (int, optional): Number of intervals for calculation. Defaults to 4.
            method (str): The method for computing the antiderivative. Options are "riemann" and "trapezoidal".

        Returns:
            float: The antiderivative result of the function.
        """
        if a >= b:
            raise ValueError("Your input a must be less than b value")

        try:
            delta_x = (b - a) / float(num_interval)

            if method == "riemann":
                result = 0.0  # Ensure result is a float
                for i in range(num_interval):
                    x_i = a + i * delta_x
                    result += f(x_i) * delta_x
                return result

            if method == "trapezoidal":
                result = 0.5 * (f(a) + f(b))
                for i in range(1, num_interval):
                    result += f(a + i * delta_x)
                return result * delta_x

        except Exception as error_integral:
            raise ValueError(f"Error: {error_integral}")

        return 0.0

    @staticmethod
    def PowerRule_antiderivative(n: int, x: float) -> float:
        """
        Calculate the antiderivative of x^n using the power rule.

        Args:
            n (int): The exponent of the power function.
            x (float): The value at which to calculate the antiderivative.

        Returns:
            float: The antiderivative result of the Power Rule.
        """
        try:
            return np.power(x, n + 1) / float(n + 1)

        except Exception as powerrule_antidifferential:
            raise ValueError(f"Error: {powerrule_antidifferential}")

    @staticmethod
    def Usubstitution(
        f: Callable[[float], float], g: Callable[[float], float], a: float, b: float
    ) -> float:
        """
        Perform integration by substitution (U-substitution).

        Args:
            f (Callable[[float], float]): First function u(x).
            g (Callable[[float], float]): Second function u(x).
            a (float): The lower limit of integration.
            b (float): The upper limit of integration.

        Returns:
            float: Result of the U-substitution calculation.
        """
        try:
            g_derivative = Differential.derivative(g, a, b)
            f_g_derivative = lambda x: f(g(x)) * g_derivative
            u_integral = AntiDifferential.general_antiderivative(f_g_derivative, a, b)
            return u_integral
        except Exception as error_value:
            raise ValueError(f"Error: {error_value}")

    @staticmethod
    def Mean_Value_AntiDifferential(
        f: Callable[[float], float], a: float, b: float
    ) -> float:
        """
        Calculate the mean value using the Mean Value Theorem for integrals.

        Args:
            f (Callable[[float], float]): The function for which to calculate the mean value.
            a (float): The lower limit of integration.
            b (float): The upper limit of integration.

        Returns:
            float: Result of the Mean Value Theorem on the integral.
        """
        return AntiDifferential.general_antiderivative(f, a, b) / (b - a)
