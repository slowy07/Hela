from __future__ import annotations
from typing import Callable

import numpy as np
from Hela.common import differential


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
            return float(result)

        if method == "trapezoidal":
            result = 0.5 * (f(a) + f(b))
            for i in range(1, num_interval):
                result += f(a + i * delta_x)
            return result * delta_x

        if method == "simpson":
            s = f(a) + f(b)
            for i in range(1, num_interval, 2):
                s += 4 * f(a + i * delta_x)
            for i in range(2, num_interval - 1, 2):
                s += 2 * f(a + i * delta_x)
            return s * delta_x / 3

    except Exception as error_integral:
        raise ValueError(f"Error: {error_integral}")

    return 0.0


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
        if not isinstance(n, (float, int)):
            raise TypeError(
                f"exponent power function (n) must be integer or float, got {type(n)}"
            )
        if not isinstance(x, (float, int)):
            raise TypeError(
                f"value to calculate antiderivative (x) must be float or integer, got {type(n)}"
            )
        if n < 1 or x < 1:
            raise ValueError("value of parameter must >= 1")
        else:
            return np.power(x, n + 1) / float(n + 1)

    except Exception as powerrule_antidifferential:
        raise ValueError(f"Error: {powerrule_antidifferential}")


def Usubstitution(
    f: Callable[[float], float], g: Callable[[float], float], a: float, b: float
) -> float:
    """
    Perform integration by substitution (U-substitution).

    Args:
        f (Callable[[float], float]): First function u(x).
        g (Callable[[float], float]): Second function u(x).
        a (float): The lower limit of integration.
        b (float): The upper limit of integrataaion.

    Returns:
        float: Result of the U-substitution calculation.
    """
    try:
        g_derivative = differential.derivative(g, a, b)
        f_g_derivative = lambda x: f(g(x)) * g_derivative
        u_integral = general_antiderivative(f_g_derivative, a, b)
        return u_integral
    except Exception as error_value:
        raise ValueError(f"Error: {error_value}")


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
    return general_antiderivative(f, a, b) / (b - a)


def Symmetry_integral(f: Callable[[float], float], points: tuple[int, int]) -> float:
    """_summary_

    Calculate the integral of a function over symmetry range of
    x-values.

    Args:

        f (Callable): the function to integrate
        points (tuple[int,int]): tuples of x-values defining the range

    """
    neg_x_points = {(-x, y) for x, y in [points]}

    x, y = points
    if (-x, y) not in neg_x_points:
        result = general_antiderivative(f, 0, y)
        return result
    else:
        return 0.0


def partialIntegral(
    f: Callable[[float], float],
    g: Callable[[float], float],
    a: int | float,
    b: int | float,
) -> float:
    """
    Calculate the integral of a partial Integral
    Args:
        f (Callable[[float],float]): f value
        g (Callable[[float],float]): g value
        a (int | float): lower interval
        b (int | float): high interval
    Returns:
        float: result of partial integral
    """
    try:
        # Calculate the result of the partial integral using the formula
        result = f(b) * g(b) - f(a) * g(a) - Usubstitution(f, g, a, b)
        return result
    except Exception as error_value:
        raise ValueError(f"Error: {error_value}")
