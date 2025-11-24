# pricers/implied_volatility.py

"""
Implied volatility solvers using Black–Scholes.
Uses:
- black_scholes_price
- bsm_vega
from pricers.black_scholes
"""

from __future__ import annotations
from math import fabs

from pricers.black_scholes import black_scholes_price, bsm_vega


def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    tau: float,
    option_type: str,
    initial_vol: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Compute implied volatility using Newton–Raphson method.
    Returns sigma, or None if fails.
    """

    sigma = initial_vol

    for _ in range(max_iter):

        model_price = black_scholes_price(S, K, r, q, sigma, tau, option_type)
        diff = model_price - market_price  # f(sigma)

        if abs(diff) < tol:
            return sigma

        v = bsm_vega(S, K, r, q, sigma, tau)
        if v < 1e-8:  # avoid division by very small numbers
            break

        sigma = sigma - diff / v  # Newton update

        # keep sigma positive
        if sigma <= 0:
            sigma = 1e-6

    return None  # did not converge


def implied_vol_bisection(
    market_price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    tau: float,
    option_type: str,
    low: float = 1e-6,
    high: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Bisection method for IV.
    Slow but guaranteed to converge.
    """

    for _ in range(max_iter):
        mid = (low + high) / 2
        price = black_scholes_price(S, K, r, q, mid, tau, option_type)

        if abs(price - market_price) < tol:
            return mid

        if price > market_price:
            high = mid
        else:
            low = mid

    return (low + high) / 2
