# pricers/black_scholes.py
"""
Black-Scholes pricing and Greeks.

Functions:
- black_scholes_price(S, K, r, q, sigma, tau, option_type)
- bsm_vega(S, K, r, q, sigma, tau)
- bsm_delta(...)
- bsm_gamma(...)
- bsm_theta(...)
- bsm_rho(...)
"""

from __future__ import annotations

from math import log, sqrt, exp
from typing import Literal, Tuple

from scipy.stats import norm

OptionType = Literal["C", "P"]


def _d1_d2(S: float, K: float, r: float, q: float, sigma: float, tau: float) -> Tuple[float, float]:
    """
    Compute d1 and d2 used in Black-Scholes formula.

    Returns (d1, d2).

    Notes:
    - If tau <= 0 or sigma <= 0 we return values that will be handled
      by callers (they typically handle tau<=0 / sigma<=0 as special cases).
    """
    if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # Return huge values that make norm.cdf(d) approach 1 or 0 as needed.
        # Callers handle tau<=0 separately, but this avoids division by zero.
        return float("inf"), float("inf")
    sqrt_t = sqrt(tau)
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def black_scholes_price(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    tau: float,
    option_type: OptionType,
) -> float:
    """
    Compute Black-Scholes (European) option price.

    Parameters
    ----------
    S : float
        Spot price of the underlying.
    K : float
        Strike price.
    r : float
        Continuous risk-free rate (annual).
    q : float
        Continuous dividend yield (annual).
    sigma : float
        Annual volatility (decimal; e.g., 0.2 for 20%).
    tau : float
        Time to expiry in years.
    option_type : "C" or "P"
        Call or Put.

    Returns
    -------
    float
        Model price (same units as S/K).
    """
    # Immediate expiry: option value is intrinsic
    if tau <= 0:
        if option_type == "C":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    # Zero sigma (volatility): return forward price minus discounted strike
    if sigma <= 0:
        df_r = exp(-r * tau)
        df_q = exp(-q * tau)
        if option_type == "C":
            return max(S * df_q - K * df_r, 0.0)
        else:
            return max(K * df_r - S * df_q, 0.0)

    d1, d2 = _d1_d2(S, K, r, q, sigma, tau)
    # risk free discount factor
    df_r = exp(-r * tau)
    # dividend discount factor
    df_q = exp(-q * tau)

    if option_type == "C":
        price = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    else:
        price = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)

    return float(price)


def bsm_vega(S: float, K: float, r: float, q: float, sigma: float, tau: float) -> float:
    """
    Vega: ∂Price / ∂sigma. Returned value is per 1.0 change in sigma (not per 1%).
    """
    if tau <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, r, q, sigma, tau)
    return float(S * exp(-q * tau) * sqrt(tau) * norm.pdf(d1))


def bsm_delta(S: float, K: float, r: float, q: float, sigma: float, tau: float, option_type: OptionType) -> float:
    """
    Delta: ∂Price / ∂S.
    For immediate expiry returns the derivative of the payoff (0/1 or -1/0).
    """
    if tau <= 0:
        if option_type == "C":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1, _ = _d1_d2(S, K, r, q, sigma, tau)
    if option_type == "C":
        return float(exp(-q * tau) * norm.cdf(d1))
    else:
        return float(exp(-q * tau) * (norm.cdf(d1) - 1.0))


def bsm_gamma(S: float, K: float, r: float, q: float, sigma: float, tau: float) -> float:
    """
    Gamma: second derivative of price w.r.t S.
    """
    if tau <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, r, q, sigma, tau)
    return float(exp(-q * tau) * norm.pdf(d1) / (S * sigma * sqrt(tau)))


def bsm_theta(
    S: float, K: float, r: float, q: float, sigma: float, tau: float, option_type: OptionType
) -> float:
    """
    Theta: ∂Price / ∂t (time decay). Returned as annualized rate (price change per year).
    Many implementations divide by 365 to get per-day; keep it per-year for consistency.
    """
    # Immediate expiry: theta is undefined in the continuous sense; return 0.0
    if tau <= 0:
        return 0.0

    d1, d2 = _d1_d2(S, K, r, q, sigma, tau)
    df_r = exp(-r * tau)
    df_q = exp(-q * tau)

    term1 = - (S * df_q * norm.pdf(d1) * sigma) / (2 * sqrt(tau))
    if option_type == "C":
        term2 = q * S * df_q * norm.cdf(d1)
        term3 = - r * K * df_r * norm.cdf(d2)
        theta = term1 + term2 + term3
    else:
        term2 = - q * S * df_q * norm.cdf(-d1)
        term3 = r * K * df_r * norm.cdf(-d2)
        theta = term1 + term2 + term3

    return float(theta)


def bsm_rho(S: float, K: float, r: float, q: float, sigma: float, tau: float, option_type: OptionType) -> float:
    """
    Rho: ∂Price / ∂r (sensitivity to risk-free rate).
    """
    if tau <= 0:
        return 0.0

    d1, d2 = _d1_d2(S, K, r, q, sigma, tau)
    df_r = exp(-r * tau)

    if option_type == "C":
        return float(K * tau * df_r * norm.cdf(d2))
    else:
        return float(-K * tau * df_r * norm.cdf(-d2))
