# pricers/implied_volatility.py
from __future__ import annotations
from typing import Optional
from math import isfinite

from src.option_pricer.utils.pricers.black_scholes import (
    black_scholes_price,
    bsm_vega,
)


def _price_bounds(S: float, K: float, r: float, q: float, tau: float, option_type: str):
    """
    Returns (lower_bound, upper_bound) arbitrage bounds for option prices under continuous rates:
      Call:  lower = max(0, S*e^{-q T} - K*e^{-r T}), upper = S*e^{-q T}
      Put :  lower = max(0, K*e^{-r T} - S*e^{-q T}), upper = K*e^{-r T}
    """
    from math import exp

    df_r = exp(-r * tau)
    df_q = exp(-q * tau)
    if option_type.upper().startswith("C"):
        lower = max(0.0, S * df_q - K * df_r)
        upper = S * df_q
    else:
        lower = max(0.0, K * df_r - S * df_q)
        upper = K * df_r
    return lower, upper


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
    max_iter: int = 20,
) -> float:

    sigma = initial_vol

    print("\n--- NEWTON START ---")
    print(f"S={S}, K={K}, tau={tau}, market={market_price}, type={option_type}")

    for i in range(max_iter):

        model_price = black_scholes_price(S, K, r, q, sigma, tau, option_type)
        diff = model_price - market_price
        v = bsm_vega(S, K, r, q, sigma, tau)

        print(
            f"Iter {i} | sigma={sigma:.6f} | model={model_price:.6f} | "
            f"market={market_price:.6f} | diff={diff:.6f} | vega={v:.6f}"
        )

        # Convergence reached
        if abs(diff) < tol:
            print(f"NEWTON SUCCESS → sigma={sigma:.6f}")
            return sigma

        # Stop if vega too small
        if v < 1e-8:
            print("STOP: Vega too small → fallback to bisection")
            return None

        # Newton step
        sigma = sigma - diff / v

        # Don't allow negative sigma
        if sigma <= 0:
            print("Sigma went negative, resetting to 1e-6")
            sigma = 1e-6

    print("NEWTON FAILED → Fallback to bisection")
    return None



def implied_vol_bisection(
    market_price, S, K, r, q, tau, option_type,
    low=1e-6, high=5.0, tol=1e-6, max_iter=100
):

    print("\n--- BISECTION START ---")
    print(f"S={S}, K={K}, tau={tau}, market={market_price}, type={option_type}")

    for i in range(max_iter):
        mid = (low + high) / 2
        price = black_scholes_price(S, K, r, q, mid, tau, option_type)

        print(
            f"Iter {i} | mid={mid:.6f} | price={price:.6f} | "
            f"low={low:.6f} | high={high:.6f}"
        )

        if abs(price - market_price) < tol:
            print(f"BISECTION SUCCESS → sigma={mid:.6f}")
            return mid

        if price > market_price:
            high = mid
        else:
            low = mid

    final_sigma = (low + high) / 2
    print(f"BISECTION END → sigma={final_sigma:.6f}")
    return final_sigma



def implied_volatility(
    price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    tau: float,
    option_type: str,
    initial_vol: float = 0.2,
    tol: float = 1e-6,
    max_iter_newton: int = 30,
    low: float = 1e-6,
    high: float = 5.0,
    max_iter_bisection: int = 200,
) -> Optional[float]:
    """
    Unified solver: Newton first (fast), fallback to robust bisection only if
    Newton explicitly returns None (indicating no root or numeric issues).
    Returns None if no plausible IV exists.
    """

    # quick input guard
    if price is None or S is None or K is None or tau is None:
        return None
    if tau <= 0:
        return None

    # try Newton
    iv_nr = implied_vol_newton(
        market_price=price,
        S=S,
        K=K,
        r=r,
        q=q,
        tau=tau,
        option_type=option_type,
        initial_vol=initial_vol,
        tol=tol,
        max_iter=max_iter_newton,
    )
    if iv_nr is not None:
        return iv_nr

    # Newton failed; try bisection (only if price is inside arbitrage bounds)
    lb, ub = _price_bounds(S, K, r, q, tau, option_type)
    if price < lb - 1e-12 or price > ub + 1e-12:
        # out of arbitrage bounds -> no valid IV
        return None

    return implied_vol_bisection(
        market_price=price,
        S=S,
        K=K,
        r=r,
        q=q,
        tau=tau,
        option_type=option_type,
        low=low,
        high=high,
        tol=tol,
        max_iter=max_iter_bisection,
    )
