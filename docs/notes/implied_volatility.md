# Implied Volatility Notes
---
## What is implied volatility used for?

- Create volatility surface -> IV x maturity x strike_price
- More realistic BSM prices
- Determines overvalued and undervalued stocks 
    - Overpriced  -> IV too high
    - Underpriced -> IV too low
- IV is required for greek hedging input (Delta, Gamma, Vega, Theta)
- Used for backtesting
---
## So how do you get the implied volatility number? 

Implied volatility requires both:
- Black–Scholes price (to compute pricing error)
- Black–Scholes vega (to adjust volatility using Newton's method)

Vega is the derivative of option price with respect to volatility. 
It cannot compute IV by itself; it is only used to refine the σ guess.

Newton iteration:
    sigma_new = sigma_old - (BS(sigma_old) - market_price) / vega(sigma_old)

### Analogy (simplified)
- Imagine driving toward a target:
- BSM price = your current location
- Market price = the destination
- Error = how far you are from the destination
- Vega = how sensitive steering is
- Newton step = adjusting steering to correct direction
You need location and steering — not just steering.

---
