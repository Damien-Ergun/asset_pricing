import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def implied_volatility_objective(sigma, S, K, T, r, market_price):
    return (black_scholes_call(S, K, T, r, sigma) - market_price) ** 2

def calculate_iv(row):
    S = row['S']
    K = row['Strike']
    T = row['T']
    r = row['interest_rate']
    market_price = row['market_price']
    result = minimize(implied_volatility_objective, x0=np.array([0.2]), 
                      args=(S, K, T, r, market_price), 
                      bounds=[(0.001, None)])
    return result.x[0] if result.success else np.nan

# Example DataFrame
data = {
    'S': [100, 105],  # Current stock prices
    'Strike': [100, 110],  # Strike prices
    'T': [1, 0.5],  # Times to maturity
    'interest_rate': [0.05, 0.05],  # Risk-free rates
    'market_price': [10, 8]  # Market prices of the options
}
df = pd.DataFrame(data)

# Calculate Implied Volatility for each row
df['Implied_Volatility'] = df.apply(calculate_iv, axis=1)
