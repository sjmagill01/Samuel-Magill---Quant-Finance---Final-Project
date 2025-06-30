"""
Black-Scholes analytic and Monte Carlo functions
"""

import numpy as np
from scipy.stats import norm

# Write functions that finds Black-Scholes value of call and put options

def bs_call(S0, K, sigma, t, r = 0):
    """
    Computes the Black-Scholes price of a European call option.

    Parameters:
        S0 (float): Current asset price
        K (float): Strike price
        sigma (float): Annualized volatility (standard deviation of log returns)
        t (float): Time to expiration (in years)
        r (float): Risk-free interest rate (annualized)

    Returns:
        float: Call option price
    """
    d1 = (np.log(S0/K) + (r+.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    
    call_price = S0*norm.cdf(d1) - K*np.exp(-r*t)*norm.cdf(d2)
    return call_price




def bs_put(S0, K, sigma, t, r=0):
    """
    Description:
    
    Computes the Black-Scholes value of a European put option.
    
    Parameters:
        S0: Current asset price
        K: Strike price
        sigma: Yearly standard deviation of log-returns (volatility)
        t: Time to expiration (in years)
        r: Risk-free interest rate
    
    Returns:
        Put option price
    """
    d1 = (np.log(S0/K) + (r+.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    
    put_price = -S0*norm.cdf(-d1) + K*np.exp(-r*t)*norm.cdf(-d2)
    return put_price
    
    
       

#Create function that does monte-carlo simulation of call option price
def monte_sim_call(S0, K, sigma, t, r, N):
    """
    Monte Carlo simulation to estimate the price of a European call option 
    under the Black-Scholes model.

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        sigma (float): Volatility of the stock
        t (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        N (int): Number of simulated paths

    Returns:
        float: Estimated call option price
    """
    random_noise = np.random.normal(0, 1, N)

    exponents = (r - .5*sigma**2)*t + sigma*np.sqrt(t)*random_noise

    end_points = S0*np.exp(exponents)

    call_payouts = np.maximum(end_points - K, 0)

    call_sim_value = np.exp(-r*t)*np.mean(call_payouts)

    return call_sim_value
    

#Create function that does monte-carlo simulation of call option price

def monte_sim_put(S0, K, sigma, t, r, N):
    """
    Monte Carlo simulation to estimate the price of a European put option 
    under the Black-Scholes model.

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        sigma (float): Volatility of the stock
        t (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        N (int): Number of simulated paths

    Returns:
        Estimated put option price
    """
    
    random_noise = np.random.normal(0, 1, N)

    exponents = (r - .5*sigma**2)*t + sigma*np.sqrt(t)*random_noise

    end_points = S0*np.exp(exponents)

    put_payouts = np.maximum(-end_points + K, 0)

    put_sim_value = np.exp(-r*t)*np.mean(put_payouts)

    return put_sim_value
    
    

import numpy as np
from scipy.stats import norm

def bs_call_delta(S0, K, sigma, t, r):
    """
    Returns the Delta (sensitivity to spot price) of a European call option
    under Black-Scholes assumptions.

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        sigma (float): Volatility of the stock
        t (float): Time to maturity (in years)
        r (float): Risk-free interest rate

    Returns:
        float: Delta of Call Option
    """
    d1 = 
    return 


def bs_put_delta(S0, K, sigma, t, r):
    """
    Returns the Delta (sensitivity to spot price) of a European put option
    under Black-Scholes assumptions.

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        sigma (float): Volatility of the stock
        t (float): Time to maturity (in years)
        r (float): Risk-free interest rate

    Returns:
        float: Delta of Put Option
    """
    d1 = 
    return 


