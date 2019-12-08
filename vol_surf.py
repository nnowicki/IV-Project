import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yahoo_fin import options
from yahoo_fin import stock_info
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
from datetime import date, datetime
import time


def Dte(day):
	'''
	Converts a maturity date string to days to expiration
	Inputs:
		day: (str) a date string of form "YYYY-MM-DD"
	Outputs:
		Days to expiration
	'''
	time_delta = datetime.strptime(day, '%Y-%m-%d').date() - date.today()
	return time_delta.days

def div_yield(ticker):
	'''
	Returns dividend yield of stock/ETF from Yahoo Finance API
	Inputs:
		ticker: (str) Stock or ETF ticker symbol
	Returns:
		(float) Dividend yield (or zero if no dividends)
	'''
	info = stock_info.get_quote_table(ticker)
	if "Yield" in info.keys():
		return float(info["Yield"].rstrip("%"))
	elif info["Forward Dividend & Yield"][:3] != "N/A":
		fdy = info["Forward Dividend & Yield"].split(" ")[1]
		fdy = fdy.replace("%)", "").lstrip("(")  
		return float(fdy)
	else:
		return 0

def expiries(ticker):
	'''
	Gets all maturities for options on a desired stock
	Inputs:
		ticker: (str) Stock or ETF ticker symbol
	Returns:
		expiries: (list) List of all maturities 
	'''
	expiries = options.get_expiration_dates(ticker)
	expiries = [x.replace(',', "") for x in expiries]
	for i, date in enumerate(expiries):
		m,d,y = date.split(" ")
		if len(d) == 1: d = "0" + d
		date = " ".join([m,d,y])
		t = time.strptime(date, "%B %d %Y")
		m,d,y = str(t.tm_mon), str(t.tm_mday), str(t.tm_year)
		if len(m) == 1: m = "0" + m
		if len(d) == 1: d = "0" + d
		date = "-".join([y,m,d])
		expiries[i] = date
	return expiries

def get_chain(ticker, date=None, c_p='Call'):
	'''
	Gets option chain from Yahoo Finance API
	Inputs:
		ticker: (str) Stock or ETF ticker symbol
		date: (str) Indicates maturity date for option (e.g: "2019-12-06")
		c_p: (str) Indicates call or put option
	Returns:
		Pandas DataFrame of option chain
	'''
	cols = {'Implied Volatility': 'IV'}
	if c_p == 'Call': 
		df = options.get_calls(ticker,date).rename(columns=cols)
		return df.set_index("Strike")
	if c_p == 'Put': 
		df = options.get_puts(ticker,date).rename(columns=cols)
		return df.set_index("Strike")

def strikes(ticker, expiries, c_p='Call', tot_dates=5):
	'''
	Returns the intersection of all strikes available for the desired 
	maturities.
	Inputs:
		ticker: (str) Stock or ETF ticker symbol
		expiries: (list) List of maturities (date strings)
		c_p: (str) Indicates call or put option
		tot_dates: (int) Number of maturities plotted on surface
	Returns:
		all_chains: (dict) Dictionary of all option chains by maturity
		intersected: (list) All strikes available for the given maturities
	'''
	tot, all_chains = [], {}
	for day in expiries[:tot_dates]:
		chain = get_chain(ticker, day, c_p)
		tot.append(list(chain.index))
		all_chains[Dte(day)] = chain
	try:
		intersected = set(tot[0]).intersection(*tot)
		intersected = sorted(list(intersected))
	except ValueError:
		intersected = set()
	return all_chains, intersected

def surface(ticker, c_p='Call', tot_dates=5):
	'''
	Plot the implied volatility surface for a given ticker
	Inputs:
		ticker: (str) Stock or ETF ticker symbol
		c_p: (str) Indicates call or put option
		tot_dates: (int) Number of maturities plotted on surface
	'''
	y_0 = expiries(ticker)[:tot_dates]
	chains, x = strikes(ticker, y_0, c_p, tot_dates)
	today = date.today()
	y, z = [Dte(day) for day in y_0], []
	X, Y = np.meshgrid(x, y)
	S, q = stock_info.get_live_price(ticker), div_yield(ticker)
	for K, dte in zip(np.ravel(X), np.ravel(Y)):
		chain = chains[dte]
		option_price = chain.loc[K]["Ask"]
		print("dte:",dte)
		z.append(BSM_vol(S, K, dte / 365, option_price, q, c_p))
	Z = np.array(z).reshape(X.shape)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
	ax.set_xlabel('Strikes')
	ax.set_ylabel('Days to Expiration')
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.title("IV Surface for {} {} Options".format(ticker,c_p))
	plt.show()

def BSM_price(S, X, T, q, sig, c_p="Call", r=0.019):
	'''
	Calculate price of a European option using Black-Scholes-Merton formula
	Inputs:
		S: (float) Current stock price
		X: (float) Current strike price
		T: (float) Days to expiration
		q: (float) Dividend yield
		c_p: (str) Indicates call or put option
		sig: (float) Volatility 
		r: (float) Risk-free rate (1y T-Bill)
	'''
	d1 = (np.log(S / X) + (r - q + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
	d2 = d1 - (sig * np.sqrt(T))
	if c_p == "Call":
		return S * np.exp(-q * T) * norm.cdf(d1) - X * \
			np.exp(-r * T) * norm.cdf(d2)
	if c_p == "Put":
		return X * np.exp(-r * T) * norm.cdf(-d2) - S * \
			np.exp(-q * T) * norm.cdf(-d1)

def BSM_vega(S, X, T, q, sig, r=0.019):
	'''
	Calculate vega of a European option using Black-Scholes-Merton formula
	Inputs:
		S: (float) Current stock price
		X: (float) Current strike price
		T: (float) Days to expiration
		q: (float) Dividend yield
		sig: (float) Volatility 
		r: (float) Risk-free rate (1y T-Bill)
	'''
	d1 = (np.log(S / X) + (r - q + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
	d2 = d1 - (sig * np.sqrt(T))
	return (1 / np.sqrt(2 * np.pi)) * S * np.exp(-q * T) * \
		np.sqrt(T) * np.exp(-(norm.cdf(d1) ** 2) / 2)


def BSM_vol(S, X, T, price, q, c_p="Call", sig=0.25, r=0.019):		
	'''
	Calculate the implied volatility of a European option via the 
	Newton-Raphson method
	Inputs:
		S: (float) Current stock price
		X: (float) Current strike price
		T: (float) Days to expiration
		price: (float) Market option price
		q: (float) Stock dividend yield
		c_p: (str) Indicates call or put option
		sig: (float) Initial volatility guess 
		r: (float) Risk-free rate (1y T-Bill)

	Returns: (float) Estimated implied volatility of option
	'''
	epsilon = 0.0001
	old = sig
	new = sig + 0.05
	if T == 0: T = T+0.5
		
	while abs(new - old) > epsilon:
		old = new
		vega = BSM_vega(S, X, T, q, new, r)
		new = new - (BSM_price(S, X, T, q, new, c_p, r) - price) / vega
	return 100 * abs(new)

if __name__ == "__main__":
	ticker = input("Enter a ticker: ")
	c_p = input('Call or Put? ')
	num_strikes = int(input('Number of strikes? '))
	surface(ticker, c_p, num_strikes)

