import yfinance as yf
import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import os
aapl = pd.read_csv('/content/drive/My Drive/AAPL_data.csv')
msft = pd.read_csv('/content/drive/My Drive/MSFT_data.csv')


msft.tail(10)

# Last Closing Price was about $435 per share

aapl.tail(10)

# Last Closing Price was about $228 per share

# Calculating Returns 
aapl['Return'] = aapl['Adj Close'].pct_change()
msft['Return'] = msft['Adj Close'].pct_change()

aapl_returns  = aapl['Return'].dropna()
msft_returns = msft['Return'].dropna()

import matplotlib.pyplot as plt

# Histograms plotting the two returns
aapl_returns.hist(bins=50, alpha=0.5, label='AAPL Returns')
msft_returns.hist(bins=50, alpha=0.5, label='MSFT Returns')
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.legend()
plt.show()


import numpy as np

# Compare with Log Normal

# Log Distribution
aapl_log_returns = np.log(1 + aapl_returns)
msft_log_returns = np.log(1 + msft_returns)

aapl_log_returns.hist(bins=50, alpha=0.5, label='AAPL Log Returns')
msft_log_returns.hist(bins=50, alpha=0.5, label='MSFT Log Returns')
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.legend()
plt.show()

from scipy import stats
from scipy.stats import norm

mu_aapl,std_aapl = norm.fit(aapl_returns)
mu_msft,std_msft = norm.fit(msft_returns)

stats.probplot(aapl_returns, dist=stats.norm, plot=plt)
plt.title("AAPL QQ-Plot (Normal)")
plt.show()

print('')

stats.probplot(msft_returns, dist=stats.norm, plot=plt)
plt.title("MSFT QQ-Plot (Normal)")
plt.show()


from scipy import stats
from scipy.stats import norm

mu_aapl,std_aapl = norm.fit(aapl_log_returns)
mu_msft,std_msft = norm.fit(msft_log_returns)


stats.probplot(aapl_log_returns, dist=stats.norm, plot=plt)
plt.title("AAPL QQ-Plot (Normal)")
plt.show()

print('')

stats.probplot(msft_log_returns, dist=stats.norm, plot=plt)
plt.title("MSFT QQ-Plot (Normal)")
plt.show()


from scipy.stats import kstest

# Test for normal distribution
ks_stat_aapl, p_value_aapl = kstest(aapl_log_returns, 'norm', args=(mu_aapl, std_aapl))
ks_stat_msft, p_value_msft = kstest(msft_log_returns, 'norm', args=(mu_msft,std_msft))

print(f"KS Test - AAPL Log Normal: statistic = {ks_stat_aapl}, p-value = {p_value_aapl}")
print(f"KS Test - MSFT  Log Normal: statistic = {ks_stat_msft}, p-value = {p_value_msft}")


from scipy.stats import kstest

# Test for normal distribution
ks_stat_aapl, p_value_aapl = kstest(aapl_returns, 'norm', args=(mu_aapl, std_aapl))
ks_stat_msft, p_value_msft = kstest(msft_returns, 'norm', args=(mu_msft,std_msft))

print(f"KS Test - AAPL Normal: statistic = {ks_stat_aapl}, p-value = {p_value_aapl}")
print(f"KS Test - MSFT Normal: statistic = {ks_stat_msft}, p-value = {p_value_msft}")



import statsmodels.api as sm

def calculate_aic_bic(model):
    """Calculates AIC and BIC for a given statistical model."""
    aic = model.aic
    bic = model.bic
    return aic, bic

# Fit a linear regression model with AAPL returns as the independent variable and MSFT returns as the dependent variable
X = sm.add_constant(aapl_returns)  # Add a constant term for the intercept
model = sm.OLS(msft_returns, X).fit()

aic, bic = calculate_aic_bic(model)

print(f"AIC: {aic}")
print(f"BIC: {bic}")


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Parameters (replace these with your calculated values)

correlation = aapl_returns.corr(msft_returns)

# Create a grid of points to evaluate the joint PDF
x = np.linspace(mu_aapl - 3*std_aapl, mu_aapl + 3*std_aapl, 100)
y = np.linspace(mu_msft - 3*std_msft, mu_msft + 3*std_msft, 100)
X, Y = np.meshgrid(x, y)

# Create the covariance matrix
covariance_matrix = np.array([[std_aapl**2, correlation * std_aapl * std_msft],
                               [correlation * std_aapl * std_msft, std_msft**2]])

# Create the bivariate normal distribution
rv = multivariate_normal(mean=[mu_aapl, mu_msft], cov=covariance_matrix)

# Evaluate the PDF over the grid
Z = rv.pdf(np.dstack((X, Y)))

# Plot the joint distribution
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar()
plt.xlabel('AAPL Returns')
plt.ylabel('MSFT Returns')
plt.title('Bivariate Normal Joint Distribution')
plt.show()


import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

window_size = 60  # Rolling window size
predictions = []

# Iterate through the DataFrame with a rolling window
for start in range(len(aapl_returns) - window_size):
    end = start + window_size
    window_data = aapl_returns.iloc[start:end]
    
    covariance_matrix = np.array([[std_aapl**2, correlation * std_aapl * std_msft],
                                   [correlation * std_aapl * std_msft, std_msft**2]])
    
    # Fit the bivariate normal distribution
    rv = multivariate_normal(mean=[mu_aapl, mu_msft], cov=covariance_matrix)

    # Generate predictions for the next day
    next_day_pred = rv.rvs(size=1)
    predictions.append(next_day_pred)

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions, columns=['Pred_AAPL', 'Pred_MSFT'])

# Concatenate with the original returns DataFrame for analysis
backtest_results = pd.concat([aapl_returns[window_size:].reset_index(drop=True), predictions_df], axis=1)

backtest_results


from sklearn.metrics import mean_squared_error

# Calculate MSE for AAPL and MSFT
mse_aapl = mean_squared_error(backtest_results['Return'], backtest_results['Pred_AAPL']) 
mse_msft = mean_squared_error(backtest_results['Return'], backtest_results['Pred_MSFT']) 

print(f'MSE for AAPL: {mse_aapl}')
print(f'MSE for MSFT: {mse_msft}')

# Calculate correlation
corr_aapl = backtest_results['Return'].corr(backtest_results['Pred_AAPL']) 
corr_msft = backtest_results['Return'].corr(backtest_results['Pred_MSFT']) 

print(f'Correlation for AAPL: {corr_aapl}')
print(f'Correlation for MSFT: {corr_msft}')


# Since MSFT and AAPL are not correlated, we dont expect much from the correlation calculations


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

# AAPL Returns
plt.subplot(1, 2, 1)
plt.plot(aapl_log_returns, label='Actual AAPL Returns', color='blue')
plt.plot(backtest_results['Pred_AAPL'], label='Predicted AAPL Returns', color='orange', alpha=0.7)
plt.title('AAPL Actual vs Predicted Returns')
plt.legend()

# MSFT Returns
plt.subplot(1, 2, 2)
plt.plot(msft_log_returns, label='Actual MSFT Returns', color='green')
plt.plot(backtest_results['Pred_MSFT'], label='Predicted MSFT Returns', color='red', alpha=0.7)
plt.title('MSFT Actual vs Predicted Returns')
plt.legend()

plt.tight_layout()
plt.show()

