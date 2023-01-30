import pandas as pd
import numpy as np
import yfinance as yf 
from itertools import combinations
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


class analysis : 


    def engle_granger_2step(x, y):
        result = coint(x, y)
        pvalue = result[1]
        if pvalue < 0.01:
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            residuals = model.resid
            return residuals
        else:
            return None

    def adf_test(residuals):
        result = adfuller(residuals)
        pvalue = result[1]
        if pvalue < 0.01:
            return True
        else:
            return False
    def z_score(residuals, window):
        '''
        Calculates the Z-score of a time series.

        Parameters:
        residuals (pd.Series): Residuals obtained from the regression model.
        window (int): The window size for the rolling mean and standard deviation calculation.
        
        Returns:
        z_score (pd.Series): The Z-score of the residuals.
        '''
        mean = residuals.rolling(window=window).mean()
        std = residuals.rolling(window=window).std()

        z_score = (residuals - mean) / std
        return z_score

    def trade_strategy(residuals, window):
        '''
        Defines the trading strategy based on Z-scores.

        Parameters:
        residuals (pd.Series): Residuals obtained from the regression model.
        window (int): The window size for the rolling mean and standard deviation calculation.
        
        Returns:
        trades (np.ndarray): An array indicating the trade signal for each day (-1 for short y and long x, 1 for short x and long y, 0 for no trade).
        '''
        z_scores = z_score(residuals, window)
        trades = np.zeros(len(residuals))
        trades[z_scores > 2.0] = 1
        trades[z_scores < -2.0] = -1
        trades[(z_scores >= -1.0) & (z_scores <= 1.0)] = 0
        return trades

    def equity_curve(data, trades, initial_capital):
        '''
        Calculates the equity curve for the trading strategy.

        Parameters:
        data (pd.Series): Close prices for the stock being traded.
        trades (np.ndarray): An array indicating the trade signal for each day (-1 for short y and long x, 1 for short x and long y, 0 for no trade).
        initial_capital (float): The starting capital for the trading strategy.
        
        Returns:
        equity (pd.Series): The equity curve for the trading strategy.
        '''
        equity = initial_capital + (trades * data).cumsum()
        return equity

    def calculate_performance_metrics(equity, trades, data):
        '''
        Calculates various performance metrics for the trading strategy.
        
        Parameters:
        equity (pd.Series): The equity curve for the trading strategy.
        trades (np.ndarray): An array indicating the trade signal for each day (-1 for short y and long x, 1 for short x and long y, 0 for no trade).
        data (pd.Series): Close prices for the stock being traded.
        
        Returns:
        profits (float): Total profit from the trades.
        total_trades (int): Total number of trades.
        win_rate (float): The win rate of the trades.
        profit_factor (float): The profit factor of the trades.
        sharpe_ratio (float): The Sharpe ratio of the equity curve.
        max_drawdown (float): The maximum drawdown of the equity curve.
        '''
        profits = (trades * (data.shift(-1) - data)).sum()
        total_trades = len(trades[trades != 0])
        win_rate = len(trades[trades == 1]) / total_trades if total_trades != 0 else 0
        profit_factor = profits / len(trades[trades == -1]) if len(trades[trades == -1]) != 0 else 0
        sharpe_ratio = equity.pct_change().mean() / equity.pct_change().std() if equity.pct_change().std() != 0 else 0
        max_drawdown = (equity.cummax() - equity) / equity.cummax()
        return profits, total_trades, win_rate, profit_factor, sharpe_ratio, max_drawdown.max()