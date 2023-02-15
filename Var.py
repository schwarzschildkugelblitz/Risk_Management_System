mport numpy as np
from scipy.stats import norm



class RiskManagementModel:
    """
    A class for calculating Value-at-Risk (VaR) of a portfolio using various risk management methods.

    Parameters:
    -----------
    returns : numpy.ndarray
        A 2D array of historical returns of the assets in the portfolio, where each row represents a period and each column represents an asset.
    weights : numpy.ndarray
        A 1D array of weights of each asset in the portfolio.

    Attributes:
    -----------
    returns : numpy.ndarray
        A 2D array of historical returns of the assets in the portfolio, where each row represents a period and each column represents an asset.
    weights : numpy.ndarray
        A 1D array of weights of each asset in the portfolio.
    """
    
    def __init__(self, returns, weights):
        self.returns = returns
        self.weights = weights
    
    def get_log_return(self):
        """
        Calculates the log returns of the portfolio.

        Returns:
        --------
        numpy.ndarray
            A 2D array of log returns of the assets in the portfolio, where each row represents a period and each column represents an asset.
        """
        log_return = np.log(1 + self.returns)
        return log_return
    
    def variance_covariance_var(self, confidence_level, horizon):
        """
        Calculates the Variance-Covariance VaR of the portfolio based on the confidence level and horizon.

        Parameters:
        -----------
        confidence_level : float
            The confidence level of the VaR calculation, e.g. 0.95.
        horizon : int
            The horizon of the VaR calculation in number of periods, e.g. 1 day = 1.

        Returns:
        --------
        float
            The VaR of the portfolio based on the Variance-Covariance method.
        """
        log_return = self.get_log_return()
        mean_log_return = np.mean(log_return, axis=0)
        cov_log_return = np.cov(log_return, rowvar=False)
        portfolio_volatility = np.sqrt(np.dot(self.weights.T, np.dot(cov_log_return, self.weights)))
        z_score = np.abs(norm.ppf(1 - confidence_level / 2))
        var_var_covar = - portfolio_volatility * z_score * np.sqrt(horizon)
        return var_var_covar
    
    def historical_simulation_var(self, confidence_level, horizon):
        """
        Calculates the Historical simulation VaR of the portfolio based on the confidence level and horizon.

        Parameters:
        -----------
        confidence_level : float
            The confidence level of the VaR calculation, e.g. 0.95.
        horizon : int
            The horizon of the VaR calculation in number of periods, e.g. 1 day = 1.

        Returns:
        --------
        float
            The VaR of the portfolio based on the Historical simulation method.
        """
        log_return = self.get_log_return()
        sorted_returns = np.sort(log_return)
        index = int(np.ceil(len(sorted_returns) * (1 - confidence_level)))
        var_returns = sorted_returns[index:]
        mean_var_returns = np.mean(var_returns)
        std_var_returns = np.std(var_returns)
        var_hist_sim_var = - mean_var_returns - std_var_returns * np.sqrt(horizon)
        return var_hist_sim_var
    
    def monte_carlo_var(self, confidence_level, horizon, num_simulations=10000):
        """
        Calculates the Monte Carlo VaR of the portfolio based on the confidence level, horizon, and number of simulations.

        Parameters:
        -----------
        confidence_level : float
            The confidence level of the VaR calculation, e.g. 0.95.
        horizon : int
            The horizon of the VaR calculation in number of periods, e.g. 1 day = 1.
        num_simulations : int, optional
            The number of simulations to run for the Monte Carlo method, by default 10000.

        Returns:
        --------
        float
            The VaR of the portfolio based on the Monte Carlo method.
        """
        log_return = self.get_log_return()
        mean_log_return = np.mean(log_return, axis=0)
        cov_log_return = np.cov(log_return, rowvar=False)
        portfolio_volatility = np.sqrt(np.dot(self.weights.T, np.dot(cov_log_return, self.weights)))
        dt = 1 / len(self.returns)
        portfolio_value = np.dot(self.weights, np.cumprod(np.exp(log_return), axis=0)[-1])
        portfolio_values = []
        for i in range(num_simulations):
            log_returns_sim = np.random.multivariate_normal(mean_log_return, cov_log_return, len(self.returns))
            portfolio_value_sim = portfolio_value * np.exp(np.cumsum(log_returns_sim, axis=0)[-1])
            portfolio_values.append(portfolio_value_sim)
        portfolio_values = np.array(portfolio_values)
        portfolio_returns_sim = (portfolio_values - portfolio_value) / portfolio_value
        sorted_returns = np.sort(portfolio_returns_sim)
        index = int(np.ceil(num_simulations * (1 - confidence_level)))
        var_returns = sorted_returns[index:]
        var_monte_carlo_var = - np.mean(var_returns) - np.std(var_returns) * np.sqrt(horizon)
        return var_monte_carlo_var

    def delta_var(self, confidence_level, horizon, delta, value, market_value):
        """
        Calculates the Delta-normal approximation VaR of the portfolio based on the confidence level, horizon, and change in portfolio value.

        Parameters:
        -----------
        confidence_level : float
            The confidence level of the VaR calculation, e.g. 0.95.
        horizon : int
            The horizon of the VaR calculation in number of periods, e.g. 1 day = 1.
        delta : float
            The change in portfolio value, e.g. 10000.
        value : float
            The current value of the portfolio, e.g. 1000000.
        market_value : numpy.ndarray
            A 1D array of market values of each asset in the portfolio.

        Returns:
        --------
        float
            The VaR of the portfolio based on the Delta-normal approximation method.
        """
        log_return = self.get_log_return()
        mean_log_return = np.mean(log_return, axis=0)
        cov_log_return = np.cov(log_return, rowvar=False)
        portfolio_volatility = np.sqrt(np.dot(self.weights.T, np.dot(cov_log_return, self.weights)))
        market_values = market_value * self.weights
        delta_market_value = np.dot(market_values, mean_log_return) * delta
        delta_portfolio_value = value * delta_market_value
        delta_var_var = - delta_portfolio_value - portfolio_volatility * np.sqrt(horizon) * norm.ppf(1 - confidence_level)
        return delta_var_var