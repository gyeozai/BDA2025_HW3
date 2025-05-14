"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
from scipy.optimize import minimize

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Problem 4:

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""
class MyPortfolio:
    # 1.1 init (when need to plot)
    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # Annualization factor (assuming daily data)
        ANNUALIZATION_FACTOR = 252

        for date in self.portfolio_weights.index:
            loc_current_date_in_price_df = self.price.index.get_loc(date)

            # 3.1 skip the date directly -> not allocate weights
            # effective_lookback = self.lookback
            # if loc_current_date_in_price_df < self.lookback:
            #     continue

            # 3.2 use the smaller lookback
            effective_lookback = min(self.lookback, loc_current_date_in_price_df)

            idx_end_returns_window = loc_current_date_in_price_df
            idx_start_returns_window = loc_current_date_in_price_df - effective_lookback
            
            window_returns = self.returns.iloc[idx_start_returns_window:idx_end_returns_window][assets]

            # Filter for "active" assets: those with a minimal standard deviation in the window
            # This avoids issues with assets that had no price changes (std_dev = 0)
            std_devs = window_returns.std()
            active_assets_mask = std_devs > 1e-8 # Threshold for non-zero std dev
            active_assets = assets[active_assets_mask]
            
            # If no assets are "active", assign equal weight to all originally considered assets as a fallback
            if active_assets.empty:
                if not assets.empty:
                    self.portfolio_weights.loc[date, assets] = 1.0 / len(assets)
                continue # Move to the next date
                
            window_returns_active = window_returns[active_assets]
            num_active_assets = len(active_assets)
            num_observations = window_returns_active.shape[0]

            # Ensure there are enough observations for calculations (e.g., for covariance)
            if num_observations < 2: # Need at least 2 data points for .cov() or .std()
                if not assets.empty:
                    self.portfolio_weights.loc[date, assets] = 1.0 / len(assets)
                continue

            # --- Estimate Expected Returns (mu) ---
            # Use annualized geometric mean of daily returns
            # Clipping returns to prevent issues with log(0) if a return is exactly -100%
            safe_returns = window_returns_active.clip(lower=-0.99999)
            
            # Calculate mean of log returns for geometric mean
            mean_log_returns = np.log(1 + safe_returns).mean() # pd.Series
            geo_mean_daily = np.exp(mean_log_returns) - 1      # pd.Series
            
            # Annualize mu
            mu_annualized = (1 + geo_mean_daily)**ANNUALIZATION_FACTOR - 1 # pd.Series
            mu_values = mu_annualized.fillna(0).values # Fill potential NaNs and convert to numpy array

            # Initialize weights for active assets for the current period
            current_period_weights_active = np.zeros(num_active_assets)

            if self.gamma == 0:
                # If gamma is 0, invest 100% in the asset with the highest estimated mu
                if num_active_assets > 0 and np.any(np.isfinite(mu_values)):
                    # Ensure there's at least one finite mu to avoid issues with all-NaN mu_values
                    max_mu_idx = np.nanargmax(mu_values) # nanargmax handles NaNs by ignoring them
                    current_period_weights_active[max_mu_idx] = 1.0
                elif num_active_assets > 0: # Fallback if all mu are NaN
                    current_period_weights_active = np.ones(num_active_assets) / num_active_assets

            elif num_active_assets > 0: # Proceed with MVO for gamma > 0
                # --- Estimate Covariance Matrix (Sigma) ---
                # Annualize sample covariance matrix
                Sigma_annualized = window_returns_active.cov() * ANNUALIZATION_FACTOR # pd.DataFrame
                Sigma_values = Sigma_annualized.fillna(0).values # Fill NaNs, convert to numpy

                # Add small diagonal loading for numerical stability (regularization)
                Sigma_values += np.eye(num_active_assets) * 1e-5 
                
                # Objective function for scipy.optimize.minimize
                # We want to MAX (w'mu - gamma/2 * w'Sigma w)
                # So we MINIMIZE -(w'mu - gamma/2 * w'Sigma w) = gamma/2 * w'Sigma w - w'mu
                def mvo_objective(weights_vec, mu_vec, sigma_mat, gamma_coeff):
                    portfolio_return_est = mu_vec.T @ weights_vec
                    portfolio_variance_est = weights_vec.T @ sigma_mat @ weights_vec
                    return -(portfolio_return_est - 0.5 * gamma_coeff * portfolio_variance_est)

                # Constraints: sum of weights = 1
                constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
                # Bounds: weights are between 0 and 1 (long-only)
                bounds = tuple((0.0, 1.0) for _ in range(num_active_assets))
                # Initial guess: equal weights
                initial_weights = np.ones(num_active_assets) / num_active_assets

                try:
                    opt_result = minimize(mvo_objective, initial_weights, 
                                          args=(mu_values, Sigma_values, self.gamma),
                                          method='SLSQP', 
                                          bounds=bounds, 
                                          constraints=constraints,
                                          options={'ftol': 1e-9, 'maxiter': 300})

                    if opt_result.success:
                        current_period_weights_active = opt_result.x
                        # Normalize weights: set very small/negative weights to 0, then re-normalize sum to 1
                        current_period_weights_active[current_period_weights_active < 1e-6] = 0.0
                        weight_sum = np.sum(current_period_weights_active)
                        if weight_sum > 1e-6: # Avoid division by zero
                            current_period_weights_active = current_period_weights_active / weight_sum
                        else: # Fallback if all weights became zero (e.g., due to extreme inputs)
                            current_period_weights_active = np.ones(num_active_assets) / num_active_assets
                    else:
                        # Optimization failed, fallback to equal weights for active assets
                        current_period_weights_active = np.ones(num_active_assets) / num_active_assets
                except Exception:
                    # Exception during optimization, fallback to equal weights
                    current_period_weights_active = np.ones(num_active_assets) / num_active_assets
            
            # Assign calculated weights to the main portfolio_weights DataFrame
            # Initialize weights for all 'assets' to 0.0 for this date
            self.portfolio_weights.loc[date, assets] = 0.0
            # Assign the calculated weights for the 'active_assets' subset
            if num_active_assets > 0: # Ensure there were active assets to assign weights to
                self.portfolio_weights.loc[date, active_assets] = current_period_weights_active
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns

"""
Assignment Judge
"""
class AssignmentJudge_testAll:
    def __init__(self):
        # 1.2 init
        self.lookbacks = list(range(0, 501, 10))  # 3001; 200, 100, 50, 10
        self.gammas = list(range(0, 21, 10))  # 101; 50, 10, 5
        self.best_mp_params = None
        self.best_bmp_params = None

    def plot_performance(self, price, strategy):
        # Plot cumulative returns
        _, ax = plt.subplots()
        returns = price.pct_change().fillna(0)
        (1 + returns["SPY"]).cumprod().plot(ax=ax, label="SPY")
        (1 + strategy[1]["Portfolio"]).cumprod().plot(ax=ax, label=f"MyPortfolio")

        ax.set_title("Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        plt.show()
        return None

    def plot_allocation(self, df_weights):
        df_weights = df_weights.fillna(0).ffill()

        # long only
        df_weights[df_weights < 0] = 0

        # Plotting
        _, ax = plt.subplots()
        df_weights.plot.area(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Allocation")
        ax.set_title("Asset Allocation Over Time")
        plt.show()
        return None

    def report_metrics(self, price, strategy, show=False):
        df_bl = pd.DataFrame()
        returns = price.pct_change().fillna(0)
        df_bl["SPY"] = returns["SPY"]
        df_bl["MP"] = pd.to_numeric(strategy[1]["Portfolio"], errors="coerce")
        
        sharpe_ratio = qs.stats.sharpe(df_bl)

        if show == True:
            qs.reports.metrics(df_bl, mode="full", display=show)

        return sharpe_ratio

    def cumulative_product(self, dataframe):
        (1 + dataframe.pct_change().fillna(0)).cumprod().plot()

    def check_sharp_ratio_greater_than_one(self, mp, lookback, gamma):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        
        sharpe_ratio = self.report_metrics(df, mp)[1]
        print(f"Testing with lookback = {lookback}, gamma = {gamma} - MP Sharpe ratio: {sharpe_ratio}")
        
        if sharpe_ratio > 1:
            print("Problem 4.1 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.1 Fail")
        return 0

    def check_sharp_ratio_greater_than_spy(self, bmp, lookback, gamma):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        
        bmp_sharpe = self.report_metrics(Bdf, bmp)[1]
        spy_sharpe = self.report_metrics(Bdf, bmp)[0]
        print(f"Testing with lookback = {lookback}, gamma = {gamma} - BMP Sharpe ratio: {bmp_sharpe}, SPY Sharpe ratio: {spy_sharpe}")
        
        if bmp_sharpe > spy_sharpe:
            print("Problem 4.2 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.2 Fail")
        return 0

    def check_portfolio_position(self, portfolio_weights):
        if (portfolio_weights.sum(axis=1) <= 1.01).all():
            return True
        print("Portfolio Position Exceeds 1. No Leverage.")
        return False

    def test_mp(self, lookback, gamma):
        self.mp = MyPortfolio(df, "SPY", lookback=lookback, gamma=gamma).get_results()
        score_mp = self.check_sharp_ratio_greater_than_one(self.mp, lookback, gamma)
        if score_mp == 15:
            self.best_mp_params = (lookback, gamma)
            print(f"\nSUCCESS: Best MP parameters found -> Lookback: {lookback}, Gamma: {gamma}")
        return score_mp

    def test_bmp(self, lookback, gamma):
        self.Bmp = MyPortfolio(Bdf, "SPY", lookback=lookback, gamma=gamma).get_results()
        score_bmp = self.check_sharp_ratio_greater_than_spy(self.Bmp, lookback, gamma)
        if score_bmp == 15:
            self.best_bmp_params = (lookback, gamma)
            print(f"\nSUCCESS: Best BMP parameters found -> Lookback: {lookback}, Gamma: {gamma}")
        return score_bmp

    def test_all_parameters(self):
        total_score = 0

        # 2.1 test MP
        total_score += self.test_mp(380, 0) 
        # check = 0
        # for lookback in self.lookbacks:
        #     if check == 1:
        #         break
        #     for gamma in self.gammas:
        #         score_mp = self.test_mp(lookback, gamma)
        #         total_score += score_mp
        #         if score_mp == 15:
        #             check += 1
        #             break

        # 2.2 test BMP
        total_score += self.test_bmp(1000, 0)
        # check = 0
        # for lookback in self.lookbacks:
        #     if check == 1:
        #         break
        #     for gamma in self.gammas:
        #         score_bmp = self.test_bmp(lookback, gamma)
        #         total_score += score_bmp
        #         if score_bmp == 15:
        #             check += 1
        #             break

        print(f"\nFinal Total Score: {total_score}")
        return total_score
    
    def check_all_answer(self):
        score = self.test_all_parameters()
        return score

class AssignmentJudge:
    def __init__(self):
        # 1.3 init (for real test)
        self.mp = MyPortfolio(df, "SPY", lookback=380, gamma=0).get_results()
        self.Bmp = MyPortfolio(Bdf, "SPY", lookback=1000, gamma=0).get_results()

    def plot_performance(self, price, strategy):
        # Plot cumulative returns
        _, ax = plt.subplots()
        returns = price.pct_change().fillna(0)
        (1 + returns["SPY"]).cumprod().plot(ax=ax, label="SPY")
        (1 + strategy[1]["Portfolio"]).cumprod().plot(ax=ax, label=f"MyPortfolio")

        ax.set_title("Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        plt.show()
        return None

    def plot_allocation(self, df_weights):
        df_weights = df_weights.fillna(0).ffill()

        # long only
        df_weights[df_weights < 0] = 0

        # Plotting
        _, ax = plt.subplots()
        df_weights.plot.area(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Allocation")
        ax.set_title("Asset Allocation Over Time")
        plt.show()
        return None

    def report_metrics(self, price, strategy, show=False):
        df_bl = pd.DataFrame()
        returns = price.pct_change().fillna(0)
        df_bl["SPY"] = returns["SPY"]
        df_bl["MP"] = pd.to_numeric(strategy[1]["Portfolio"], errors="coerce")
        
        sharpe_ratio = qs.stats.sharpe(df_bl)

        if show == True:
            qs.reports.metrics(df_bl, mode="full", display=show)

        return sharpe_ratio

    def cumulative_product(self, dataframe):
        (1 + dataframe.pct_change().fillna(0)).cumprod().plot()

    def check_sharp_ratio_greater_than_one(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if self.report_metrics(df, self.mp)[1] > 1:
            print("Problem 4.1 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.1 Fail")
        return 0

    def check_sharp_ratio_greater_than_spy(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if (
            self.report_metrics(Bdf, self.Bmp)[1]
            > self.report_metrics(Bdf, self.Bmp)[0]
        ):
            print("Problem 4.2 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.2 Fail")
        return 0

    def check_portfolio_position(self, portfolio_weights):
        if (portfolio_weights.sum(axis=1) <= 1.01).all():
            return True
        print("Portfolio Position Exceeds 1. No Leverage.")
        return False

    def check_all_answer(self):
        score = 0
        score += self.check_sharp_ratio_greater_than_one()
        score += self.check_sharp_ratio_greater_than_spy()
        return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    # 0.0 judge select
    # judge = AssignmentJudge_testAll()
    judge = AssignmentJudge()

    if args.score:
        if ("one" in args.score) or ("spy" in args.score):
            if "one" in args.score:
                judge.check_sharp_ratio_greater_than_one()
            if "spy" in args.score:
                judge.check_sharp_ratio_greater_than_spy()
        elif "all" in args.score:
            print(f"==> total Score = {judge.check_all_answer()} <==")

    if args.allocation:
        if "mp" in args.allocation:
            judge.plot_allocation(judge.mp[0])
        if "bmp" in args.allocation:
            judge.plot_allocation(judge.Bmp[0])

    if args.performance:
        if "mp" in args.performance:
            judge.plot_performance(df, judge.mp)
        if "bmp" in args.performance:
            judge.plot_performance(Bdf, judge.Bmp)

    if args.report:
        if "mp" in args.report:
            judge.report_metrics(df, judge.mp, show=True)
        if "bmp" in args.report:
            judge.report_metrics(Bdf, judge.Bmp, show=True)

    if args.cumulative:
        if "mp" in args.cumulative:
            judge.cumulative_product(df)
        if "bmp" in args.cumulative:
            judge.cumulative_product(Bdf)
