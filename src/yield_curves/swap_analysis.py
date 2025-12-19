"""
Currency swap and bond return analysis.

This module provides functionality for analyzing currency swaps,
calculating bond returns with FX effects, and computing cumulative returns.
"""

import numpy as np
import pandas as pd
from typing import Optional
from .helpers import monthly_bond_return, round_month_to_maturity


class SwapAnalysis:
    """
    Analyze currency swaps and bond returns with FX effects.

    This class manages the calculation of bond returns, currency returns,
    and swap strategies over time.

    Attributes
    ----------
    results : pd.DataFrame
        DataFrame containing all calculations indexed by month-end dates
    """

    def __init__(
        self,
        origination_date: pd.Timestamp | str,
        maturity_date: pd.Timestamp | str,
        num_digits_maturity: int = 4
    ):
        """
        Initialize swap analysis.

        Parameters
        ----------
        origination_date : pd.Timestamp or str
            Start date for the analysis
        maturity_date : pd.Timestamp or str
            Maturity date for the bond
        num_digits_maturity : int, default=4
            Number of decimal places for maturity calculations
        """
        self.orig_date = pd.Timestamp(origination_date)
        self.mat_date = pd.Timestamp(maturity_date)
        self.num_digits = num_digits_maturity

        # Initialize results DataFrame
        columns = [
            'time_to_maturity',
            'long_bond_yield',
            'short_bond_yield',
            'lb_return',
            'sb_return',
            'long_ccy_return',
            'short_ccy_return',
            'lb_return_w_ccy',
            'sb_return_w_ccy',
            'swap_return',
            'lb_cumulative',
            'sb_cumulative',
            'swap_cumulative'
        ]

        # Generate month-end dates
        today = pd.Timestamp.now().normalize()
        final_date = min(self.mat_date, today)
        month_ends = pd.date_range(
            start=self.orig_date,
            end=final_date,
            freq='M'
        )

        self.results = pd.DataFrame(
            index=month_ends,
            columns=columns,
            dtype=float
        )

        # Initialize cumulative return indices to 1
        self.results.loc[self.results.index[0], 'lb_cumulative'] = 1.0
        self.results.loc[self.results.index[0], 'sb_cumulative'] = 1.0
        self.results.loc[self.results.index[0], 'swap_cumulative'] = 1.0

    def set_yields(
        self,
        long_yields: pd.Series,
        short_yields: pd.Series
    ) -> None:
        """
        Set bond yields for long and short positions.

        Parameters
        ----------
        long_yields : pd.Series
            Yields for long bond position (indexed by date and maturity)
        short_yields : pd.Series
            Yields for short bond position (indexed by date and maturity)
        """
        for i, date in enumerate(self.results.index):
            # Calculate time to maturity
            ttm = round_month_to_maturity(
                date, self.mat_date, self.num_digits
            )
            self.results.loc[date, 'time_to_maturity'] = ttm

            # Get yields at the appropriate maturity
            # This assumes yields is a DataFrame with columns as maturities
            maturity_str = str(ttm)
            if maturity_str in long_yields.columns:
                self.results.loc[date, 'long_bond_yield'] = \
                    long_yields.loc[date, maturity_str]
            if maturity_str in short_yields.columns:
                self.results.loc[date, 'short_bond_yield'] = \
                    short_yields.loc[date, maturity_str]

    def set_currency_returns(
        self,
        long_ccy_returns: pd.Series,
        short_ccy_returns: Optional[pd.Series] = None
    ) -> None:
        """
        Set currency returns for FX effects.

        Parameters
        ----------
        long_ccy_returns : pd.Series
            Monthly returns for the long currency (vs base currency)
        short_ccy_returns : pd.Series, optional
            Monthly returns for short currency (if None, assumes USD base = 0)
        """
        self.results['long_ccy_return'] = long_ccy_returns

        if short_ccy_returns is None:
            # Assume short currency is USD (base), so returns are 0
            self.results['short_ccy_return'] = 0.0
        else:
            self.results['short_ccy_return'] = short_ccy_returns

    def calculate_returns(self) -> None:
        """
        Calculate bond returns, returns with FX, and swap returns.

        This method computes:
        - Monthly bond returns for long and short positions
        - Returns including currency effects
        - Swap strategy returns
        - Cumulative returns for all strategies
        """
        long_yields = self.results['long_bond_yield'].values
        short_yields = self.results['short_bond_yield'].values

        for i in range(1, len(self.results)):
            # Calculate bond returns
            try:
                lb_ret = monthly_bond_return(long_yields, i)
                self.results.iloc[i, self.results.columns.get_loc('lb_return')] = lb_ret
            except ValueError:
                pass

            try:
                sb_ret = monthly_bond_return(short_yields, i)
                self.results.iloc[i, self.results.columns.get_loc('sb_return')] = sb_ret
            except ValueError:
                pass

            # Get values for this month
            lb_ret = self.results.iloc[i]['lb_return']
            sb_ret = self.results.iloc[i]['sb_return']
            long_ccy = self.results.iloc[i]['long_ccy_return']
            short_ccy = self.results.iloc[i]['short_ccy_return']

            # Returns with currency effects
            if pd.notna(lb_ret) and pd.notna(long_ccy):
                lb_ret_w_ccy = (1 + lb_ret) * (1 + long_ccy) - 1
                self.results.iloc[i, self.results.columns.get_loc('lb_return_w_ccy')] = lb_ret_w_ccy

            if pd.notna(sb_ret) and pd.notna(short_ccy):
                sb_ret_w_ccy = (1 + sb_ret) * (1 + short_ccy) - 1
                self.results.iloc[i, self.results.columns.get_loc('sb_return_w_ccy')] = sb_ret_w_ccy

            # Swap return: long bond with FX - short bond with FX
            if pd.notna(lb_ret) and pd.notna(sb_ret):
                swap_ret = (
                    (1 + lb_ret) * (1 + long_ccy) -
                    (1 + sb_ret) * (1 + short_ccy)
                )
                self.results.iloc[i, self.results.columns.get_loc('swap_return')] = swap_ret

            # Cumulative returns
            prev_lb_cum = self.results.iloc[i-1]['lb_cumulative']
            prev_sb_cum = self.results.iloc[i-1]['sb_cumulative']
            prev_swap_cum = self.results.iloc[i-1]['swap_cumulative']

            if pd.notna(lb_ret) and pd.notna(prev_lb_cum):
                self.results.iloc[i, self.results.columns.get_loc('lb_cumulative')] = \
                    prev_lb_cum * (1 + lb_ret) * (1 + long_ccy)

            if pd.notna(sb_ret) and pd.notna(prev_sb_cum):
                self.results.iloc[i, self.results.columns.get_loc('sb_cumulative')] = \
                    prev_sb_cum * (1 + sb_ret) * (1 + short_ccy)

            swap_ret_val = self.results.iloc[i]['swap_return']
            if pd.notna(swap_ret_val) and pd.notna(prev_swap_cum):
                self.results.iloc[i, self.results.columns.get_loc('swap_cumulative')] = \
                    prev_swap_cum * (1 + swap_ret_val)

    def get_performance_summary(self) -> pd.DataFrame:
        """
        Calculate annualized performance metrics.

        Returns
        -------
        pd.DataFrame
            Performance summary with annualized returns, volatility, and Sharpe ratios
        """
        # Select return columns
        return_cols = ['lb_return_w_ccy', 'sb_return_w_ccy', 'swap_return', 'long_ccy_return']
        returns = self.results[return_cols].dropna()

        # Calculate metrics
        metrics = pd.DataFrame(index=['Annualized Return', 'Annualized Vol', 'Sharpe Ratio'])

        for col in return_cols:
            clean_returns = returns[col].dropna()
            if len(clean_returns) > 0:
                # Annualized return (geometric)
                cum_return = (1 + clean_returns).prod()
                n_years = len(clean_returns) / 12
                ann_return = cum_return ** (1 / n_years) - 1

                # Annualized volatility
                ann_vol = clean_returns.std() * np.sqrt(12)

                # Sharpe ratio (assuming 0 risk-free rate for simplicity)
                sharpe = ann_return / ann_vol if ann_vol > 0 else 0

                metrics[col] = [ann_return, ann_vol, sharpe]

        # Rename columns for clarity
        metrics.columns = ['Long Bond w/FX', 'Short Bond w/FX', 'Swap', 'FX Only']

        return metrics
