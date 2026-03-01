"""
Tech Skill Forecaster
====================
Predicts future market share for tech skills using exponential smoothing.

Algorithm:
- Simple Exponential Smoothing (SES) with configurable smoothing factor (alpha)
- Processes vectorized (per-skill) to efficiently handle multiple skills
- Generates one-step-ahead forecasts for the next period

Output:
- forecasts.csv: Skill predictions with trends and deltas
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import logging

# Configure structured logging for monitoring forecast runs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ForecasterConfig:
    """
    Configuration for the TechForecaster.
    
    Args:
        alpha (float): Smoothing factor for exponential smoothing (0.0 to 1.0).
                      Higher values give more weight to recent data.
                      Default: 0.5 (equal weight to recent and historical data)
        input_file (str): CSV file with skill market share data to forecast
        output_file (str): Output CSV file for forecast results
    """
    alpha: float = 0.5
    input_file: str = "latest_trends.csv"
    output_file: str = "forecasts.csv"

class TechForecaster:
    """
    Forecasts future tech skill market share using exponential smoothing.
    
    Usage:
        forecaster = TechForecaster()
        forecast_df = forecaster.run_forecast()
    """
    
    def __init__(self, config: ForecasterConfig = None):
        """
        Initialize the forecaster with configuration and file paths.
        
        Args:
            config (ForecasterConfig, optional): Custom configuration.
                                                 Defaults to standard config.
        """
        self.config = config or ForecasterConfig()
        self.base_path = Path(__file__).parent.parent.parent
        self.data_path = self.base_path / "data" / "analysis" / self.config.input_file
        self.output_path = self.base_path / "data" / "analysis" / self.config.output_file

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        """
        Validate that input data has required columns.
        
        Required columns:
        - period: Time period (e.g., "2026-01")
        - skills: Skill name (e.g., "Python")
        - market_share: Market share percentage (numeric)
        
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = {"period", "skills", "market_share"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @staticmethod
    def _simple_forecast_vectorized(values: np.ndarray, alpha: float) -> np.ndarray:
        """
        Compute one-step-ahead forecast using Simple Exponential Smoothing (SES).
        
        Formula:
            y_{t+1} = alpha * y_t + (1 - alpha) * y_{t-1}
        
        This formula is applied vectorized across all skills simultaneously.
        
        Args:
            values (np.ndarray): 2D array of shape (n_periods, n_skills)
            alpha (float): Smoothing factor (0.0 to 1.0)
        
        Returns:
            np.ndarray: 1D array of predicted values for each skill
        """
        if values.shape[0] < 2:
            # If only one period, use that as forecast
            return values[-1, :]
        return alpha * values[-1, :] + (1 - alpha) * values[-2, :]

    @staticmethod
    def _next_period(period: str) -> str:
        """
        Calculate the next month period from a given period string.
        
        Args:
            period (str): Period in format "YYYY-MM" (e.g., "2026-02")
        
        Returns:
            str: Next period in "YYYY-MM" format, or "next_period" if parsing fails
        
        Example:
            "2026-02" -> "2026-03"
        """
        period_ts = pd.to_datetime(f"{period}-01", format="%Y-%m-%d", errors="coerce")
        if pd.isna(period_ts):
            return "next_period"
        return (period_ts + pd.offsets.MonthBegin(1)).strftime("%Y-%m")

    def run_forecast(self) -> pd.DataFrame | None:
        """
        Execute the forecasting pipeline.
        
        Process:
        1. Load market share data from CSV
        2. Validate required columns
        3. Pivot data into time-series format (periods × skills)
        4. Apply exponential smoothing to forecast next period
        5. Compute trends and deltas
        6. Save results to CSV and return dataframe
        
        Returns:
            pd.DataFrame: Forecast results with columns:
                - latest_period: Current period
                - forecast_period: Next period
                - skill: Skill name
                - current_share: Current market share (%)
                - predicted_share: Predicted market share (%)
                - delta_share: Change in market share
                - trend: "Increasing", "Decreasing", or "Stable"
            None: If data file not found or is empty
        """
        # Step 1: Verify data file exists
        if not self.data_path.exists():
            logger.error("❌ No data found. Run the aggregator first!")
            return None

        # Step 2: Load and validate data
        df = pd.read_csv(self.data_path)
        self._validate_input(df)

        # Step 3: Pivot to time series format (periods × skills)
        # Using float32 for memory efficiency when handling large datasets
        pivot_df = (
            df.pivot(index='period', columns='skills', values='market_share')
            .fillna(0)
            .astype("float32")
        )

        if pivot_df.empty:
            logger.warning("⚠️  No data available for forecasting.")
            return None

        # Step 4: Apply exponential smoothing forecast (vectorized across all skills)
        values = pivot_df.to_numpy()  # Shape: (n_periods, n_skills)
        preds = self._simple_forecast_vectorized(values, self.config.alpha)

        # Step 5: Compute trends based on movement from current to predicted
        current_vals = values[-1, :]
        trend = np.where(
            preds > current_vals,
            "Increasing",
            np.where(preds < current_vals, "Decreasing", "Stable")
        )
        latest_period = str(pivot_df.index[-1])
        forecast_period = self._next_period(latest_period)

        # Step 6: Assemble results dataframe
        forecast_df = pd.DataFrame({
            "latest_period": latest_period,
            "forecast_period": forecast_period,
            "skill": pivot_df.columns,
            "current_share": np.round(current_vals, 2),
            "predicted_share": np.round(preds, 2),
            "delta_share": np.round(preds - current_vals, 2),
            "trend": trend
        }).sort_values(by="predicted_share", ascending=False, kind="mergesort")

        # Step 7: Save and log results
        forecast_df.to_csv(self.output_path, index=False)
        logger.info("🔮 Forecast generated successfully.")
        logger.info(f"📄 Output saved to: {self.output_path}")
        return forecast_df

if __name__ == "__main__":
    """
    Entry point: Run forecasting from command line.
    Usage: python forecaster.py
    """
    TechForecaster().run_forecast()