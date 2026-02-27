import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ForecasterConfig:
    alpha: float = 0.5
    input_file: str = "latest_trends.csv"
    output_file: str = "forecasts.csv"

class TechForecaster:
    def __init__(self, config: ForecasterConfig = None):
        self.config = config or ForecasterConfig()
        self.base_path = Path(__file__).parent.parent.parent
        self.data_path = self.base_path / "data" / "analysis" / self.config.input_file
        self.output_path = self.base_path / "data" / "analysis" / self.config.output_file

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        required_cols = {"period", "skills", "market_share"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @staticmethod
    def _simple_forecast_vectorized(values: np.ndarray, alpha: float) -> np.ndarray:
        """
        Vectorized SES for each skill column.
        Uses last two time points:
        y_{t+1} = alpha * y_t + (1 - alpha) * y_{t-1}
        """
        if values.shape[0] < 2:
            return values[-1, :]
        return alpha * values[-1, :] + (1 - alpha) * values[-2, :]

    def run_forecast(self) -> None:
        if not self.data_path.exists():
            logger.error("❌ No data found. Run the aggregator first!")
            return

        df = pd.read_csv(self.data_path)
        self._validate_input(df)

        # Pivot to time series per skill (float32 to reduce memory)
        pivot_df = (
            df.pivot(index='period', columns='skills', values='market_share')
            .fillna(0)
            .astype("float32")
        )

        if pivot_df.empty:
            logger.warning("⚠️  No data available for forecasting.")
            return

        # Vectorized forecast
        values = pivot_df.to_numpy()
        preds = self._simple_forecast_vectorized(values, self.config.alpha)

        current_vals = values[-1, :]
        trend = np.where(preds > current_vals, "📈 Increasing", "📉 Decreasing")

        forecast_df = pd.DataFrame({
            "skill": pivot_df.columns,
            "current_share": np.round(current_vals, 2),
            "predicted_share": np.round(preds, 2),
            "trend": trend
        }).sort_values(by="predicted_share", ascending=False, kind="mergesort")

        forecast_df.to_csv(self.output_path, index=False)
        logger.info("🔮 Forecast generated successfully.")
        logger.info(f"📄 Output saved to: {self.output_path}")

if __name__ == "__main__":
    TechForecaster().run_forecast()