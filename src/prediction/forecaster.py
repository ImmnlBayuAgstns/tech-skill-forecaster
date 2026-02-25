import pandas as pd
import numpy as np
from pathlib import Path

class TechForecaster:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.data_path = self.base_path / "data" / "analysis" / "latest_trends.csv"
        self.output_path = self.base_path / "data" / "analysis" / "forecasts.csv"

    def simple_forecast(self, series, alpha=0.5):
        """
        Implements Simple Exponential Smoothing (SES).
        Formula: $y_{t+1} = \alpha y_t + (1 - \alpha) y_{t-1}$
        """
        if len(series) < 2:
            return series.iloc[-1] # Naive forecast if only 1 month exists
        
        # SES Calculation
        forecast = alpha * series.iloc[-1] + (1 - alpha) * series.iloc[-2]
        return forecast

    def run_forecast(self):
        if not self.data_path.exists():
            print("❌ No data found. Run the aggregator first!")
            return

        df = pd.read_csv(self.data_path)
        
        # 1. Pivot data so we have a time series per skill
        pivot_df = df.pivot(index='period', columns='skills', values='market_share').fillna(0)
        
        predictions = []
        for skill in pivot_df.columns:
            series = pivot_df[skill]
            
            # 2. Generate March 2026 Prediction
            march_pred = self.simple_forecast(series)
            
            # 3. Calculate "Momentum Direction"
            current_val = series.iloc[-1]
            direction = "📈 Increasing" if march_pred > current_val else "📉 Decreasing"
            
            predictions.append({
                "skill": skill,
                "current_share": round(current_val, 2),
                "predicted_march_share": round(march_pred, 2),
                "trend": direction
            })

        # 4. Save and Report
        forecast_df = pd.DataFrame(predictions).sort_values(by="predicted_march_share", ascending=False)
        forecast_df.to_csv(self.output_path, index=False)
        
        print(f"🔮 Forecast for March 2026 generated!")
        print(forecast_df.head(5)) # Show top 5 predicted winners

if __name__ == "__main__":
    TechForecaster().run_forecast()