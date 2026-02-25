import pandas as pd
from pathlib import Path
from datetime import datetime

class TrendAggregator:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.proc_path = self.base_path / "data" / "processed"
        
        # 📂 NEW: Snapshot-based structure
        self.run_date = datetime.now().strftime("%Y-%m-%d")
        self.analysis_path = self.base_path / "data" / "analysis" / f"snapshot_{self.run_date}"
        self.analysis_path.mkdir(parents=True, exist_ok=True)
        self.growth_path = self.base_path / "data" / "analysis"
        self.current_date = datetime.now().strftime("%Y-%m")

    def generate_momentum_report(self, trends_df):
        """Calculates Month-over-Month growth automatically."""
        pivot_df = trends_df.pivot(index='period', columns='skills', values='market_share').fillna(0)
        
        # Calculate % Change (Velocity)
        growth_df = pivot_df.pct_change().fillna(0) * 100
        
        # Save to analysis folder
        growth_file = self.growth_path / "growth_momentum.csv"
        growth_df.to_csv(growth_file)
        
        # Print top 3 breakout skills to terminal
        if len(growth_df) > 1:
            latest = growth_df.iloc[-1].sort_values(ascending=False).head(3)
            print(f"🔥 BREAKOUT TECH THIS MONTH: {', '.join(latest.index.tolist())}")

    def run(self):
        print(f"🔄 Starting Aggregation Engine...")

        # 1. Dynamically find all ML-Ready files (Hive Structure)
        files = list(self.proc_path.glob(f"year=*/month=*/{self.current_date}_ml_ready.csv"))
        
        if not files:
            print("❌ No data found to aggregate.")
            return

        all_data = []
        for f in files:
            # We still read FROM the Hive format dynamically
            year = f.parts[-3].split('=')[1]
            month = f.parts[-2].split('=')[1]
            
            df = pd.read_csv(f)
            df['period'] = f"{year}-{month}"
            all_data.append(df)

        master_df = pd.concat(all_data, ignore_index=True)

        # 2. Transformation Logic (Explode & Clean)
        master_df['skills'] = master_df['skills'].str.split('|')
        exploded = master_df.explode('skills')
        exploded = exploded[exploded['skills'] != "Unclassified"]

        # 3. Aggregation (Grouping for Trends)
        trends = exploded.groupby(['period', 'skills']).size().reset_index(name='count')
        
        # Calculate Market Share
        total_jobs = master_df.groupby('period').size().reset_index(name='total_posts')
        trends = trends.merge(total_jobs, on='period')
        trends['market_share'] = (trends['count'] / trends['total_posts']) * 100

        self.generate_momentum_report(trends)
        
        # 4. SAVE: One main file for the dashboard, one for the history
        # We save a 'latest' version AND a 'snapshot' version
        trends.to_csv(self.analysis_path / "aggregated_trends.csv", index=False)
        trends.to_csv(self.base_path / "data" / "analysis" / "latest_trends.csv", index=False)
        
        print(f"✅ Aggregation Complete!")
        print(f"📍 Snapshot: {self.analysis_path}/aggregated_trends.csv")
        print(f"📍 Latest Link: data/analysis/latest_trends.csv")

if __name__ == "__main__":
    TrendAggregator().run()