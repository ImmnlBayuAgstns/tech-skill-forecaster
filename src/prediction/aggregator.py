import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AggregatorConfig:
    """Centralized configuration for tuning."""
    chunk_size: int = 50000  # Process files in chunks to save memory
    min_skill_posts: int = 5  # Filter out skills with < 5 mentions
    top_breakout_skills: int = 3  # Number of top skills to report

class TrendAggregator:
    def __init__(self, config: AggregatorConfig = None):
        self.config = config or AggregatorConfig()
        self.base_path = Path(__file__).parent.parent.parent
        self.proc_path = self.base_path / "data" / "processed"
        self.analysis_path = self.base_path / "data" / "analysis"
        
        # Snapshot-based structure
        self.run_date = datetime.now().strftime("%Y-%m-%d")
        self.snapshot_path = self.analysis_path / f"snapshot_{self.run_date}"
        self.snapshot_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📂 Snapshot path: {self.snapshot_path}")

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data validation: handle missing columns, invalid rows."""
        required_cols = {'period', 'role', 'skills', 'id'}
        missing = required_cols - set(df.columns)
        
        if missing:
            logger.warning(f"⚠️  Missing columns: {missing}. Skipping.")
            return df[required_cols & set(df.columns)]
        
        # Remove rows with null skills
        before = len(df)
        df = df.dropna(subset=['skills'])
        logger.debug(f"Dropped {before - len(df)} rows with null skills")
        
        return df

    def _explode_skills(self, df: pd.DataFrame) -> pd.DataFrame:
        """Explode pipe-separated skills, filter junk."""
        # Convert skills to list (handle already-list cases)
        df = df.copy()
        df['skills'] = df['skills'].astype(str).str.split('|')
        exploded = df.explode('skills')
        
        # Remove "Unclassified" and empty strings
        exploded = exploded[
            (exploded['skills'] != "Unclassified") & 
            (exploded['skills'].str.strip() != "")
        ]
        
        # Standardize skill names (lowercase, strip whitespace)
        exploded['skills'] = exploded['skills'].str.strip().str.lower()
        
        logger.debug(f"Exploded to {len(exploded)} skill records")
        return exploded

    def _calculate_global_trends(self, exploded_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate global market share efficiently."""
        # Total posts per period
        total_posts = exploded_df.groupby('period').size().reset_index(name='total_period_posts')
        
        # Skill counts per period
        global_trends = (
            exploded_df
            .groupby(['period', 'skills'], as_index=False)
            .size()
            .rename(columns={'size': 'count'})
        )
        
        # Merge and calculate market share
        global_trends = global_trends.merge(total_posts, on='period', how='left')
        global_trends['market_share'] = (
            global_trends['count'] / global_trends['total_period_posts'] * 100
        ).round(2)
        
        # Filter out rare skills (noise reduction)
        global_trends = global_trends[global_trends['count'] >= self.config.min_skill_posts]
        
        logger.info(f"✅ Calculated {len(global_trends)} global trend records")
        return global_trends

    def _calculate_role_trends(self, exploded_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate role-specific market share (segment analysis)."""
        # Total posts per role per period
        role_totals = (
            exploded_df
            .groupby(['period', 'role'], as_index=False)
            .size()
            .rename(columns={'size': 'total_role_posts'})
        )
        
        # Skill counts per role per period
        role_trends = (
            exploded_df
            .groupby(['period', 'role', 'skills'], as_index=False)
            .size()
            .rename(columns={'size': 'count'})
        )
        
        # Merge and calculate role-specific market share
        role_trends = role_trends.merge(role_totals, on=['period', 'role'], how='left')
        role_trends['role_market_share'] = (
            role_trends['count'] / role_trends['total_role_posts'] * 100
        ).round(2)
        
        # Filter out rare skills
        role_trends = role_trends[role_trends['count'] >= self.config.min_skill_posts]
        
        logger.info(f"✅ Calculated {len(role_trends)} role-based trend records")
        return role_trends

    def _generate_momentum_report(self, trends_df: pd.DataFrame) -> None:
        """Calculate Month-over-Month growth and detect breakout skills."""
        if len(trends_df) < 2:
            logger.warning("⚠️  Not enough periods for growth calculation.")
            return
        
        # Pivot: periods × skills
        pivot_df = trends_df.pivot(
            index='period',
            columns='skills',
            values='market_share'
        ).fillna(0)
        
        # Calculate % change (velocity)
        growth_df = pivot_df.pct_change().fillna(0) * 100
        
        # Save growth report
        growth_file = self.analysis_path / "growth_momentum.csv"
        growth_df.to_csv(growth_file)
        logger.info(f"📊 Growth report saved: {growth_file}")
        
        # Detect and report top breakout skills
        if len(growth_df) > 0:
            latest_growth = growth_df.iloc[-1].sort_values(ascending=False)
            top_breakouts = latest_growth.head(self.config.top_breakout_skills)
            
            breakout_msg = ", ".join(
                [f"{skill} (+{growth:.1f}%)" for skill, growth in top_breakouts.items()]
            )
            logger.info(f"🔥 BREAKOUT TECH THIS MONTH: {breakout_msg}")

    def _load_and_process_files(self) -> pd.DataFrame:
        """Efficiently load CSV files from hive-partitioned directory."""
        files = sorted(self.proc_path.glob("year=*/month=*/ml_ready.csv"))
        
        if not files:
            logger.error("❌ No ml_ready.csv files found.")
            return pd.DataFrame()
        
        logger.info(f"📂 Found {len(files)} files to process")
        
        all_data = []
        for file_idx, file in enumerate(files):
            try:
                # Extract year/month from path
                year = file.parts[-3].split('=')[1]
                month = file.parts[-2].split('=')[1]
                
                # Read CSV with efficient dtypes
                df = pd.read_csv(
                    file,
                    usecols=['id', 'skills', 'role'],  # Only load necessary columns
                    dtype={'id': 'string', 'skills': 'string'},
                    low_memory=False
                )
                
                # Add period metadata
                df['period'] = f"{year}-{month}"
                df['role'] = df.get('role', 'General')  # Default if missing
                
                all_data.append(df)
                logger.debug(f"✅ Loaded {len(df)} records from {file.name}")
                
            except Exception as e:
                logger.error(f"⚠️  Error reading {file}: {e}")
                continue
        
        if not all_data:
            logger.error("❌ No data loaded from any files.")
            return pd.DataFrame()
        
        # Concatenate all files
        master_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"✅ Master dataset: {len(master_df)} rows")
        
        return master_df

    def run(self):
        """Main orchestration function."""
        try:
            logger.info("🔄 Starting Trend Aggregation Engine...")
            
            # 1. Load all processed files
            master_df = self._load_and_process_files()
            if master_df.empty:
                return
            
            # 2. Validate data integrity
            master_df = self._validate_data(master_df)
            
            # 3. Explode and clean skills
            exploded_df = self._explode_skills(master_df)
            
            if exploded_df.empty:
                logger.warning("⚠️  No valid skills after processing.")
                return
            
            # 4. Calculate trends
            global_trends = self._calculate_global_trends(exploded_df)
            role_trends = self._calculate_role_trends(exploded_df)
            
            # 5. Generate reports
            self._generate_momentum_report(global_trends)
            
            # 6. Save outputs
            global_trends.to_csv(self.snapshot_path / "aggregated_trends.csv", index=False)
            role_trends.to_csv(self.snapshot_path / "role_trends.csv", index=False)
            global_trends.to_csv(self.analysis_path / "latest_trends.csv", index=False)
            
            logger.info(f"✅ Aggregation Complete!")
            logger.info(f"📍 Snapshot: {self.snapshot_path}/aggregated_trends.csv")
            logger.info(f"📍 Latest: {self.analysis_path}/latest_trends.csv")
            
        except Exception as e:
            logger.error(f"❌ Critical error: {e}", exc_info=True)
            raise

if __name__ == "__main__":
        # Default config
    TrendAggregator().run()

    # Custom config for larger datasets
    # config = AggregatorConfig(chunk_size=100000, min_skill_posts=10)
    # TrendAggregator(config).run()