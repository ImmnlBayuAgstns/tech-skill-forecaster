"""
Tech Skill Trends Forecaster Dashboard
======================================
Interactive Streamlit dashboard for analyzing tech hiring trends and forecasting future skill demand.

Features:
- Real-time skill popularity metrics
- Month-over-month growth momentum visualization
- Future skill forecasts using exponential smoothing
- Interactive filters and customizable charts
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.append(str(ROOT_PATH))

from src.prediction.forecaster import TechForecaster

# Page Configuration: Set wide layout for better visualization
st.set_page_config(page_title="Tech Skill Trends", layout="wide")

class TrendDashboard:
    """
    Main dashboard class for visualizing tech skill trends and forecasts.
    
    Attributes:
        analysis_path (Path): Path to the analysis data directory containing CSV files
    """
    def __init__(self):
        self.analysis_path = Path(__file__).parent.parent / "data" / "analysis"

    @st.cache_data(show_spinner=False)
    def load_data(_self):
        """
        Load and preprocess all required data files for the dashboard.
        
        Loads:
        - latest_trends.csv: Current market share data per skill and period
        - growth_momentum.csv: Month-over-month growth rates
        - forecasts.csv: Predicted future skill market share
        
        Returns:
            tuple: (global_df, momentum_df, forecast_df)
                - global_df: Global skill trends across all periods
                - momentum_df: Growth rates by skill and period
                - forecast_df: Predicted future skill shares
        
        """
        # Load core datasets
        global_df = pd.read_csv(_self.analysis_path / "latest_trends.csv")
        momentum_df = pd.read_csv(_self.analysis_path / "growth_momentum.csv")
        forecast_path = _self.analysis_path / "forecasts.csv"

        # Auto-generate forecast if not present
        if forecast_path.exists():
            forecast_df = pd.read_csv(forecast_path)
        else:
            generated_forecast = TechForecaster().run_forecast()
            forecast_df = generated_forecast.copy() if generated_forecast is not None else pd.DataFrame()

        # Normalize period columns to string type across all dataframes
        for frame in (global_df, momentum_df, forecast_df):
            if "period" in frame.columns:
                frame["period"] = frame["period"].astype(str)

        # Clean skill names (trim whitespace)
        if "skills" in global_df.columns:
            global_df["skills"] = global_df["skills"].astype(str).str.strip()

        # Process forecast dataframe if present
        if not forecast_df.empty:
            if "skill" in forecast_df.columns:
                forecast_df["skill"] = forecast_df["skill"].astype(str).str.strip()

            if "forecast_period" not in forecast_df.columns:
                forecast_df["forecast_period"] = "Next Period"

            # Calculate delta (change) if predicted_share and current_share exist
            if "delta_share" not in forecast_df.columns and {"predicted_share", "current_share"}.issubset(forecast_df.columns):
                forecast_df["delta_share"] = forecast_df["predicted_share"] - forecast_df["current_share"]

            # Ensure numeric columns are properly typed
            for numeric_col in ["predicted_share", "current_share", "delta_share"]:
                if numeric_col in forecast_df.columns:
                    forecast_df[numeric_col] = pd.to_numeric(forecast_df[numeric_col], errors="coerce")

            # Generate trend labels if not present
            if "trend" not in forecast_df.columns and "delta_share" in forecast_df.columns:
                forecast_df["trend"] = forecast_df["delta_share"].apply(
                    lambda value: "Increasing" if value > 0 else ("Decreasing" if value < 0 else "Stable")
                )

        return global_df, momentum_df, forecast_df

    def render(self):
        """
        Render the complete dashboard with all visualizations and controls.
        
        Dashboard sections:
        1. Top metrics: Current top skill, top grower, skills tracked
        2. Future forecast: Predictions with trend analysis
        3. Global trends: Market share evolution over time
        4. Growth momentum: Month-over-month skill growth rates
        """
        st.title("🚀 Hiring Trends & Skill Momentum")
        global_df, momentum_df, forecast_df = self.load_data()

        # === SIDEBAR FILTERS ===
        # Allow users to select which time periods to analyze
        st.sidebar.header("Filters")
        available_periods = sorted(global_df["period"].unique())
        selected_periods = st.sidebar.multiselect(
            "Periods",
            options=available_periods,
            default=available_periods,
        )

        if not selected_periods:
            st.warning("Select at least one period to display charts.")
            return

        # Filter global data to selected periods
        filtered_global = global_df[global_df["period"].isin(selected_periods)].copy()
        latest_period = sorted(filtered_global["period"].unique())[-1]
        latest_global = filtered_global[filtered_global["period"] == latest_period].copy()

        # === SECTION 1: TOP LEVEL METRICS ===
        # Display 3 key performance indicators
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Metric 1: Most popular skill by count
            top_skill = latest_global.sort_values("count", ascending=False).iloc[0]
            st.metric("Top Skill", top_skill["skills"], f"{top_skill['market_share']:.1f}%")
        
        with col2:
            # Metric 2: Skill with highest growth rate
            momentum_latest = momentum_df[momentum_df["period"] == latest_period]
            if momentum_latest.empty:
                st.metric("Top Grower", "N/A", "No momentum data")
            else:
                breakout = (
                    momentum_latest
                    .melt(id_vars=["period"], var_name="skills", value_name="growth")
                    .replace([float("inf"), float("-inf")], pd.NA)
                    .dropna(subset=["growth"])
                    .sort_values("growth", ascending=False)
                    .iloc[0]
                )
                st.metric("Top Grower", breakout["skills"], f"+{breakout['growth']:.1f}%")

        with col3:
            # Metric 3: Total unique skills tracked
            skill_count = latest_global["skills"].nunique()
            st.metric("Skills Tracked", int(skill_count), latest_period)

        st.divider()
        
        # === SECTION 2: FUTURE SKILL FORECAST ===
        # Shows predicted market share for future periods using exponential smoothing
        col_forecast_title, col_forecast_btn = st.columns([0.8, 0.2])
        with col_forecast_title:
            st.subheader("Future Skill Forecast")
        with col_forecast_btn:
            # Button to regenerate forecasts without restarting Streamlit
            if st.button("🔄 Regenerate", key="regen_forecast"):
                st.cache_data.clear()
                st.rerun()

        if forecast_df.empty or "skill" not in forecast_df.columns:
            st.info("Forecast data is not available yet.")
        else:
            # Display forecast period information
            forecast_period = (
                forecast_df["forecast_period"].iloc[0]
                if "forecast_period" in forecast_df.columns and not forecast_df["forecast_period"].isna().all()
                else "Next Period"
            )
            st.caption(f"Forecast period: {forecast_period}")

            # Forecast metrics: top predicted skill and highest expected gainer
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                top_predicted = forecast_df.sort_values("predicted_share", ascending=False).iloc[0]
                st.metric("Top Predicted Skill", top_predicted["skill"], f"{top_predicted['predicted_share']:.2f}%")
            with f_col2:
                top_gainer = forecast_df.sort_values("delta_share", ascending=False).iloc[0]
                st.metric("Highest Expected Gain", top_gainer["skill"], f"{top_gainer['delta_share']:+.2f}%")

            # Interactive controls for forecast visualization
            top_n = st.slider(
                "Top forecasted skills",
                min_value=5,
                max_value=min(30, len(forecast_df)),
                value=min(12, len(forecast_df)),
                help="Number of top skills to display in the chart"
            )
            trend_filter = st.multiselect(
                "Forecast trend",
                options=sorted(forecast_df["trend"].dropna().unique()),
                default=sorted(forecast_df["trend"].dropna().unique()),
                help="Filter skills by trend type (Increasing/Decreasing/Stable)"
            )

            # Apply filters to forecast data
            filtered_forecast = forecast_df.copy()
            if trend_filter:
                filtered_forecast = filtered_forecast[filtered_forecast["trend"].isin(trend_filter)]

            filtered_forecast = filtered_forecast.sort_values("predicted_share", ascending=False).head(top_n)

            if filtered_forecast.empty:
                st.info("No forecast rows match your selected trend filter.")
            else:
                # Horizontal bar chart showing predicted market share with color-coded trends
                fig_forecast = px.bar(
                    filtered_forecast.sort_values("predicted_share", ascending=True),
                    x="predicted_share",
                    y="skill",
                    orientation="h",
                    color="trend",
                    title="Predicted Market Share by Skill",
                    text="predicted_share",
                )
                fig_forecast.update_traces(texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False)
                fig_forecast.update_layout(xaxis_title="Predicted Market Share (%)", yaxis_title="Skill")
                st.plotly_chart(fig_forecast, use_container_width=True)

        # === SECTION 3: GLOBAL SKILL TRENDS ===
        # Time series visualization of market share across all periods
        st.subheader("Global Skill Popularity Over Time")
        
        # Build options with case-insensitive defaults
        options_skills = sorted(filtered_global["skills"].unique())
        options_lower_map = {skill.lower(): skill for skill in options_skills}
        preferred_skills = ["python", "typescript", "aws", "llm"]
        default_skills = [
            options_lower_map[skill] for skill in preferred_skills if skill in options_lower_map
        ]

        # Multi-select for skill filtering
        selected_skills = st.multiselect(
            "Filter Skills",
            options=options_skills,
            default=default_skills if default_skills else options_skills[:4],
            help="Select one or more skills to display in the trend chart"
        )

        selected_skills_lower = {skill.lower() for skill in selected_skills}
        chart_global = filtered_global[
            filtered_global["skills"].str.lower().isin(selected_skills_lower)
        ]

        # Line chart showing market share evolution
        fig_global = px.line(
            chart_global,
            x="period",
            y="market_share",
            color="skills",
            markers=True,
            title="Market Share % per Month",
        )
        fig_global.update_layout(xaxis_title="Period", yaxis_title="Market Share (%)")
        st.plotly_chart(fig_global, use_container_width=True)

        # === SECTION 4: GROWTH MOMENTUM ===
        # Month-over-month growth rates for selected skills
        momentum_long = (
            momentum_df[momentum_df["period"].isin(selected_periods)]
            .melt(id_vars=["period"], var_name="skills", value_name="growth")
            .replace([float("inf"), float("-inf")], pd.NA)
            .dropna(subset=["growth"])
        )
        momentum_long["growth"] = pd.to_numeric(momentum_long["growth"], errors="coerce")
        momentum_long = momentum_long.dropna(subset=["growth"])

        # Filter momentum data by selected skills
        if selected_skills_lower:
            momentum_long = momentum_long[
                momentum_long["skills"].str.lower().isin(selected_skills_lower)
            ]

        st.subheader("Skill Growth Momentum")

        # Interactive controls for momentum visualization
        ctl1, ctl2 = st.columns(2)
        with ctl1:
            # Option to hide periods where all selected skills have 0% growth
            hide_zero_periods = st.checkbox(
                "Hide periods with all 0% growth (selected skills)",
                value=True,
                help="Removes flat months to focus on periods with actual changes"
            )
        with ctl2:
            # Option to show growth percentages as labels on bars
            show_labels = st.checkbox("Show value labels", value=True)

        # Filter out zero-growth periods if requested
        if hide_zero_periods and not momentum_long.empty:
            non_zero_by_period = momentum_long.groupby("period")["growth"].apply(lambda s: (s != 0).any())
            keep_periods = non_zero_by_period[non_zero_by_period].index.tolist()
            momentum_long = momentum_long[momentum_long["period"].isin(keep_periods)]

        # Display momentum chart or no-data message
        if momentum_long.empty:
            st.info("No momentum records available for the selected filters.")
        else:
            # Format growth values as percentage labels
            momentum_long["growth_label"] = momentum_long["growth"].map(lambda v: f"{v:+.1f}%")
            
            # Grouped bar chart with facets by period
            fig_momentum = px.bar(
                momentum_long,
                x="skills",
                y="growth",
                color="growth",
                facet_col="period",
                facet_col_spacing=0.08,
                title="Month-over-Month Growth by Skill",
                text="growth_label" if show_labels else None,
            )
            fig_momentum.update_layout(
                showlegend=False,
                xaxis_title="Skill",
                yaxis_title="Growth (%)",
            )
            if show_labels:
                fig_momentum.update_traces(textposition="outside", cliponaxis=False)

            st.plotly_chart(fig_momentum, use_container_width=True)
            
if __name__ == "__main__":
    """
    Entry point: Instantiate the dashboard and render all visualizations.
    Run with: streamlit run app/dashboard.py
    """
    TrendDashboard().render()