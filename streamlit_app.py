import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error
from scipy.stats import zscore
import altair as alt

st.set_page_config(page_title="📊 Demand Forecasting", layout="wide")
st.title("📈 Demand Forecasting System using Prophet")

uploaded_file = st.file_uploader("Upload a CSV file with 'date' and 'sales' columns", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    try:
        df['ds'] = pd.to_datetime(df['date'])
        df['y'] = df['sales']
        df = df[['ds', 'y']]

        st.success("✅ File uploaded and parsed successfully.")
        st.write("Preview of uploaded data:")
        preview_df = df[['ds', 'y']].copy()
        preview_df.columns = ['Date', 'Sales']
        st.dataframe(preview_df.tail())


        # Train Prophet model
        model = Prophet()
        model.fit(df)

        # Create future DataFrame
        future = model.make_future_dataframe(periods=60)
        forecast = model.predict(future)

        st.subheader("📉 Forecast Plot")
        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧩 Forecast Components")
        st.plotly_chart(plot_plotly(model, forecast, trend=True), use_container_width=True)

        st.subheader("📊 Forecast Table (next 30 days)")
        forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_table.columns = ['Date', 'Predicted Sales', 'Lower Bound', 'Upper Bound']
        st.dataframe(forecast_table.tail(30))

        # Download Forecast CSV
        st.subheader("📥 Download Forecast")
        download_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        download_forecast.columns = ['Date', 'Predicted Sales', 'Lower Bound', 'Upper Bound']

        csv = download_forecast.to_csv(index=False).encode('utf-8')

        st.download_button(
        label="📁 Download Forecast CSV",
        data=csv,
        file_name='sales_forecast.csv',
        mime='text/csv'
        )


        # --- Business Insights and Inventory Suggestions ---
        st.subheader("💡 Business Suggestions")

        past_30_avg = df['y'].tail(30).mean()
        future_df = forecast[forecast['ds'] > df['ds'].max()]
        future_30_avg = future_df['yhat'].head(30).mean()

        if future_30_avg > past_30_avg * 1.25:
            st.info("📦 Demand is expected to increase significantly. Consider increasing inventory.")
        elif future_30_avg < past_30_avg * 0.75:
            st.warning("📉 Demand is forecasted to drop. You may reduce stock or run promotions.")
        else:
            st.success("✅ Forecasted demand is stable. Continue with regular inventory.")

        lead_time_days = 7
        recommended_restock_date = future_df.iloc[lead_time_days]['ds'].date()

        if future_30_avg > past_30_avg * 1.1:
            st.subheader("📅 Inventory Refill Recommendation")
            st.write(f"🛒 With a lead time of **{lead_time_days} days**, you should consider **reordering by {recommended_restock_date}**.")

        # --- Anomaly Detection ---
        st.subheader("🚨 Anomaly Detection (Historical Data)")
        df['zscore'] = zscore(df['y'])
        anomalies = df[(df['zscore'].abs() > 2)]

        if not anomalies.empty:
            st.warning(f"⚠️ Found {len(anomalies)} historical anomalies (spikes/drops).")
            # Rename columns for display
            anomaly_table = anomalies[['ds', 'y', 'zscore']].copy()
            anomaly_table.columns = ['Date', 'Sales', 'Z-Score']

            st.dataframe(anomaly_table.sort_values('Date'))

        else:
            st.success("✅ No significant anomalies detected in historical sales.")

        # --- Anomaly Chart ---
        st.subheader("📈 Sales with Anomalies Highlighted")
        base = alt.Chart(df).mark_line().encode(
        x=alt.X('ds:T', title='Date'),
        y=alt.Y('y:Q', title='Sales'),
        tooltip=[alt.Tooltip('ds:T', title='Date'), alt.Tooltip('y:Q', title='Sales')]
                ).properties(title='Sales Over Time')

        points = alt.Chart(anomalies).mark_circle(color='red', size=60).encode(
        x=alt.X('ds:T', title='Date'),
        y=alt.Y('y:Q', title='Sales'),
        tooltip=[alt.Tooltip('ds:T', title='Date'), alt.Tooltip('y:Q', title='Sales'), 'zscore']
                )

        st.altair_chart(base + points, use_container_width=True)


    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

else:
    st.info("👆 Please upload a CSV file to get started.")
