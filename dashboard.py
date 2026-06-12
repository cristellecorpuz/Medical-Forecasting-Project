import streamlit as st
import plotly.express as px
import pandas as pd
import requests

st.set_page_config(page_title="Medical Forecast AI", layout="wide")
st.title("Medical Demand Forecast Dashboard")

unit_selection = st.sidebar.selectbox("Select Hospital Unit", ["Total_Hospital", "Emergency", "Outpatient"])

def get_forecast_data():
    try:
        response = requests.post("http://127.0.0.1:8000/predict")
        return pd.DataFrame(response.json()['forecast'])
    except:
        st.error("Please ensure the FastAPI server is running.")
        return None

if st.button("Refresh Forecast"):
    df = get_forecast_data()
    
    if df is not None:
        df['ds'] = pd.to_datetime(df['ds'])
        filtered_df = df[df['unique_id'] == unit_selection].copy()
        plot_df = filtered_df.melt(
            id_vars=['ds'], 
            value_vars=['SeasonalNaive'], 
            var_name='Model', 
            value_name='Demand'
        )

        fig = px.line(
            plot_df, 
            x='ds', 
            y='Demand', 
            color='Model',
            title=f"Forecast Analysis: {unit_selection}",
            markers=True
        )
        
        fig.update_layout(
            template="plotly_dark", 
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Forecasted Demand"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Raw Data"):
            st.dataframe(df)