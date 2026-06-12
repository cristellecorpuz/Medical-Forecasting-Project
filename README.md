# Medical-Demand-Forecasting
This repository contains my implementation and deployment work for a machine learning-based hospital demand forecasting task using patient volume data.

# 📌 Overview

This project is an overview of patient demand forecasting across multiple hospital units using statistical baseline models deployed as a live, containerized service.

Topics covered include:
- Time series data preprocessing
- Hierarchical aggregation and tag generation
- Statistical baseline modeling (Seasonal Naive)
- REST API development for model inference
- Interactive data visualization and dashboarding
- Docker containerization

The dataset used is based on hospital patient volume records with unit-level categorizations (e.g., Emergency, Outpatient).

#🧪 Project Pipeline
---

# Preprocessing & Modeling
The modeling pipeline prepares the dataset for forecasting by:
- Loading historical patient volume data
- Formatting temporal data and establishing structural hierarchies
- Training a Seasonal Naive baseline model to capture 7-day weekly cycles (mapping the drop in non-emergency volume during weekends)
- Serializing the trained forecasting engine into a .pkl format for production use

# Inference API
---
A custom FastAPI application is implemented to:
- Load the serialized forecasting model into memory at startup
- Expose REST API endpoints to generate real-time future demand predictions
- Format the forecasted time-series data for downstream consumption
- Ensure environment independence by containerizing the entire backend service using Docker

# Interactive Dashboard
---
The frontend notebook implements the visual analysis pipeline using Streamlit and Plotly.
The workflow includes:
- Live data fetching directly from the REST API container
- Data transformation (melting wide-format forecasts into long-format for plotting)
- Interactive line charts visualizing the baseline forecast model
- Unit-level drill-down capabilities via sidebar filtering
- Raw data tabular inspection

# ⚙️ Technical Implementation
---
- Machine Learning Workflow
- Time series data preprocessing
- Baseline forecasting pipeline
- Model serialization and API deployment
- Containerized environment build
- Experiment visualization and dashboarding

# Techniques Used
---
- Time Series Forecasting
- REST API Architecture
- Containerization (Docker)
- Interactive Data Visualization (Plotly)
- Web Dashboard Development (Streamlit)
- Model Deployment
