import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sqlalchemy import create_engine
    import random

    exog_df = pd.read_excel('Health_Dataset.xlsx')

    exog_df['Year'] = exog_df['Year'].ffill()
    #----------
    date_strings = exog_df['Year'].astype(int).astype(str) + '-' + exog_df['Week'].astype(int).astype(str) + '-1'
    exog_df['Date'] = pd.to_datetime(date_strings, format="%Y-%W-%w")

    exog_df.set_index('Date', inplace=True)
    exog_df.sort_index(inplace=True)

    exog_df = exog_df.resample('W-MON').mean() 
    exog_df = exog_df.interpolate(method='linear').ffill().bfill()

    target_disease = 'Dengue'

    min_cases = exog_df[target_disease].min()
    max_cases = exog_df[target_disease].max()
    exog_df['Demand_Multiplier'] = 1.0 + ((exog_df[target_disease] - min_cases) / (max_cases - min_cases)) * 1.5

    exog_daily = exog_df[['Demand_Multiplier']].resample('D').ffill()

    start_date = exog_daily.index.min()
    end_date = exog_daily.index.max()
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    #Hierarchies
    regions = ['NCR_North', 'NCR_South']
    hospitals = {
        'NCR_North': ['Quezon_City_Gen', 'Caloocan_Med'],
        'NCR_South': ['PGH_Manila', 'Makati_Med']
    }
    products = ['IV_Fluids', 'Paracetamol_Drops', 'N95_Masks']

    #Generate Dataset
    data = []

    for region in regions:
        for hosp in hospitals[region]:
            for prod in products:
                unique_id = f"{region}_{hosp}_{prod}"

                base_demand = 200 if prod =="IV_Fluids" else 100
                day_of_week_effect = np.where(dates.dayofweek < 5, 1.1, 0.8)
                df_temp = pd.DataFrame({
                    'unique_id': unique_id,
                    'ds': dates
                })
                df_temp['multiplier'] = df_temp['ds'].map(exog_daily['Demand_Multiplier']).fillna(1.0)
                noise = np.random.normal(0, 15, size=len(dates))
                y_demand = (base_demand * day_of_week_effect * df_temp['multiplier']) + noise
                df_temp['y'] = np.maximum(y_demand, 0).astype(int)
                df_temp.rename(columns={'multiplier': 'disease_surge_index'}, inplace=True)
                data.append(df_temp)

    master_df = pd.concat(data, ignore_index=True)

    drop_indices = master_df.sample(frac=0.05, random_state=42).index
    master_df.loc[drop_indices, 'y'] = np.nan

    engine = create_engine('sqlite:///healthcare_demand.db')

    master_df.to_sql('daily_demand', con=engine, index=False, if_exists='replace')

    master_df.head(15)


    return


if __name__ == "__main__":
    app.run()
