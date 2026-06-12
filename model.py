import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import sqlite3
    from statsforecast import StatsForecast
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    from statsforecast.models import AutoARIMA, SeasonalNaive

    conn = sqlite3.connect('healthcare_demand.db')
    df = pd.read_sql_query("SELECT * FROM daily_demand", conn)
    conn.close()

    df['ds'] = pd.to_datetime(df['ds'])

    df = df.sort_values(['unique_id', 'ds'])
    df['y'] = df.groupby('unique_id')['y'].transform(
        lambda x: x.interpolate(method='linear').ffill().bfill()
    )

    test_size = 30
    test_df = df.groupby('unique_id').tail(test_size)
    train_df = df.drop(test_df.index)

    models = [
        SeasonalNaive(season_length=7),
        AutoARIMA()
    ]

    sf = StatsForecast(
        models=models,
        freq='D',
        n_jobs=1  
    )

    train_df_baseline = train_df[['unique_id', 'ds', 'y']]

    forecast_df = sf.forecast(df=train_df_baseline, h=test_size)

    forecast_df.head(15)

    #eval metrics
    eval_df = test_df.merge(forecast_df, on=['unique_id', 'ds'], how='inner')

    def evaluate_model(y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)

        return {
            'Model': model_name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE': f"{round(mape * 100, 2)}%"
        }

    metrics = []
    metrics.append(evaluate_model(eval_df['y'], eval_df['SeasonalNaive'], 'Seasonal Naive'))
    metrics.append(evaluate_model(eval_df['y'], eval_df['AutoARIMA'], 'AutoARIMA'))

    metrics_df = pd.DataFrame(metrics)
    metrics_df
    return (
        SeasonalNaive,
        StatsForecast,
        eval_df,
        evaluate_model,
        np,
        pd,
        test_df,
        train_df_baseline,
    )


@app.cell
def _(eval_df, evaluate_model, np, pd, test_df, train_df_baseline):
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled_train = train_df_baseline.copy()
    scaled_train['y_scaled'] = scaler.fit_transform(scaled_train[['y']])

    sequence_length = 30
    forecast_horizon = 30

    X_train, y_train = [], []

    for uid, group in scaled_train.groupby('unique_id'):
        values = group['y_scaled'].values
        for i in range(len(values) - sequence_length - forecast_horizon + 1):
            X_train.append(values[i : i + sequence_length])
            y_train.append(values[i + sequence_length : i + sequence_length + forecast_horizon])

    X_tensor = torch.FloatTensor(np.array(X_train)).unsqueeze(-1)
    y_tensor = torch.FloatTensor(np.array(y_train))

    class MedicalDemandLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, output_size=30):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, output_size)

        def forward(self, input_seq):
            lstm_out, _ = self.lstm(input_seq)
            last_time_step = lstm_out[:, -1, :]
            predictions = self.linear(last_time_step)
            return predictions

    model = MedicalDemandLSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    epochs = 50
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = loss_function(y_pred, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch + 1:3} | MSE Loss: {loss.item():.4f}')

    model.eval()
    lstm_predictions = []

    with torch.no_grad():
        for uid, group in scaled_train.groupby('unique_id'):
            last_30_days = group['y_scaled'].values[-sequence_length:]
            seq = torch.FloatTensor(last_30_days).view(1, sequence_length, 1)

            pred_scaled = model(seq).numpy()[0]
            pred_real = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

            future_dates = test_df[test_df['unique_id'] == uid]['ds'].values
            for ds, val in zip(future_dates, pred_real):
                lstm_predictions.append({'unique_id': uid, 'ds': ds, 'LSTM': max(0, val)}) 

    lstm_preds_df = pd.DataFrame(lstm_predictions)

    #Metrics
    final_eval_df = eval_df.merge(lstm_preds_df, on=['unique_id', 'ds'], how='inner')

    final_metrics = [
        evaluate_model(final_eval_df['y'], final_eval_df['SeasonalNaive'], 'Seasonal Naive'),
        evaluate_model(final_eval_df['y'], final_eval_df['AutoARIMA'], 'AutoARIMA'),
        evaluate_model(final_eval_df['y'], final_eval_df['LSTM'], 'Pure PyTorch LSTM')
    ]

    final_metrics_df = pd.DataFrame(final_metrics)
    final_metrics_df
    return


@app.cell
def _(
    SeasonalNaive,
    StatsForecast,
    eval_df,
    evaluate_model,
    pd,
    train_df_baseline,
):
    from hierarchicalforecast.utils import aggregate
    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import BottomUp

    h_df_clean = train_df_baseline[train_df_baseline['unique_id'] != 'Total_Hospital'].copy()
    h_df_clean = h_df_clean.rename(columns={'unique_id': 'Item'})
    h_df_clean['Total'] = 'Total_Hospital'

    h_spec = [
        ['Total'],
        ['Total', 'Item']
    ]

    h_Y_df, h_S_df, h_tags = aggregate(h_df_clean, spec=h_spec)
    h_sf_engine = StatsForecast(
        models=[SeasonalNaive(season_length=7)], 
        freq='D', 
        n_jobs=1
    )

    h_sf_engine.fit(h_Y_df)
    h_Y_hat_df = h_sf_engine.predict(h=30)

    h_reconciler = HierarchicalReconciliation(reconcilers=[BottomUp()])
    h_Y_rec_df = h_reconciler.reconcile(
        Y_hat_df=h_Y_hat_df, 
        Y_df=h_Y_df, 
        S_df=h_S_df, 
        tags=h_tags
    )

    cols_to_clip = h_Y_rec_df.select_dtypes(include=['number']).columns
    h_Y_rec_df[cols_to_clip] = h_Y_rec_df[cols_to_clip].clip(lower=0)

    h_reconciled_bottom = h_Y_rec_df.reset_index()
    h_reconciled_bottom['unique_id'] = h_reconciled_bottom['unique_id'].str.split('/').str[1]
    h_eval_merged = eval_df.merge(
        h_reconciled_bottom[['unique_id', 'ds', 'SeasonalNaive/BottomUp']], 
        on=['unique_id', 'ds'], 
        how='inner'
    )

    # Run scoring 
    h_final_metrics_list = [
        evaluate_model(h_eval_merged['y'], h_eval_merged['SeasonalNaive/BottomUp'], 'Reconciled (BottomUp)')
    ]

    h_final_metrics_df = pd.DataFrame(h_final_metrics_list)
    h_final_metrics_df
    return BottomUp, HierarchicalReconciliation, aggregate


@app.cell
def _(
    BottomUp,
    HierarchicalReconciliation,
    SeasonalNaive,
    StatsForecast,
    aggregate,
    train_df_baseline,
):
    import joblib
    import os

    new_df = train_df_baseline[train_df_baseline['unique_id'] != 'Total_Hospital'].copy()
    new_df = new_df.rename(columns={'unique_id': 'Item'})
    new_df['Total'] = 'Total_Hospital'

    new_Y_df, new_S_df, new_tags = aggregate(new_df, spec=[['Total'], ['Total', 'Item']])

    new_sf = StatsForecast(models=[SeasonalNaive(season_length=7)], freq='D', n_jobs=1)
    new_sf.fit(new_Y_df)

    os.makedirs('models', exist_ok=True)
    joblib.dump(new_sf, 'models/model.pkl')

    new_sf_hat = new_sf.predict(h=30).reset_index()

    new_sf_hat = new_sf.predict(h=30).reset_index()

    new_reconciler = HierarchicalReconciliation(reconcilers=[BottomUp()])
    new_Y_rec_df = new_reconciler.reconcile(
        Y_hat_df=new_sf_hat, 
        Y_df=new_Y_df, 
        S_df=new_S_df, 
        tags=new_tags
    )

    new_cols = new_Y_rec_df.select_dtypes(include=['number']).columns
    new_Y_rec_df[new_cols] = new_Y_rec_df[new_cols].clip(lower=0)
    new_Y_rec_df.head()
    return


if __name__ == "__main__":
    app.run()
