============================================================

IndiGo Standby Crew Forecasting – Production‑Ready POC

Author: Krishna Jaiswal

Objective: Improve forecast accuracy (MAE, MAPE, SMAPE)

============================================================

-----------------------------

1. Imports & Configuration

-----------------------------

import pandas as pd import numpy as np import lightgbm as lgb from sklearn.metrics import mean_absolute_error import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

-----------------------------

2. Load & Prepare Data

-----------------------------

NOTE: Update path as per environment

DATA_PATH = 'DEL_SBY_prepared_data (UAT).csv'

df = pd.read_csv(DATA_PATH)

df['Date'] = pd.to_datetime(df['Date']) df = df.sort_values('Date').reset_index(drop=True)

-----------------------------

3. Business Target Definition

-----------------------------

Raw target is volatile for low counts

We predict ACTIVATION RATIO instead of raw count

df['activation_ratio'] = ( df['Standby Activation Count'] / df['Pairing Start Count'].replace(0, np.nan) )

df['activation_ratio'] = df['activation_ratio'].fillna(0)

-----------------------------

4. Categorical Encoding

-----------------------------

Rank encoding (domain driven)

df['Rank'] = df['Rank'].map({'FO': 0, 'CP': 1})

Station encoding (single station for POC, scalable later)

df['Station_enc'] = df['Station'].astype('category').cat.codes

-----------------------------

5. Time‑Based Features

-----------------------------

df['dayofweek'] = df['Date'].dt.dayofweek df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int) df['week'] = df['Date'].dt.isocalendar().week.astype(int) df['month'] = df['Date'].dt.month

-----------------------------

6. Lag & Rolling Features (Operational Memory)

-----------------------------

GROUP_COLS = ['Station', 'Duty Window Number', 'Rank'] TARGET = 'activation_ratio'

Lags

df['lag_1'] = df.groupby(GROUP_COLS)[TARGET].shift(1) df['lag_7'] = df.groupby(GROUP_COLS)[TARGET].shift(7)

df['pairing_lag_1'] = df.groupby(GROUP_COLS)['Pairing Start Count'].shift(1)

Rolling mean

df['rolling_7'] = ( df.groupby(GROUP_COLS)[TARGET] .shift(1) .rolling(7, min_periods=1) .mean() )

-----------------------------

7. Clean Modeling Frame

-----------------------------

df_model = df.dropna(subset=['lag_1']).reset_index(drop=True)

FEATURES = [ 'Duty Window Number', 'Rank', 'Station_enc', 'dayofweek', 'is_weekend', 'week', 'month', 'lag_1', 'lag_7', 'rolling_7', 'pairing_lag_1' ]

-----------------------------

8. Train / Validation Split (Time‑Based)

-----------------------------

SPLIT_DATE = df_model['Date'].quantile(0.8)

train = df_model[df_model['Date'] <= SPLIT_DATE] valid = df_model[df_model['Date'] > SPLIT_DATE]

X_train, y_train = train[FEATURES], train[TARGET] X_valid, y_valid = valid[FEATURES], valid[TARGET]

-----------------------------

9. LightGBM Model

-----------------------------

model = lgb.LGBMRegressor( n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42 )

model.fit(X_train, y_train)

-----------------------------

10. Predictions (Convert back to counts)

-----------------------------

valid['pred_ratio'] = model.predict(X_valid) valid['pred_ratio'] = valid['pred_ratio'].clip(lower=0)

valid['pred_standby'] = valid['pred_ratio'] * valid['Pairing Start Count'] valid['actual_standby'] = valid['Standby Activation Count']

-----------------------------

11. Evaluation Metrics

-----------------------------

def mape(y_true, y_pred): y_true = np.array(y_true) y_pred = np.array(y_pred) return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

def smape(y_true, y_pred): y_true = np.array(y_true) y_pred = np.array(y_pred) return 100 * np.mean( 2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6) )

MAE = mean_absolute_error(valid['actual_standby'], valid['pred_standby']) MAPE = mape(valid['actual_standby'], valid['pred_standby']) SMAPE = smape(valid['actual_standby'], valid['pred_standby'])

print('Evaluation Results:') print(f'MAE   : {MAE:.2f}') print(f'MAPE  : {MAPE:.2f}%') print(f'SMAPE : {SMAPE:.2f}%')

-----------------------------

12. Visual Validation

-----------------------------

plt.figure(figsize=(10, 5)) plt.plot(valid['Date'], valid['actual_standby'], label='Actual', marker='o') plt.plot(valid['Date'], valid['pred_standby'], label='Predicted', marker='x') plt.title('Actual vs Predicted Standby Activation') plt.xlabel('Date') plt.ylabel('Standby Count') plt.legend() plt.grid(True) plt.show()

-----------------------------

13. Business Conclusion

-----------------------------

• Ratio‑based forecasting reduces variance

• MAE reflects crew‑count accuracy

• SMAPE provides stable performance view for low counts

• Model is suitable for 3‑month operational planning

============================================================
