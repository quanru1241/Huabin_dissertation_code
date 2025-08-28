
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import pearsonr
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# Optional: AutoML
try:
    from pycaret.regression import setup, compare_models, predict_model
    has_pycaret = True
except ImportError:
    has_pycaret = False
    print("⚠️ PyCaret not found. AutoML will be skipped.")

# ------------------ 1. Load Data ------------------ #
def load_data():
    met_ds = xr.open_dataset("AU-ASM_2011-2017_OzFlux_Met.nc")
    flux_ds = xr.open_dataset("AU-ASM_2011-2017_OzFlux_Flux.nc")
    met_df = met_ds.to_dataframe().reset_index()
    flux_df = flux_ds.to_dataframe().reset_index()
    df = pd.merge_asof(met_df.sort_values('time'), flux_df.sort_values('time'), on='time')

    features_raw = ['SWdown', 'LWdown', 'Tair', 'Qair', 'RH', 'Psurf', 'Wind',
                    'CO2air', 'VPD', 'LAI', 'Ustar']
    target_vars = ['GPP', 'NEE']
    df = df[['time'] + features_raw + target_vars].dropna()

    # Derived features
    df['SW_LAI'] = df['SWdown'] * df['LAI']
    df['RH_Tair'] = df['RH'] * df['Tair']
    df['SWdown_lag1'] = df['SWdown'].shift(1)
    df['Tair_lag1'] = df['Tair'].shift(1)
    df = df.dropna()

    df = df.set_index('time').sort_index()
    features = features_raw + ['SW_LAI', 'RH_Tair', 'SWdown_lag1', 'Tair_lag1']
    return df, features

# ------------------ 2. Evaluation ------------------ #
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    rho, _ = pearsonr(y_true, y_pred)
    return r2, rmse, mae, rho

# ------------------ 3. Model Runners ------------------ #
def run_xgb(X_train, y_train, X_test):
    model = xgb.XGBRegressor(objective='reg:squarederror',
                             n_estimators=100, max_depth=3, learning_rate=0.05,
                             colsample_bytree=0.6, subsample=0.8, reg_alpha=0.5, reg_lambda=0.5)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def run_rf(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=200, min_samples_split=2, min_samples_leaf=4,
                                   max_features='sqrt', max_depth=None, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def run_automl(train_df, test_df, features, target):
    if not has_pycaret:
        return None
    train_data = pd.concat([train_df[features], train_df[target]], axis=1).reset_index(drop=True)
    setup(data=train_data, target=target, session_id=42,
          train_size=0.999, fold_strategy='timeseries', fold=3,
          preprocess=True, numeric_features=features,
          remove_multicollinearity=True, multicollinearity_threshold=0.95,
          silent=True, verbose=False, data_split_shuffle=False)
    best = compare_models(sort='R2')
    test_data = test_df[features].reset_index(drop=True)
    pred = predict_model(best, data=test_data)
    return pred['prediction_label'].values

# ------------------ 4. Plotting ------------------ #
def plot_comparison(index, y_true, predictions, target):
    plt.figure(figsize=(12, 4))
    plt.plot(index, y_true, label='Observed', color='black', linestyle='--')
    for name, pred in predictions.items():
        plt.plot(index, pred, label=name)
    plt.title(f"{target} – Observed vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{target}_comparison.png")
    plt.close()

    plt.figure()
    for name, pred in predictions.items():
        sns.histplot(y_true - pred, kde=True, label=name, bins=50)
    plt.title(f"{target} – Residual Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{target}_residuals.png")
    plt.close()

# ------------------ 5. Main Execution ------------------ #
def run_all_models():
    df, features = load_data()
    split_idx = int(len(df) * 0.7)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    for target in ['GPP', 'NEE']:
        X_train, y_train = train_df[features], train_df[target]
        X_test, y_test = test_df[features], test_df[target]

        preds = {}
        metrics = {}

        preds['XGBoost'] = run_xgb(X_train, y_train, X_test)
        preds['RandomForest'] = run_rf(X_train, y_train, X_test)
        if has_pycaret:
            preds['AutoML'] = run_automl(train_df, test_df, features, target)

        print(f"\n📊 Results for {target}")
        for name, y_pred in preds.items():
            r2, rmse, mae, rho = evaluate(y_test, y_pred)
            print(f"{name}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, ρ={rho:.4f}")

        plot_comparison(y_test.index, y_test.values, preds, target)

if __name__ == '__main__':
    run_all_models()
