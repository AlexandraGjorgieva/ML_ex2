from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

def evaluate_model(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def plot_results(test_results_df):
    models = test_results_df.index
    mse_values = test_results_df['MSE']
    mae_values = test_results_df['MAE']
    r2_values = test_results_df['R2']

    #MSE
    plt.figure(figsize=(10, 6))
    plt.bar(models, mse_values, color='blue', alpha=0.7)
    plt.title("Mean Squared Error (MSE) by Model")
    plt.ylabel("MSE")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    #MAE
    plt.figure(figsize=(10, 6))
    plt.bar(models, mae_values, color='green', alpha=0.7)
    plt.title("Mean Absolute Error (MAE) by Model")
    plt.ylabel("MAE")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    #R2
    plt.figure(figsize=(10, 6))
    plt.bar(models, r2_values, color='orange', alpha=0.7)
    plt.title("R² Score by Model")
    plt.ylabel("R²")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def draw_histograms(df):
    vars = df.columns
    n_vars = len(vars)
    ngrid = int(sqrt(n_vars)) + 1

    fig = plt.figure(figsize=(ngrid * 6, ngrid * 4))

    for ix, var in enumerate(vars):
        if df[var].dtype in ['int64', 'float64']:  #only plotting numeric variables
            ax = fig.add_subplot(ngrid, ngrid, ix + 1)
            df[var].hist(bins=10, ax=ax)
            ax.set_title(f"{ix + 1}: " + var + " values distribution")

    fig.tight_layout()
    plt.show()