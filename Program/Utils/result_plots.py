import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def prediction_vs_real_price(df, arma_model, arma_sent_model):
    # Predicted values for the same period as your data
    # ARIMA without sentiment
    df['ARIMA_Pred'] = arma_model.fittedvalues
    # ARIMA with sentiment
    df['ARIMA_Sent_Pred'] = arma_sent_model.fittedvalues

    df['ARIMA_Pred_Price'] = df['Close'].iloc[0] * (1 + df['ARIMA_Pred']).cumprod()
    df['ARIMA_Sent_Pred_Price'] = df['Close'].iloc[0] * (1 + df['ARIMA_Sent_Pred']).cumprod()

    plt.figure(figsize=(14, 6))

    # Actual price
    plt.plot(df['Date'], df['Close'], label='Actual Price', color='black', linewidth=2)

    # Predicted prices from ARIMA
    plt.plot(df['Date'], df['ARIMA_Pred_Price'], label='ARIMA(1,0,1)', linestyle='--', color='blue')

    # Predicted prices from ARIMA + Sentiment
    plt.plot(df['Date'], df['ARIMA_Sent_Pred_Price'], label='ARIMA + Sentiment', linestyle='--', color='red')

    # Title, labels, legend
    plt.title('Predicted vs Actual Prices', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def prediction_vs_real_priceChange(df, arma_model, arma_sent_model, start_date=None, end_date=None):
    # Predicted values for the same period as your data
    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Strip timezone if present to avoid comparison errors
    if pd.api.types.is_datetime64tz_dtype(df['Date']):
        df['Date'] = df['Date'].dt.tz_convert(None)

    # Predicted values for the same period as your data
    df['ARIMA_Pred'] = arma_model.fittedvalues
    df['ARIMA_Sent_Pred'] = arma_sent_model.fittedvalues

    # Initialize predicted price lists with first actual price
    pred_prices = [df['Close'].iloc[0]]
    pred_sent_prices = [df['Close'].iloc[0]]

    # Reconstruct predicted prices from predicted changes
    for i in range(1, len(df)):
        pred_prices.append(df['Close'].iloc[i - 1] * (1 + df['ARIMA_Pred'].iloc[i]))
        pred_sent_prices.append(df['Close'].iloc[i - 1] * (1 + df['ARIMA_Sent_Pred'].iloc[i]))

    df['ARIMA_Pred_Price'] = pred_prices
    df['ARIMA_Sent_Pred_Price'] = pred_sent_prices

    # Filter by date if specified (convert both sides safely)
    plot_df = df.copy()
    if start_date is not None:
        plot_df = plot_df[plot_df['Date'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        plot_df = plot_df[plot_df['Date'] <= pd.to_datetime(end_date)]

    plt.figure(figsize=(14, 6))
    plt.plot(plot_df['Date'], plot_df['Close'], label='Actual Price', color='black', linewidth=2)
    plt.plot(plot_df['Date'], plot_df['ARIMA_Pred_Price'], label='ARIMA(1,0,1)', linestyle='--', color='blue')
    plt.plot(plot_df['Date'], plot_df['ARIMA_Sent_Pred_Price'], label='ARIMA + Sentiment', linestyle='--', color='red')
    plt.title('Predicted vs Actual Prices (Reconstructed from Price Changes)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def sentiment_price_plot(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot price
    ax1.plot(df['Date'], df['Close'], color='blue', label='Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create second y-axis for sentiment
    ax2 = ax1.twinx()
    ax2.plot(df['Date'], df['sentiment_lag0'], color='red', alpha=0.7, label='Daily Sentiment')
    ax2.set_ylabel('Daily Sentiment', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Title & legend
    fig.suptitle('Price vs Daily Sentiment Over Time', fontsize=14)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_price_change_sentiment_scatter(df, lag):
    df_copy = df.copy()

    plt.figure(figsize=(10, 6))
    sns.regplot(x=df_copy[f'sentiment_lag{lag}'], y=df['Pct_Change'], line_kws={"color": "red"})
    plt.title(f"Daily Sentiment lag{lag} vs Price Change")
    plt.xlabel("Daily Sentiment")
    plt.ylabel("Price Change in %")
    plt.grid(alpha=0.3)
    plt.show()

def plot_arima_pvalues(model, significance_level=0.05):
    """
    Plots p-values of ARIMA/ARMA model coefficients with a significance threshold.

    Parameters:
        model: Fitted ARIMA/ARMA model from statsmodels
        significance_level: Threshold for statistical significance (default 0.05)
    """
    # Extract p-values from the fitted model
    pvalues = model.pvalues

    # Convert to DataFrame for easier plotting
    pval_df = pd.DataFrame({
        'Coefficient': pvalues.index,
        'p-value': pvalues.values
    })

    plt.figure(figsize=(10, 5))
    sns.barplot(x='Coefficient', y='p-value', data=pval_df, palette='coolwarm')
    plt.axhline(significance_level, color='black', linestyle='--',
                label=f'Significance Threshold ({significance_level})')
    plt.title("ARIMA/ARMA Coefficient Significance (p-values)")
    plt.ylabel("p-value")
    plt.xlabel("Coefficients")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_arma_aic_heatmap(results_arima, results_arima_sentiment):
    """
    Plots three heatmaps side by side:
    1. ARIMA baseline AIC
    2. ARIMA + Sentiment AIC
    3. ΔAIC = (sentiment - baseline)
       -> Green = improvement (lower AIC), Red = worse

    Parameters
    ----------
    results_arima : list[dict]
        [{'p': int, 'q': int, 'AIC': float}, ...]
    results_arima_sentiment : list[dict]
        Same structure, for sentiment-augmented models.
    """

    df_base = pd.DataFrame(results_arima)
    df_sent = pd.DataFrame(results_arima_sentiment)

    # Merge on p,q for ΔAIC calculation
    df_merge = pd.merge(df_base, df_sent, on=['p', 'q'], suffixes=('_base', '_sent'))
    df_merge['delta_AIC'] = df_merge['AIC_sent'] - df_merge['AIC_base']

    # Pivot tables
    pivot_base = df_merge.pivot(index='p', columns='q', values='AIC_base')
    pivot_sent = df_merge.pivot(index='p', columns='q', values='AIC_sent')
    pivot_diff = df_merge.pivot(index='p', columns='q', values='delta_AIC')

    # Create 3 plots side by side
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

    sns.heatmap(
        pivot_base, annot=True, fmt=".1f", cmap="coolwarm",
        cbar_kws={'label': 'AIC'}, ax=axes[0]
    )
    axes[0].set_title("ARIMA baseline")
    axes[0].set_xlabel("q")
    axes[0].set_ylabel("p")

    sns.heatmap(
        pivot_sent, annot=True, fmt=".1f", cmap="coolwarm",
        cbar_kws={'label': 'AIC'}, ax=axes[1]
    )
    axes[1].set_title("ARIMA + Sentiment")
    axes[1].set_xlabel("q")
    axes[1].set_ylabel("")

    sns.heatmap(
        pivot_diff, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0,
        cbar_kws={'label': 'ΔAIC (sent - base)'}, ax=axes[2]
    )
    axes[2].set_title("ΔAIC (Improvement)")
    axes[2].set_xlabel("q")
    axes[2].set_ylabel("")

    plt.suptitle("AIC Comparison: ARIMA vs ARIMA + Sentiment", fontsize=14, y=1.02)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
