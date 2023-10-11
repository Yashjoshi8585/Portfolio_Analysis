import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import date
from nsepy import get_history as gh

plt.style.use('fivethirtyeight')  # Setting matplotlib style

stocksymbols = ['TATAMOTORS', 'DABUR', 'ICICIBANK', 'WIPRO', 'BPCL', 'IRCTC', 'INFY', 'RELIANCE']
startdate = date(2019, 10, 14)
end_date = date.today()
print(end_date)
print(f"You have {len(stocksymbols)} assets in your portfolio")

df = pd.DataFrame()

for symbol in stocksymbols:
    data = gh(symbol=symbol, start=startdate, end=end_date)[['Symbol', 'Close']]
    data.rename(columns={'Close': symbol}, inplace=True)
    data.drop(['Symbol'], axis=1, inplace=True)

    if df.empty:
        df = data
    else:
        df = df.join(data)

# Visualizing the DataFrame
print(df)

# Plotting the Portfolio Close Price History
fig, ax = plt.subplots(figsize=(15, 8))
for col in df.columns:
    ax.plot(df.index, df[col], label=col)

ax.set_title("Portfolio Close Price History")
ax.set_xlabel('Date', fontsize=18)
ax.set_ylabel('Close Price INR (₨)', fontsize=18)
ax.legend(df.columns, loc='upper left')
plt.show()

# Calculating and visualizing the correlation matrix
correlation_matrix = df.corr(method='pearson')
print('Correlation between Stocks in your portfolio')
fig1 = plt.figure()
sb.heatmap(correlation_matrix, xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns,
            cmap='YlGnBu', annot=True, linewidth=0.5)
plt.show(fig1)

# Calculate daily simple returns
daily_simple_return = df.pct_change(1)
daily_simple_return.dropna(inplace=True)
print('Daily simple returns')
fig, ax = plt.subplots(figsize=(15, 8))

for col in daily_simple_return.columns:
    ax.plot(daily_simple_return.index, daily_simple_return[col], lw=2, label=col)

ax.legend(loc='upper right', fontsize=10)
ax.set_title('Volatility in Daily simple returns')
ax.set_xlabel('Date')
ax.set_ylabel('Daily simple returns')
plt.show()

# Calculate and print average daily returns
print('Average Daily returns(%) of stocks in your portfolio')
avg_daily = daily_simple_return.mean()
print(avg_daily * 100)

# Plot a box plot to visualize risk
daily_simple_return.plot(kind="box", figsize=(20, 10), title="Risk Box Plot")

# Calculate and print annualized standard deviation
print('Annualized Standard Deviation (Volatility(%), 252 trading days) of individual stocks in your portfolio on the basis of daily simple returns.')
print(daily_simple_return.std() * np.sqrt(252) * 100)

# Calculate and print the Sharpe ratio
sharpe_ratio = avg_daily / (daily_simple_return.std() * np.sqrt(252)) * 100
print('Sharpe Ratio')
print(sharpe_ratio)

# Calculate and visualize daily cumulative simple returns
daily_cumulative_simple_return = (daily_simple_return + 1).cumprod()
print('Cummulative Returns')
fig, ax = plt.subplots(figsize=(18, 8))

for col in daily_cumulative_simple_return.columns:
    ax.plot(daily_cumulative_simple_return.index, daily_cumulative_simple_return[col], lw=2, label=col)

ax.legend(loc='upper left', fontsize=10)
ax.set_title('Daily Cumulative Simple returns/growth of investment')
ax.set_xlabel('Date')
ax.set_ylabel('Growth of ₨ 1 investment')
plt.show(fig)
