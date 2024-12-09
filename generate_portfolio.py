

# # Unsupervised Learning Trading Strategy
# 
# * Download/Load SP500 stocks prices data.
# * Calculate different features and indicators on each stock.
# * Aggregate on monthly level and filter top 150 most liquid stocks.
# * Calculate Monthly Returns for different time-horizons.
# * Download Fama-French Factors and Calculate Rolling Factor Betas.
# * For each month fit a K-Means Clustering Algorithm to group similar assets based on their features.
# * For each month select assets based on the cluster and form a portfolio based on Efficient Frontier max sharpe ratio optimization.
# * Visualize Portfolio returns and compare to SP500 returns.
# * Video at: https://youtu.be/9Y3yaoi9rUQ?si=JnKro_HeAoDGfiht
# * Source at: https://github.com/Luchkata/Algorithmic_Trading_Machine_Learning

# # All Packages Needed:
# * pandas, numpy, matplotlib, statsmodels, pandas_datareader, datetime, yfinance, sklearn, PyPortfolioOpt

# In[2]:


from statsmodels.regression.rolling import RollingOLS
from datetime import datetime, timedelta
import pandas_datareader.data as web
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import requests
import csv
import warnings
import sys
warnings.filterwarnings('ignore')


# ### Download tickers from S&P500

# In[3]:


# sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

# sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

# symbols_list = sp500['Symbol'].unique().tolist()

# # see https://fingpt.bot/ for stock prediction
# additional_symbols = ['ASML', 'VICI', 'WST','WYNN','ZBH','ABNB','AMZN','FTNT','GEHC','MRVL']
# # je bent heiiiiiir
# COMBINED_LIST = symbols_list + additional_symbols
# # remove double entries
# symbols_list = list(set(COMBINED_LIST))


# get the symbols from the nasdaq from Alpaca markets API
API_KEY = 'PKAFE6FQ1ZTHK0KNPDBU'
SECRET_KEY = 'x6DmwTMWv4Bfbdu9TyCxpV0hNTzIqGmSEtrazsrR'

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetExchange

# Initialize the TradingClient
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Get all assets
assets = trading_client.get_all_assets()

# Filter for active, tradeable stocks on the NASDAQ exchange
nasdaq_stocks = [
    asset for asset in assets 
    if asset.tradable  # Check if the asset is tradable
    and asset.asset_class == AssetClass.US_EQUITY  # Ensure the asset is a US equity (stock)
    and asset.exchange == AssetExchange.NASDAQ  # Ensure the asset is listed on NASDAQ
]

## from nasdaq_stocks, get the symbols of the stocks and create a list of symbols


# Extract the symbols from the nasdaq_stocks list
symbols_list = [asset.symbol for asset in nasdaq_stocks]


# print(symbols_list)
# print(len(symbols_list))
# sys.exit()




# In[ ]:





# ### Download tickers from S&P500

# ### Download tickers from Nasdaq 

# In[4]:


# def get_nasdaq_tickers(no_tickers=3000):
#     url = 'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25&offset=0&download=true'

#     headers = {
#             'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
#             }

#     resp = requests.get(url,headers=headers)
#     json_data = resp.json()
#     df = pd.DataFrame(json_data['data']['rows'],columns=json_data['data']['headers'])

#     ## convert columns to the correct data types
#     # df['lastsale'] = df['lastsale'].astype(float)

#     ## convert lastsale remove $ and convert to float
#     df['lastsale'] = df['lastsale'].str.replace('$','').astype(float)
#     ## convert netchange into float
#     df['netchange'] = df['netchange'].str.replace('$','').astype(float)
#     # convert pctchange remove % and convert to float
#     df['pctchange'] = pd.to_numeric(df['pctchange'].str.replace('%', ''), errors='coerce')
#     ## convert marketCap to string to float
#     df['marketCap'] = pd.to_numeric(df['marketCap'], errors='coerce')
#     ## convert volume to int
#     df['volume'] = df['volume'].str.replace(',','').astype(int)

#     df.to_csv('nasdaq.csv',index=False)

#     ### Select top 50 stocks with the highigest trade volume.

#     # Filter df and select the 100 rows with the highest volume
#     # df = df.sort_values('volume',ascending=False).head(no_tickers)
    
#    # hier moet je beter gaan filteren en de meest kansrijke aandelen selecteren

#     # tickers contains a list of df symbol and name 
#     tickers = list(df[['symbol','name']].itertuples(index=False, name=None))
#     # take tickers and copy the column symbol to a list
#     tickers = list(df['symbol'])
    

#     return tickers, df

# tickers, df = get_nasdaq_tickers(100)
# print(tickers)


# ## 1. Download/Load  stocks prices data.

# In[5]:


# end_date = '2023-09-27'
# # end_date = '2024-05-1'

# start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)

# Calculate the date 8 years ago from today 8*365
end_date = datetime.now()
start_date = end_date - timedelta(days=8*365)  # Approximation, does not account for leap years

# Format the dates in a way that yfinance expects
start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')


# symbols_list = ['AAPL', 'MSFT', 'AMZN', 'GOOGL'] ## ARRE Remove for debugging only


df = yf.download(tickers=symbols_list,
                 start=start_date,
                 end=end_date).stack()

df.to_csv('downloaded_stocks.csv')

df.index.names = ['date', 'ticker']

df.columns = df.columns.str.lower()

# Filter out tickers with fewer than 20 rows of data so that we can calculate the 20-day moving average
df_filtered = df.groupby('ticker').filter(lambda x: len(x) >= 20)

# Ensure the DataFrame `df` retains the same format
df = df_filtered

df


# In[ ]:





# ## 2. Calculate features and technical indicators for each stock.
# 
# * Garman-Klass Volatility
# * RSI
# * Bollinger Bands
# * ATR
# * MACD
# * Dollar Volume

# \begin{equation}
# \text{Garman-Klass Volatility} = \frac{(\ln(\text{High}) - \ln(\text{Low}))^2}{2} - (2\ln(2) - 1)(\ln(\text{Adj Close}) - \ln(\text{Open}))^2
# \end{equation}

# 

# In[6]:


df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
                                                        
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
                                                        
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])

def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        length=14)
    return atr.sub(atr.mean()).div(atr.std())

df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

def compute_macd(close):
   
    # # if len(close) < 25:
    # #     print("Not enough data to compute MACD")
    # #     return None
    # macd = pandas_ta.macd(close=close, length=20)
    # print ('Macd - ',macd.info())
    # macd = macd.iloc[:,0]
    # return macd.sub(macd.mean()).div(macd.std())

   
    if close.size < 25:  # Ensure there are enough data points for MACD calculation
        return pd.Series([None] * len(close), index=close.index)

    try:
        # Assuming 'close' is a pandas Series of closing prices
        macd = pandas_ta.macd(close=close, length=20)
    except Exception as e:
        print(f"An error occurred: {e}")
        # Handle the error or set macd to None or an empty DataFrame
        return pd.Series([None] * len(close), index=close.index)

    if macd is None or macd.empty:
        print("Debug: MACD calculation returned None or empty for data:", close)
        return pd.Series([None] * len(close), index=close.index)
    macd_series = macd.iloc[:, 0]  # Assuming the first column is the MACD line
    return macd_series.sub(macd_series.mean()).div(macd_series.std())


df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

df['dollar_volume'] = (df['adj close']*df['volume'])/1e6

# print(df)


# In[ ]:





# ## 3. Aggregate to monthly level and filter top 150 most liquid stocks for each month.
# 
# * To reduce training time and experiment with features and strategies, we convert the business-daily data to month-end frequency.

# In[7]:


last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open',
                                                          'high', 'low', 'close']]


data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                   df.unstack()[last_cols].resample('M').last().stack('ticker')],
                  axis=1)).dropna()



data


# * Calculate 5-year rolling average of dollar volume for each stocks before filtering.

# In[8]:


data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())

data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))

data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)

data


# ## 4. Calculate Monthly Returns for different time horizons as features.
# 
# * To capture time series dynamics that reflect, for example, momentum patterns, we compute historical returns using the method .pct_change(lag), that is, returns over various monthly periods as identified by lags.

# In[9]:


def calculate_returns(df):

    outlier_cutoff = 0.005

    lags = [1, 2, 3, 6, 9, 12]

    for lag in lags:

        df[f'return_{lag}m'] = (df['adj close']
                              .pct_change(lag)
                              .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                     upper=x.quantile(1-outlier_cutoff)))
                              .add(1)
                              .pow(1/lag)
                              .sub(1))
    return df
    
    
data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()


# ## 5. Download Fama-French Factors and Calculate Rolling Factor Betas.
# 
# * We will introduce the Fama—French data to estimate the exposure of assets to common risk factors using linear regression.
# 
# * The five Fama—French factors, namely market risk, size, value, operating profitability, and investment have been shown empirically to explain asset returns and are commonly used to assess the risk/return profile of portfolios. Hence, it is natural to include past factor exposures as financial features in models.
# 
# * We can access the historical factor returns using the pandas-datareader and estimate historical exposures using the RollingOLS rolling linear regression.

# In[10]:


factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                               'famafrench',
                               start='2010')[0].drop('RF', axis=1)

factor_data.index = factor_data.index.to_timestamp()

factor_data = factor_data.resample('M').last().div(100)

factor_data.index.name = 'date'

# Ensure data index is tz-naive BY ARRE
# data.index = data.index.set_levels([data.index.levels[0].tz_localize(None), data.index.levels[1]], level=[0, 1])

# Adjust factor_data to match return_1m's timezone
factor_data.index = pd.to_datetime(factor_data.index).tz_localize('UTC')

factor_data = factor_data.join(data['return_1m']).sort_index()

factor_data


# * Filter out stocks with less than 10 months of data.

# In[11]:


observations = factor_data.groupby(level=1).size()

valid_stocks = observations[observations >= 10]

factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

factor_data


# * Calculate Rolling Factor Betas.

# In[12]:


betas = (factor_data.groupby(level=1,
                            group_keys=False)
         .apply(lambda x: RollingOLS(endog=x['return_1m'], 
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(24, x.shape[0]),
                                     min_nobs=len(x.columns)+1)
         .fit(params_only=True)
         .params
         .drop('const', axis=1)))

betas 


# * Join the rolling factors data to the main features dataframe.

# In[13]:


factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

data = (data.join(betas.groupby('ticker').shift()))

data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

data = data.drop('adj close', axis=1)

data = data.dropna()

data.info()


# ### At this point we have to decide on what ML model and approach to use for predictions etc.
# 

# ## 6. For each month fit a K-Means Clustering Algorithm to group similar assets based on their features.
# 
# ### K-Means Clustering
# * You may want to initialize predefined centroids for each cluster based on your research.
# 
# * For visualization purpose of this tutorial we will initially rely on the ‘k-means++’ initialization.
# 
# * Then we will pre-define our centroids for each cluster.

# ### Apply pre-defined centroids.

# In[14]:


target_rsi_values = [30, 45, 55, 70]

initial_centroids = np.zeros((len(target_rsi_values), 18))

initial_centroids[:, 6] = target_rsi_values

initial_centroids


# In[15]:


from sklearn.cluster import KMeans

if 'cluster' in data.columns:
    data = data.drop('cluster', axis=1)


def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4,
                           random_state=0,
                           init=initial_centroids).fit(df).labels_
    return df

data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)

data


# In[16]:


def plot_clusters(data):

    cluster_0 = data[data['cluster']==0]
    cluster_1 = data[data['cluster']==1]
    cluster_2 = data[data['cluster']==2]
    cluster_3 = data[data['cluster']==3]

# Plot the clusters
    plt.scatter(cluster_0.iloc[:,0] , cluster_0.iloc[:,6] , color = 'red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:,0] , cluster_1.iloc[:,6] , color = 'green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:,0] , cluster_2.iloc[:,6] , color = 'blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:,0] , cluster_3.iloc[:,6] , color = 'black', label='cluster 3')
    
    plt.legend()
    plt.show()
    return


# In[17]:


plt.style.use('ggplot')

for i in data.index.get_level_values('date').unique().tolist():
    
    g = data.xs(i, level=0)
    
    plt.title(f'Date {i}')
    
    # Plot the clusters
    plot_clusters(g) # Hide plots by arre


# ## 7. For each month select assets based on the cluster and form a portfolio based on Efficient Frontier max sharpe ratio optimization
# 
# * First we will filter only stocks corresponding to the cluster we choose based on our hypothesis.
# 
# * Momentum is persistent and my idea would be that stocks clustered around RSI 70 centroid should continue to outperform in the following month - thus I would select stocks corresponding to cluster 3.
# 

# In[18]:


filtered_df = data[data['cluster']==3].copy()

filtered_df = filtered_df.reset_index(level=1)

filtered_df.index = filtered_df.index+pd.DateOffset(1)

filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

dates = filtered_df.index.get_level_values('date').unique().tolist()

fixed_dates = {}

for d in dates:
    
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
    
fixed_dates

# convert fixed_dates to a pandas DataFrame
# Identify the last key in the dictionary
last_key = list(fixed_dates.keys())[-1]

# Extract the data associated with the last key
last_column_data = fixed_dates[last_key]

# Create a pandas DataFrame from the extracted data
watchlist = pd.DataFrame({last_key: last_column_data})
watchlist.columns.values[0] = 'Symbol'
watchlist['Exchange'] = 'NASDAQ'
watchlist['Type'] = 'Stock'

watchlist.to_csv('watchlist.csv', index=False)


# Display the DataFrame
print("This is the watchlist ==== ", watchlist)



# In[19]:


# Step 1: Extract the last entry
# Get the last key based on maximum date
last_date = max(fixed_dates.keys())
last_date_stocks = fixed_dates[last_date]

# Step 2: Copy the values to a list (already in list form, directly use it)
stocks_to_save = last_date_stocks

# Step 3: Save the list to a CSV file
csv_filename = 'stocks_list.csv'
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Stocks'])  # Writing a header, optional
    for stock in stocks_to_save:
        writer.writerow([stock])  # Each stock in its own row

print(f"Data from the last date {last_date} has been saved to {csv_filename}.")


# ### Define portfolio optimization function
# 
# * We will define a function which optimizes portfolio weights using PyPortfolioOpt package and EfficientFrontier optimizer to maximize the sharpe ratio.
# 
# * To optimize the weights of a given portfolio we would need to supply last 1 year prices to the function.
# 
# * Apply signle stock weight bounds constraint for diversification (minimum half of equaly weight and maximum 10% of portfolio).

# In[20]:


from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def optimize_weights(prices, lower_bound=0):
    
    returns = expected_returns.mean_historical_return(prices=prices,
                                                      frequency=252)
    
    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)
    
    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, .1),
                           solver='SCS')
    
    weights = ef.max_sharpe()
    
    return ef.clean_weights()


# * Download Fresh Daily Prices Data only for short listed stocks.

# In[21]:


stocks = data.index.get_level_values('ticker').unique().tolist()

new_df = yf.download(tickers=stocks,
                     start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                     end=data.index.get_level_values('date').unique()[-1])

print("This is new_def", new_df)


# * Calculate daily returns for each stock which could land up in our portfolio.
# 
# * Then loop over each month start, select the stocks for the month and calculate their weights for the next month.
# 
# * If the maximum sharpe ratio optimization fails for a given month, apply equally-weighted weights.
# 
# * Calculated each day portfolio return.

# In[22]:


returns_dataframe = np.log(new_df['Adj Close']).diff()

portfolio_df = pd.DataFrame()

for start_date in fixed_dates.keys():
    
    try:

        end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')

        cols = fixed_dates[start_date]

        optimization_start_date = (pd.to_datetime(start_date)-pd.DateOffset(months=12)).strftime('%Y-%m-%d')

        optimization_end_date = (pd.to_datetime(start_date)-pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        
        optimization_df = new_df[optimization_start_date:optimization_end_date]['Adj Close'][cols]
        
        success = False
        try:
            weights = optimize_weights(prices=optimization_df,
                                   lower_bound=round(1/(len(optimization_df.columns)*2),3))

            weights = pd.DataFrame(weights, index=pd.Series(0))
            
            success = True
        except:
            print(f'Max Sharpe Optimization failed for {start_date}, Continuing with Equal-Weights')
        
        if success==False:
            weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))],
                                     index=optimization_df.columns.tolist(),
                                     columns=pd.Series(0)).T
        
        temp_df = returns_dataframe[start_date:end_date]

        temp_df = temp_df.stack().to_frame('return').reset_index(level=0)\
                   .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True),
                          left_index=True,
                          right_index=True)\
                   .reset_index().set_index(['Date', 'index']).unstack().stack()

        temp_df.index.names = ['date', 'ticker']

        temp_df['weighted_return'] = temp_df['return']*temp_df['weight']

        temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')

        portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)
    
    except Exception as e:
        print(e)

portfolio_df = portfolio_df.drop_duplicates()

print(portfolio_df)


# ## 8. Visualize Portfolio returns and compare to SP500 returns.

# In[23]:


# spy = yf.download(tickers='SPY',
#                   start='2015-01-01',
#                   end=dt.date.today())

# spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close':'SPY Buy&Hold'}, axis=1)

# portfolio_df = portfolio_df.merge(spy_ret,
#                                   left_index=True,
#                                   right_index=True)

# portfolio_df


# # In[24]:


# import matplotlib.ticker as mtick

# plt.style.use('ggplot')

# portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum())-1

# portfolio_cumulative_return[:'2023-09-29'].plot(figsize=(16,6))

# # Plot the SPY returns
# plt.title('Unsupervised Learning Trading Strategy Returns Over Time')

# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

# plt.ylabel('Return')

# plt.show() # Hide plots by arre
