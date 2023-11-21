import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

bist100 = ['AEFES', 'AGHOL', 'AKBNK', 'AKCNS', 'AKSA', 'AKSEN', 'ALARK',
           'ALGYO', 'ALKIM', 'ARCLK', 'ARDYZ', 'ASELS', 'AYDEM', 'ALBRK',
           'AYGAZ', 'BERA', 'BIMAS', 'BIOEN', 'BRISA', 'CANTE', 'CCOLA', 'CEMTS',
           'CIMSA', 'DEVA', 'DOAS', 'DOHOL', 'ECILC', 'EGEEN', 'EKGYO', 'ENJSA', 'ENKAI',
           'ERBOS', 'EREGL', 'ESEN', 'ESEN', 'FROTO', 'GARAN', 'GLYHO', 'GLYHO', 'GLYHO',
           'GOZDE', 'GUBRF', 'HALKB', 'HEKTS', 'HEKTS', 'HEKTS', 'HLGYO', 'INDES', 'ISCTR',
           'ISDMR', 'ISFIN', 'ISGYO', 'ISMEN', 'IZMDC', 'IZMDC', 'IZMDC', 'KARSN', 'KARTN',
           'KCHOL', 'KERVT', 'KORDS', 'KOZAA', 'KOZAL', 'KRDMD', 'KRVGD', 'LOGO', 'MAVI', 'MGROS',
           'MPARK', 'NETAS', 'ODAS', 'ODAS', 'ODAS', 'OTKAR', 'OYAKC', 'PARSN', 'PETKM', 'PGSUS',
           'QUAGR', 'QUAGR', 'SAHOL', 'SARKY', 'SASA', 'SELEC', 'SISE', 'SKBNK', 'SKBNK', 'SOKM',
           'TAVHL', 'TCELL', 'THYAO', 'TKFEN', 'TKNSA', 'TKNSA', 'TKNSA', 'TMSN', 'TOASO',
           'TRGYO', 'TRILC', 'TSKB', 'TTKOM', 'TTRAK', 'TUPRS', 'TURSG', 'ULKER', 'VAKBN',
           'VERUS', 'VESBE', 'VESTL', 'YATAS', 'YKBNK', 'ZOREN', 'ZOREN', 'ZRGYO']
bist100 = [s + '.IS' for s in bist100]

stc = ['AEFES', 'AKSEN', 'AKBNK', 'ARCLK', 'ASELS', 'BIMAS',
       'DOHOL', 'ECILC', 'EKGYO', 'ENJSA', 'EREGL', 'FROTO', 'GARAN', 'GUBRF',
       'HALKB', 'HEKTS', 'ISGYO', 'ISCTR', 'KARSN', 'KCHOL', 'KOZAA', 'KOZAL',
       'KRDMD', 'MGROS', 'ODAS', 'OYAKC', 'PETKM', 'PGSUS', 'SAHOL', 'SASA',
       'SISE', 'SKBNK', 'SOKM', 'TAVHL', 'TCELL', 'THYAO', 'TKFEN', 'TOASO',
       'TSKB', 'TTKOM', 'TUPRS', 'VAKBN', 'VESTL', 'YKBNK']
stc = [x + '.IS' for x in stc]


def getData(ticker, period='5y', interval='1d'):
    """
    :param stockName: The ticker symbol for the stock (e.g., 'AAPL' for Apple Inc.).
    :param period: The time frame for the data ('1d', '1mo', '1y', '5y', etc.). Default is '5y' for five years of data.
    :param interval: The data point frequency ('1m', '1d', '1wk', '1mo', etc.). Default is '1d' for daily data.
    :return: Returns the result dataframe for selected stock for given interval and period
    """
    return yf.download(tickers=ticker, period=period, interval=interval)

def getPairPlotClose(stock1,stock2,ts_args):
    """
    :param stock1:  stock ticker symbol for the first stock.
    :param stock2: stock ticker symbol for the second stock.
    :param ts_args:     A dictionary of arguments to be passed to the getData function.
    :return:  Plots the closing prices of the two stocks on the same plot.
    """
    getData(stock1,**ts_args).Close.plot()
    getData(stock2,**ts_args).Close.plot()
    plt.grid(True)
    plt.legend([stock1, stock2])

def getPairPlotReturn(stock1, stock2, ts_args):
    """
    :param stock1:  stock ticker symbol for the first stock.
    :param stock2:  stock ticker symbol for the second stock.
    :param ts_args:     A dictionary of arguments to be passed to the getData function.
    :return:    Plots the log returns of the two stocks on the same plot.
    """
    # Get the data and calculate returns for both stocks
    df_stock1 = calcReturn(getData(stock1, **ts_args), stock1)
    df_stock2 = calcReturn(getData(stock2, **ts_args), stock2)

    # Create a single plot with both sets of returns
    plt.figure(figsize=(10, 5))  # You can adjust the figure size to your liking
    plt.plot(df_stock1.index, df_stock1.iloc[:, 0], label=stock1)
    plt.plot(df_stock2.index, df_stock2.iloc[:, 0], label=stock2)

    # Adding grid, legend, and showing the plot
    plt.grid(True)
    plt.legend()
    plt.show()

def vizPairShifted(stock1,stock1_shift,stock2,stock2_shift,ts_args):
    """
    :param stock1:  stock ticker symbol for the first stock.
    :param stok1_shift:  The number of days to shift the first stock's log returns.
    :param stock2:  stock ticker symbol for the second stock.
    :param stock2_shift:  The number of days to shift the second stock's log returns.
    :param ts_args:     A dictionary of arguments to be passed to the getData function.
    :return:    Plots the log returns of the two stocks on the same plot.
    """
    # Get the data and calculate returns for both stocks
    df_stock1 = getData(stock1, **ts_args).Close.shift(stock1_shift)
    df_stock2 = getData(stock2, **ts_args).Close.shift(stock2_shift)

# Create a single plot with both sets of returns
    plt.figure(figsize=(10, 5))  # You can adjust the figure size to your liking
    plt.plot(df_stock1.index, df_stock1, label=stock1)
    plt.plot(df_stock2.index, df_stock2, label=stock2)

# Adding grid, legend, and showing the plot
    plt.grid(True)
    plt.legend()
    plt.show()

def getShiftedCorrelation(stock1,stock1_shift,stock2,stock2_shift,ts_args):
    # Get the data and calculate returns for both stocks
    df_stock1 = calcReturn(getData(stock1, **ts_args), stock1)
    df_stock2 = calcReturn(getData(stock2, **ts_args), stock2)

    # Shift the returns in the second DataFrame by the number of days specified
    df_stock2_shift = df_stock2.shift(stock2_shift)
    df_stock1_shift = df_stock1.shift(stock1_shift)

    # Concatenate the two DataFrames and drop NaN values
    df = pd.concat([df_stock1_shift, df_stock2_shift], axis=1).dropna()
    correlation = df.corr().iloc[0, 1]
    return correlation


def findShiftCorrelations(stock1, stock2, ts_args, max_shift=10):
    """
    Calculate and store the correlation for each shift between two stocks.

    :param stock1: Ticker symbol for the first stock.
    :param stock2: Ticker symbol for the second stock.
    :param ts_args: Arguments for the getData function, packed in a dictionary.
    :param max_shift: The maximum lag to test.
    :return: A DataFrame with each shift and its corresponding correlation.
    """
    # List to store the shift and the corresponding correlation
    shift_correlations = []

    # Loop over the range of shifts
    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            # Zero shift means no lag, compare directly
            correlation = getShiftedCorrelation(stock1, 0, stock2, 0, ts_args)
        elif shift > 0:
            # Positive shift: stock1 leads, stock2 lags
            correlation = getShiftedCorrelation(stock1, 0, stock2, shift, ts_args)
        else:
            # Negative shift: stock2 leads, stock1 lags
            correlation = getShiftedCorrelation(stock1, -shift, stock2, 0, ts_args)

        # Append the shift and correlation to the list
        shift_correlations.append((shift, correlation))

    # Convert the list to a DataFrame
    correlation_df = pd.DataFrame(shift_correlations, columns=['Shift', 'Correlation'])

    return correlation_df


def getHistoricalData(ticker, start_date, end_date, interval='1d'):
    """
    Fetches historical stock data from Yahoo Finance.

    :param ticker: The stock symbol to fetch data for (e.g., 'AAPL').
    :param start_date: The starting date (inclusive) for the data in format 'YYYY-MM-DD'.
    :param end_date: The ending date (inclusive) for the data in format 'YYYY-MM-DD'.
    :param interval: The data interval. Valid intervals: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d',
     '1wk', '1mo', '3mo'.
    :return: A DataFrame with the stock data.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return stock_data

def getMultipleStocksData(tickers, start_date, end_date, interval='1d'):
    """
    Fetches historical stock data for multiple stocks from Yahoo Finance and arranges it in columns.

    :param tickers: A list of stock symbols to fetch data for (e.g., ['AAPL', 'MSFT']).
    :param start_date: The starting date (inclusive) for the data in format 'YYYY-MM-DD'.
    :param end_date: The ending date (inclusive) for the data in format 'YYYY-MM-DD'.
    :param interval: The data interval.
    :return: A DataFrame with the stock data for each ticker in separate columns.
    """
    all_data = {}

    for ticker in tickers:
        print(f"Fetching data for {ticker}")
        stock_data = getHistoricalData(ticker, start_date, end_date, interval)
        stock_data.columns = [f"{ticker}_{col}" for col in stock_data.columns]  # Prefix columns with ticker
        all_data[ticker] = stock_data

    combined_data = pd.concat(all_data.values(), axis=1)
    return combined_data

def plot_scatter_with_line(df, x_col, y_col):
    """
    Plots a scatter plot with a 1:1 line.

    :param df: DataFrame containing the data.
    :param x_col: The column name to be plotted on the x-axis.
    :param y_col: The column name to be plotted on the y-axis.
    """
    # Create the scatter plot and get the axes object
    ax = sns.scatterplot(data=df, x=x_col, y=y_col)

    # Get the current axis limits directly from the axes object
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    # Determine the limits for the 1:1 line
    line_limits = [min(x_limits[0], y_limits[0]), max(x_limits[1], y_limits[1])]

    # Draw the 1:1 line
    ax.plot(line_limits, line_limits, 'k--')  # 'k--' specifies a black dashed line

    # Set the axis limits back to the original
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    # Add labels and title if desired
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'Scatter Plot of {x_col} vs {y_col} with y=x Line')

    plt.show()



def calcReturn(df, ticker):
    """
    :param df: The DataFrame containing stock data with a 'Close' column representing closing prices.
    :param ticker: A string representing the stock ticker, used to name the new returns column.
    :return: Logaritmic returns for the selected dataframe based on close prices
    """
    df[f'Return_{ticker}'] = np.log(df['Close'] / df['Close'].shift(1))
    return df[f'Return_{ticker}'].to_frame()


def getReturn(stocks,s_args):
    """
    :param s_args:
    :param stocks:A list of stock ticker symbols for which you want to calculate log returns.
    :return: Dataframe that shows log returns of list of stocks
    The getReturn function iterates over a list of stock tickers, retrieves their price data,
    calculates log returns for each, and concatenates the results into a single DataFrame before returning it.
    """
    dfs_to_concat = []
    for stc in stocks:
        st = calcReturn(getData(stc,**s_args), stc)
        dfs_to_concat.append(st)

    df = pd.concat(dfs_to_concat, axis=1)
    df = df.dropna()
    return df


def vizClusterHierarcy(df):
    """
    :param df: the DataFrame with the data to be clustered, using its correlation matrix.
    The vizClusterHierarchy function visualizes dendrograms of hierarchical clustering using four linkage methods
    on the DataFrame's correlation matrix, displaying them in a 2x2 subplot layout.
    """
    methods = ['single', 'complete', 'average', 'ward']
    corr = df.corr()
    # Plot dendrograms for different linkage methods
    plt.figure(figsize=(20, 10))

    for i, method in enumerate(methods, 1):
        plt.subplot(2, 2, i)

        # Compute linkage matrix
        linked = linkage(corr, method=method)

        # Plot dendrogram
        dendrogram(linked, labels=corr.columns)

        plt.title(f'Hierarchical Clustering with {method.capitalize()} Linkage')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')

    plt.tight_layout()
    plt.show()

def getVariances(df):
    """
    :param pca: PCA Object
    :return:
    """
    pca = PCA(n_components=2)
    pca.fit_transform(df)
    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Displaying the explained variance
    print("Explained Variance Ratio:", explained_variance_ratio)
    print("Cumulative Explained Variance:", cumulative_variance)

def getComponentDataByStock(df):
    """
    :param pca: PCA object
    :param df: input data
    :return: Coefficients of components for each stock
    """
    pca = PCA(n_components=2)
    pca.fit_transform(df)
    # Get the loadings for each principal component (PC)
    loadings = pca.components_

    # Create a DataFrame to display the loadings
    pc_names = [f'PC{i}' for i in range(1, len(loadings) + 1)]
    loadings_df = pd.DataFrame(loadings.T, columns=pc_names, index=df.columns)
    return loadings_df

def getComponentDataByDate(data):
    """
    :param data:  input data for PCA  (stock returns)
    :return: PCA dataframe
    """
    # Conduct PCA
    pca = PCA(n_components=2)  # We want to keep the first two principal components
    principal_components = pca.fit_transform(data)

    # Create a DataFrame with the principal components and the date as index
    pca_df = pd.DataFrame(data=principal_components,
                          columns=['PC1', 'PC2'],
                          index=data.index)
    return pca_df

def visualizePcaByDate(pca_df):
    """
    :param pca_df: PCA dataframe
    :return: Plots the first two principal components over time for the selected stocks.
    """
    # Plotting the first two principal components
    plt.figure(figsize=(14, 7))

    # Plot PC1
    plt.subplot(2, 1, 1)
    plt.plot(pca_df.index, pca_df['PC1'], label='PC1')
    plt.xlabel('Date')
    plt.ylabel('PC1 Value')
    plt.title('PC1 over Time')
    plt.legend()

    # Plot PC2
    plt.subplot(2, 1, 2)
    plt.plot(pca_df.index, pca_df['PC2'], label='PC2')
    plt.xlabel('Date')
    plt.ylabel('PC2 Value')
    plt.title('PC2 over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()