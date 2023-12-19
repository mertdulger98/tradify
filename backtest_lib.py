import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
import datetime
from pykalman import KalmanFilter


def getData(stocks,start,end,timeframe):
    stc = []
    for s in stocks:
        stc.append(yf.download(s,start,end, interval=timeframe)['Adj Close'].rename(f'{s}'))
    df = pd.concat(stc,axis=1)
    return df.tz_localize(None)

def calc_zscore(df,window,kalman=True):
    s1_name = df.iloc[:, 0].name
    s2_name = df.iloc[:, 1].name

    S1 = df[s1_name]
    S2 = df[s2_name]

    S1 = np.log(S1)
    S2 = np.log(S2)
    S1 = sm.add_constant(S1)
    results = sm.OLS(S2, S1).fit()
    S1 = S1[s1_name]
    b = results.params[s1_name]
    spread = S2 - b * S1
    if kalman:
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=0,
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01,
        )
        state_means, state_cov = kf.filter(spread)
        state_means, state_std = state_means.squeeze(), np.std(
            state_cov.squeeze()
        )
        ma = spread.rolling(window=window, center=False).mean()
        zsc = (ma - state_means) / state_std
    else:
        zsc = (spread - spread.mean()) / np.std(spread)
    return zsc[-1]

# bunu rolling yapmak lazım

def sign(
        pair,
        start_date,
        end_date,
        interval='1d',
        window: int = 10,
        bulk_days: int= 10,
):
    df = getData(stocks=pair,start=start_date,end=end_date,timeframe=interval)
    dt = []
    dz = []
    for t in range(bulk_days+window,len(df)):
        dy = df.iloc[:t]
        dt.append(dy.iloc[-1])
        dz.append(calc_zscore(dy,window=window))


    data = pd.DataFrame(dt)
    data['zscore'] = dz
    return data

def bt_regular(df,zscore_treshold=1,shifted_signal=True):
    s1 = df.columns[0]
    s2 = df.columns[1]

    df[f'{s1}_returns'] = df[s1].pct_change()
    df[f'{s2}_returns'] = df[s2].pct_change()

    df['EntrySignal'] = np.nan
    df['ExitSignal'] = np.nan
    df[f'{s1}_position'] = np.nan
    df[f'{s2}_position'] = np.nan

    df[f'{s1}_return_nom'] = 0.0
    df[f'{s2}_return_nom'] = 0.0
    df[f'{s1}_return_cum'] = 1.0
    df[f'{s2}_return_cum'] = 1.0

    df = df.reset_index()
    current_position = None
    df['z_shift'] = df['zscore'].shift(1)
    for index, row in df.iterrows():
        zs = row['zscore']
        zs_p = row['z_shift']

        if zs <zscore_treshold and zs_p >= zscore_treshold and current_position is None:
            current_position = 'long'
            if shifted_signal:
                df.at[index + 1, f'{s1}_position'] = 'long'
                df.at[index + 1, f'{s2}_position'] = 'short'
            else:
                df.at[index, f'{s1}_position'] = 'long'
                df.at[index, f'{s2}_position'] = 'short'
            df.at[index, 'EntrySignal'] = 'LongEntry'


        elif zs > -zscore_treshold and zs_p <= -zscore_treshold and current_position is None:
            current_position = 'short'
            if shifted_signal:
                df.at[index + 1, f'{s1}_position'] = 'short'
                df.at[index + 1, f'{s2}_position'] = 'long'
            else:
                df.at[index, f'{s1}_position'] = 'short'
                df.at[index, f'{s2}_position'] = 'long'
            df.at[index, 'EntrySignal'] = 'ShortEntry'

        if ((np.abs(zs) < 0.01) or ((zs_p > 0) and (zs < 0)) or (
                (zs_p < 0) and (zs > 0))) and current_position is not None:
            df.at[index, 'ExitSignal'] = 'Exit'
            df.at[index, f'{s1}_position'] = 'cover'
            df.at[index, f'{s2}_position'] = 'cover'
            current_position = None

    df[f'{s1}_position'] = df[f'{s1}_position'].fillna(method='ffill')
    df[f'{s2}_position'] = df[f'{s2}_position'].fillna(method='ffill')

    for i in range(len(df)):
        if df.iloc[i][f'{s1}_position'] == 'long' and df.iloc[i][f'{s2}_position'] == 'short':
            df.at[i, f'{s1}_return_nom'] = df.iloc[i - 1][f'{s1}_return_nom'] + df.iloc[i][f'{s1}_returns']
            df.at[i, f'{s2}_return_nom'] = df.iloc[i - 1][f'{s2}_return_nom'] - df.iloc[i][f'{s2}_returns']
            df.at[i, f'{s1}_return_cum'] = df.iloc[i - 1][f'{s1}_return_cum'] + df.iloc[i][f'{s1}_returns']
            df.at[i, f'{s2}_return_cum'] = df.iloc[i - 1][f'{s2}_return_cum'] - df.iloc[i][f'{s2}_returns']
        elif df.iloc[i][f'{s1}_position'] == 'short' and df.iloc[i][f'{s2}_position'] == 'long':
            df.at[i, f'{s1}_return_nom'] = df.iloc[i - 1][f'{s1}_return_nom'] - df.iloc[i][f'{s1}_returns']
            df.at[i, f'{s2}_return_nom'] = df.iloc[i - 1][f'{s2}_return_nom'] + df.iloc[i][f'{s2}_returns']
            df.at[i, f'{s1}_return_cum'] = df.iloc[i - 1][f'{s1}_return_cum'] - df.iloc[i][f'{s1}_returns']
            df.at[i, f'{s2}_return_cum'] = df.iloc[i - 1][f'{s2}_return_cum'] + df.iloc[i][f'{s2}_returns']
        else:
            df.at[i, f'{s1}_return_cum'] = df.iloc[i - 1][f'{s1}_return_cum']
            df.at[i, f'{s2}_return_cum'] = df.iloc[i - 1][f'{s2}_return_cum']

    df['nominal_return'] = df[f'{s1}_return_nom'] + df[f'{s2}_return_nom']
    df[f'{s1}_buy'] = df[f'{s1}_returns'].cumsum() + 1
    df[f'{s2}_buy'] = df[f'{s2}_returns'].cumsum() + 1
    df['cumulative_return'] = (df[f'{s1}_return_cum'] + df[f'{s2}_return_cum']) / 2
    df[f'{s1}_hold'] = df[f'{s1}'] / df[f'{s1}'].iloc[0]
    df[f'{s2}_hold'] = df[f'{s2}'] / df[f'{s2}'].iloc[0]

    return df


def bt_lag(df,z_lag=7,zscore_treshold=1,shifted_signal=True):
    s1 = df.columns[0]
    s2 = df.columns[1]

    df[f'{s1}_returns'] = df[s1].pct_change()
    df[f'{s2}_returns'] = df[s2].pct_change()

    df['EntrySignal'] = np.nan
    df['ExitSignal'] = np.nan
    df[f'{s1}_position'] = np.nan
    df[f'{s2}_position'] = np.nan

    df[f'{s1}_return_nom'] = 0.0
    df[f'{s2}_return_nom'] = 0.0
    df[f'{s1}_return_cum'] = 1.0
    df[f'{s2}_return_cum'] = 1.0

    df = df.reset_index()
    current_position = 'short'
    df[f'z_lag_{z_lag}']=df['zscore'].shift(z_lag)
    for index, row in df.iterrows():
        zs = row['zscore']
        zs_p = row[f'z_lag_{z_lag}']

        if zs > zs_p and current_position == 'short':
            current_position = 'long'
            if shifted_signal:
                df.at[index + 1, f'{s1}_position'] = 'long'
                df.at[index + 1, f'{s2}_position'] = 'short'
            else:
                df.at[index, f'{s1}_position'] = 'long'
                df.at[index, f'{s2}_position'] = 'short'
            df.at[index, 'EntrySignal'] = 'LongEntry'
            df.at[index, 'ExitSignal'] = 'ShortCover'


        elif zs < zs_p and current_position == 'long':
            current_position = 'short'
            if shifted_signal:
                df.at[index + 1, f'{s1}_position'] = 'short'
                df.at[index + 1, f'{s2}_position'] = 'long'
            else:
                df.at[index, f'{s1}_position'] = 'short'
                df.at[index, f'{s2}_position'] = 'long'
            df.at[index, 'EntrySignal'] = 'ShortEntry'
            df.at[index, 'ExitSignal'] = 'LongCover'

        # if ((np.abs(zs) < 0.01) or ((zs_p > 0) and (zs < 0)) or (
        #         (zs_p < 0) and (zs > 0))) and current_position is not None:
        #     df.at[index, 'ExitSignal'] = 'Exit'
        #     df.at[index, f'{s1}_position'] = 'cover'
        #     df.at[index, f'{s2}_position'] = 'cover'
        #     current_position = None

    df[f'{s1}_position'] = df[f'{s1}_position'].fillna(method='ffill')
    df[f'{s2}_position'] = df[f'{s2}_position'].fillna(method='ffill')

    for i in range(len(df)):
        if df.iloc[i][f'{s1}_position'] == 'long' and df.iloc[i][f'{s2}_position'] == 'short':
            df.at[i, f'{s1}_return_nom'] = df.iloc[i - 1][f'{s1}_return_nom'] + df.iloc[i][f'{s1}_returns']
            df.at[i, f'{s2}_return_nom'] = df.iloc[i - 1][f'{s2}_return_nom'] - df.iloc[i][f'{s2}_returns']
            df.at[i, f'{s1}_return_cum'] = df.iloc[i - 1][f'{s1}_return_cum'] + df.iloc[i][f'{s1}_returns']
            df.at[i, f'{s2}_return_cum'] = df.iloc[i - 1][f'{s2}_return_cum'] - df.iloc[i][f'{s2}_returns']
        elif df.iloc[i][f'{s1}_position'] == 'short' and df.iloc[i][f'{s2}_position'] == 'long':
            df.at[i, f'{s1}_return_nom'] = df.iloc[i - 1][f'{s1}_return_nom'] - df.iloc[i][f'{s1}_returns']
            df.at[i, f'{s2}_return_nom'] = df.iloc[i - 1][f'{s2}_return_nom'] + df.iloc[i][f'{s2}_returns']
            df.at[i, f'{s1}_return_cum'] = df.iloc[i - 1][f'{s1}_return_cum'] - df.iloc[i][f'{s1}_returns']
            df.at[i, f'{s2}_return_cum'] = df.iloc[i - 1][f'{s2}_return_cum'] + df.iloc[i][f'{s2}_returns']
        else:
            df.at[i, f'{s1}_return_cum'] = df.iloc[i - 1][f'{s1}_return_cum']
            df.at[i, f'{s2}_return_cum'] = df.iloc[i - 1][f'{s2}_return_cum']

    df['nominal_return'] = df[f'{s1}_return_nom'] + df[f'{s2}_return_nom']
    df[f'{s1}_buy'] = df[f'{s1}_returns'].cumsum() + 1
    df[f'{s2}_buy'] = df[f'{s2}_returns'].cumsum() + 1
    df['cumulative_return'] = (df[f'{s1}_return_cum'] + df[f'{s2}_return_cum']) / 2
    df[f'{s1}_hold'] = df[f'{s1}'] / df[f'{s1}'].iloc[0]
    df[f'{s2}_hold'] = df[f'{s2}'] / df[f'{s2}'].iloc[0]

    return df

def bt_ma(df,z_ma=7,zscore_treshold=1,shifted_signal=True):
    s1 = df.columns[0]
    s2 = df.columns[1]

    df[f'{s1}_returns'] = df[s1].pct_change()
    df[f'{s2}_returns'] = df[s2].pct_change()

    df['EntrySignal'] = np.nan
    df['ExitSignal'] = np.nan
    df[f'{s1}_position'] = np.nan
    df[f'{s2}_position'] = np.nan

    df[f'{s1}_return_nom'] = 0.0
    df[f'{s2}_return_nom'] = 0.0
    df[f'{s1}_return_cum'] = 1.0
    df[f'{s2}_return_cum'] = 1.0

    df = df.reset_index()
    current_position = 'short'
    df[f'z_ma_{z_ma}']=df['zscore'].shift(z_ma)
    for index, row in df.iterrows():
        zs = row['zscore']
        zs_p = row[f'z_ma_{z_ma}']

        if zs > zs_p and current_position == 'short':
            current_position = 'long'
            if shifted_signal:
                df.at[index + 1, f'{s1}_position'] = 'long'
                df.at[index + 1, f'{s2}_position'] = 'short'
            else:
                df.at[index, f'{s1}_position'] = 'long'
                df.at[index, f'{s2}_position'] = 'short'
            df.at[index, 'EntrySignal'] = 'LongEntry'
            df.at[index, 'ExitSignal'] = 'ShortCover'


        elif zs < zs_p and current_position == 'long':
            current_position = 'short'
            if shifted_signal:
                df.at[index + 1, f'{s1}_position'] = 'short'
                df.at[index + 1, f'{s2}_position'] = 'long'
            else:
                df.at[index, f'{s1}_position'] = 'short'
                df.at[index, f'{s2}_position'] = 'long'
            df.at[index, 'EntrySignal'] = 'ShortEntry'
            df.at[index, 'ExitSignal'] = 'LongCover'

        # if ((np.abs(zs) < 0.01) or ((zs_p > 0) and (zs < 0)) or (
        #         (zs_p < 0) and (zs > 0))) and current_position is not None:
        #     df.at[index, 'ExitSignal'] = 'Exit'
        #     df.at[index, f'{s1}_position'] = 'cover'
        #     df.at[index, f'{s2}_position'] = 'cover'
        #     current_position = None

    df[f'{s1}_position'] = df[f'{s1}_position'].fillna(method='ffill')
    df[f'{s2}_position'] = df[f'{s2}_position'].fillna(method='ffill')

    for i in range(len(df)):
        if df.iloc[i][f'{s1}_position'] == 'long' and df.iloc[i][f'{s2}_position'] == 'short':
            df.at[i, f'{s1}_return_nom'] = df.iloc[i - 1][f'{s1}_return_nom'] + df.iloc[i][f'{s1}_returns']
            df.at[i, f'{s2}_return_nom'] = df.iloc[i - 1][f'{s2}_return_nom'] - df.iloc[i][f'{s2}_returns']
            df.at[i, f'{s1}_return_cum'] = df.iloc[i - 1][f'{s1}_return_cum'] + df.iloc[i][f'{s1}_returns']
            df.at[i, f'{s2}_return_cum'] = df.iloc[i - 1][f'{s2}_return_cum'] - df.iloc[i][f'{s2}_returns']
        elif df.iloc[i][f'{s1}_position'] == 'short' and df.iloc[i][f'{s2}_position'] == 'long':
            df.at[i, f'{s1}_return_nom'] = df.iloc[i - 1][f'{s1}_return_nom'] - df.iloc[i][f'{s1}_returns']
            df.at[i, f'{s2}_return_nom'] = df.iloc[i - 1][f'{s2}_return_nom'] + df.iloc[i][f'{s2}_returns']
            df.at[i, f'{s1}_return_cum'] = df.iloc[i - 1][f'{s1}_return_cum'] - df.iloc[i][f'{s1}_returns']
            df.at[i, f'{s2}_return_cum'] = df.iloc[i - 1][f'{s2}_return_cum'] + df.iloc[i][f'{s2}_returns']
        else:
            df.at[i, f'{s1}_return_cum'] = df.iloc[i - 1][f'{s1}_return_cum']
            df.at[i, f'{s2}_return_cum'] = df.iloc[i - 1][f'{s2}_return_cum']

    df['nominal_return'] = df[f'{s1}_return_nom'] + df[f'{s2}_return_nom']
    df[f'{s1}_buy'] = df[f'{s1}_returns'].cumsum() + 1
    df[f'{s2}_buy'] = df[f'{s2}_returns'].cumsum() + 1
    df['cumulative_return'] = (df[f'{s1}_return_cum'] + df[f'{s2}_return_cum']) / 2
    df[f'{s1}_hold'] = df[f'{s1}'] / df[f'{s1}'].iloc[0]
    df[f'{s2}_hold'] = df[f'{s2}'] / df[f'{s2}'].iloc[0]

    return df

def latest_signal(df):
    signal = {}
    lts = df.dropna(subset=['EntrySignal']).iloc[-1]
    signal['Date']=lts['index']
    signal['latest_entry']= lts['EntrySignal']
    signal['latest_exit'] = lts['ExitSignal']

    return signal

