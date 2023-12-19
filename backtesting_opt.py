import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import statsmodels.api as sm
import datetime
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import coint
from stock_pairs_trading import StockPairsTrading
import optuna


def getData(stocks, start_date, end_date, timeframe):
    stc = []
    for s in stocks:
        stc.append(yf.download(s, start=start_date, end=end_date, interval=timeframe)['Adj Close'].rename(f'{s}'))
    df = pd.concat(stc, axis=1)
    return df.tz_localize(None)


def fit_ols(df):
    s1_name = df.iloc[:, 0].name
    s2_name = df.iloc[:, 1].name

    S1 = df[s1_name]
    S2 = df[s2_name]
    S1_log = np.log(S1)
    S2_log = np.log(S2)
    S1_log = sm.add_constant(S1_log)
    results = sm.OLS(S2_log, S1_log).fit()

    return results


def calculate_spread(new_df, fitted_model):
    """
    Calculates the spread for new data points using the fitted OLS model.

    :param new_S1: New data points for Series 1 (independent variable)
    :param new_S2: New data points for Series 2 (dependent variable)
    :param fitted_model: Fitted OLS model from 'fit_ols_model' function
    :param s1_name: Name of the S1 series
    :return: Calculated spread
    """

    s1_name = new_df.iloc[:, 0].name
    s2_name = new_df.iloc[:, 1].name

    new_S1 = new_df[s1_name]
    new_S2 = new_df[s2_name]

    new_S1_log = np.log(new_S1)
    new_S2_log = np.log(new_S2)
    new_S1_log = sm.add_constant(new_S1_log)
    b = fitted_model.params[s1_name]

    spread = new_S2_log - b * new_S1_log[s1_name]

    return spread


def calc_kalman(spread, window):
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01,
    ) # moving residualı mean ve stdyi initiallar olarak verebilirsin.

    state_means, state_cov = kf.filter(spread)
    state_means, state_std = state_means.squeeze(), np.std(
        state_cov.squeeze()
    )
    ma = spread.rolling(window=window, center=False).mean()
    zsc = (ma - state_means) / state_std
    return zsc


def bt_ma(df, z_ma, kalman_ma, fitted_model, shifted_signal=True):
    zsc = calc_kalman(calculate_spread(df, fitted_model), kalman_ma)
    df['zscore'] = zsc

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
    df[f'z_ma_{z_ma}'] = df['zscore'].shift(z_ma)
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
    df = df.dropna(subset=df.columns[0])
    return df


def evaluate_backtest(bt_data):
    st1 = bt_data.columns[1]
    st2 = bt_data.columns[2]
    results = {}
    results['cumulative_return'] = bt_data['cumulative_return'].iloc[-1]
    results['max_return'] = bt_data['cumulative_return'].max()
    results['max_drawdown'] = bt_data['cumulative_return'].min()
    # win_trades, lose_trades, num_trades
    return results


def objective(trial, df, fitted_model):
    z_ma = trial.suggest_int('z_ma', 7, 14)
    kalman_ma = trial.suggest_int('kalman_ma', 7, 14)

    obj_df = bt_ma(df, z_ma, kalman_ma, fitted_model)
    score = obj_df.iloc[-1]['cumulative_return']
    if (score == np.nan) or (score is None):
        score = 1.0
    return score


def tune_parameters(df, fitted_model):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, df, fitted_model), n_trials=100)
    return study


def latest_signal(pair1,pair2,freq,z_ma,kalman_ma,model):
    signal = {}
    last_day = datetime.datetime.today().strftime("%Y-%m-%d")
    start_day = (datetime.datetime.today() - datetime.timedelta(days=120)).strftime("%Y-%m-%d")
    df = getData([pair1,pair2],start_day,last_day,freq)
    sg_df = bt_ma(df,z_ma,kalman_ma,model)
    lts = sg_df.dropna(subset=['EntrySignal']).iloc[-1]
    signal['Pair1']=pair1
    signal['Pair2']=pair2
    signal['Date']=lts[0]
    signal['latest_entry']= lts['EntrySignal']
    signal['latest_exit'] = lts['ExitSignal']

    return signal