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


# Class for PairsTrading
class PairsTrading:
    def __init__(self,
                 start: str = "2020-01-01",
                 end: str = datetime.datetime.today().strftime("%Y-%m-%d"),
                 commision: float = 0.0,
                 scope: str = "equity",
                 start_balance: float = 100000.0,
                 window: int = 1,
                 approach: str = "spread", # This could be ratio or spread,
                 shifted_signal: bool = True,
                 zscore_treshold: float = 1.0,
                 timeframe: str = "1d",
                 ):
        self.start = start
        self.end = end
        self.commision = commision
        self.scope = scope
        self.start_balance = start_balance
        self.window = window
        self.approach = approach
        self.shifted_signal = shifted_signal
        self.zscore_treshold = zscore_treshold
        self.timeframe = timeframe
        self.stc = ['AEFES', 'AKSEN', 'AKBNK', 'ARCLK', 'ASELS', 'BIMAS',
                    'DOHOL', 'ECILC', 'EKGYO', 'ENJSA', 'EREGL', 'FROTO', 'GARAN', 'GUBRF',
                    'HALKB', 'HEKTS', 'ISGYO', 'ISCTR', 'KARSN', 'KCHOL', 'KOZAA', 'KOZAL',
                    'KRDMD', 'MGROS', 'ODAS', 'OYAKC', 'PETKM', 'PGSUS', 'SAHOL', 'SASA',
                    'SISE', 'SKBNK', 'SOKM', 'TAVHL', 'TCELL', 'THYAO', 'TKFEN', 'TOASO',
                    'TSKB', 'TTKOM', 'TUPRS', 'VAKBN', 'VESTL', 'YKBNK']
        self.stc = [x + '.IS' for x in self.stc]

    def getData(self, stocks):
        df = yf.download(stocks, self.start, self.end,interval=self.timeframe)["Adj Close"]
        return df

    def findPairs(self, data):
        n = data.shape[1]
        score_matrix = np.zeros((n, n))
        pvalue_matrix = np.ones((n, n))
        keys = data.keys()
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                S1 = data[keys[i]]
                S2 = data[keys[j]]
                result = coint(S1, S2)
                score = result[0]
                pvalue = result[1]
                score_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
                if pvalue < 0.05:
                    pairs.append((keys[i], keys[j]))
        pvalue_matrix_df = pd.DataFrame(pvalue_matrix, index=keys, columns=keys)
        return score_matrix, pvalue_matrix, pairs, pvalue_matrix_df

    def getPairs(self, stocks):
        data = self.getData(stocks)
        score_matrix, pvalue_matrix, pairs, pvalue_matrix_df = self.findPairs(data)
        sns.heatmap(
            pvalue_matrix,
            xticklabels=stocks,
            yticklabels=stocks,
            cmap="RdYlGn_r",
            mask=(pvalue_matrix >= 0.05),
        )
        return pairs

    def backtest(self, pair):
        details = {}
        s1 = pair[0]
        s2 = pair[1]
        df = self.getData([s1, s2])

        S1 = df[s1]
        S2 = df[s2]

        score, pvalue, _ = coint(S1, S2)
        details['coinegration'] = pvalue

        if self.approach == "spread":
            S1 = sm.add_constant(S1)
            results = sm.OLS(S2, S1).fit()
            S1 = S1[s1]
            b = results.params[s1]
            spread = S2 - b * S1
            zsc = self.zscore(spread)

        if self.approach == "log":
            S1 = np.log(S1)
            S2 = np.log(S2)
            S1 = sm.add_constant(S1)
            results = sm.OLS(S2, S1).fit()
            S1 = S1[s1]
            b = results.params[s1]
            spread = S2 - b * S1
            zsc = self.zscore(spread)

        elif self.approach == "ratio":
            ratio = S1 / S2
            zsc = self.zscore(ratio)

        elif self.approach == "kalman":
            kf = KalmanFilter(
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_mean=0,
                initial_state_covariance=1,
                observation_covariance=1,
                transition_covariance=0.01,
            )
            state_means, state_cov = kf.filter(df[s1] / df[s2])
            state_means, state_std = state_means.squeeze(), np.std(
                state_cov.squeeze()
            )
            ma = (df[s1] / df[s2]).rolling(window=self.window, center=False).mean()
            zsc = (ma - state_means) / state_std

        df['zscore'] = zsc
        df['z_shift'] = df['zscore'].shift(1)

        df[f'{s1}_returns']=df[s1].pct_change()
        df[f'{s2}_returns']=df[s2].pct_change()

        df['EntrySignal'] = np.nan
        df['ExitSignal'] = np.nan
        df[f'{s1}_position'] = np.nan
        df[f'{s2}_position'] = np.nan

        df[f'{s1}_return_nom']=0.0
        df[f'{s2}_return_nom']=0.0
        df[f'{s1}_return_cum']=1.0
        df[f'{s2}_return_cum']=1.0

        df=df.reset_index()
        current_position = None
        for index, row in df.iterrows():
            zs = row['zscore']
            zs_p = row['z_shift']

            if zs < self.zscore_treshold and zs_p >= self.zscore_treshold and current_position is None:
                current_position = 'long'
                if self.shifted_signal:
                    df.at[index+1,f'{s1}_position'] = 'long'
                    df.at[index+1,f'{s2}_position'] = 'short'
                else:
                    df.at[index,f'{s1}_position'] = 'long'
                    df.at[index,f'{s2}_position'] = 'short'
                df.at[index, 'EntrySignal'] = 'LongEntry'


            elif zs > -self.zscore_treshold and zs_p <= -self.zscore_treshold and current_position is None:
                current_position = 'short'
                if self.shifted_signal:
                    df.at[index+1,f'{s1}_position'] = 'short'
                    df.at[index+1,f'{s2}_position'] = 'long'
                else:
                    df.at[index,f'{s1}_position'] = 'short'
                    df.at[index,f'{s2}_position'] = 'long'
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


        df['nominal_return']=df[f'{s1}_return_nom']+df[f'{s2}_return_nom']
        df[f'{s1}_buy']=df[f'{s1}_returns'].cumsum() + 1
        df[f'{s2}_buy']=df[f'{s2}_returns'].cumsum() + 1
        df['cumulative_return'] = (df[f'{s1}_return_cum'] + df[f'{s2}_return_cum']) / 2
        df[f'{s1}_hold']=df[f'{s1}']/df[f'{s1}'].iloc[0]
        df[f'{s2}_hold']=df[f'{s2}']/df[f'{s2}'].iloc[0]

        return df

    def show(self, pair):

        s1 = pair[0]
        s2 = pair[1]
        df = self.getData([s1, s2])

        c1 = self.calc_spread(df, s1, s2)
        c2 = self.calc_spread(df, s2, s1)

        return c1, c2

    def calc_spread(self,df, s1,s2):

        df = self.getData([s1, s2])
        S1 = df[s1]
        S2 = df[s2]

        S1 = np.log(S1)
        S2 = np.log(S2)
        S1 = sm.add_constant(S1)
        results = sm.OLS(S2, S1).fit()
        S1 = S1[s1]
        b = results.params[s1]
        spread = S2 - b * S1
        zsc = self.zscore(spread)
        df['zscore'] = zsc

        return df

    def cacl_calman(self, df, s1, s2):

        kf1 = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=0,
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01,
        )
        state_means, state_cov = kf1.filter(df[s1] / df[s2])
        state_means, state_std = state_means.squeeze(), np.std(
            state_cov.squeeze()
        )
        ma = (df[s1] / df[s2]).rolling(window=self.window, center=False).mean()
        zsc = (ma - state_means) / state_std
        df['zscore'] = zsc

        return df

    def zscore(self, series):
        return (series - series.mean()) / np.std(series)

    def backtestExternalSingle(self, pair):
        spt = StockPairsTrading(start=self.start,
                                end=self.end,
                                window=self.window)
        bt = spt.backtest(pair)
        return bt

    def backtestExternalMultiple(self, pairs):
        spt = StockPairsTrading(start=self.start,
                                end=self.end,
                                window=self.window)
        data = {}
        for pair in pairs:
            str_p = pair[0] + '-' + pair[1]
            bt = spt.backtest(pair)
            data[str_p] = bt
        return pd.DataFrame().from_dict(data, orient='index')
