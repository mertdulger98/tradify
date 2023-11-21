import numpy as np
import pandas as pd
import yfinance as yf
import math
import matplotlib.pyplot as plt
import datetime

today = datetime.datetime.today().strftime("%Y-%m-%d")
import os


def getData(stockName, period, interval):
    return yf.download(tickers=stockName, period=period, interval=interval)


def calc(tick, per, inter, shf):
    if inter == '1d':
        df = getData(tick, per, inter).reset_index().drop(columns=['Date', 'Close']).rename(
            columns={'Adj Close': 'Close'})
    else:
        df = getData(tick, per, inter).reset_index().drop(columns=['Datetime', 'Close']).rename(
            columns={'Adj Close': 'Close'})

    df['r_var'] = df['Close'].rolling(30).var()
    df['Volatility'] = np.sqrt(df['r_var'])
    df['Range'] = np.sqrt(30 / 365) * df['Volatility']
    df['RH'] = (df['Close'] + df['Range']).shift(shf)
    df['RL'] = (df['Close'] - df['Range']).shift(shf)

    ax = df[['Close', 'RH', 'RL']][-60:].plot(figsize=(15, 6))

    ax.get_lines()[0].set_color('yellow')  # Change the color of 'Value1' to red
    ax.get_lines()[1].set_color('green')
    ax.get_lines()[2].set_color('red')

    ax.set_title(f'{tick}-{inter}')
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.set_facecolor('#222222')
    ax.tick_params(axis='both', colors='black')
    ax.legend(loc='upper left')
    plt.savefig(f"recs/{today}/{tick}-{inter}.png")


stc = ['XU030', 'XU100', 'TOASO', 'KCHOL', 'PETKM', 'THYAO', 'BIMAS', 'FROTO', 'ASELS', 'TUPRS', 'SAHOL', 'TCELL',
       'SISE', 'KOZAA', 'SOKM']
stc = [x + '.IS' for x in stc]

t = {
    '1d': '6mo',
    '1h': '60d',
    '15m': '30d'
}
sh = {
    '1d': 3,
    '1h': 5,
    '15m': 7
}

os.mkdir(f"recs/{today}")
for s in stc:
    for v, k in t.items():
        print(s, k, v, sh[v])
        calc(s, k, v, sh[v])
