import numpy as np
import pandas as pd
import yfinance as yf
import math
import matplotlib.pyplot as plt
import datetime
import plotly.express as px

today = datetime.datetime.today().strftime("%Y-%m-%d")
import os
import streamlit as st


def getData(stockName, period, interval):
    if interval == '1d':
        dtt = 'Date'
    else:
        dtt = 'Datetime'
    df = yf.download(tickers=stockName, period=period, interval=interval).reset_index().drop(columns=['Close']).rename(
        columns={'Adj Close': 'Close', dtt: 'Date'})

    return df


def calc(tick, per, inter, mov, window):
    df = getData(tick, per, inter)
    df['r_var'] = df['Close'].rolling(mov).var()
    df['Volatility'] = np.sqrt(df['r_var'])
    df['Range'] = np.sqrt(mov / 365) * df['Volatility']
    df['RH'] = (df['Close'] + df['Range']).shift(window)
    df['RL'] = (df['Close'] - df['Range']).shift(window)

    return df


tfs = {
    '1d': '6mo',
    '1h': '60d',
    '15m': '30d'
}

st.set_page_config(layout="wide",
                   page_icon=":sunglasses:",
                   initial_sidebar_state="collapsed",
                   # theme="dark"
                   )
st.markdown("""
<style>
body {
    background-color: #121212; /* Dark background color */
    color: #FFFFFF; /* Text color for contrast */
}
</style>
""", unsafe_allow_html=True)

st.title("Bist Volatility App")

col1, col2 = st.columns((1, 4))

with col1:
    tick = st.text_input("Hisse", "XU100")
    tick = tick.upper() + '.IS'

    intrv = st.selectbox(
        "Zaman Aralığı",
        ('1d', '1h', '15m')
    )

    period = tfs[intrv]

    ma_w = st.number_input("Hareketli Ortalama", value=30)
    sf_w = st.number_input("Kaydırma Aralığı", value=3)

    typ = st.radio(
        "Grafik Tipi",
        ['Line', 'Scatter']
    )

    df = calc(tick, period, intrv, ma_w, sf_w)

with col2:
    color_map = {
        'Close': 'yellow',
        'RH': 'green',
        'RL': 'red'
    }
    if typ == 'Line':
        fig = px.line(df[-60:], x='Date', y=['Close', 'RH', 'RL'], markers=True, color_discrete_map=color_map)


    elif typ == 'Scatter':
        fig = px.scatter(df[-45:], x='Date', y=['Close', 'RH', 'RL'])

    if intrv != '1d':
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(bounds=[18, 9], pattern="hour")
            ])

    fig.update_layout(xaxis_showspikes=True,
                      hovermode='x'
                      # 'x unified'
                      )

    st.plotly_chart(fig, use_container_width=True)
