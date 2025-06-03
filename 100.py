# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 13:39:25 2025

@author: student
"""

# -*- coding: utf-8 -*-
"""
金融資料視覺化看板
"""

# 載入必要模組
import os
import numpy as np
import indicator_f_Lo2_short, datetime, indicator_forKBar_short
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 
from order_streamlit import Record
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#%% 初始化設定
st.set_page_config(layout="wide", page_title="金融看板與程式交易平台")

###### 設定網頁標題介面 
html_temp = """
<div style="background-color:#3872fb;padding:10px;border-radius:10px">   
<h1 style="color:white;text-align:center;">金融看板與程式交易平台</h1>
<h2 style="color:white;text-align:center;">Financial Dashboard and Program Trading</h2>
</div>
"""
stc.html(html_temp)

# 初始化 session_state
if 'order_record' not in st.session_state:
    st.session_state.order_record = None

#%% 資料載入函數
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def load_data(path):
    df = pd.read_pickle(path)
    return df

#%% 資料轉換函數
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def To_Dictionary_1(df, product_name):
    KBar_dic = df.to_dict()
    for col in ['open', 'low', 'high', 'close', 'volume', 'amount']:
        KBar_dic[col] = np.array(list(KBar_dic[col].values()))
    
    KBar_dic['product'] = np.repeat(product_name, len(KBar_dic['open']))
    
    time_list = [t.to_pydatetime() for t in KBar_dic['time'].values()]
    KBar_dic['time'] = np.array(time_list)
    
    return KBar_dic

#%% K線週期轉換函數
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Change_Cycle(Date, cycle_duration, KBar_dic, product_name):
    KBar = indicator_forKBar_short.KBar(Date, cycle_duration)
    
    for i in range(len(KBar_dic['time'])):
        time = KBar_dic['time'][i]
        open_price = KBar_dic['open'][i]
        close_price = KBar_dic['close'][i]
        low_price = KBar_dic['low'][i]
        high_price = KBar_dic['high'][i]
        qty = KBar_dic['volume'][i]
        amount = KBar_dic['amount'][i]
        KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)
    
    new_KBar_dic = {
        'time': KBar.TAKBar['time'],
        'product': np.repeat(product_name, len(KBar.TAKBar['time'])),
        'open': KBar.TAKBar['open'],
        'high': KBar.TAKBar['high'],
        'low': KBar.TAKBar['low'],
        'close': KBar.TAKBar['close'],
        'volume': KBar.TAKBar['volume']
    }
    
    return new_KBar_dic

#%% 技術指標計算函數
@st.cache_data(ttl=3600, show_spinner="正在計算指標...")
def Calculate_MA(df, period=10):
    return df['close'].rolling(window=period).mean()

@st.cache_data(ttl=3600, show_spinner="正在計算指標...")
def Calculate_RSI(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600, show_spinner="正在計算指標...")
def Calculate_Bollinger_Bands(df, period=20, num_std_dev=2):
    df = df.copy()
    df['SMA'] = df['close'].rolling(window=period).mean()
    df['Standard_Deviation'] = df['close'].rolling(window=period).std()
    df['Upper_Band'] = df['SMA'] + (df['Standard_Deviation'] * num_std_dev)
    df['Lower_Band'] = df['SMA'] - (df['Standard_Deviation'] * num_std_dev)
    return df

@st.cache_data(ttl=3600, show_spinner="正在計算指標...")
def Calculate_MACD(df, fast_period=12, slow_period=26, signal_period=9):
    df = df.copy()
    df['EMA_Fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_Slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    return df

#%% 交易策略相關函數
def ChartOrder_MA(df, trade_record):
    # 買方下單點位紀錄
    buy_orders = [trade for trade in trade_record if trade[0] in ['Buy', 'B']]
    buy_dates = [trade[2] for trade in buy_orders]
    buy_prices = [df[df['time'] == date]['low'].values[0] * 0.999 for date in buy_dates]
    
    # 買方出場點位紀錄
    buy_covers = [trade for trade in trade_record if trade[0] in ['Sell', 'S'] and trade[4] > 0]
    cover_dates = [trade[2] for trade in buy_covers]
    cover_prices = [df[df['time'] == date]['high'].values[0] * 1.001 for date in cover_dates]
    
    # 賣方下單點位紀錄
    sell_orders = [trade for trade in trade_record if trade[0] in ['Sell', 'S'] and trade[4] < 0]
    sell_dates = [trade[2] for trade in sell_orders]
    sell_prices = [df[df['time'] == date]['high'].values[0] * 1.001 for date in sell_dates]
    
    # 賣方出場點位紀錄
    sell_covers = [trade for trade in trade_record if trade[0] in ['Buy', 'B'] and trade[4] < 0]
    cover_sell_dates = [trade[2] for trade in sell_covers]
    cover_sell_prices = [df[df['time'] == date]['low'].values[0] * 0.999 for date in cover_sell_dates]
    
    # 繪製圖表
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        yaxis=dict(fixedrange=False, autorange=True),
        xaxis=dict(rangeslider=dict(visible=True)))
    
    # K線
    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K線'), secondary_y=True)
    
    # 移動平均線
    last_valid_index = df['MA_long'].last_valid_index()
    if last_valid_index is not None:
        df_plot = df.iloc[last_valid_index+1:]
        fig.add_trace(go.Scatter(
            x=df_plot['time'],
            y=df_plot['MA_long'],
            mode='lines',
            line=dict(color='orange', width=2),
            name=f'長MA'), secondary_y=True)
        
        fig.add_trace(go.Scatter(
            x=df_plot['time'],
            y=df_plot['MA_short'],
            mode='lines',
            line=dict(color='pink', width=2),
            name=f'短MA'), secondary_y=True)
    
    # 交易點位
    if buy_dates:
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_prices,
            mode='markers',
            marker=dict(color='red', symbol='triangle-up', size=10),
            name='作多進場點'), secondary_y=True)
    
    if cover_dates:
        fig.add_trace(go.Scatter(
            x=cover_dates, y=cover_prices,
            mode='markers',
            marker=dict(color='blue', symbol='triangle-down', size=10),
            name='作多出場點'), secondary_y=True)
    
    if sell_dates:
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_prices,
            mode='markers',
            marker=dict(color='green', symbol='triangle-down', size=10),
            name='作空進場點'), secondary_y=True)
    
    if cover_sell_dates:
        fig.add_trace(go.Scatter(
            x=cover_sell_dates, y=cover_sell_prices,
            mode='markers',
            marker=dict(color='black', symbol='triangle-up', size=10),
            name='作空出場點'), secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

def ChartOrder_Trade(df, trade_record):
    fig = go.Figure()
    
    # K線
    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K線'))
    
    # 交易點位
    for trade in trade_record:
        action = trade[0]
        trade_time = trade[2]
        trade_price = trade[3]
        
        if action in ['Buy', 'B']:
            color = 'green'
            symbol = 'triangle-up'
            name = '買入點'
        elif action in ['Sell', 'S'] and trade[4] > 0:  # 平多倉
            color = 'blue'
            symbol = 'triangle-down'
            name = '賣出點'
        elif action in ['Sell', 'S'] and trade[4] < 0:  # 開空倉
            color = 'red'
            symbol = 'triangle-down'
            name = '做空點'
        elif action in ['Buy', 'B'] and trade[4] < 0:  # 平空倉
            color = 'purple'
            symbol = 'triangle-up'
            name = '回補點'
        else:
            continue
            
        fig.add_trace(go.Scatter(
            x=[trade_time], y=[trade_price],
            mode='markers',
            marker=dict(color=color, size=10, symbol=symbol),
            name=name))
    
    fig.update_layout(
        title='交易點位圖',
        xaxis_title='時間',
        yaxis_title='價格',
        xaxis_rangeslider_visible=True,
        template='plotly_white',
        height=600)
    
    st.plotly_chart(fig, use_container_width=True)

def Show_Performance(order_record):
    if order_record is None or len(order_record.GetTradeRecord()) == 0:
        st.warning("沒有交易記錄可供分析")
        return
    
    trades = order_record.GetTradeRecord()
    profits = []
    returns = []
    trade_dates = []
    
    # 計算每筆交易的盈虧
    for i in range(0, len(trades), 2):
        if i+1 >= len(trades):
            break
            
        entry = trades[i]
        exit = trades[i+1]
        
        if entry[0] in ['Buy', 'B']:
            profit = exit[3] - entry[3]
        else:
            profit = entry[3] - exit[3]
            
        returns.append(profit / entry[3])
        profits.append(profit)
        trade_dates.append(entry[2])
    
    if not profits:
        st.warning("沒有完整的交易對（進場+出場）可供分析")
        return
    
    # 基本績效指標
    total_profit = sum(profits)
    num_trades = len(profits)
    win_rate = len([p for p in profits if p > 0]) / num_trades * 100
    avg_profit = total_profit / num_trades
    max_profit = max(profits)
    max_loss = min(profits)
    
    # 最大回撤
    equity = [0]
    for p in profits:
        equity.append(equity[-1] + p)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_drawdown = max(drawdown) if len(drawdown) > 0 else 0
    
    # 顯示績效儀表板
    st.subheader("交易績效摘要")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("總利潤", f"{total_profit:.2f}")
    col2.metric("交易次數", num_trades)
    col3.metric("勝率", f"{win_rate:.2f}%")
    col4.metric("平均利潤", f"{avg_profit:.2f}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最大利潤", f"{max_profit:.2f}")
    col2.metric("最大虧損", f"{max_loss:.2f}")
    col3.metric("最大回撤", f"{max_drawdown:.2f}")
    col4.metric("累計報酬率", f"{(equity[-1] / equity[0] - 1)*100 if equity[0] != 0 else 0:.2f}%")
    
    # 繪製累計盈虧曲線
    st.subheader("累計盈虧曲線")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trade_dates, y=equity[1:],
        mode='lines+markers',
        name='累計盈虧',
        line=dict(color='royalblue', width=2)))
    
    fig.update_layout(
        xaxis_title='交易時間',
        yaxis_title='累計盈虧',
        template='plotly_white')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 顯示交易明細
    st.subheader("交易明細")
    trade_details = []
    for i in range(0, len(trades), 2):
        if i+1 >= len(trades):
            break
            
        entry = trades[i]
        exit = trades[i+1]
        
        if entry[0] in ['Buy', 'B']:
            profit = exit[3] - entry[3]
            trade_type = "多頭"
        else:
            profit = entry[3] - exit[3]
            trade_type = "空頭"
            
        trade_details.append({
            "類型": trade_type,
            "進場時間": entry[2],
            "進場價格": entry[3],
            "出場時間": exit[2],
            "出場價格": exit[3],
            "盈虧": profit
        })
    
    st.dataframe(pd.DataFrame(trade_details))

#%% 主程式開始
st.sidebar.header("設定參數")

###### 選擇金融商品
st.sidebar.subheader("選擇金融商品")
choices = ['富邦金期貨: 2023.4.15 至 2025.4.16', '華碩: 2023.4.17至2025.4.16', '聯電期貨: 2023.4.17至2025.4.16']
choice = st.sidebar.selectbox('選擇金融商品', choices, index=0)

###### 讀取資料
if choice == choices[0]:
    df_original = load_data('future_KBar_CEF2023.4.15-2025.4.16.pkl')
    product_name = '富邦金期貨'
elif choice == choices[1]:
    df_original = load_data('stock_KBar_2357 2023.4.17-2025.4.16.pkl')
    product_name = '華碩'
else:
    df_original = load_data('future_KBar_CCF 2023.4.17-2025.4.16.pkl')
    product_name = '聯電期貨'

###### 選擇資料區間
st.sidebar.subheader("選擇資料時間區間")
start_date = st.sidebar.text_input('輸入開始日期 (格式: 2023.4.15)', '2023.4.15')
end_date = st.sidebar.text_input('輸入結束日期 (格式: 2025.4.16)', '2025.4.16')

try:
    start_date = datetime.datetime.strptime(start_date, '%Y.%m.%d')
    end_date = datetime.datetime.strptime(end_date, '%Y.%m.%d')
    
    # 使用條件篩選選擇時間區間的數據
    df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]
    
    if df.empty:
        st.warning("選擇的時間區間內沒有資料，請重新選擇日期範圍")
        st.stop()
        
except ValueError:
    st.error("日期格式錯誤，請使用 'YYYY.MM.DD' 格式")
    st.stop()

###### 轉化為字典
KBar_dic = To_Dictionary_1(df, product_name)

###### 設定K棒週期
st.sidebar.subheader("設定K棒週期")
choices_unit = ['以分鐘為單位','以日為單位','以週為單位','以月為單位']
choice_unit = st.sidebar.selectbox('選擇計算K棒時間長度之單位', choices_unit, index=1)

if choice_unit == '以分鐘為單位':
    cycle_duration = st.sidebar.number_input('輸入一根K棒的時間長度(分鐘)', min_value=1, value=60, step=1)
elif choice_unit == '以日為單位':
    cycle_duration = st.sidebar.number_input('輸入一根K棒的時間長度(日)', min_value=1, value=1, step=1) * 1440
elif choice_unit == '以週為單位':
    cycle_duration = st.sidebar.number_input('輸入一根K棒的時間長度(週)', min_value=1, value=1, step=1) * 7 * 1440
else:
    cycle_duration = st.sidebar.number_input('輸入一根K棒的時間長度(月)', min_value=1, value=1, step=1) * 30 * 1440

###### 進行K棒更新
KBar_dic = Change_Cycle(start_date.strftime("%Y-%m-%d"), cycle_duration, KBar_dic, product_name)
KBar_df = pd.DataFrame(KBar_dic)

#%% 技術指標設定
st.sidebar.subheader("技術指標參數")

with st.sidebar.expander("移動平均線(MA)"):
    ma_long = st.number_input('長移動平均線週期', min_value=5, value=20, step=1)
    ma_short = st.number_input('短移動平均線週期', min_value=1, value=5, step=1)
    KBar_df['MA_long'] = Calculate_MA(KBar_df, period=ma_long)
    KBar_df['MA_short'] = Calculate_MA(KBar_df, period=ma_short)

with st.sidebar.expander("相對強弱指數(RSI)"):
    rsi_long = st.number_input('長RSI週期', min_value=5, value=14, step=1)
    rsi_short = st.number_input('短RSI週期', min_value=1, value=6, step=1)
    KBar_df['RSI_long'] = Calculate_RSI(KBar_df, period=rsi_long)
    KBar_df['RSI_short'] = Calculate_RSI(KBar_df, period=rsi_short)
    KBar_df['RSI_Middle'] = 50

with st.sidebar.expander("布林通道(Bollinger Bands)"):
    bb_period = st.number_input('布林通道週期', min_value=5, value=20, step=1)
    bb_std = st.number_input('標準差倍數', min_value=1, value=2, step=1)
    KBar_df = Calculate_Bollinger_Bands(KBar_df, period=bb_period, num_std_dev=bb_std)

with st.sidebar.expander("移動平均收斂發散(MACD)"):
    macd_fast = st.number_input('MACD快速線週期', min_value=5, value=12, step=1)
    macd_slow = st.number_input('MACD慢速線週期', min_value=10, value=26, step=1)
    macd_signal = st.number_input('MACD訊號線週期', min_value=5, value=9, step=1)
    KBar_df = Calculate_MACD(KBar_df, fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)

#%% 技術指標視覺化
st.header("技術指標視覺化")

tab1, tab2, tab3, tab4 = st.tabs(["K線與移動平均線", "RSI指標", "布林通道", "MACD指標"])

with tab1:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        yaxis=dict(fixedrange=False, autorange=True),
        xaxis=dict(rangeslider=dict(visible=True)))
    
    # K線
    fig.add_trace(go.Candlestick(
        x=KBar_df['time'],
        open=KBar_df['open'],
        high=KBar_df['high'],
        low=KBar_df['low'],
        close=KBar_df['close'],
        name='K線'), secondary_y=True)
    
    # 成交量
    fig.add_trace(go.Bar(
        x=KBar_df['time'],
        y=KBar_df['volume'],
        name='成交量',
        marker=dict(color='black')), secondary_y=False)
    
    # 移動平均線
    last_valid_index = KBar_df['MA_long'].last_valid_index()
    if last_valid_index is not None:
        df_plot = KBar_df.iloc[last_valid_index+1:]
        fig.add_trace(go.Scatter(
            x=df_plot['time'],
            y=df_plot['MA_long'],
            mode='lines',
            line=dict(color='orange', width=2),
            name=f'{ma_long}期移動平均線'), secondary_y=True)
        
        fig.add_trace(go.Scatter(
            x=df_plot['time'],
            y=df_plot['MA_short'],
            mode='lines',
            line=dict(color='pink', width=2),
            name=f'{ma_short}期移動平均線'), secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        yaxis=dict(fixedrange=False, autorange=True),
        xaxis=dict(rangeslider=dict(visible=True)))
    
    # RSI指標
    last_valid_index = KBar_df['RSI_long'].last_valid_index()
    if last_valid_index is not None:
        df_plot = KBar_df.iloc[last_valid_index+1:]
        fig.add_trace(go.Scatter(
            x=df_plot['time'],
            y=df_plot['RSI_long'],
            mode='lines',
            line=dict(color='red', width=2),
            name=f'{rsi_long}期RSI'), secondary_y=False)
        
        fig.add_trace(go.Scatter(
            x=df_plot['time'],
            y=df_plot['RSI_short'],
            mode='lines',
            line=dict(color='blue', width=2),
            name=f'{rsi_short}期RSI'), secondary_y=False)
    
        fig.add_trace(go.Scatter(
            x=df_plot['time'],
            y=df_plot['RSI_Middle'],
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            name='中線(50)'), secondary_y=False)
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        yaxis=dict(fixedrange=False, autorange=True),
        xaxis=dict(rangeslider=dict(visible=True)))
    
    # K線
    fig.add_trace(go.Candlestick(
        x=KBar_df['time'],
        open=KBar_df['open'],
        high=KBar_df['high'],
        low=KBar_df['low'],
        close=KBar_df['close'],
        name='K線'), secondary_y=True)
    
    # 布林通道
    last_valid_index = KBar_df['SMA'].last_valid_index()
    if last_valid_index is not None:
        df_plot = KBar_df.iloc[last_valid_index+1:]
        fig.add_trace(go.Scatter(
            x=df_plot['time'],
            y=df_plot['SMA'],
            mode='lines',
            line=dict(color='black', width=2),
            name='中軌道(SMA)'), secondary_y=True)
        
        fig.add_trace(go.Scatter(
            x=df_plot['time'],
            y=df_plot['Upper_Band'],
            mode='lines',
            line=dict(color='red', width=1),
            name='上軌道'), secondary_y=True)
        
        fig.add_trace(go.Scatter(
            x=df_plot['time'],
            y=df_plot['Lower_Band'],
            mode='lines',
            line=dict(color='blue', width=1),
            name='下軌道'), secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        yaxis=dict(fixedrange=False, autorange=True),
        xaxis=dict(rangeslider=dict(visible=True)))
    
    # MACD指標
    last_valid_index = KBar_df['MACD'].last_valid_index()
    if last_valid_index is not None:
        df_plot = KBar_df.iloc[last_valid_index+1:]
        fig.add_trace(go.Bar(
            x=df_plot['time'],
            y=df_plot['MACD_Histogram'],
            name='MACD柱狀圖',
            marker=dict(color=np.where(df_plot['MACD_Histogram'] > 0, 'green', 'red'))), 
            secondary_y=False)
        
        fig.add_trace(go.Scatter(
            x=df_plot['time'],
            y=df_plot['MACD'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='DIF線'), secondary_y=True)
        
        fig.add_trace(go.Scatter(
            x=df_plot['time'],
            y=df_plot['Signal_Line'],
            mode='lines',
            line=dict(color='orange', width=2),
            name='訊號線(DEA)'), secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

#%% 程式交易策略
st.header("程式交易策略")

# 策略選擇
strategy = st.selectbox("選擇交易策略", 
                       ["移動平均線交叉策略", "RSI策略", "MACD策略", "布林通道策略"])

# 通用參數設置
st.subheader("交易參數設置")
move_stop_loss = st.slider('停損點數', min_value=1, max_value=100, value=30, step=1)
order_quantity = st.slider('交易數量', min_value=1, max_value=10, value=1, step=1)
initial_capital = st.number_input('初始資金', min_value=10000, value=100000, step=10000)

# 策略執行按鈕
if st.button("執行回測"):
    OrderRecord = Record()
    
    if strategy == "移動平均線交叉策略":
        with st.spinner("執行移動平均線交叉策略..."):
            for i in range(1, len(KBar_df)):
                # 略過無效數據
                if pd.isna(KBar_df['MA_short'].iloc[i-1]) or pd.isna(KBar_df['MA_long'].iloc[i-1]):
                    continue
                    
                # 無持倉
                if OrderRecord.GetOpenInterest() == 0:
                    # 黃金交叉 - 買入
                    if (KBar_df['MA_short'].iloc[i-1] < KBar_df['MA_long'].iloc[i-1] and 
                        KBar_df['MA_short'].iloc[i] > KBar_df['MA_long'].iloc[i]):
                        OrderRecord.Order('Buy', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], order_quantity)
                        stop_loss_price = KBar_df['close'].iloc[i] - move_stop_loss
                    
                    # 死亡交叉 - 賣出
                    elif (KBar_df['MA_short'].iloc[i-1] > KBar_df['MA_long'].iloc[i-1] and 
                          KBar_df['MA_short'].iloc[i] < KBar_df['MA_long'].iloc[i]):
                        OrderRecord.Order('Sell', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], order_quantity)
                        stop_loss_price = KBar_df['close'].iloc[i] + move_stop_loss
                
                # 持有多頭倉位
                elif OrderRecord.GetOpenInterest() > 0:
                    # 更新停損價
                    if KBar_df['close'].iloc[i] - move_stop_loss > stop_loss_price:
                        stop_loss_price = KBar_df['close'].iloc[i] - move_stop_loss
                    
                    # 觸發停損
                    if KBar_df['close'].iloc[i] < stop_loss_price:
                        OrderRecord.Cover('Sell', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], order_quantity)
                
                # 持有空頭倉位
                elif OrderRecord.GetOpenInterest() < 0:
                    # 更新停損價
                    if KBar_df['close'].iloc[i] + move_stop_loss < stop_loss_price:
                        stop_loss_price = KBar_df['close'].iloc[i] + move_stop_loss
                    
                    # 觸發停損
                    if KBar_df['close'].iloc[i] > stop_loss_price:
                        OrderRecord.Cover('Buy', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], -OrderRecord.GetOpenInterest())
    
    elif strategy == "RSI策略":
        with st.spinner("執行RSI策略..."):
            overbought = 70
            oversold = 30
            
            for i in range(1, len(KBar_df)):
                if pd.isna(KBar_df['RSI_short'].iloc[i]) or pd.isna(KBar_df['RSI_short'].iloc[i-1]):
                    continue
                    
                # 無持倉
                if OrderRecord.GetOpenInterest() == 0:
                    # RSI低於超賣區 - 買入
                    if KBar_df['RSI_short'].iloc[i-1] < oversold and KBar_df['RSI_short'].iloc[i] >= oversold:
                        OrderRecord.Order('Buy', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], order_quantity)
                        stop_loss_price = KBar_df['close'].iloc[i] - move_stop_loss
                    
                    # RSI高於超買區 - 賣出
                    elif KBar_df['RSI_short'].iloc[i-1] > overbought and KBar_df['RSI_short'].iloc[i] <= overbought:
                        OrderRecord.Order('Sell', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], order_quantity)
                        stop_loss_price = KBar_df['close'].iloc[i] + move_stop_loss
                
                # 持有多頭倉位
                elif OrderRecord.GetOpenInterest() > 0:
                    # 更新停損價
                    if KBar_df['close'].iloc[i] - move_stop_loss > stop_loss_price:
                        stop_loss_price = KBar_df['close'].iloc[i] - move_stop_loss
                    
                    # RSI高於超買區 - 賣出
                    if KBar_df['RSI_short'].iloc[i] > overbought or KBar_df['close'].iloc[i] < stop_loss_price:
                        OrderRecord.Cover('Sell', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], order_quantity)
                
                # 持有空頭倉位
                elif OrderRecord.GetOpenInterest() < 0:
                    # 更新停損價
                    if KBar_df['close'].iloc[i] + move_stop_loss < stop_loss_price:
                        stop_loss_price = KBar_df['close'].iloc[i] + move_stop_loss
                    
                    # RSI低於超賣區 - 買入
                    if KBar_df['RSI_short'].iloc[i] < oversold or KBar_df['close'].iloc[i] > stop_loss_price:
                        OrderRecord.Cover('Buy', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], -OrderRecord.GetOpenInterest())
    
    elif strategy == "MACD策略":
        with st.spinner("執行MACD策略..."):
            for i in range(1, len(KBar_df)):
                if pd.isna(KBar_df['MACD'].iloc[i]) or pd.isna(KBar_df['Signal_Line'].iloc[i]):
                    continue
                    
                # 無持倉
                if OrderRecord.GetOpenInterest() == 0:
                    # MACD上穿信號線 - 買入
                    if (KBar_df['MACD'].iloc[i-1] < KBar_df['Signal_Line'].iloc[i-1] and 
                        KBar_df['MACD'].iloc[i] > KBar_df['Signal_Line'].iloc[i]):
                        OrderRecord.Order('Buy', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], order_quantity)
                        stop_loss_price = KBar_df['close'].iloc[i] - move_stop_loss
                    
                    # MACD下穿信號線 - 賣出
                    elif (KBar_df['MACD'].iloc[i-1] > KBar_df['Signal_Line'].iloc[i-1] and 
                          KBar_df['MACD'].iloc[i] < KBar_df['Signal_Line'].iloc[i]):
                        OrderRecord.Order('Sell', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], order_quantity)
                        stop_loss_price = KBar_df['close'].iloc[i] + move_stop_loss
                
                # 持有多頭倉位
                elif OrderRecord.GetOpenInterest() > 0:
                    # 更新停損價
                    if KBar_df['close'].iloc[i] - move_stop_loss > stop_loss_price:
                        stop_loss_price = KBar_df['close'].iloc[i] - move_stop_loss
                    
                    # MACD下穿信號線或觸發停損 - 賣出
                    if (KBar_df['MACD'].iloc[i] < KBar_df['Signal_Line'].iloc[i] or 
                        KBar_df['close'].iloc[i] < stop_loss_price):
                        OrderRecord.Cover('Sell', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], order_quantity)
                
                # 持有空頭倉位
                elif OrderRecord.GetOpenInterest() < 0:
                    # 更新停損價
                    if KBar_df['close'].iloc[i] + move_stop_loss < stop_loss_price:
                        stop_loss_price = KBar_df['close'].iloc[i] + move_stop_loss
                    
                    # MACD上穿信號線或觸發停損 - 買入
                    if (KBar_df['MACD'].iloc[i] > KBar_df['Signal_Line'].iloc[i] or 
                        KBar_df['close'].iloc[i] > stop_loss_price):
                        OrderRecord.Cover('Buy', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], -OrderRecord.GetOpenInterest())
    
    elif strategy == "布林通道策略":
        with st.spinner("執行布林通道策略..."):
            for i in range(1, len(KBar_df)):
                if pd.isna(KBar_df['Upper_Band'].iloc[i]) or pd.isna(KBar_df['Lower_Band'].iloc[i]):
                    continue
                    
                # 無持倉
                if OrderRecord.GetOpenInterest() == 0:
                    # 價格觸及下軌 - 買入
                    if KBar_df['close'].iloc[i] < KBar_df['Lower_Band'].iloc[i]:
                        OrderRecord.Order('Buy', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], order_quantity)
                        stop_loss_price = KBar_df['close'].iloc[i] - move_stop_loss
                    
                    # 價格觸及上軌 - 賣出
                    elif KBar_df['close'].iloc[i] > KBar_df['Upper_Band'].iloc[i]:
                        OrderRecord.Order('Sell', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], order_quantity)
                        stop_loss_price = KBar_df['close'].iloc[i] + move_stop_loss
                
                # 持有多頭倉位
                elif OrderRecord.GetOpenInterest() > 0:
                    # 更新停損價
                    if KBar_df['close'].iloc[i] - move_stop_loss > stop_loss_price:
                        stop_loss_price = KBar_df['close'].iloc[i] - move_stop_loss
                    
                    # 價格觸及中軌或觸發停損 - 賣出
                    if (KBar_df['close'].iloc[i] > KBar_df['SMA'].iloc[i] or 
                        KBar_df['close'].iloc[i] < stop_loss_price):
                        OrderRecord.Cover('Sell', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], order_quantity)
                
                # 持有空頭倉位
                elif OrderRecord.GetOpenInterest() < 0:
                    # 更新停損價
                    if KBar_df['close'].iloc[i] + move_stop_loss < stop_loss_price:
                        stop_loss_price = KBar_df['close'].iloc[i] + move_stop_loss
                    
                    # 價格觸及中軌或觸發停損 - 買入
                    if (KBar_df['close'].iloc[i] < KBar_df['SMA'].iloc[i] or 
                        KBar_df['close'].iloc[i] > stop_loss_price):
                        OrderRecord.Cover('Buy', product_name, KBar_df['time'].iloc[i], 
                                        KBar_df['close'].iloc[i], -OrderRecord.GetOpenInterest())
    
    # 保存交易記錄
    st.session_state.order_record = OrderRecord
    st.success("回測完成！")

# 顯示交易結果
if st.session_state.order_record is not None and len(st.session_state.order_record.GetTradeRecord()) > 0:
    st.header("交易結果分析")
    
    # 顯示交易點位圖
    if strategy == "移動平均線交叉策略":
        ChartOrder_MA(KBar_df, st.session_state.order_record.GetTradeRecord())
    else:
        ChartOrder_Trade(KBar_df, st.session_state.order_record.GetTradeRecord())
    
    # 顯示績效分析
    Show_Performance(st.session_state.order_record)
else:
    st.info("請執行回測以查看交易結果")