# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:52:54 2025

@author: student
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:24:20 2025

@author: student
"""

# -*- coding: utf-8 -*-
"""
金融資料視覺化看板
"""
# 載入必要模組
import os
import numpy as np
import datetime
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 
from order_streamlit import Record
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import indicator_f_Lo2_short

# 重新定義 KBar 類（修正空列表問題）
class KBar():
    def __init__(self, Date, cycle_duration):
        self.Time = [datetime.datetime.strptime("2023-08-15 00:00:00", "%Y-%m-%d %H:%M:%S")]
        self.Open = []
        self.Close = []
        self.Low = []
        self.High = []
        self.Volume = []
        self.Cycle = cycle_duration
        self.TAKBar = {'time':[],'open':[],'high':[],'low':[],'close':[],'volume':[]}
        self.Product = None
        
    def AddPrice(self, time, open_price, close_price, low_price, high_price, qty):
        # 如果是第一筆資料，直接建立新的 K 棒
        if len(self.TAKBar['time']) == 0:
            self.TAKBar['time'].append(time)
            self.TAKBar['open'].append(open_price)
            self.TAKBar['close'].append(close_price)
            self.TAKBar['low'].append(low_price)
            self.TAKBar['high'].append(high_price)
            self.TAKBar['volume'].append(qty)
            return
        
        # 計算當前時間與最後一根 K 棒的時間差（分鐘）
        time_diff = (time - self.TAKBar['time'][-1]).total_seconds() / 60
        
        # 如果時間差小於週期，更新最後一根 K 棒
        if time_diff < self.Cycle:
            self.TAKBar['close'][-1] = close_price
            self.TAKBar['volume'][-1] += qty
            
            if high_price > self.TAKBar['high'][-1]:
                self.TAKBar['high'][-1] = high_price
                
            if low_price < self.TAKBar['low'][-1]:
                self.TAKBar['low'][-1] = low_price
        else:
            # 建立新的 K 棒
            self.TAKBar['time'].append(time)
            self.TAKBar['open'].append(close_price)  # 新 K 棒開盤價 = 前一根收盤價
            self.TAKBar['close'].append(close_price)
            self.TAKBar['high'].append(high_price)
            self.TAKBar['low'].append(low_price)
            self.TAKBar['volume'].append(qty)

#%% (1) 開始設定
html_temp = """
<div style="background-color:#3872fb;padding:10px;border-radius:10px">   
<h1 style="color:white;text-align:center;">金融看板與程式交易平台 </h1>
<h2 style="color:white;text-align:center;">Financial Dashboard and Program Trading </h2>
</div>
"""
stc.html(html_temp)

# 讀取資料
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def load_data(path):
    if not os.path.exists(path):
        st.error(f"檔案 {path} 不存在！")
        return pd.DataFrame()
    return pd.read_pickle(path)

st.subheader("選擇金融商品: ")
choices = ['富邦金期貨: 2023.4.15 至 2025.4.16', '華碩: 2023.4.17至2025.4.16', '聯電期貨: 2023.4.17至2025.4.16']
choice = st.selectbox('選擇金融商品', choices, index=0)

if choice == choices[0]:
    file_path = 'future_KBar_CEF2023.4.15-2025.4.16.pkl'
    product_name = '富邦金期貨'
elif choice == choices[1]:
    file_path = 'stock_KBar_2357 2023.4.17-2025.4.16.pkl'
    product_name = '華碩'
else:
    file_path = 'future_KBar_CCF 2023.4.17-2025.4.16.pkl'
    product_name = '聯電期貨'

df_original = load_data(file_path)

# 檢查資料是否載入成功
if df_original.empty:
    st.error(f"沒有找到 {product_name} 的資料！請檢查資料檔案是否存在。")
    st.stop()

# 顯示資料預覽
with st.expander("資料預覽"):
    st.dataframe(df_original.head())
    st.write(f"總資料量: {len(df_original)} 筆")
    st.write(f"時間範圍: {df_original['time'].min()} 至 {df_original['time'].max()}")

st.subheader("選擇資料時間區間")
if choice == choices[0]:
    default_start = '2023.4.15'
    default_end = '2025.4.16'
elif choice == choices[1]:
    default_start = '2023.4.17'
    default_end = '2025.4.16'
else:
    default_start = '2023.4.17'
    default_end = '2025.4.16'

start_date_str = st.text_input(f'輸入開始日期(日期格式: {default_start}), 區間:{default_start} 至 {default_end}', default_start)
end_date_str = st.text_input(f'輸入結束日期 (日期格式: {default_end}), 區間:{default_start} 至 {default_end}', default_end)

# 日期格式轉換
try:
    start_date = datetime.datetime.strptime(start_date_str, '%Y.%m.%d')
    end_date = datetime.datetime.strptime(end_date_str, '%Y.%m.%d')
except ValueError:
    st.error("日期格式錯誤，請使用 YYYY.MM.DD 格式（例如：2023.04.15）")
    st.stop()

# 過濾資料
df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]

# 檢查過濾後是否有資料
if df.empty:
    st.error(f"在選定的時間範圍內沒有 {product_name} 的資料！請調整日期範圍。")
    st.stop()

#%% (2) 轉化為字典
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def To_Dictionary_1(df, product_name):
    KBar_dic = df.to_dict()
    KBar_dic['product'] = np.repeat(product_name, len(df))
    
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        KBar_dic[col] = np.array(list(KBar_dic[col].values()))
    
    KBar_time_list = [pd.Timestamp(t).to_pydatetime() for t in df['time'].values]
    KBar_dic['time'] = np.array(KBar_time_list)
    
    return KBar_dic

KBar_dic = To_Dictionary_1(df, product_name)

#%% (3) 改變 KBar 時間長度
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")
def Change_Cycle(Date, cycle_duration, KBar_dic, product_name):
    kbar = KBar('time', cycle_duration)

    for i in range(len(KBar_dic['time'])):
        time = KBar_dic['time'][i]
        open_price = KBar_dic['open'][i]
        close_price = KBar_dic['close'][i]
        low_price = KBar_dic['low'][i]
        high_price = KBar_dic['high'][i]
        qty = KBar_dic['volume'][i]
        kbar.AddPrice(time, open_price, close_price, low_price, high_price, qty)

    new_dict = {
        'time': kbar.TAKBar['time'],
        'product': np.repeat(product_name, len(kbar.TAKBar['time'])),
        'open': kbar.TAKBar['open'],
        'high': kbar.TAKBar['high'],
        'low': kbar.TAKBar['low'],
        'close': kbar.TAKBar['close'],
        'volume': kbar.TAKBar['volume']
    }
    return new_dict


st.subheader("設定技術指標視覺化圖形之相關參數:")
with st.expander("設定K棒相關參數:"):
    choices_unit = ['以分鐘為單位','以日為單位','以週為單位','以月為單位']
    choice_unit = st.selectbox('選擇計算K棒時間長度之單位', choices_unit, index=1)
    
    if choice_unit == '以分鐘為單位':
        cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:分鐘, 一日=1440分鐘)', value=1440, key="KBar_duration_分")
        cycle_duration = float(cycle_duration)
    elif choice_unit == '以日為單位':
        cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:日)', value=1, key="KBar_duration_日")
        cycle_duration = float(cycle_duration) * 1440
    elif choice_unit == '以週為單位':
        cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:週)', value=1, key="KBar_duration_週")
        cycle_duration = float(cycle_duration) * 7 * 1440
    else:
        cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:月, 一月=30天)', value=1, key="KBar_duration_月")
        cycle_duration = float(cycle_duration) * 30 * 1440

Date = start_date.strftime("%Y-%m-%d")
try:
    KBar_dic = Change_Cycle(Date, cycle_duration, KBar_dic, product_name)
    KBar_df = pd.DataFrame(KBar_dic)
    
    # 檢查轉換後的KBar資料
    if KBar_df.empty:
        st.error("K棒轉換後沒有資料！請調整參數。")
        st.stop()
except Exception as e:
    st.error(f"轉換K棒週期時出錯: {str(e)}")
    st.stop()

#%% (4) 計算各種技術指標
def Calculate_MA(df, period=10):
    return df['close'].rolling(window=period).mean()

def Calculate_RSI(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def Calculate_Bollinger_Bands(df, period=20, num_std_dev=2):
    df['SMA'] = df['close'].rolling(window=period).mean()
    df['Standard_Deviation'] = df['close'].rolling(window=period).std()
    df['Upper_Band'] = df['SMA'] + (df['Standard_Deviation'] * num_std_dev)
    df['Lower_Band'] = df['SMA'] - (df['Standard_Deviation'] * num_std_dev)
    return df

def Calculate_MACD(df, fast_period=12, slow_period=26, signal_period=9):
    df['EMA_Fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_Slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    return df

with st.expander("設定長短移動平均線的 K棒 長度:"):
    LongMAPeriod = st.slider('設定計算長移動平均線(MA)的 K棒週期數目', 5, 100, 20, key='visualization_MA_long')
    ShortMAPeriod = st.slider('設定計算短移動平均線(MA)的 K棒週期數目', 1, 50, 5, key='visualization_MA_short')

KBar_df['MA_long'] = Calculate_MA(KBar_df, period=LongMAPeriod)
KBar_df['MA_short'] = Calculate_MA(KBar_df, period=ShortMAPeriod)
last_nan_index_MA = KBar_df['MA_long'].isna().sum()

# 檢查移動平均線計算
if KBar_df['MA_long'].isna().all():
    st.error("移動平均線計算失敗，請檢查參數設置")
    st.stop()

with st.expander("設定長短 RSI 的 K棒 長度:"):
    LongRSIPeriod = st.slider('設定計算長RSI的 K棒週期數目', 5, 50, 14, key='visualization_RSI_long')
    ShortRSIPeriod = st.slider('設定計算短RSI的 K棒週期數目', 1, 20, 6, key='visualization_RSI_short')

KBar_df['RSI_long'] = Calculate_RSI(KBar_df, LongRSIPeriod)
KBar_df['RSI_short'] = Calculate_RSI(KBar_df, ShortRSIPeriod)
KBar_df['RSI_Middle'] = 50
last_nan_index_RSI = KBar_df['RSI_long'].isna().sum()

with st.expander("設定布林通道(Bollinger Band)相關參數:"):
    period = st.slider('設定計算布林通道的K棒週期數目', 10, 60, 20, key='BB_period')
    num_std_dev = st.slider('設定布林通道的標準差倍數', 1.0, 3.0, 2.0, key='BB_heigh')

KBar_df = Calculate_Bollinger_Bands(KBar_df, period, num_std_dev)
last_nan_index_BB = KBar_df['SMA'].isna().sum()

with st.expander("設定MACD三種週期的K棒長度:"):
    fast_period = st.slider('設定計算 MACD快速線的K棒週期數目', 5, 20, 12, key='visualization_MACD_quick')
    slow_period = st.slider('設定計算 MACD慢速線的K棒週期數目', 15, 50, 26, key='visualization_MACD_slow')
    signal_period = st.slider('設定計算 MACD訊號線的K棒週期數目', 5, 20, 9, key='visualization_MACD_signal')

KBar_df = Calculate_MACD(KBar_df, fast_period, slow_period, signal_period)
last_nan_index_MACD = KBar_df['MACD'].isna().sum()

#%% (5) 畫圖
st.subheader("技術指標視覺化圖形")

# K線圖, 移動平均線MA
with st.expander("K線圖, 移動平均線"):
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Candlestick(
        x=KBar_df['time'],
        open=KBar_df['open'], 
        high=KBar_df['high'],
        low=KBar_df['low'], 
        close=KBar_df['close'], 
        name='K線'
    ), secondary_y=True)
    
    fig1.add_trace(go.Bar(
        x=KBar_df['time'], 
        y=KBar_df['volume'], 
        name='成交量', 
        marker=dict(color='rgba(100, 100, 100, 0.5)')
    ), secondary_y=False)
    
    if last_nan_index_MA < len(KBar_df):
        fig1.add_trace(go.Scatter(
            x=KBar_df['time'][last_nan_index_MA:], 
            y=KBar_df['MA_long'][last_nan_index_MA:], 
            mode='lines',
            line=dict(color='orange', width=2), 
            name=f'{LongMAPeriod}日均線'
        ), secondary_y=True)
        
        fig1.add_trace(go.Scatter(
            x=KBar_df['time'][last_nan_index_MA:], 
            y=KBar_df['MA_short'][last_nan_index_MA:], 
            mode='lines',
            line=dict(color='blue', width=2), 
            name=f'{ShortMAPeriod}日均線'
        ), secondary_y=True)
    
    fig1.update_layout(
        title=f'{product_name} K線圖與移動平均線',
        xaxis_title='時間',
        yaxis_title='價格',
        height=600,
        showlegend=True
    )
    st.plotly_chart(fig1, use_container_width=True)

# RSI圖
with st.expander("長短 RSI"):
    fig2 = make_subplots(specs=[[{"secondary_y": False}]])
    
    if last_nan_index_RSI < len(KBar_df):
        fig2.add_trace(go.Scatter(
            x=KBar_df['time'][last_nan_index_RSI:], 
            y=KBar_df['RSI_long'][last_nan_index_RSI:], 
            mode='lines',
            line=dict(color='red', width=2), 
            name=f'{LongRSIPeriod}日RSI'
        ))
        
        fig2.add_trace(go.Scatter(
            x=KBar_df['time'][last_nan_index_RSI:], 
            y=KBar_df['RSI_short'][last_nan_index_RSI:], 
            mode='lines',
            line=dict(color='blue', width=2), 
            name=f'{ShortRSIPeriod}日RSI'
        ))
        
        fig2.add_trace(go.Scatter(
            x=KBar_df['time'][last_nan_index_RSI:], 
            y=KBar_df['RSI_Middle'][last_nan_index_RSI:], 
            mode='lines',
            line=dict(color='green', width=1, dash='dash'), 
            name='中線(50)'
        ))
    
    fig2.update_layout(
        title=f'{product_name} RSI指標',
        xaxis_title='時間',
        yaxis_title='RSI值',
        height=400,
        showlegend=True
    )
    st.plotly_chart(fig2, use_container_width=True)

# 布林通道圖
with st.expander("K線圖,布林通道"):
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Candlestick(
        x=KBar_df['time'],
        open=KBar_df['open'], 
        high=KBar_df['high'],
        low=KBar_df['low'], 
        close=KBar_df['close'], 
        name='K線'
    ), secondary_y=True)
    
    if last_nan_index_BB < len(KBar_df):
        fig3.add_trace(go.Scatter(
            x=KBar_df['time'][last_nan_index_BB:], 
            y=KBar_df['SMA'][last_nan_index_BB:], 
            mode='lines',
            line=dict(color='blue', width=2), 
            name='中軌'
        ), secondary_y=True)
        
        fig3.add_trace(go.Scatter(
            x=KBar_df['time'][last_nan_index_BB:], 
            y=KBar_df['Upper_Band'][last_nan_index_BB:], 
            mode='lines',
            line=dict(color='red', width=1), 
            name='上軌'
        ), secondary_y=True)
        
        fig3.add_trace(go.Scatter(
            x=KBar_df['time'][last_nan_index_BB:], 
            y=KBar_df['Lower_Band'][last_nan_index_BB:], 
            mode='lines',
            line=dict(color='green', width=1), 
            name='下軌',
            fill='tonexty',
            fillcolor='rgba(100, 200, 100, 0.2)'
        ), secondary_y=True)
    
    fig3.update_layout(
        title=f'{product_name} 布林通道',
        xaxis_title='時間',
        yaxis_title='價格',
        height=600,
        showlegend=True
    )
    st.plotly_chart(fig3, use_container_width=True)

# MACD圖
with st.expander("MACD(異同移動平均線)"):
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    
    if last_nan_index_MACD < len(KBar_df):
        # MACD柱狀圖
        colors = ['green' if val >= 0 else 'red' for val in KBar_df['MACD_Histogram'][last_nan_index_MACD:]]
        fig4.add_trace(go.Bar(
            x=KBar_df['time'][last_nan_index_MACD:], 
            y=KBar_df['MACD_Histogram'][last_nan_index_MACD:], 
            name='MACD柱',
            marker_color=colors
        ), secondary_y=False)
        
        # DIF線
        fig4.add_trace(go.Scatter(
            x=KBar_df['time'][last_nan_index_MACD:], 
            y=KBar_df['MACD'][last_nan_index_MACD:], 
            mode='lines',
            line=dict(color='blue', width=2), 
            name='DIF'
        ), secondary_y=True)
        
        # DEA線
        fig4.add_trace(go.Scatter(
            x=KBar_df['time'][last_nan_index_MACD:], 
            y=KBar_df['Signal_Line'][last_nan_index_MACD:], 
            mode='lines',
            line=dict(color='orange', width=2), 
            name='DEA'
        ), secondary_y=True)
    
    fig4.update_layout(
        title=f'{product_name} MACD指標',
        xaxis_title='時間',
        height=500,
        showlegend=True
    )
    st.plotly_chart(fig4, use_container_width=True)

#%% (6) 程式交易
st.subheader("程式交易:")

# 策略选择 
choices_strategies = ['移動平均線策略', 'RSI 策略', 'MACD 策略', '布林通道策略'] 
choice_strategy = st.selectbox('選擇交易策略', choices_strategies, index=0)

# 初始化交易记录
OrderRecord = Record()

# 移動平均線策略
if choice_strategy == choices_strategies[0]:
    with st.expander("移動平均線策略參數設定"):
        MoveStopLoss = st.slider('停損點數', 1, 100, 30)
        LongMAPeriod = st.slider('長移動平均線週期', 10, 100, 20)
        ShortMAPeriod = st.slider('短移動平均線週期', 1, 20, 5)
        Order_Quantity = st.slider('交易數量', 1, 10, 1)
    
    # 重新計算移動平均線
    KBar_df['MA_long'] = Calculate_MA(KBar_df, period=LongMAPeriod)
    KBar_df['MA_short'] = Calculate_MA(KBar_df, period=ShortMAPeriod)
    last_nan_index = max(KBar_df['MA_long'].isna().sum(), KBar_df['MA_short'].isna().sum())
    
    # 初始化交易狀態
    position = None  # 當前持倉方向: 'long' 或 'short'
    entry_price = 0  # 進場價格
    highest_after_entry = 0  # 進場後最高價
    lowest_after_entry = float('inf')  # 進場後最低價
    
    # 策略邏輯
    for i in range(last_nan_index + 1, len(KBar_df)):
        # 獲取當前價格
        NowClose = KBar_df['close'][i]
        NowHigh = KBar_df['high'][i]
        NowLow = KBar_df['low'][i]
        NowTime = KBar_df['time'][i]
        
        # 獲取指標值
        NowMA_short = KBar_df['MA_short'][i]
        NowMA_long = KBar_df['MA_long'][i]
        PreMA_short = KBar_df['MA_short'][i-1]
        PreMA_long = KBar_df['MA_long'][i-1]
        
        # 更新進場後最高/最低價
        if position == 'long':
            if NowHigh > highest_after_entry:
                highest_after_entry = NowHigh
        elif position == 'short':
            if NowLow < lowest_after_entry:
                lowest_after_entry = NowLow
        
        # 移動停損檢查
        if position == 'long':
            stop_loss_price = highest_after_entry - MoveStopLoss
            if NowLow <= stop_loss_price:
                # 觸發停損賣出
                OrderRecord.Cover('Sell', product_name, NowTime, stop_loss_price, Order_Quantity)
                position = None
                st.success(f"📉 移動停損觸發 (賣出): {NowTime} @ {stop_loss_price}")
                
        elif position == 'short':
            stop_loss_price = lowest_after_entry + MoveStopLoss
            if NowHigh >= stop_loss_price:
                # 觸發停損買回
                OrderRecord.Cover('Buy', product_name, NowTime, stop_loss_price, Order_Quantity)
                position = None
                st.success(f"📈 移動停損觸發 (買回): {NowTime} @ {stop_loss_price}")
        
        # 交易條件 - 只在沒有持倉時進場
        if position is None:
            cross_over = (PreMA_short < PreMA_long) and (NowMA_short > NowMA_long)
            cross_under = (PreMA_short > PreMA_long) and (NowMA_short < NowMA_long)
            
            if cross_over:
                # 金叉買入
                OrderRecord.Order('Buy', product_name, NowTime, NowClose, Order_Quantity)
                position = 'long'
                entry_price = NowClose
                highest_after_entry = NowHigh
                st.success(f"🚀 黃金交叉買入: {NowTime} @ {NowClose}")
                
            elif cross_under:
                # 死叉賣出
                OrderRecord.Order('Sell', product_name, NowTime, NowClose, Order_Quantity)
                position = 'short'
                entry_price = NowClose
                lowest_after_entry = NowLow
                st.success(f"💥 死亡交叉賣出: {NowTime} @ {NowClose}")
    
    # 回測結束時，若有持倉則平倉
    if position == 'long':
        OrderRecord.Cover('Sell', product_name, KBar_df['time'].iloc[-1], KBar_df['close'].iloc[-1], Order_Quantity)
        st.success(f"🏁 回測結束平倉 (賣出): {KBar_df['time'].iloc[-1]} @ {KBar_df['close'].iloc[-1]}")
    elif position == 'short':
        OrderRecord.Cover('Buy', product_name, KBar_df['time'].iloc[-1], KBar_df['close'].iloc[-1], Order_Quantity)
        st.success(f"🏁 回測結束平倉 (買回): {KBar_df['time'].iloc[-1]} @ {KBar_df['close'].iloc[-1]}")

# 顯示交易記錄
if OrderRecord and OrderRecord.GetTradeRecord():
    st.subheader("交易記錄")
    trades = []
    for trade in OrderRecord.GetTradeRecord():
        trades.append({
            "類型": trade[0],
            "商品": trade[1],
            "時間": trade[2],
            "價格": trade[3],
            "數量": trade[4]
        })
    st.dataframe(pd.DataFrame(trades))
    
    # 顯示交易摘要
    buy_count = sum(1 for trade in trades if trade['類型'] == 'Buy')
    sell_count = sum(1 for trade in trades if trade['類型'] == 'Sell')
    cover_buy_count = sum(1 for trade in trades if trade['類型'] == 'Cover' and trade['類型'] == 'Buy')
    cover_sell_count = sum(1 for trade in trades if trade['類型'] == 'Cover' and trade['類型'] == 'Sell')
    
    st.subheader("交易摘要")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("買入次數", buy_count)
    col2.metric("賣出次數", sell_count)
    col3.metric("平倉買回", cover_buy_count)
    col4.metric("平倉賣出", cover_sell_count)
else:
    st.warning("沒有產生任何交易")
    
# ... (前面的程式碼保持不變) ...

#%% (6) 程式交易
st.subheader("程式交易:")

# 策略選擇 
choices_strategies = ['移動平均線策略', 'RSI 策略', 'MACD 策略', '布林通道策略'] 
choice_strategy = st.selectbox('選擇交易策略', choices_strategies, index=0,key='strategy_select_1') 

# 初始化交易記錄
OrderRecord = None

# 移動平均線策略
if choice_strategy == choices_strategies[0]:
    with st.expander("<策略參數設定>: 移動平均線策略"):
        MoveStopLoss = st.slider('停損點數', 1, 100, 30, key='ma_stop')
        LongMAPeriod = st.slider('長移動平均線週期', 5, 100, 20, key='ma_long')
        ShortMAPeriod = st.slider('短移動平均線週期', 1, 20, 5, key='ma_short')
        Order_Quantity = st.slider('交易數量', 1, 10, 1, key='ma_qty')
    
    # 重新計算移動平均線
    KBar_df['MA_long'] = Calculate_MA(KBar_df, period=LongMAPeriod)
    KBar_df['MA_short'] = Calculate_MA(KBar_df, period=ShortMAPeriod)
    last_nan_index_MA = max(KBar_df['MA_long'].isna().sum(), KBar_df['MA_short'].isna().sum())
    
    try:
        # 初始化交易記錄
        OrderRecord = Record()
        
        # 初始化交易狀態
        position = None  # 當前持倉方向: 'long' 或 'short'
        entry_price = 0  # 進場價格
        highest_after_entry = 0  # 進場後最高價
        lowest_after_entry = float('inf')  # 進場後最低價
        
        # 策略邏輯
        for i in range(last_nan_index_MA + 1, len(KBar_df)):
            # 獲取當前價格
            NowClose = KBar_df['close'][i]
            NowHigh = KBar_df['high'][i]
            NowLow = KBar_df['low'][i]
            NowTime = KBar_df['time'][i]
            
            # 獲取指標值
            NowMA_short = KBar_df['MA_short'][i]
            NowMA_long = KBar_df['MA_long'][i]
            PreMA_short = KBar_df['MA_short'][i-1]
            PreMA_long = KBar_df['MA_long'][i-1]
            
            # 更新進場後最高/最低價
            if position == 'long':
                if NowHigh > highest_after_entry:
                    highest_after_entry = NowHigh
            elif position == 'short':
                if NowLow < lowest_after_entry:
                    lowest_after_entry = NowLow
            
            # 移動停損檢查
            if position == 'long':
                stop_loss_price = highest_after_entry - MoveStopLoss
                if NowLow <= stop_loss_price:
                    # 觸發停損賣出
                    OrderRecord.Cover('Sell', product_name, NowTime, stop_loss_price, Order_Quantity)
                    position = None
                    st.success(f"📉 移動停損觸發 (賣出): {NowTime} @ {stop_loss_price}")
                    
            elif position == 'short':
                stop_loss_price = lowest_after_entry + MoveStopLoss
                if NowHigh >= stop_loss_price:
                    # 觸發停損買回
                    OrderRecord.Cover('Buy', product_name, NowTime, stop_loss_price, Order_Quantity)
                    position = None
                    st.success(f"📈 移動停損觸發 (買回): {NowTime} @ {stop_loss_price}")
            
            # 交易條件 - 只在沒有持倉時進場
            if position is None:
                cross_over = (PreMA_short < PreMA_long) and (NowMA_short > NowMA_long)
                cross_under = (PreMA_short > PreMA_long) and (NowMA_short < NowMA_long)
                
                if cross_over:
                    # 金叉買入
                    OrderRecord.Order('Buy', product_name, NowTime, NowClose, Order_Quantity)
                    position = 'long'
                    entry_price = NowClose
                    highest_after_entry = NowHigh
                    st.success(f"🚀 黃金交叉買入: {NowTime} @ {NowClose}")
                    
                elif cross_under:
                    # 死叉賣出
                    OrderRecord.Order('Sell', product_name, NowTime, NowClose, Order_Quantity)
                    position = 'short'
                    entry_price = NowClose
                    lowest_after_entry = NowLow
                    st.success(f"💥 死亡交叉賣出: {NowTime} @ {NowClose}")
        
        # 回測結束時，若有持倉則平倉
        if position == 'long':
            OrderRecord.Cover('Sell', product_name, KBar_df['time'].iloc[-1], KBar_df['close'].iloc[-1], Order_Quantity)
            st.success(f"🏁 回測結束平倉 (賣出): {KBar_df['time'].iloc[-1]} @ {KBar_df['close'].iloc[-1]}")
        elif position == 'short':
            OrderRecord.Cover('Buy', product_name, KBar_df['time'].iloc[-1], KBar_df['close'].iloc[-1], Order_Quantity)
            st.success(f"🏁 回測結束平倉 (買回): {KBar_df['time'].iloc[-1]} @ {KBar_df['close'].iloc[-1]}")
            
    except Exception as e:
        st.error(f"執行移動平均線策略時發生錯誤: {str(e)}")
        st.stop()

# 其他策略類似，這裡省略以節省空間...
# RSI 策略、MACD 策略、布林通道策略的實現方式類似

#%% 績效計算部分
def 計算績效_大台指期貨(order_record): 
    if order_record is None or len(order_record.GetTradeRecord()) == 0: 
        return 0, 0, 0, 0, 0, 0, 0, 0, 0 
    # ... (保持原有的绩效计算逻辑) ...
def 計算績效_華碩(order_record): 
    if order_record is None or len(order_record.GetTradeRecord()) == 0: 
        return 0, 0, 0, 0, 0, 0, 0, 0, 0 
def 計算績效_聯電期貨(order_record): 
    if order_record is None or len(order_record.GetTradeRecord()) == 0: 
        return 0, 0, 0, 0, 0, 0, 0, 0, 0 
# 策略选择 
if OrderRecord and len(OrderRecord.GetTradeRecord()) > 0:
    st.subheader("交易記錄")
    trades = []
    for trade in OrderRecord.GetTradeRecord():
        trades.append({
            "類型": trade[0],
            "商品": trade[1],
            "時間": trade[2],
            "價格": trade[3],
            "數量": trade[4]
        })
    st.dataframe(pd.DataFrame(trades))
    
    st.subheader("績效分析")
    
    # 根據商品類型選擇績效計算函數
    if choice in [choices[0], choices[1]]:  # 富邦金期貨和大台指
        績效結果 = 計算績效_大台指期貨(OrderRecord)
    elif choice == choices[2]:  # 小台指
        績效結果 = 計算績效_華碩(OrderRecord)
    else:  # 股票
        績效結果 = 計算績效_聯電期貨(OrderRecord)
    
    # 顯示績效表格
    st.table(pd.DataFrame({
        "績效指標": ["交易總盈虧", "平均每次盈虧", "平均投資報酬率", "平均獲利(只看獲利)", 
                  "平均虧損(只看虧損)", "勝率", "最大連續虧損", "最大盈虧回落(MDD)", "報酬風險比"],
        "數值": 績效結果
    }))
    
    # 累計盈虧圖
    try:
        OrderRecord.GeneratorProfitChart(choice='future' if '期貨' in choice else 'stock', StrategyName=choice_strategy)
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.error(f"生成累計盈虧圖時出錯: {str(e)}")
    
    # 累計投資報酬率圖
    try:
        OrderRecord.GeneratorProfit_rateChart(StrategyName=choice_strategy)
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.error(f"生成累計投資報酬率圖時出錯: {str(e)}")
else:
    st.warning("沒有交易記錄可供分析")

# ... (後面的程式碼保持不變) ... 
#%% 
####### (7) 呈現即時資料 ####### 


#%%
###### 計算績效:
# OrderRecord.GetTradeRecord()          ## 交易紀錄清單
# OrderRecord.GetProfit()               ## 利潤清單


def 計算績效_股票():
    交易總盈虧 = OrderRecord.GetTotalProfit()*1000          ## 取得交易總盈虧
    平均每次盈虧 = OrderRecord.GetAverageProfit()*1000         ## 取得交易 "平均" 盈虧(每次)
    平均投資報酬率 = OrderRecord.GetAverageProfitRate()    ## 取得交易 "平均" 投資報酬率(每次)  
    平均獲利_只看獲利的 = OrderRecord.GetAverEarn()*1000              ## 平均獲利(只看獲利的) 
    平均虧損_只看虧損的 = OrderRecord.GetAverLoss()*1000              ## 平均虧損(只看虧損的)
    勝率 = OrderRecord.GetWinRate()              ## 勝率
    最大連續虧損 = OrderRecord.GetAccLoss()*1000               ## 最大連續虧損
    最大盈虧回落_MDD = OrderRecord.GetMDD()*1000                   ## 最大利潤(盈虧)回落(MDD). 這個不是一般的 "資金" 或 "投資報酬率" 的回落
    if 最大盈虧回落_MDD>0:
        報酬風險比 = 交易總盈虧/最大盈虧回落_MDD
    else:
        報酬風險比='資料不足無法計算'
    return (交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比)


def 計算績效_大台指期貨():
    交易總盈虧 = OrderRecord.GetTotalProfit()*200          ## 取得交易總盈虧
    平均每次盈虧 = OrderRecord.GetAverageProfit()*200         ## 取得交易 "平均" 盈虧(每次)
    平均投資報酬率 = OrderRecord.GetAverageProfitRate()    ## 取得交易 "平均" 投資報酬率(每次)  
    平均獲利_只看獲利的 = OrderRecord.GetAverEarn()*200              ## 平均獲利(只看獲利的) 
    平均虧損_只看虧損的 = OrderRecord.GetAverLoss()*200              ## 平均虧損(只看虧損的)
    勝率 = OrderRecord.GetWinRate()              ## 勝率
    最大連續虧損 = OrderRecord.GetAccLoss()*200               ## 最大連續虧損
    最大盈虧回落_MDD = OrderRecord.GetMDD()*200                   ## 最大利潤(盈虧)回落(MDD). 這個不是一般的 "資金" 或 "投資報酬率" 的回落
    if 最大盈虧回落_MDD>0:
        報酬風險比 = 交易總盈虧/最大盈虧回落_MDD
    else:
        報酬風險比='資料不足無法計算'
    return 交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比


def 計算績效_小台指期貨():
    交易總盈虧 = OrderRecord.GetTotalProfit()*50          ## 取得交易總盈虧
    平均每次盈虧 = OrderRecord.GetAverageProfit()*50         ## 取得交易 "平均" 盈虧(每次)
    平均投資報酬率 = OrderRecord.GetAverageProfitRate()    ## 取得交易 "平均" 投資報酬率(每次)  
    平均獲利_只看獲利的 = OrderRecord.GetAverEarn()*50              ## 平均獲利(只看獲利的) 
    平均虧損_只看虧損的 = OrderRecord.GetAverLoss()*50              ## 平均虧損(只看虧損的)
    勝率 = OrderRecord.GetWinRate()              ## 勝率
    最大連續虧損 = OrderRecord.GetAccLoss()*50               ## 最大連續虧損
    最大盈虧回落_MDD = OrderRecord.GetMDD()*50                   ## 最大利潤(盈虧)回落(MDD). 這個不是一般的 "資金" 或 "投資報酬率" 的回落
    if 最大盈虧回落_MDD>0:
        報酬風險比 = 交易總盈虧/最大盈虧回落_MDD
    else:
        報酬風險比='資料不足無法計算'
    return 交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比





if choice == choices[0]:
   交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_大台指期貨()

# 交易總盈虧 = OrderRecord.GetTotalProfit()*1000          ## 取得交易總盈虧
    # 平均每次盈虧 = OrderRecord.GetAverageProfit()*1000         ## 取得交易 "平均" 盈虧(每次)
    # 平均投資報酬率 = OrderRecord.GetAverageProfitRate()    ## 取得交易 "平均" 投資報酬率(每次)  
    # 平均獲利_只看獲利的 = OrderRecord.GetAverEarn()*1000              ## 平均獲利(只看獲利的) 
    # 平均虧損_只看虧損的 = OrderRecord.GetAverLoss()*1000              ## 平均虧損(只看虧損的)
    # 勝率 = OrderRecord.GetWinRate()              ## 勝率
    # 最大連續虧損 = OrderRecord.GetAccLoss()*1000               ## 最大連續虧損
    # 最大盈虧回落_MDD = OrderRecord.GetMDD()*1000                   ## 最大利潤(盈虧)回落(MDD). 這個不是一般的 "資金" 或 "投資報酬率" 的回落
    # if 最大盈虧回落_MDD>0:
    #     報酬風險比 = 交易總盈虧/最大盈虧回落_MDD
    # else:
    #     報酬風險比='資料不足無法計算'

if choice == choices[1] :   #'大台指期貨2024.12到期: 2023.12 至 2024.4.11':
    交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_大台指期貨()

    # 交易總盈虧 = OrderRecord.GetTotalProfit()*200          ## 取得交易總盈虧
    # 平均每次盈虧 = OrderRecord.GetAverageProfit() *200       ## 取得交易 "平均" 盈虧(每次)
    # 平均投資報酬率 = OrderRecord.GetAverageProfitRate()    ## 取得交易 "平均" 投資報酬率(每次)  
    # 平均獲利_只看獲利的 = OrderRecord.GetAverEarn() *200            ## 平均獲利(只看獲利的) 
    # 平均虧損_只看虧損的 = OrderRecord.GetAverLoss()*200             ## 平均虧損(只看虧損的)
    # 勝率 = OrderRecord.GetWinRate()              ## 勝率
    # 最大連續虧損 = OrderRecord.GetAccLoss()*200              ## 最大連續虧損
    # 最大盈虧回落_MDD = OrderRecord.GetMDD()*200                  ## 最大利潤(盈虧)回落(MDD). 這個不是一般的 "資金" 或 "投資報酬率" 的回落
    # if 最大盈虧回落_MDD>0:
    #     報酬風險比 = 交易總盈虧/最大盈虧回落_MDD
    # else:
    #     報酬風險比='資料不足無法計算'

if choice == choices[2] :   #'小台指期貨2024.12到期: 2023.12 至 2024.4.11':
    交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_小台指期貨()
    # 交易總盈虧 = OrderRecord.GetTotalProfit()*50          ## 取得交易總盈虧
    # 平均每次盈虧 = OrderRecord.GetAverageProfit() *50       ## 取得交易 "平均" 盈虧(每次)
    # 平均投資報酬率 = OrderRecord.GetAverageProfitRate()    ## 取得交易 "平均" 投資報酬率(每次)  
    # 平均獲利_只看獲利的 = OrderRecord.GetAverEarn() *50            ## 平均獲利(只看獲利的) 
    # 平均虧損_只看虧損的 = OrderRecord.GetAverLoss()*50             ## 平均虧損(只看虧損的)
    # 勝率 = OrderRecord.GetWinRate()              ## 勝率
    # 最大連續虧損 = OrderRecord.GetAccLoss()*50              ## 最大連續虧損
    # 最大盈虧回落_MDD = OrderRecord.GetMDD()*50                  ## 最大利潤(盈虧)回落(MDD). 這個不是一般的 "資金" 或 "投資報酬率" 的回落
    # if 最大盈虧回落_MDD>0:
    #     報酬風險比 = 交易總盈虧/最大盈虧回落_MDD
    # else:
    #     報酬風險比='資料不足無法計算'
    # 交易總盈虧 = OrderRecord.GetTotalProfit()*1000          ## 取得交易總盈虧
    # 平均每次盈虧 = OrderRecord.GetAverageProfit()*1000         ## 取得交易 "平均" 盈虧(每次)
    # 平均投資報酬率 = OrderRecord.GetAverageProfitRate()    ## 取得交易 "平均" 投資報酬率(每次)  
    # 平均獲利_只看獲利的 = OrderRecord.GetAverEarn()*1000              ## 平均獲利(只看獲利的) 
    # 平均虧損_只看虧損的 = OrderRecord.GetAverLoss()*1000              ## 平均虧損(只看虧損的)
    # 勝率 = OrderRecord.GetWinRate()              ## 勝率
    # 最大連續虧損 = OrderRecord.GetAccLoss()*1000               ## 最大連續虧損
    # 最大盈虧回落_MDD = OrderRecord.GetMDD()*1000                   ## 最大利潤(盈虧)回落(MDD). 這個不是一般的 "資金" 或 "投資報酬率" 的回落
    # if 最大盈虧回落_MDD>0:
    #     報酬風險比 = 交易總盈虧/最大盈虧回落_MDD
    # else:
    #     報酬風險比='資料不足無法計算'



# OrderRecord.GetCumulativeProfit()         ## 累計盈虧
# OrderRecord.GetCumulativeProfit_rate()    ## 累計投資報酬率

##### 将投資績效存储成一个DataFrame並以表格形式呈現各項績效數據
# 在程式交易部分之後，添加以下績效顯示程式碼

# 績效儀表板
if len(OrderRecord.Profit) > 0:
    st.subheader("績效分析")
    
    # 計算績效指標
    total_return = (1 + pd.Series(OrderRecord.Profit_rate)).prod() - 1
    annual_return = np.mean(OrderRecord.Profit_rate) * 252
    volatility = np.std(OrderRecord.Profit_rate) * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility != 0 else 0
    
    # 計算最大回撤
    cumulative_returns = (1 + pd.Series(OrderRecord.Profit_rate)).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # 計算勝率
    win_rate = len([x for x in OrderRecord.Profit if x > 0]) / len(OrderRecord.Profit) if len(OrderRecord.Profit) > 0 else 0
    
    # 顯示績效指標
    col1, col2, col3 = st.columns(3)
    col1.metric("總報酬率", f"{total_return:.2%}")
    col2.metric("年化報酬率", f"{annual_return:.2%}")
    col3.metric("夏普比率", f"{sharpe_ratio:.2f}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("最大回撤", f"{max_drawdown:.2%}")
    col2.metric("波動率", f"{volatility:.2%}")
    col3.metric("勝率", f"{win_rate:.2%}")
    
    # 繪製累計報酬曲線
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=KBar_df['time'][:len(cumulative_returns)],
        y=cumulative_returns,
        mode='lines',
        name='策略累計報酬'
    ))
    
    # 添加買賣點標記
    trade_dates = [trade[2] for trade in OrderRecord.GetTradeRecord()]
    trade_prices = [trade[3] for trade in OrderRecord.GetTradeRecord()]
    trade_types = [trade[0] for trade in OrderRecord.GetTradeRecord()]
    
    buy_dates = [date for date, typ in zip(trade_dates, trade_types) if typ == 'Buy']
    sell_dates = [date for date, typ in zip(trade_dates, trade_types) if typ == 'Sell']
    
    if buy_dates:
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=[cumulative_returns[KBar_df['time'] == date].values[0] for date in buy_dates],
            mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            name='買入點'
        ))
    
    if sell_dates:
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=[cumulative_returns[KBar_df['time'] == date].values[0] for date in sell_dates],
            mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            name='賣出點'
        ))
    
    fig.update_layout(
        title='策略累計報酬曲線',
        xaxis_title='日期',
        yaxis_title='累計報酬率',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 顯示交易明細
    st.subheader("交易明細")
    trades = []
    for trade in OrderRecord.GetTradeRecord():
        trades.append({
            "類型": trade[0],
            "商品": trade[1],
            "時間": trade[2],
            "價格": trade[3],
            "數量": trade[4],
            "盈虧": trade[5] if len(trade) > 5 else None
        })
    
    st.dataframe(pd.DataFrame(trades))
else:
    st.warning("沒有交易記錄可供分析")






# ###### 累計盈虧 & 累計投資報酬率
# with st.expander("累計盈虧 & 累計投資報酬率"):
#     fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    
#     #### include a go.Bar trace for volumes
#     # fig4.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['MACD_Histogram'], name='MACD Histogram', marker=dict(color='black')),secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
#     fig4.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MACD+1:], y=KBar_df['Signal_Line'][last_nan_index_MACD+1:], mode='lines',line=dict(color='orange', width=2), name='訊號線(DEA)'), 
#                   secondary_y=True)
#     fig4.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MACD+1:], y=KBar_df['MACD'][last_nan_index_MACD+1:], mode='lines',line=dict(color='pink', width=2), name='DIF'), 
#                   secondary_y=True)
    
#     fig4.layout.yaxis2.showgrid=True
#     st.plotly_chart(fig4, use_container_width=True)



# #### 定義圖表
# matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ax1 = plt.subplot(2,1,1)
# ax2 = plt.subplot(2,1,2)




##### 畫累計盈虧圖:
if choice == '富邦金期貨: 2023.4.15 至 2025.4.16':
    OrderRecord.GeneratorProfitChart(choice='stock',StrategyName='MA')
if choice == '大台指期貨2024.12到期: 2023.12 至 2024.4.11':
    OrderRecord.GeneratorProfitChart(choice='future1',StrategyName='MA')
if choice == '小台指期貨2024.12到期: 2023.12 至 2024.4.11':
    OrderRecord.GeneratorProfitChart(choice='future2',StrategyName='MA')
if choice == '華碩: 2023.4.17至2025.4.16':
    OrderRecord.GeneratorProfitChart(choice='stock',StrategyName='MA')
if choice == '聯電期貨: 2023.4.17至2025.4.16':
    OrderRecord.GeneratorProfitChart(choice='stock',StrategyName='MA')

    

# matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# plt.figure()

# #### 計算累計績效
# TotalProfit=[0]
# for i in OrderRecord.Profit:
#     TotalProfit.append(TotalProfit[-1]+i)

# #### 繪製圖形
# if choice == '台積電: 2022.1.1 至 2024.4.9':
#     # ax.plot( TotalProfit[1:]  , '-', marker='o', linewidth=1 )
#     plt.plot( TotalProfit[1:]*1000  , '-', marker='o', linewidth=1 )
# if choice == '大台指2024.12到期: 2024.1 至 2024.4.9':
#     # ax.plot( TotalProfit[1:]  , '-', marker='o', linewidth=1 )
#     plt.plot( TotalProfit[1:]*200  , '-', marker='o', linewidth=1 )


# ####定義標頭
# # # ax.set_title('Profit')
# # ax.set_title('累計盈虧')
# # ax.set_xlabel('交易編號')
# # ax.set_ylabel('累計盈虧(元/每股)')
# plt.title('累計盈虧(元)')
# plt.xlabel('交易編號')
# plt.ylabel('累計盈虧(元)')
# # if choice == '台積電: 2022.1.1 至 2024.4.9':
# #     plt.ylabel('累計盈虧(元/每股)')
# # if choice == '大台指2024.12到期: 2024.1 至 2024.4.9':
# #     plt.ylabel('累計盈虧(元/每口)')

# #### 设置x轴的刻度
# ### 获取TotalProfit的长度
# length = len(TotalProfit)
# ### 创建新的x轴刻度列表，每个值都加1
# new_ticks = range(1, length + 1)
# ### 应用新的x轴刻度
# plt.xticks(ticks=range(length), labels=new_ticks)

# #### 顯示繪製圖表
# # plt.show()    # 顯示繪製圖表
# # plt.savefig(StrategyName+'.png') #儲存繪製圖表
# ### 在Streamlit中显示
# st.pyplot(plt)






##### 畫累計投資報酬率圖:
OrderRecord.GeneratorProfit_rateChart(StrategyName='MA')
# matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# plt.figure()

# #### 計算累計計投資報酬
# TotalProfit_rate=[0]
# for i in OrderRecord.Profit_rate:
#     TotalProfit_rate.append(TotalProfit_rate[-1]+i)

# #### 繪製圖形
# plt.plot( TotalProfit_rate[1:]  , '-', marker='o', linewidth=1 )
# # if choice == '台積電: 2022.1.1 至 2024.4.9':
# #     # ax.plot( TotalProfit[1:]  , '-', marker='o', linewidth=1 )
# #     plt.plot( TotalProfit_rate[1:]  , '-', marker='o', linewidth=1 )
# # if choice == '大台指2024.12到期: 2024.1 至 2024.4.9':
# #     # ax.plot( TotalProfit[1:]  , '-', marker='o', linewidth=1 )
# #     plt.plot( TotalProfit_rate[1:]  , '-', marker='o', linewidth=1 )


# ####定義標頭
# plt.title('累計投資報酬率')
# plt.xlabel('交易編號')
# plt.ylabel('累計投資報酬率')
# # if choice == '台積電: 2022.1.1 至 2024.4.9':
# #     plt.ylabel('累計投資報酬率')
# # if choice == '大台指2024.12到期: 2024.1 至 2024.4.9':
# #     plt.ylabel('累計投資報酬率')

# #### 设置x轴的刻度
# ### 获取TotalProfit的长度
# length = len(TotalProfit_rate)
# ### 创建新的x轴刻度列表，每个值都加1
# new_ticks = range(1, length + 1)
# ### 应用新的x轴刻度
# plt.xticks(ticks=range(length), labels=new_ticks)

# #### 顯示繪製圖表
# # plt.show()    # 顯示繪製圖表
# # plt.savefig(StrategyName+'.png') #儲存繪製圖表
# ### 在Streamlit中显示
# st.pyplot(plt)


#%%
####### (7) 呈現即時資料 #######






   