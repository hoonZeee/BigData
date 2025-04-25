import pandas as pd
import numpy as np

'''
시계열
-시간변수
-시간 변수 흐름에 따른 종속변수의 움직임을 이해하고 예측하는 것을 목표로 한다.
-특정 시점, 기간으로 구분하는 자료형 제공
타임스탬프
- 특정 시점을 의미하는 자료형
-to_datetime()함수로 생성 가능하며 날짜 형태의 자료형을 시계열 타입으로 변환
기간
-일정 기간을 의미
-timestamp를 기간에 따른 자료형으로 이용하고자 할때 사용
'''
#%%

dates = pd.date_range('20201001',periods=6)
dates

df = pd.DataFrame(np.random.randn(6,4),index=dates, columns=list('ABCD'))
df

#%%
file_path= "/Users/ijihun/bigData/timeseries.csv"

df = pd.read_csv(file_path)
df.info()

#%%
df['new_Date'] = pd.to_datetime(df['Date'])
df.info()

df['Year'] = df['new_Date'].dt.year
df['Month'] = df['new_Date'].dt.month
df['Day'] = df['new_Date'].dt.day

df['Date_Yr'] = df['new_Date'].dt.to_period(freq='Y')
df['Date m'] = df['new_Date'].dt.to_period(freq='M')

df.info()


#%%
df.drop('Date',axis=1,inplace=True)
#%%

df.set_index('new_Date',inplace=True)
df

#%%
today = pd.to_datetime('2023-11-13')
df['time_diff'] = today - df.index
df

#%% 시계열 그래프 그리기

file_path= "/Users/ijihun/bigData/time_series2.csv"

df2 = pd.read_csv(file_path)

df2['new_Date'] = pd.to_datetime(df2['date'])
df2.drop('date',axis=1,inplace=True)
df2.set_index('new_Date',inplace=True)

#%%
from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'figure.figsize':(10,7),'figure.dpi':120})

def plot_df2(df2,x,y,title="",xlabel='Date',ylabel='Value',dpi=100):
    plt.figure(figsize=(16,5),dpi=dpi)
    plt.plot(x,y,color='tab:red')
    plt.gca().set(title=title,xlabel=xlabel,ylabel=ylabel)
    plt.show()

plot_df2(df2,x=df2.index,y=df2.value,title='Time series data')






























