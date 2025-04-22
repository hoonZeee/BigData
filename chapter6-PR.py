import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/ijihun/bigData/state_x77.csv")

#%% 문맹률과 수입 사이에 연관성이 있는시 산점도와 상관계수를 통해 분석
df.plot.scatter(x='Illiteracy',y='Income')
plt.show()

df['Illiteracy'].corr(df['Income'])

#%% 수입과 기대수명 사이에 연관성이 있는지

df.plot.scatter(x='Income',y='Life_Exp')
plt.show()

df['Income'].corr(df['Life_Exp'])

#%% 전체 변수에 대한 다중 산점도 작성 


df2= df.loc[:, ~df.columns.isin(['State'])]
df2.columns

#%%
sm = pd.read_csv("/Users/ijihun/bigData/user_behavior_dataset.csv")

sm = sm.iloc[:,3:10]
sm.columns=['App_Usage_Time','Screen_On_Time','Battery_Drain',
            'No_of_Apps','Data_Usage','Age','Gender']

#%%
sm.plot.scatter(x='App_Usage_Time',y='No_of_Apps')
plt.show()



dict = {'Male':'red','Female':'blue'}
colors = list(dict[key] for key in sm['Gender'])
sm.plot.scatter(x='App_Usage_Time',
                y='No_of_Apps',
                s=30,
                c=colors,
                marker='o'
                )
plt.show()

#%%

pd.plotting.scatter_matrix(sm.loc[:,['App_Usage_Time', 'Screen_On_Time',\
                                     'Battery_Drain','No_of_Apps']])
plt.show()

#%%

sm['App_Usage_Time'].corr(sm['No_of_Apps'])

sm2 = sm.loc[:, ~sm.columns.isin(['Gender'])]

#%%
sm3 = sm2.corr()
top_corr = sm3[sm3 < 1].stack().sort_values(ascending=False).head(1)




