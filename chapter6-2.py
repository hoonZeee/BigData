import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
선그래프 작성
- x축이 시간이면 '시계열 데이터' 라고 한다.
'''

late = pd.Series([5,8,7,9,4,6,12,13,8,6,6,4],
                 index = list(range(1,13)))

late.plot(title='Late Student per month',
          xlabel='month',
          ylabel='fequency',
          linestyle='solid') #실선
plt.show()

late.plot(title='Late student per month',
          xlabel='month',
          ylabel='frequency',
          linestyle='dashed', #점선
          marker='o')
plt.show()


#%% 복수의 선그래프

late1= [5,8,7,9,4,6,12,13,8,6,6,4]
late2 = [4,6,5,8,7,8,10,11,6,5,7,3]

dict = {'late1':late1, 'late2':late2}

mul = pd.DataFrame(dict,index= list(range(1,13)))

mul.plot(title='Late Student per month',
         xlabel='month',
         ylabel='frequency',
         marker='o')
plt.legend(loc='upper right')
plt.show()

#%%
from pandas.api.types import CategoricalDtype

plt.rcParams['font.family'] = 'AppleGothic' 

df = pd.read_csv("/Users/ijihun/bigData/bostonHousing.csv")
df = df[['crim','rm','dis','tax','medv']]
df

titles = ['1인당 범죄율', '방의 개수','직업센터까지의 거리','재산세','주택가격']

grp = pd.Series(['M' for i in range(len(df))])
grp.loc[df['medv'] >= 25.0] = 'H'
grp.loc[df['medv']<= 17.0] = 'L'
df['grp'] = grp

new_order=['H','M','L']
new_dtype = CategoricalDtype(categories=new_order,ordered=True)
df['grp'] = df['grp'].astype(new_dtype)


df.shape
df.head()
df.dtypes
df.grp.value_counts(sort=False)


fig, axes = plt.subplots(nrows=2,ncols=3)
fig.subplots_adjust(hspace=0.5, wspace=0.3)


#각 분할 영역에서 그래프 작성
for i in range(5):
    df[df.columns[i]].plot.hist(ax=axes[i//3,i%3],
                                ylabel='',xlabel=titles[i])

fig.suptitle('Histogram',fontsize=14)
plt.show()


#%% 상자그림

fig, axes = plt.subplots(nrows=2, ncols=3)
fig.subplots_adjust(hspace=0.5,wspace=0.3)

for i in range(5):
    df[df.columns[i]].plot.box(ax=axes[i//3,i%3],
                               label=titles[i])
    
fig.suptitle('Boxplot', fontsize=14)
plt.show()


#%%


fig, axes = plt.subplots(nrows=2, ncols=3)
fig.subplots_adjust(hspace=0.5,wspace=0.3)

for i in range(5):
    df.boxplot(column=df.columns[i],by='grp',grid=False,
               ax=axes[i//3,i%3], xlabel= titles[i])
    
fig.suptitle('Boxplot by group',fontsize=14)
plt.show()


#%%
pd.plotting.scatter_matrix(df.iloc[:,:5])
plt.show()


#%%
dict = {'H':'red','M':'green','L':'gray'}
colors=list(dict[key] for key in df.grp)

pd.plotting.scatter_matrix(df.iloc[:,:5],c=colors)
plt.show()

#%%
df.iloc[:,:5].corr()





