import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

titanic = pd.read_csv("/Users/ijihun/bigData/titanic.csv")

#%%

#연속형 데이터는 관측값들이 크기를 가지므로, 범주형 데이터보다 다양한 분석 방법 존재

'''
중앙값(median)
평균과 중앙값은 같을 수 있지만, 대부분 다르다.
평균은 일부 큰값이나 작은 값들에 영향을 많이 받는다.
중앙값은 이상치에 크게 영향을 받지 않는다.
'''

'''
절사평균(trimmed mean)
데이터 하위 %와 상위%를 제외한후 남은 값들로 평균을 계산하는 방법
극단적인 값의 영향을 줄이기 위한 방법
'''

#%%
from scipy import stats

ds = [60,62,64,65,68,69]
ds2 = [60,62,64,65,68,69,120]
weight = pd.Series(ds)
weight_heavy = pd.Series(ds2)

print(weight.mean(), weight_heavy.mean())
print(weight.median(),weight_heavy.median())
print(stats.trim_mean(weight,0.2),stats.trim_mean(weight_heavy,0.2))
print(weight_heavy.quantile(0.25))
print(weight_heavy.quantile([0.25,0.5,0.75]))
print(weight_heavy.std())
print(weight_heavy.var())

'''
분산과 표준편차
값이 크다면 : 학생들 간 실력차이가 크다.
값이 작다면 : 학생들 간 실력차이가 작다.
'''


#%% 히스토그램
t = titanic.dropna(subset='age',axis=0) #dropna(axis=1) nan이 있는 열자체를 삭제
titanic['age'].plot.hist(ylabel="Frequency")
plt.show()

#%%

titanic.query('sex=="male"')['age'].plot.hist(ylabel="man",alpha=0.5)
titanic.query('sex=="female"')['age'].plot.hist(ylabel="woman",alpha=0.5)
plt.show()

#%%
titanic['age'].plot.hist(bins=6,color="green")
plt.show()
titanic['age'].value_counts(bins=6,sort=False) # bins : 총 6개로



#%% 한 화면에 여러 개의 그래프 출력하기

df=  pd.read_csv("/Users/ijihun/bigData/iris.csv")

#화면 분할 처리

fig, axes = plt.subplots(nrows=2,ncols=2)

df['Petal_Length'].plot.hist(ax=axes[0,0])
df['Petal_Length'].plot.box(ax=axes[0,1])

fd = df['Species'].value_counts()
fd.plot.pie(ax=axes[1,0])
fd.plot.barh(ax=axes[1,1])

fig.suptitle('Multiple Graph Example', fontsize=14)

plt.show()


#%% 실습

#실전 1

plt.rcParams['font.family'] = 'AppleGothic' 

pr = pd.Series(['등산','낚시','골프','수영','등산','등산','낚시','수영','등산','낚시','수영','골프'])


fig,axes = plt.subplots(nrows=1, ncols=2)

pr1 = pr.value_counts()

pr1.plot.bar(xlabel="Species",
             ylabel="counts",
             rot=0,
             title="hobby",
             ax=axes[0]
             )


pr1.plot.pie(ylabel='',
             autopct='%1.0f%%',
             title='hobby',
             ax=axes[1]
             )
plt.subplots_adjust(left=0.2)

plt.show()

#%%

pr_category = pd.Categorical(pr, categories=['낚시','골프','수영','등산'])

pr2 = pr_category.value_counts().sort_values(ascending=False)

fig,axes = plt.subplots(nrows=1,ncols=2)

pr2.plot.bar(ax=axes[0])
pr2.plot.pie(ax=axes[1])

plt.show()


#%%

bs = pd.read_csv("/Users/ijihun/bigData/bostonHousing.csv")

hp = pd.Series(bs['medv'])

print(hp.mean())
print(hp.median())
print(hp.quantile([0.25,0.5,0.75]))


q1 = hp[hp <= hp.quantile(0.25)].mean()
q2 = hp[(hp > hp.quantile(0.25))&(hp<= hp.quantile(0.5))].mean()
q3 = hp[(hp > hp.quantile(0.5) )&(hp<= hp.quantile(0.75))].mean()
q4 = hp[hp > hp.quantile(0.75)].mean()

mean_hp = pd.Series([q1,q2,q3,q4])

mean_hp.plot.bar(xlabel='mean',
                 ylabel='Frequency',
                 rot=0,
                 title='mean HP'
                 )

plt.show()

hp.plot.box()
plt.show()

hp.plot.hist(ylabel='counts',bins=8)

plt.show()

#%%

up = pd.read_csv("/Users/ijihun/bigData/user_behavior_dataset.csv")






















