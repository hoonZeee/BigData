import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%% 산점도
'''
다중변수 데이터 (다변량 데이터)
- 변수가 2개 이상인 데이터 ex 키와 몸무게
- iris 데이터 셋은 5개의 변수로 구성된 데이터셋
- 다중변수 데이터에에서 변수는 컬럼으로 표현이 되고 개별 관측값들이 행을 이룬다.

핵심 : 변수간의 관계 파악

데이터 분석할때 일반적인 절차
- 각 컬럼에 대해 단일 변수 데이터 분석 실시
- 이후 변수들 간의 관계 분석
- 다중 변수 데이터 분석에는 다양한 기법을 사용하며, 기본적인 탐색 기법인 산점도와 상관 분석 학습

두 변수 사이의 산점도
- 산점도 : 두 변수로 구성된 데이터의 분포를 시각적으로 확인하는 그래프
- 관측값들의 분포를 통해 두 변수 간의 관계를 파악하는데 유용하다.

mtcars 데이터셋
- 여러 자동차 모델의 스펙 정보 포함
- 자동차의 중량wt 와 연비 mpg  사이의 관계를 산점도를 통하여 분석.
'''

#%%

df = pd.read_csv("/Users/ijihun/bigData/mtcars.csv")
df

#기본 산점
df.plot.scatter(x='wt',y='mpg')
plt.show()

#매개변수 조정 산점도
df.plot.scatter(x='wt',
                y='mpg',
                s=50,
                c='red',
                marker='s'
                )
plt.show()

#%%

'''
산점도는 두개의 변수로 비교하지만, 만약 여러개를 비교해야한다면? 두개씩 짝지어야하나?
이때 여러 변수 간의 짝지어진 산점도를 한번에 그리는 방법
'''
#%%

vars=['mpg','disp','drat','wt']
pd.plotting.scatter_matrix(df[vars])
plt.show()

#%%

df = pd.read_csv("/Users/ijihun/bigData/iris.csv")

dict = {'setosa':'red', 'versicolor':'green','virginica':'blue'}
colors = list(dict[key] for key in df['Species'])

df.plot.scatter(x='Petal_Length',
                y='Petal_Width',
                s=30,
                c=colors,
                marker='o')
plt.show()


#%% 산점도에 그룹 범례 표현하기

fig,ax = plt.subplots()

for label, data in df.groupby('Species'):
    ax.scatter(x=data['Petal_Length'],
               y=data['Petal_Width'],
               s=30,
               c=dict[label],
               marker='o',
               label=label
               )

ax.set_xlabel('Petal_Length')
ax.set_ylabel('Petal_Width')
ax.legend()
plt.show()


#%%

'''
중량이 증가할 수록 연비는 감소하는 추세가 보인다. 이 추세가 선모양을 띄면 '선형적 관계'라고 한다.
'강한 선형적 관계'
'약한 선형적 관계'
- 변수간의 선형적 관계 정도를 수치적으로 나타내는 방법이 

피어슨 상관계수 r 
-1 <= r <= 1  
r > 0 : 양의 상관관계 x,y둘다 증가
r < 0 : 음의 상관간계 하나증가 하나감소
r 이 1 이나 -1에 가까울수록 x,y의 상관성이 높다.
'''


#%%

beers = [5,2,9,8,3,7,3,5,3,5]
bal = [0.10,0.03,0.19,0.12,0.04,0.095,0.07,0.06,0.02,0.05]

dict = {'beer':beers, 'bal':bal}

bb = pd.DataFrame(dict)
bb

bb.plot.scatter(x='beer',y='bal',title='Beers~Blood Alchol Level')

#회귀식 계산
m, b = np.polyfit(beers, bal, 1)
#회귀식 출력
plt.plot(beers, m* np.array(beers)+b)
plt.show()

#두 변수 간 상관계수 계산
bb['beer'].corr(bb['bal'])


#%%
df2 = pd.read_csv("/Users/ijihun/bigData/iris.csv")
df2.columns

df2 = df2.loc[:, ~df2.columns.isin(['Species'])]
df2.columns

df2.corr()






















