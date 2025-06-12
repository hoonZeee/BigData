import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

'''
seaborn은 데이터 시각화 라이브러리이다.
불연속 데이터용 팔레트
연속 데이터용 팔레트
'''


tips = sns.load_dataset('tips')
plt.rc('font',family='AppleGothic')
sns.set_palette('Set2') #색상 설

sns.barplot(x='day', y='total_bill', data = tips)
#요일별로 total_bill의 평균을 막대그래프로 그린다.
#seaborn은 자동으로 에러바를 계산해준다.
plt.show()

sns.barplot(x='day', y='total_bill',data=tips, palette='hls')
plt.show()

sns.countplot(x='day', data= tips, palette='Set2')
# 요일의 개수를 막대그래프로 표현
plt.show()

sns.countplot(x='day', data=tips, palette='Set2', hue='smoker')
#smoker를 기준으로 색깔을 나눠서 표현 즉 위에 그래프에서 흡연자,비흡연자로 나눈거임
plt.show()

sns.boxplot(x='sex', y='tip', data=tips, palette='pastel')
# IQR = Q3 - Q1 : 중앙값
# Q3 + 1.5 * IQR : 최대 이상치
# Q1 - 1.5 * IQR : 최소 이상치
plt.show()


#%%

import numpy as np

data = np.array([1,2,5,7,8,10,15,18,20,28])

Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)

IQR = Q3 - Q1

lower_whisker = Q1 - 1.5 * IQR
higher_whisker = Q3 + 1.5 * IQR

print("하단 수염 ", lower_whisker)
print("상단 수엽", higher_whisker)

#%%
sns.boxenplot(data=tips, orient='h', palette='Set1')
# oreint = h : horizontal 수평방
plt.show()

#%%
'''
KDE(Kernel Density Estimation)
- 데이터의 분포를 부드러운 곡선으로 시각화
- 연속형 데이터에 사용
- 2d KDE
'''

sns.kdeplot(data=tips, x='total_bill', y='tip', color='r', cmap='Reds',shade=True)
#히스토그램보다 데이터가 어디 몰려있는지 파악하기 위함
plt.show()


#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

mpg = pd.read_excel('/Users/ijihun/bigData/mpg.xlsx')
mpg2 = mpg.query('category=="compact"') #compact 인거만 묶어서, 근데 여기 코드에선 안쓰였네?
sns.scatterplot(data=mpg,  
                x='displ', 
                y='hwy',
                size=mpg['cty']*10, 
                palette='viridis',
                hue='displ', # 배기량에 따른 색깔구분
                legend=False
                )
plt.show()

#%%
iris = sns.load_dataset('iris')
sns.scatterplot(data=iris,
                x='sepal_length',
                y='petal_length',
                hue='species')

plt.show()


#%%
'''
조인트 플롯
- 두 변수가 연속형 실수형일 경우
- 차트의 가장자리에 각 변수의 히스토그램 표현
'''
sns.jointplot(x='sepal_length', y='petal_length', data=iris, palette='viridis')
plt.show()
#산점도 : 어디에 가장 많이 몰려있고, 양의 상관관계인지 음의 상관관계인지 분석

#%%
'''
violinplot: 세로 방향 커널 밀도 히스토그램, 왼쪽과 오른쪽 대칭 바이올린 모양
stripplot: 스캐터 플롯처럼 모든 데이터를 점으로 표현 jitter=True로 설정하면 가로축 상의 위치를 무작위로
바꾸어서 데이터가 많아도 겹치지 않는다.
swarmplot: stripplot과 비슷하지만 데이터를 나타내는 점이 겹치지 않도록 옆으로 이동하여 표현
'''

tips= sns.load_dataset('tips')
sns.catplot(data=tips, x='day',y='total_bill', hue='day')
#catplot은 이것들 상위적인거고 kind= 에따라서 변하는게 보임
plt.show()

#swarmplot
sns.catplot(x='day',y='total_bill',hue='sex', kind='swarm',data=tips)
plt.show()


sns.violinplot(x='day',y='total_bill',data=tips)
plt.show()


#%%
'''
다변량 시각화
- 연속형 변수가 3개 이상이면 페어플롯을 사용
- 데이터 프레임을 인수로 받아 grid 형태로 각 데이터열의 조합에 대해 스캐터 플롯을 표현
- 같은 데이터가 만나는 대각선 영역에 해당 데이터의 히스토그램을 그린다
'''
iris2=iris.drop(['species'],axis=1)
iris_corr = iris2.corr()
sns.heatmap(data=iris_corr, 
            annot=True, #각 셀에 수치 표현
            fmt='.2f', # 소수점 둘째 자리까지
            linewidths=.5, #셀 경계선 두
            cmap='Blues')

plt.show()

#%%
from plotnine import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure() # 내부적으로 도화지 생성
flights = sns.load_dataset('flights')


ggplot(flights, aes(x='year', y='month'))\
    + geom_tile(aes(fill='passengers'))\
    + scale_fill_gradientn(colors=['#9ebcda', '#8c6bb1', '#88419d', '#6e016b'])\
    + ggtitle('Heatmap of Flights by plotnine')
# 연도 월 조합의 히트맵 생성
# 승객이 많을 수록 색이 진해진다. 

#%% 실습 문제

iris3 = sns.load_dataset('iris')

q_low = iris['sepal_length'].quantile(0.2)
q_high = iris['sepal_length'].quantile(0.9)


iris3 = iris[(iris['sepal_length'] >= q_low) & (iris['sepal_length'] <= q_high)]

iris3['species'].value_counts().plot.pie(autopct='%1.1f%%')
plt.ylabel("")
plt.title("iris3의 species 를 비율을 pie로 나타내어 보기")
plt.show()

iris3= iris.drop(['species'],axis=1)

iris3_corr = iris3.corr()
sns.heatmap(data = iris3_corr,
            annot=True,
            fmt='.2f',
            linewidths=.5,
            cmap='Blues')
plt.show()








