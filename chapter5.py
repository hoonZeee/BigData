import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#%%

#범주형 데이터 : 크기를 가지지 않는 것, 평균 계산이나 대소 비교 불가능 -> spring, winter 이런거

#단일 변수 범주형 데이터 탐색

favorite = pd.Series(['WINTER', 'SUMMER', 'SPRING', 'SUMMER', 'SUMMER',
                      'FALL', 'FALL', 'SUMMER', 'SPRING', 'SPRING'])

favorite
favorite.value_counts() # 도수분표 계산
favorite.value_counts()/favorite.size

#막대 그래프 작성
fd = favorite.value_counts()
fd

fd.plot.bar(xlabel='Season',
            ylabel='Frequency',
            rot=0,
            title='Favorite Season'
            )

plt.show()

favorite = pd.Categorical(favorite,categories=['SPRING','SUMMER','FALL','WINTER'],ordered=True)
fd = favorite.value_counts()

fd.plot.bar(xlabel='Season',
            ylabel='Frequency',
            rot=0, # x축 글자 회전 각도 (세글자아님)
            title='Favorite Seasons'
            )

plt.show()


#가로 막대 그리기
fd.plot.barh(xlabel='Frequency',  #barh : 가로 막대 그리
            ylabel='Seasons',
            rot=0,
            title='Favorite Seasons'
            )
plt.subplots_adjust(left=0.2)
plt.show()

#원그래프
fd.plot.pie(ylabel='',
            autopct='%1.1f%%',
            title='Favorite Season')
plt.show()

#%%
#문제 : mpg.csv에서 cls의 개수에 따른 분포에서 최대 5위까지를 cls 막대와 원그래프로 그려보기
mpg = pd.read_excel("/Users/ijihun/bigData/mpg.xlsx")
cl = mpg['category'].value_counts().head(5)

cl.plot.bar(xlabel='Species',
            ylabel='Frequency',
            rot=0,
            title='cls bar')
plt.show()

cl.plot.pie(ylabel='',
            autopct='%1.0f%%',
            title='cls pie')
plt.show()





