#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 19:36:00 2025

@author: ijihun
"""

import pandas as pd
import numpy as np

df = pd.read_csv("/Users/ijihun/bigData/iris.csv")


df.info() 
df.shape
df.shape[0] # 행개수
df.shape[1] # 열개수
df.head() # 데이터 일부 보기 5개
df.tail()
df['Species'].unique() #Species의 정보 확인

#컬럼 제외
#Species 컬럼을 제외하고 나머지 컬럼만 
df.loc[:, ~df.columns.isin(['Species'])]

#%% exam.csv
ex = pd.read_csv("/Users/ijihun/bigData/exam.csv")

ex.index
ex.columns
ex.describe()

ex['math'].mean()
ex['science'].std()
ex['nclass'].var()


#정렬하기
ex1 = ex.sort_values(by='math')
ex1 = ex.sort_values(by='math',ascending=False)
ex1.head()

ex['total'] = ex['math'] + ex['science']
ex['test'] = np.where(ex['total']>=160, 'pass', 'fail')

#%%

#조건에 맞는 행 추출하기
ex.query('nclass == 2')
ex.query('nclass == 1 & math>=50')
ex.query('nclass in [1,3,5]')

#%%
import seaborn as sns
w = pd.read_csv("/Users/ijihun/bigData/weather.csv")

#상위 10%만 추출하기
w1 = w['temp'] >= w['temp'].quantile(0.9)
w[w1]

#열 삽입하기 - insesrt는 열삽입만 가능 
w.insert(1,'num1',[n for n in range(3653)] )

tips = sns.load_dataset('tips')
print(tips.dtypes)

tips['smoker_str'] = tips['smoker'].astype(str)
tips.dtypes # 자료형 변환

'''
Category VS Object
category 
- 문자열의 특수한 형태, 몇 가지 번주가 가능한 문자열
- 범주형 데이터는 카테고리로 지정
- 분석할 때 용량과 속도 면에서 매우 효과적
object
- 이름 등과 같은 범위가 큰 문자열
'''

#%%

#열 삭제
w3 = w.drop(['temp'],axis=1) # axis1 이면 열, axis0 이면 행
w3

#행 삭제
w3 = w.drop([3],axis=0)
w3

#행/열 합계 df.sum()

tips1 = tips[['total_bill','tip']]
tips1.sum(axis=0) #열 합계
tips1.sum(axis=1) #행 합계

w[w['max_wind'].isna()]

w5 = w.notnull().sum()
w5












