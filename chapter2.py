#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:14:54 2025

@author: ijihun
"""

import pandas as pd
import numpy as np

#%% 판다스 시리즈 객체 생성하기

age = pd.Series([25, 34, 19 ,45 ,60])
age
type(age)

data = ['spring','summer','april','winter']
season = pd.Series(data)  # data 는 list니까 pd.Series()를 통해 시리즈로 바꿔줘야함.
season.iloc[2] #Series 는 numpy배열과 동일하게 인덱스를 이용하여 원소를 다룰 수 있다.

#%% 판다스 데이터프레임 생성

score = pd.DataFrame([[10,11,12,13,14],
                     [21,22,23,24,25],
                     [31,32,33,34,35]])

score
type(score)

score.index #행방향 인덱스 (새로)
score.columns #열방향 인덱스 (가로)

score.iloc[1,2] 

#%% 넘파이 배열 <-> 판다스 배열

#넘파이 1차원 배열을 판다스로 변환
w_np = np.array([65.4,71.3,np.nan,57.8])
weight = pd.Series(w_np)
weight

# 판다스 시리즈를 넘파이로 변환
w_np = pd.Index.to_numpy(weight)
w_np


#넘파이 2차원 배열을 판다스로 변환
s_np = np.array([[1,2,3,4,5],
                 [11,12,13,14,15],
                 [21,22,23,24,25]])

score2 = pd.DataFrame(s_np)
score2

#데이터 프레이를 넘파이 2차원 배열로
score_np = score2.to_numpy()
score_np

# 판다스의 위치에 따른 인덱싱
# score.loc['spring'] = 값의 레이블에 의한 인덱싱
# score.iloc[2] = 값의 절대 위치에 의한 인덱

#%% 행과 열에 레이블을 부여하는 방법

#시리즈에 레이블 부여
age = pd.Series([25,34,19,45])
age
age.index= ['John','Jane','Tom','Luka']
age

age.loc['John']
age.iloc[0]

#데이터프레임에 레이블 부여
score = pd.DataFrame([[10,11,12,13,14],
                      [21,22,23,24,25],
                      [31,32,33,34,35]])

score.index=['A','B','C']
score.columns = ['a','b','c','d','e']

score.iloc[1,2]
score.loc['A']  #컬럼에 지정하려면 score.loc[:,'a']

# 판다스 개요
# - 절대 위치 인덱스는 중복되지 않으며, 시스템에 의해 자동 관리된다.
# - 레이블 인덱스는 사용자가 임의로 지정할 수 있으며, 중복이 존재할 수 있다.


#%% 실습해보기

# 1. 데이터프레임에 저장
tb = pd.DataFrame([[-0.0, 0.0, -0.1, -0.2],
                   [1.8, 2.0, 1.6, 1.6],
                   [6.4, 6.8, 5.8, 5.9],
                   [12.3, 12.9, 11.5, 11.5],
                   [17.9, 18.5, 17.1, 17.1],
                   [22.2, 22.8, 21.6, 21.5]])

tb.index = ['1월','2월','3월','4월','5월','6월']
tb.columns = ['전북','전주','군산','부안']

#2. 절대위치로 전주의 3월 평균 기온
print(tb.iloc[2,0])

#3. 절대위치로 부안의 4월 평균 기온
print(tb.iloc[3,3])

#4. 레이블 인덱스를 이용하여 군산의 1월 평균 기온을 출력하시오.
print(tb.loc['1월','군산'])

#5. 레이블 인덱스를 이용하여 전북의 6월 평균 기온을 출력하시오.
print(tb.loc['6월','전북'])




