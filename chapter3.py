#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 21:24:14 2025

@author: ijihun
"""

import pandas as pd

#%% 시리즈 정보 확인하기

temp = pd.Series([1,2,3,4,5,6,7,8,9,10,11,12])
temp.index = ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']

#시리즈 정보 확인하기
print(temp.size)  # 배열의 크기(값의 개수)
print(len(temp))  # 배열의 크기(값의 개수)
print(temp.shape) # 배열의 형태
print(temp.dtype) # 배열의 타입

#%% 인덱싱과 슬라이싱

#인덱싱
temp.iloc[2]
temp.loc['3월']
temp.iloc[[3,5,7]]
temp.loc[['4월','6월','8월']]

#슬라이싱
temp.iloc[5:8]
temp.loc['6월':'9월']
temp.iloc[:4]
temp.iloc[9:]
temp.iloc[:]

#조건문 슬라이싱
temp.loc[temp >= 15]
temp.loc[(temp>15) & (temp<25)]
temp.loc[(temp<15)|(temp>25)] # 괄호없는 조건문은 에러발생

#조건문
temp.where(temp>=10) # 조건에 맞지않는 조건들은 결측값(NaN)으로 표시된다. 
temp.where(temp>=10).dropna() # 결측값들을 제거한다.


#%% 시리즈 객체의 산술연산

#각각 값에 개별적으로 적용된다.
#두 시리즈 객체 간의 산술 연산은 인덱스가 같은 값들끼리 수행된다.

temp+1
2*temp+1
temp+temp
temp.loc[temp>=10]+1 # 10보다 큰것들에 대해서 +1 

#통계관련 메서드
temp.sum()
temp.mean()
temp.median() #짝수라면 가운데 두 값의 평균
temp.max() 
temp.min()
temp.std() #표준편차
temp.var() #분산
temp.describe() #기초통계정보


#시리즈 객체 내용 변경
salary = pd.Series([20,15,18,30])
score = pd.Series([75,80,90,65], index = ['kor','eng','math','soc'])
salary
score

#값의 변경
score.iloc[0] = 85
score

score.loc['soc'] = 60
score

score.loc[['eng','math']] = [70,80] # 값 변경에도 [] 
score

#값의 추가는 loc[]를 통해서만 가능하다.
score.loc['phy'] = 50
score

# score.iloc[6] = 40 이거 오류발생

#값의 추가 (레이블 인덱스가 없는 경우)
next_idx = salary.size
#salary.iloc[next_idx] = 33 이러면 오류남 값 추가는 무조건 loc로 해야됨
salary.loc[next_idx] = 33
salary

new = pd.Series({'MUS' : 95}) #딕셔너리 기반 시리즈 생성 , Index= MUS 값은 95
score._append(new)
score
score = score._append(new) #잠깐! .loc[] 같은건 값을 바꾼거니까 = 필요없는데, _append처럼 값을 추가하는건 = 해줘야함.
score


salary = salary._append(pd.Series([66]), ignore_index=True) # ignore 없으면 0 추가됨. 추가되는 인데그에 맞춰 새로운 인덱스 부여
salary

#객체 삭제
score = score.drop('phy') # 이거 score.drop('phy',inplace=True) 이렇게 해도됨
score

salary = salary.drop(1)
salary

#시리즈 객체 복사
score2= score # 동일한 객체
score3=score.copy() #독립된 객


#%% 실습해보기
test = pd.Series([781, 650, 705, 406, 580, 450, 550, 640], index= ['A','B','C','D','E','F','G','H'])
test

test.where((test<500) | (test>700)).dropna()
test.loc[(test<500) | (test>700)]

test.loc[test>test['B']]

test.loc[test<600]

test.loc[test<600] * 1.2

test.mean()
test.sum()
test.std()

test.loc[['A','C']] = 810,820
test

test.loc['J'] = 400
test

test.drop('J',inplace=True)
test

test2 = test.copy()
test2 + 500
test2





