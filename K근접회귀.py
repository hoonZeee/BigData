#%%
import numpy as np

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data= np.column_stack((fish_length,fish_weight))
#1이라는 배열 35의 길이를 만들고, 0이라는 배열 14길이를 만든다
fish_target = np.concatenate([np.ones(35),np.zeros(14)])


#%%

#  훈련 세트와 테스트 세트 데이터를 나누는 함수
from sklearn.model_selection import train_test_split 

# 전체 비율을 학습용, 테스트용으로 나누지만 도미/빙어의 비율은 일정하게 유지시키면서 랜덤하게 나눈다.
train_input,test_input,train_target,test_target= \
    train_test_split(fish_data, fish_target,stratify=fish_target,random_state=42)

print(train_input.shape,test_input.shape)


#%%                          
# k- 최근접 학습하기
from sklearn.neighbors import KNeighborsClassifier # knn 분류기 가져오기

kn = KNeighborsClassifier()   # kn 분류기 가져오기
kn.fit(train_input, train_target)   #훈련 데이터로 모델 학습하기. 


print(kn.score(test_input,test_target)) # 테스트 데이터로 정확도 평가 score

#%%
#predict 로 예측하기
print(kn.predict([[25,150]])) #길이가 25 무게가 150인 물고기예측 > 1이면 도미, 0이면 빙어

#예측을 시각화한거
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0],train_input[:,1])
plt.scatter(25,150,marker='^') # 25,150인거 위치 표현
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


#%%
distances, indexes = kn.kneighbors([[25,150]]) # kn.kneighbors로 이웃한 5개를 보여준다.

plt.scatter(train_input[:,0],train_input[:,1]) #이건 전체 훈련 데이터셋
plt.scatter(25,150,marker='^') # 이건 예측하고자 하는 데이터 표
plt.scatter(train_input[indexes,0],train_input[indexes,1],marker='D') #이건 근처 5개 데이터
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


print(train_input[indexes]) #근처 5개 정보
print(train_target[indexes]) # 도민지 빙언지
print(distances) # 거리정

'''
길이, 무게 : 특성값을 일정한 기준으로 맞춰줘야함 
> 데이터 전처리 : 왜냐면, 머신러닝에선 단위나 크기가 다르면 이상하게 학습함

표준점수 : 각 특성값이 평균에서 표춘편차의 몇 배만큼이나 떨어져 있는지를 나타냄, 평균을 빼고 표준편차를 나누어 줌
'''

#%% 표준 점수
mean=np.mean(train_input,axis=0) 
std=np.std(train_input,axis=0)
train_scaled= (train_input-mean) / std #전체 데이터 표존화
new= ([25,150] - mean) / std # 예측할 데이터 표준화
plt.scatter(train_scaled[:,0],train_scaled[:,1]) #표준화된 훈련데이터
plt.scatter(new[0],new[1],marker='^') #표준화된 예측 데이터
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

#%% 표준 점수로 예측 > 표준화된 데이터 : 표준화를 했기때문 스케일 차이를 보정해서 보다 정확한 값을 도출해 낼 수 있다.
# 모든 특성을 0과 1로 맞춰서 공정한 비교가 가능하다.

# train_scaled , train_target 으로 학습하기
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled=(train_input - mean) /std
kn.fit(train_scaled, train_target)
test_scaled= (test_input - mean) / std
new= ([25,150] - mean ) / std
distance,indexes= kn.kneighbors([new])

#%% scaled 로 그래프 그리기

new= ([25,150] - mean ) / std
distance,indexes= kn.kneighbors([new])

plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1],marker='^')
plt.scatter(train_scaled[indexes,0],train_scaled[indexes,1],marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(kn.predict([new]))
#%% 농어의 길이와 무게 k- 최근접 회귀 
#linear regression
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

#%%  데이터 분리
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)

#sklearn사이킷런 모델은 2차원 입력데이터를 요구 > reshape(-1,1)은 행은 자유, 열은 1로 지정한다는 의미
train_input = train_input.reshape(-1,1) 
test_input = test_input.reshape(-1,1) 


#%% 최근접 이웃 개수를 3으로 하는 모델을 훈련

from sklearn.neighbors import KNeighborsClassifier

knr = KNeighborsClassifier(n_neighbors=3) #가까운 3개로 부터 예측

knr.fit(train_input, train_target)

print(knr.predict([[50]])) # 길이가 50인 농어 무게 예측 결과


#%%

distances, indexes = knr.kneighbors([[50]])


plt.scatter(train_input,train_target)
plt.scatter(25,150,marker='^')
plt.scatter(train_input[indexes],train_target[indexes],marker='D')
pred_data = knr.predict([[50]])  # 이거랑 아래50 을 100으로 바꾸면 범위를 벗어나 엉뚱한 값을 예측한다.
plt.scatter(50,pred_data,marker='^')
plt.show()

'''
이때 50에서 100으로 늘리면 엉뚱한 값이 나오는 이유 
KNN은 가장 가까운값에서 흔한값을 예측하는 것이다. 우리 데이터의 최대범위는 44정도인데, 100까지해버리면
훈련 범위를 벗어나서 예측을 하지 못한다.
'''


#%% 선형 회귀를 이용하여 단점 극복

from sklearn.linear_model import LinearRegression

lr = LinearRegression()  #선형 회귀 모델 생성


#%%


from sklearn.model_selection import train_test_split

#훈련 세트와 테스트를 세트로 나눔
train_input, test_input, train_target, test_target = train_test_split(
    perch_length,perch_weight,random_state=42)

#훈련세트와 테스트 세트를 2차원 배열로 나눔
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)


lr.fit(train_input,train_target)

print(lr.predict([[50]]))
print(lr.coef_,lr.intercept_)

#%%
import matplotlib.pyplot as plt 

# 선형 회귀를 이용하여 단점 극복
plt.scatter(train_input, train_target)
plt.plot(train_input, lr.coef_*train_input+lr.intercept_)
plt.scatter(50, lr.predict([[50]]), marker="^") # 여길 100으로 바꾸면?
plt.xlabel('Length')
plt.ylabel('weight')
plt.show()


'''
결론 : 직선의 방정식으로 그었으니까 100쯤이면 그직선의 저 멀리 한100쯤되는 위치, 직선위에 올라와있겠지? 라고 예측
'''


#%%

'''
그렇다면 다항회귀
직선보다는 2차 곡선이면 어떨까? 무게 : a * length 제곱 + b * length + c 
훈련데이터 추가 : 길이의 제곱 데이터 추가
'''

train_poly = np.column_stack((train_input **2, train_input))
test_poly = np.column_stack((test_input **2, test_input))

lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.predict([[50 **2, 50]]))
print("lr", lr.coef_, lr.intercept_)


point = np.arange(15,50)
plt.scatter(train_input, train_target)
plt.plot(point, 1.01*point**2 -21.6*point + 116.05)
plt.scatter(50,1573,marker='^')
plt.ylabel('length')
plt.xlabel('weight')
plt.show()

'''
데이터가 포물선의 형태를 띄니까 선형보다 2차 곡선에 더 적합하다.
'''

#%%

'''
예측모델 테스트 점수 
score() : 테스트 세트에 있는 샘플을 정확하게 분류한 개수의 비율
'''

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state = 42)

train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors = 3)

knr.fit(train_input, train_target)

print("train", knr.score(train_input,train_target))
print("test", knr.score(test_input, test_target))

'''
score 은 회귀에서 결정개수 R제곱을 반환한다. 1에 가까울수록 잚자는거임.
여기서 회귀란: 숫자(연속형 값)을 예측하는 문제. 입력에 따른 결과를 예측
분류 : 이게 스팸메일일까 아닐까? 이 꽃은 어떤 품종일까?
회귀 : 이사람의 키가 이러면 그 몸무게는? 집평수가 이렇다면 그에 따른 가격은? 


결정계수
- 회귀의 경우 평가하는 방법
- R **2(결정계수) 으로 지칭
- R **2 = 1 - (타깃 - 예측)**2 의 합 / (타깃 - 평균)**2의 합
분자와 분모가 비슷하면 0에 가까워지고, 예측이 타깃에 아주 가까워지면 분자가 0에 가까워 지기 때문에 1에 가까워진다.


과대적합 vs 과소적합

과대적합 : 훈련세트에서 점수가 좋았는데 반해 테스트 세트에서 점수가 굉장히 나쁜경우
- 훈련세트에만 잘 맞는 모델

과소적합
- 훈련 세트보다 테스트 세트의 점수가 높거나, 두 점수 모두 낮은경우
- 모델이 너무 단순하여 훈련 세트에 적절히 훈련 되지 않은 경우
- 훈련 세트가 전체 세트를 대표한다고 가정하기에 훈련 세트를 학습하는 것이 더 중요
- k-최근접 이웃 알고리즘 모델을 복잡하게 만드는 방법 - 이웃의 개수를 줄이는 것
'''

#%% 과소적합 코드는 위쪽 코드와 동일한데 n_neighbors = 5로 설정

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state = 42)

train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors = 5) 

knr.fit(train_input, train_target)

print("train", knr.score(train_input,train_target))
print("test", knr.score(test_input, test_target))

'''결과
train 0.9698823289099254
test 0.992809406101064
'''

#%%

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state = 42)

train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors = 3)  # 즉 3으로 해야 트레이닝 셋이 더 높다. 

knr.fit(train_input, train_target)

print("train", knr.score(train_input,train_target))
print("test", knr.score(test_input, test_target))


'''
선형 회귀 - 다중 회귀
- 여러 개의 특성을 사용한 선형 회귀
- 1개의 특성일 때 : 직선이 나온다.
- 2개의 특성일 때 : 평면을 학습한다.(3차 공간 형성)
- 특성이 많은 고차원에서는 선형 회귀가 매우 복잡한 모델을 표현할 수 있다.
예제) 농어의 길이, 높이, 두께를 같이 사용 / 3개의 특성을 각각 제곱하여 추가, 특성을 서로 곱해서 추가.
'''



#%%
'''
특성 공학
- 기존의 특성을 사용하여 새로운 특성을 뽑아내는 작업
- perch_csv_data.csv
'''

import pandas as pd

df = pd.read_csv('https://bit.ly/perch_csv_data')

perch_full=df.to_numpy()
print(perch_full)

perch_weight = np.array([
  5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
  110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
  130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
  197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
  514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
  820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
  1000.0, 1000.0
])

# 훈련 세트 나누기
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state = 42)


'''
새로운 특성 만들기
사이킷런 변환기
- 전처리 하거나 특성을 만들기 위한 클래스
- fit(), transform()메서드
- polynomialfeature 클래스
- sklearn.preporocessing 패키지에 포함되어 있음.
- fit() 과 transform()을 차례대로 호출
'''

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(include_bias=False)
poly.fit([[2,3]])
print(poly.transform([[2,3]]))

'''
기존 특성으로 부터 제곱 , 곱 등을 자동으로 추가하는 다항 특성 생성

여기서 사이킷런 선형 모델이 자동으로 절편 추가해주므로 include_bias=False 이걸 통해서 특성을 만드는걸 해제해줘야함.

그래서 왜하냐? 데이터가 곡선 형태일때 선형회귀를 곡선 특성을 추가해서 곡선처럼 예측한다.
'''
#%%

print(train_poly.shape)

print(poly.get_feature_names_out())


'''
사이킷런 변환기
fit() 
-만들 특성의 조합을 준비, 별도의 통계 값을 구하지 않는다.
- 훈련세트에 적용했던 변환기로 테스트 세트를 변환하는 습관이 좋다.

다중회귀 모델 훈련하기
- 농어 길이만 가지고 발생한 과소적합 해결(길이, 높이, 두께)
- 농어 길이,높이,두께를 가지고 발생한 과소 적합 해결
'''


'''
아오 여기서 부터는 걍 강의자료보면서 하
'''
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
test_poly = poly.transform(test_input)
print(lr.score(test_poly, test_target))


#%% 특성 정규화 하기

from sklearn.preprocessing import StandardScaler

ss= StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled =  ss.transform(test_poly)

'''
StandardScaler는 평균이 0, 표준편차가 1이 되도록 데이터를 표준 정규화해주는 클래스
'''

#%%

'''
선형 회귀 모델에 규제를 추가한 모델
- 릿지와 라쏘
릿지 : 계수를 제곱한 값을 기준으로 규제를 적용, 라쏘보다 선호
라소 : 계수의 절대값을 기준으로 규제를 적
용
'''

























