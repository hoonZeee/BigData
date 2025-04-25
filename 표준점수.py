import pandas as pd
import numpy as np

'''
표준점수 : 어떤 값이 평균에서 얼마나 떨어져 있는지를 표준 편차 기준으로 나타낸 값
표준 점수 : (원래 값  - 평균 ) / 표준편차
'''

from scipy import stats

data = [55,60,65,70,75]

mean = np.mean(data)
std = np.std(data)

z_scores = [(x-mean) / std for x in data]
z_scores

z_scores_scipy = stats.zscore(data)
z_scores_scipy


#%%

from scipy.stats import zscore

students = ["철수", "영희", "민수", "지민", "하늘", "수진", "태호", "서연", "도윤", "유진"]
scores = [72, 85, 90, 65, 78, 95, 88, 70, 60, 80]

df = pd.DataFrame({'이름':students, '점수':scores})

df['표준점수'] = zscore(df['점수'])
df