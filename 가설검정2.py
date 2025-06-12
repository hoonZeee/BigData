import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

'''
맨 휘트니 검정 : 독립성은 만족하지만 정규 분포를 만족하지 않을 때 > 독립표본
'''

df = pd.read_csv('/Users/ijihun/bigData/mw_test.csv')

df.groupby('group').count()  # 그룹크기가 5, 4 여서 정규성 만족 못한다.

#%%
df.groupby('group').mean()

#%%
df.groupby('group').boxplot(grid=False)
plt.show()

#%%

group_1 = df.loc[df.group=='A','score']
group_2 = df.loc[df.group=='B', 'score']

stats.mannwhitneyu(group_1, group_2)
'''
pvlaue = 0.085로 유의수준보다 크다. 즉 유의미한 차이가 있다고 말할 수 없다.
결론 : 개발된 필기도구의 만족도가 일반 필기도구의 만족도보다 높다고 볼 근거가 없다.
'''

#%%
'''
윌콕슨 부호 순위 검정
: 짝지어진 두 그룹이 정규분포를 만족하지 않을때 두 그룹 평균을 비교하귀 이해 시행하는 검정 방법 > 대응표본
'''

df = pd.read_csv('/Users/ijihun/bigData/wilcoxon_test.csv')

(df['post']-df['pre']).mean() # 불만도가 4.5점 낮아저서 개선된것처럼 보인다.

#%%
fig, axes = plt.subplots(nrows=1, ncols=2)

df['pre'].plot.box(grid=False, ax=axes[0])

plt.ylim([60,100])

df['post'].plot.box(grid=False, ax=axes[1])

plt.show() # 개선 후의 상자그림에서도 불만도가 낮아진 것으로 확인가능하다.

#%%
#윌콕슨 부호 순위 검정

stats.wilcoxon(df['pre'],df['post'])

'''
윌콕슨 부호 순위 검정 결과 p-value가 0.25로 유의수준 0.05보다 크기때문에 귀무가설 기각 안됨.
즉 불만도가 낮아졌다고 볼 수 있는 근거가 없다.

즉, 육안상으로는 불만도가 낮아진것처럼 보여도, 통계적으론 그렇지 않다. 그래서 검정이 필요하다.
'''

#%%
from scipy import stats
men = [10,10]
women = [15,65]

stats.chi2_contingency([men,women]) #stats검정통계랑,pavlue, 기대빈도 확인 .다 5이상이니까 카이제곱가능
'''
pvalue가 0.00937... 이므로 귀무가설을 기각하고 대립가설 채택
'''


#%%
from scipy import stats

Group_A = [7,3]
Group_B = [2,9]

stats.chi2_contingency([Group_A,Group_B])[3] 
'''
5이하인 셀이 2개나 있어서 카이제곱 사용불가, 피셔 검정 사용 
'''
#%%
stats.fisher_exact([Group_A,Group_B])
'''
pvalue 0.029 대립가설 채택. 즉 두반의 정답률에는 차이가 있다.
'''


#%% 실습문제

import seaborn as sns

df = sns.load_dataset('titanic')

df2 = df[['pclass','alive']]

cross_tab = pd.crosstab(df['pclass'], df['alive'])

print(cross_tab['yes'] / (cross_tab['yes'] + cross_tab['no']))

stats.chi2_contingency(cross_tab)

'''
기대 빈도를 보니 다 5이상으로 충분히 카이검정 사용가능
pvalue가 4.549251711298793e-23 매우작다. 대립가설 채택!
'''








