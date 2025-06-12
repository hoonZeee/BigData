import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/ijihun/bigData/ind_ttest.csv')

#%%
print(df.head())

print(df.groupby('group').count())
print(df.groupby('group').mean())
df.groupby('group').boxplot(grid=False)
plt.show()

#%%
group_1 = df.loc[df.group == 'A','height']
group_2 = df.loc[df.group == 'B','height']

#%%

#정규성 검정
print(stats.shapiro(group_1))
print(stats.shapiro(group_2))
'''
두집단 모두 표본 크기가 30보다 크기 때문에 정규성 검정이 필요없으나 검정 방법을 설명하기 위해 실
두 그룹모두 p-value가 0.05보다 훨씬 크기 때문에 정규성을 만족한다.
왜냐 > 정규성이라는건 귀무가설이 가치가 있냐는건데, 귀무가설이 체택되려면 대립가설과 반대여야겠지? 즉 pvalue도 커야좋은거임 0.05이상이어야!
'''
#%%
#등분산성 검정
stats.levene(group_1, group_2)
'''
마찬가지로 분산도 귀무가설을 입증하는거니까 pvalue 가 0.05이상이어야 만족하는거임!
'''

#%%

#독립 표본 검정
result = stats.ttest_ind(group_1, group_2, equal_var=True)
result

'''
1. 두 그룹은 정규성과 분산성을 만족하므로 독립표본 T-검정을 실시할 수 있다.
- stats.ttest_ind() : 독립표본 T-검정을 위한 메서드 , 파라미터 equal_Var=True 등분산 여부 지정. 우리 분산 만족하니까 True

2. 즉 결과를 보면 pvalue는 0.003으로 유의수준 0.05보다 작다! 따라서 두 그룹은 유의미한 차이가 있다.
그룹 B의 평균이 그룹 A의 평균보다 높으므로 "신제품 B가 신제품 A보다 효과가 좋다."

- tip : 가설검정하다보면 pvalue가 0.689e-28 이렇게 나올수도 있는데 < 이건 그냥 0에가까운 매우작은숫자를 의미
'''


#%%

#대응표본 검정

df = pd.read_csv('/Users/ijihun/bigData/paired_ttest.csv')
df.head()
df[['before','after']].mean()
(df['after']-df['before']).mean()


#%%
fig, axes = plt.subplots(nrows=1, ncols=2)
df['before'].plot.box(grid=False, ax=axes[0])
plt.ylim([60,100])
df['after'].plot.box(grid=False, ax=axes[1])
plt.show()

#%%
stats.shapiro(df['after']-df['before']) # pvalue는 좀큰데? 정규성은 만족하고

#%%
result = stats.ttest_rel(df['before'], df['after'])

result # pvalue가 0.05보다 훨크네? 그러면 뭐, 대립가설 채택못하고 새로운 교수법이 효과 있다고 말하기 어렵다.

'''
근데 왜 대응표본은 등분산성 검사안해?
대응표본은 같은사람한테서 비교하는거니까, 정규성 검사만 하면된다.
독립표본빌때만 등분산성 검사하는거임!
'''


















