import pandas as pd
import numpy as np

exam = pd.read_csv("/Users/ijihun/bigData/exam.csv")
mpg = pd.read_excel("/Users/ijihun/bigData/mpg.xlsx")

#%% 집단 별로 요약하기 : df.groupby(), df.agg()

exam.agg(mean_math=('math','mean'))

exam.groupby('nclass').agg(mean_math=('math','mean'))

#nclass 가 인덱스로 처리됨, as_index= False 로 설정하면 인덱스는 바뀌지 않는다.
exam.groupby('nclass',as_index=False).agg(mean_math=('math','mean'))


exam.groupby('nclass',as_index = False).agg(mean_Eng=('english','sum'))

#그럼 한번에 여러개 해보기

exam.groupby('nclass',as_index=False).\
    agg(mean_math = ('math','mean'),
        mean_eng = ('english','sum'),
        mean_sci = ('science','median'),
        n = ('nclass','count')
        )
    
#agg()에서 자주 사용되는 요약 통계랑 함수
# mean(), std(), sum(), median(), min(), max(), count()

#모든 변수의 요약 통계량 한번에 구하기
exam.groupby('nclass').mean()
exam.groupby('nclass').sum()

#집단을 나눈 다음 다시 하위 집단으로 나누기
mpg.groupby(['manufacturer','drv'])\
    .agg(mean_cty=('cty','mean'))
    
mpg.groupby('drv').agg(n=('drv','count'))

mpg['drv'].value_counts() #내림차순 정렬
#데이터프레임이 아니라 시리즈 이므로 query()적용 불가
# mpg['drv'].values_counts().query('n>100') << 이거 안됨.



mpg.query('category=="suv"')\
    .assign(total = (mpg['cty'] + mpg['hwy'])/2)\
        .groupby('manufacturer')\
            .agg(mean_tot = ('total','mean'))\
                .sort_values('mean_tot',ascending = False)\
                    .head()
                    
#%%

#'suv','compact'등 일곱 종류 분류해서 어떤 차종의 도시 연비가 높은지 비교해봐라.

mpg.groupby('category')\
    .agg(mean_cty=('cty','mean'))\
        .sort_values('mean_cty',ascending=False)\
            .head(3)

#어떤회사에서 'compact'차종을 가장 많이 생산하는지

mpg.query('category=="compact"')\
    .groupby('manufacturer')\
        .agg(n=('manufacturer','count'))\
            .sort_values('n',ascending=False)
            


#%% 등급 나누기

exam.loc[0,'math']=1
bins=[-1,20,40,60,80,100]

math_grade=pd.cut(exam.math,bins,labels=['F','D','C','B','A'])

#%% category 정렬 순서 만들기

dfiris = pd.read_csv("/Users/ijihun/bigData/iris.csv")

sort1 = dfiris.sort_values(by="Species")
dfiris.info()
dfiris['Species'].unique()

species_category = ['virginica', 'setosa', 'versicolor']
dfiris['Species']=pd.Categorical(dfiris['Species'],
                                 categories=species_category,ordered=True)
dfiris.info()
dfiris['Species'].unique()
sort2 = dfiris.sort_values(by="Species")








