#%%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 데이터 준비
df = pd.read_csv('/Users/ijihun/bigData/iris.csv') 
df = df.drop('Species', axis=1)
df.head()
#%%
# 데이터 표준화
scaler = StandardScaler() # StandardScaler : 각 컬럼의 값들이 평균0, 표준편차 1이 되도록 변환
result = scaler.fit_transform(df)
df_scaled = pd.DataFrame(result, columns=df.columns)
df_scaled.head()

#%%
# 차원축소
pca = PCA(n_components=2)           #차원 축소를 위한 pca객체 생성
#%%
transform = pca.fit_transform(df_scaled)                 # 2차원으로 축소 

transform = pd.DataFrame(transform)
transform.head()

#%%
# 시각화
transform.plot.scatter(x=0, y=1,                         # 산점도
                       title='PCA plot')
plt.show()

#%%
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 준비
df = pd.read_csv('/Users/ijihun/bigData/iris.csv') 
df = df.drop('Species', axis=1)
df.head()

# 데이터 표준화
scaler = StandardScaler()
result = scaler.fit_transform(df)
df_scaled = pd.DataFrame(result, columns=df.columns)
df_scaled.head()

# 군집화
model = KMeans(n_clusters=3, n_init=10, random_state=123) 
model.fit(df_scaled) 

# 군집화 결과 확인
print(model.cluster_centers_)                  # 군집 중심점 좌표
print(model.labels_)                           # 각행의 군집 번호
print(model.inertia_)                          # 군집 평가 점수 : 모든 점과 중심점 간 거리의 제곱합 ,작을수록 좋다.

#%%
# 차원축소
pca = PCA(n_components=2)          
transform = pca.fit_transform(df_scaled)          # 2차원으로 축소 
transform = pd.DataFrame(transform)
transform['cluster'] = model.labels_              # 군집정보 추가
transform.head()

#%%
# 시각화
sns.scatterplot(
    data=transform,
    x = 0,                              # x축 
    y = 1,                              # y축
    hue="cluster",                      # 원의 색 
    palette='Set2',                     # 팔레트 선택
    legend=False                        # 범례표시 여부
)

plt.show()
#%%

ks = range(1,10)                    # 군집의 개수
inertias = pd.Series([])            # 군집화 평가 결과

for k in ks:
    model = KMeans(n_clusters=k,
                   n_init=10, random_state=123)
    model.fit(df_scaled)
    inertias.loc[k] = model.inertia_

plt.figure(figsize=(7, 4))
inertias.plot.line(title = 'Inertias Score',
                   xlabel= 'number of clusters, k',
                   ylabel = 'inertia')


plt.show()

 


