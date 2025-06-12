import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 폰트 설정 
plt.rcParams['font.family'] = 'AppleGothic'
mpl.rcParams['axes.unicode_minus'] = False

# 전국 데이터 불러오기 및 전처리
df = pd.read_csv('/Users/ijihun/bigData/project/age_sido.csv', header=1)
df_national = df[(df['행정구역별'] == '전국') & (df['각세별'] != '계')]

df_age_move = df_national[['각세별', '시도간전입', '시도간전출']].copy()
df_age_move.columns = ['age', 'inflow', 'outflow']
df_age_move['age'] = df_age_move['age'].str.replace('세', '', regex=False)
df_age_move['age'] = df_age_move['age'].replace('100이상', '100').astype(int)
df_age_move['inflow'] = pd.to_numeric(df_age_move['inflow'], errors='coerce')
df_age_move['outflow'] = pd.to_numeric(df_age_move['outflow'], errors='coerce')
df_age_move.sort_values(by='age', inplace=True)
df_age_move.reset_index(drop=True, inplace=True)

# 전국 기준 상위 나이대 출력
print("시도간 전입(inflow) 기준 상위 10")
print(df_age_move.sort_values(by='inflow', ascending=False)[['age', 'inflow']].head(10))
print("\n시도간 전출(outflow) 기준 상위 10")
print(df_age_move.sort_values(by='outflow', ascending=False)[['age', 'outflow']].head(10))

# 반복 처리 대상 지역
region_paths = [
    ('서울', '/Users/ijihun/bigData/project/seoul.csv'),
    ('부산', '/Users/ijihun/bigData/project/busan.csv'),
    ('대구', '/Users/ijihun/bigData/project/daegu.csv'),
    ('충청남도', '/Users/ijihun/bigData/project/choongchung.csv')
]

# 지역 전처리 함수
def preprocess_region_data(filepath):
    df = pd.read_csv(filepath, header=1)
    df = df[df['각세별'] != '계'][['각세별', '시도간전입', '시도간전출']].copy()
    df['age'] = df['각세별'].str.replace('세', '', regex=False)
    df['age'] = df['age'].replace('100이상', '100').astype(int)
    df['inflow'] = pd.to_numeric(df['시도간전입'], errors='coerce')
    df['outflow'] = pd.to_numeric(df['시도간전출'], errors='coerce')
    df.sort_values(by='age', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# 지역별 데이터프레임 저장
region_dataframes = {}
for name, path in region_paths:
    df_region = preprocess_region_data(path)
    region_dataframes[name] = df_region

    print(f"\n{name} 시도간 전입 상위 10")
    print(df_region.sort_values(by='inflow', ascending=False)[['age', 'inflow']].head(10))
    print(f"\n{name} 시도간 전출 상위 10")
    print(df_region.sort_values(by='outflow', ascending=False)[['age', 'outflow']].head(10))

# 시각화 함수: 각 지역 inflow/outflow
def plot_all_regions(regions):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, (df, name) in enumerate(regions):
        axes[i].bar(df['age'], df['inflow'], alpha=0.7, label='전입')
        axes[i].bar(df['age'], df['outflow'], alpha=0.7, label='전출')
        axes[i].set_title(f'{name} 시도간 전입/전출')
        axes[i].set_xlabel('나이')
        axes[i].set_ylabel('인원수')
        axes[i].legend()
        axes[i].grid(True)
    plt.tight_layout()
    plt.show()

# 시각화 실행
plot_input = [(df, name) for name, df in region_dataframes.items()]
plot_all_regions(plot_input)

# 25~35세 전입/전출 합계 계산 함수
def filter_and_sum(df):
    df_range = df[(df['age'] >= 25) & (df['age'] <= 35)]
    inflow_sum = df_range['inflow'].sum()
    outflow_sum = df_range['outflow'].sum()
    return inflow_sum, outflow_sum

# 지역별 25~35세 이동량 계산
filtered_results = {
    name: filter_and_sum(df) for name, df in region_dataframes.items()
}

# 막대그래프 시각화 (25~35세 전입/전출)
region_names = list(filtered_results.keys())
inflows = [val[0] for val in filtered_results.values()]
outflows = [val[1] for val in filtered_results.values()]

x = range(len(region_names))
bar_width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x, inflows, width=bar_width, label='전입', alpha=0.8)
plt.bar([i + bar_width for i in x], outflows, width=bar_width, label='전출', alpha=0.8)
plt.xticks([i + bar_width / 2 for i in x], region_names)
plt.xlabel('지역')
plt.ylabel('25~35세 이동 인구수')
plt.title('25~35세 시도간 전입/전출 비교 (지역별)')
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
