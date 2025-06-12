import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CSV 파일 불러오기
df = pd.read_csv('/Users/ijihun/bigData/project/age.csv') 

# 2. 20대, 30대만 필터링 + '계' 제외
df_filtered = df[
    df['연령별'].isin(['20대', '30대']) &
    (df['전입사유별'] != '계')
].copy()

# 3. 연도별 합계 컬럼 생성
df_filtered['합계'] = df_filtered[['2022', '2023', '2024']].sum(axis=1)

# 4. 주요 전입 사유만 필터링
main_reasons = ['직업', '가족', '주택', '교육', '주거환경', '기타']
df_main = df_filtered[df_filtered['전입사유별'].isin(main_reasons)].copy()

# 5. 피벗테이블 생성 (행: 전입사유, 열: 연령대)
pivot = pd.pivot_table(
    df_main,
    index='전입사유별',
    columns='연령별',
    values='합계',
    aggfunc='sum',
    fill_value=0
)

# 6. 카이제곱 독립성 검정
chi2, p, dof, expected = chi2_contingency(pivot)

# 7. Cramér's V 계산 (효과 크기)
n = pivot.to_numpy().sum()
phi2 = chi2 / n
r, k = pivot.shape
cramers_v = np.sqrt(phi2 / min(k - 1, r - 1))

# 8. 결과 출력
print("[카이제곱 독립성 검정 결과]")
print("chi² 통계량:", round(chi2, 2))
print("p-value:", p)
print("자유도:", dof)
print("Cramér's V (효과크기):", round(cramers_v, 3))
print()

# 9. 기대빈도와 실제값 비교
result_df = pd.DataFrame({
    '전입사유': pivot.index,
    '20대 실제값': pivot['20대'],
    '30대 실제값': pivot['30대'],
    '20대 기대값': expected[:, 0],
    '30대 기대값': expected[:, 1]
})

print("[전입사유별 실제값 vs 기대값]")
print(result_df)

# 한글 폰트 설정 (필요 시 수정)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터프레임 인덱스 설정
viz_df = result_df.copy()
viz_df.set_index('전입사유', inplace=True)

# 실제값 비교 시각화
viz_df[['20대 실제값', '30대 실제값']].plot(kind='bar', figsize=(10, 6), alpha=0.85)
plt.title('전입 사유별 실제값 비교 (20대 vs 30대)')
plt.ylabel('합계 인원수')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()