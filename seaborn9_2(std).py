#%%

'''
버블차트
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('/Users/ijihun/bigData/crimeRatesByState2005.csv')

sns.set_theme(rc={'figure.figsize':(7,7)}) #7정도 사이즈로 1:1 비율설

sns.scatterplot(
    data =df,
    x="murder",y =  "burglary",
    size="population", sizes=(20,4000), #인구에 따른 원의 크기를 20에서 400으로 설정
    hue="state", alpha = 0.5, #hue 원의 색을 결정할 기준
    legend = False)

for i in range(0,df.shape[0]):
    if df.murder[i] <=12 :
        plt.text(x=df.murder[i],y=df.burglary[i],s=df.state[i],
                 horizontalalignment='center',size="small",color='dimgray')
        
plt.xlim(0,12) #x축의 범위를 12까지만

plt.show()

#%%
'''
방사형 차트 : 변수의 수에 따라 원을 동일한 간격으로 나누고, 중심으로 부터 일정한 간격으로 등심원을 그리며
척도를 나타내는 축을 생성
'''

from plotly.offline import plot
import plotly.express as px
fig =px.line_polar(r=[1,5,2,3,2],theta=['a','b','c','d','e'],
                   line_close=True)

plot(fig)
#%%

import plotly.graph_objects as go


fig = go.Figure(
    data = [go.Bar(x=[1,2,3], y=[1,3,2])],
    layout = go.Layout(
            title = go.layout.Title(text="A Figure Specified By A Graph Object")
        )
    )

fig.show()
#%%
# 실습전 라이브러리 설치
# pip install plotly
# pip install kaleido==0.1.0post1


import matplotlib.image as mpimg
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
#####################################################################

def radar(df, fills, min_max, title=''):
    fig = go.Figure()
    categories = df.columns.to_list()
    categories.append(categories[0])
    i = 0
    while (i < len(df)):
        scores = df.iloc[i, :].to_list()
        scores.append(scores[0])
        fig.add_trace(go.Scatterpolar(
            r=scores,  # 축의 값
            theta=categories,  # 축의 레이블
            fill=fills[i],  # 다각형 채우기 색
            name=df.index[i]  # 다각형 레이블
        ))
        i += 1
        
        fig.update_layout(
            polar_radialaxis_visible=True,
            polar_radialaxis_range=min_max,  # 축의 값 범위
            showlegend=True,
            margin_t=50,  # 상단 여백
            margin_l=100,  # 좌측 여백
            margin_r=100,  # 우측 여백
            margin_b=25,  # 하단 여백
            width=700,  # 그래프의 폭(pixel)
            height=700,  # 그래프의 높이(pixel)
            title_text=title,  # 그래프 제목
            title_font_size=30,  # 제목 폰트 사이즈
            font_size=20  # 폰트 사이즈
            )
        plot(fig)
"""
    # 그래프 저장 & display
    plt.axis('off')
    fig.write_image('rader.png')
    plt.imshow(mpimg.imread('rader.png'))
    plt.show()
 """
#####################################################################

# 데이터 입력
df = pd.DataFrame({
    'Kor': [72, 70, 90, 60, 66],
    'Eng': [84, 85, 95, 70, 85],
    'Math': [71, 40, 88, 80, 75],
    'Sci': [83, 80, 91, 90, 70],
    'Phy': [60, 60, 60, 70, 50]
})
df.index = ['AVG', 'John', 'Tom', 'Smith', 'Grace']
df
fills = [None, 'toself']
radar(df=df.iloc[[0, 3], :],
      fills=fills,
      min_max=[0, 100],
      title='Scores of Smith'
      )

#%%

fills = ['toself', 'toself', 'toself', 'toself']
radar(df=df.iloc[1:, :],
      fills=fills,
      min_max=[0, 100],
      title='Scores of Student'
      )
#%%

