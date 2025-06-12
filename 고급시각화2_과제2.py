import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/ijihun/bigData/prestige.csv")


#%%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rc('font', family='AppleGothic')
matplotlib.rcParams['axes.unicode_minus'] = False



df = pd.read_csv("/Users/ijihun/bigData/prestige.csv")

sns.scatterplot(
    data = df,
    x= "education", y= "income",
    size= "women", sizes=(20,4000),
    hue = "type", alpha = 0.5,
    legend =True )

for i in range(6, df.shape[0]):
    if df.education[i] <= 17 :
        plt.text(x=df.education[i],y=df.income[i], s=df.job[i],
                 horizontalalignment='center',size= 'small',color='dimgray')
        



plt.xlim(6, 17)

plt.title("직종별 교육연수(education)과 수입(income)에 대해 버블 차트를 작성하시오")
plt.show()




#%%
df_new = df[['job', 'education', 'income', 'women', 'prestige']].copy()
df_new.set_index('job', inplace=True)


df_new = (df_new - df_new.min()) / (df_new.max() - df_new.min())

df_new.loc['avg'] = df_new.mean()

df_prog = df_new.query("index == 'computer.programers'")


df_radar = pd.concat([df_new.loc[['avg']], df_prog])


labels = df_radar.columns
num_vars = len(labels)


angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]


fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))


ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)


plt.xticks(angles[:-1], labels)


ax.set_ylim(0, 1)


for idx, row in df_radar.iterrows():
    values = row.tolist()
    values += values[:1]  
    ax.plot(angles, values, label=idx)
    ax.fill(angles, values, alpha=0.25)


plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.show()


#%%
df = pd.read_csv("/Users/ijihun/bigData/prestige.csv")


df2 = df[['job','education','income','women','prestige']].copy()
df2.set_index('job',inplace=True)
df2 = (df2 - df2.min()) / (df2.max() - df2.min()) #단위가 다르니까 각 값을 0 ~ 1 사이로 스케일링


df2.loc['avg'] = df2.mean()

df_prog = df2.query("index == 'computer.programers'")

df_rader = pd.concat([df2.loc[['avg']], df_prog])

labels = df_rader.columns
num_vars = len(labels)








