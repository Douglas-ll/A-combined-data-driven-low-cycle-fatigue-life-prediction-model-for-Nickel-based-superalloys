import pandas as pd
from minepy import MINE
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
def mic(x, y):
    m = MINE(alpha=0.6, c=15, est="mic_approx")
    m.compute_score(x, y)
    return m.mic()

df=pd.read_excel('data.xlsx')
col=df.columns.values
per=df.corr()
j = 18
for i in range(len(col)):
        m=mic(df[col[i]],df[col[j]])
        per.iloc[i,j]=m
        print(m)
per=per.round(2)

# k=1
# for i in range(len(col)):
#     for j in range(i,len(col)):
#         plt.subplot(1, 1, 1)
#         k+=1
#         plt.scatter(df[col[i]],df[col[j]])
#         plt.xlabel(col[i])
#         plt.ylabel(col[j])
#         plt.title('Pearson r='+str(per.iloc[j,i])+'\nMic='+str(per.iloc[i,j]))
#         plt.show()
