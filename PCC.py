import pandas as pd
import xlwt
df = pd.read_excel('D:/python/machine-learning/随机森林/随机森林/疲劳数据分布图/皮尔逊相关系数/data.xlsx')
result = df.corr(method='pearson')['Fatigue life']    # 计算疲劳寿命与其他变量之间的皮尔逊系数
print(result)
# l_1d = result.values.tolist()
# print(type(l_1d))
# Index = ['Pearson']
# columns = ['Ni','Cr','Nb','Mo','Ti','Al','C','Co','Fe','W','Si','Solution','temperature','Solution time','Aging processing','Strain rate','Plastic strain','Stress ratio','Experimental temperature','Fatigue life']
# df = pd.DataFrame("l_1d", index=Index,columns=columns)
#
# # 保存到本地excel
# df.to_excel("cl_1d.xlsx", index=False)
# def data_write(file_path, datas):
#     f = xlwt.Workbook()
#     sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
#     #将数据写入第 i 行，第 j 列
#     i = 0
#     for data in datas:
#         for j in range(19):
#             sheet1.write(i,j,data[j])
#         i = i + 1
#     f.save(file_path) #保存文件
# if __name__ == '__main__':
#     data_write('pearson1.xlsx',l_1d)
