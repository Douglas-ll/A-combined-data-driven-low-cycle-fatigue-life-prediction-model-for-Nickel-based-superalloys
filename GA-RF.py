# coding=utf-8
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import random
import math
import csv
import joblib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
generations = 30  # 繁殖代数
pop_size = 30  # 种群数量
max_value = 512  # 基因中允许出现的最大值
max_value_BOUND = [0,512]
chrom_length = 15 # 染色体长度
pc = 0.6  # 交配概率
pm = 0.2  # 变异概率
results = [[]]  # 存储每一代的最优解，N个三元组（auc最高值, n_estimators, max_depth）
fit_value = []  # 个体适应度
fit_mean = []  # 平均适应度
n_estimators3d = []
max_depth3d = []
R23d=[]
pop = [[0,1,1,1,0,0,0,0,0,0, 0, 1, 0,0,0] for i in range(pop_size)]  # 初始化种群中所有个体的基因初始序列

'''
n_estimators 取 {10、20、30、40、50、60、70、80、90、100、110、120、130、140、150、160}
max_depth 取 {1、2、3、4、5、6、7、8、9、10、11、12、13、14、15、16} 
（1111，1111）基因组8位长
'''


def randomForest(n_estimators_value, max_depth_value):
    # print("n_estimators_value: " + str(n_estimators_value))
    # print("max_depth_value: " + str(max_depth_value))
    features = pd.read_csv('D:/python/machine-learning/实验二/class3.csv')
    # print(features.head(5))#查看前五行数据
    # print(np.isnan(features).any())#查看是否有缺失值
    # 标签
    labels = np.array(features['Fatigue life'])

    # 在特征中去掉标签
    features = features.drop('Fatigue life', axis=1)

    # 名字单独保存一下，以备后患
    feature_list = list(features.columns)

    # 转换成合适的格式
    features = np.array(features)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,
                                                                                random_state=42)

    # train_xy = loadFile("data.csv")
    # train_xy = train_xy.drop('Fatigue life', axis=1)  # 删除训练集的ID
    # # 将训练集划分成7:3（训练集与测试集比例）的比例
    # train, val = train_test_split(
    #     train_xy, test_size=0.3, random_state=80)
    # train_y = train['Kind']  # 训练集类标
    # val_y = val['Kind']  # 测试集类标
    #
    # train = train.drop('Kind', axis=1)  # 删除训练集的类标
    # val = val.drop('Kind', axis=1)  # 删除测试集的类标

    rf = RandomForestRegressor(n_estimators=n_estimators_value,
                                max_depth=max_depth_value,
                               random_state=42)
    rf.fit(train_features, train_labels)  # 训练
    #R2X = rf.score(train_features, train_labels)
    predictions = rf.predict(test_features)
    R2 = r2_score(test_labels, predictions)
    #print('训练集特征:', train_features.shape)
    #print(R2)
    # predict_test = rf.predict_proba(val)[:, 1]
    # roc_auc = metrics.roc_auc_score(val_y, predict_test)
    return R2
def randomForestbest(n_estimators_valuebest, max_depth_valuebest):
    # print("n_estimators_value: " + str(n_estimators_value))
    # print("max_depth_value: " + str(max_depth_value))
    features = pd.read_csv('D:/python/machine-learning/实验二/class3.csv')
    # print(features.head(5))#查看前五行数据
    # print(np.isnan(features).any())#查看是否有缺失值
    # 标签
    labels = np.array(features['Fatigue life'])

    # 在特征中去掉标签
    features = features.drop('Fatigue life', axis=1)

    # 名字单独保存一下，以备后患
    feature_list = list(features.columns)

    # 转换成合适的格式
    features = np.array(features)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,
                                                                                random_state=42)
    print(test_labels)


    rf = RandomForestRegressor(n_estimators=n_estimators_valuebest,
                                max_depth=max_depth_valuebest,
                               random_state=42)
    rf.fit(train_features, train_labels)  # 训练回归器
    predictions = rf.predict(test_features)
    predictions1 = rf.predict(train_features)
    R2 = r2_score(test_labels, predictions)
    print(rf.score(train_features, train_labels))
    #print('训练集特征:', train_features.shape)
    print("测试集精度",R2)
    csv_file_path = 'predictions/svm3.csv'
    rounded_data = np.round(predictions, 2)
    formatted_predictions = []
    it = np.nditer(rounded_data)
    while not it.finished:
        formatted_predictions.append("{:.2f}".format(it[0]))
        it.iternext()
    # 将格式化的预测结果写入CSV文件
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Predictions'])  # 写入表头
        writer.writerows([[prediction] for prediction in formatted_predictions])  # 逐行写入预测结果

    print("预测结果已保存为CSV文件")
    # print ('MAPE:',np.mean(mape))
    #print("测试集精度", r2_score(test_labels, predictions))
    ##print("训练集RMSE", MAPE1)
    # joblib.dump(rf, 'save/clf.pkl')
    # clf1 = joblib.load('save/clf.pkl')
    # true_data = pd.DataFrame(data = {'actual': test_labels})
    # predictions_data = pd.DataFrame(data={'prediction': predictions})
    # plt.plot(true_data['actual'], 'b-', label='actual')
    # plt.plot(predictions_data['prediction'], 'r-', label='prediction')
    # plt.xticks(rotation='60');
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # plt.legend()
    # # 图名
    # plt.xlabel('serial number')
    # plt.ylabel('Fatigue life');
    # plt.title('Actual and Predicted Values');
    # plt.show()

    # predict_test = rf.predict_proba(val)[:, 1]
    # roc_auc = metrics.roc_auc_score(val_y, predict_test)



def loadFile(filePath):
    fileData = pd.read_csv(filePath)
    return fileData


# Step 1 : 对参数进行编码（用于初始化基因序列，可以选择初始化基因序列，本函数省略）
def geneEncoding(pop_size, chrom_length):
    pop = [[]]
    for i in range(pop_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.randint(0, 1))
        pop.append(temp)
    return pop[1:]


# Step 2 : 计算个体的目标函数值
def cal_obj_value(pop):
    objvalue = []
    variable = decodechrom(pop)
    for i in range(len(variable)):
        tempVar = variable[i]
        n_estimators_value = (tempVar[0]+1)
        #print(n_estimators_value)
        max_depth_value = tempVar[1]+1
        #print(max_depth_value)

        aucValue = randomForest(n_estimators_value, max_depth_value)
        objvalue.append(aucValue)
    return objvalue  # 目标函数值objvalue[m] 与个体基因 pop[m] 对应


# 对每个个体进行解码，并拆分成单个变量，返回 n_estimators 和 max_depth
def decodechrom(pop):
    variable = []
    n_estimators_value = []
    max_depth_value = []
    for i in range(len(pop)):
        res = []

        # 计算第一个变量值，即 0101->10(逆转)
        temp1 = pop[i][0:9]
        preValue = 0
        for pre in range(9):
            preValue += temp1[pre] * (math.pow(2, pre))
        res.append(int(preValue))

        # 计算第二个变量值
        temp2 = pop[i][9:14]
        aftValue = 0
        for aft in range(5):
            aftValue += temp2[aft] * (math.pow(2, aft))
        res.append(int(aftValue))
        variable.append(res)
    return variable

# Step 3: 计算个体的适应值（计算最大值，于是就淘汰负值就好了）
def calfitvalue(obj_value):
    fit_value = []
    temp = 0.0
    Cmin = 0
    for i in range(len(obj_value)):
        if (obj_value[i] + Cmin > 0):
            temp = Cmin + obj_value[i]
        else:
            temp = 0.0
        fit_value.append(temp)
    return fit_value


# Step 4: 找出适应函数值中最大值，和对应的个体
def best(pop, fit_value):
    best_individual = pop[0]
    best_fit = fit_value[0]
    for i in range(1, len(pop)):
        if (fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
    return [best_individual, best_fit]


# Step 5: 每次繁殖，将最好的结果记录下来(将二进制转化为十进制)
def b2d(best_individual):
    temp1 = best_individual[0:9]
    preValue = 0
    for pre in range(9):
        preValue += temp1[pre] * (math.pow(2, pre))
    preValue = preValue + 1
    #preValue = preValue * 10

    # 计算第二个变量值
    temp2 = best_individual[9:14]
    aftValue = 0
    for aft in range(5):
        aftValue += temp2[aft] * (math.pow(2, aft))
    aftValue = aftValue + 1
    return int(preValue), int(aftValue)


# Step 6: 自然选择（轮盘赌算法）
def selection(pop, fit_value):
    # 计算每个适应值的概率
    new_fit_value = []
    total_fit = sum(fit_value)
    for i in range(len(fit_value)):

        new_fit_value.append(fit_value[i] / total_fit)
    # 计算每个适应值的累积概率
    cumsum(new_fit_value)
    # 生成随机浮点数序列
    ms = []
    pop_len = len(pop)
    for i in range(pop_len):
        ms.append(random.random())
    # 对生成的随机浮点数序列进行排序
    ms.sort()
    # 轮盘赌算法（选中的个体成为下一轮，没有被选中的直接淘汰，被选中的个体代替）
    fitin = 0
    newin = 0
    newpop = pop
    while newin < pop_len:
        if (ms[newin] < new_fit_value[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    pop = newpop


# 求适应值的总和
def sum(fit_value):
    total = 0
    for i in range(len(fit_value)):
        total += fit_value[i]
    return total


# 计算累积概率
def cumsum(fit_value):
    temp = []
    for i in range(len(fit_value)):
        t = 0
        j = 0
        while (j <= i):
            t += fit_value[j]
            j = j + 1
        temp.append(t)
    for i in range(len(fit_value)):
        fit_value[i] = temp[i]


# Step 7: 交叉繁殖
def crossover(pop, pc):  # 个体间交叉，实现基因交换
    poplen = len(pop)
    for i in range(poplen - 1):
        if (random.random() < pc):
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0: cpoint])
            temp1.extend(pop[i + 1][cpoint: len(pop[i])])
            temp2.extend(pop[i + 1][0: cpoint])
            temp2.extend(pop[i][cpoint: len(pop[i])])
            pop[i] = temp1
            pop[i + 1] = temp2


# Step 8: 基因突变
def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])
    for i in range(px):
        if (random.random() < pm):
            mpoint = random.randint(0, py - 1)
            if (pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1


if __name__ == '__main__':
    # pop = geneEncoding(pop_size, chrom_length)

    for i in range(generations):
        print("第 " + str(i) + " 代开始繁殖......")
        obj_value = cal_obj_value(pop)  # 计算目标函数值
        #print(obj_value)
        fit_value = calfitvalue(obj_value)  # 计算个体的适应值
        mean = np.mean(obj_value)
        print("平均适应度",mean)
        fit_mean.append(mean)
        dataframe = pd.DataFrame({'a_name': fit_mean})

        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv("test.csv", sep=',')
        [best_individual, best_fit] = best(pop, fit_value)  # 选出最好的个体和最好的函数值
        #print("best_individual: "+ str(best_individual))
        #print("fit_value: "+ str(fit_value))
        temp_n_estimator, temp_max_depth = b2d(best_individual)
        results.append([best_fit, temp_n_estimator, temp_max_depth])  # 每次繁殖，将最好的结果记录下来
        print(temp_n_estimator)
        print(temp_max_depth)
        print(str(best_individual) + " " + str(best_fit))
        n_estimators3d.append(temp_n_estimator)
        max_depth3d.append(temp_max_depth)
        R23d.append(best_fit)
        #print(n_estimators3d)
        #print(best_value)
        selection(pop, fit_value)  # 自然选择，淘汰掉一部分适应性低的个体
        crossover(pop, pc)  # 交叉繁殖
        mutation(pop, pc)  # 基因突变
    # print(results)
    dataframe = pd.DataFrame({'a_name': n_estimators3d})
    dataframe.to_csv("n_estimators3d.csv", sep=',')
    dataframe = pd.DataFrame({'a_name': max_depth3d})
    dataframe.to_csv("max_depth3d.csv", sep=',')
    dataframe = pd.DataFrame({'a_name': R23d})
    dataframe.to_csv("R23d.csv", sep=',')
    results.sort()
    print(results[-1])
    n_estimators_best = results[-1][1]
    max_depth_best = results[-1][2]
    randomForestbest(n_estimators_best,max_depth_best)
   #print(results[-1][1])