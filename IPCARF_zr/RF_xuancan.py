import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from sklearn.ensemble import RandomForestClassifier
import random
from math import log
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from k_cross_zr import k_cross_zr
from sklearn.decomposition import IncrementalPCA
from sklearn import model_selection
from sklearn.model_selection import train_test_split,StratifiedKFold

#data progress
#以下两句用来忽略版本错误信息
import warnings
warnings.filterwarnings("ignore")

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return
#定义函数来显示柱状上的数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%.3f' % float(height))

# 读取源文件
SampleFeature = []
ReadMyCsv(SampleFeature, "D:\\1 python\\zr\\zr_lncRNA_disease_predict\\data\\SampleFeaturezr.csv")
# SampleLabel
SampleLabel = []
counter = 0
while counter < len(SampleFeature) / 2:
    # Row = []
    # Row.append(1)
    SampleLabel.append(1)
    counter = counter + 1
counter1 = 0
while counter1 < len(SampleFeature) / 2:
    # Row = []
    # Row.append(0)
    SampleLabel.append(0)
    counter1 = counter1 + 1


X = np.array(SampleFeature)
y = np.array(SampleLabel)


seed = 7 #重现随机生成的训练
test_size = 0.3 #30%测试，70%训练
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
#model =XGBClassifier()
model =RandomForestClassifier()
n_estimators=[100,500,1000,1500,2000,2500]
#learning_rate = [0.0001,0.001,0.01,0.1,0.2,0.3] #学习率
#gamma = [1, 0.1, 0.01, 0.001]
#param_grid = dict(learning_rate = learning_rate,gamma = gamma)#转化为字典格式，网络搜索要求
param_grid=dict(n_estimators = n_estimators)
kflod = StratifiedKFold(n_splits=10, shuffle = True,random_state=7)#将训练/测试数据集划分10个互斥子集，
#grid_search = GridSearchCV(model,param_grid,scoring = 'neg_log_loss',n_jobs = -1,cv = kflod)
grid_search = GridSearchCV(model,param_grid,scoring = 'roc_auc',n_jobs = -1,cv = kflod)
#scoring指定损失函数类型，n_jobs指定全部cpu跑，cv指定交叉验证
grid_result = grid_search.fit(X_train, Y_train) #运行网格搜索
print("Best: %f using %s" % (grid_result.best_score_,grid_search.best_params_))