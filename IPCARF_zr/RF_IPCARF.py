from zr_confusion_heatmap import confusion_heatmap
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import random
from math import log
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import IncrementalPCA
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

x_train, x_test, y_train, y_test = train_test_split(SampleFeature, SampleLabel, test_size=0.2)

#建模
model = RandomForestClassifier()

names=['RF','IPCARF']
accuracy_results=[]
precision_results=[]
recall_results=[]
f1_results=[]
fpr_results=[]
tpr_results=[]
auc_results=[]
times_results=[]
#直接运行RF
t1=time.clock()
model.fit(x_train, y_train)
pred_test = model.predict(x_test)
accuracy=accuracy_score(y_test, pred_test)
precision = precision_score(y_test, pred_test)  
recall = recall_score(y_test, pred_test)  
f1 = f1_score(y_test, pred_test) 
accuracy_results.append(accuracy)
precision_results.append(precision)
recall_results.append(recall)
f1_results.append(f1)
print("RF: %-8.2f %-8.2f %-8.2f %-8.2f" %(accuracy,precision,recall,f1))
fpr, tpr, thresholds = roc_curve(y_test, pred_test, pos_label=1.0)
roc_auc = auc(fpr,tpr)
fpr_results.append(fpr)
tpr_results.append(tpr)
auc_results.append(roc_auc)
t2=time.clock()-t1
times_results.append(t2)

#IPCA降维
ipca = IncrementalPCA(n_components=128)
X_ipca = ipca.fit_transform(SampleFeature)
##PCA降维
#pca = PCA(n_components=2)
#X_pca = pca.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_ipca, SampleLabel, test_size=0.2)
t1=time.clock()
model.fit(x_train, y_train)
pred_test = model.predict(x_test)
accuracy=accuracy_score(y_test, pred_test)
precision = precision_score(y_test, pred_test)  
recall = recall_score(y_test, pred_test)  
f1 = f1_score(y_test, pred_test) 
accuracy_results.append(accuracy)
precision_results.append(precision)
recall_results.append(recall)
f1_results.append(f1)
print("IPCARF : %-8.2f %-8.2f %-8.2f %-8.2f" %(accuracy,precision,recall,f1))
fpr, tpr, thresholds = roc_curve(y_test, pred_test, pos_label=1.0)
roc_auc = auc(fpr,tpr)
fpr_results.append(fpr)
tpr_results.append(tpr)
auc_results.append(roc_auc)
t2=time.clock()-t1
times_results.append(t2)
#绘制ROC
#参数值
linestyles=['--','-.']
colors=['b','m']
markers=['+','D']

plt.figure()
for k in range(2):
    plt.plot(fpr_results[k],tpr_results[k],linestyle=linestyles[k],color=colors[k],marker=markers[k],lw=2, alpha=.8,label=names[k]+'   auc = %0.3f' % auc_results[k])
plt.plot([0, 1], [0, 1], color='black', linewidth=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=12)
plt.ylabel('True Positive Rate',fontsize=12)
plt.title('ROC',fontsize=12)
plt.legend(loc="lower right")
plt.show()
plt.savefig('RF_IPCARF.tif',dpi=300)

#绘制条形图
#准确率
plt.figure()
plt.title("accuracy")
plt.ylabel('accuracy')
plt.xlabel('method')
plt.ylim([0.0, 1.05])
a=plt.bar(names,accuracy_results,fc = 'g') 
autolabel(a)
plt.savefig('accuracy_IPCARF.tif',dpi=300)
plt.show()
#精确率
plt.figure()
plt.title("precision")
plt.ylabel('precision')
plt.xlabel('method')
plt.ylim([0.0, 1.05])
a=plt.bar(names,precision_results,fc = 'g') 
autolabel(a)
plt.savefig('precision_IPCARF.tif',dpi=300)
plt.show()
#召回率
plt.figure()
plt.title("recall")
plt.ylabel('recall')
plt.xlabel('method')
plt.ylim([0.0, 1.05])
a=plt.bar(names,recall_results,fc = 'g') 
autolabel(a)
plt.savefig('recall_IPCARF.tif',dpi=300)
plt.show()
#f1得分
plt.figure()
plt.title("f1_score")
plt.ylabel('f1_score')
plt.xlabel('method')
plt.ylim([0.0, 1.05])
a=plt.bar(names,f1_results,fc = 'g') 
autolabel(a)
plt.savefig('f1_score_IPCARF.tif',dpi=300)
plt.show()
#运行时间
plt.figure()
plt.title("run time",fontsize=12)
plt.ylabel('run time',fontsize=12)
plt.xlabel('method',fontsize=12)
plt.ylim([0.0, 1])
a=plt.bar(names,times_results,fc = 'g')
autolabel(a)
plt.show()
plt.savefig('runtime_IPCARF.tif',dpi=300)