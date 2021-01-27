#k_cross_zr
# k折交叉验证
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def k_cross_zr(X, y,fold,model,model_name,linestyles,colors,markers):
    SplitNum = fold
    cv = StratifiedKFold(n_splits=SplitNum)
    tprs = []
    aucs = []
    fprs=[]
#    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(X, y):
        model1 = model
        model1.fit(X[train], y[train])
        y_score1 = model1.predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], y_score1[:, 1])
        tprs.append(tpr)
        fprs.append(fpr)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        i += 1
    # 画均值
    mean_tpr = np.mean(tprs, axis=0)
    mean_fpr=np.mean(fprs,axis=0)
    roc_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, label=model_name+' (AUC = %0.4f)' % (roc_auc), linestyle=linestyles,color=colors,marker=markers,lw=2, alpha=.8)
    # 画标题坐标轴
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate',fontsize=12)
    plt.ylabel('True Positive Rate',fontsize=12)
    # plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")