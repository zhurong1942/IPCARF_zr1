from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
def confusion_heatmap(y_test,y_pred,methodname):
    plt.figure()
    sns.heatmap(pd.DataFrame(confusion_matrix(y_test,y_pred)), annot=True, cmap="rainbow" ,fmt='g')
#    plt.tight_layout()
    plt.title(methodname+'_Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(methodname+'_confusion_heatmap.tif',dpi=300,bbox_inches='tight')
    plt.show()