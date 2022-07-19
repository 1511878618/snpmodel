from sklearn import metrics
import matplotlib.pyplot as plt 
import seaborn as sns 
def pred_threshold(y_pred, threshold=0.5):
    return (y_pred > threshold).astype(int)
def accuracy(y_true, y_pred, threshold=0.5, **kwargs):
    y_pred = pred_threshold(y_pred, threshold)

    return accuracy_score(y_true, y_pred, **kwargs)

def roc_curve(true, pred):
    plt.figure(dpi=200)
    sns.set(font_scale=1.5)
    fpr, tpr, thresholds = metrics.roc_curve(true, pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr,roc_auc=roc_auc,estimator_name='example estimator')
    display.plot(name="ROC")