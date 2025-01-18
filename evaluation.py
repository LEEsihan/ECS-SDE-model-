import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score

def evaluation_measures(y_test, prob):
    # AUC_ROC
    tpr_list = []
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, threshold = roc_curve(y_test, prob)
    tpr_list.append(np.interp(mean_fpr, fpr, tpr))
    tpr_list[-1][0] = 0.0
    auc_roc = auc(fpr, tpr)
    
    # AUC_PR
    auc_pr = average_precision_score(y_test, prob)
    
    # Brier Scores
    min_class = []
    maj_class = []
    for i in range(len(y_test)):
        if y_test[i] == 1:
            min_class.append((prob[i] - 1) ** 2)
        else:
            maj_class.append(prob[i] ** 2)
    bs_min = sum(min_class) / len(min_class)
    bs_maj = sum(maj_class) / len(maj_class)

    #bs = np.mean((y_test - prob) ** 2)
    total_bs = 0
    total_samples = len(y_test)
    batch_size=10000
    for i in range(0, total_samples, batch_size):
        batch_y = y_test[i:i+batch_size]
        batch_prob = prob[i:i+batch_size]
        total_bs += np.mean(np.square(batch_y - batch_prob)) * len(batch_y)
    
    bs = total_bs / total_samples
    
    return [auc_roc, auc_pr, bs_min, bs_maj, bs]

# Example usage:
# auc_roc, auc_pr, bs_min, bs_maj = evaluation_measures(y_test, prob)
# print('ROC AUC:', auc_roc)
# print('PR AUC:', auc_pr)
# print('BS-:', bs_min)
# print('BS+:', bs_maj)
