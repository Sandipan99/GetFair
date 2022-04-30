from sklearn.metrics import confusion_matrix


def disparateImpact(ts_s, ts_pred, ts_c=None):
    '''
    ts_s - sensitive attribute of the datapoint
    ts_predict - predicted class of the datapoint by the classifier
    The metric essentially checks whether the classifier is more probable to classify a datapoint
    as positive belonging to a particular class. Note that ground-truth class values are not required
    in this case.
    '''
    acc_s_0 = []
    acc_s_1 = []

    for i in range(ts_pred.shape[0]):
        if ts_s[i] == 0:
            acc_s_0.append(ts_pred[i])
        else:
            acc_s_1.append(ts_pred[i])

    '''
    The code below only calculates a modified version of disparate impact which also includes ground truth class
    We essentially check how what is the predicted accuracy with respect to the ground truth for a given sensitive 
    class attribute
    '''
    #for i in range(ts_pred.shape[0]):
    #    if ts_s[i] == 0:
    #        if ts_pred[i] == ts_c[i]:
    #            acc_s_0.append(1)
    #        else:
    #            acc_s_0.append(0)
    #    else:
    #        if ts_pred[i] == ts_c[i]:
    #            acc_s_1.append(1)
    #        else:
    #            acc_s_1.append(0)

    # print(f'accuracy for sensistive attribute 0 - {sum(acc_s_0)/len(acc_s_0)}')
    # print(f'accuracy for sensistive attribute 1 - {sum(acc_s_1)/len(acc_s_1)}')

    return sum(acc_s_0) / len(acc_s_0), sum(acc_s_1) / len(acc_s_1)


def disparateMistreatment(ts_s, ts_pred, ts_c, verbose=False):
    s_0_p = []
    s_0_or = []
    s_1_p = []
    s_1_or = []

    for i in range(ts_pred.shape[0]):
        if ts_s[i] == 0:
            s_0_p.append(ts_pred[i])
            s_0_or.append(ts_c[i])
        else:
            s_1_p.append(ts_pred[i])
            s_1_or.append(ts_c[i])

    tn_0, fp_0, fn_0, tp_0 = confusion_matrix(s_0_or, s_0_p).ravel()
    tn_1, fp_1, fn_1, tp_1 = confusion_matrix(s_1_or, s_1_p).ravel()

    fpr_0 = fp_0 / (fp_0 + tn_0)
    fpr_1 = fp_1 / (fp_1 + tn_1)

    fnr_0 = fn_0 / (fn_0 + tp_0)
    fnr_1 = fn_1 / (fn_1 + tp_1)

    # print(f'False positive rate for attr 0: {fpr_0}, false negative rate: {fnr_0}')
    # print(f'False positive rate for attr 1: {fpr_1}, false negative rate: {fnr_1}')
    if verbose:
        return (abs(fpr_0 - fpr_1) + abs(fnr_0 - fnr_1))

    return fpr_0, fnr_0, fpr_1, fnr_1


def equal_opportunity(ts_s, ts_pred, ts_c, verbose=False):
    s_0_p = []
    s_0_or = []
    s_1_p = []
    s_1_or = []

    for i in range(ts_pred.shape[0]):
        if ts_s[i] == 0:
            s_0_p.append(ts_pred[i])
            s_0_or.append(ts_c[i])
        else:
            s_1_p.append(ts_pred[i])
            s_1_or.append(ts_c[i])

    tn_0, fp_0, fn_0, tp_0 = confusion_matrix(s_0_or, s_0_p).ravel()
    tn_1, fp_1, fn_1, tp_1 = confusion_matrix(s_1_or, s_1_p).ravel()

    tpr_0 = tp_0/(tp_0 + fn_0)
    tpr_1 = tp_1/(tp_1 + fn_1)



    if verbose:
        return abs(tpr_0 - tpr_1)

    return tpr_0, tpr_1


def equalized_odds(ts_s, ts_pred, ts_c, verbose=False):
    tpr_0, tpr_1 = equal_opportunity(ts_s, ts_pred, ts_c)
    fpr_0, fnr_0, fpr_1, fnr_1 = disparateMistreatment(ts_s, ts_pred, ts_c)
    if verbose:
        return abs(tpr_0 - tpr_1) + abs(fpr_0 - fpr_1)
    return tpr_0, tpr_1, fpr_0, fpr_1
