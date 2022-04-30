from fairnessMetrics import disparateMistreatment, equal_opportunity, equalized_odds, disparateImpact


def stat_parity(ts_s, ts_pred, ts_c=None, verbose=False):
    r_0, r_1 = disparateImpact(ts_s, ts_pred)
    if r_0 == 0 or r_1 == 0:
        return None
    return min(r_0/r_1, r_1/r_0)


def diff_FPR(ts_s, ts_pred, ts_c):
    fpr_0, fnr_0, fpr_1, fnr_1 = disparateMistreatment(ts_s, ts_pred, ts_c)
    return 1-abs(fpr_0 - fpr_1)


def diff_FNR(ts_s, ts_pred, ts_c):
    fpr_0, fnr_0, fpr_1, fnr_1 = disparateMistreatment(ts_s, ts_pred, ts_c)
    return 1-abs(fnr_0 - fnr_1)


def diff_FPR_FNR(ts_s, ts_pred, ts_c, verbose=False):
    fpr_0, fnr_0, fpr_1, fnr_1 = disparateMistreatment(ts_s, ts_pred, ts_c)
    #print(f'false positive rate class 0: {fpr_0}')
    #print(f'false positive rate class 1: {fpr_1}')
    #print(f'false negative rate class 0: {fnr_0}')
    #print(f'false negative rate class 1: {fnr_1}')
    if verbose:
        return abs(fnr_0 - fnr_1) + abs(fpr_0 - fpr_1)
    return (1 - abs(fnr_0 - fnr_1)) + (1 - abs(fpr_0 - fpr_1))


def diff_Eoppr(ts_s, ts_pred, ts_c, verbose=False):
    tpr_0, tpr_1 = equal_opportunity(ts_s, ts_pred, ts_c)
    #print(f'true positive rate class 0: {tpr_0}')
    #print(f'true positive rate class 1: {tpr_1}')
    if verbose:
        return abs(tpr_0 - tpr_1)
    return 1 - abs(tpr_0 - tpr_1)


def diff_Eodd(ts_s, ts_pred, ts_c, verbose=False):
    tpr_0, tpr_1, fpr_0, fpr_1 = equalized_odds(ts_s, ts_pred, ts_c)
    if verbose:
        return abs(tpr_0 - tpr_1) + abs(fpr_0 - fpr_1)
    return (1 - abs(tpr_0 - tpr_1)) + (1 - abs(fpr_0 - fpr_1))