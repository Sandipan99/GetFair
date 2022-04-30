from Rewardfunctions import stat_parity, diff_FPR, diff_FNR, \
    diff_FPR_FNR, diff_Eodd, diff_Eoppr
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from utils import DataDispImp, Data, DataDispMis, Dataset

data = Dataset('Datasets/compas_data.p')

data.train_test_split(train_size=0.8, val=False)
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(data.tr_dt, data.tr_c)
ts_pred = clf.predict(data.ts_dt)

print(stat_parity(data.ts_s, ts_pred))
print(diff_FPR(data.ts_s, ts_pred, data.ts_c))
print(diff_FNR(data.ts_s, ts_pred, data.ts_c))
print(diff_FPR_FNR(data.ts_s, ts_pred, data.ts_c))
print(diff_Eoppr(data.ts_s, ts_pred, data.ts_c))
print(diff_Eodd(data.ts_s, ts_pred, data.ts_c))
