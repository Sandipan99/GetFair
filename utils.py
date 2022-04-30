import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from scipy.stats import multivariate_normal
import math
import pickle


def genMulNormal(mean, covar, n_samples): # genertaes samples based on multivariate normal distribution
    return np.random.multivariate_normal(mean, covar, n_samples)


def genSyntData(mean, covar, n_samples, sens_at, cls_at): # generate samples with sensitive and class attribute
    sample = genMulNormal(mean, covar, n_samples)
    data = [(sample[i], sens_at, cls_at) for i in range(sample.shape[0])]
    return data


def gen_gaussian(mean_in, cov_in, n_samples, class_label):
    nv = multivariate_normal(mean=mean_in, cov=cov_in)
    X = nv.rvs(n_samples)
    y = np.ones(n_samples, dtype=float) * class_label
    return nv, X, y


def gen_gaussian_diff_size(mean_in, cov_in, z_val, class_label, n):
    nv = multivariate_normal(mean = mean_in, cov = cov_in)
    X = nv.rvs(n)
    y = np.ones(n, dtype=float) * class_label
    z = np.ones(n, dtype=float) * z_val # all the points in this cluster get this value of the sensitive attribute

    return nv, X, y, z


class Data:
    def __init__(self, mean, covar, n_samples):
        self.d_0_1 = genSyntData(mean[0], covar[0], int(n_samples//4), 0, 1)  # sens = 0, cls = 1
        self.d_1_1 = genSyntData(mean[1], covar[1], int(n_samples//4), 1, 1)  # sens = 1, cls = 1
        self.d_0_0 = genSyntData(mean[2], covar[2], int(n_samples//4), 0, 0)  # sens = 0, cls = 0
        self.d_1_0 = genSyntData(mean[3], covar[3], int(n_samples//4), 1, 0)  # sens = 1, cls = 0

        self.tr_dt = None
        self.tr_s = None
        self.tr_c = None
        self.ts_dt = None
        self.ts_s = None
        self.ts_c = None

    def traintestGen(self, train_size=0.2):
        d_0_0_test, d_0_0_train = train_test_split(self.d_0_0, train_size=train_size)
        d_0_1_test, d_0_1_train = train_test_split(self.d_0_1, train_size=train_size)
        d_1_0_test, d_1_0_train = train_test_split(self.d_1_0, train_size=train_size)
        d_1_1_test, d_1_1_train = train_test_split(self.d_1_1, train_size=train_size)

        train_set = d_0_0_train + d_0_1_train + d_1_0_train + d_1_1_train
        test_set = d_0_0_test + d_0_1_test + d_1_0_test + d_1_1_test

        random.shuffle(train_set)

        train_data, train_sens, train_cls = zip(*train_set)
        test_data, test_sens, test_cls = zip(*test_set)

        self.tr_dt = np.array(train_data)
        self.tr_s = np.array(train_sens)
        self.tr_c = np.array(train_cls)
        self.ts_dt = np.array(test_data)
        self.ts_s = np.array(test_sens)
        self.ts_c = np.array(test_cls)


class DataDispImp:
    def __init__(self, mean, covar, n_samples, disc_factor):
        nv1, X1, y1 = gen_gaussian(mean[0], covar[0], n_samples, 1)  # positive class
        nv2, X2, y2 = gen_gaussian(mean[1], covar[1], n_samples, 0)  # negative class
        X = np.vstack((X1, X2))
        y = np.hstack((y1, y2))
        data_ = list(zip(X, y))
        random.shuffle(data_)
        rotation_mult = np.array(
            [[math.cos(disc_factor), -math.sin(disc_factor)], [math.sin(disc_factor), math.cos(disc_factor)]])
        self.data_aux = [(np.dot(r[0], rotation_mult), r[1]) for r in data_]
        self.x_control = []  # this array holds the sensitive feature value
        for i in range(0, len(self.data_aux)):
            x = self.data_aux[i][0]

            # probability for each cluster that the point belongs to it
            p1 = nv1.pdf(x)
            p2 = nv2.pdf(x)

            # normalize the probabilities from 0 to 1
            s = p1 + p2
            p1 = p1 / s
            p2 = p2 / s

            r = np.random.uniform()  # generate a random number from 0 to 1

            if r < p1:  # the first cluster is the positive class
                self.x_control.append(1.0)  # 1.0 means its male
            else:
                self.x_control.append(0.0)  # 0.0 -> female

        self.x_control = np.array(self.x_control)
        d, cls = zip(*self.data_aux)
        self.data = list(zip(d, cls, self.x_control))

    def train_test_split(self, train_size=0.7, val=True):
        random.shuffle(self.data)
        if val:
            tr_size = int(train_size*len(self.data))
            val_size = int((train_size+0.15)*len(self.data))
            train_data, train_cls, train_sens = zip(*self.data[:tr_size])
            test_data, test_cls, test_sens = zip(*self.data[tr_size:val_size])
            val_data, val_cls, val_sens = zip(*self.data[val_size:])
            self.vl_dt = np.array(val_data)
            self.vl_s = np.array(val_sens)
            self.vl_c = np.array(val_cls)
        else:
            size = int(train_size*len(self.data))
            train_data, train_cls, train_sens = zip(*self.data[:size])
            test_data, test_cls, test_sens = zip(*self.data[size:])

        self.tr_dt = np.array(train_data)
        self.tr_s = np.array(train_sens)
        self.tr_c = np.array(train_cls)
        self.ts_dt = np.array(test_data)
        self.ts_s = np.array(test_sens)
        self.ts_c = np.array(test_cls)

    def test_train_split_comp(self, tune=None):
        '''
        Splits the data into 80% train and 20% test sets
        Tunes on a subset of the test set if None then on the entire test set
        '''
        random.shuffle(self.data)
        size = int(0.8*(len(self.data)))
        train_data_ = self.data[:size]
        val_data_ = self.data[size:]
        train_data, train_cls, train_sens = zip(*train_data_)
        val_data, val_cls, val_sens = zip(*val_data_)
        if tune is None:
            test_data, test_cls, test_sens = zip(*train_data_)
        else:
            random.shuffle(train_data_)
            size = int(tune*len(train_data_))
            test_data, test_cls, test_sens = zip(*train_data_[:size])

        self.vl_dt = np.array(val_data)
        self.vl_s = np.array(val_sens)
        self.vl_c = np.array(val_cls)
        self.tr_dt = np.array(train_data)
        self.tr_s = np.array(train_sens)
        self.tr_c = np.array(train_cls)
        self.ts_dt = np.array(test_data)
        self.ts_s = np.array(test_sens)
        self.ts_c = np.array(test_cls)


class DataDispMis:
    def __init__(self, mean, covar, n_samples):
        nv1, X1, y1, z1 = gen_gaussian_diff_size(mean[0], covar[0], 1, 1, int(n_samples * 1))  # z=1, +
        nv2, X2, y2, z2 = gen_gaussian_diff_size(mean[1], covar[1], 0, 1, int(n_samples * 1))  # z=0, +
        nv3, X3, y3, z3 = gen_gaussian_diff_size(mean[2], covar[2], 1, 0, int(n_samples * 1))  # z=1, -
        nv4, X4, y4, z4 = gen_gaussian_diff_size(mean[3], covar[3], 0, 0, int(n_samples * 1))  # z=0, -

        X = np.vstack((X1, X2, X3, X4))
        y = np.hstack((y1, y2, y3, y4))
        x_control = np.hstack((z1, z2, z3, z4))

        self.data = list(zip(X, y, x_control))

    def train_test_split(self, train_size=0.7, val=True):
        random.shuffle(self.data)
        if val:
            tr_size = int(train_size*len(self.data))
            val_size = int((train_size+0.15)*len(self.data))
            train_data, train_cls, train_sens = zip(*self.data[:tr_size])
            test_data, test_cls, test_sens = zip(*self.data[tr_size:val_size])
            val_data, val_cls, val_sens = zip(*self.data[val_size:])
            self.vl_dt = np.array(val_data)
            self.vl_s = np.array(val_sens)
            self.vl_c = np.array(val_cls)
        else:
            size = int(train_size*len(self.data))
            train_data, train_cls, train_sens = zip(*self.data[:size])
            test_data, test_cls, test_sens = zip(*self.data[size:])

        self.tr_dt = np.array(train_data)
        self.tr_s = np.array(train_sens)
        self.tr_c = np.array(train_cls)
        self.ts_dt = np.array(test_data)
        self.ts_s = np.array(test_sens)
        self.ts_c = np.array(test_cls)

    def test_train_split_comp(self, tune=None):
        '''
        Splits the data into 80% train and 20% test sets
        Tunes on a subset of the test set if None then on the entire test set
        '''
        random.shuffle(self.data)
        size = int(0.8*(len(self.data)))
        train_data_ = self.data[:size]
        val_data_ = self.data[size:]
        train_data, train_cls, train_sens = zip(*train_data_)
        val_data, val_cls, val_sens = zip(*val_data_)
        if tune is None:
            test_data, test_cls, test_sens = zip(*train_data_)
        else:
            random.shuffle(train_data_)
            size = int(tune*len(train_data_))
            test_data, test_cls, test_sens = zip(*train_data_[:size])

        self.vl_dt = np.array(val_data)
        self.vl_s = np.array(val_sens)
        self.vl_c = np.array(val_cls)
        self.tr_dt = np.array(train_data)
        self.tr_s = np.array(train_sens)
        self.tr_c = np.array(train_cls)
        self.ts_dt = np.array(test_data)
        self.ts_s = np.array(test_sens)
        self.ts_c = np.array(test_cls)


class Dataset: # class for real-world datasets
    def __init__(self, fname, scale=True):
        with open(fname, 'rb') as fs:
            data_ = pickle.load(fs)
        if scale:
            scaler = StandardScaler()
            scaler.fit(data_['data'])
            scaled_data = scaler.transform(data_['data'])
            self.data = list(zip(scaled_data, data_['s_attr'], data_['class']))
        else:
            self.data = list(zip(data_['data'], data_['s_attr'], data_['class']))

    def train_test_split(self, train_size=0.7, val=True):
        random.shuffle(self.data)
        if val:
            tr_size = int(len(self.data)*train_size)
            val_size = int((train_size + 0.15) * len(self.data))
            train_data, train_sens, train_cls = zip(*self.data[:tr_size])
            test_data, test_sens, test_cls = zip(*self.data[tr_size:val_size])
            val_data, val_sens, val_cls = zip(*self.data[val_size:])
            self.vl_dt = np.array(val_data)
            self.vl_s = np.array(val_sens)
            self.vl_c = np.array(val_cls)

        else:
            size = int(len(self.data)*train_size)
            train_data, train_sens, train_cls = zip(*self.data[:size])
            test_data, test_sens, test_cls = zip(*self.data[size:])

        self.tr_dt = np.array(train_data)
        self.tr_s = np.array(train_sens)
        self.tr_c = np.array(train_cls)
        self.ts_dt = np.array(test_data)
        self.ts_s = np.array(test_sens)
        self.ts_c = np.array(test_cls)

    def test_train_split_comp(self, tune=None):
        '''
        Splits the data into 80% train and 20% test sets
        Tunes on a subset of the test set if None then on the entire test set
        '''
        random.shuffle(self.data)
        tr_size = int(len(self.data)*0.8)
        train_data_ = self.data[:tr_size]
        self.tr_dt, self.tr_s, self.tr_c = zip(*train_data_)
        self.vl_dt, self.vl_s, self.vl_c = zip(*self.data[tr_size:])
        if tune is None:
            self.ts_dt, self.ts_s, self.ts_c = zip(*train_data_)
        else:
            size = int(tune*len(train_data_))
            self.ts_dt, self.ts_s, self.ts_c = zip(*train_data_[:size])


if __name__ == "__main__":
    means = [[2, 2], [2, 2], [-2, -2], [-1, 0]]
    covars = [[[3, 1], [1, 3]], [[3, 1], [1, 3]], [[3, 1], [1, 3]], [[3, 1], [1, 3]]]
    n_samples = 2000
    disc_factor = math.pi / 4.0
    data = DataDispImp(means, covars, n_samples, disc_factor)
    print(data.data_aux[100])
    print(data.x_control[100])
    print(data.data[100])
    #data = Dataset('Datasets/compas_data.p')
    data.train_test_split(train_size=0.8, val=False)
    print(len(data.tr_dt), len(data.tr_c), len(data.tr_s))
    print(len(data.ts_dt), len(data.ts_c), len(data.ts_s))
    #print(len(data.vl_dt), len(data.vl_c), len(data.vl_s))