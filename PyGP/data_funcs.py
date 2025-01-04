import math
import numpy as np
import random
import pmlb
from glob import glob
from yaml import load, Loader
import gzip
from PyGP.read_file import read_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

sc_y = StandardScaler()

def Hartman_func(data):
    return [-(math.exp(-(
            data[0][i] ** 2 + data[1][i] ** 2 + data[2][i] ** 2 +
            data[3][i] ** 2))) for i in range(len(data[0]))]

def Nonic_func(data):
    fitness = []
    for i in range(len(data[0])):
        data_tmp = 0
        for j in range(1, 9):
            data_tmp += data[0][i] ** j
        fitness.append(data_tmp)
    return fitness

def F1_funcs(data):
    return [0.2 + 0.4 * data[0][i] ** 2 + 0.3 * np.sin(15 * data[0][i]) + 0.05 * np.cos(50 * data[0][i]) for i in range(len(data[0]))]

def Nguyen5(data):
    return [np.sin(data[0][i] ** 2) * np.cos(data[0][i]) - 1 for i in range(len(data[0]))]

def R1(data):
    return [(data[0][i] + 1) ** 3 / (data[0][i] ** 2 - data[0][i] + 1) for i in range(len(data[0]))]

def EXP1(data):
    return [np.exp(data[0][i] * data[1][i]) + 20 * (data[2][i] * data[3][i]) ** 2 + 10 * data[4][i] for i in range(len(data[0]))]

def RatPol3D(data):
    return [30 * (data[0][i] - 1) * (data[2][i] - 1) / (data[1][i] ** 2 * (data[0][i] - 10)) for i in range(len(data[0]))]

dataset_dict = {
    "Nonic": [Nonic_func, [[-2, 2]], 20, 1000],
    "R1": [R1, [[-2, 2]], 20, 1000],
    "RatPol3D": [RatPol3D, [[0.05, 2], [1, 2], [0.05, 2]], 300, 2700],
    "EXP1": [EXP1, [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]], 650, 80],
    "F1": [F1_funcs, [[0., 1.]], 100, 1000],
    "Hartman": [Hartman_func, [[0, 2], [0, 2], [0, 2], [0, 2]], 100, 900],
    "Nguyen5": [Nguyen5, [[-1, 1]], 20, 1000],
}

# fixed seeds for the experiment
SEEDS = [
	    23654, 15795,   860,  5390, 16850, 29910,  4426, 21962, 14423, 28020,
        29802, 21575, 11964, 11284, 22118,  6265, 11363, 27495, 16023,  8322,
        1685,  32052,   769, 26967, 30187, 32157, 23333,  2433,  5311,  5051,
        6420,  17568, 20939, 19769, 28693,  6396, 29419, 27480, 32304,  8666,
        25658, 18942, 24233, 18431, 32219,  2747, 25551, 26382,   189, 31677,
        19118,  3005, 21042,  1899, 24118,  1267, 31551, 17912, 11394,  3556,
        3890,   8838, 30740, 27464, 14502, 21777, 10627,  8792, 10555, 10253,
        8433,  10233, 11016, 23897,  2612, 23425, 25939, 22619, 21870, 23483,
        26054, 15787, 27132, 17159, 12206,  8226, 14541,  3152, 26531,  1585,
        3943,  23939, 19457,  1021, 11653, 10805, 13417, 20227,  7989, 9692
]

def dataset_save(run, train_data, test_data):
    with open('F' + str(run[0]) + '_' + str(run[1]) + '_training_data.txt', 'w') as f:
        print('{0}-{1}'.format(len(train_data[1]), len(train_data[0])))
        f.write('{0} {1}\n'.format(len(train_data[1]), len(train_data[0])))
        for i in range(len(train_data[1])):
            o_file = []
            for j in range(len(train_data[0])):
                o_file.append(str(train_data[0][j][i]) + ' ')
            o_file.append(str(train_data[1][i]))
            o_file.append('\n')
            f.write(''.join(o_file))

    with open('F' + str(run[0]) + '_' + str(run[1]) + '_test_data.txt', 'w') as f:
        print('{0}-{1}'.format(len(test_data[1]), len(test_data[0])))
        f.write('{0} {1}\n'.format(len(test_data[1]), len(test_data[0])))
        for i in range(len(test_data[1])):
            o_file = []
            for j in range(len(test_data[0])):
                o_file.append(str(test_data[0][j][i]) + ' ')
            o_file.append(str(test_data[1][i]))
            o_file.append('\n')
            f.writelines(''.join(o_file))



def read_data_(train_file, test_file, init_seed=1234, scale_x=False, scale_y=False):
    np.random.seed(init_seed)
    random.seed(init_seed)
    np.random.SeedSequence(init_seed)
    input = pd.read_csv(train_file, header=None, sep=' ')
    print('train_file: ', train_file)
    print(input.values)
    print(input.columns[-1])
    X_train = input.drop(input.columns[-1], axis=1).values
    y_train = input[input.columns[-1]].values
    # y_train = input.columns[-1]

    input = pd.read_csv(test_file, header=None, sep=' ')
    X_test = input.drop(input.columns[-1], axis=1).values
    y_test = input[input.columns[-1]].values
    # y_test = input.columns[-1]

    # scale and normalize the data
    if scale_x:
        print('scaling X')
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)

    if scale_y:
        print('scaling y')
        global sc_y
        y_train = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    ##################################################
    # noise
    ##################################################
    target_noise = 0
    feature_noise = 0
    if target_noise > 0:
        print('adding', target_noise, 'noise to target')
        y_train += np.random.normal(0,
                                    target_noise * np.sqrt(np.mean(np.square(y_train))),
                                    size=len(y_train))
    # add noise to the features
    if feature_noise > 0:
        print('adding', target_noise, 'noise to features')
        X_train = np.array([x + np.random.normal(0, feature_noise * np.sqrt(np.mean(np.square(x))),
                                                 size=len(x))
                            for x in X_train.T]).T

    data_size = len(y_train) + len(y_test)
    train_size = len(y_train)
    test_size = len(y_test)
    r_vals = np.random.uniform(0, 1, data_size)
    n_terms = X_train.shape[1]
    train_set = range(train_size)
    data = np.array([[X_train[train_set[j]][i] for j in range(train_size)] for i in
                     range(n_terms)])  # np.array([data_all[i][arg_train] for i in range(n_terms)])
    fitness = y_train[train_set]  # fitness_all[arg_train]
    data_candidate = np.array([[X_test[j][i] for j in range(test_size)] for i in
                               range(n_terms)])  # np.array([data_all[i][arg_test] for i in range(n_terms)])
    fitness_candidate = y_test  # fitness_all[arg_test]
    data_candidate_1 = np.array([[X_test[j][i] for j in range(test_size)] for i in
                                 range(n_terms)])  # np.array([data_all[i][arg_test] for i in range(n_terms)])
    fitness_candidate_1 = y_test  # fitness_all[arg_test]
    data_curve = data_candidate.copy()
    fitness_curve = fitness_candidate.copy()
    curve_size = len(fitness_curve)
    range_curve_1 = [[np.min(data[i]), np.max(data[i])] for i in range(n_terms)]
    range_curve_2 = [[np.min(data_candidate[i]), np.max(data_candidate[i])] for i in range(n_terms)]
    # range_curve = range_ = [[np.min([range_curve_1[i], range_curve_2[i]]), np.max([range_curve_1[i], range_curve_2[i]])] for i in range(n_terms)]

    # range_curve = range_ = [[-1, 1] for i in range(n_terms)]

    range_curve = range_ = [[float(math.floor(np.min(X_train[:][i]))), float(math.ceil(np.max(X_train[:][i])))] for i in
                            range(n_terms)]
    # assert (0 == 1)
    return (
    curve_size, n_terms, range_, data_size, data, fitness, range_curve, data_curve, fitness_curve, data_candidate,
    fitness_candidate, data_candidate_1, fitness_candidate_1)

def read_data_Feyn(train_file, test_file, init_seed=1234, scale_x=False, scale_y=False):
    np.random.seed(init_seed)
    random.seed(init_seed)
    np.random.SeedSequence(init_seed)
    input = pd.read_csv(train_file, header=None, sep=' ')
    X_train = input.drop(input.columns[-1], axis=1).values
    y_train = input[input.columns[-1]].values

    input = pd.read_csv(test_file, header=None, sep=' ')
    X_test = input.drop(input.columns[-1], axis=1).values
    y_test = input[input.columns[-1]].values

    if scale_x:
        print('scaling X')
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)

    if scale_y:
        print('scaling y')
        global sc_y
        y_train = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    ##################################################
    # noise
    ##################################################
    target_noise = 0
    feature_noise = 0
    if target_noise > 0:
        print('adding', target_noise, 'noise to target')
        y_train += np.random.normal(0,
                                    target_noise * np.sqrt(np.mean(np.square(y_train))),
                                    size=len(y_train))
    # add noise to the features
    if feature_noise > 0:
        print('adding', target_noise, 'noise to features')
        X_train = np.array([x + np.random.normal(0, feature_noise * np.sqrt(np.mean(np.square(x))),
                                                 size=len(x))
                            for x in X_train.T]).T

    data_size = len(y_train) + len(y_test)
    train_size = len(y_train)
    test_size = len(y_test)
    r_vals = np.random.uniform(0, 1, data_size)
    n_terms = X_train.shape[1]
    train_set = range(train_size)
    data = np.array([[X_train[train_set[j]][i] for j in range(train_size)] for i in
                     range(n_terms)])  # np.array([data_all[i][arg_train] for i in range(n_terms)])
    fitness = y_train[train_set]  # fitness_all[arg_train]
    data_candidate = np.array([[X_test[j][i] for j in range(test_size)] for i in
                               range(n_terms)])  # np.array([data_all[i][arg_test] for i in range(n_terms)])
    fitness_candidate = y_test  # fitness_all[arg_test]
    range_ = [[float(math.floor(np.min(X_train[:,i]))), float(math.ceil(np.max(X_train[:,i])))] for i in
                            range(n_terms)]
    return (
    n_terms, range_, data_size, data, fitness, data_candidate, fitness_candidate)

def read_data(train_file, test_file, init_seed=1234, scale_x=False, scale_y=False):
    np.random.seed(init_seed)
    random.seed(init_seed)
    np.random.SeedSequence(init_seed)
    input = pd.read_csv(train_file, header=None, sep=',')
    print('train_file: ', train_file)
    print(input.values)
    print(input.columns[-1])
    X_train = input.drop(input.columns[-1], axis=1).values
    y_train = input[input.columns[-1]].values
    # y_train = input.columns[-1]

    input = pd.read_csv(test_file, header=None, sep=',')
    X_test = input.drop(input.columns[-1], axis=1).values
    y_test = input[input.columns[-1]].values
    # y_test = input.columns[-1]

    # scale and normalize the data
    if scale_x:
        print('scaling X')
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)

    if scale_y:
        print('scaling y')
        global sc_y
        y_train = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    ##################################################
    # noise
    ##################################################
    target_noise = 0
    feature_noise = 0
    if target_noise > 0:
        print('adding', target_noise, 'noise to target')
        y_train += np.random.normal(0,
                                    target_noise * np.sqrt(np.mean(np.square(y_train))),
                                    size=len(y_train))
    # add noise to the features
    if feature_noise > 0:
        print('adding', target_noise, 'noise to features')
        X_train = np.array([x + np.random.normal(0, feature_noise * np.sqrt(np.mean(np.square(x))),
                                                 size=len(x))
                            for x in X_train.T]).T

    data_size = len(y_train) + len(y_test)
    train_size = len(y_train)
    test_size = len(y_test)
    r_vals = np.random.uniform(0, 1, data_size)
    n_terms = X_train.shape[1]
    train_set = range(train_size)
    data = np.array([[X_train[train_set[j]][i] for j in range(train_size)] for i in
                     range(n_terms)])  # np.array([data_all[i][arg_train] for i in range(n_terms)])
    fitness = y_train[train_set]  # fitness_all[arg_train]
    data_candidate = np.array([[X_test[j][i] for j in range(test_size)] for i in
                               range(n_terms)])  # np.array([data_all[i][arg_test] for i in range(n_terms)])
    fitness_candidate = y_test  # fitness_all[arg_test]
    data_candidate_1 = np.array([[X_test[j][i] for j in range(test_size)] for i in
                                 range(n_terms)])  # np.array([data_all[i][arg_test] for i in range(n_terms)])
    fitness_candidate_1 = y_test  # fitness_all[arg_test]
    data_curve = data_candidate.copy()
    fitness_curve = fitness_candidate.copy()
    curve_size = len(fitness_curve)
    range_curve_1 = [[np.min(data[i]), np.max(data[i])] for i in range(n_terms)]
    range_curve_2 = [[np.min(data_candidate[i]), np.max(data_candidate[i])] for i in range(n_terms)]
    # range_curve = range_ = [[np.min([range_curve_1[i], range_curve_2[i]]), np.max([range_curve_1[i], range_curve_2[i]])] for i in range(n_terms)]

    # range_curve = range_ = [[-1, 1] for i in range(n_terms)]

    range_curve = range_ = [[float(math.floor(np.min(X_train[:][i]))), float(math.ceil(np.max(X_train[:][i])))] for i in
                            range(n_terms)]
    # assert (0 == 1)
    return (
    curve_size, n_terms, range_, data_size, data, fitness, range_curve, data_curve, fitness_curve, data_candidate,
    fitness_candidate, data_candidate_1, fitness_candidate_1)


class Dataset:
    def __init__(self, func, range_, train, test):
        self.func = func
        self.n_terms = len(range_)
        self.range_ = range_
        self.data_size = train
        self.test_size = test
        self.datasets = None
    def funcs(self, data):
        return self.func(data)
    def set_data_size(self, train, test):
        self.data_size = train
        self.test_size = test

    def set_dataset(self, dataset_dir):
        if dataset_dir.endswith('.tsv.gz'):
            self.datasets = [dataset_dir]
        elif dataset_dir.endswith('*'):
            print('capturing glob', dataset_dir + '/*.tsv.gz')
            self.datasets = sorted(glob(dataset_dir + '*/*.tsv.gz'))
        else:
            self.datasets = sorted(glob(dataset_dir + '/*/*.tsv.gz'))

    def from_pmlb(self, data_id, seed_id, SYM_DATA=True, scale_x=True, scale_y=True, use_dataframe=True):
        seed = SEEDS[seed_id]
        
        np.random.seed(seed)
        # random.seed(1234)
        # np.random.SeedSequence(1234)
        if self.datasets is not None:
            # print(len(self.datasets), self.datasets[data_id], self.datasets)
            dataset = self.datasets[data_id]
            # if not SYM_DATA and any([n in dataset for n in ['feynman', 'strogatz', 'fri']]):
            #     return None
            # elif SYM_DATA:
            #     if not any([n in dataset for n in ['fri']]):
            #         return None
            # print('----------------------------------', dataset, '/'.join(dataset.split('/')[:-1]) + '/metadata.yaml', 'r')
            metadata = load(
                open('/'.join(dataset.split('/')[:-1]) + '/metadata.yaml', 'r'),
                Loader=Loader)
            print('here..............')
            if metadata['task'] != 'regression':
                return None
            print('here..............')
            dataname = dataset.split('/')[-1].split('.tsv.gz')[0]
            print(metadata)
            print(dataname, len(metadata), dataset, type(dataset))
            (X, y, feature_names) = read_file(dataset)
        assert (len(X) == len(y))
        print(len(X), X.shape[1])
        n_terms = X.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=0.75,
                                                            test_size=0.25,
                                                            random_state=seed)
        X = X.values
        X_train = X_train.values
        X_test = X_test.values
        max_train_samples=10000
        if max_train_samples > 0 and len(y_train) > max_train_samples:
            # return None
            print('subsampling training data from', len(X_train),
                  'to', max_train_samples)
            sample_idx = np.random.choice(np.arange(len(X_train)),
                                          size=8000, replace=False)
            sample_idx_test = np.random.choice(np.arange(len(X_test)),
                                          size=int(max_train_samples / 4), replace=False)
            y_train = y_train[sample_idx]
            if isinstance(X_train, pd.DataFrame):
                X_train.reindex(np.arange(len(X_train)))
                X_train = X_train.loc[sample_idx]
            else:
                X_train = X_train[sample_idx]

            y_test = y_test[sample_idx_test]
            if isinstance(X_test, pd.DataFrame):
                # print(X_test.index)
                X_test.reindex(np.arange(len(X_test)))
                X_test = X_test.loc[sample_idx_test]
            else:
                X_test = X_test[sample_idx_test]

        # scale and normalize the data
        if scale_x:
            print('scaling X')
            sc_X = StandardScaler()
            X_train_scaled = sc_X.fit_transform(X_train)
            X_test_scaled = sc_X.transform(X_test)
            if use_dataframe:
                X_train = pd.DataFrame(X_train_scaled,
                                              columns=feature_names)
                X_test = pd.DataFrame(X_test_scaled,
                                             columns=feature_names)

        if scale_y:
            print('scaling y')
            global sc_y
            y_train = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()


        ##################################################
        # noise
        ##################################################
        target_noise=0.0
        feature_noise=0
        if target_noise > 0:
            print('adding', target_noise, 'noise to target')
            y_train += np.random.normal(0,
                                               target_noise * np.sqrt(np.mean(np.square(y_train))),
                                               size=len(y_train))
        # add noise to the features
        if feature_noise > 0:
            print('adding', target_noise, 'noise to features')
            X_train = np.array([x
                                       + np.random.normal(0, feature_noise * np.sqrt(np.mean(np.square(x))),
                                                          size=len(x))
                                       for x in X_train.T]).T


        data_size = len(y)
        train_size = len(y_train)
        test_size = len(y_test)
        data_all = np.array([[X[j][i]for j in range(data_size)] for i in range(n_terms)])
        fitness_all = y#np.array([X[j][n_terms] for j in range(data_size)])
        r_vals = np.random.uniform(0, 1, data_size)
        arg_train = np.where(r_vals < 0.75)
        arg_test = np.where(r_vals >= 0.75)
        train_set = range(train_size)
        data = np.array([[X_train[train_set[j]][i]for j in range(train_size)] for i in range(n_terms)])#np.array([data_all[i][arg_train] for i in range(n_terms)])
        fitness = y_train[train_set]#fitness_all[arg_train]
        data_candidate = np.array([[X_test[j][i]for j in range(test_size)] for i in range(n_terms)])#np.array([data_all[i][arg_test] for i in range(n_terms)])
        fitness_candidate = y_test#fitness_all[arg_test]
        data_candidate_1 = np.array([[X_test[j][i]for j in range(test_size)] for i in range(n_terms)])#np.array([data_all[i][arg_test] for i in range(n_terms)])
        fitness_candidate_1 = y_test#fitness_all[arg_test]
        data_curve = data_candidate.copy()
        fitness_curve = fitness_candidate.copy()
        curve_size = len(fitness_curve)
        range_curve_1 = [[np.min(data[i]), np.max(data[i])] for i in range(n_terms)]
        range_curve_2 = [[np.min(data_candidate[i]), np.max(data_candidate[i])] for i in range(n_terms)]
        # range_curve = range_ = [[np.min([range_curve_1[i], range_curve_2[i]]), np.max([range_curve_1[i], range_curve_2[i]])] for i in range(n_terms)]

        range_curve = range_ = [[float(math.floor(np.min(X_train[:][i]))), float(math.ceil(np.max(X_train[:][i])))] for i in range(n_terms)]
        # assert (0 == 1)
        return (curve_size, n_terms, range_, data_size, data, fitness, range_curve, data_curve, fitness_curve, data_candidate,
         fitness_candidate, data_candidate_1, fitness_candidate_1)

    def get_data(self):
        rand_val = [
            np.linspace(self.range_[0][0], self.range_[0][1], self.data_size) if self.n_terms == 1 else
            np.random.uniform(self.range_[j][0], self.range_[j][1], self.data_size) for j in range(self.n_terms)]
        # rand_val = [np.arange(self.range_[i][0], self.range_[i][1], (self.range_[i][1] - self.range_[i][0]) / self.data_size) for i in range(self.n_terms)]
        data = rand_val#[np.random.uniform(self.range_[j][0], self.range_[j][1], self.data_size) for j in range(self.n_terms)]
        data_candidate = [np.random.uniform(self.range_[j][0], self.range_[j][1], self.test_size) for j in range(self.n_terms)]
        data_candidate_1 = [np.random.uniform(self.range_[j][0] - 0.2, self.range_[j][1] + 0.2, self.test_size) for j in range(self.n_terms)]
        fitness = self.funcs(data)
        fitness_candidate = self.funcs(data_candidate)
        fitness_candidate_1 = self.funcs(data_candidate_1)

        curve_size = 100 * (self.range_[0][1] - self.range_[0][0])
        data_curve = data_candidate if self.n_terms > 1 else [
            np.arange(self.range_[i][0], self.range_[i][1], (self.range_[i][1] - self.range_[i][0]) / curve_size) for i in range(self.n_terms) ]
        curve_size = len(data_curve[0])
        fitness_curve = self.funcs(data_curve)
        # return [(data, fitness), (data_candidate, fitness_candidate), (data_candidate_1, fitness_candidate_1), (data_curve, fitness_curve)]
        print(len(data[0]), len(data_candidate[0]), len(fitness_candidate))
        return (curve_size, self.n_terms, self.range_, self.data_size, data, fitness, self.range_, data_curve, fitness_curve, data_candidate,
         fitness_candidate, data_candidate_1, fitness_candidate_1)
