from dataset import Dataset
from config import data_path
from EDA import MyPlot
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from timeit import default_timer as timer
from config import data_msg
from sklearn.pipeline import Pipeline
from parm_select import parm_optimize
import warnings
warnings.filterwarnings("ignore")

def parm(model, ds, start, end, step):
    n_neighbors = np.arange(start,end,step)
    weights = ['uniform', 'distance']
    param_grid = dict(n_neighbors=n_neighbors, weights=weights)
    parm_optimize(model, param_grid, scoring[2], cv, ds, is_all=False)

if __name__ == '__main__':

    # config
    scoring = ['precision_weighted', 'recall_weighted', 'f1_weighted']
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    # 1） iris
    data_name = 'iris'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()
    ds.select_feature()

    knn = KNeighborsClassifier(n_neighbors=50, weights='uniform')
    tic = timer()
    scores = cross_validate(knn, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation fl_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(knn, ds, 10, 60, 10)

    #  2）wine
    data_name = 'wine'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()
    ds.select_feature()

    knn = KNeighborsClassifier(n_neighbors=40, weights='distance')
    tic = timer()
    scores = cross_validate(knn, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation fl_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(knn, ds, 10, 60, 10)

    #  3）skin
    data_name = 'skin'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()

    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    tic = timer()
    scores = cross_validate(knn, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.6f} ± {:.6f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.6f} ± {:.6f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation fl_weighted: {:.6f} ± {:.6f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(knn, ds, 1000, 10001, 1000) # 0.999539 ± 0.000124

    #  4） balance-scale
    data_name = 'balance-scale'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()

    knn = KNeighborsClassifier(n_neighbors=25, weights='distance')
    tic = timer()
    scores = cross_validate(knn, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(knn, ds, 20, 80, 10) # 0.866

    #  5） flowmeter
    data_name = 'flowmeter'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.fill_nan()
    ds.standardization()
    ds.normalization()
    ds.select_feature()

    knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
    tic = timer()
    scores = cross_validate(knn, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(knn, ds, 1, 12, 1)

    #  6） adult
    data_name = 'adult'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)
    ds.encode_onehotencode() # KNN对数值不敏感可以直接编码

    ds.standardization(is_sparse=True)
    ds.normalization(is_sparse=True)
    ds.select_feature()

    knn = KNeighborsClassifier()
    tic = timer()
    scores = cross_validate(knn, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(knn, ds, 10000, 30001, 10000) # 0.829 ± 0.006

    #  7）student
    data_name = 'student'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()
    ds.select_feature()

    knn = KNeighborsClassifier(n_neighbors=15, weights='uniform')
    tic = timer()
    scores = cross_validate(knn, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(knn, ds, 10, 15, 1) # 0.861 ± 0.017

    # 8）sensor24
    data_name = 'sensor24'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()
    ds.select_feature()

    knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
    tic = timer()
    scores = cross_validate(knn, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(knn, ds, 1, 11, 1) # 0.961 ± 0.008

    # 9） har-PUC
    data_name = 'har-PUC'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.object_convert([8, 10, 11, 13, 14, 16, 17, 18, 19], 'int')
    ds.encode_onehotencode()
    ds.standardization(is_sparse=True)
    ds.normalization(is_sparse=True)

    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    tic = timer()
    scores = cross_validate(knn, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.6f} ± {:.6f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.6f} ± {:.6f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.6f} ± {:.6f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(knn, ds, 100, 1001, 100)

    '''
    # 10 - fold cross - validation precision_weighted: 0.991 ± 0.001
    # 10 - fold cross - validation recall_weighted: 0.991 ± 0.001
    # 10 - fold cross - validation f1_weighted: 0.991 ± 0.001
    # total time: 1409.073
    '''


    # 10） MNIST
    data_name = 'MNIST'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()
    # ds.select_feature()

    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    tic = timer()
    scores = cross_validate(knn, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.6f} ± {:.6f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.6f} ± {:.6f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.6f} ± {:.6f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(knn, ds, 100, 1001, 100)

    # 11) chronic
    data_name = 'chronic'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    # ds.encode_onehotencode()
    # ds.fill_nan()
    # ds.standardization()
    # ds.normalization()
    # ds.select_feature()
    # ds.save_feature()
    ds.load_feature('./feature/'+ ds.name + '.pkl')

    knn = KNeighborsClassifier(n_neighbors=2, weights='uniform')
    tic = timer()
    scores = cross_validate(knn, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.6f} ± {:.6f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.6f} ± {:.6f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.6f} ± {:.6f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(knn, ds, 1, 11, 1)

    '''
    # 10 - fold cross - validation precision_weighted: 0.976660 ± 0.037303
    # 10 - fold cross - validation recall_weighted: 0.980000 ± 0.031225
    # 10 - fold cross - validation f1_weighted: 0.977677 ± 0.035333
    # total time: 0.042
    '''
