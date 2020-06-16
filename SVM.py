from dataset import Dataset
from config import data_path
from EDA import MyPlot
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import svm
import numpy as np
from timeit import default_timer as timer
from config import data_msg
from sklearn.pipeline import Pipeline
from parm_select import parm_optimize
import warnings
warnings.filterwarnings("ignore")

def parm(model, ds):
    C = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    gamma = [0.001, 0.0001]
    param_grid = dict(C=C, gamma=gamma)
    parm_optimize(model, param_grid, scoring[2], cv, ds, is_all=False)

if __name__ == '__main__':
    # config
    scoring = ['precision_weighted', 'recall_weighted', 'f1_weighted']
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    # TODO 注意svm这种数值模型往往不需要特征选择
    # 1） iris
    data_name = 'iris'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()

    svc = svm.SVC(kernel='rbf', probability=True, C=100, gamma=0.001)
    tic = timer()
    scores = cross_validate(svc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation fl_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(svc, ds)

    #  2）wine
    data_name = 'wine'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()

    svc = svm.SVC(kernel='rbf', probability=True, C=10, gamma=0.001)
    tic = timer()
    scores = cross_validate(svc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation fl_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(svc, ds) # 0.983 ± 0.027

    #  3）skin
    data_name = 'skin'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()

    svc = svm.SVC(kernel='rbf', probability=True)
    tic = timer()
    scores = cross_validate(svc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.6f} ± {:.6f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.6f} ± {:.6f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation fl_weighted: {:.6f} ± {:.6f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(svc, ds) # 0.998337 ± 0.000206

    #  4） balance-scale
    data_name = 'balance-scale'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()

    svc = svm.SVC(kernel='rbf', probability=True, C=1000, gamma=0.001)
    tic = timer()
    scores = cross_validate(svc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(svc, ds)

    #  5） flowmeter
    data_name = 'flowmeter'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.fill_nan()
    ds.standardization()
    ds.normalization()
    ds.select_feature()

    svc = svm.SVC(kernel='rbf', probability=True, C=1000, gamma=0.001)
    tic = timer()
    scores = cross_validate(svc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(svc, ds)

    #  6） adult
    data_name = 'adult'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)
    ds.encode_onehotencode() # KNN对数值不敏感可以直接编码

    ds.standardization(is_sparse=True)
    ds.normalization(is_sparse=True)

    svc = svm.SVC(kernel='rbf', probability=True, C=100, gamma=0.001)
    tic = timer()
    scores = cross_validate(svc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(svc, ds) # 0.843 ± 0.005

    #  7）student
    data_name = 'student'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()
    ds.select_feature()

    svc = svm.SVC(kernel='rbf', probability=True, C=1000, gamma=0.001)
    tic = timer()
    scores = cross_validate(svc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(svc, ds) # 0.862 ± 0.018   20.156

    # 8）sensor24
    data_name = 'sensor24'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()

    svc = svm.SVC(kernel='rbf', probability=True, C=1000, gamma=0.001)
    tic = timer()
    scores = cross_validate(svc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(svc, ds) # 0.903

    # 9） har-PUC
    data_name = 'har-PUC'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.object_convert([8, 10, 11, 13, 14, 16, 17, 18, 19], 'int')
    ds.encode_onehotencode()
    ds.standardization(is_sparse=True)
    ds.normalization(is_sparse=True)
    ds.select_feature()

    svc = svm.SVC(kernel='rbf', probability=True, C=1, gamma='scale')
    tic = timer()
    scores = cross_validate(svc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.6f} ± {:.6f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.6f} ± {:.6f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.6f} ± {:.6f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(svc, ds) # 0.911796 ± 0.002185


    # 10） MNIST
    # TODO : 没有特征选择再跑一次
    data_name = 'MNIST'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()
    ds.select_feature()

    svc = svm.SVC(kernel='rbf', probability=True, C=1, gamma='scale')
    tic = timer()
    scores = cross_validate(svc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.6f} ± {:.6f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.6f} ± {:.6f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.6f} ± {:.6f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(svc, ds)
    '''
    # 10 - fold cross - validation precision_weighted: 0.973815 ± 0.001836
    # 10 - fold cross - validation recall_weighted: 0.973771 ± 0.001800
    # 10 - fold cross - validation f1_weighted: 0.973756 ± 0.001810
    # total time: 7326.255
    '''

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

    svc = svm.SVC(kernel='rbf', probability=True, C=1, gamma='scale')
    tic = timer()
    scores = cross_validate(svc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.6f} ± {:.6f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.6f} ± {:.6f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.6f} ± {:.6f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(svc, ds)  # 0.972756 ± 0.038565
    '''
    # 10 - fold cross - validation precision_weighted: 0.972473 ± 0.038187
    # 10 - fold cross - validation recall_weighted: 0.975000 ± 0.035355
    # 10 - fold cross - validation f1_weighted: 0.972756 ± 0.038565
    # total time: 0.072
    '''