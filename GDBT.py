from dataset import Dataset
from config import data_path
from EDA import MyPlot
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, cross_validate
import lightgbm as lgb
import numpy as np
from timeit import default_timer as timer
from config import data_msg
from sklearn.pipeline import Pipeline
from parm_select import parm_optimize
import warnings
import joblib
warnings.filterwarnings("ignore")

def parm(model, ds):
    max_depth = [25, 50, 75]
    learning_rate = [0.01, 0.05, 0.1]
    num_leaves = [300, 900, 1200]
    n_estimators = [100, 200]
    param_grid = dict(max_depth=max_depth, learning_rate=learning_rate, num_leaves=num_leaves, n_estimators=n_estimators)
    parm_optimize(model, param_grid, scoring[2], cv, ds, is_all=False)

if __name__ == '__main__':

    # config
    scoring = ['precision_weighted', 'recall_weighted', 'f1_weighted']
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    param = {
        'num_leaves': 150,
        'objective': 'multiclass', # binary
        'max_depth': 7,
        'learning_rate': 0.05,
        'max_bin': 200,
        'is_training_metric': True
    }
    param['metric'] = ['multi_error']

    # 1） iris
    data_name = 'iris'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()

    lc = lgb.LGBMClassifier()
    lc.set_params(**param)
    tic = timer()
    scores = cross_validate(lc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation fl_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # joblib.dump(lc, './model/iris.pkl')
    # parm(lc, ds)

    #  2）wine
    data_name = 'wine'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()
    ds.select_feature()

    lc = lgb.LGBMClassifier()
    lc.set_params(**param)
    tic = timer()
    scores = cross_validate(lc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation fl_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic)) # 0.961 ± 0.056

    # parm(lc, ds)

    #  3）skin
    data_name = 'skin'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()

    lc = lgb.LGBMClassifier()
    param['objective'] = ['binary']
    param['metric'] = ['binary_error']
    lc.set_params(**param)
    tic = timer()
    scores = cross_validate(lc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.6f} ± {:.6f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.6f} ± {:.6f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation fl_weighted: {:.6f} ± {:.6f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(lc, ds) # 0.998333 ± 0.000303

    #  4） balance-scale
    data_name = 'balance-scale'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()

    lc = lgb.LGBMClassifier()
    lc.set_params(**param)
    tic = timer()
    scores = cross_validate(lc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    #joblib.dump(lc, './model/balance-scale.pkl')
    # parm(lc, ds) # 0.857 ± 0.030

    #  5） flowmeter
    data_name = 'flowmeter'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()
    ds.select_feature()

    lc = lgb.LGBMClassifier()
    lc.set_params(**param)
    tic = timer()
    scores = cross_validate(lc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # joblib.dump(lc, './model/flowmeter.pkl')
    # parm(lc, ds) # 0.932 ± 0.050

    #  6） adult
    data_name = 'adult'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)
    ds.encode_onehotencode() # KNN对数值不敏感可以直接编码

    ds.standardization(is_sparse=True)
    ds.normalization(is_sparse=True)

    lc = lgb.LGBMClassifier()
    param['objective'] = ['binary']
    param['metric'] = ['binary_error']
    lc.set_params(**param)
    tic = timer()
    scores = cross_validate(lc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic)) # 0.864 ± 0.006

    # parm(lc, ds)

    #  7）student
    data_name = 'student'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()

    lc = lgb.LGBMClassifier()
    lc.set_params(**param)
    tic = timer()
    scores = cross_validate(lc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(lc, ds) # 0.861 ± 0.017

    # 8）sensor24
    data_name = 'sensor24'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.standardization()
    ds.normalization()

    lc = lgb.LGBMClassifier()
    lc.set_params(**param)
    tic = timer()
    scores = cross_validate(lc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.3f} ± {:.3f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.3f} ± {:.3f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.3f} ± {:.3f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(lc, ds) # 0.995 ± 0.003

    # 9） har-PUC
    data_name = 'har-PUC'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.object_convert([8, 10, 11, 13, 14, 16, 17, 18, 19], 'int')
    ds.encode_onehotencode()
    ds.standardization(is_sparse=True)
    ds.normalization(is_sparse=True)

    lc = lgb.LGBMClassifier()
    lc.set_params(**param)
    tic = timer()
    scores = cross_validate(lc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.6f} ± {:.6f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.6f} ± {:.6f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.6f} ± {:.6f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(lc, ds) # 0.961 ± 0.008
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

    lc = lgb.LGBMClassifier()
    lc.set_params(**param)
    tic = timer()
    scores = cross_validate(lc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.6f} ± {:.6f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.6f} ± {:.6f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.6f} ± {:.6f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(lc, ds) # 0.962272 ± 0.001846

    # 11) chronic
    data_name = 'chronic'
    ds = Dataset(path=data_path[data_name], lable_idx=data_msg[data_name])
    print(ds)

    ds.encode_onehotencode()
    ds.fill_nan()
    ds.standardization()
    ds.normalization()
    # ds.select_feature()

    lc = lgb.LGBMClassifier()
    lc.set_params(**param)
    tic = timer()
    scores = cross_validate(lc, ds.data, ds.lable, cv=cv, scoring=scoring)
    toc = timer()

    print('10-fold cross-validation precision_weighted: {:.6f} ± {:.6f}'.format(scores['test_precision_weighted'].mean(), scores['test_precision_weighted'].std()))
    print('10-fold cross-validation recall_weighted: {:.6f} ± {:.6f}'.format(scores['test_recall_weighted'].mean(), scores['test_recall_weighted'].std()))
    print('10-fold cross-validation f1_weighted: {:.6f} ± {:.6f}'.format(scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))
    print('total time: {:.3f}'.format(toc - tic))

    # parm(lc, ds) # 0.990066 ± 0.020631

    '''
    # 10-fold cross-validation precision_weighted: 0.987756 ± 0.025090
    # 10-fold cross-validation recall_weighted: 0.992500 ± 0.016008
    # 10-fold cross-validation f1_weighted: 0.990066 ± 0.020631
    # total time: 0.693
    '''