import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from scipy import sparse
import joblib
class Dataset:
    def __init__(self, path, sep=',', lable_idx=-1, is_fill=False):
        self.path = path
        df = pd.read_table(self.path, sep=sep, header=None)

        # get lable
        if df[df.columns[lable_idx]].dtypes == 'object':
            self.lable = LabelEncoder().fit_transform(df[df.columns[lable_idx]])
        else:
            self.lable =df[:][df.columns[lable_idx]]
        df[df.columns[lable_idx]] = self.lable
        # get data
        data_idx = list(df.columns)
        data_idx.remove(df.columns[lable_idx])
        self.data = df[data_idx]

        if is_fill:
            self.fill_nan()

        # self.X_y = df
        self.name = os.path.basename(self.path).split('.')[0]
        self.size = self.data.shape
        self.data.columns = list(range(self.size[1]))

    def describe(self):
        fea_lst = []
        for i in range(self.size[1]):
            fea_lst.append([self.data[:][i].dtypes, self.data[:][i].unique().size, self.data[:][i].describe()])
            print(fea_lst[-1])

    def normalization(self, is_sparse=False):
        self.data = StandardScaler(with_mean= not is_sparse).fit_transform(self.data)

    def standardization(self, is_sparse=False):
        if is_sparse:
            self.data = MaxAbsScaler().fit_transform(self.data)
        else:
            self.data = MinMaxScaler().fit_transform(self.data)

    def fill_nan(self):
        imp = IterativeImputer(RandomForestRegressor(), max_iter=10, random_state=0)
        self.data = pd.DataFrame(imp.fit_transform(self.data))

    def select_feature(self):
        self.data = SelectFromModel(GradientBoostingClassifier()).fit_transform(self.data, self.lable)

    def encode_lableencode(self):
        for i in self.data.columns:
            if self.data[i].dtypes == 'object':
                not_Nan_lst = np.where(self.data[i].notnull())
                if len(not_Nan_lst[0]) == self.size[0]:
                    self.data[i] = LabelEncoder().fit_transform(self.data[i])
                else:
                    self.data[i].iloc[not_Nan_lst] = LabelEncoder().fit_transform(self.data[i].iloc[not_Nan_lst])


    def encode_onehotencode(self, is_sparse=False):
        ls = []
        for i in self.data.columns:
            if self.data[i].dtypes == 'object':
                ls.append(i)
        self.data = pd.get_dummies(self.data, prefix=ls, drop_first=True)
        if is_sparse:
            self.data = sparse.csr_matrix(self.data)
        self.size = self.data.shape

    def object_convert(self,ls,type):
        self.data[ls] = self.data[ls].astype(type)

    def save_feature(self):
        joblib.dump((self.data, self.lable), './feature/'+ self.name + '.pkl')
        print('save finish')

    def load_feature(self, path):
        self.data, self.lable = joblib.load(path)
        self.data = pd.DataFrame(self.data)
        print('load finish')

    def __str__(self):
        return 'name: {} size: {} lable: {}'.format(self.name, self.size, len(set(self.lable)))
