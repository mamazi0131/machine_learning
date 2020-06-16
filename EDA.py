from dataset import Dataset
from config import data_path, data_msg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import joblib
import math

# 1D/2D/3D normal title font
font_title_1 = {'family': 'DejaVu Sans',
              'weight': 'normal',
              'size': 25,
              }

# 1D multiple title font
font_title_2 = {'family': 'DejaVu Sans',
              'weight': 'normal',
              'size': 10,
              }

# 1D/2D/3D axis font
font_axis = {'family': 'DejaVu Sans',
             'weight': 'normal',
             'size': 13,
             }

class MyPlot:

    def distributin_1D(self, name, x, title, is_save=False, save_path='./fig/'):
        sns.set_style('white')
        num_bins = 20
        plt.figure(figsize=(8, 6))
        sns.distplot(x, hist=True, kde=True, rug=False, bins=num_bins,
                     kde_kws={"color": "y", "lw": 1.5, 'linestyle': '--'},  # 设置密度曲线颜色，线宽，标注、线形，#控制是否显示核密度估计图
                     hist_kws={"color": "b"},
                     label='male_height')
        plt.grid(linestyle='--')
        plt.title('Attribute '+ name, font_title_1, pad=10)
        plt.xlabel('')
        plt.ylabel('')
        if is_save:
            plt.savefig(save_path + title + "_1D.png")
        plt.show()


    def mul_distribution_1D(self, x, title, is_save=False, save_path='./fig/'):
        sns.set_style('white')
        num_bins = 20
        n = math.ceil(np.sqrt(x.shape[1]))
        fig, axes = plt.subplots(n, n)
        fig.set_size_inches(12, 9)
        fig.subplots_adjust(wspace=0.35, hspace=0.35)
        ax = axes.flatten()
        for i in range(x.shape[1]):
            fig_ = sns.distplot(np.array(x)[:, i], hist=True, kde=True, rug=False, bins=num_bins,
                         kde_kws={"color": "y", "lw": 1.5, 'linestyle': '--'},  # 设置密度曲线颜色，线宽，标注、线形，#控制是否显示核密度估计图
                         hist_kws={"color": "b"},
                         label='male_height', ax=ax[i])
            fig_.grid(linestyle='--')
            fig_.set_title('Attribute ' + str(i), font_title_2, pad=10)
            fig_.set_xlabel('')
            fig_.set_ylabel('')
        if is_save:
            plt.savefig(save_path + title + "_describe_1D.png")
        plt.show()

    def distribution_2D(self, x, y, title, is_save=False, save_path='./fig/'):
        tsne = manifold.TSNE(n_components=2, perplexity=10.0,init='pca', random_state=501)
        X_tsne = tsne.fit_transform(x)
        X_norm = MinMaxScaler().fit_transform(X_tsne)

        joblib.dump((x, y), './feature/' + title + '.pkl')


        plt.figure(figsize=(8, 8))
        plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y, cmap=plt.cm.get_cmap('coolwarm'))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('t-SNE X1', font_axis)
        plt.ylabel('t-SNE X2', font_axis)
        plt.title(title, font_title_1, pad=10)
        if is_save:
            plt.savefig(save_path + title + "_2D.png")
        plt.show()

    def distribution_3D(self, x, y, title, is_save=False, save_path='./fig/'):
        tsne = manifold.TSNE(n_components=3, perplexity=10.0,init='pca', random_state=501)
        X_tsne = tsne.fit_transform(x)
        X_norm = MinMaxScaler().fit_transform(X_tsne)
        # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_norm[:, 0], X_norm[:, 1], X_norm[:, 2], c=y, cmap=plt.cm.get_cmap('coolwarm'))
        ax.set_xlabel('t-SNE X1', font_axis, labelpad=10)
        ax.set_ylabel('t-SNE X2', font_axis, labelpad=10)
        ax.set_zlabel('t-SNE X3', font_axis, labelpad=10)
        plt.title(title, font_title_1, pad=10)
        if is_save:
            plt.savefig(save_path + title + "_3D.png")
        plt.show()

    def pca_2D(self, x, y, title, is_save=False, save_path='./fig/'):
        pca = PCA(n_components=2, copy=False, whiten=False)
        X_pca = pca.fit_transform(x)
        X_norm = MinMaxScaler().fit_transform(X_pca)
        plt.figure(figsize=(8, 8))
        plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y, cmap=plt.cm.get_cmap('coolwarm'))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('t-SNE X1', font_axis)
        plt.ylabel('t-SNE X2', font_axis)
        plt.title(title, font_title_1, pad=10)
        if is_save:
            plt.savefig(save_path + title + "_pca_2D.png")
        plt.show()



if __name__ == '__main__':
    for i, name in enumerate(data_path.keys()):
        ds = Dataset(path=data_path[name], lable_idx=data_msg[name], is_fill=False)
        print(ds)
        # ds.object_convert([8, 10, 11, 13, 14, 16, 17, 18, 19], 'int')
        # ds.encode_lableencode()
        # ds.describe()
        mp = MyPlot()
        mp.mul_distribution_1D(ds.data, name, True, save_path='./fig/')
        mp.distribution_2D(ds.data, ds.lable, ds.name, True)
        mp.pca_2D(ds.data, ds.lable, ds.name, True)



