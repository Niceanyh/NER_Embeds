#--coding:utf-8--
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from tsne import bh_sne

class CostumVisualizer:
    def __init__(self):
        pass
    @staticmethod
    def visualize_pca(embeddings_dict):
        #plt.rcParams['font.family'] = 'SimHei' # 用来正常显示中文标签
        font = FontProperties(fname=r"./font/SimHei.ttf", size=8)  # 请根据你的系统路径修改
        pca = PCA(n_components=2, random_state=0)
        
        words = list(embeddings_dict.keys())
        vectors = list(embeddings_dict.values())

        # 使用 PCA 进行降维
        Y = pca.fit_transform(vectors)

        plt.scatter(Y[:, 0], Y[:, 1], alpha=0.5)  # 添加透明度，以便更好地观察重叠点

        for i, label in enumerate(words):
            plt.annotate(label, xy=(Y[i, 0], Y[i, 1]), xytext=(0, 0),textcoords="offset points",fontsize=8,fontproperties=font)

        plt.title('PCA Visualization of Word Embeddings',fontproperties=font)
        plt.xlabel('PCA Dimension 1',fontproperties=font)
        plt.ylabel('PCA Dimension 2',fontproperties=font)

        plt.show()
    
    @staticmethod
    def visualize_tsne(embeddings_dict):
        # Font settings
        font = FontProperties(fname=r"./font/SimHei.ttf", size=8)  # Modify the path based on your system

        # t-SNE dimensionality reduction
        tsne = TSNE(n_components=2,init='pca',random_state=0)
        print("init success")
        
        words = list(embeddings_dict.keys())
        vectors = list(embeddings_dict.values())

        # Use t-SNE for dimensionality reduction
        Y = tsne.fit_transform(vectors)

        # Scatter plot
        plt.scatter(Y[:, 0], Y[:, 1], alpha=0.5)

        # Annotate points with words
        for i, label in enumerate(words):
            plt.annotate(label, xy=(Y[i, 0], Y[i, 1]), xytext=(0, 0), textcoords="offset points", fontsize=8, fontproperties=font)

        # Plot settings
        plt.title('t-SNE Visualization of Word Embeddings', fontproperties=font)
        plt.xlabel('t-SNE Dimension 1', fontproperties=font)
        plt.ylabel('t-SNE Dimension 2', fontproperties=font)

        # Show the plot
        plt.show()

    @staticmethod
    def visualize_tsne1(embeddings_dict,perplexity=1):
        # Font settings
        font = FontProperties(fname=r"./font/SimHei.ttf", size=8)  # Modify the path based on your system

        words = list(embeddings_dict.keys())
        vectors = np.array(list(embeddings_dict.values()),dtype=np.float64)

        # t-SNE dimensionality reduction
        Y = bh_sne(vectors,perplexity=perplexity)

        # Scatter plot
        plt.scatter(Y[:, 0], Y[:, 1], alpha=0.5)

        # Annotate points with words
        for i, label in enumerate(words):
            plt.annotate(label, xy=(Y[i, 0], Y[i, 1]), xytext=(0, 0), textcoords="offset points", fontsize=8, fontproperties=font)

        # Plot settings
        plt.title('t-SNE Visualization of Word Embeddings', fontproperties=font)
        plt.xlabel('t-SNE Dimension 1', fontproperties=font)
        plt.ylabel('t-SNE Dimension 2', fontproperties=font)

        # Show the plot
        plt.show()

# Example usage:
# embeddings_dict = {word1: vector1, word2: vector2, ...}
# YourClassName.visualize_tsne(embeddings_dict)
