# TODO:
#   1.word cloud
#   2.scatter plot

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD, PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import jieba
import matplotlib
from wordcloud import WordCloud
import re


def vectorize(data):
    count_vector = CountVectorizer()
    emb = count_vector.fit_transform(data)
    return emb, count_vector


def plot_lsa(test_data, test_labels, size_change=False):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    colors = ['orange', 'blue']
    plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=test_labels,
                cmap=matplotlib.colors.ListedColormap(colors))
    orange_patch = mpatches.Patch(color='orange', label='Not')
    blue_patch = mpatches.Patch(color='blue', label='Real')
    plt.legend(handles=[orange_patch, blue_patch], prop={'size': 30})

    if size_change:
        plt.xlim(0, 0.3)
        plt.ylim(-0.1, 0.2)
    else:
        plt.xlim(0, 5)
        plt.ylim(-5, 5)


def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    train = tfidf_vectorizer.fit_transform(data)
    return train, tfidf_vectorizer


def generate_png(X, y, saved_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # visualize with CountVectorizer
    X_train_counts, count_vector = vectorize(X_train)
    fig = plt.figure(figsize=(16, 16))
    plot_lsa(X_train_counts, y_train)
    plt.savefig(saved_path[0])

    # visualize with tf_idf
    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
    fig = plt.figure(figsize=(16, 16))
    plot_lsa(X_train_tfidf, y_train)
    # tfidf初始图片尺度调整
    if saved_path[1] == 'pics/tfidf.png':
        print('1')
        plot_lsa(X_train_counts, y_train, size_change=True)
    plt.savefig(saved_path[1])


def generate_png_pca(X, y, saved_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pca = PCA(n_components='mle', copy=True)

    # visualize with CountVectorizer
    X_train_counts, count_vector = vectorize(X_train)
    pca_X_train_counts = pca.fit_transform(X_train_counts)

    plot_lsa(pca_X_train_counts, y_train)
    plt.savefig(saved_path[0])

    # visualize with tf_idf
    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
    pca_X_train_tfidf = pca.fit_transform(X_train_tfidf)

    plot_lsa(pca_X_train_tfidf, y_train)
    plt.savefig(saved_path[1])


def generate_png_svd(X, y, saved_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svd = TruncatedSVD(n_components=200)
    fig = plt.figure(figsize=(16, 16))

    # visualize with CountVectorizer
    X_train_counts, count_vector = vectorize(X_train)
    pca_X_train_counts = svd.fit_transform(X_train_counts)

    plot_lsa(pca_X_train_counts, y_train)
    plt.savefig(saved_path[0])

    # visualize with tf_idf
    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
    pca_X_train_tfidf = svd.fit_transform(X_train_tfidf)

    plot_lsa(pca_X_train_tfidf, y_train)
    plt.savefig(saved_path[1])


def generate_word_list(text):
    words = []
    for sentence in text:
        content = jieba.lcut(sentence)
        for s in content:
            words.append(s)
    return words


def words_cloud(data_path, pic_path):
    data = pd.read_csv(data_path)
    data = data.dropna()
    X = data['text']
    words = generate_word_list(X)

    tx = ''.join(words)
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 16))
    wordcl = WordCloud(font_path="C:\\Windows\\Fonts\\STFANGSO.ttf",
                       width=1000,
                       height=1000,
                       background_color='white',
                       # min_font_size=10,
                       mode='RGBA'
                       # mask=np.array(Image.open("testpic.png"))
                       ).generate(tx)
    wordcl.to_file(pic_path)
    plt.imshow(wordcl, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most used words for {}".format(re.split('[/.]', pic_path)[1]), fontsize=24)
    plt.show()
    print("词云生成完成！")


if __name__ == '__main__':

    all_data_path = "data/processed_all_data.csv"
    dataset = pd.read_csv(all_data_path)
    data = dataset.dropna()

    X = data['text']
    y = data['target']

    png_paths = ['pics/vectorization.png', 'pics/tfidf.png']
    pca_png_paths = ['pics/vectorization_pca.png', 'pics/tfidf_pca.png']
    svd_png_paths = ['pics/vectorization_svd.png', 'pics/tfidf_svd.png']
    # generate_png(X, y, png_paths)
    # generate_png_svd(X, y, svd_png_paths)

    # pca doesn't support sparse input
    # generate_png_pca(X, y, pca_png_paths)

    processed_all_data_path = 'data/all_data.csv'
    processed_rumor_data_path = "data/rumor_data.csv"
    processed_non_rumor_data_path = "data/non_rumor_data.csv"
    processed_all_data_paths = [processed_all_data_path, processed_rumor_data_path, processed_non_rumor_data_path]

    all_data_wc_path = 'pics/wordcloud_all.png'
    rumor_data_wc_path = 'pics/wordcloud_rumor.png'
    non_rumor_data_wc_path = 'pics/wordcloud_non_rumor.png'
    word_cloud_paths = [all_data_wc_path, rumor_data_wc_path, non_rumor_data_wc_path]

    for i in range(3):
        words_cloud(processed_all_data_paths[i], word_cloud_paths[i])
