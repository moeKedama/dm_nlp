# TODO:
#   1.get data and transform to csv
#   2.from data to dict
#   3.process data

import numpy as np
import pandas as pd
import jieba
import os
import random
import json


def txt_to_csv(data_list, csv_path):
    txt = np.array(data_list)
    pd_data = pd.DataFrame(txt, columns=['target', 'text'])
    pd_data.to_csv(csv_path)
    print(csv_path + ' done')


def analyze_data(class_dirs, label):
    original_microblog = "Chinese_Rumor_Dataset-master/CED_Dataset/original-microblog/"
    data_list = []
    num = 0
    for class_dir in class_dirs:
        if class_dir != '.DS_Store':
            with open(original_microblog + class_dir, 'r', encoding='UTF-8') as f:
                content = f.read()
            dict = json.loads(content)
            data_list.append([label, dict["text"]])
            num += 1
    return data_list, num


# 生成数据字典
def create_dict(data_path, dict_path):
    '''
    :param data_path: path of data to read
    :param dict_path: path of dict to write
    :return:          words, list of words, repeatable
    '''
    dict_set = set()
    dataframe = pd.read_csv(data_path)
    words = []
    for indexs in dataframe.index:
        content = jieba.lcut(dataframe.loc[indexs].values[-1])
        for s in content:
            dict_set.add(s)
            words.append(s)
    # 把元组转换成字典，一个字对应一个数字
    dict_list = []
    i = 0
    for s in dict_set:
        dict_list.append([s, i])
        i += 1
    # 添加未知字符
    dict_txt = dict(dict_list)
    end_dict = {"<unk>": i}
    dict_txt.update(end_dict)
    # 把这些字典保存到本地中
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(str(dict_txt))
    print(dict_path + ' done')
    return words


def data_process(data_list, stopwords, csv_path):
    nd_txt = np.array(data_list)
    pd_data = pd.DataFrame(nd_txt, columns=['target', 'text'])

    target = pd_data['target'].values.tolist()
    text = pd_data['text']
    sentences = []
    text_list = []

    for txt in text:
        segs = jieba.lcut(txt)
        segs = list(filter(lambda x: len(x) > 1, segs))  # 没有解析出来的新闻过滤掉
        segs = list(filter(lambda x: x not in stopwords, segs))  # 把停用词过滤掉
        text_list.append(segs)
        sentences.append(" ".join(segs))

    data = [[0 for j in range(2)] for i in range(len(target))]
    for i in range(len(target)):
        data[i][0] = target[i]
        data[i][1] = sentences[i]

    nd_txt = np.array(data)
    pd_data = pd.DataFrame(nd_txt, columns=['target', 'text'])
    pd_data.to_csv(csv_path)
    print(csv_path + ' done')
    return pd_data


if __name__ == '__main__':
    # 分别为谣言数据、非谣言数据、全部数据的文件路径
    rumor_class_dirs = os.listdir("Chinese_Rumor_Dataset-master/CED_Dataset/rumor-repost/")
    non_rumor_class_dirs = os.listdir("Chinese_Rumor_Dataset-master/CED_Dataset/non-rumor-repost/")
    original_microblog = "Chinese_Rumor_Dataset-master/CED_Dataset/original-microblog/"

    # 谣言标签为0，非谣言标签为1
    rumor_label = "1"
    non_rumor_label = "0"

    all_rumor_list, rumor_num = analyze_data(rumor_class_dirs, rumor_label)
    all_non_rumor_list, non_rumor_num = analyze_data(non_rumor_class_dirs, non_rumor_label)

    print("谣言数据总量为：" + str(rumor_num))
    print("非谣言数据总量为：" + str(non_rumor_num))

    all_data_list = all_rumor_list + all_non_rumor_list
    random.shuffle(all_data_list)

    data_lists = [all_data_list, all_rumor_list, all_non_rumor_list]

    # 全部数据进行乱序后写入all_data.csv, 正例写入rumor_data.csv 负例写入non_rumor_data.csv
    all_data_path = "data/all_data.csv"
    rumor_data_path = "data/rumor_data.csv"
    non_rumor_data_path = "data/non_rumor_data.csv"
    data_paths = [all_data_path, rumor_data_path, non_rumor_data_path]

    # dict_paths为数据字典存放路径
    dict_all_path = "data/dict.txt"
    dict_rumor_path = "data/dict_rumor.txt"
    dict_non_rumor_path = "data/dict_non_rumor.txt"
    dict_paths = [dict_all_path, dict_rumor_path, dict_non_rumor_path]

    # processed_all_data_paths为处理后文件存放位置
    processed_all_data_path = 'data/processed_all_data.csv'
    processed_rumor_data_path = "data/processed_rumor_data.csv"
    processed_non_rumor_data_path = "data/processed_non_rumor_data.csv"
    processed_all_data_paths = [processed_all_data_path, processed_rumor_data_path, processed_non_rumor_data_path]

    stopwords = pd.read_csv("stopwords-master/baidu_stopwords.txt", index_col=False, quoting=3,
                            sep="\t", names=['stopword'], encoding='utf-8')
    stopwords = stopwords['stopword'].values

    # 数据写入csv & 字典写入txt
    for i in range(3):
        txt_to_csv(data_lists[i], data_paths[i])
        create_dict(data_paths[i], dict_paths[i])
