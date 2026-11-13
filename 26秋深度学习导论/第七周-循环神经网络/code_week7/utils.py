import torch 
import string
import numpy as np
import datetime

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

def load_data(file_path):

    # Read the data and separate the name and nationality
    languages = []
    data = []
    X = []
    y = []

    with open(file_path, 'r') as f:
        text = f.readlines()
        for line in text[1:]:  
            line = line.split(",")
            name = line[0].strip()
            lang = line[1].strip()
            if not lang in languages:
                languages.append(lang)
            X.append(name)
            y.append(lang)
            data.append((name, lang))
    return X, y, data, languages

# Define a function to convert the name into one-hot vector 
def name_vector(name, all_letters):
    # Build the Vocabulary
    # get all the letters
    # all_letters = string.ascii_letters + ".,;"
    vocab_size = len(all_letters)
    rep = torch.zeros(len(name), 1, vocab_size)     # 初始化字符的标识向量，其维度由给定的词袋大小
    for index, letter in enumerate(name):
        pos = all_letters.find(letter)
        rep[index][0][pos] = 1
    return rep


def nationality_vector(lang, languages):
    """
    repersent the nationality in vectors 
    It will return the index for crossponding nationality in languages array 
    """
    # todo
    # 参考 name_vector()是如何构建one-hot类型的字符张量的，把nationality的字符张量构建出来
    # 返回 shape=[1] 的类别索引
    return torch.tensor([languages.index(lang)], dtype=torch.long)


def dataloader(all_letters, languages, batch_size, X, y):
    """
    将输入的字符型数据转换成vector
    返回的to_ret包含转换后的向量
    name: str
    lang: str
    name_vector(name, all_letters).shape = [len(name),1,len(all_letters)]
    nationality_vector(lang, languages).shape = [1]
    """
    to_ret = []
    for i in range(batch_size): 
        index = np.random.randint(len(X))
        name, lang = X[index], y[index] 
        to_ret.append((name, lang, name_vector(name, all_letters), nationality_vector(lang, languages)))

    return to_ret