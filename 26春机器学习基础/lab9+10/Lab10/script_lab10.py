"""
Introduction to Machine Learning

Lab 10: Primal SVM and Suppress Unfairness

TODO: Add your information here.
    IMPORTANT: Please ensure this script
    (1) Run script_lab10.py on Python >=3.6;
    (2) No errors;
    (3) Finish in tolerable time on a single CPU (e.g., <=10 mins);
Student name(s): 黄浩博
Student ID(s): 2024201630
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
# don't add any other packages


def adult_income_data_loader() -> Dict[str, List[np.ndarray]]:
    df = pd.read_csv("adult.csv")
    df.drop(df.index[df['workclass'] == '?'], inplace=True)
    df.drop(df.index[df['occupation'] == '?'], inplace=True)
    df.drop(df.index[df['native-country'] == '?'], inplace=True)
    df.dropna(how='any', inplace=True)
    df = df.drop_duplicates()
    df.drop(['education'], axis=1, inplace=True)
    df['net_capital'] = (df['capital-gain'] - df['capital-loss']).astype(int)
    df.drop(['capital-gain', 'capital-loss'], axis=1, inplace=True)

    # 使用 np.where 彻底避免 Pandas replace 的警告
    df['income'] = np.where(df['income'].astype(str).str.contains('>50K'), 1, -1)
    df['gender'] = np.where(df['gender'].astype(str).str.contains('Female'), 0, 1)

    b = df.iloc[:, [0, 2, 3, 9, 12]]
    ys = df['income'].to_numpy()
    ys = ys.reshape(ys.shape[0], 1)
    genders = df['gender'].to_numpy()
    names = b.columns
    xs = pd.DataFrame(b, columns=names).to_numpy()
    xs = np.float64(xs)
    # normalize features
    xs /= np.max(xs, axis=0, keepdims=True)
    idx = np.random.RandomState(42).permutation(xs.shape[0])
    data = {'train': [xs[idx[:10000], :], ys[idx[:10000], :], genders[idx[:10000]]],
            'test': [xs[idx[10000:20000], :], ys[idx[10000:20000], :], genders[idx[10000:20000]]]}
    return data


def hinge_loss_with_grad(z: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The hinge loss L = max(0, 1-yz) and its gradient dL/dz
    """
    margin = 1.0 - y * z
    loss = np.maximum(0, margin)
    grad = np.zeros_like(z)
    grad[margin > 0] = -y[margin > 0]
    return loss, grad


def linear_model_with_grad(xs: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The linear model: y = x^T w and its gradient dy/dw
    """
    y_pred = xs @ weights
    grad = xs
    return y_pred, grad


def sgd_primal_svm(xs: np.ndarray, ys: np.ndarray, batch_size: int = 100,
                   epochs: int = 50, lr: float = 1e-1) -> np.ndarray:
    """
    Training a Logistic regression model based on stochastic gradient descent
    """
    num, dim = xs.shape
    xs = np.concatenate((xs, -np.ones((xs.shape[0], 1))), axis=1)
    # 采用0初始化增强SVM起步的稳定性
    weights = np.zeros((dim + 1, 1))

    idx = np.arange(num)
    for epoch in range(epochs):
        np.random.shuffle(idx)
        for i in range(0, num, batch_size):
            batch_idx = idx[i:i+batch_size]
            batch_xs = xs[batch_idx]
            batch_ys = ys[batch_idx]

            z, dz_dw = linear_model_with_grad(batch_xs, weights)
            _, dl_dz = hinge_loss_with_grad(z, batch_ys)

            dl_dw = dz_dw.T @ dl_dz / batch_xs.shape[0]
            weights -= lr * (dl_dw + 0.001 * weights)

    return weights


def test_svm(xs: np.ndarray, ys: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get prediction accuracy of the logistic regression model
    """
    xs_aug = np.concatenate((xs, -np.ones((xs.shape[0], 1))), axis=1)
    preds = np.sign(xs_aug @ weights)
    preds[preds == 0] = 1
    acc = np.mean(preds == ys)
    return acc, preds


def gender_fairness_check(preds: np.ndarray, genders: np.ndarray) -> Tuple[float, float]:
    """
    Find a way to check whether your classification results are fair with respect to gender or not
    """
    p1 = preds[genders == 1]
    p0 = preds[genders == 0]
    return np.sum(p1 == 1) / p1.shape[0], np.sum(p0 == 1) / p0.shape[0]


def data_augment(data: List) -> List:
    """
    Find a way to augment data for training a model with better fairness on gender
    """
    xs, ys, genders = data[0], data[1], data[2]

    idx_f_pos = np.where((genders == 0) & (ys[:, 0] == 1))[0]
    idx_m_pos = np.where((genders == 1) & (ys[:, 0] == 1))[0]
    idx_f_neg = np.where((genders == 0) & (ys[:, 0] == -1))[0]
    idx_m_neg = np.where((genders == 1) & (ys[:, 0] == -1))[0]

    # 提取4个子群体的最大样本数
    max_size = max(len(idx_f_pos), len(idx_m_pos), len(idx_f_neg), len(idx_m_neg))

    def resample(idx, size):
        return np.random.choice(idx, size, replace=True)

    # 对所有子群体进行上采样至最大规模，彻底平衡数据，防止坍缩
    final_idx = np.concatenate([
        resample(idx_f_pos, max_size),
        resample(idx_m_pos, max_size),
        resample(idx_f_neg, max_size),
        resample(idx_m_neg, max_size)
    ])

    np.random.shuffle(final_idx)

    return [xs[final_idx], ys[final_idx], genders[final_idx]]


if __name__ == '__main__':
    data = adult_income_data_loader()
    weights1 = sgd_primal_svm(xs=data['train'][0], ys=data['train'][1])
    accuracy1, preds1 = test_svm(xs=data['test'][0], ys=data['test'][1], weights=weights1)
    p1, p0 = gender_fairness_check(preds1[:, 0], genders=data['test'][2])
    print('SVM: p(high income | male)={:.4f}, p(high income | female)={:.4f}'.format(p1, p0))
    print('SVM: Acc={:.4f}'.format(accuracy1))

    data_new = data_augment(data['train'])
    weights2 = sgd_primal_svm(xs=data_new[0], ys=data_new[1])
    accuracy2, preds2 = test_svm(xs=data['test'][0], ys=data['test'][1], weights=weights2)
    q1, q0 = gender_fairness_check(preds2[:, 0], genders=data['test'][2])
    print('After DA, SVM: p(high income | male)={:.4f}, p(high income | female)={:.4f}'.format(q1, q0))
    print('After DA, SVM: Acc={:.4f}'.format(accuracy2))