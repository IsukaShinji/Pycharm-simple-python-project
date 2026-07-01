"""
Introduction to Machine Learning

Lab 9: LDA and Logistic Regression

TODO: Add your information here.
    IMPORTANT: Please ensure this script
    (1) Run script_lab9.py on Python >=3.6;
    (2) No errors;
    (3) Finish in tolerable time on a single CPU (e.g., <=10 mins);
Student name(s):黄浩博
Student ID(s):2024201630
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
    # changing class from >50K and <=50K to 1 and 0
    df['income'] = df['income'].astype(str)
    df['income'] = df['income'].replace('>50K', 1)
    df['income'] = df['income'].replace('<=50K', 0)
    # changing class from Male and Female to 1 and 0
    df['gender'] = df['gender'].astype(str)
    df['gender'] = df['gender'].replace('Male', 1)
    df['gender'] = df['gender'].replace('Female', 0)
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


def linear_discriminant_analysis_2class(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Learning a LDA model for two classes: learning w and c for checking x^T w > c or not
    :param xs: training data with size (N, D)
    :param ys: training labels with size (N, 1), whose element is 0 or 1

    :return:
        the weights "w" of LDA with size (D, 1),
        the criterion "c"
    """
    # TODO: Change the code below and implement LDA
    xs0 = xs[ys[:, 0] == 0]
    xs1 = xs[ys[:, 0] == 1]

    mu0 = np.mean(xs0, axis=0, keepdims=True).T
    mu1 = np.mean(xs1, axis=0, keepdims=True).T

    sw = (xs0.T - mu0) @ (xs0.T - mu0).T + (xs1.T - mu1) @ (xs1.T - mu1).T

    w = np.linalg.pinv(sw) @ (mu1 - mu0)
    c = float(0.5 * w.T @ (mu0 + mu1))

    return w, c


def test_lda(xs: np.ndarray, ys: np.ndarray, w: np.ndarray, c: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Testing the LDA model and output prediction results and accuracy
    :param xs: testing data with size (N, D)
    :param ys: the ground truth labels with size (N, 1)
    :param w: the model parameters with size (D, 1)
    :param c: the threshold to make classification x^Tw > c => 1, otherwise => 0
    :return:
        prediction accuracy in the range in [0, 1]
        prediction results with size (N, 1)
    """
    # TODO: Implement the testing script of LDA, output prediction accuracy and prediction results
    preds = (xs @ w > c).astype(int)
    accuracy = np.mean(preds == ys)
    return accuracy, preds


def sigmoid_function_with_grad(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The sigmoid function y = 1 / (1 + exp(-x)) and its gradient
    :param x: an array with arbitrary size
    :return:
        the output of the sigmoid function
        the gradient dy/dx
    """
    # TODO: Change the code below, implement the sigmoid function and calculate its gradient
    y = 1 / (1 + np.exp(-x))
    grad = y * (1 - y)
    return y, grad


def binary_cross_entropy_with_grad(ps: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The BCE loss:
        L = -1/N * sum_n yn * log pn + (1-yn) * log (1-pn)
    And its gradient dL/dp
    :param ps: the probabilities of labels
    :param ys: the binary labels
    :return:
        the value of loss function
        th gradient dL/dp, whose size is the same with ps
    """
    # TODO: implement the cross-entropy loss and the gradient with respect to ps
    eps = 1e-15
    N = ps.shape[0]
    ps = np.clip(ps, eps, 1 - eps)

    loss = -1/N * np.sum(ys * np.log(ps) + (1 - ys) * np.log(1 - ps))
    grad = -1/N * (ys / ps - (1 - ys) / (1 - ps))

    return np.array([loss]), grad


def linear_model_with_grad(xs: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The linear model: y = x^T w and its gradient dy/dw
    :param xs: the data with size (N, D), where N is the number of sample, D is the dimension of feature
    :param weights: the parameters of linear model with size (D, 1)
    :return:
        the output of the model with size (N, 1)
        the gradient of the model with size (N, D)
    """
    # TODO: implement linear map and its gradient
    y = xs @ weights
    grad = xs
    return y, grad


def sgd_logistic_regression(xs: np.ndarray, ys: np.ndarray, batch_size: int = 100,
                            epochs: int = 50, lr: float = 1e-1) -> np.ndarray:
    """
    Training a Logistic regression model based on stochastic gradient descent
    :param xs: training data with size (N, D)
    :param ys: training labels with size (N, 1)
    :param batch_size: the batch size of SGD
    :param epochs: the number of epochs
    :param lr: the learning rate
    :return:
        the model parameters with size (D, 1)
    """
    num, dim = xs.shape
    weights = np.random.RandomState(1).randn(dim, 1)
    # TODO: Implement the SGD algorithm of logistic regression
    for epoch in range(epochs):
        idx = np.random.permutation(num)
        for i in range(0, num, batch_size):
            batch_xs = xs[idx[i:i+batch_size]]
            batch_ys = ys[idx[i:i+batch_size]]

            z, dz_dw = linear_model_with_grad(batch_xs, weights)
            p, dp_dz = sigmoid_function_with_grad(z)
            loss, dL_dp = binary_cross_entropy_with_grad(p, batch_ys)

            dL_dw = dz_dw.T @ (dL_dp * dp_dz)
            weights -= lr * dL_dw

    return weights


def test_logistic_regression(xs: np.ndarray, ys: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get prediction accuracy of the logistic regression model
    :param xs: testing data with size (N, D)
    :param ys: the ground truth labels with size (N, 1)
    :param weights: the model parameters with size (D, 1)
    :return:
        prediction accuracy in the range in [0, 1]
        prediction results with size (N, 1)
    """
    # TODO: Implement the testing script of logistic regression, hint: reuse the above functions you implement
    z, _ = linear_model_with_grad(xs, weights)
    ps, _ = sigmoid_function_with_grad(z)
    preds = (ps >= 0.5).astype(int)
    accuracy = np.mean(preds == ys)
    return accuracy, preds


def gender_fairness_check(preds: np.ndarray, genders: np.ndarray) -> Tuple[float, float]:
    """
    Find a way to check whether your classification results are fair with respect to gender or not
    :param preds: the results with size (N, )
    :param genders: the gender info with size (N, ), 1 for male and 0 for female
    :return:
        p(y=1|male) and p(y=1|female)
    """
    p1 = preds[genders == 1]
    p0 = preds[genders == 0]
    return np.sum(p1 == 1) / p1.shape[0], np.sum(p0 == 1) / p0.shape[0]


if __name__ == '__main__':
    data = adult_income_data_loader()
    weights = sgd_logistic_regression(xs=data['train'][0], ys=data['train'][1])
    accuracy1, preds1 = test_logistic_regression(xs=data['test'][0], ys=data['test'][1], weights=weights)

    w, c = linear_discriminant_analysis_2class(xs=data['train'][0], ys=data['train'][1])
    accuracy2, preds2 = test_lda(xs=data['test'][0], ys=data['test'][1], w=w, c=c)

    print('LR: Acc={:.4f}'.format(accuracy1))
    print('LDA: Acc={:.4f}'.format(accuracy2))

    p1, p0 = gender_fairness_check(preds1[:, 0], genders=data['test'][2])
    q1, q0 = gender_fairness_check(preds2[:, 0], genders=data['test'][2])

    print('LR: p(high income | male)={:.4f}, p(high income | female)={:.4f}'.format(p1, p0))
    print('LDA: p(high income | male)={:.4f}, p(high income | female)={:.4f}'.format(q1, q0))

    # TODO: An open problem: Given the output of the gender_fairness_check,
    #  could you propose a measurement for the fairness of classifier? Implement your measurement below
    dpd_lr = abs(p1 - p0)
    dpd_lda = abs(q1 - q0)
    print('Measurement: Demographic Parity Difference (DPD) -> |P(y=1|male) - P(y=1|female)|')
    print('LR DPD = {:.4f}'.format(dpd_lr))
    print('LDA DPD = {:.4f}'.format(dpd_lda))