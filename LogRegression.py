import math
import pandas as pd


class Logistic_Regression:
    def __init__(self, koefs=(), lr=0.0001, epoch=100):
        self.lr = lr
        self.epoch = epoch
        self.koefs = koefs
        self.X = pd.DataFrame()
        self.y_true = pd.DataFrame()

    def __sigmoid__(self, x):
        return 1 / (1 + math.e ** -x)

    def __our_prediction__(self, X, koefs):
        res = []
        for _, row in X.iterrows():
            s = 0
            for i in range(len(koefs)):
                s += koefs[i] * row[i]
            res.append(self.__sigmoid__(s))
        return res

    def __error__(self, y_true, y_pred):
        delta = 10 ** -10  # 0.0000000001
        return -y_true * math.log(delta + y_pred) - (1 - y_true) * math.log(delta + 1 - y_pred)

    def __total_error__(self, y_pred, y_true):
        _sum = 0
        for pred, true in zip(y_pred, y_true):
            _sum += self.__error__(pred, true)
        return _sum / len(y_pred)

    def __d_error__(self, f, X, koefs, y_true):
        derivatives = []
        delta = 0.00001
        y_pred_0 = f(X, koefs)
        f1 = self.__total_error__(y_pred_0, y_true)
        for i in range(len(koefs)):
            new_koefs = list(koefs)
            new_koefs[i] += delta
            y_pred_1 = f(X, new_koefs)
            f2 = self.__total_error__(y_pred_1, y_true)
            derivative = (f2 - f1) / delta
            derivatives.append(derivative)
        return derivatives

    def fit(self, X, y_true):
        self.X = X
        self.y_true = y_true
        for i in range(self.epoch):
            derivatives = self.__d_error__(self.__our_prediction__, X, self.koefs, y_true)
            for j in range(len(self.koefs)):
                self.koefs[j] -= self.lr * derivatives[j]
        print(self.koefs)

    def predict(self):
        my_preds = []
        for i in range(len(self.X)):
            temp = sum(self.X.iloc[i, :] * self.koefs)
            if self.__sigmoid__(temp) > 0.5:
                my_preds.append(1)
            else:
                my_preds.append(0)
        return my_preds

    def score(self):
        y_pred = self.predict()
        right = sum([1 for i, j in zip(y_pred, self.y_true) if i == j])
        return right / len(self.X)







