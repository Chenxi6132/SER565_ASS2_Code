from sklearn.naive_bayes import MultinomialNB
import numpy as np


class NaiveBayer(object):
    def __init__(self, classes):
        self.models = {}
        self.classes = classes

        for cls in self.classes:
            model = MultinomialNB()
            self.models[cls] = model

    def fit(self, data_x, data_y):
        for idx, cls in enumerate(self.classes):
            class_labels = data_y[:, idx]
            self.models[cls].fit(data_x, class_labels)

    # calculate accurate, classification report. it returns label
    def predict(self, test_x):
        predictions = np.zeros((test_x.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            predictions[:, idx] = self.models[cls].predict(test_x)
        return predictions

    def predict_prob(self, test_x):
        predictions = np.zeros((test_x.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):  # 遍历每一个分类
            predictions[:, idx] = self.models[cls].predict_proba(test_x)[:, 1]  # 把结果写进去
        return predictions
