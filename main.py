import pandas as pd
from logisticRegression.LogRegression import Logistic_Regression
from logisticRegression.metrics import precision, recall, f_score

path = "exam_data.csv"
df = pd.read_csv(path)
X = df.drop('is admitted', axis=1)
y = df['is admitted']
new_X = X.copy()
new_X['beta'] = 1


clf = Logistic_Regression(epoch=100, koefs=[0.2, 0.2, -20])
clf.fit(new_X, y)
print(clf.score())
df['y_pred'] = clf.predict()
print("precision ", precision(df, right="is admitted"))
print("recall ", recall(df, right="is admitted"))
print("f_score ", f_score(df, right="is admitted"))


