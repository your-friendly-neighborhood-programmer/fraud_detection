import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Load data
transactions = pd.read_csv('fraud_detection/transactions.csv')
print(transactions.head())
print(transactions.info())

# Number of fraudulent transactions
sum_frauds = transactions.isFraud.sum() # 282 fraudulent transactions
print(sum_frauds)

# Summary statistics on amount column
print(transactions.amount.describe())

# create fields
payments = ['PAYMENT', 'DEBIT']
transactions['isPayment'] = np.where(transactions['type'].isin(payments), 1, 0)
movements = ['CASH_OUT', 'TRANSFER']
transactions['isMovement'] = np.where(transactions['type'].isin(movements), 1, 0)
transactions['accountDiff'] = abs(transactions['oldbalanceOrg'] - transactions['oldbalanceDest'])

# create features & label variables
features = transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']]
label = transactions[['isFraud']]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=27)

# scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train model
model = LogisticRegression()
model.fit(X_train, y_train)

# score model on training and test data
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

# print coefficients
print(model.coef_)

# confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()

# save plot
plt.savefig('fraud_detection/confusion_matrix.png')

# accuracy, precision, recall
true_positives = cm[0, 0]
true_negatives = cm[1, 1]
false_positives = cm[1, 0]
false_negatives = cm[0, 1]

accuracy = (true_positives + true_negatives) / len(y_test)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

print(accuracy)
print(precision)
print(recall)

# This model performs well with an accuracy of 0.999, precision of 0.999, and recall of 1.0.

