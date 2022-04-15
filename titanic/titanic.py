import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
import pandas as pd

# Data preprocess: train, test and 
data_path = './'

df_train_orginal_data = pd.read_csv(data_path + 'train.csv')

df_train_removal = df_train_orginal_data.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
df_train_dummy = pd.get_dummies(df_train_removal)

X = df_train_dummy.drop(['Survived'], axis=1)
y = df_train_dummy['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

df_submission = pd.read_csv(data_path + 'test.csv').drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
df_submission_dummy = pd.get_dummies(df_submission)

ID = pd.read_csv(data_path + 'test.csv')['PassengerId']

# Logistic regression
lg = LogisticRegression()
lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)

print('The accuracy_score of logistic regression is {}'.format(accuracy_score(y_test, y_pred)))

submission_pred = lg.predict(df_submission_dummy)

submission = pd.concat([pd.DataFrame(ID), pd.DataFrame(submission_pred)], axis=1)
submission.columns = ['PassengerId','Survived']
submission.to_csv('submission_lg.csv', index=False)

# XGBoost
xgb_classifer = xgb.XGBClassifier()

xgb_classifer.fit(X_train, y_train)
y_pred = xgb_classifer.predict(X_test)
print('The accuracy_score of XGBoost is {}'.format(accuracy_score(y_test, y_pred)))

submissioin_pred = xgb_classifer.predict(df_submission_dummy)

submission = pd.concat([pd.DataFrame(ID), pd.DataFrame(submissioin_pred)], axis=1)
submission.columns = ['PassengerId','Survived']
submission.to_csv('submission.csv', index=False)