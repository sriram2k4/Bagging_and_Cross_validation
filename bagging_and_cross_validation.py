import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

### Data Preprocessing
dataset = pd.read_csv("diabetes.csv")

print(dataset.head())

print(dataset.describe()) # Description about dataset

print(dataset.Outcome.value_counts())# Checking if the dataset is unbalanced or balanced

print(dataset.isnull().sum()) # No Null

X = dataset.drop("Outcome",axis=1).values
y = dataset["Outcome"].values

### Standardization 
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_test = scaler.transform(X)


### Split train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

### Bagging using Decision Tree Classifier and Cross Validation
decision_tree_bagging_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=50,
    max_samples=80,
    oob_score=True,
    random_state=0
)

decision_tree_bagging_model.fit(X_train, y_train)
print("Decision Tree Bagging oob score -",decision_tree_bagging_model.oob_score_) # 0.762214983713355

print("Decision Tree Bagging model score - ",decision_tree_bagging_model.score(X_test, y_test)) # 0.7922077922077922

dt_score = cross_val_score(decision_tree_bagging_model, X, y, cv=10)
print("Decision Tree Bagging model cross validation score -",dt_score.mean()) # 0.7578947368421052

### Random forest classification and Cross Validation
clf = RandomForestClassifier(n_estimators=25)
rf_score = cross_val_score(clf, X, y, cv=15)
print("Random Forest model cross validation score -",rf_score.mean()) # 0.7669180492709906 - Optimal Estimate for the Random Forest Model