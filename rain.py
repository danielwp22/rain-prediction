import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Directory for 
wdata = '/Users/danielpalin/Downloads/Weather_Data.csv'

df = pd.read_csv(wdata)

#Replace class columns with binary values
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)

#Split dataset into features and target
df_sydney_processed.drop('Date',axis=1,inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
target = df_sydney_processed['RainTomorrow']

#Train test split
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=10)


#linear regression model
LinearReg = LinearRegression()
LinearReg.fit(x_train, y_train)
print("Intercept:", LinearReg.intercept_)  # Model intercept
print("Coefficients:", LinearReg.coef_)   # Model coefficients

predictions = LinearReg.predict(x_test)

#Looking at the errors
LinearRegression_MAE = metrics.mean_absolute_error(y_test, predictions)
LinearRegression_MSE = metrics.mean_squared_error(y_test, predictions)
LinearRegression_R2 = metrics.r2_score(y_test, predictions)

Report = pd.DataFrame({
    'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'R² Score'],
    'Value': [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2]
})
print(Report)

#K nearest neighbours model
KNN = KNeighborsClassifier(n_neighbors = 4)
KNN.fit(x_train, y_train)

predictions = KNN.predict(x_test)

KNN_Accuracy_Score = accuracy_score(y_test, predictions)
KNN_JaccardIndex = jaccard_score(y_test, predictions)
KNN_F1_Score = f1_score(y_test, predictions)

#Decision tree model
Tree = DecisionTreeClassifier()
Tree.fit(x_train, y_train)

predictions = Tree.predict(x_test)

Tree_Accuracy_Score = accuracy_score(y_test, predictions)
Tree_JaccardIndex = jaccard_score(y_test, predictions)
Tree_F1_Score = f1_score(y_test, predictions)

#Logistic Regression
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)
LR = LogisticRegression(solver = 'liblinear')
LR.fit(x_train, y_train)

predictions = LR.predict(x_test)
predict_proba = LR.predict_proba(x_test)

LR_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
LR_JaccardIndex = metrics.jaccard_score(y_test, predictions)
LR_F1_Score = metrics.f1_score(y_test, predictions)
LR_Log_Loss = metrics.log_loss(y_test, predict_proba)

#SVM
SVM = svm.SVC()
SVM.fit(x_train,y_train)
predictions = SVM.predict(x_test)

SVM_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
SVM_JaccardIndex = metrics.jaccard_score(y_test, predictions)
SVM_F1_Score = metrics.f1_score(y_test, predictions)

Report = pd.DataFrame({
    'Model': ['KNN', 'Decision Tree', 'Logistic Regression', 'SVM'],
    'Accuracy': [ 
        KNN_Accuracy_Score, Tree_Accuracy_Score, LR_Accuracy_Score, SVM_Accuracy_Score
    ],
    'Jaccard Index': [
        KNN_JaccardIndex, Tree_JaccardIndex, LR_JaccardIndex, SVM_JaccardIndex
    ],
    'F1 Score': [
        KNN_F1_Score, Tree_F1_Score, LR_F1_Score, SVM_F1_Score
    ],
    'Log Loss': [
        None,  # Log Loss is not for KNN
        None,  # Log Loss is not for Decision Tree
        LR_Log_Loss,  # Only Logistic Regression has Log Loss
        None  # Log Loss is not for SVM
    ]
})

# Display the summary report
print(Report)

import matplotlib.pyplot as plt
import numpy as np

SVM_Accuracy_Score = metrics.accuracy_score(y_test, predictions)
SVM_JaccardIndex = metrics.jaccard_score(y_test, predictions)
SVM_F1_Score = metrics.f1_score(y_test, predictions)

# Define metrics and scores for the models
models = ['Accuracy', 'Jaccard Index', 'F1 Score']
knn_scores = [KNN_Accuracy_Score, KNN_JaccardIndex, KNN_F1_Score]
tree_scores = [Tree_Accuracy_Score, Tree_JaccardIndex, Tree_F1_Score]
logistic_scores = [LR_Accuracy_Score, LR_JaccardIndex, LR_F1_Score]
svm_scores = [SVM_Accuracy_Score, SVM_JaccardIndex, SVM_F1_Score]  

# Bar width and positions
bar_width = 0.2
x = np.arange(len(models))  # Position for the first group of bars

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(x, knn_scores, width=bar_width, label='KNN', color='blue')
plt.bar(x + bar_width, tree_scores, width=bar_width, label='Decision Tree', color='green')
plt.bar(x + 2 * bar_width, logistic_scores, width=bar_width, label='Logistic Regression', color='red')
plt.bar(x + 3 * bar_width, svm_scores, width=bar_width, label='SVM', color='purple')

# Add labels and title
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.title('Model Performance Metrics', fontsize=14)
plt.xticks(x + 1.5 * bar_width, models)  # Center the labels under the grouped bars
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()
