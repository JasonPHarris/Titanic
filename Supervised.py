# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:13:43 2022
@author: Jason Harris
This is the Supervised evaluation of survivor possibility based on the
provided tutorials and CS379T-Week-1-IP.xls file.I used 
https://betterprogramming.pub/df-survival-prediction-using-machine-learning-4c5ff1e3fa16
and https://www.youtube.com/watch?v=rODWw2_1mCI&ab_channel=ComputerScience to help with the code.
"""
# importing necessary extensions to handle the Excel file, and the df
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Reading the file
df = pd.read_excel('F:/DropBox/CTU/CS379 Machine Learning/IP1/CS379T-Week-1-IP.xls')  # importing xls file as df
# get some numerical statistics
print(df.describe())
# who lived and who died?
#print(df['survived'].value_counts())
# making visual aids
sns.countplot(df['survived'], label='Count')
# making a big set of tables
cols = ['sex', 'pclass', 'sibsp', 'parch', 'embarked']
n_rows = 1
n_cols = 5
# subplots and figure sizes
fig, axs = plt.subplots(n_rows, n_cols, figsize = (n_cols * 3.2, n_rows * 3.2))
# loop through each column to get data
for r in range(0, n_rows):
    for c in range(0, n_cols):
        i = r * n_cols + c  # index for number of columns
        ax = axs[c]         # showing subplot positions
        sns.countplot(df[cols[0+i]], hue=df['survived'], ax=ax)
        ax.set_title(cols[i])
        ax.legend(title='survived', loc='upper right')
plt.tight_layout()
# comparing some data
print(df.groupby('sex')[['survived']].mean()) # survival by gender
print()
print('****************************************************')
genClassSurv = df.pivot_table('survived', index='sex', columns='pclass')
print(genClassSurv) # survival by sex and class 
print()
print('****************************************************')
df.pivot_table('survived', index='sex', columns='pclass').plot() # display a line graph of this
plt.clf() # clearing plots for next table
sns.barplot(x='pclass', y='survived', data=df)
plt.clf() # clearing plots for next table
age = pd.cut(df['age'], [0, 18, 80])                        # separating based on adult and child
print(df.pivot_table('survived', ['sex', age], 'pclass'))   # survival broken down by age groups and gender
print('*****************************************************')
plt.scatter(df['fare'], df['pclass'], color = 'purple', label = 'Passenger Paid') # scatter graph price broken down by class
plt.ylabel('Class') # y axis label
plt.xlabel('Price/Fare') # x axis label
plt.title('Price of Each Class') # graph title
plt.legend() # showing the legend of the graph
plt.show() # show the graph itself
plt.clf() # clearing plots for next table
# print(df.isna().sum()) # find empty data
# =============================================================================
# for val in df: # checking redundant values
#     print(df[val].value_counts()) 
# =============================================================================
# doing some housekeeping 
df.drop(['name','body', 'cabin', 'ticket', 'home.dest', 'boat'], 1, inplace=True)         # removing unnecessary columns
#print('**************Columns Dropped******************')
df.dropna(subset=['embarked', 'age', 'fare'], inplace=True)      # dropping NaN
#print('**************NaN Dropped**********************')
# =============================================================================
# print(df.shape)                             # see the new df shape
# print(df.dtypes)                            # see the data types
# print(df['sex'].unique())                   # see the unique data
# print(df['embarked'].unique())              # see the unique data
# =============================================================================
# starting encoder
labelencoder = LabelEncoder()               
df.iloc[:, 2]=labelencoder.fit_transform(df.iloc[:, 2].values) # changing sex to a numerical value
df.iloc[:, 7]=labelencoder.fit_transform(df.iloc[:, 7].values) # changing embarked to a numerical value
# print(df.dtypes)                            # verify new data types
anyNull=np.any(np.isnan(df))
if anyNull == True:
    print('**** Only printed if there are any Null****')
    print(df.isna().sum()) # find empty data
anyInfin=np.all(np.isfinite(df))
if anyInfin != True:
    print('*** Only printed if there are any infinites***')
# split data into x and y variables
X = df.iloc[:, [0, 2, 3, 4, 5, 6, 7]].values  # setting X axis values
Y = df.iloc[:, 1].values                                # setting Y to the survived values (train the model)
from sklearn.model_selection import train_test_split    #  80/20 training/testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
sc = StandardScaler()                       # Scaling the data
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
# Plethora of machine learning models
def models(X_train, Y_train):
    from sklearn.linear_model import LogisticRegression # LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)
    from sklearn.neighbors import KNeighborsClassifier  # KNeighbors
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn.fit(X_train, Y_train)
    from sklearn.svm import SVC                         # linear SVC kernel
    svc_lin = SVC(kernel='linear', random_state=0)
    svc_lin.fit(X_train, Y_train)
    svc_rbf = SVC(kernel='rbf', random_state=0)         # rbf SVC kernel
    svc_rbf.fit(X_train, Y_train)
    from sklearn.naive_bayes import GaussianNB          # Gaussian NB 
    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)
    from sklearn.tree import DecisionTreeClassifier     # Decision Tree
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)
    from sklearn.ensemble import RandomForestClassifier # Forest
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)        
    forest.fit(X_train, Y_train)
    # Testing Accuracy
    print('[0]LogisticRegression Training Accuracy:', log.score(X_train, Y_train))
    print('[1]K Neighbors Training Accuracy:', knn.score(X_train, Y_train))
    print('[2]SVC Linear Training Accuracy:', svc_lin.score(X_train, Y_train))
    print('[3]SVC RBF Training Accuracy:', svc_rbf.score(X_train, Y_train))
    print('[4]Gaussian Training Accuracy:', gauss.score(X_train, Y_train))
    print('[5]DecisionTreeClassifier Training Accuracy:', tree.score(X_train, Y_train))
    print('[6]RandomForestClassifier Training Accuracy:', forest.score(X_train, Y_train))
    print()
    print('*********************************************')
    print()
    return log, knn, svc_lin, svc_rbf, gauss, tree, forest
model = models(X_train, Y_train)
# Confusion Matrix with accuracy on test data
from sklearn.metrics import confusion_matrix
for i in range(len(model)):
    cm=confusion_matrix(Y_test, model[i].predict(X_test))
    # Extract Ture Neg, False Pos, False Neg, True Pos
    TN, FP, FN, TP = cm.ravel()
    test_score = (TP+TN)/(TP+TN+FN+FP)
    #print(cm)
    print('Model [{}] Test Accuracy = "{}"'.format(i,test_score))
usedModel=model[6] # setting which model will be used
importances=pd.DataFrame({'feature':df.iloc[:, [0, 2, 3, 4, 5, 6, 7]].columns, 'importance': np.round(usedModel.feature_importances_, 3)})
importances=importances.sort_values('importance', ascending=False).set_index('feature')
print('****************************************************')
print(importances) # printing the results of what can help you survive
importances.plot.bar() # visual results
pred = model[6].predict(X_test)
# =============================================================================
# print(pred) #printing prediction
# print()
# print(Y_test) # printing actual
# =============================================================================
# [pclass survived(omitted) sex age sibsp parch fare embarked] 
my_death = [[1, 0, 1, 4, 4, 510, 2]] # creating my prediction
sc = StandardScaler()
my_scaled_survival = sc.fit_transform(my_death) # scaling my information
pred_my = model[6].predict(my_scaled_survival)  # generating the results 
print('***************************************************************')
print(pred_my)                                  # printing the results
if pred_my == 0:
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Oh no! You bought the farm!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
else:
    print('YAYAYAYAYAYAYAAYAYAYAYAYAYAAYAYAYAYAYAYA')
    print('How many women and children could have survived in your sted????')
    print('YAYAYAYAYAYAYAAYAYAYAYAYAYAAYAYAYAYAYAYA')
