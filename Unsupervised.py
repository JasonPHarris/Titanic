# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:13:43 2022
@author: Jason Harris
This is the Unsupervised evaluation of survivor possiblity based on the 
provided tutorials and Titanic.xls file. I used 
https://www.youtube.com/watch?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&v=8p6XaQSIFpY 
videos 35 and 36 to help with the code.
"""

# importing necessary extenstions to handle the excel file, and the df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
sns.set()
df = pd.read_excel('F:/DropBox/CTU/CS379 Machine Learning/IP1/CS379T-Week-1-IP.xls')  # importing xls file as df (aka data frame)
# doing some housekeeping
df.drop(['name','ticket', 'cabin', 'body', 'home.dest'], 1, inplace=True)         # removing unnecessary columns
df.dropna(subset=['embarked', 'age', 'fare'], inplace=True)      # dropping NaN
# df.convert_objects(convert_numeric=True)    # converting cells to numeric - commented out due to errors that halt
# processing
df.fillna(0, inplace=True)  # converting NaN values
# handling other data that has not converted yet
def handle_non_numeric_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df
df = handle_non_numeric_data(df)  # calls the above function to handle non-numeric data
X = np.array(df.drop(['survived'], 1).astype(float))  # don't include 'survived' in x axis comparison data
X = preprocessing.scale(X)  # preprocessing of all pertinent data
y = np.array(df['survived'])  # set the y-axis as survived or deceased

clf = KMeans(n_clusters=2)  # 2 clusters ('survived' == 1 and 'survived' == 0 (dead))
clf.fit(X)

correct = 0
for i in range(len(X)):  # iterate over the data in the array
    predict_me = np.array(X[i].astype(float))  # setting array type to float
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct / len(X))  # printing the correct result.
