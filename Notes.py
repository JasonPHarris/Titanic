# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 19:45:56 2022

@author: Jason Harris
"""


# handling other data that has not converted yet
df['sex'] = [1 if item=='male' else 0 for item in df['sex']]   # changing sex from male and female to 1 and 2
df['embarked'].fillna = 'Q' 
df['embarked'] = [0 if item=='S' else 1 if item=='C' else 2 for item in df['embarked']]
df.head()

def handle_non_numeric_data(df):
    columns = df.columns.values
    
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if df[column].dtype != np.int64 and df[column].dtype !=np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
                    
            df[column] = list(map(convert_to_int, df[column]))
            
    return df

df = handle_non_numeric_data(df)    # calls the above function to handle non-numeric data

#converting string data to numerical data
df['sex'] = [1 if item=='male' else 0 for item in df['sex']]   # changing sex from male and female to 1 and 0
df['embarked'] = [0 if item=='S' else 1 if item=='C' else 2 for item in df['embarked']]
df.head()
 
# survival based on gender
plt.scatter(df['survived'],df['sex'], marker='^', color = 'blue', s=10)
plt.xlim(-1,2)
plt.ylim(0,2)
plt.show()
 
# survival based on age
plt.scatter(df['survived'], df['age'], marker='o', color = 'orange', s=10)
plt.xlim(-1, 2)
plt.ylim(0, 100)
plt.show()
 
# survival based on embarking station 0 or 'S' == Southampton, 1 or 'C' == Cherbourg, and 2 or 'Q' == Queenstown
plt.scatter(df['survived'], df['embarked'], marker='H', color = 'brown', s=10)
plt.xlim(-1, 2)
plt.ylim(-2, 4)
plt.show()
 
# survival based on fare in British pounds
plt.scatter(df['survived'], df['fare'], marker='P', color = 'green', s=10)
plt.xlim(-1, 2)
plt.ylim(0, 600)
plt.show()
 
# survival based on siblings/spouses aboard
plt.scatter(df['survived'], df['sibsp'], marker='.', color = '#ffc0cb', s=10)
plt.xlim(-1, 2)
plt.ylim(0, 6)
plt.show()
 
# survival based on parents/children aboard
plt.scatter(df['survived'], df['parch'], marker='x', color = 'red', s=10)
plt.xlim(-1, 2)
plt.ylim(0, 6)
plt.show()
 
# survival based on passenger class
plt.scatter(df['survived'], df['pclass'], marker='X',color = 'black', s=10)
plt.xlim(-1, 2)
plt.ylim(0, 4)
plt.show()


# =============================================================================
# X = np.array([[1,2],
#               [1.5, 1.8],
#               [5, 8],
#               [8, 8],
#               [1, 0.6],
#               [9, 11]])
# 
# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()
# 
# colors = 10*["g.", "r.", "c.", "b.", "k."]
# 
# class K_Means: # defining our custom kmeans class
#     # tol = centroid movement in percent, max_iter is max mnumber of iterations before failure
#     def _init_(self, k=2, tol=0.001, max_iter=300) 
#         # define starting values    
#         self.k = k
#         self.tol = tol
#         self.max_iter = max_iter
#         
#     # creating 
#     def fit(self, data):
#         
#         self.centroids = {} # dictionary for centroids
#         
#         for i in range(self.k):
#             self.centroids[i] = data[i] # passing first 2 data as first 2 centroids
#             
#         for i in range(self.max_iter): 
#             # dictionary for optimization keys are centroids empties and redoes the centroids every iteration
#             self.classifications = {} 
#             
#             for i in range(self.k):
#                 self.classifications[i] = [] # passes features set of keys
#                 
#             for featureset in data:
#                 # calculate the distances of the data set
#                 distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
#                 classification = distances.index(min(distances))
#                 self.classifications[classification].append(featureset)
#                 
#             prev_centroids = dict(self.centroids)    #compare versus previous centroids
#             
#             for classification in self.classifications:
#                 pass
#             
#                 # finding the mean of all the features for any given values
#                 #self.centroids[classification] = np.average(self.classifications[classification], axis=0) 
#     
#     def predict(self, data):
#         pass
#     
# =============================================================================
    
