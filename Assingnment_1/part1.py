import pandas as pd
import numpy as np
import sys
from statistics import mode


#Save data to dataframes
df_train = pd.read_csv(sys.argv[1], delimiter=r"\s+");
df_test = pd.read_csv(sys.argv[2], delimiter=r"\s+");

#Split the class from the training data
X = df_train.drop('Class', axis=1)
y = df_train.Class

#Normalise
X=(X-X.min())/(X.max()-X.min())

#remove the class from the test data
X_test = df_test.drop('Class', axis=1)
y_test = df_test.Class

#Normalise
X_test=(X_test-X_test.min())/(X_test.max()-X_test.min())

def euclidean_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

    from statistics import mode

def KNN(k):
    results = []

    for j in X_test.index:
        
        distances = []  

        for i in X.index:
            distances.append(euclidean_dist(X_test.iloc[j], X.iloc[i]))

        sorted_distances = np.argsort(distances)[:k]
        labels = y[sorted_distances]

        results.append(mode(labels))        

    accuracy = np.sum(np.equal(y_test, results)) / len(y_test)

    print(f"K={k} Accuracy: {accuracy}")


KNN(1)
KNN(3)