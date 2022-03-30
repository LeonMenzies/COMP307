import pandas as pd
import numpy as np
import sys

df = pd.read_csv(sys.argv[1], delimiter=r"\s+");

learning_rate = 0.001

#Split the class from the training data
X = df.drop('class', axis=1)
y = df["class"].replace({"g": 1, "b": 0})

#Add the dummy row for bias
X.insert(0, "f0", 1)

np.random.seed(0)
weights = np.random.rand(len(X.columns))

#Take a dot product
def predict(inputs, w):
    if np.dot(inputs, w) > 0:
        return 1
    else:
        return 0

def perceptron(data, w):
    #Run a loop until ddesired accuracy or given amount of loops (500 in this case)
    epochs = 0
    while True:
        predictions = []
        for inputs, label in zip(data.to_numpy(), y):
            #Mkae preication and the ajust weights
            prediction = predict(inputs, w)
            predictions.append(prediction)

            w = w + learning_rate * (label - prediction) * inputs
        
        #Calcualte the accuracy
        accuracy = np.sum(np.equal(y, predictions)) / len(y)
        epochs+=1

        if(accuracy > 0.9 or epochs > 500):
            print(f"Accuracy: {accuracy}")
            break
        
    print(f"Epcohs: {epochs}")    
    return w

new_weights = perceptron(X, weights)
print(f"Final weights: {new_weights}")

print("========================================")
print("Spliting the data and running train test")
print("========================================")

df = pd.read_csv("./part3/ionosphere.data", delimiter=r"\s+");

#Add the dummy row for bias
df.insert(0, "f0", 1)

#Split the data
split = (int)(len(df) * 0.70)

np.random.seed(0)
weights = np.random.rand(len(X.columns))


X_train = df.loc[0:split]
X_test = df.loc[split: len(df)]


#Seperate features form class
X = X_train.drop('class', axis=1)
y = X_train["class"].replace({"g": 1, "b": 0})

new_weights = perceptron(X, weights)

#prepare test data
X_test_split = X_test.drop('class', axis=1)
y_test_split = X_test["class"].replace({"g": 1, "b": 0})


count = 0

for inputs, label in zip(X_test_split.to_numpy(), y_test_split):
    if predict(inputs, new_weights) == label:
        count += 1


print(f"Accuracry on test data: {count/len(X_test_split)}")
