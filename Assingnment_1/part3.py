import pandas as pd
import numpy as np
import sys


df = pd.read_csv(sys.argv[1], delimiter=r"\s+");


learning_rate = 0.001

#Split the class from the training data
X = df.drop('class', axis=1)
y = df["class"].replace({"g": 1, "b": 0})

w = np.zeros(len(X.columns))

np.random.seed(0)
w = np.random.rand(len(X.columns))


def predict(inputs):
    if np.dot(inputs, w) > 0:
        return 1
    else:
        return 0


for i in range(100):
    predictions = []
    for inputs, label in zip(X.to_numpy(), y):
        prediction = predict(inputs)
        predictions.append(prediction)

        w = w + learning_rate * (label - prediction) * inputs
    
    accuracy = np.sum(np.equal(y, predictions)) / len(y)
    print(f"Accuracy: {accuracy}")

print(f"Final weights: {w}")
    