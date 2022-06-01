import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from NeuralNetwork import Neural_Network

# Parameters. As per the handout.
n_in = 4
n_hidden = 2
n_out = 3
learning_rate = 0.2

initial_hidden_layer_weights = np.array([[-0.28, -0.22], [0.08, 0.20], [-0.30, 0.32], [0.10, 0.01]])
initial_output_layer_weights = np.array([[-0.29, 0.03, 0.21], [0.08, 0.13, -0.36]])
hidden_bias_values = np.array([-0.02, -0.20])
output_bias_values = np.array([-0.33, 0.26, 0.06])



def encode_labels(labels):
    # encode 'Adelie' as 1, 'Chinstrap' as 2, 'Gentoo' as 3
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    # don't worry about this
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    # encode 1 as [1, 0, 0], 2 as [0, 1, 0], and 3 as [0, 0, 1] (to fit with our network outputs!)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return label_encoder, integer_encoded, onehot_encoder, onehot_encoded

data = pd.read_csv('part1/penguins307-train.csv')
# the class label is last!
labels = data.iloc[:, -1]
# seperate the data from the labels
instances = data.iloc[:, :-1]
#scale features to [0,1] to improve training
scaler = MinMaxScaler()
instances = scaler.fit_transform(instances)
# We can't use strings as labels directly in the network, so need to do some transformations
label_encoder, integer_encoded, onehot_encoder, onehot_encoded = encode_labels(labels)
labels = onehot_encoded

pd_data_ts = pd.read_csv('part1/penguins307-test.csv')
test_labels = pd_data_ts.iloc[:, -1]
test_instances = pd_data_ts.iloc[:, :-1]
#scale the test according to our training data.
test_instances = scaler.transform(test_instances)

def sensitivity_testing():
    learning_rates = [0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 1]


    learning_rate_acc = []

    #Run through each learning rate
    for n in learning_rates:
        nn = Neural_Network(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,
                        n, hidden_bias_values, output_bias_values)

        nn.train(instances, labels, 100, integer_encoded.flatten())

        #Obtain the accuracy
        test_prediction = nn.predict(test_instances)
        integer_encoded_test = label_encoder.fit_transform(test_labels)
        learning_rate_acc.append(np.sum(np.equal(integer_encoded_test.flatten(), test_prediction)) / len(test_prediction))



    fig = plt.figure(figsize=(10,5))
    plt.plot(learning_rates, learning_rate_acc)
    fig.suptitle('Sensitivity Testing', fontsize=20)
    plt.xlabel('learning_rate', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)


 
    plt.show()



sensitivity_testing()
