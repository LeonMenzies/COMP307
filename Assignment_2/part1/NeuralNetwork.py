import numpy as np
import math

class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, learning_rate, hidden_bias_values, output_bias_values):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights

        self.learning_rate = learning_rate
        self.hidden_bias_values = hidden_bias_values
        self.output_bias_values = output_bias_values

    # Calculate neuron activation for an input
    def sigmoid(self, input):
        # TODO!
        output = 1 / (1 + math.exp(-input))
        return output

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        hidden_layer_outputs = []
        for i in range(self.num_hidden):
            # TODO! Calculate the weighted sum, and then compute the final output.
            weighted_sum = 0.
            for input, weights in zip(inputs, self.hidden_layer_weights): 
                weighted_sum += input * weights[i]

            #Add bias
            weighted_sum += self.hidden_bias_values[i]
            hidden_layer_outputs.append(self.sigmoid(weighted_sum))

        output_layer_outputs = []
        for i in range(self.num_outputs):
            # TODO! Calculate the weighted sum, and then compute the final output.

            weighted_sum = 0.
            for input, weights in zip(hidden_layer_outputs, self.output_layer_weights): 
                weighted_sum += input * weights[i]

            #Add bias
            weighted_sum += self.output_bias_values[i]
            output_layer_outputs.append(self.sigmoid(weighted_sum))

        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):

        output_layer_betas = np.zeros(self.num_outputs)
        # TODO! Calculate output layer betas.
        output_layer_betas = desired_outputs - output_layer_outputs;
       
        #print('OL betas: ', output_layer_betas)

        hidden_layer_betas = np.zeros(self.num_hidden)
        # TODO! Calculate hidden layer betas.
        
        for i in range(self.num_hidden):
            vals = []
            for j in range(self.num_outputs):
                vals.append(self.output_layer_weights[i][j] * output_layer_outputs[j] * (1 - output_layer_outputs[j]) * output_layer_betas[j])
            hidden_layer_betas[i] = sum(vals)

        #print('HL betas: ', hidden_layer_betas)

        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        # TODO! Calculate output layer weight changes.
        for i in range(self.num_hidden):
            for j in range (self.num_outputs):
                delta_output_layer_weights[i][j] = self.learning_rate * hidden_layer_outputs[i] * output_layer_outputs[j] * (1 - output_layer_outputs[j]) * output_layer_betas[j]

        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        # TODO! Calculate hidden layer weight changes.
        for i in range(self.num_inputs):
                    for j in range (self.num_hidden):
                        delta_hidden_layer_weights[i][j] = self.learning_rate * inputs[i] * hidden_layer_outputs[j] * (1 - hidden_layer_outputs[j]) * hidden_layer_betas[j]

        #Update hidden layer bias 
        hidden_layer_bias = np.zeros(self.num_hidden)
        for j in range(self.num_hidden):
            hidden_layer_bias[j] = self.learning_rate * hidden_layer_outputs[j] * (1 - hidden_layer_outputs[j]) * hidden_layer_betas[j]

        #Update output layer bias 
        output_layer_bias = np.zeros(self.num_outputs)
        for j in range (self.num_hidden):
            output_layer_bias[j] = self.learning_rate * output_layer_outputs[j] * (1 - output_layer_outputs[j]) * output_layer_betas[j]

        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights, output_layer_bias, hidden_layer_bias

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights, output_layer_bias, hidden_layer_bias):
        # TODO! Update the weights.
        self.output_layer_weights += delta_output_layer_weights
        self.hidden_layer_weights += delta_hidden_layer_weights
        self.hidden_bias_values += hidden_layer_bias
        self.output_bias_values += output_layer_bias

    def train(self, instances, desired_outputs, epochs, desired_outputs_decoded):

        for epoch in range(epochs):
            print('epoch = ', epoch)
            predictions = []
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights, output_layer_bias, hidden_layer_bias = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i])
                # TODO!
                predicted_class = max(range(len(output_layer_outputs)), key=output_layer_outputs.__getitem__)  
                predictions.append(predicted_class)

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights, output_layer_bias, hidden_layer_bias)

            # Print new weights
            # print('Hidden layer weights \n', self.hidden_layer_weights)
            # print('Output layer weights  \n', self.output_layer_weights)

            # TODO: Print accuracy achieved over this epoch
            acc = np.sum(np.equal(desired_outputs_decoded, predictions)) / len(predictions)
            print('acc = ', acc)

    def predict(self, instances):
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            # TODO! Should be 0, 1, or 2.
            predicted_class = max(range(len(output_layer_outputs)), key=output_layer_outputs.__getitem__)  
            predictions.append(predicted_class)
        return predictions