import random
import math

def get_weighted_sum(inputs, weights):
    weighted_sum = 0

    for input_index, input in enumerate(inputs):
        weighted_sum += input * weights[input_index]

    return weighted_sum

def sigmoid(x):

    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig

# Set the size of the array
input_size = 784
layer_size = 16

lower_weight_limit = -0.4
upper_weight_limit = 0.4

bias = 0

# Create an empty array
input_array = []
input_weights_matrix = []

layer1_array = []
layer1_weights_matrix = []

layer2_array = []
layer2_weights_matrix = []

output_array = []

# Initialize input values as random numbers between 0 and 1 in an array that represents each pixel in a 28x28 grayscale image
for _ in range(input_size):
    random_number = random.random()
    input_array.append(random_number)

# Initialize a matrix of weights initialized to a random number between 0 and 1 where the width is the layer size and the height is the input size
for layer_index in range(layer_size):
    input_weights_matrix.insert(layer_index, [])
    for input_index in range(input_size):
        input_weights_matrix[layer_index].insert(input_index, random.uniform(lower_weight_limit, upper_weight_limit))

# Initialize a matrix of weights initialized to a random number between 0 and 1 for the first layer
for next_layer_index in range(layer_size):
    layer1_weights_matrix.insert(next_layer_index, [])
    for layer1_index in range(layer_size):
        layer1_weights_matrix[next_layer_index].insert(layer1_index, random.uniform(lower_weight_limit, upper_weight_limit))

# Initialize a matrix of weights initialized to a random number between 0 and 1 for the second layer
for next_layer_index in range(layer_size):
    layer2_weights_matrix.insert(next_layer_index, [])
    for layer2_index in range(layer_size):
        layer2_weights_matrix[next_layer_index].insert(layer2_index, random.uniform(lower_weight_limit, upper_weight_limit))

# For each layer 1 element, calculate the weighted sum of the inputs and store it in the layer 1 array
for layer_index in range(layer_size):
    layer_weights = input_weights_matrix[layer_index]
    layer1_element_weighted_sum = sigmoid(get_weighted_sum(input_array, layer_weights) + bias)

    layer1_array.insert(layer_index, layer1_element_weighted_sum)

# For each layer 2 element, calculate the weighted sum of the inputs and store it in the layer 2 array
for layer_index in range(layer_size):
    layer_weights = layer1_weights_matrix[layer_index]
    layer2_element_weighted_sum = sigmoid(get_weighted_sum(layer1_array, layer_weights) + bias)

    layer2_array.insert(layer_index, layer2_element_weighted_sum)

print(layer1_array)
print(layer2_array)


# for index, input in enumerate(input_array):
#     print(index)