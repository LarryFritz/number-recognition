import random
import math
import numpy as np
from pathlib import Path
from utils import *

def init(input_size, layer_size, output_digits, lower_weight_limit, upper_weight_limit, bias):

    # Create an empty array
    input_array = []
    input_weights = []

    layer1_array = []
    layer1_weights = []
    layer1_biases = [0] * layer_size


    layer2_array = []
    layer2_weights = []
    layer2_biases = [0] * layer_size

    output_array = []
    output_biases = [0] * output_digits

    # Initialize input values as random numbers between 0 and 1 in an array that represents each pixel in a 28x28 grayscale image
    for _ in range(input_size):
        random_number = random.random()
        input_array.append(random_number)

    # Initialize a matrix of weights initialized to a random number between 0 and 1 where the width is the layer size and the height is the input size
    for layer_index in range(layer_size):
        input_weights.insert(layer_index, [])
        for input_index in range(input_size):
            input_weights[layer_index].insert(input_index, random.uniform(lower_weight_limit, upper_weight_limit))

    # Randomize layer 1 biases
    for bias_index, bias in enumerate(layer1_biases):
        layer1_biases[bias_index] = random.randint(lower_weight_limit, upper_weight_limit)        

    # Initialize a matrix of weights initialized to a random number between 0 and 1 for the first layer
    for next_layer_index in range(layer_size):
        layer1_weights.insert(next_layer_index, [])
        for layer1_index in range(layer_size):
            layer1_weights[next_layer_index].insert(layer1_index, random.uniform(lower_weight_limit, upper_weight_limit))

    # Randomize layer 2 biases
    for bias_index, bias in enumerate(layer2_biases):
        layer2_biases[bias_index] = random.randint(lower_weight_limit, upper_weight_limit)     

    # Initialize a matrix of weights initialized to a random number between 0 and 1 for the second layer
    for next_layer_index in range(output_digits):
        layer2_weights.insert(next_layer_index, [])
        for layer2_index in range(layer_size):
            layer2_weights[next_layer_index].insert(layer2_index, random.uniform(lower_weight_limit, upper_weight_limit))

    # Randomize output biases
    for bias_index, bias in enumerate(output_biases):
        output_biases[bias_index] = random.randint(lower_weight_limit, upper_weight_limit)        

    # For each layer 1 element, calculate the weighted sum of the inputs and store it in the layer 1 array
    for layer_index in range(layer_size):
        layer_weights = input_weights[layer_index]
        layer1_element_weighted_sum = sigmoid(get_weighted_sum(input_array, layer_weights) + layer1_biases[layer_index])

        layer1_array.insert(layer_index, layer1_element_weighted_sum)

    # For each layer 2 element, calculate the weighted sum of the inputs and store it in the layer 2 array
    for layer_index in range(layer_size):
        layer_weights = layer1_weights[layer_index]
        layer2_element_weighted_sum = sigmoid(get_weighted_sum(layer1_array, layer_weights) + layer2_biases[layer_index])

        layer2_array.insert(layer_index, layer2_element_weighted_sum)

    # For each output element, calculate the weighted sum of the inputs and store it in the output array
    for layer_index in range(output_digits):
        layer_weights = layer2_weights[layer_index]
        output_element_weighted_sum = sigmoid(get_weighted_sum(layer2_array, layer_weights) + output_biases[layer_index])

        output_array.insert(layer_index, output_element_weighted_sum)

    #print(layer1_array)
    #print(layer2_array)
    #print(output_array)
        
    p = Path('./output_data')
    p.mkdir(exist_ok=True)
        
    with (p / 'input_weights.txt').open('wb+') as input_weights_write:
        np.savetxt(input_weights_write, input_weights)

    with (p / 'layer1_biases.txt').open('wb+') as layer1_biases_write:
        np.savetxt(layer1_biases_write, layer1_biases)

    with (p / 'layer1_weights.txt').open('wb+') as layer1_weights_write:
        np.savetxt(layer1_weights_write, layer1_weights)

    with (p / 'layer2_biases.txt').open('wb+') as layer2_biases_write:
        np.savetxt(layer2_biases_write, layer2_biases)

    with (p / 'layer2_weights.txt').open('wb+') as layer2_weights_write:
        np.savetxt(layer2_weights_write, layer2_weights)

    with (p / 'output_biases.txt').open('wb+') as output_biases_write:
        np.savetxt(output_biases_write, output_biases)

    # f_read = (p / 'input_weights.txt').open('rb')
    # test = np.loadtxt(f_read).tolist()
    # print(test)