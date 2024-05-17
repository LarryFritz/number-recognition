import numpy as np
from utils import *

def run(input_array):
    layer1_array = []
    layer2_array = []
    output_array = []

    input_weights = np.loadtxt('./output_data/input_weights.txt').tolist()
    layer1_biases = np.loadtxt('./output_data/layer1_biases.txt').tolist()
    layer1_weights = np.loadtxt('./output_data/layer1_weights.txt').tolist()
    layer2_biases = np.loadtxt('./output_data/layer2_biases.txt').tolist()
    layer2_weights = np.loadtxt('./output_data/layer2_weights.txt').tolist()
    output_biases = np.loadtxt('./output_data/output_biases.txt').tolist()

        # For each layer 1 element, calculate the weighted sum of the inputs and store it in the layer 1 array
    for layer_index, layer_weights in enumerate(input_weights):
        layer1_element_weighted_sum = sigmoid(get_weighted_sum(input_array, layer_weights) + layer1_biases[layer_index])

        layer1_array.insert(layer_index, layer1_element_weighted_sum)

    # For each layer 2 element, calculate the weighted sum of the inputs and store it in the layer 2 array
    for layer_index, layer_weights in enumerate(layer1_weights):
        layer2_element_weighted_sum = sigmoid(get_weighted_sum(layer1_array, layer_weights) + layer2_biases[layer_index])

        layer2_array.insert(layer_index, layer2_element_weighted_sum)

    # For each output element, calculate the weighted sum of the inputs and store it in the output array
    for layer_index, layer_weights in enumerate(layer2_weights):
        output_element_weighted_sum = sigmoid(get_weighted_sum(layer2_array, layer_weights) + output_biases[layer_index])

        output_array.insert(layer_index, output_element_weighted_sum)

    return output_array