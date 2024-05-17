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
    
def relu(x):
    return max(0, x)

def get_cost(output_array, label):
    sum = 0
    for output_index, output in enumerate(output_array):
        diff = 0
        
        if output_index == label:
            diff = output - 1
        else:
            diff = output
        
        sum += math.pow(diff, 2)
    return sum