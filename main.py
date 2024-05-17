import idx2numpy
from init import *
from run import *

# Set the size of the array
input_size = 784
layer_size = 16
output_digits = 10

lower_weight_limit = -4
upper_weight_limit = 4

bias = 0

#init(input_size, layer_size, output_digits, lower_weight_limit, upper_weight_limit, bias)

training_images = idx2numpy.convert_from_file('./training_data/train-images.idx3-ubyte')
training_labels = idx2numpy.convert_from_file('./training_data/train-labels.idx1-ubyte')

label = training_labels[0]
image = training_images[0]

input_array = image.flatten().tolist()

output_array = run(input_array)

cost = get_cost(output_array, label)

print(cost)