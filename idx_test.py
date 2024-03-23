import idx2numpy

training_images = idx2numpy.convert_from_file('./training_data/train-images.idx3-ubyte')
training_labels = idx2numpy.convert_from_file('./training_data/train-labels.idx1-ubyte')

print(training_labels)