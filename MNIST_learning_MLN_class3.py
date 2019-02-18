from matplotlib import pyplot as plt
import numpy as np
import sys


def get_MNIST_data(filename):

    print("loading file")
    file_to_load = np.genfromtxt(filename, delimiter=',')
    print("file loaded")
    return file_to_load


def extract_labels(dataset):

    x_label = dataset[:,0]
    dataset = np.delete(dataset, 0, 1)
    return x_label, dataset


def display_image(image, image_size):
    image_resize = image.reshape(image_size, image_size)
    plt.imshow(image_resize, cmap='gray')
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_back(x):
    return x * (1 - x)


def relu_activation(x, flag):

    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                pass  # do nothing since it would be effectively replacing x with x
            else:
                x[i][k] = 0
    return x


def relu_back(x):
    for i in range(0, len(x)):
        for j in range(0,len(x[i])):
            if x[i][j] > 0:
                x[i][j] = 1
            else:
                x[i][j] = 0
    return x


def tanh(x):
    return (2 / (1 + np.exp(-2*x))) - 1.0


def tanh_back(x):
    return 1 - x ** 2


def softmax(z):

    y = np.exp(z)
    soft = y / np.sum(y, axis=1, keepdims=True)
    return soft


def softmax_loss(activation, target):

    indices = np.argmax(target).astype(int)
    predicted = activation[np.arange(len(activation)), indices]
    log_preds = np.log(predicted)
    loss = -1.0*np.sum(log_preds) / len(log_preds)
    return loss


def cross_entropy_softmax_loss_array(softmax_probs_array, y_onehot):

    log_preds = np.multiply(y_onehot, np.log(softmax_probs_array))
    loss = -1.0 * np.sum(log_preds)
    return loss


def initialize_weights(input_size, layer_size, type):

    if type == 1:
        print("Uniform Distrution")
        weights = np.random.uniform(-1, 1, size=(input_size, layer_size))
    elif type == 2:
        sigma = np.sqrt(2 / input_size)
        print("Normal dist " + str(sigma))
        weights = np.random.normal(0.0, sigma, size=(input_size, layer_size))
    bias = np.zeros(layer_size)

    return weights, bias


def create_exp(num, size):
    output = np.zeros(shape=(size, 10))
    for i in range(0, size):
        output[i, int(num[i])] = 1
    # print(output)
    return output


def accuracy(predictions, labels):
    correct_predictions = 0
    preds = np.argmax(predictions, 1)

    for i in range(0,len(labels)):
        if preds[i] == labels[i]:
            correct_predictions += 1
    total_accuracy = correct_predictions / predictions.shape[0]
    return total_accuracy


class FullyConnectedLayer:

    def __init__(self, input_size, layer_size):
        self.weights, self.bias = initialize_weights(input_size, layer_size, 2)
        self.input_size = input_size
        self.layer_size = layer_size
        self.activation = np.ones(layer_size)
        self.output = np.ones(layer_size)

        print("Fully Connected Layer {} input {} output".format(input_size, layer_size))

    def feedforward(self, input_layer):
        self.activation = np.dot(input_layer, self.weights) + self.bias
        self.output = relu_activation(self.activation, 0)
        return self.output

    def fftest(self, input_layer):
        layer_input = np.dot(input_layer, self.weights) + self.bias
        layer_activation = relu_activation(layer_input, 0)
        return layer_activation

    def backprop(self, next_activation, loss, layer_weights, layer_bias):
        gradient_output_weights = np.dot(next_activation.T, loss)
        gradient_output_bias = np.sum(loss)
        error_next = np.dot(loss, layer_weights.T) * relu_back(next_activation)
        layer_weights -= step_size*gradient_output_weights
        layer_bias -= step_size*gradient_output_bias
        return error_next, layer_weights, layer_bias


class OutputLayer:

    def __init__(self, input_size, layer_size):
        self.weights, self.bias = initialize_weights(input_size, layer_size, 2)
        self.input_size = input_size
        self.layer_size = layer_size
        self.activation = np.ones(layer_size)
        print("Output Layer Defined {} input {} output".format(input_size, layer_size))

    def outputlayer(self, input_layer, y_train):
        self.activation = np.dot(input_layer, self.weights) + self.bias
        output = softmax(self.activation)
        _output_exp = create_exp(y_train, batch_size)
        return output, _output_exp

    def backprop(self, next_activation, loss, layer_weights, layer_bias):
        gradient_output_weights = np.dot(next_activation.T, loss)
        gradient_output_bias = np.sum(loss)
        error_next = np.dot(loss, layer_weights.T) * relu_back(next_activation)
        layer_weights -= step_size*gradient_output_weights
        layer_bias -= step_size*gradient_output_bias
        return error_next, layer_weights, layer_bias


#
#
# main program
# opti1 implements multiple hidden layers and arbitrary batch size
# opti2 implements dropout
# class2 implements class' for layer types
# class3 implements class list with in a dictionary for easy addition of layers and layer types
#
#

# import and organize training and test data

x_train = get_MNIST_data("mnist_train_100.csv")
x_test = get_MNIST_data("mnist_test_10.csv")

#x_train = get_MNIST_data("mnist_train_5000.csv")
# x_train = get_MNIST_data("mnist_train_20000.csv")
# x_train = get_MNIST_data("mnist_train.csv")
#x_test = get_MNIST_data("mnist_test.csv")

y_train, x_train = extract_labels(x_train)
y_test, x_test = extract_labels(x_test)

x_train /= 255.
x_test /= 255.

print(y_train)

# display_image(x_train[0], 28)
#
# define network hyper-parameters
#

input_size = len(x_train[0])
hidden1_size = 400
hidden2_size = 200
output_size = 10
batch_size = 2
step_size = .01
step_size_scaling = 1.0
counter = 0
dropout = True
dropout_rate = .1
if not dropout:
    dropout_rate = 0.0

print("****************************")
print("Model Hyper Parameters")
print("Training Set Size--> " + str(len(x_train)))
print("Input Size --> " + str(len(x_train[0])))
print("Hidden Layer #1 size --> " + str(hidden1_size))
print("Hidden Layer #2 size --> " + str(hidden2_size))
print("Batch Size --> " + str(batch_size))
print("Step Size & Step Size scaling --> " + str(step_size) + "    " + str(step_size_scaling))
print("Dropout and Rate --> " + str(dropout) + "     " + str(dropout_rate))

#
# define layer_map
#

layer_map = {'fc_layer': FullyConnectedLayer,
             'classify_layer': OutputLayer}

#
# define model within layer_config
#

layer_config = [{'fc_layer': {
                 'layer_size': 400}},
                {'fc_layer': {
                 'layer_size': 200}},
                {'classify_layer': {'layer_size': 10}}]

num_layrs = len(layer_config)

#
# initialize layers
#

layers = []

for layer_spec in layer_config:

    layer_class = layer_map[list(layer_spec.keys())[0]]
    layer_kwargs = list(layer_spec.values())[0]
    layer = layer_class(input_size, **layer_kwargs)
    layers.append(layer)
    print(layer)
    input_size = layer.layer_size

print(layers)

#
# begin training process
#

print(len(x_train))

for epoch in range(0, 10):

    print("epoch #" + str(epoch))
    step_size *= step_size_scaling

    total_batches = len(x_train) / batch_size
    batch_location = 0

    for i in range(0, len(x_train), batch_size):

        x_train_batch = x_train[i:i+batch_size, :]
        y_train_batch = y_train[i:i+batch_size]

        input_layer = x_train_batch
        prev_activation = x_train_batch

        #
        # Forward learn
        #

        for layer in layers:
            input_layer = prev_activation
            if isinstance(layer, FullyConnectedLayer):
                layer_activation = layer.feedforward(input_layer)

            if isinstance(layer, OutputLayer):
                output_activation, output_exp = layer.outputlayer(input_layer, y_train)

            prev_activation = layer_activation



        # Error at Output Layer

        loss_slope = (output_activation - output_exp) / output_activation.shape[0]

        # Back prop thru each layer

        exit()

        for layer in reversed(layers):

            if isinstance(layer, OutputLayer):
                error_next, weights, bias = layer.backprop()





        error_hidden2, output_layer.weights, output_layer.bias = output_layer.backprop(
            layer2_activation, loss_slope, output_layer.weights, output_layer.bias)

        error_hidden1, layer2.weights, layer2.bias = layer2.backprop(
            layer1_activation, error_hidden2, layer2.weights, layer2.bias)

        dummy_error, layer1.weights, layer1.bias = layer1.backprop(
            x_train_batch, error_hidden1, layer1.weights, layer1.bias)

    #
    # Run Test Set to assess model
    #

    layer1_activation = layer1.fftest(x_test)
    layer2_activation = layer2.fftest(layer1_activation)
    output_activation, output_exp = output_layer.outputlayer(layer2_activation, y_test)


    # print("+++++++++++++++++++++++++++++++++++++++++++++++++")
    #
    # print(output_activation.shape)
    # print(output_activation)
    # print(y_test.shape)
    # print(y_test)

    print("Epoch Accuracy ---------> " + str(accuracy(output_activation, y_test)))
