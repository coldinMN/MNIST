from matplotlib import pyplot as plt
import numpy as np


def get_MNIST_data(filename):

    print("loading file")
    file_to_load = np.genfromtxt(filename, delimiter=',')
    print("file loaded")
    return file_to_load


def extract_labels(dataset):

    # print(dataset)
    # print("********************")
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


def relu_activation(x):
    # print("relu_input")
    # print(x)
    # print("relu activation")
    # print(np.maximum(x, 0))
    return np.maximum(x, 0)


def relu_back(x):
    for i in range(0, len(x)):
        if x[i] > 0:
            x[i] = 1
        else:
            x[i] = 0
    return x


def tanh(x):
    return (2 / (1 + np.exp(-2*x))) - 1.0


def tanh_back(x):
    return 1 - x ** 2


def softmax(z):
    # print("softmax activation")
    # print(np.exp(z)/np.sum(np.exp(z)))
    y = np.exp(z)
    return y / np.sum(y)


def softmax_loss(activation, target):
    sum_error = 0.0
    for i in range(len(activation)):
        sum_error += target[i]*np.log(activation[i])
    # print("error = " + str(sum_error))
    return sum_error*(-1.0)


def cross_entropy_softmax_loss_array(softmax_probs_array, y_onehot):
    log_preds = np.multiply(y_onehot, np.log(softmax_probs_array))
    loss = -1.0 * np.sum(log_preds)
    return loss


def relu(x):
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                pass  # do nothing since it would be effectively replacing x with x
            else:
                x[i][k] = 0
    return x


def initialize_weights(input_size, layer_size, type):
    if type == 1:
        print("Uniform Distrution")
        weights = np.random.uniform(-1, 1, size=(input_size, layer_size))
    elif type == 2:
        sigma = np.sqrt(2 / input_size)
        print("Normal dist " + str(sigma))
        weights = np.random.normal(0.0, sigma, size=(input_size, layer_size))
        # print(weights)
    bias = np.zeros(layer_size)
    # print("Bias")
    # print(bias)

    return weights, bias


def train_layer(input_layer, weights):
    layer_inputs = np.dot(input_layer, weights)
    return layer_inputs


def create_exp(num):
    output = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    output[int(num)] = 1
    # print(output)
    return output


#
#
# main program
#
#

# import and organize training and test data

x_train = get_MNIST_data("mnist_train_100.csv")
x_test = get_MNIST_data("mnist_test_10.csv")

# x_train = get_MNIST_data("mnist_train_5000.csv")
# x_train = get_MNIST_data("mnist_train_20000.csv")
# x_test = get_MNIST_data("mnist_test.csv")


y_train, x_train = extract_labels(x_train)
y_test, x_test = extract_labels(x_test)


print(y_train)
# display_image(x_train[0], 28)

# define network and initialize weighting functions

input_size = len(x_train[0])

print(str(len(x_train)) + "             " + str(len(x_train[0])))

hidden1_size = 400
hidden2_size = 200
output_size = 10
batch_size = 10
step_size = .0105
step_size_scaling = .95
counter = 0

layer1_weights, layer1_bias = initialize_weights(input_size, hidden1_size, 2)
layer2_weights, layer2_bias = initialize_weights(hidden1_size, hidden2_size, 2)
output_weights, output_bias = initialize_weights(hidden2_size, output_size, 2)
# print(output_weights)

# begin training process

train = 0

print(len(x_train))

for epoch in range(0, 40):

    print("epoch #" + str(epoch))
    step_size *= step_size_scaling
    print("Step Size = " + str(step_size))

    total_batches = len(x_train / step_size)
    print("Total number of batches : " + str(total_batches))
    exit()

    for i in range(0,len(x_train)):

        # Forward learn

        # input to layer 1

        input_layer = x_train[i]/255.0
        layer1_input = np.dot(input_layer, layer1_weights) + layer1_bias
        layer1_activation = relu_activation(layer1_input)
        # layer1_activation = sigmoid(layer1_input)

        # hidden layer1 to hidden layer2

        layer2_input = np.dot(layer1_activation, layer2_weights) + layer2_bias
        layer2_activation = relu_activation(layer2_input)

        # hidden layer2 to output layer

        output_input = np.dot(layer2_activation, output_weights) + output_bias
        output_activation = softmax(output_input)
        output_exp = create_exp(y_train[i])

        # Back propagate Output Weights

        # Error at output layer

        loss_slope = output_activation - output_exp


        # Error at hidden layer2

        error_hidden2 = np.dot(loss_slope, output_weights.T)
        error_hidden2 *= relu_back(layer2_activation)

        # Error at hidden layer1

        error_hidden1 = np.dot(error_hidden2, layer2_weights.T)
        error_hidden1 *= relu_back(layer1_activation)

        # at this point we have pushed error to both hidden layers

        gradient_output_weights = np.outer(layer2_activation.T, loss_slope)
        gradient_layer2_weights = np.outer(layer1_activation.T, error_hidden2)
        gradient_layer1_weights = np.outer(input_layer.T, error_hidden1)

        gradient_output_bias = np.sum(loss_slope)
        gradient_layer2_bias = np.sum(error_hidden2)
        gradient_layer1_bias = np.sum(error_hidden1)

        layer1_weights -= step_size*gradient_layer1_weights
        layer2_weights -= step_size*gradient_layer2_weights
        output_weights -= step_size*gradient_output_weights

        output_bias -= step_size*gradient_output_bias
        layer2_bias -= step_size*gradient_layer2_bias
        layer1_bias -= step_size*gradient_layer1_bias


        # display_image(x_train[5],28)

    #
    # Run Test Set to assess model
    #

    counter_correct = 0
    counter_incorrect = 0

    for i in range(0,len(x_test)):

        input_layer = x_test[i]/255.0
        layer1_input = np.dot(input_layer, layer1_weights)
        layer1_activation = relu_activation(layer1_input)
        # layer1_activation = sigmoid(layer1_input)

        # hidden layer1 to hidden layer2

        layer2_input = np.dot(layer1_activation, layer2_weights)
        layer2_activation = relu_activation(layer2_input)

        # hidden layer2 to output layer

        output_input = np.dot(layer2_activation, output_weights)
        output_activation = softmax(output_input)
        output_exp = create_exp(y_test[i])
        cross_entropy_loss2 = cross_entropy_softmax_loss_array(output_activation, output_exp)
        predicted = np.argmax(output_activation)
        if int(predicted) == y_test[i]:
            counter_correct += 1


    #
    # Final Assessment
    #

    accuracy = counter_correct / float(len(x_test))

    print("Final Accuracy --> " + str(accuracy))

