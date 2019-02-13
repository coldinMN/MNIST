from matplotlib import pyplot as plt
import numpy as np


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

    # if flag == 1:
    #     # print("test")
    #     for i in range(len(x)):
    #         if x[i] > 0:
    #             pass
    #         else:
    #             x[i] = 0
    # else:
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
    # print("softmax activation")
    # print(np.exp(z)/np.sum(np.exp(z)))
    # print(z)
    y = np.exp(z)
    soft = y / np.sum(y, axis=1, keepdims=True)
    # print(len(z))
    # exit()
    return soft


def softmax_loss(activation, target):
    sum_error = 0.0
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


def train_layer(input_layer, weights):
    layer_inputs = np.dot(input_layer, weights)
    return layer_inputs


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


#
#
# main program
#
#

# import and organize training and test data

# x_train = get_MNIST_data("mnist_train_100.csv")
# x_test = get_MNIST_data("mnist_test_10.csv")

# x_train = get_MNIST_data("mnist_train_5000.csv")
# x_train = get_MNIST_data("mnist_train_20000.csv")
x_train = get_MNIST_data("mnist_train.csv")
x_test = get_MNIST_data("mnist_test.csv")


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
batch_size = 1
step_size = .01
step_size_scaling = 1.0
counter = 0

layer1_weights, layer1_bias = initialize_weights(input_size, hidden1_size, 2)
layer2_weights, layer2_bias = initialize_weights(hidden1_size, hidden2_size, 2)
output_weights, output_bias = initialize_weights(hidden2_size, output_size, 2)

# begin training process

train = 0

print(len(x_train))

for epoch in range(0, 4000):

    print("epoch #" + str(epoch))
    step_size *= step_size_scaling

    total_batches = len(x_train) / batch_size
    batch_location = 0
    print("Total number of batches : " + str(total_batches))

    for i in range(0, len(x_train), batch_size):
        # print(i)

        x_train_batch = x_train[i:i+batch_size, :]
        y_train_batch = y_train[i:i+batch_size]

        # Forward learn

        # Normalize input data

        input_layer = x_train_batch/255.0

        # input to layer 1

        layer1_input = np.dot(input_layer, layer1_weights) + layer1_bias
        layer1_activation = relu_activation(layer1_input, 0)

        # hidden layer1 to hidden layer2

        layer2_input = np.dot(layer1_activation, layer2_weights) + layer2_bias
        layer2_activation = relu_activation(layer2_input, 0)

        # hidden layer2 to output layer

        output_input = np.dot(layer2_activation, output_weights) + output_bias
        output_activation = softmax(output_input)

        output_exp = create_exp(y_train_batch, batch_size)

        # Back propagate Output Weights

        # Error at output layer

        loss_slope = (output_activation - output_exp) / output_activation.shape[0]

        # Error at hidden layer2

        error_hidden2 = np.dot(loss_slope, output_weights.T)
        error_hidden2 *= relu_back(layer2_activation)



        # Error at hidden layer1

        error_hidden1 = np.dot(error_hidden2, layer2_weights.T)
        error_hidden1 *= relu_back(layer1_activation)

        # at this point we have pushed error to both hidden layers

        gradient_output_weights = np.dot(layer2_activation.T, loss_slope)
        gradient_layer2_weights = np.dot(layer1_activation.T, error_hidden2)
        gradient_layer1_weights = np.dot(input_layer.T, error_hidden1)

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

    input_layer = x_test / 255
    layer1_input = np.dot(input_layer, layer1_weights) + layer1_bias
    layer1_activation = relu_activation(layer1_input, 0)
    layer2_input = np.dot(layer1_activation, layer2_weights) + layer2_bias
    layer2_activation = relu_activation(layer2_input, 0)
    output_input = np.dot(layer2_activation, output_weights) + output_bias
    output_activation = softmax(output_input)

    # print("+++++++++++++++++++++++++++++++++++++++++++++++++")
    #
    # print(output_activation.shape)
    # print(output_activation)
    # print(y_test.shape)
    # print(y_test)

    print("Epoch Accuracy ---------> " + str(accuracy(output_activation, y_test)))
    # print(softmax_loss(output_activation, y_test))


