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
    return np.maximum(x, 0)


def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))


def softmax_loss(activation, target):
    sum_error = 0.0
    for i in range(len(activation)):
        sum_error += target[i]*np.log(activation[i])
    # print("error = " + str(sum_error))
    return sum_error*(-1.0)

def cross_entropy_softmax_loss_array(softmax_probs_array, y_onehot):

    indices = np.argmax(y_onehot).astype(int)
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


def initialize_weights(input_size, layer_size):
    # weights = np.random.uniform(size=(input_size,layer_size)) / layer_size
    weights = np.random.uniform(-1, 1, size=(input_size,layer_size))
    # print(weights)
    return weights


def train_layer(input_layer, weights):
    layer_inputs = np.dot(input_layer, weights)
    return layer_inputs


def create_exp(num):
    output = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    output[int(num)] = 1
    return(output)




#
#
# main program
#
#

# import and organze training and test data

x_train = get_MNIST_data("mnist_train_100.csv")
x_test = get_MNIST_data("mnist_test_10.csv")

#x_train = get_MNIST_data("mnist_train_5000.csv")
# x_train = get_MNIST_data("mnist_train_20000.csv")
#x_test = get_MNIST_data("mnist_test.csv")


y_train, x_train = extract_labels(x_train)
y_test, x_test = extract_labels(x_test)

# display_image(x_train[0], 28)

# define network and initialize weighting functions

input_size = len(x_train[0])

print(str(len(x_train)) + "             " + str(len(x_train[0])))

hidden1_size = 400
output_size = 10
batch_size = 1
step_size = .01
counter = 0

layer1_weights = initialize_weights(input_size, hidden1_size)
output_weights = initialize_weights(hidden1_size, output_size)

# begin training process

train = 0

print(len(x_train))

for epoch in range(0, 40):

    print("epoch #" + str(epoch))

    for i in range(0,len(x_train)):

        # Forward learn

        x_train_norm = x_train[i]/255.0
        layer1_input = np.dot(x_train_norm, layer1_weights)
        # layer1_activation = sigmoid(layer1_input)
        layer1_activation = relu_activation(layer1_input)
        output_input = np.dot(layer1_activation, output_weights)
        output_activation = softmax(output_input)
        output_exp = create_exp(y_train[i])
        # cross_entropy_loss2 = cross_entropy_softmax_loss_array(output_activation,output_exp)

        # print result and error

        # print("Tgt -> " + str(y_train[i]) + "  Pre -> " + str(predicted) + " TSLE -> "
        #       + str(counter) + " epoc " + str(i) + " loss = " + str(cross_entropy_loss2))

        # Back propagate Output Weights

        loss_slope = output_activation - output_exp

        error_hidden = np.dot(loss_slope, output_weights.T)
        error_hidden[layer1_activation <= 0] = 0
        # error_hidden = sigmoid_back(error_hidden)
        gradient_layer2_weights = np.dot(layer1_activation.T, error_hidden)

        gradient_layer2_weights = np.outer(layer1_activation.T, loss_slope)
        gradient_layer1_weights = np.outer(x_train_norm.T, error_hidden)

        layer1_weights -= step_size*gradient_layer1_weights
        output_weights -= step_size*gradient_layer2_weights

        # display_image(x_train[5],28)

    #
    # Run Test Set to assess model
    #

    counter_correct = 0
    counter_incorrect = 0

    for i in range(0,len(x_test)):

        x_test_norm = x_test[i]/255.0
        layer1_input = np.dot(x_test_norm, layer1_weights)
        # layer1_activation = sigmoid(layer1_input)
        layer1_activation = relu_activation(layer1_input)
        output_input = np.dot(layer1_activation, output_weights)
        output_activation = softmax(output_input)

        output_exp = create_exp(y_test[i])
        cross_entropy_loss2 = cross_entropy_softmax_loss_array(output_activation,output_exp)
        predicted = np.argmax(output_activation)
        if (int(predicted)==y_test[i]):
            counter_correct += 1


    #
    # Final Assesment
    #

    accuracy = counter_correct / float(len(x_test))

    print("Final Accuracy --> " + str(accuracy))












