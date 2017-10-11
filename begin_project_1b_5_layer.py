import time
import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# scale and normalize input data
def scale(X, X_min, X_max):
    return (X - X_min) / (X_max - X_min)

def normalize(X, X_mean, X_std):
    return (X - X_mean) / X_std

def shuffle_data(samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    # print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

def init_bias(n=1):
    return (theano.shared(np.zeros(n) if n != 1 else 0., theano.config.floatX))

def init_weights(n_in=1, n_out=1, logistic=True):
    W_values = np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                 high=np.sqrt(6. / (n_in + n_out)),
                                 size=(n_in, n_out))
    if logistic == True:
        W_values *= 4
    return (theano.shared(W_values, theano.config.floatX))

def set_bias(b, n=1):
    b.set_value(np.zeros(n) if n != 1 else 0.)

def set_weights(w, n_in=1, n_out=1, logistic=True):
    W_values = np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                 high=np.sqrt(6. / (n_in + n_out)),
                                 size=(n_in, n_out))
    if logistic == True:
        W_values *= 4
    w.set_value(W_values)

def preprocess_data(dataset):
    # read and divide data into test and train sets
    cal_housing = np.loadtxt(dataset, delimiter=',')
    X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
    Y_data = (np.asmatrix(Y_data)).transpose()

    X_data, Y_data = shuffle_data(X_data, Y_data)

    # separate train and test data
    m = 3 * X_data.shape[0] // 10
    testX, testY = X_data[:m], Y_data[:m]
    trainX, trainY = X_data[m:], Y_data[m:]

    # scale and normalize data
    trainX_max, trainX_min = np.max(trainX, axis=0), np.min(trainX, axis=0)
    testX_max, testX_min = np.max(testX, axis=0), np.min(testX, axis=0)

    trainX = scale(trainX, trainX_min, trainX_max)
    testX = scale(testX, testX_min, testX_max)

    trainX_mean, trainX_std = np.mean(trainX, axis=0), np.std(trainX, axis=0)
    testX_mean, testX_std = np.mean(testX, axis=0), np.std(testX, axis=0)

    trainX = normalize(trainX, trainX_mean, trainX_std)
    testX = normalize(testX, testX_mean, testX_std)

    return trainX, testX, trainY, testY

def initialize_weights_bias():
    w_o = init_weights(no_hidden3, no_output, False)
    w_h1 = init_weights(no_features, no_hidden1)
    w_h2 = init_weights(no_hidden1, no_hidden2)
    w_h3 = init_weights(no_hidden2, no_hidden3)
    b_o = init_bias(no_output)
    b_h1 = init_bias(no_hidden1)
    b_h2 = init_bias(no_hidden2)
    b_h3 = init_bias(no_hidden3)
    return w_o, w_h1, w_h2, w_h3, b_o, b_h1, b_h2, b_h3

def reset_weights():
    set_weights(w_o, no_hidden3, no_output, False)
    set_weights(w_h1, no_features, no_hidden1)
    set_weights(w_h2, no_hidden1, no_hidden2)
    set_weights(w_h3, no_hidden2, no_hidden3)
    set_bias(b_o, no_output)
    set_bias(b_h1, no_hidden1)
    set_bias(b_h2, no_hidden2)
    set_bias(b_h3, no_hidden3)

def create_nn():
    x = T.matrix('x')  # data sample
    d = T.matrix('d')  # desired output
    no_samples = T.scalar('no_samples')

    # Define mathematical expression:
    h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
    h2_out = T.nnet.sigmoid(T.dot(h1_out, w_h2) + b_h2)
    h3_out = T.nnet.sigmoid(T.dot(h2_out, w_h3) + b_h3)
    y = T.dot(h3_out, w_o) + b_o

    cost = T.abs_(T.mean(T.sqr(d - y)))
    accuracy = T.mean(d - y)

    # define gradients
    dw_o, db_o, dw_h, db_h, dw_h2, db_h2, dw_h3, db_h3 = T.grad(cost, [w_o, b_o, w_h1, b_h1, w_h2, b_h2, w_h3, b_h3])

    train = theano.function(
        inputs=[x, d],
        outputs=cost,
        updates=[[w_o, w_o - alpha * dw_o],
                 [b_o, b_o - alpha * db_o],
                 [w_h1, w_h1 - alpha * dw_h],
                 [b_h1, b_h1 - alpha * db_h],
                 [w_h2, w_h2 - alpha * dw_h2],
                 [b_h2, b_h2 - alpha * db_h2],
                 [w_h3, w_h3 - alpha * dw_h3],
                 [b_h3, b_h3 - alpha * db_h3]],
        allow_input_downcast=True
    )

    test = theano.function(
        inputs=[x, d],
        outputs=[y, cost, accuracy],
        allow_input_downcast=True
    )

    return train, test

def run_nn(train, test, batch_size, trainX, trainY, testX, testY, epochs):
    min_error = 1e+15
    best_iter = 0
    best_w_o = np.zeros(no_hidden3)
    best_w_h1 = np.zeros([no_features, no_hidden1])
    best_w_h2 = np.zeros([no_hidden1, no_hidden2])
    best_w_h3 = np.zeros([no_hidden2, no_hidden3])
    best_b_o = 0
    best_b_h1 = np.zeros(no_hidden1)
    best_b_h2 = np.zeros(no_hidden2)
    best_b_h3 = np.zeros(no_hidden3)

    train_cost = np.zeros(epochs)
    test_cost = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)

    reset_weights()

    # train with best value
    for iter in range(epochs):
        if iter % 500 == 0:
            print("Iter:", iter)

        trainX, trainY = shuffle_data(trainX, trainY)
        for start, end in zip(range(0, len(trainX), batch_size),
                              range(batch_size, len(trainX), batch_size)):
            train_cost[iter] += train(trainX[start:end], trainY[start:end])
        train_cost[iter] /= (len(trainX) // batch_size)
        pred, test_cost[iter], test_accuracy[iter] = test(testX, testY)

        if test_cost[iter] < min_error:
            best_iter = iter
            min_error = test_cost[iter]
            best_w_o = w_o.get_value()
            best_w_h1 = w_h1.get_value()
            best_w_h2 = w_h2.get_value()
            best_w_h3 = w_h3.get_value()
            best_b_o = b_o.get_value()
            best_b_h1 = b_h1.get_value()
            best_b_h2 = b_h2.get_value()
            best_b_h3 = b_h3.get_value()

    # set weights and biases to values at which performance was best
    w_o.set_value(best_w_o)
    b_o.set_value(best_b_o)
    w_h1.set_value(best_w_h1)
    b_h1.set_value(best_b_h1)
    w_h2.set_value(best_w_h2)
    b_h2.set_value(best_b_h2)
    w_h3.set_value(best_w_h3)
    b_h3.set_value(best_b_h3)

    best_pred, best_cost, best_accuracy = test(testX, testY)

    print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d' % (best_cost, best_accuracy, best_iter))

    # Plots
    plt.figure()
    plt.plot(range(epochs), train_cost, label='train error')
    plt.plot(range(epochs), test_cost, label='test error')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Squared Error')
    plt.title('Training and Test Errors at Alpha = %.3f' % alpha.get_value())
    plt.legend()
    plt.savefig('p_1b_mse.png')
    plt.show()

    plt.figure()
    plt.plot(range(epochs), test_accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.savefig('p_1b_accuracy.png')
    plt.show()


# Scale, normalize, and separate data to train/test
np.random.seed(10)
epochs = 100
batch_size = 32
no_hidden1 = 60  # num of neurons in hidden layer 1
no_hidden3 = no_hidden2 = 20
learning_rate = 0.0001

trainX, testX, trainY, testY = preprocess_data('cal_housing.data')

no_features = trainX.shape[1]
no_output = trainY.shape[1]

alpha = theano.shared(learning_rate, theano.config.floatX)

w_o, w_h1, w_h2, w_h3, b_o, b_h1, b_h2, b_h3 = initialize_weights_bias()
train, test = create_nn()
run_nn(train, test, batch_size, trainX, trainY, testX, testY, epochs)