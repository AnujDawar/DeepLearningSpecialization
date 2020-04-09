import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
# from PIL import Image
from lr_utils import load_dataset

###################################################################################
#####################    reshape training/testing set      ########################
###################################################################################

def flat_matrix(matrix):
    matrix_flatten = matrix.reshape(matrix.shape[0], -1).T
    return matrix_flatten

###################################################################################
########################    normalize train/test set      #########################
###################################################################################

def normalize_matrix(matrix):
    '''
    Dividing each pixel by 255 coz range would be 1 to 255
    '''
    normalized_matrix = matrix/255
    return normalized_matrix

def initialize_with_zeros(dimension):
    w = np.zeros((dimension, 1))
    b = 0

    assert(w.shape == (dimension, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


##########  forward and backward propagation        ########
def propagate(w, b, X, Y):
    '''
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    '''

    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)

    cost = -(1/m) * np.sum(np.multiply(Y, np.log(A)) + np.multiply(1-Y, np.log(1-A)))

    cost = np.squeeze(cost)

    dw = (1/m) * np.dot(X, (A-Y).T)
    db = (1/m) * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db }

    return grads, cost

########################### gradient descent algo ###############################

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    '''
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
            
    '''

    costs = []

    for i in range(num_iterations):

        grad, cost = propagate(w, b, X, Y)

        dw = grad["dw"]
        db = grad["db"]

        w -= (learning_rate * dw)
        b -= (learning_rate * db)

        if(i % 100 == 0):
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}


    grads = { "dw": dw,
              "db": db }

    return params, grads, costs


################## prediction #################

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(m):
        Y_prediction[0, i] = A[0, i] > 0.5

    return Y_prediction


###########     final model     #############

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """

    w, b = initialize_with_zeros(X_train.shape[0])

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = params["w"]
    b = params["b"]

    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d



##############      Main Code   #################


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]

print("Number of training examples " + str(m_train))
print("Number of testing examples " + str(m_test))
print("Length/Height of each image : " + str(train_set_x_orig.shape[1]))
print("Shape of each image : " + str(train_set_x_orig.shape[1]) + ", " + str(train_set_x_orig.shape[2]) + ", " + str(train_set_x_orig.shape[3]))
print("Shape of training set X: " + str(train_set_x_orig.shape))
print("Shape of testing set X : " + str(test_set_x_orig.shape))
print("Shape of training set Y: " + str(train_set_y.shape))
print("Shape of testing set Y: " + str(test_set_y.shape))

train_x_flat = flat_matrix(train_set_x_orig)
test_x_flat = flat_matrix(test_set_x_orig)

train_set_x = normalize_matrix(train_x_flat)
test_set_x = normalize_matrix(test_x_flat)

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)


# Plot learning curve (with costs)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


##  Choice of learning rates

learning_rates = [0.01, 0.001, 0.005, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
