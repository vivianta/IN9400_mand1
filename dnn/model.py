#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of mandatory assignment 1 in                                             #
# IN5400 - Machine Learning for Image analysis                                  #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2019.02.12                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    # TODO: Task 1.1
    
    dims = conf.get('layer_dimensions')
    L0 = dims[0]
    L1 = dims[1]
    L2 = dims[2]
    L3 = dims[3]
    sigma1 = 2/L0
    sigma2 = 2/L1
    sigma3 = 2/L2
    W1 = np.random.normal(0,sigma1,(L0,L1))
    W2 = np.random.normal(0,sigma2,(L1,L2))
    W3 = np.random.normal(0,sigma3,(L2,L3))
    B1 = np.zeros((L1,1))
    B2 = np.zeros((L2,1))
    B3 = np.zeros((L3,1))
    

    params = {
        'W_1': W1, 
        'W_2': W2,
        'W_3': W3,
        'b_1': B1,
        'b_2': B2,
        'b_3': B3}

    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.2 a)
    if activation_function == 'relu':
        acv = Z*(Z>0)
        return acv
    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    e_x = np.exp(Z - np.max(Z, axis=0))
    return e_x / e_x.sum(axis=0)


def forward(conf, X_batch, params, is_training):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    # TODO: Task 1.2 c)
    features = dict()
    layers = conf['layer_dimensions']
    num_layers = len(layers)
    n_x, m = X_batch.shape
    #print('forward poin a, params keys:',params.keys())
    #input layer
    Z_prev = X_batch.transpose()
    features['A_0'] = X_batch
    
    #hiden layer
    for idx in range(1,(np.size(layers)-1)):
        B = np.repeat(params['b_'+str(idx)].transpose(), m, axis=0)
        Z = Z_prev.dot(params['W_'+str(idx)])+B
        L = activation(Z, 'relu')
        Z_prev = L
        if is_training:
            features['Z_'+str(idx)] = Z.transpose()
            features['A_'+str(idx)] = L.transpose()
            
    # output layer
    #print('forward point b params keys:',params.keys())
    B = np.repeat(params['b_'+str(num_layers-1)].transpose(), m, axis=0)
    Z = Z_prev.dot(params['W_'+str(num_layers-1)])+B
    Y_proposed = softmax(Z.transpose())
    if is_training:
            features['Z_'+str(num_layers-1)] = Z.transpose()
            features['A_'+str(num_layers-1)] = Y_proposed
    #print('forward point c params keys:',params.keys())        
    return Y_proposed, features


def cross_entropy_cost(Y_proposed, Y_reference):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    # TODO: Task 1.3
    n_y, m = Y_reference.shape
    cost = -(1/m)*np.dot(np.dot(np.ones(shape=(1,n_y)),Y_reference*np.log(Y_proposed)),np.ones(shape=(m,1)))
    max_probability_indices = np.vstack((np.argmax(Y_proposed, axis=0), np.arange(m)))
    num_correct = np.sum(Y_reference[max_probability_indices[0], max_probability_indices[1]])

    return cost.item(), num_correct


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.4 a)
    dg = np.zeros_like(Z)
    if activation_function == 'relu':
        dg[Z>=0]=1
        return dg
    else:
        print("Error: Unimplemented derivative of activation function: {}", activation_function)
        return None


def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    # TODO: Task 1.4 b)
    n_y, m = Y_proposed.shape
    layers = conf['layer_dimensions']
    num_layers = len(layers)
    grad_params = dict()
    
    #output layer
    idx = num_layers-1
    J = Y_proposed - Y_reference
    A_prev = features['A_'+str(idx -1)]
    grad_W = np.dot(A_prev, J.transpose())/m
    grad_b = np.dot(J, np.ones((m,1)))/m
    grad_params['grad_W_'+str(idx)] = grad_W
    grad_params['grad_b_'+str(idx)] = grad_b
    J_prev = J
    
    #hiden layer
    for idx in range((num_layers-2),0,-1):
        J = activation_derivative(features["Z_" + str(idx)], activation_function="relu")*\
            np.dot(params['W_'+str(idx+1)], J_prev)
        A_prev = features['A_'+str(idx-1)]
        J_prev = J
        
        grad_W = np.dot(A_prev, J.transpose())/m
        grad_b = np.dot(J, np.ones((m,1)))/m
        grad_params['grad_W_'+str(idx)] = grad_W
        grad_params['grad_b_'+str(idx)] = grad_b
        
    return grad_params


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        params: Updated parameter dictionary.
    """
    # TODO: Task 1.5
    lamda = conf['learning_rate']
    num_layers = int(len(params.keys())/2)+1
    updated_params = dict()
    
    for idx in range(1, num_layers):
        updated_params['W_'+str(idx)] = params['W_'+str(idx)] - lamda * grad_params['grad_W_'+str(idx)]
        updated_params['b_'+str(idx)] = params['b_'+str(idx)] - lamda * grad_params['grad_b_'+str(idx)]
    
    return updated_params
