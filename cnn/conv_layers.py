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

"""Implementation of convolution forward and backward pass"""

import numpy as np
from scipy import signal

def conv_layer_forward(input_layer, weight, bias, pad_size=1, stride=1):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of M data points, each with C channels, height H and
    width W. We convolve each input with C_o different filters, where each filter
    spans all C_i channels and has height H_w and width W_w.

    Args:
        input_alyer: The input layer with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        output_layer: The output layer with shape (batch_size, num_filters, height_y, width_y)
    """
    # TODO: Task 2.1
    output_layer = None # Should have shape (batch_size, num_filters, height_y, width_y)

    (batch_size, channels_x, height_x, width_x) = input_layer.shape
    (num_filters, channels_w, height_w, width_w) = weight.shape
    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")

    # with zero-padding for x
    height_y = int((height_x + 2*pad_size - height_w)/stride +1)
    width_y = int((width_x + 2*pad_size - width_w)/stride +1)
    output_layer = np.zeros((batch_size, num_filters, height_y, width_y))
    K = pad_size # where H == W == 2K+1 for the filter
    Std = stride

    # with zero-padding for x
    for i in range(batch_size):
        for j in range(num_filters):
            out = np.zeros_like(output_layer[i,j,:,:])
            for cc in range(channels_x):
                x = input_layer[i,cc,:,:]
                filter_j = weight[j,cc,:,:]

                out_cc = np.zeros_like(out)
                x_pad = np.pad(x,(K,K),'constant', constant_values=(0, 0))

                """
                if(i==0 and j==0 and cc==0):
                    print("filter_j,",filter_j)
                    print("compute convolution....")
                    print("x_pad,", x_pad)
                """

                for p in range(0,height_x, Std):
                    for q in range(0,width_x, Std):
                        x_masked = x_pad[p:(p+height_w),q:(q+width_w)]
                        ypq = np.sum(x_masked*filter_j)
                        """
                        ypq = 0
                        for r in range(-1*K,(K+1)):
                            for s in range(-1*K,(K+1)):
                                if((i+j+cc+p+q)==0 ):
                                    print("i=",(1-r),", j=",(1-s),"filter[r,s]",filter_j[1-r,1-s])
                                if((i+j+cc)==0):
                                    print("i=",(p+K+r),", j=",(q+K+s),"x[r,s]",x_pad[p+K+r,q+K+s])
                                ypq += x_pad[p+K+r,q+K+s]*filter_j[1-r,1-s]
                        """
                        pp = int(p/Std)
                        qq = int(q/Std)

                        out_cc[pp,qq]=ypq
                out+=out_cc
            output_layer[i,j,:,:] = bias[j]+out
    return output_layer


def conv_layer_backward(output_layer_gradient, input_layer, weight, bias, pad_size=1):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Args:
        output_layer_gradient: Gradient of the loss L wrt the next layer y, with shape
            (batch_size, num_filters, height_y, width_y)
        input_layer: Input layer x with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        input_layer_gradient: Gradient of the loss L with respect to the input layer x
        weight_gradient: Gradient of the loss L with respect to the filters w
        bias_gradient: Gradient of the loss L with respect to the biases b
    """
    # TODO: Task 2.2
    input_layer_gradient, weight_gradient, bias_gradient = np.zeros_like(input_layer),\
                                                           np.zeros_like(weight),\
                                                           np.zeros_like(bias)

    batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
    batch_size, channels_x, height_x, width_x = input_layer.shape
    num_filters, channels_w, height_w, width_w = weight.shape

    assert num_filters == channels_y, (
        "The number of filters must be the same as the number of output layer channels")
    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")

    K = pad_size
    Std=1
    
    #bias_gradient
    bias_gradient = np.zeros_like(bias)
    #for i in range(batch_size): not necessary because the gradients are also summed over samples
    for j in range(num_filters):
        bias_gradient[j] += np.sum(output_layer_gradient[:,j,:,:])

    #weight_gradient
    weight_gradient = np.zeros_like(weight)
    for i in range(batch_size):
        for j in range(num_filters):
            for cc in range(channels_x):
                x_pad = np.pad(input_layer[i,cc,:,:],(K,K),'constant', constant_values=(0, 0))
                #x_pad = np.flip(np.pad(input_layer[i,cc,:,:],(K,K),'constant', constant_values=(0, 0)))
                for r in range(height_w):
                    for s in range(width_w):
                        x_mask = x_pad[r:(r+height_y),s:(s+width_y)]
                        weight_gradient[j,cc,r,s] += \
                            np.sum(output_layer_gradient[i,j,:,:]*x_mask)

    #input_layer_gradient
    input_padded = np.pad(input_layer, ((0,), (0,), (K,), (K,)), mode="constant", constant_values=0)
    input_gradient_padded = np.zeros_like(input_padded)

    for i in range(batch_size):
        for j in range(height_y):
            for k in range(width_y):
                input_gradient_padded[i, :, j*Std:j*Std+height_w, k*Std:k*Std+width_w] += \
                np.sum((weight[:, :, :, :] * (output_layer_gradient[i, :, j, k])[:, None, None, None]), axis=0)

    input_layer_gradient = input_gradient_padded[:, :, K:-K, K:-K]

    return input_layer_gradient, weight_gradient, bias_gradient


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad
