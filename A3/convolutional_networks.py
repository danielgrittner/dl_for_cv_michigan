"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import torch.nn.functional as func
import random
from eecs598 import Solver
from a3_helper import svm_loss, softmax_loss
from fully_connected_networks import *

def hello_convolutional_networks():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from convolutional_networks.py!')


class Conv(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
      
    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ##############################################################################
    # TODO: Implement the convolutional forward pass.                            #
    # Hint: you can use the function torch.nn.functional.pad for padding.        #
    # Note that you are NOT allowed to use anything in torch.nn in other places. #
    ##############################################################################
    
    N, C, H, W = x.size()
    F, _, HH, WW = w.size()

    padding = conv_param['pad']
    stride = conv_param['stride']
    
    # Calculate output size
    H_prime = int(1 + (H + 2 * padding - HH) / stride)
    W_prime = int(1 + (W + 2 * padding - WW) / stride)
    out_shape = (N, F, H_prime, W_prime)

    # Allocate memory for output
    out = torch.zeros(out_shape, dtype=x.dtype)
    out = out.to(x.device)

    # Pad the input
    x_padded = func.pad(x, (padding, padding, padding, padding), 'constant', 0)

    # Compute the convolution
    for sample in range(N): # process each sample of the batch
      for kernel_idx in range(F): # process each kernel
        # Select the correct conv kernel
        kernel = w[kernel_idx]  # (C, HH, WW)
        bias = b[kernel_idx]

        in_row = 0
        for out_h in range(H_prime):
          in_col = 0
          for out_w in range(W_prime):
            # Initialize with bias
            out[sample, kernel_idx, out_h, out_w] = bias
            # Slice out input
            x_sliced = x_padded[sample, :, in_row:in_row + HH, in_col:in_col + WW]
            # Apply kernel
            out[sample, kernel_idx, out_h, out_w] += (kernel * x_sliced).sum()
            
            in_col += stride
          in_row += stride

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives. (N, F, H', W')
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################

    x, w, b, conv_param = cache

    stride = conv_param['stride']
    padding = conv_param['pad']

    N, F, H_prime, W_prime = dout.size()
    _, C, HH, WW = w.size()
    _, _, H, W = x.size()

    # Pad the input
    x_padded = func.pad(x, (padding, padding, padding, padding), 'constant', 0)

    ############### Gradient of the biases
    db = dout.sum(axis=(2, 3)).sum(axis=0)
    ###############

    ############### Gradient of the kernels
    # The gradient of the kernels can be calculated by convoluting dout over x
    dw = torch.zeros_like(w, dtype=w.dtype) # (F, C, HH, WW)

    for sample_id in range(N): # Process sample-by-sample
      # Input of the current sample
      sample_input = x_padded[sample_id]# (C, H + 2 * pad, W + 2 * pad)
      # dout of the current sample
      sample_dout = dout[sample_id] # (F, H', W')

      for kernel_id in range(F): # Process kernel-by-kernel
        # Calculate the gradients for the current kernel
        sample_dout_for_current_kernel = sample_dout[kernel_id] # (H', W')

        # Each kernel: (C, HH, WW)
        for c in range(C): # Process each channel
          current_input = sample_input[c] # (H + 2 * pad, W + 2 * pad)

          # TODO: Can we rewrite this into more efficient, compact code? ==> current_input[from:to:stride, from:to:stride] ? Would this work?
          for dw_h in range(HH):
            for dw_w in range(WW):

              current_input_row = dw_h
              for dout_row in range(H_prime):
                current_input_col = dw_w
                for dout_col in range(W_prime):
                  dw[kernel_id, c, dw_h, dw_w] += current_input[current_input_row, current_input_col] * \
                                                sample_dout_for_current_kernel[dout_row, dout_col]
                  current_input_col += stride
                current_input_row += stride
    ###############

    ############### Gradient of x
    # dx with temporary padding
    H_padding = H + 2 * padding
    W_padding = W + 2 * padding
    dx_temp = torch.zeros((N, C, H_padding, W_padding), dtype=x.dtype, device=dout.device)

    for sample_id in range(N): # Process sample-by-sample
      sample_dout = dout[sample_id] # (F, H', W')

      for kernel_id in range(F): # Process kernel-by-kernel
        sample_dout_for_current_kernel = sample_dout[kernel_id] # (H', W')
        kernel = w[kernel_id] # (C, HH, WW)

        for c in range(C):
          kernel_layer = kernel[c]

          dx_h = 0
          for h_prime in range(H_prime):
            dx_w = 0
            for w_prime in range(W_prime):
              dout_single = sample_dout_for_current_kernel[h_prime, w_prime]
              dx_temp[sample_id, c, dx_h:dx_h+HH, dx_w:dx_w+WW] += kernel_layer * dout_single

              dx_w += stride
            dx_h += stride


    # Remove padding again
    dx = dx_temp[:, :, padding:padding+H, padding:padding+W]
    
    ###############

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return dx, dw, db


class MaxPool(object):

  @staticmethod
  def forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here.

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max-pooling forward pass                              #
    #############################################################################
    
    N, C, H, W = x.size()

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_prime = int(1 + (H - pool_height) / stride)
    W_prime = int(1 + (W - pool_width) / stride)

    # Allocate output tensor
    out = torch.zeros((N, C, H_prime, W_prime), device=x.device, dtype=x.dtype)

    for sample_id in range(N):
      for channel in range(C):
        # For each (pool_height, pool_width) compute max
        h_in = 0
        for h in range(H_prime):
          w_in = 0
          for w in range(W_prime):
            out[sample_id, channel, h, w] = x[sample_id, channel, h_in:h_in+pool_height, w_in:w_in+pool_width].max()
            w_in += stride
          h_in += stride

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    cache = (x, pool_param)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max-pooling backward pass                             #
    #############################################################################

    x, pool_param = cache

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    N, C, H, W = x.size()

    # Allocate memory for output
    dx = torch.zeros_like(x)

    for sample_id in range(N):
      for channel in range(C):

        h_dout = 0
        for h_x in range(0, H, stride):
          w_dout = 0
          for w_x in range(0, W, stride):
            grad = dout[sample_id, channel, h_dout, w_dout]
            # Find the correct position which gets the gradient, i.e.
            # the max position from this area from the forward pass.
            area = x[sample_id, channel, h_x:h_x+pool_height, w_x:w_x+pool_width]
            h_target, w_target = (area == torch.max(area)).nonzero()[0]
            h_target, w_target = h_x + h_target.item(), w_x + w_target.item()
            # Add the gradient
            dx[sample_id, channel, h_target, w_target] += grad
            
            w_dout += 1
          h_dout += 1
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return dx


class BatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the PyTorch
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
    running_var = bn_param.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))

    out, cache = None, None
    if mode == 'train':
      #######################################################################
      # TODO: Implement the training-time forward pass for batch norm.      #
      # Use minibatch statistics to compute the mean and variance, use      #
      # these statistics to normalize the incoming data, and scale and      #
      # shift the normalized data using gamma and beta.                     #
      #                                                                     #
      # You should store the output in the variable out. Any intermediates  #
      # that you need for the backward pass should be stored in the cache   #
      # variable.                                                           #
      #                                                                     #
      # You should also use your computed sample mean and variance together #
      # with the momentum variable to update the running mean and running   #
      # variance, storing your result in the running_mean and running_var   #
      # variables.                                                          #
      #                                                                     #
      # Note that though you should be keeping track of the running         #
      # variance, you should normalize the data based on the standard       #
      # deviation (square root of variance) instead!                        # 
      # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
      # might prove to be helpful.                                          #
      #######################################################################
      
      # Build mean for each feature over samples
      sample_mean = x.mean(axis=0)

      # Build variance for each feature over samples
      shifted_x = x - sample_mean
      sample_var = (shifted_x * shifted_x).mean(axis=0)

      # Normalize x
      std = torch.sqrt(sample_var + eps)
      normalized_x = shifted_x / std

      # Update running_mean and running_var
      running_mean = momentum * running_mean + (1 - momentum) * sample_mean
      running_var = momentum * running_var + (1 - momentum) * sample_var

      out = gamma * normalized_x + beta

      cache = (normalized_x, std, shifted_x, sample_var, sample_mean, x, gamma, eps)

      #######################################################################
      #                           END OF YOUR CODE                          #
      #######################################################################
    elif mode == 'test':
      #######################################################################
      # TODO: Implement the test-time forward pass for batch normalization. #
      # Use the running mean and variance to normalize the incoming data,   #
      # then scale and shift the normalized data using gamma and beta.      #
      # Store the result in the out variable.                               #
      #######################################################################
      
      normalized_x = (x - running_mean) / torch.sqrt(running_var + eps)
      out = gamma * normalized_x + beta

      #######################################################################
      #                           END OF YOUR CODE                          #
      #######################################################################
    else:
      raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean.detach()
    bn_param['running_var'] = running_var.detach()

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    # Don't forget to implement train and test mode separately.               #
    ###########################################################################
    
    normalized_x, std, shifted_x, sample_var, sample_mean, x, gamma, eps = cache

    m = x.size()[0]

    # Compute gradient for beta
    dbeta = dout.sum(axis=0)

    # Compute gradient for gamma
    dgamma = (dout * normalized_x).sum(axis=0)

    # Compute gradient for x
    dx = torch.zeros_like(x)
    
    dx_temp1 = dout * gamma
    
    dx_temp2 = dx_temp1 / std
    dmean = - dx_temp2
    dx += dx_temp2
    
    dx_temp3 = dx_temp1 * (- shifted_x / (sample_var + eps))
    dx_temp4 = dx_temp3 * 0.5 / std
    dx_temp5 = dx_temp4.sum(axis=0) / m
    dx_temp6 = dx_temp5 * 2 * shifted_x

    dmean -= dx_temp6
    dx += dx_temp6

    dx += dmean.sum(axis=0) / m

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta

  @staticmethod
  def backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
    
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    
    normalized_x, std, shifted_x, sample_var, sample_mean, x, gamma, eps = cache

    m = x.size()[0]

    # Compute gradient for beta
    dbeta = dout.sum(axis=0)

    # Compute gradient for gamma
    dgamma = (dout * normalized_x).sum(axis=0)

    # Compute the gradient for x
    # Working solution:
    # dldy = gamma * dout
    # dldvar = (-0.5) * (dldy * shifted_x).sum(axis=0) * (sample_var + eps)**(-1.5)
    # dldmean = (-1) / std * dldy.sum(axis=0)
    # dx = 1/std * dldy + 2/m * shifted_x * dldvar + 1/m * dldmean

    # Single line:
    dx = (gamma / std) * (dout - 1/m * shifted_x / (sample_var + eps) * (dout * shifted_x).sum(axis=0) - 1/m * dout.sum(axis=0))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


class SpatialBatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (C,) giving running mean of features
      - running_var Array of shape (C,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################

    N, C, H, W = x.size()
    x_reshaped = x.permute(0, 2, 3, 1) # (N, H, W, C)
    x_reshaped = x_reshaped.reshape(-1, C) # (N * H * W, C)
    out, cache = BatchNorm.forward(x_reshaped, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C) # (N, H, W, C)
    out = out.permute(0, 3, 1, 2) # (N, C, H, W)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    
    N, C, H, W = dout.size()
    dout_reshaped = dout.permute(0, 2, 3, 1) # (N, H, W, C)
    dout_reshaped = dout_reshaped.reshape(-1, C) # (N * H * W, C)
    dx, dgamma, dbeta = BatchNorm.backward_alt(dout_reshaped, cache)
    dx = dx.reshape(N, H, W, C) # (N, H, W, C)
    dx = dx.permute(0, 3, 1, 2) # (N, C, H, W)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


################################################################################
################################################################################
#################   Fast Implementations and Sandwich Layers  ##################
################################################################################
################################################################################

class FastConv(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
    layer.weight = torch.nn.Parameter(w)
    layer.bias = torch.nn.Parameter(b)
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    cache = (x, w, b, conv_param, tx, out, layer)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    try:
      x, _, _, _, tx, out, layer = cache
      out.backward(dout)
      dx = tx.grad.detach()
      dw = layer.weight.grad.detach()
      db = layer.bias.grad.detach()
      layer.weight.grad = layer.bias.grad = None
    except RuntimeError:
      dx, dw, db = torch.zeros_like(tx), torch.zeros_like(layer.weight), torch.zeros_like(layer.bias)
    return dx, dw, db


class FastMaxPool(object):

  @staticmethod
  def forward(x, pool_param):
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width), stride=stride)
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    cache = (x, pool_param, tx, out, layer)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    try:
      x, _, tx, out, layer = cache
      out.backward(dout)
      dx = tx.grad.detach()
    except RuntimeError:
      dx = torch.zeros_like(tx)
    return dx


class Conv_ReLU(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    out, relu_cache = ReLU.forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = ReLU.backward(dout, relu_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db
    

class Conv_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, b, conv_param, pool_param):
    """
    A convenience layer that performs a convolution, a ReLU, and a pool.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    s, relu_cache = ReLU.forward(a)
    out, pool_cache = FastMaxPool.forward(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = FastMaxPool.backward(dout, pool_cache)
    da = ReLU.backward(ds, relu_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db


class Linear_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an linear transform, batch normalization,
    and ReLU.
    Inputs:
    - x: Array of shape (N, D1); input to the linear layer
    - w, b: Arrays of shape (D2, D2) and (D2,) giving the weight and bias for
      the linear transform.
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.
    Returns:
    - out: Output from ReLU, of shape (N, D2)
    - cache: Object to give to the backward pass.
    """
    a, fc_cache = Linear.forward(x, w, b)
    a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
    out, relu_cache = ReLU.forward(a_bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the linear-batchnorm-relu convenience layer.
    """
    fc_cache, bn_cache, relu_cache = cache
    da_bn = ReLU.backward(dout, relu_cache)
    da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
    dx, dw, db = Linear.backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    out, relu_cache = ReLU.forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = ReLU.backward(dout, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    s, relu_cache = ReLU.forward(an)
    out, pool_cache = FastMaxPool.forward(s, pool_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    ds = FastMaxPool.backward(dout, pool_cache)
    dan = ReLU.backward(ds, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  conv - relu - 2x2 max pool - linear - relu - linear - softmax
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dims=(3, 32, 32), num_filters=32, filter_size=7,
         hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
         dtype=torch.float, device='cpu'):
    """
    Initialize a new network.
    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Width/height of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian centered at 0.0   #
    # with standard deviation equal to weight_scale; biases should be          #
    # initialized to zero. All weights and biases should be stored in the      #
    # dictionary self.params. Store weights and biases for the convolutional   #
    # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
    # weights and biases of the hidden linear layer, and keys 'W3' and 'b3'    #
    # for the weights and biases of the output linear layer.                   #
    #                                                                          #
    # IMPORTANT: For this assignment, you can assume that the padding          #
    # and stride of the first convolutional layer are chosen so that           #
    # **the width and height of the input are preserved**. Take a look at      #
    # the start of the loss() function to see how that happens.                #               
    ############################################################################
    
    C, H, W = input_dims

    # Weights for first conv layer
    self.params['W1'] = weight_scale * torch.randn((num_filters, C, filter_size, filter_size), 
                                      dtype=dtype, device=device)
    self.params['b1'] = torch.zeros(num_filters, dtype=dtype, device=device)

    height_after_2x2_max_pool = int(1 + (H - 2) / 2)
    width_after_2x2_max_pool = int(1 + (W - 2) / 2)
    input_dim = num_filters * height_after_2x2_max_pool * width_after_2x2_max_pool
    # Weights for the second linear layer
    self.params['W2'] = weight_scale * torch.randn((input_dim, hidden_dim), dtype=dtype, device=device)
    self.params['b2'] = torch.zeros(hidden_dim, dtype=dtype, device=device)

    # Weights for the third linear layer
    self.params['W3'] = weight_scale * torch.randn((hidden_dim, num_classes), dtype=dtype, device=device)
    self.params['b3'] = torch.zeros(num_classes, dtype=dtype, device=device)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

  def save(self, path):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.params,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))

  def load(self, path):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint['params']
    self.dtype = checkpoint['dtype']
    self.reg = checkpoint['reg']
    print("load checkpoint file: {}".format(path))

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    Input / output: Same API as TwoLayerNet.
    """
    X = X.to(self.dtype)
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    #                                                                          #
    # Remember you can use the functions defined in your implementation above. #
    ############################################################################
    
    # conv - relu - 2x2 max pool - ...
    out, cache1 = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)

    # ... - linear - relu - ...
    out, cache2 = Linear_ReLU.forward(out, W2, b2)

    # ... - linear (- ...)
    out, cache3 = Linear.forward(out, W3, b3)

    scores = out

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization does not include  #
    # a factor of 0.5                                                          #
    ############################################################################
    
    # ... - softmax
    loss, dx = softmax_loss(scores, y)

    # Add L2-regularization to the loss
    l2_reg_sum = 0
    l2_reg_sum += (W1 * W1).sum()
    l2_reg_sum += (W2 * W2).sum()
    l2_reg_sum += (W3 * W3).sum()
    loss += self.reg * l2_reg_sum

    # ... - linear - ...
    dx, grads['W3'], grads['b3'] = Linear.backward(dx, cache3)
    grads['W3'] += self.reg * 2 * W3

    # ... - linear - relu - ...
    dx, grads['W2'], grads['b2'] = Linear_ReLU.backward(dx, cache2)
    grads['W2'] += self.reg * 2 * W2

    # conv - relu - 2x2 max pool - ...
    _, grads['W1'], grads['b1'] = Conv_ReLU_Pool.backward(dx, cache1)
    grads['W1'] += self.reg * 2 * W1

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
  """
  Implement Kaiming initialization for linear and convolution layers.
  
  Inputs:
  - Din, Dout: Integers giving the number of input and output dimensions for
    this layer
  - K: If K is None, then initialize weights for a linear layer with Din input
    dimensions and Dout output dimensions. Otherwise if K is a nonnegative
    integer then initialize the weights for a convolution layer with Din input
    channels, Dout output channels, and a kernel size of KxK.
  - relu: If ReLU=True, then initialize weights with a gain of 2 to account for
    a ReLU nonlinearity (Kaiming initializaiton); otherwise initialize weights
    with a gain of 1 (Xavier initialization).
  - device, dtype: The device and datatype for the output tensor.

  Returns:
  - weight: A torch Tensor giving initialized weights for this layer. For a
    linear layer it should have shape (Din, Dout); for a convolution layer it
    should have shape (Dout, Din, K, K).
  """
  gain = 2. if relu else 1.
  weight = None
  if K is None:
    ###########################################################################
    # TODO: Implement Kaiming initialization for linear layer.                #
    # The weight scale is sqrt(gain / fan_in),                                #
    # where gain is 2 if ReLU is followed by the layer, or 1 if not,          #
    # and fan_in = num_in_channels (= Din).                                   #
    # The output should be a tensor in the designated size, dtype, and device.#
    ###########################################################################

    weight_scale = torch.sqrt(torch.tensor(gain / Din))
    weight = weight_scale * torch.randn((Din, Dout), device=device, dtype=dtype)

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  else:
    ###########################################################################
    # TODO: Implement Kaiming initialization for convolutional layer.         #
    # The weight scale is sqrt(gain / fan_in),                                #
    # where gain is 2 if ReLU is followed by the layer, or 1 if not,          #
    # and fan_in = num_in_channels (= Din) * K * K                            #
    # The output should be a tensor in the designated size, dtype, and device.#
    ###########################################################################

    num_in_channels = Din
    fan_in = num_in_channels * K * K

    weight_scale = torch.sqrt(torch.tensor(gain / fan_in))
    weight = weight_scale * torch.randn((Dout, Din, K, K), device=device, dtype=dtype)

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  return weight

class DeepConvNet(object):
  """
  A convolutional neural network with an arbitrary number of convolutional
  layers in VGG-Net style. All convolution layers will use kernel size 3 and 
  padding 1 to preserve the feature map size, and all pooling layers will be
  max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
  size of the feature map.

  The network will have the following architecture:
  
  {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

  Each {...} structure is a "macro layer" consisting of a convolution layer,
  an optional batch normalization layer, a ReLU nonlinearity, and an optional
  pooling layer. After L-1 such macro layers, a single fully-connected layer
  is used to predict the class scores.

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dims=(3, 32, 32),
               num_filters=[8, 8, 8, 8, 8],
               max_pools=[0, 1, 2, 3, 4],
               batchnorm=False,
               num_classes=10, weight_scale=1e-3, reg=0.0,
               weight_initializer=None,
               dtype=torch.float, device='cpu'):
    """
    Initialize a new network.

    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: List of length (L - 1) giving the number of convolutional
      filters to use in each macro layer.
    - max_pools: List of integers giving the indices of the macro layers that
      should have max pooling (zero-indexed).
    - batchnorm: Whether to include batch normalization in each macro layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights, or the string "kaiming" to use Kaiming initialization instead
    - reg: Scalar giving L2 regularization strength. L2 regularization should
      only be applied to convolutional and fully-connected weight matrices;
      it should not be applied to biases or to batchnorm scale and shifts.
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'    
    """
    self.params = {}
    self.num_layers = len(num_filters) + 1
    self.max_pools = max_pools
    self.batchnorm = batchnorm
    self.reg = reg
    self.dtype = dtype
  
    if device == 'cuda':
      device = 'cuda:0'
    
    ############################################################################
    # TODO: Initialize the parameters for the DeepConvNet. All weights,        #
    # biases, and batchnorm scale and shift parameters should be stored in the #
    # dictionary self.params.                                                  #
    #                                                                          #
    # Weights for conv and fully-connected layers should be initialized        #
    # according to weight_scale. Biases should be initialized to zero.         #
    # Batchnorm scale (gamma) and shift (beta) parameters should be initilized #
    # to ones and zeros respectively.                                          #           
    ############################################################################
    
    C, H, W = input_dims

    filter_size = 3

    # {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - ...
    self.has_pool_layer = [False] * (self.num_layers - 1)
    for pool_layer_idx in max_pools:
      self.has_pool_layer[pool_layer_idx] = True

    prev_filter = C
    for i in range(self.num_layers - 1):
      id = i + 1
      # conv kernel size = (3, 3)
      out_filter = num_filters[i]
      kernel_shape = (out_filter, prev_filter, filter_size, filter_size)
      
      self.params[f'b{id}'] = torch.zeros(out_filter, device=device, dtype=dtype)

      if weight_scale != 'kaiming':
        self.params[f'W{id}'] = weight_scale * torch.randn(kernel_shape, device=device, dtype=dtype)
      else:
        # Use kaiming initialization
        self.params[f'W{id}'] = kaiming_initializer(prev_filter, out_filter, K=3, relu=True, device=device, dtype=dtype)

      if self.batchnorm:
        self.params[f'gamma{id}'] = torch.ones(out_filter, device=device, dtype=dtype)
        self.params[f'beta{id}'] = torch.zeros(out_filter, device=device, dtype=dtype)

      prev_filter = out_filter

    out_channels = prev_filter
    out_h = int(H / (1 << len(max_pools)))
    out_w = int(W / (1 << len(max_pools)))

    # ... - linear
    input_dim = out_channels * out_h * out_w
    
    if weight_scale != 'kaiming':
      self.params[f'W{self.num_layers}'] = weight_scale * torch.randn((input_dim, num_classes), device=device, dtype=dtype)
    else:
      # Use kaiming initialization
      self.params[f'W{self.num_layers}'] = kaiming_initializer(input_dim, num_classes, relu=False, device=device, dtype=dtype)
    
    self.params[f'b{self.num_layers}'] = torch.zeros(num_classes, device=device, dtype=dtype)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.batchnorm:
      self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))]
      
    # Check that we got the right number of parameters
    if not self.batchnorm:
      params_per_macro_layer = 2  # weight and bias
    else:
      params_per_macro_layer = 4  # weight, bias, scale, shift
    num_params = params_per_macro_layer * len(num_filters) + 2
    msg = 'self.params has the wrong number of elements. Got %d; expected %d'
    msg = msg % (len(self.params), num_params)
    assert len(self.params) == num_params, msg

    # Check that all parameters have the correct device and dtype:
    for k, param in self.params.items():
      msg = 'param "%s" has device %r; should be %r' % (k, param.device, device)
      assert param.device == torch.device(device), msg
      msg = 'param "%s" has dtype %r; should be %r' % (k, param.dtype, dtype)
      assert param.dtype == dtype, msg


  def save(self, path):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.params,
      'num_layers': self.num_layers,
      'max_pools': self.max_pools,
      'batchnorm': self.batchnorm,
      'bn_params': self.bn_params,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))


  def load(self, path, dtype, device):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint['params']
    self.dtype = dtype
    self.reg = checkpoint['reg']
    self.num_layers = checkpoint['num_layers']
    self.max_pools = checkpoint['max_pools']
    self.batchnorm = checkpoint['batchnorm']
    self.bn_params = checkpoint['bn_params']


    for p in self.params:
      self.params[p] = self.params[p].type(dtype).to(device)

    for i in range(len(self.bn_params)):
      for p in ["running_mean", "running_var"]:
        self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)

    print("load checkpoint file: {}".format(path))


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the deep convolutional network.
    Input / output: Same API as ThreeLayerConvNet.
    """
    X = X.to(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params since they
    # behave differently during training and testing.
    bn_param = {}
    if self.batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode

    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = 3
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    ############################################################################
    # TODO: Implement the forward pass for the DeepConvNet, computing the      #
    # class scores for X and storing them in the scores variable.              #
    #                                                                          #
    # You should use the fast versions of convolution and max pooling layers,  #
    # or the convolutional sandwich layers, to simplify your implementation.   #
    ############################################################################

    out = X

    cache_container = []

    # {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - ...
    for i in range(self.num_layers - 1):
      layer_id = i + 1

      if self.has_pool_layer[i]:
        if self.batchnorm:
          # conv - batchnorm - relu - pool
          out, cache = Conv_BatchNorm_ReLU_Pool.forward(out, self.params[f'W{layer_id}'], self.params[f'b{layer_id}'], self.params[f'gamma{layer_id}'], self.params[f'beta{layer_id}'], conv_param, self.bn_params[i], pool_param)
        else:
          # conv - relu - pool
          out, cache = Conv_ReLU_Pool.forward(out, self.params[f'W{layer_id}'], self.params[f'b{layer_id}'], conv_param, pool_param)
      else:
        if self.batchnorm:
          # conv - batchnorm - relu
          out, cache = Conv_BatchNorm_ReLU.forward(out, self.params[f'W{layer_id}'], self.params[f'b{layer_id}'], self.params[f'gamma{layer_id}'], self.params[f'beta{layer_id}'], conv_param, self.bn_params[i])
        else:
          # conv - relu
          out, cache = Conv_ReLU.forward(out, self.params[f'W{layer_id}'], self.params[f'b{layer_id}'], conv_param)

      cache_container.append(cache)

    # ... - linear
    scores, cache = Linear.forward(out, self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}'])
    cache_container.append(cache)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the DeepConvNet, storing the loss  #
    # and gradients in the loss and grads variables. Compute data loss using   #
    # softmax, and make sure that grads[k] holds the gradients for             #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization does not include  #
    # a factor of 0.5                                                          #
    ############################################################################

    loss, dx = softmax_loss(scores, y)

    # Add L2 regularization
    if self.reg != 0:
      for i in range(self.num_layers):
        layer_id = i + 1
        loss += self.reg * (self.params[f'W{layer_id}'] * self.params[f'W{layer_id}']).sum()

    # ... - linear
    dx, grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = Linear.backward(dx, cache_container[self.num_layers - 1])
    grads[f'W{self.num_layers}'] += self.reg * 2 * self.params[f'W{self.num_layers}']

    # {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - ...
    for layer_id in range(self.num_layers - 1, 0, -1):
      cache = cache_container[layer_id - 1]

      if self.has_pool_layer[layer_id - 1]:
        if self.batchnorm:
          # conv - batchnorm - relu - pool
          dx, grads[f'W{layer_id}'], grads[f'b{layer_id}'], grads[f'gamma{layer_id}'], grads[f'beta{layer_id}'] = Conv_BatchNorm_ReLU_Pool.backward(dx, cache)
        else:
          # conv - relu - pool
          dx, grads[f'W{layer_id}'], grads[f'b{layer_id}'] = Conv_ReLU_Pool.backward(dx, cache)
      else:
        if self.batchnorm:
          # conv - batchnorm - relu
          dx, grads[f'W{layer_id}'], grads[f'b{layer_id}'], grads[f'gamma{layer_id}'], grads[f'beta{layer_id}'] = Conv_BatchNorm_ReLU.backward(dx, cache)
        else:
          # conv - relu
          dx, grads[f'W{layer_id}'], grads[f'b{layer_id}'] = Conv_ReLU.backward(dx, cache)

      # Apply L2 regularization
      grads[f'W{layer_id}'] += self.reg * 2 * self.params[f'W{layer_id}']

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return loss, grads

def find_overfit_parameters():
  ############################################################################
  # TODO: Change weight_scale and learning_rate so your model achieves 100%  #
  # training accuracy within 30 epochs.                                      #
  ############################################################################
  weight_scale = 24e-2   # Experiment with this!
  learning_rate = 51e-4  # Experiment with this!
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################
  return weight_scale, learning_rate

def create_convolutional_solver_instance(data_dict, dtype, device):
  ################################################################################
  # TODO: Train the best DeepConvNet that you can on CIFAR-10 within 60 seconds. #
  ################################################################################
  model = DeepConvNet(input_dims=(3, 32, 32),
                      num_filters=[32, 32, 64], # TODO:
                      max_pools=[0, 1, 2], # TODO:
                      batchnorm=False,
                      num_classes=10, # TODO:
                      weight_scale='kaiming',
                      reg=0.0, # TODO:
                      dtype=dtype, 
                      device=device)

  solver = Solver(model,
                  data_dict,
                  optim_config={
                    'learning_rate': 70e-3 # TODO: tune
                  },
                  lr_decay=0.99,
                  num_epochs=20,
                  batch_size=128,
                  print_every=1000,
                  print_acc_every=5,
                  device=device)
  ################################################################################
  #                              END OF YOUR CODE                                #
  ################################################################################
  return solver
