# Authors: Michael Eickenberg
#          Kyle Kastner
# License: BSD 3 clause

import theano
import numbers
import numpy as np
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d


def _relu(x):
    return T.maximum(x, 0)


def _identity(x):
    return x

ACTIVATIONS = {'sigmoid': T.nnet.sigmoid,
               'identity': _identity,
               'linear': _identity,
               'relu': _relu,
               None: _identity}


class Feedforward(object):
    def __init__(self, weights, biases=None, activation='relu',
                 input_dtype='float32'):
        pass  # should maybe initialize the type of variable
        self.input_dtype = input_dtype
        self.weights = theano.shared(weights.astype(self.input_dtype))
        self.activation_function = ACTIVATIONS[activation]
        if biases is not None:
            self.biases = theano.shared(biases.astype(self.input_dtype))
        else:
            self.biases = None
        self._build_expression()  # maybe have a flag

    def _build_expression(self):
        self.input_ = T.matrix(dtype=self.input_dtype)
        self.expression_ = T.dot(self.input_, self.weights)
        if self.biases is not None:
            self.expression_ += self.biases
        self.expression_ = self.activation_function(self.expression_)


class Convolution(object):
    """A wrapper for a 2D convolution building block"""
    def __init__(self,
                 convolution_filter,
                 biases=None,
                 activation='relu',
                 border_mode='valid',
                 subsample=None,
                 cropping=None,
                 input_dtype='float32'):
        self.convolution_filter = convolution_filter
        self.biases = biases
        self.activation_function = ACTIVATIONS[activation]
        self.subsample = subsample
        self.border_mode = border_mode
        self.cropping = cropping
        self.input_dtype = input_dtype

        self._build_expression()  # not sure whether this is legit here

    def _build_expression(self):
        if self.cropping is None:
            self.cropping_ = [(0, None), (0, None)]
        else:
            self.cropping_ = self.cropping

        if self.subsample is None:
            self.subsample_ = (1, 1)
        else:
            self.subsample_ = self.subsample

        cf = self.convolution_filter
        if not isinstance(cf, T.sharedvar.TensorSharedVariable):
            if isinstance(cf, np.ndarray):
                self.convolution_filter_ = theano.shared(cf)
            else:
                raise ValueError("Variable type not understood")
        else:
            self.convolution_filter_ = cf

        biases = self.biases
        if biases is not None:
            if not isinstance(biases, T.sharedvar.TensorSharedVariable):
                if isinstance(biases, np.ndarray):
                    self.biases_ = theano.shared(biases)
                else:
                    raise ValueError("Variable type not understood")
            else:
                self.biases_ = biases

        self.input_ = T.tensor4(dtype=self.input_dtype)
        c = self.cropping_
        self.expression_ = T.nnet.conv2d(self.input_,
            self.convolution_filter_,
            border_mode=self.border_mode,
            subsample=self.subsample_)[:, :, c[0][0]:c[0][1],
                                             c[1][0]:c[1][1]]
        if self.biases is not None:
            self.expression_ += self.biases_.dimshuffle('x', 0, 'x', 'x')
        self.expression_ = self.activation_function(self.expression_)


class MarginalConvolution(object):
    """Same as Convolution if there is only one input channel. If there are
    several input channels, then it convolves each with the same filter,
    instead of using a 3D filter as does Convolution.
    Used for scattering transform.
    And for global smoothing.
    """
    def __init__(self,
                 convolution_filter,
                 activation=None,
                 border_mode='valid',
                 subsample=None,
                 input_dtype='float32'):
        self.convolution_filter = convolution_filter
        self.activation = activation
        self.border_mode = border_mode
        self.subsample = subsample
        self.input_dtype = input_dtype

        self._build_expression()

    def _build_expression(self):

        if self.subsample is None:
            self.subsample_ = (1, 1)
        else:
            self.subsample_ = self.subsample

        cf = self.convolution_filter
        if not isinstance(cf, T.sharedvar.TensorSharedVariable):
            if isinstance(cf, np.ndarray):
                self.convolution_filter_ = theano.shared(cf)
            else:
                raise ValueError("Variable type not understood")
        else:
            self.convolution_filter_ = cf

        self.input_ = T.tensor4(dtype=self.input_dtype)
        border_mode_ = self.border_mode
        if border_mode_ == 'same':
            border_mode_ = 'full'
        self.expression_ = T.nnet.conv2d(
            self.input_.reshape((-1, 1,
                                  self.input_.shape[-2],
                                  self.input_.shape[-1])),
            self.convolution_filter_.reshape((-1, 1,
                                  self.convolution_filter_.shape[-2],
                                  self.convolution_filter_.shape[-1])),
            border_mode=border_mode_,
            subsample=self.subsample_)
        if border_mode_ == 'valid':
            output_shape = (
                self.input_.shape[-2] -
                self.convolution_filter_.shape[-2] + 1,
                self.input_.shape[-1] -
                self.convolution_filter_.shape[-1] + 1)
        elif border_mode_ == 'full':
            output_shape = (
                self.input_.shape[-2] +
                self.convolution_filter_.shape[-2] - 1,
                self.input_.shape[-1] +
                self.convolution_filter_.shape[-1] - 1)
        self.expression_ = self.expression_.reshape(
            (self.input_.shape[0], -1) + output_shape)
        if self.border_mode == 'same':
            start_crop = (self.convolution_filter_.shape[-2:] - 1) // 2
            end_crop = (self.convolution_filter_.shape[-2:] - 1
                          - start_crop)
            self.expression_ = self.expression_[:, :,
                start_crop[0]:-end_crop[0],
                start_crop[1]:-end_crop[1]]
        activation_function = ACTIVATIONS[self.activation]
        self.expression_ = activation_function(self.expression_)


class PassThrough(object):
    def __init__(self, input_dtype='float32'):
        pass  # should maybe initialize the type of variable
        self.input_dtype = input_dtype
        self._build_expression()  # maybe have a flag

    def _build_expression(self):
        self.input_ = T.tensor4(dtype=self.input_dtype)
        self.expression_ = self.input_


class Standardize(object):
    def __init__(self, mean=None, std=None, input_dtype='float32'):
        self.mean = mean
        self.std = std
        self.input_dtype = input_dtype

        self._build_expression()

    def _build_expression(self):
        self.input_ = T.tensor4(dtype=self.input_dtype)
        if self.mean is not None:
            centered = self.input_ - self.mean
        else:
            raise NotImplementedError("One could default to mean per sample")
        if self.std is not None:
            scaled = centered / self.std
        else:
            raise NotImplementedError("One could default to std per sample")
        self.expression_ = scaled


class MaxPool(object):
    def __init__(self, max_pool_stride, input_dtype='float32'):
        self.max_pool_stride = max_pool_stride
        self.input_dtype = input_dtype

        self._build_expression()

    def _build_expression(self):
        self.input_ = T.tensor4(dtype=self.input_dtype)
        self.expression_ = max_pool_2d(self.input_, self.max_pool_stride,
                                       ignore_border=True)


class ZeroPad(object):
    """Zero-padding using a convolution with an appropriate padded Dirac.
    Any input welcome as to how to make this more simple."""

    def __init__(self, padding=1, input_dtype='float32'):
        self.padding = padding
        self.input_dtype = input_dtype

        self._build_expression()

    def _build_expression(self):
        if isinstance(self.padding, numbers.Number):
            self.padding_ = (self.padding,) * 4
        elif len(self.padding) == 1:
            self.padding_ = tuple(self.padding) * 4
        elif len(self.padding) == 2:
            self.padding_ = tuple(self.padding) * 2
        elif len(self.padding) == 4:
            self.padding_ = self.padding
        else:
            raise ValueError("padding must be of length 1, 2 or 4")

        p = self.padding_
        shape = (p[0] + 1 + p[2], p[1] + 1 + p[3])

        padding_indicator = np.zeros(shape, dtype=np.dtype(self.input_dtype))
        padding_indicator[p[0], p[1]] = 1.
        self.padding_indicator_ = theano.shared(
            padding_indicator[np.newaxis, np.newaxis])
        self.input_ = T.tensor4(dtype=self.input_dtype)
        input_shape = self.input_.shape
        output_shape = (input_shape[0], input_shape[1],
                        input_shape[2] + p[0] + p[2],
                        input_shape[3] + p[1] + p[3])
        intermediate_shape = (-1, 1, input_shape[2], input_shape[3])
        self.expression_ = T.nnet.conv2d(
            self.input_.reshape(intermediate_shape),
            self.padding_indicator_,
            border_mode='full').reshape(output_shape)


class Relu(object):
    def __init__(self, input_type=T.tensor4):
        self.input_type = input_type

        self._build_expression()

    def _build_expression(self):
        self.input_ = self.input_type()
        self.expression_ = T.maximum(self.input_, 0)


def fuse(building_blocks, fuse_dim=4, input_variables=None, entry_expression=None,
         output_expressions=-1, input_dtype='float32'):

    num_blocks = len(building_blocks)

    if isinstance(output_expressions, numbers.Number):
        output_expressions = [output_expressions]

    # account for indices -1, -2 etc
    output_expressions = [oe % num_blocks for oe in output_expressions]

    if fuse_dim == 4:
        fuse_block = T.tensor4
    else:
        fuse_block = T.matrix

    if input_variables is None and entry_expression is None:
        input_variables = fuse_block(dtype=input_dtype)
        entry_expression = input_variables

    current_expression = entry_expression
    outputs = []

    for i, block in enumerate(building_blocks):
        current_expression = theano.clone(
            block.expression_,
            replace={block.input_: current_expression},
            strict=False)
        if i in output_expressions:
            outputs.append(current_expression)

    return outputs, input_variables

