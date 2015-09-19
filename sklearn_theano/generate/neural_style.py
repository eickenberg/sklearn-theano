"""Module implementing neural style from Gathys, Ecker, Bethge 8/2015"""

import theano
import theano.tensor as T
import numpy as np


def feature_covariance(expression):
    """Calculates the spatially agnostic feature covariance.

    Parameters
    ==========
    expression: theano expression, shape (n_features, ...)

    Note:
    =====
    Assumes first axis represents feature maps,
    following axes contain the features that the covariance sums across.
    """

    # new_shape = T.as_tensor_array((expression.shape[0], -1))

    reshaped_expression = expression.reshape((expression.shape[0], -1))
    gram = reshaped_expression.dot(reshaped_expression.T)
    return gram / reshaped_expression.shape[1] ** 2


def style_loss(network_layers_moving,
               network_layers_fixed=None,
               weights=None,
               feature_covariances_fixed=None):
    """Computes the style loss as a weighted sum of feature covariances.

    Parameters
    ==========
    network_layers_moving: theano expression, 
        represents a network layer or list of network layers containing style 
        network layers of image to be changed.

    network_layers_fixed: theano expression, optional 
        represents a network layer or list of network layers containing style 
        network layers of target image.

    weights: list or array of floats
        indicates the weights for summation of loss terms. Defaults to 1

    feature_covariances_fixed: theano expression, optional
        represents feature covariances of the fixed image. If specified, it
        overrides network_layers_fixed. Can be filled with a constant shared
        variable. This is to avoid unnecessary calculations at runtime.

    Notes
    =====
    Network_layer expressions are assumed to be convolutional of shape
        (n_samples=1, n_features, *feature_shape_dimensions).
        Summation occurs over feature_shape_dimension axes.

"""

    if not isinstance(network_layers_moving, list):
        network_layers_moving = [network_layers_moving]

    if feature_covariances_fixed is None:
        if network_layers_fixed is None:
            raise ValueError("Must specify either feature_covariances_fixed"
                             " or network_layers_fixed")
        if not isinstance(network_layers_fixed, list):
            network_layers_fixed = [network_layers_fixed]

        # use only the first sample to create these feature maps
        feature_covariances_fixed = [feature_covariance(fixed_layer[0])
                                     for fixed_layer in network_layers_fixed]

    if weights is None:
        weights = np.ones(len(network_layers_moving))

    feature_covariances_moving = [feature_covariance(moving_layer[0])
                                  for moving_layer in network_layers_moving]

    if len(feature_covariances_moving) != len(feature_covariances_fixed):
        raise ValueError("Must specify same number of moving and fixed"
                         "layers for loss")

    loss = 0
    for w, moving, fixed in zip(weights, feature_covariances_moving,
                                feature_covariances_fixed):
        loss = loss + w * ((moving - fixed) ** 2).sum() / (
                4 * moving.shape[0] ** 2)

    return loss


def content_loss(network_layers_moving,
                 network_layers_fixed,
                 weights=None):
    """Computes the loss between target image activation and moving image
    activation. It amounts to the squared error between the two.

    Parameters
    ==========

    network_layers_moving: theano expression of list thereof
        specifies 

"""

    if not isinstance(network_layers_moving, list):
        network_layers_moving = [network_layers_moving]

    if not isinstance(network_layers_fixed, list):
        network_layers_fixed = [network_layers_fixed]

    if len(network_layers_moving) != len(network_layers_fixed):
        raise ValueError("must specify same number of fixed and moving"
                         " layers for correct evaluation of loss.")

    if weights is None:
        weights = np.ones(len(network_layers_moving))

    loss = 0

    for w, fixed, moving in zip(weights, network_layers_fixed,
                                network_layers_moving):
        loss = loss + w * ((moving[0] - fixed[0]) ** 2).sum()

    return 0.5 * loss


def neural_style_loss(content_layers, style_layers,
                      content_input_expr, style_input_expr,
                      target_content_input, target_style_input,
                      alpha, beta,
                      content_layer_weights=None,
                      style_layer_weights=None):

    # First, calculate target activation and style

    style_matrices = [feature_covariance(style[0]) for style in style_layers]
    f_style_matrices = theano.function([style_input_expr], style_matrices)
    target_styles = map(theano.shared, f_style_matrices(target_style_input))

    f_content = theano.function([content_input_expr], content_layers)
    target_content = map(theano.shared, f_content(target_content_input))

    # compute content loss
    c_loss = content_loss(content_layers, target_content,
                          content_layer_weights)

    # compute style loss, placing precomputed covariance matrices to
    # ensure that they don't get evaluated every time
    s_loss = style_loss(style_layers,
                        feature_covariances_fixed=target_styles,
                        weights=style_layer_weights)

    return alpha * c_loss + beta * s_loss


def get_neural_style_loss_and_gradient(content_layers, style_layers,
                                       content_input_expr, style_input_expr,
                                       target_content, target_style,
                                       alpha, beta,
                                       content_layer_weights=None,
                                       style_layer_weights=None,
                                       ):
    
