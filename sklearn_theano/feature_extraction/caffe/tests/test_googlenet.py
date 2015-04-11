from skimage.data import coffee, camera
from sklearn_theano.feature_extraction.caffe import googlenet
from sklearn_theano.feature_extraction import (
    GoogLeNetTransformer, GoogLeNetClassifier)
import numpy as np
import theano
from nose import SkipTest
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import os

co = coffee().astype(np.float32)
ca = camera().astype(np.float32)[:, :, np.newaxis] * np.ones((1, 1, 3),
                                                             dtype='float32')


def test_googlenet_transformer():
    """smoke test for googlenet transformer"""
    if os.environ.get('CI', None) is not None:
        raise SkipTest("Skipping heavy data loading on CI")
    t = GoogLeNetTransformer()

    t.transform(co)
    t.transform(ca)


def test_googlenet_classifier():
    """smoke test for googlenet classifier"""
    if os.environ.get('CI', None) is not None:
        raise SkipTest("Skipping heavy data loading on CI")
    c = GoogLeNetClassifier()

    c.predict(co)
    c.predict(ca)


def _fetch_caffe_layers_for_coffee():
    """Loads a file containing all caffe layer outputs for
    skimage.data.coffee."""

    # TODO: write a generic pycaffe model finder
    #       and feedforward any image or load from disk

    f = np.load('coffee.npz')
    output = dict()
    for k in f.files:
        output[k] = f[k]
    return output


def test_caffe_correspondence(verbose=1):
    """Test correspondence of all layers to caffe model"""

    c = _fetch_caffe_layers_for_coffee()

    layer_expr, input_expr = googlenet.create_theano_expressions(
        verbose=verbose)
    layer_names = layer_expr.keys()

    # restrict layers to those available in c
    exclude = ['data']
    layer_names = [name for name in layer_names if name in c.keys()
                   and not name in exclude]

    all_outputs = theano.function([input_expr],
                                  [layer_expr[name] for name in layer_names])

    fprop = all_outputs(c['data'][np.newaxis])

    for i, name in enumerate(layer_names):
        if verbose > 0:
            print "Comparing %s" % name
        assert_array_almost_equal(fprop[i][0], c[name], decimal=2)
        assert_almost_equal(
            ((fprop[i][0] - c[name]) ** 2).sum() / (c[name] ** 2).sum(), 0)

