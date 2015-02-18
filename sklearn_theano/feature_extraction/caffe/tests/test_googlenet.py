from skimage.data import coffee, camera
from sklearn_theano.feature_extraction.caffe.googlenet import (
    GoogLeNetTransformer, GoogLeNetClassifier)
import numpy as np


co = coffee().astype(np.float32)
ca = camera().astype(np.float32)[:, :, np.newaxis] * np.ones((1, 1, 3),
                                                             dtype='float32')


def test_googlenet_transformer():
    """smoke test for googlenet transformer"""
    t = GoogLeNetTransformer()

    t.transform(co)
    t.transform(ca)


def test_googlenet_classifier():
    """smoke test for googlenet classifier"""
    c = GoogLeNetClassifier()

    c.predict(co)
    c.predict(ca)
