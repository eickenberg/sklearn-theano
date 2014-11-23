"""Implements the scattering transform"""
import theano
import theano.tensor as T
import numpy as np


def _rotation_matrices(angles):
    """Returns 2D rotation matrix expressions for all angles.

    Parameters
    ==========

    angles : tensor variable or shared variable or array-like
        array-like will be converted to shared variable.
    """

    if not (isinstance(angles, T.TensorVariable) or
            isinstance(angles, T.sharedvar.TensorSharedVariable)):
        angles = theano.shared(np.asarray(angles))
    if angles.ndim > 1:
        raise Exception("angles must be 0 or 1d. It is %dd." % angles.ndim)

    scalar = angles.ndim == 0
    if scalar:
        angles = angles.reshape((1, 1))

    sin_angles = T.sin(angles).reshape((-1, 1, 1))
    cos_angles = T.cos(angles).reshape((-1, 1, 1))

    first_row = T.concatenate([cos_angles, -sin_angles], axis=2)
    second_row = T.concatenate([sin_angles, cos_angles], axis=2)

    matrices = T.concatenate([first_row, second_row], axis=1)

    if scalar:
        matrices = matrices[0]

    return matrices


class _nd_grid(object):
    """Implements the mgrid and ogrid functionality for theano tensor
    variables.

    Parameters
    ==========
        sparse : boolean, optional, default=True
            Specifying False leads to the equivalent of numpy's mgrid
            functionality. Specifying True leads to the equivalent of ogrid.
    """
    def __init__(self, sparse=False):
        self.sparse = sparse

    def __getitem__(self, *args):

        ndim = len(args[0])
        ranges = [T.arange(sl.start, sl.stop, sl.step) for sl in args[0]]
        shapes = [tuple([1] * j + [r.shape[0]] + [1] * (ndim - 1 - j))
                  for j, r in enumerate(ranges)]
        ranges = [r.reshape(shape) for r, shape in zip(ranges, shapes)]
        ones = [T.ones_like(r) for r in ranges]
        if self.sparse:
            grids = ranges
        else:
            grids = []
            for i in range(ndim):
                grid = 1
                for j in range(ndim):
                    if j == i:
                        grid = grid * ranges[j]
                    else:
                        grid = grid * ones[j]
                grids.append(grid)
        return grids


_mgrid = _nd_grid()
_ogrid = _nd_grid(sparse=True)

