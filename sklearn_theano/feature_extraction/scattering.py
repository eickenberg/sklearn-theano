"""Implements the scattering transform"""
import theano
import theano.tensor as T
import numpy as np


def _rotation_matrices(angles):
    """Returns 2D rotation matrix expressions for all angles.

    Parameters
    ==========

    angles : tensor variable or shared variable or array-like.
        angles of rotation.
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


def morlet_filter_2d(shape, sigmas, slant, xis, thetas, offset=None):

    xgrid, ygrid = _mgrid[0:shape[0]:1., 0:shape[1]:1.]

    if offset is not None:
        xgrid = xgrid - offset[0]
        ygrid = ygrid - offset[1]

    xgrid = xgrid - (shape[0] // 2)
    ygrid = ygrid - (shape[1] // 2)

    stacked_grid = T.vertical_stack(xgrid.reshape((1, -1)),
                                    ygrid.reshape((1, -1)))
    rotations = _rotation_matrices(thetas)
    rotated_grids = T.dot(rotations, stacked_grid)

    dilations = T.as_tensor_variable(
        [sigmas ** -1, slant / sigmas]).T.reshape((sigmas.shape[0], 1, 2, 1))
    dilated_grids = rotated_grids * dilations
    dilated_distances_squared = (dilated_grids ** 2).sum(2)
    gaussian_envelopes = T.exp(-dilated_distances_squared / 2)

    oscillations_cos = T.cos(rotated_grids[np.newaxis, :, 1, :] *
                             xis.reshape((xis.shape[0], 1, 1)))
    oscillations_sin = T.sin(rotated_grids[np.newaxis, :, 1, :] *
                             xis.reshape((xis.shape[0], 1, 1)))

    gabor_cos = gaussian_envelopes * oscillations_cos
    gabor_sin = gaussian_envelopes * oscillations_sin

    return rotated_grids, dilated_grids, dilated_distances_squared, gaussian_envelopes, oscillations_cos, oscillations_sin, gabor_cos, gabor_sin




