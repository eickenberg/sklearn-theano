"""Implements the scattering transform"""
import theano
import theano.tensor as T
import numpy as np
from theano.sandbox import fourier


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


def morlet_filter_2d(shape, sigmas, slant, xis, thetas, 
                     noDC=True, offset=None,
                     return_complex=False):

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
        [slant * sigmas ** -1, sigmas ** -1]).T.reshape((sigmas.shape[0], 1, 2, 1))
    dilated_grids = rotated_grids * dilations
    dilated_distances_squared = (dilated_grids ** 2).sum(2)
    gaussian_envelopes = T.exp(-dilated_distances_squared / 2)

    oscillations_cos = T.cos(rotated_grids[np.newaxis, :, 1, :] *
                             xis.reshape((xis.shape[0], 1, 1)))
    oscillations_sin = T.sin(rotated_grids[np.newaxis, :, 1, :] *
                             xis.reshape((xis.shape[0], 1, 1)))

    gabor_cos = gaussian_envelopes * oscillations_cos
    gabor_sin = gaussian_envelopes * oscillations_sin

    # because we are generating morlet filters, we need to make them have
    # zero sum. This can be done analytically, which would be better in this
    # symbolic setting, but the empirical way of doing it will have to
    # suffice for now.

    if noDC:
        envelope_sums = gaussian_envelopes.sum(axis=-1)
        gabor_cos_DCs = gabor_cos.sum(axis=-1)
        K = gabor_cos_DCs / envelope_sums

        gabor_cos = gabor_cos - K.dimshuffle(0, 1, 'x') * gaussian_envelopes

    # scale the filters
    prefactors = (1 / (2 * np.pi * sigmas ** 2 / slant ** 2)
                  ).dimshuffle(0, 'x', 'x')
    gabor_cos = gabor_cos * prefactors
    gabor_sin = gabor_sin * prefactors

    if not return_complex:
        bcast_shp = (sigmas.shape[0], thetas.shape[0], 1, shape[0], shape[1])
        return T.concatenate([
                gabor_cos.reshape(bcast_shp),
                gabor_sin.reshape(bcast_shp)
                              ], axis=2)
    else:
        out_shp = (sigmas.shape[0], thetas.shape[0], shape[0], shape[1])
        return (gabor_cos + 1j * gabor_sin).reshape(out_shp)


def _fft2(inp):
    """Use 1D fourier transform provided in theano to do 2d fft on last 2"""
    out_shape = inp.shape
    other_axes = out_shape[:-2]
    nr, nc = out_shape[-2], out_shape[-1]
    ns = T.prod(other_axes)
    proc_shape = (ns, nr, nc)

    inp3 = inp.reshape(proc_shape)

    fft_lines = fourier.fft(inp3.reshape((ns * nr, nc)), nc, 1)
    fft_lines_reshaped = fft_lines.reshape(
        (ns, nr, nc)).dimshuffle(1, 0, 2).reshape((nr, ns * nc))
    fft_columns = fourier.fft(fft_lines_reshaped, nr, 0)

    output = fft_columns.reshape(
        (nr, ns, nc)).dimshuffle(1, 0, 2).reshape(out_shape)

    return output


def morlet_filter_bank_2d(shape, J=4, L=8, Q=1,
                          sigma_phi=0.8, sigma_psi=0.8,
                          xi_psi=None, slant_psi=None,
                          return_complex=False,
                          littlewood_paley_normalization=False):
    if xi_psi is None:
        xi_psi = .5 * (2 ** (-1. / Q) + 1) * np.pi
    if slant_psi is None:
        slant_psi = 4. / L

    res = 0  # this variable indicates resolution. It will be kept at 0

    angles = T.arange(L) * np.pi / L
    scales = 2 ** (T.arange(J) / (1. * Q) - res)

    sigmas = sigma_psi * scales
    xis = xi_psi / scales

    filters = morlet_filter_2d(shape, sigmas, slant_psi, xis, angles,
                               return_complex=return_complex)
    lowpass = morlet_filter_2d(shape, sigmas[J - 1:], 1.,
                               np.array([0.]), np.array([0.]),
                               return_complex=False,
                               noDC=False)[:, :, 0]

    if not littlewood_paley_normalization:
        return filters, lowpass
    else:
        c_filters = filters
        if not return_complex:
            c_filters = filters[:, :, 0] + 1j * filters[:, :, 1]
        fft_filters = abs(T.real(_fft2(c_filters)))
        littlewood_paley = (fft_filters ** 2).sum(0).sum(0)
        K = littlewood_paley.max()

        return filters / T.sqrt(K / 2), lowpass


