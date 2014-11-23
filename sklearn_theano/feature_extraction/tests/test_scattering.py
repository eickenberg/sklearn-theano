import theano
import theano.tensor as T
import numpy as np
from sklearn_theano.feature_extraction.scattering import _rotation_matrices
from sklearn_theano.feature_extraction.scattering import _mgrid, _ogrid
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test__rotation_matrices():

    angles = (np.array([0., .25, .1, .5, .6, 3.2]) * 2 * np.pi
              ).astype('float32')
    sin_angles, cos_angles = np.sin(angles), np.cos(angles)
    matrices = np.array([[cos_angles, -sin_angles], [sin_angles, cos_angles]])
    matrices = matrices.transpose(2, 0, 1)

    angles_var = T.fvector()

    shared_rot_mat_expression = _rotation_matrices(angles)
    tensor_rot_mat_expression = _rotation_matrices(angles_var)

    f_shared = theano.function([], shared_rot_mat_expression)
    f_tensor = theano.function([angles_var], tensor_rot_mat_expression)

    matrices_shared = f_shared()
    matrices_tensor = f_tensor(angles)

    assert_array_equal(matrices, matrices_shared)
    assert_array_equal(matrices, matrices_tensor)

    # one single matrix
    angle = angles[1]
    angle_var = T.fscalar()
    matrix = matrices[1]

    shared_rot_mat_expression = _rotation_matrices(angle)
    tensor_rot_mat_expression = _rotation_matrices(angle_var)

    f_shared = theano.function([], shared_rot_mat_expression)
    f_tensor = theano.function([angle_var], tensor_rot_mat_expression)
    matrix_shared = f_shared()
    matrix_tensor = f_tensor(angle)

    assert_array_equal(matrix, matrix_shared)
    assert_array_equal(matrix, matrix_tensor)


def test__mgrid__ogrid():

    fmgrid = np.mgrid[0:1:.1, 1:10:1., 10:100:10.]
    imgrid = np.mgrid[0:2:1, 1:10:1, 10:100:10]

    fogrid = np.ogrid[0:1:.1, 1:10:1., 10:100:10.]
    iogrid = np.ogrid[0:2:1, 1:10:1, 10:100:10]

    tfmgrid = _mgrid[0:1:.1, 1:10:1., 10:100:10.]
    timgrid = _mgrid[0:2:1, 1:10:1, 10:100:10]

    tfogrid = _ogrid[0:1:.1, 1:10:1., 10:100:10.]
    tiogrid = _ogrid[0:2:1, 1:10:1, 10:100:10]

    for g1, g2 in zip([fmgrid, imgrid, fogrid, iogrid],
                      [tfmgrid, timgrid, tfogrid, tiogrid]):
        for v1, v2 in zip(g1, g2):
            assert_array_almost_equal(v1, v2.eval(), decimal=6)
