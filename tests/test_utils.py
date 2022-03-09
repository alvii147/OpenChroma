import numpy as np
import pytest

from openchroma.utils import (
    is_array_like,
    require_array_like,
    is_shape,
    require_shape,
    is_dim,
    require_dim,
    is_axis_size,
    require_axis_size,
)

is_array_like_parameters = [
    [[2, 4, 6], True],
    [np.array([12, 15]), True],
    [2, False],
    ['squidward', False],
]


@pytest.mark.parametrize('arr, result', is_array_like_parameters)
def test_is_array_like(arr, result):
    assert is_array_like(arr) == result

    if result is True:
        require_array_like(arr)
    else:
        with pytest.raises(TypeError):
            require_array_like(arr)


is_shape_parameters = [
    [[2, 5], 2, True],
    [np.arange(0, 20).reshape(10, 2), (10, 2), True],
    [[2, 4, 6], (14,), False],
    [np.array([[12, 15], [8, -1]]), (2, 3), False],
]


@pytest.mark.parametrize('arr, shape, result', is_shape_parameters)
def test_is_shape(arr, shape, result):
    assert is_shape(arr, shape) == result

    if result is True:
        require_shape(arr, shape)
    else:
        with pytest.raises(ValueError):
            require_shape(arr, shape)


is_dim_parameters = [
    [np.zeros(3), 1, True],
    [np.zeros((2, 3, 8)), 3, True],
    [np.zeros(9), 4, False],
    [np.zeros((7, 8)), 1, False],
]


@pytest.mark.parametrize('arr, dim, result', is_dim_parameters)
def test_is_dim(arr, dim, result):
    assert is_dim(arr, dim) == result

    if result is True:
        require_dim(arr, dim)
    else:
        with pytest.raises(ValueError):
            require_dim(arr, dim)


is_axis_size_parameters = [
    [np.zeros(3), 3, -1, True],
    [np.zeros((6, 2, 3)), 2, 1, True],
    [np.zeros(9), 4, -1, False],
    [np.zeros((7, 8)), 3, 1, False],
]


@pytest.mark.parametrize('arr, size, axis, result', is_axis_size_parameters)
def test_is_axis_size(arr, size, axis, result):
    assert is_axis_size(arr, size, axis=axis) == result

    if result is True:
        require_axis_size(arr, size, axis=axis)
    else:
        with pytest.raises(ValueError):
            require_axis_size(arr, size, axis=axis)
