import numpy as np
import pytest

from openchroma.utils import (
    isArrayLike,
    requireArrayLike,
    isShape,
    requireShape,
    isDim,
    requireDim,
    isAxisSize,
    requireAxisSize,
)

isArrayLike_parameters = [
    [[2, 4, 6], True],
    [np.array([12, 15]), True],
    [2, False],
    ['squidward', False],
]

@pytest.mark.parametrize('arr, result', isArrayLike_parameters)
def test_isArrayLike(arr, result):
    assert isArrayLike(arr) == result

    if result == True:
        requireArrayLike(arr)
    else:
        with pytest.raises(TypeError):
            requireArrayLike(arr)

isShape_parameters = [
    [[2, 5], 2, True],
    [np.arange(0, 20).reshape(10, 2), (10, 2), True],
    [[2, 4, 6], (14,), False],
    [np.array([[12, 15], [8, -1]]), (2, 3), False],
]

@pytest.mark.parametrize('arr, shape, result', isShape_parameters)
def test_isShape(arr, shape, result):
    assert isShape(arr, shape) == result

    if result == True:
        requireShape(arr, shape)
    else:
        with pytest.raises(ValueError):
            requireShape(arr, shape)

isDim_parameters = [
    [np.zeros(3), 1, True],
    [np.zeros((2, 3, 8)), 3, True],
    [np.zeros(9), 4, False],
    [np.zeros((7, 8)), 1, False],
]

@pytest.mark.parametrize('arr, dim, result', isDim_parameters)
def test_isDim(arr, dim, result):
    assert isDim(arr, dim) == result

    if result == True:
        requireDim(arr, dim)
    else:
        with pytest.raises(ValueError):
            requireDim(arr, dim)

isAxisSize_parameters = [
    [np.zeros(3), 3, -1, True],
    [np.zeros((6, 2, 3)), 2, 1, True],
    [np.zeros(9), 4, -1, False],
    [np.zeros((7, 8)), 3, 1, False],
]

@pytest.mark.parametrize('arr, size, axis, result', isAxisSize_parameters)
def test_isAxisDim(arr, size, axis, result):
    assert isAxisSize(arr, size, axis=axis) == result

    if result == True:
        requireAxisSize(arr, size, axis=axis)
    else:
        with pytest.raises(ValueError):
            requireAxisSize(arr, size, axis=axis)
