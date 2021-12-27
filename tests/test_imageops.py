import numpy as np
import pytest

from openchroma.imageops import (
    openImage,
    saveImage,
    splitChannels,
    combineChannels,
    slidingWindowOperation,
)

def generateRandomImage(height, width):
    img = np.around(np.random.rand(height, height, 3) * 255)

    return img

def test_openImage_saveImage():
    img = openImage('img/popcat.png')
    saveImage(img, 'img/popcat2.png')

def test_splitChannels_combineChannels():
    img = generateRandomImage(100, 100)
    img_shape = np.shape(img)

    r, g, b = splitChannels(img)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            assert r[i][j] == img[i][j][0]

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            assert g[i][j] == img[i][j][1]

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            assert b[i][j] == img[i][j][2]

    img_combined = combineChannels(r, g, b)

    assert np.array_equal(img, img_combined)

slidingWindowOperation_parameters = [
    [
        np.array([
            [0, 9, 18],
            [27, 36, 45],
            [54, 63, 72],
            [81, 90, 99],
        ], dtype=np.float64),
        (2, 2),
        np.mean,
        np.float64,
        False,
        np.array([
            [18, 27],
            [45, 54],
            [72, 81],
        ], dtype=np.float64),
    ],
    [
        np.array([
            [0, 9, 18],
            [27, 36, 45],
            [54, 63, 72],
            [81, 90, 99],
        ], dtype=np.float64),
        (2, 2),
        np.mean,
        np.float64,
        True,
        np.array([
            [0, 4.5, 13.5, 18],
            [13.5, 18, 27, 31.5],
            [40.5, 45, 54, 58.5],
            [67.5, 72, 81, 85.5],
            [81, 85.5, 94.5, 99],
        ], dtype=np.float64),
    ],
]

@pytest.mark.parametrize('img, window, op, dtype, edges, output', slidingWindowOperation_parameters)
def test_slidingWindowOperation(img, window, op, dtype, edges, output):
    output_computed = slidingWindowOperation(img, window, op=op, dtype=dtype, edges=edges)
    assert np.array_equal(output, output_computed)