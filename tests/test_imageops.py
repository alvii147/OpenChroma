import numpy as np

from openchroma.imageops import (
    openImage,
    saveImage,
    splitChannels,
    combineChannels,
)

def generateRandomImage(height, width):
    img = np.around(np.random.rand(height, height, 3) * 255)

    return img

def test_openImage_saveImage():
    img = openImage('img/popcat.png')
    saveImage(img, 'img/popcat2.png')

def test_splitChannels_combineChannels():
    img = generateRandomImage(100, 100)
    r, g, b = splitChannels(img)
    img_combined = combineChannels(r, g, b)

    assert np.array_equal(img, img_combined)