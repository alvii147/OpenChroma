import numpy as np
from openchroma.imageops import cropImage

img = np.arange(0, 100, 5).reshape(4, 5)
print(img)
print('')

cropped = cropImage(img, (1, 1), bottom_right=(3, 3))
print(cropped)