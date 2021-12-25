import numpy as np
from openchroma.colorspace import RGB_to_CMYK, CMYK_to_RGB

arr1 = np.array([34, 21, 121])
arr2 = np.arange(0, 100, 9).reshape(4, 3)
arr3 = np.arange(0, 100, 9).reshape(2, 2, 3)

print(arr1)
print('')
print(arr2)
print('')
print(arr3)
print('')

print(RGB_to_CMYK(arr1))
print('')
print(RGB_to_CMYK(arr1, 3))
print('')
print(RGB_to_CMYK(arr2))
print('')
print(RGB_to_CMYK(arr2, precision=4))
print('')
print(RGB_to_CMYK(arr3))