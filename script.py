import numpy as np
from openchroma.imageops import slidingWindowOperation

img = np.arange(0, 100, 9).reshape(4, 3)
print(img)
print('')

o = slidingWindowOperation(img, (2, 2), op=np.mean, edges=False)
print(o)