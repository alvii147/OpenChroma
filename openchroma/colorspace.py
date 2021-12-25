import numpy as np

from .utils import requireArrayLike, requireAxisDim

RGB_SHAPE = (3,)
RGB_RANGE = (0, 255)
CMYK_SHAPE = (4,)
CMYK_RANGE = (0, 100)

def _RGB_to_CMYK(rgb, precision=2):
    '''
    Convert array from RGB space to CMYK space values.

    Parameters
    ----------
    rgb : array-like
        1D input array in RGB space.
    precision : int, optional
        Number of decimal places to round values to.

    Returns
    -------
    cmyk : numpy.ndarray
        1D output array in CMYK space.
    '''

    # truncate values between 0 and 255
    rgb = np.array(rgb, dtype=np.float64)
    for i in range(RGB_SHAPE[0]):
        rgb[i] = max(rgb[i], RGB_RANGE[0])
        rgb[i] = min(rgb[i], RGB_RANGE[1])

    # convert to CMYK space
    k = 1 - (np.max(rgb) / 255)
    cmyk = np.zeros(CMYK_SHAPE, dtype=np.float64)
    for i in range(CMYK_SHAPE[0]):
        if i < 3:
            val = ((1 - (rgb[i] / 255) - k) / (1 - k)) * 100
        else:
            val = k * 100

        cmyk[i] = max(round(val, precision), 0)

    return cmyk

def RGB_to_CMYK(rgb, precision=2):
    '''
    Convert array from RGB space to CMYK space values on the last axis.

    Parameters
    ----------
    rgb : array-like
        Input array in RGB space.
    precision : int, optional
        Number of decimal places to round values to.

    Returns
    -------
    cmyk : numpy.ndarray
        Output array in CMYK space.
    '''

    # check if input is array-like
    requireArrayLike(rgb, var_name='rgb')
    # check if last axis is 3-dimensional
    requireAxisDim(rgb, RGB_SHAPE[-1], -1, var_name='rgb')

    # convert to CMYK along last axis
    return np.apply_along_axis(_RGB_to_CMYK, -1, rgb, precision=precision)

def _CMYK_to_RGB(cmyk, precision=2):
    '''
    Convert array from CMYK space to RGB space values.

    Parameters
    ----------
    cmyk : array-like
        1D input array in CMYK space.
    precision : int, optional
        Number of decimal places to round values to.

    Returns
    -------
    rgb : numpy.ndarray
        1D output array in RGB space.
    '''

    # truncate values between 0 and 100
    cmyk = np.array(cmyk, dtype=np.float64)
    for i in range(CMYK_SHAPE[0]):
        cmyk[i] = max(cmyk[i], CMYK_RANGE[0])
        cmyk[i] = min(cmyk[i], CMYK_RANGE[1])

    # convert to RGB space
    rgb = np.zeros(RGB_SHAPE, dtype=np.float64)
    k = cmyk[3]
    for i in range(RGB_SHAPE[0]):
        val = (1 - (cmyk[i] / 100)) * (1 - (k / 100)) * 255
        rgb[i] = max(round(val, precision), 0)

    return rgb

def CMYK_to_RGB(cmyk, precision=2):
    '''
    Convert array from CMYK space to RGB space values on the last axis.

    Parameters
    ----------
    cmyk : array-like
        Input array in RGB space.
    precision : int, optional
        Number of decimal places to round values to.

    Returns
    -------
    rgb : numpy.ndarray
        Output array in rgb space.
    '''

    # check if input is array-like
    requireArrayLike(cmyk, var_name='cmyk')
    # check if last axis is 3-dimensional
    requireAxisDim(cmyk, CMYK_SHAPE[-1], -1, var_name='cmyk')

    # convert to RGB along last axis
    return np.apply_along_axis(_CMYK_to_RGB, -1, cmyk, precision=precision)