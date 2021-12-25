import numpy as np

def isArrayLike(arr):
    '''
    Check if object is array-like.

    Parameters
    ----------
    arr : any
        Object to be tested.

    Returns
    -------
    is_array_like : bool
        Indicates whether given object is array-like.
    '''

    return isinstance(arr, (list, tuple, np.ndarray))

def requireArrayLike(arr, var_name='Array', exception=TypeError):
    '''
    Raise exception if object is array-like.

    Parameters
    ----------
    arr : any
        Object to be tested.
    var_name : str, optional
        Name of the variable to be tested, used to construct exception message.
    exception : Exception, optional
        Exception to raise if test fails.
    '''

    if not isArrayLike(arr):
        message = f'`{var_name}` must be an array-like object'
        raise exception(message)

def isShape(arr, shape):
    '''
    Check if array is of desired shape.

    Parameters
    ----------
    arr : array-like
        Array to be tested.
    shape : array-like
        Desired shape.

    Returns
    -------
    is_shape : bool
        Indicates whether given array is of desired shape.
    '''

    arr_shape = np.shape(arr)
    if np.isscalar(shape):
        shape = (shape,)

    return np.array_equal(arr_shape, shape)

def requireShape(arr, shape, var_name='Array', exception=ValueError):
    '''
    Raise exception if array is of desired shape.

    Parameters
    ----------
    arr : any
        Array to be tested.
    shape : array-like
        Desired shape.
    var_name : str, optional
        Name of the variable to be tested, used to construct exception message.
    exception : Exception, optional
        Exception to raise if test fails.
    '''

    if not isShape(arr, shape):
        message = f'`{var_name}` must be of shape {shape}'
        raise exception(message)

def isAxisDim(arr, dim, axis=0):
    '''
    Check if given axis of array is of desired dimension.

    Parameters
    ----------
    arr : any
        Array to be tested.
    dim : int
        Desired dimension.
    axis : int, optional
        Axis of array to test for desired dimension.

    Returns
    -------
    is_axis_dimm : bool
        Indicates whether given axis of array is of desired dimension.
    '''

    arr_shape = np.shape(arr)

    return arr_shape[axis] == dim

def requireAxisDim(arr, dim, axis=0, var_name='Array', exception=ValueError):
    '''
    Raise exception if given axis of array is of desired dimension.

    Parameters
    ----------
    arr : any
        Array to be tested.
    dim : int
        Desired dimension.
    axis : int, optional
        Axis of array to test for desired dimension.
    var_name : str, optional
        Name of the variable to be tested, used to construct exception message.
    exception : Exception, optional
        Exception to raise if test fails.

    Returns
    -------
    is_axis_dimm : bool
        Indicates whether given axis of array is of desired dimension.
    '''

    if not isAxisDim(arr, dim, axis=axis):
        message = f'`{var_name}` must be {dim}-dimensional for axis {axis}'
        raise exception(message)