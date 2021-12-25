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

    is_array_like = isinstance(arr, (list, tuple, np.ndarray))

    return is_array_like


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

    is_shape = np.array_equal(arr_shape, shape)

    return is_shape


def requireShape(arr, shape, var_name='Array', exception=ValueError):
    '''
    Raise exception if array is not of desired shape.

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


def isDim(arr, dim):
    '''
    Check if array is of desired shape.

    Parameters
    ----------
    arr : array-like
        Array to be tested.
    dim : int
        Desired dimension.

    Returns
    -------
    is_dim : bool
        Indicates whether given array is of desired dimension.
    '''

    arr_shape = np.shape(arr)
    is_dim = len(arr_shape) == dim

    return is_dim


def requireDim(arr, dim, var_name='Array', exception=ValueError):
    '''
    Raise exception if array is not of desired dimension.

    Parameters
    ----------
    arr : any
        Array to be tested.
    dim : int
        Desired dimension.
    var_name : str, optional
        Name of the variable to be tested, used to construct exception message.
    exception : Exception, optional
        Exception to raise if test fails.
    '''

    if not isDim(arr, dim):
        message = f'`{var_name}` must be {dim}-dimensional'
        raise exception(message)


def isAxisSize(arr, size, axis=0):
    '''
    Check if given axis of array is of desired size.

    Parameters
    ----------
    arr : any
        Array to be tested.
    size : int
        Desired size.
    axis : int, optional
        Axis of array to test for desired size.

    Returns
    -------
    is_axis_size : bool
        Indicates whether given axis of array is of desired size.
    '''

    arr_shape = np.shape(arr)
    is_axis_size = arr_shape[axis] == size

    return is_axis_size


def requireAxisSize(arr, size, axis=0, var_name='Array', exception=ValueError):
    '''
    Raise exception if given axis of array is not of desired size.

    Parameters
    ----------
    arr : any
        Array to be tested.
    size : int
        Desired size.
    axis : int, optional
        Axis of array to test for desired size.
    var_name : str, optional
        Name of the variable to be tested, used to construct exception message.
    exception : Exception, optional
        Exception to raise if test fails.
    '''

    if not isAxisSize(arr, size, axis=axis):
        message = f'`{var_name}` must be of size {size} at axis {axis}'
        raise exception(message)
