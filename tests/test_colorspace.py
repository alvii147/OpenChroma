import numpy as np
import numpy.testing as npt
import pytest

from openchroma.colorspace import RGB_to_CMYK, CMYK_to_RGB

RGB_to_CMYK_parameters = [
    [
        np.array([117.0, 95.0, 143.0], dtype=np.float64),
        np.array([18.0, 34.0, 0.0, 44.0], dtype=np.float64),
    ],
    [
        np.array([13.0, 31.0, 52.0], dtype=np.float64),
        np.array([75.0, 40.0, 0.0, 80.0], dtype=np.float64),
    ],
    [
        np.array(
            [
                [58.0, 150.0, 176.0],
                [229.0, 168.0, 91.0],
                [13.0, 99.0, 202.0],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [67.0, 15.0, 0.0, 31.0],
                [0.0, 27.0, 60.0, 10.0],
                [94.0, 51.0, 0.0, 21.0],
            ],
            dtype=np.float64,
        ),
    ],
    [
        np.array(
            [
                [
                    [98.0, 91.0, 36.0],
                    [200.0, 101.0, 164.0],
                    [168.0, 191.0, 214.0],
                ],
                [
                    [31.0, 44.0, 222.0],
                    [82.0, 168.0, 77.0],
                    [0.0, 17.0, 44.0],
                ],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [
                    [0.0, 7.0, 63.0, 62.0],
                    [0.0, 50.0, 18.0, 22.0],
                    [21.0, 11.0, 0.0, 16.0],
                ],
                [
                    [86.0, 80.0, 0.0, 13.0],
                    [51.0, 0.0, 54.0, 34.0],
                    [100.0, 61.0, 0.0, 83.0],
                ],
            ],
            dtype=np.float64,
        ),
    ],
]


@pytest.mark.parametrize('rgb, cmyk_expected', RGB_to_CMYK_parameters)
def test_RGB_to_CMYK(rgb, cmyk_expected):
    cmyk_computed = RGB_to_CMYK(rgb, precision=0)
    npt.assert_almost_equal(cmyk_expected, cmyk_computed)


CMYK_to_RGB_parameters = [
    [
        np.array([89.0, 37.0, 79.0, 33.0], dtype=np.float64),
        np.array([19.0, 108.0, 36.0], dtype=np.float64),
    ],
    [
        np.array([84.0, 23.0, 0.0, 89.0], dtype=np.float64),
        np.array([4.0, 22.0, 28.0], dtype=np.float64),
    ],
    [
        np.array(
            [
                [19.0, 49.0, 55.0, 21.0],
                [0.0, 16.0, 78.0, 40.0],
                [42.0, 100.0, 8.0, 17.0],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [163.0, 103.0, 91.0],
                [153.0, 129.0, 34.0],
                [123.0, 0.0, 195.0],
            ],
            dtype=np.float64,
        ),
    ],
    [
        np.array(
            [
                [
                    [100.0, 9.0, 67.0, 20.0],
                    [88.0, 78.0, 31.0, 22.0],
                    [13.0, 17.0, 95.0, 82.0],
                ],
                [
                    [13.0, 17.0, 6.0, 96.0],
                    [83.0, 91.0, 96.0, 4.0],
                    [33.0, 54.0, 62.0, 30.0],
                ],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [
                    [0.0, 186.0, 67.0],
                    [24.0, 44.0, 137.0],
                    [40.0, 38.0, 2.0],
                ],
                [
                    [9.0, 8.0, 10.0],
                    [42.0, 22.0, 10.0],
                    [120.0, 82.0, 68.0],
                ],
            ],
            dtype=np.float64,
        ),
    ],
]


@pytest.mark.parametrize('cmyk, rgb_expected', CMYK_to_RGB_parameters)
def test_CMYK_to_RGB(cmyk, rgb_expected):
    rgb_computed = CMYK_to_RGB(cmyk, precision=0)
    npt.assert_almost_equal(rgb_expected, rgb_computed)
