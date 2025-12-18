import numpy as np
from typing import Tuple, Callable
import numpy.typing as npt

FunctionInfo = Tuple[Callable[[npt.NDArray], float], Tuple[float, float], npt.NDArray]


def sphere(x: npt.NDArray) -> float:
    """F1: Sphere function - Unimodal"""
    return np.sum(x ** 2)


def schwefel_222(x: npt.NDArray) -> float:
    """F10: Schwefel 2.22 function - Unimodal"""
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def rastrigin(x: npt.NDArray) -> float:
    """F11: Rastrigin function - Multimodal"""
    A = 10
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def ackley(x: npt.NDArray) -> float:
    """F14: Ackley function - Multimodal"""
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.e


def griewank(x: npt.NDArray) -> float:
    """F15: Griewank function - Multimodal"""
    sum_part = np.sum(x ** 2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_part - prod_part + 1



TEST_FUNCTIONS_2D = {
    'Sphere': (sphere, (-100, 100), np.array([0, 0])),
    'Rastrigin': (rastrigin, (-5.12, 5.12), np.array([0, 0])),
    'Ackley': (ackley, (-32.768, 32.768), np.array([0, 0])),
    'Griewank': (griewank, (-600, 600), np.array([0, 0])),
}

TEST_FUNCTIONS_3D = {
    'Sphere 3D': (sphere, (-100, 100), np.array([0, 0, 0])),
    'Rastrigin 3D': (rastrigin, (-5.12, 5.12), np.array([0, 0, 0])),
    'Ackley 3D': (ackley, (-32.768, 32.768), np.array([0, 0, 0])),
}


def get_function_2d_grid(func: Callable, bounds: Tuple[float, float],
                         resolution: int = 100) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    return X, Y, Z