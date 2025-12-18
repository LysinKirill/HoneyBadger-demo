import numpy as np
from typing import Tuple, Dict, Callable
import numpy.typing as npt


def spring_design(x: npt.NDArray) -> Tuple[float, npt.NDArray]:
    """
    Minimize weight of tension/compression spring
    x = [d, D, N] where:
    d = wire diameter
    D = mean coil diameter
    N = number of active coils
    """
    d, D, N = x[0], x[1], x[2]

    f = (N + 2) * D * d ** 2

    g1 = 1 - (D ** 3 * N) / (71785 * d ** 4)
    g2 = (4 * D ** 2 - d * D) / (12566 * (d ** 3 * D - d ** 4)) + 1 / (5108 * d ** 2) - 1
    g3 = 1 - (140.45 * d) / (D ** 2 * N)
    g4 = (D + d) / 1.5 - 1

    penalty = 0
    penalty += 1000 * max(0, -g1) ** 2
    penalty += 1000 * max(0, -g2) ** 2
    penalty += 1000 * max(0, -g3) ** 2
    penalty += 1000 * max(0, -g4) ** 2

    return f + penalty, np.array([g1, g2, g3, g4])


def pressure_vessel(x: npt.NDArray) -> Tuple[float, npt.NDArray]:
    """
    Minimize total cost of pressure vessel
    x = [Ts, Th, R, L] where:
    Ts = shell thickness
    Th = head thickness
    R = inner radius
    L = length of cylindrical section
    """
    Ts = 0.0625 * np.round(x[0] / 0.0625)
    Th = 0.0625 * np.round(x[1] / 0.0625)
    R, L = x[2], x[3]

    f = 0.6224 * Ts * R * L + 1.7781 * Th * R ** 2 + 3.1661 * Ts ** 2 * L + 19.84 * Ts ** 2 * R

    g1 = -Ts + 0.0193 * R
    g2 = -Th + 0.00954 * R
    g3 = -np.pi * R ** 2 * L - (4 / 3) * np.pi * R ** 3 + 1296000
    g4 = L - 240

    penalty = 0
    penalty += 1000 * max(0, -g1) ** 2
    penalty += 1000 * max(0, -g2) ** 2
    penalty += 1000 * max(0, -g3) ** 2
    penalty += 1000 * max(0, -g4) ** 2

    return f + penalty, np.array([g1, g2, g3, g4])


def welded_beam(x: npt.NDArray) -> Tuple[float, npt.NDArray]:
    """
    Minimize fabrication cost of welded beam
    x = [h, l, t, b] where:
    h = weld thickness
    l = weld length
    t = beam thickness
    b = beam width
    """
    h, l, t, b = x[0], x[1], x[2], x[3]

    P = 6000
    L = 14
    E = 30e6
    G = 12e6

    f = 1.10471 * h ** 2 * l + 0.04811 * t * b * (L + l)

    tau_prime = P / (np.sqrt(2) * h * l)
    tau_double_prime = (P * (L + l / 2) * np.sqrt(l ** 2 + (h + t) ** 2)) / \
                       (2 * np.sqrt(2) * h * l * (l ** 2 / 12 + (h + t) ** 2 / 4))
    tau = np.sqrt(tau_prime ** 2 + tau_double_prime ** 2 +
                  (tau_prime * tau_double_prime * l) / np.sqrt(l ** 2 + (h + t) ** 2))

    sigma = 6 * P * L / (t ** 2 * b)
    delta = 4 * P * L ** 3 / (E * t ** 3 * b)
    Pc = (4.013 * E * np.sqrt(t ** 2 * b ** 6 / 36) / L ** 2) * (1 - t / (2 * L) * np.sqrt(E / (4 * G)))

    g1 = tau - 13600
    g2 = sigma - 30000
    g3 = h - b
    g4 = 0.125 - h
    g5 = delta - 0.25
    g6 = P - Pc

    constraints = np.array([g1, g2, g3, g4, g5, g6])
    penalty = 1000 * np.sum(np.maximum(0, -constraints) ** 2)

    return f + penalty, constraints


def speed_reducer(x: npt.NDArray) -> Tuple[float, npt.NDArray]:
    """
    Minimize weight of speed reducer
    7 variables: b, m, z, l1, l2, d1, d2
    """
    b, m, z, l1, l2, d1, d2 = x[0], x[1], x[2], x[3], x[4], x[5], x[6]

    f = 0.7854 * b * m ** 2 * (3.3333 * z ** 2 + 14.9334 * z - 43.0934) - \
        1.508 * b * (d1 ** 2 + d2 ** 2) + 7.477 * (d1 ** 3 + d2 ** 3) + \
        0.7854 * (l1 * d1 ** 2 + l2 * d2 ** 2)

    g1 = 27 / (b * m ** 2 * z) - 1
    g2 = 397.5 / (b * m ** 2 * z ** 2) - 1
    g3 = 1.93 * l1 ** 3 / (m * z * d1 ** 4) - 1
    g4 = 1.93 * l2 ** 3 / (m * z * d2 ** 4) - 1
    g5 = np.sqrt((745 * l1 / (m * z)) ** 2 + 16.9e6) / (110 * d1 ** 3) - 1
    g6 = np.sqrt((745 * l2 / (m * z)) ** 2 + 157.5e6) / (85 * d2 ** 3) - 1
    g7 = m * z / 40 - 1
    g8 = 5 * m / b - 1
    g9 = b / (12 * m) - 1
    g10 = (1.5 * d1 + 1.9) / l1 - 1
    g11 = (1.1 * d2 + 1.9) / l2 - 1

    constraints = np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])
    penalty = 1000 * np.sum(np.maximum(0, -constraints) ** 2)

    return f + penalty, constraints


ENGINEERING_PROBLEMS: Dict[str, Dict] = {
    'Spring Design': {
        'function': lambda x: spring_design(x)[0],
        'bounds': [(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)],
        'dim': 3,
        'optimal': 0.012665,
        'description': 'Minimize weight of tension/compression spring'
    },
    'Pressure Vessel': {
        'function': lambda x: pressure_vessel(x)[0],
        'bounds': [(0.0625, 99 * 0.0625), (0.0625, 99 * 0.0625), (10, 200), (10, 200)],
        'dim': 4,
        'optimal': 5885.332,
        'description': 'Minimize fabrication cost of pressure vessel'
    },
    'Welded Beam': {
        'function': lambda x: welded_beam(x)[0],
        'bounds': [(0.1, 2.0), (0.1, 10.0), (0.1, 10.0), (0.1, 2.0)],
        'dim': 4,
        'optimal': 1.724852,
        'description': 'Minimize cost of welded beam'
    },
    'Speed Reducer': {
        'function': lambda x: speed_reducer(x)[0],
        'bounds': [(2.6, 3.6), (0.7, 0.8), (17, 28), (7.3, 8.3),
                   (7.3, 8.3), (2.9, 3.9), (5.0, 5.5)],
        'dim': 7,
        'optimal': 2994.471,
        'description': 'Minimize weight of speed reducer'
    }
}