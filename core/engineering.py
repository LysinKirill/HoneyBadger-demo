from typing import Tuple, Dict, List

import numpy as np
import numpy.typing as npt


class EngineeringProblem:
    def __init__(self, name: str, description: str, diagram: str,
                 variables: List[dict], constraints: List[dict]):
        self.name = name
        self.description = description
        self.diagram = diagram
        self.variables = variables
        self.constraints = constraints

    def get_variable_info(self, idx: int) -> dict:
        return self.variables[idx]

    def get_constraint_info(self, idx: int) -> dict:
        return self.constraints[idx]


def spring_design(x: npt.NDArray) -> Tuple[float, npt.NDArray, List[bool]]:
    d, D, N = x[0], x[1], x[2]

    f = (N + 2) * D * d ** 2

    g1 = 1 - (D ** 3 * N) / (71785 * d ** 4)
    g2 = (4 * D ** 2 - d * D) / (12566 * (d ** 3 * D - d ** 4)) + 1 / (5108 * d ** 2) - 1
    g3 = 1 - (140.45 * d) / (D ** 2 * N)
    g4 = (D + d) / 1.5 - 1

    constraints = np.array([g1, g2, g3, g4])
    satisfied = constraints <= 0

    penalty = 1000 * np.sum(np.maximum(0, -constraints) ** 2)

    return f + penalty, constraints, satisfied.tolist()


def pressure_vessel(x: npt.NDArray) -> Tuple[float, npt.NDArray, List[bool]]:
    Ts = 0.0625 * np.round(x[0] / 0.0625)
    Th = 0.0625 * np.round(x[1] / 0.0625)
    R, L = x[2], x[3]

    f = 0.6224 * Ts * R * L + 1.7781 * Th * R ** 2 + 3.1661 * Ts ** 2 * L + 19.84 * Ts ** 2 * R

    g1 = -Ts + 0.0193 * R
    g2 = -Th + 0.00954 * R
    g3 = -np.pi * R ** 2 * L - (4 / 3) * np.pi * R ** 3 + 1296000
    g4 = L - 240

    constraints = np.array([g1, g2, g3, g4])
    satisfied = constraints <= 0

    penalty = 1000 * np.sum(np.maximum(0, -constraints) ** 2)

    return f + penalty, constraints, satisfied.tolist()


def welded_beam(x: npt.NDArray) -> Tuple[float, npt.NDArray, List[bool]]:
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
    satisfied = constraints <= 0
    penalty = 1000 * np.sum(np.maximum(0, -constraints) ** 2)

    return f + penalty, constraints, satisfied.tolist()


def speed_reducer(x: npt.NDArray) -> Tuple[float, npt.NDArray, List[bool]]:
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
    satisfied = constraints <= 0
    penalty = 1000 * np.sum(np.maximum(0, -constraints) ** 2)

    return f + penalty, constraints, satisfied.tolist()


def spring_design_objective(x: npt.NDArray) -> float:
    """Wrapper that returns only the objective value"""
    return spring_design(x)[0]

def pressure_vessel_objective(x: npt.NDArray) -> float:
    return pressure_vessel(x)[0]

def welded_beam_objective(x: npt.NDArray) -> float:
    return welded_beam(x)[0]

def speed_reducer_objective(x: npt.NDArray) -> float:
    return speed_reducer(x)[0]


SPRING_PROBLEM = EngineeringProblem(
    name="Spring Design",
    description="Design a helical compression spring for minimum weight subject to:\n"
                "1. Shear stress constraint\n"
                "2. Surge frequency constraint\n"
                "3. Deflection constraint\n"
                "4. Diameter constraint",
    diagram="┌──────────────────────┐\n"
            "│     Spring Design    │\n"
            "│                      │\n"
            "│      ╭───╮           │\n"
            "│   d  │   │           │\n"
            "│  ───►│   │ D         │\n"
            "│      ╰───╯◄─────────►│\n"
            "│                      │\n"
            "│  N active coils      │\n"
            "│  ╭─╮╭─╮╭─╮╭─╮╭─╮     │\n"
            "└──────────────────────┘",
    variables=[
        {"name": "d", "description": "Wire diameter (inches)", "unit": "in", "symbol": "d"},
        {"name": "D", "description": "Mean coil diameter (inches)", "unit": "in", "symbol": "D"},
        {"name": "N", "description": "Number of active coils", "unit": "", "symbol": "N"}
    ],
    constraints=[
        {"name": "Shear Stress", "description": "τ ≤ 13600 psi", "formula": "g₁ = 1 - D³N/(71785d⁴) ≤ 0"},
        {"name": "Surge Frequency", "description": "Avoid natural frequency resonance",
         "formula": "g₂ = (4D²-dD)/(12566(d³D-d⁴)) + 1/(5108d²) - 1 ≤ 0"},
        {"name": "Deflection", "description": "δ ≥ 0.1 inches", "formula": "g₃ = 1 - (140.45d)/(D²N) ≤ 0"},
        {"name": "Diameter", "description": "Outer diameter ≤ 1.5 in", "formula": "g₄ = (D+d)/1.5 - 1 ≤ 0"}
    ]
)

PRESSURE_VESSEL_PROBLEM = EngineeringProblem(
    name="Pressure Vessel",
    description="Design a cylindrical pressure vessel with hemispherical heads\n"
                "for minimum fabrication cost subject to ASME constraints.",
    diagram="┌─────────────────────────┐\n"
            "│     Pressure Vessel     │\n"
            "│                         │\n"
            "│     ┌─────────────┐     │\n"
            "│     │    Th       │     │\n"
            "│  Ts │             │ Ts  │\n"
            "│  ▲  │             │ ▲   │\n"
            "│  │  │      R      │ │   │\n"
            "│  │  │   ◄──────►  │ │   │\n"
            "│  └──┼─────────────┼─┘   │\n"
            "│     │             │     │\n"
            "│     │     L       │     │\n"
            "│     │   ◄─────►   │     │\n"
            "│     └─────────────┘     │\n"
            "└─────────────────────────┘",
    variables=[
        {"name": "Ts", "description": "Shell thickness (inches)", "unit": "in", "symbol": "Tₛ"},
        {"name": "Th", "description": "Head thickness (inches)", "unit": "in", "symbol": "Tₕ"},
        {"name": "R", "description": "Inner radius (inches)", "unit": "in", "symbol": "R"},
        {"name": "L", "description": "Cylinder length (inches)", "unit": "in", "symbol": "L"}
    ],
    constraints=[
        {"name": "Shell Thickness", "description": "Ts ≥ 0.0193R", "formula": "g₁ = -Tₛ + 0.0193R ≤ 0"},
        {"name": "Head Thickness", "description": "Th ≥ 0.00954R", "formula": "g₂ = -Tₕ + 0.00954R ≤ 0"},
        {"name": "Volume", "description": "Volume ≥ 750 ft³", "formula": "g₃ = -πR²L - (4/3)πR³ + 1296000 ≤ 0"},
        {"name": "Length", "description": "L ≤ 240 in", "formula": "g₄ = L - 240 ≤ 0"}
    ]
)
WELDED_BEAM_PROBLEM = EngineeringProblem(
    name="Welded Beam Design",
    description="Design a welded beam for minimum cost subject to:\n"
                "1. Shear stress constraint\n"
                "2. Bending stress constraint\n"
                "3. Buckling load constraint\n"
                "4. Deflection constraint\n"
                "5. Side constraints",
    diagram="┌──────────────────────┐\n"
            "│     Welded Beam      │\n"
            "│                      │\n"
            "│    P = 6000 lb       │\n"
            "│      ↓               │\n"
            "│  ┌───┬─────┬───┐     │\n"
            "│  │ h │  l  │ h │     │\n"
            "│  │   │     │   │     │\n"
            "│  └───┴─────┴───┘     │\n"
            "│        t             │\n"
            "│        ▲             │\n"
            "│        │ b           │\n"
            "│        ▼             │\n"
            "└──────────────────────┘",
    variables=[
        {"name": "h", "description": "Weld thickness (inches)", "unit": "in", "symbol": "h"},
        {"name": "l", "description": "Weld length (inches)", "unit": "in", "symbol": "l"},
        {"name": "t", "description": "Beam thickness (inches)", "unit": "in", "symbol": "t"},
        {"name": "b", "description": "Beam width (inches)", "unit": "in", "symbol": "b"}
    ],
    constraints=[
        {"name": "Shear Stress", "description": "τ ≤ 13600 psi", "formula": "g₁ = τ - 13600 ≤ 0"},
        {"name": "Bending Stress", "description": "σ ≤ 30000 psi", "formula": "g₂ = σ - 30000 ≤ 0"},
        {"name": "Geometry", "description": "h ≤ b", "formula": "g₃ = h - b ≤ 0"},
        {"name": "Min Weld", "description": "h ≥ 0.125 in", "formula": "g₄ = 0.125 - h ≤ 0"},
        {"name": "Deflection", "description": "δ ≤ 0.25 in", "formula": "g₅ = δ - 0.25 ≤ 0"},
        {"name": "Buckling", "description": "P ≤ P_c", "formula": "g₆ = P - P_c ≤ 0"}
    ]
)

SPEED_REDUCER_PROBLEM = EngineeringProblem(
    name="Speed Reducer Design",
    description="Design a speed reducer for minimum weight subject to:\n"
                "1. Bending stress constraints\n"
                "2. Surface stress constraints\n"
                "3. Transverse deflection constraints\n"
                "4. Geometry constraints",
    diagram="┌─────────────────────────┐\n"
            "│     Speed Reducer       │\n"
            "│                         │\n"
            "│   Input Shaft ◄─┐       │\n"
            "│       │         │       │\n"
            "│  Gear 1         │       │\n"
            "│    m,z          │       │\n"
            "│       │  Gear 2 │       │\n"
            "│  ┌────▼────┐    │       │\n"
            "│  │  Bearing│    │       │\n"
            "│  │   d₁,l₁ │    │       │\n"
            "│  └─────────┘    │       │\n"
            "│  Output Shaft ◄─┘       │\n"
            "└─────────────────────────┘",
    variables=[
        {"name": "b", "description": "Face width (inches)", "unit": "in", "symbol": "b"},
        {"name": "m", "description": "Module (inches)", "unit": "in", "symbol": "m"},
        {"name": "z", "description": "Number of teeth", "unit": "", "symbol": "z"},
        {"name": "l₁", "description": "Length of shaft 1 (inches)", "unit": "in", "symbol": "l₁"},
        {"name": "l₂", "description": "Length of shaft 2 (inches)", "unit": "in", "symbol": "l₂"},
        {"name": "d₁", "description": "Diameter of shaft 1 (inches)", "unit": "in", "symbol": "d₁"},
        {"name": "d₂", "description": "Diameter of shaft 2 (inches)", "unit": "in", "symbol": "d₂"}
    ],
    constraints=[
        {"name": "Bending Stress 1", "description": "σ₁ ≤ allowable", "formula": "g₁ = 27/(bm²z) - 1 ≤ 0"},
        {"name": "Bending Stress 2", "description": "σ₂ ≤ allowable", "formula": "g₂ = 397.5/(bm²z²) - 1 ≤ 0"},
        {"name": "Surface Stress 1", "description": "Contact stress limit", "formula": "g₃ = 1.93l₁³/(mzd₁⁴) - 1 ≤ 0"},
        {"name": "Surface Stress 2", "description": "Contact stress limit", "formula": "g₄ = 1.93l₂³/(mzd₂⁴) - 1 ≤ 0"},
        {"name": "Deflection 1", "description": "Shaft 1 deflection", "formula": "g₅ = √((745l₁/(mz))² + 16.9e6)/(110d₁³) - 1 ≤ 0"},
        {"name": "Deflection 2", "description": "Shaft 2 deflection", "formula": "g₆ = √((745l₂/(mz))² + 157.5e6)/(85d₂³) - 1 ≤ 0"},
        {"name": "Geometry 1", "description": "mz/40 ≤ 1", "formula": "g₇ = mz/40 - 1 ≤ 0"},
        {"name": "Geometry 2", "description": "5m/b ≤ 1", "formula": "g₈ = 5m/b - 1 ≤ 0"},
        {"name": "Geometry 3", "description": "b/(12m) ≤ 1", "formula": "g₉ = b/(12m) - 1 ≤ 0"},
        {"name": "Geometry 4", "description": "(1.5d₁ + 1.9)/l₁ ≤ 1", "formula": "g₁₀ = (1.5d₁ + 1.9)/l₁ - 1 ≤ 0"},
        {"name": "Geometry 5", "description": "(1.1d₂ + 1.9)/l₂ ≤ 1", "formula": "g₁₁ = (1.1d₂ + 1.9)/l₂ - 1 ≤ 0"}
    ]
)

ENGINEERING_PROBLEMS: Dict[str, Dict] = {
    'Spring Design': {
        'problem': SPRING_PROBLEM,
        'function': spring_design,
        'bounds': [(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)],
        'objective': spring_design_objective,
        'dim': 3
    },
    'Pressure Vessel': {
        'problem': PRESSURE_VESSEL_PROBLEM,
        'function': pressure_vessel,
        'objective': pressure_vessel_objective,
        'bounds': [(0.0625, 99 * 0.0625), (0.0625, 99 * 0.0625), (10, 200), (10, 200)],
        'dim': 4
    },
    'Welded Beam': {
        'problem': WELDED_BEAM_PROBLEM,
        'function': welded_beam,
        'objective': welded_beam_objective,
        'bounds': [(0.1, 2.0), (0.1, 10.0), (0.1, 10.0), (0.1, 2.0)],
        'dim': 4
    },
    'Speed Reducer': {
        'problem': SPEED_REDUCER_PROBLEM,
        'function': speed_reducer,
        'objective': speed_reducer_objective,
        'bounds': [(2.6, 3.6), (0.7, 0.8), (17, 28), (7.3, 8.3),
                   (7.3, 8.3), (2.9, 3.9), (5.0, 5.5)],
        'dim': 7
    }
}