# Honey Badger Algorithm Visualizer

A Python-based visualization tool for the Honey Badger Algorithm (HBA) metaheuristic optimization method with graphical interface. This implementation follows the original HBA paper by Hashim et al. (2022) and applies it to both benchmark functions and real-world engineering design problems.

## Project Structure
```tree
project/
├── core/
│ ├── honey_badger.py   # Main HBA implementation
│ ├── functions.py      # Benchmark test functions
│ └── engineering.py    # Engineering problem definitions
├── ui/
│ ├── main_window.py    # Main application window
│ ├── plot_2d.py        # 2D visualization window
│ ├── plot_3d.py        # 3D visualization window
│ └── engineering_visualizer.py # Engineering problem interface
├── config.py           # Configuration parameters
├── main.py             # Application entry point
└── requirements.txt    # Dependencies
```

## Installation
1. Ensure Python 3.8 or higher is installed
2. Install required dependencies:
```bash
pip install -r requirements.txt
```
### Dependencies
- PyQt6 >= 6.5.0
- numpy >= 1.24.0
- pyqtgraph >= 0.13.0

## Usage
### Run the application:

```bash
python main.py
```

### Application Features
#### Main Interface:

- Select between 2D benchmark functions, 3D benchmark functions, or engineering design problems
- Configure algorithm parameters (population size, iterations)

#### 2D/3D Visualization Modes:
- Real-time visualization of population movement
- Step-by-step iteration control
- Animation playback with adjustable speed
- Convergence curve plotting
- Trail lines showing agent movement paths

#### Engineering Problem Mode:
- Solve constrained engineering design problems
- Visualize design variables and constraints
- Monitor constraint satisfaction during optimization

#### Implemented Benchmark Functions
- Sphere Function (F1) - Unimodal
- Schwefel 2.22 Function (F10) - Unimodal
- Rastrigin Function (F11) - Multimodal
- Ackley Function (F14) - Multimodal
- Griewank Function (F15) - Multimodal

#### Engineering Design Problems
- Spring Design: Minimize weight of helical compression spring subject to shear stress, surge frequency, deflection, and diameter constraints.
- Pressure Vessel: Minimize fabrication cost of cylindrical pressure vessel with hemispherical heads subject to ASME constraints.
- Welded Beam Design: Minimize cost of welded beam subject to shear stress, bending stress, buckling load, and deflection constraints.
- Speed Reducer Design: Minimize weight of speed reducer subject to bending stress, surface stress, deflection, and geometry constraints.

## Algorithm Implementation
The implementation follows the original HBA paper (Hashim et al., 2022) with:
- Population initialization (Equation 3)
- Intensity calculation (Equation 7)
- Density factor update (Equation 9)
- Digging phase movement (Equation 10)
- Honey phase movement (Equation 11)

### Default parameters: C=2.0, β=6.0 as specified in the paper

#### Configuration
Edit config.py to modify:
- Default algorithm parameters (population size, max iterations)
- Plot resolution and colormaps
- Window sizes and update intervals


## References
Hashim, F. A., et al. (2022). "Honey Badger Algorithm: New metaheuristic algorithm for solving optimization problems." Mathematics and Computers in Simulation, 192, 84-110.

Engineering problem formulations adapted from various constrained optimization benchmark papers in engineering design literature.