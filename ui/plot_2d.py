import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox)
import pyqtgraph as pg
from core.honey_badger import HoneyBadgerAlgorithm, HBAParams


class Plot2DWindow(QWidget):
    def __init__(self, func, bounds, optimum, func_name):
        super().__init__()
        self.func = func
        self.bounds = bounds
        self.optimum = optimum
        self.func_name = func_name

        self.setWindowTitle(f"2D HBA - {func_name}")
        self.setGeometry(150, 150, 800, 600)

        self.init_ui()
        self.setup_plot()
        self.setup_algorithm()

    def init_ui(self):
        layout = QVBoxLayout()

        top_layout = QHBoxLayout()

        self.btn_run = QPushButton("Run Optimization")
        self.btn_run.clicked.connect(self.run_optimization)
        top_layout.addWidget(self.btn_run)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset)
        top_layout.addWidget(self.btn_reset)

        top_layout.addWidget(QLabel("Population:"))
        self.spin_pop = QSpinBox()
        self.spin_pop.setRange(10, 100)
        self.spin_pop.setValue(30)
        top_layout.addWidget(self.spin_pop)

        top_layout.addWidget(QLabel("Iterations:"))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(100, 2000)
        self.spin_iter.setValue(500)
        top_layout.addWidget(self.spin_iter)

        layout.addLayout(top_layout)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Y')
        self.plot_widget.setLabel('bottom', 'X')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.convergence_plot = pg.PlotWidget()
        self.convergence_plot.setBackground('w')
        self.convergence_plot.setLabel('left', 'Best Fitness')
        self.convergence_plot.setLabel('bottom', 'Iteration')
        self.convergence_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.convergence_plot)

        self.setLayout(layout)

    def setup_plot(self):
        resolution = 100
        x = np.linspace(self.bounds[0], self.bounds[1], resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = self.func(np.array([X[i, j], Y[i, j]]))

        self.surface = pg.ImageItem(Z.T)
        self.surface.setRect(self.bounds[0], self.bounds[0],
                             self.bounds[1] - self.bounds[0],
                             self.bounds[1] - self.bounds[0])
        self.plot_widget.addItem(self.surface)

        self.colorbar = pg.ColorBarItem(colorMap=pg.colormap.get('viridis'),
                                        values=(Z.min(), Z.max()))
        self.colorbar.setImageItem(self.surface)

        self.optimum_point = pg.ScatterPlotItem(
            [self.optimum[0]], [self.optimum[1]],
            pen=pg.mkPen(color='r', width=2),
            brush=pg.mkBrush('r'), size=10, symbol='x'
        )
        self.plot_widget.addItem(self.optimum_point)

        self.population_scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(color='b'), brush=pg.mkBrush('b'),
            size=8, symbol='o'
        )
        self.plot_widget.addItem(self.population_scatter)

    def setup_algorithm(self):
        params = HBAParams(
            pop_size=self.spin_pop.value(),
            max_iter=self.spin_iter.value()
        )
        self.hba = HoneyBadgerAlgorithm(params)
        self.hba.initialize_population(2, self.bounds)
        self.update_population_plot()

    def update_population_plot(self):
        if self.hba.population is not None:
            x = self.hba.population[:, 0]
            y = self.hba.population[:, 1]
            self.population_scatter.setData(x, y)

    def run_optimization(self):
        self.btn_run.setEnabled(False)
        self.status_label.setText("Optimizing...")

        params = HBAParams(
            pop_size=self.spin_pop.value(),
            max_iter=self.spin_iter.value()
        )
        self.hba = HoneyBadgerAlgorithm(params)

        best_solution, best_fitness = self.hba.optimize(
            self.func, 2, self.bounds
        )

        # Update plots
        self.update_population_plot()
        self.plot_convergence()

        self.status_label.setText(
            f"Optimization complete! Best fitness: {best_fitness:.6f}\n"
            f"Best solution: [{best_solution[0]:.4f}, {best_solution[1]:.4f}]"
        )
        self.btn_run.setEnabled(True)

    def plot_convergence(self):
        self.convergence_plot.clear()
        self.convergence_plot.plot(
            self.hba.convergence_curve,
            pen=pg.mkPen(color='b', width=2)
        )

    def reset(self):
        self.setup_algorithm()
        self.convergence_plot.clear()
        self.status_label.setText("Reset complete")
