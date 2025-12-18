import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox)
import pyqtgraph.opengl as gl
from core.honey_badger import HoneyBadgerAlgorithm, HBAParams


class Plot3DWindow(QWidget):
    def __init__(self, func, bounds, optimum, func_name):
        super().__init__()
        self.func = func
        self.bounds = bounds
        self.optimum = optimum
        self.func_name = func_name

        self.setWindowTitle(f"3D HBA - {func_name}")
        self.setGeometry(150, 150, 800, 600)

        self.init_ui()
        self.setup_3d_plot()
        self.setup_algorithm()

    def init_ui(self):
        layout = QVBoxLayout()

        top_layout = QHBoxLayout()

        self.btn_run = QPushButton("Run 3D Optimization")
        self.btn_run.clicked.connect(self.run_optimization)
        top_layout.addWidget(self.btn_run)

        self.btn_animate = QPushButton("Animate")
        self.btn_animate.clicked.connect(self.toggle_animation)
        top_layout.addWidget(self.btn_animate)

        top_layout.addWidget(QLabel("Population:"))
        self.spin_pop = QSpinBox()
        self.spin_pop.setRange(10, 100)
        self.spin_pop.setValue(30)
        top_layout.addWidget(self.spin_pop)

        layout.addLayout(top_layout)

        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=50)
        layout.addWidget(self.gl_widget)

        self.status_label = QLabel("Ready for 3D optimization")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

        self.animating = False
        self.animation_step = 0

    def setup_3d_plot(self):
        resolution = 30
        x = np.linspace(self.bounds[0], self.bounds[1], resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = self.func(np.array([X[i, j], Y[i, j], 0]))

        self.surface = gl.GLSurfacePlotItem(
            x=x, y=y, z=Z,
            shader='shaded',
            glOptions='translucent'
        )
        self.surface.scale(1, 1, 0.1)
        self.gl_widget.addItem(self.surface)

        self.population_scatter = gl.GLScatterPlotItem(
            size=10,
            color=(1, 0, 0, 1)  # Red
        )
        self.gl_widget.addItem(self.population_scatter)

        self.best_scatter = gl.GLScatterPlotItem(
            size=15,
            color=(0, 1, 0, 1)
        )
        self.gl_widget.addItem(self.best_scatter)

        self.trajectory_lines = gl.GLLinePlotItem(
            width=2,
            color=(0, 0, 1, 0.5)
        )
        self.gl_widget.addItem(self.trajectory_lines)

    def setup_algorithm(self):
        params = HBAParams(
            pop_size=self.spin_pop.value(),
            max_iter=200
        )
        self.hba = HoneyBadgerAlgorithm(params)
        self.hba.initialize_population(3, self.bounds)
        self.update_3d_plot()

    def update_3d_plot(self):
        if self.hba.population is not None:
            pos = np.zeros((self.hba.population.shape[0], 3))
            pos[:, :2] = self.hba.population[:, :2]
            pos[:, 2] = np.array([self.func(ind) * 10 for ind in self.hba.population])

            self.population_scatter.setData(pos=pos)
            if self.hba.best_solution is not None:
                best_pos = np.zeros((1, 3))
                best_pos[0, :2] = self.hba.best_solution[:2]
                best_pos[0, 2] = self.func(self.hba.best_solution) * 10
                self.best_scatter.setData(pos=best_pos)

    def run_optimization(self):
        self.btn_run.setEnabled(False)
        self.status_label.setText("Running 3D optimization...")

        params = HBAParams(
            pop_size=self.spin_pop.value(),
            max_iter=100
        )
        self.hba = HoneyBadgerAlgorithm(params)

        def func_3d(x):
            if len(x) == 3:
                return self.func(x)
            else:
                x_full = np.zeros(3)
                x_full[:len(x)] = x
                return self.func(x_full)

        best_solution, best_fitness = self.hba.optimize(
            func_3d, 3, self.bounds
        )

        self.update_3d_plot()
        self.status_label.setText(
            f"3D Optimization complete! Best fitness: {best_fitness:.6f}"
        )
        self.btn_run.setEnabled(True)

    def toggle_animation(self):
        if not self.animating:
            self.animating = True
            self.btn_animate.setText("Stop Animation")
            self.animate_optimization()
        else:
            self.animating = False
            self.btn_animate.setText("Animate")

    def animate_optimization(self):
        """Animate the optimization process step by step"""
        if not self.animating:
            return

        if self.animation_step < self.hba.params.max_iter:
            self.hba.run_one_iteration()
            self.update_3d_plot()
            self.animation_step += 1

            from PyQt6.QtCore import QTimer
            QTimer.singleShot(50, self.animate_optimization)
        else:
            self.animating = False
            self.btn_animate.setText("Animate")
            self.animation_step = 0
