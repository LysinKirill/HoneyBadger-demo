import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox, QSlider,
                             QGroupBox)
from PyQt6.QtCore import Qt, QTimer
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
        self.setGeometry(150, 150, 1000, 800)

        self.init_ui()
        self.setup_plot()
        self.setup_algorithm()

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.step_optimization)
        self.is_animating = False

    def init_ui(self):
        layout = QVBoxLayout()

        top_layout = QHBoxLayout()

        self.btn_run_full = QPushButton("Run Full")
        self.btn_run_full.clicked.connect(self.run_full_optimization)
        top_layout.addWidget(self.btn_run_full)

        self.btn_step = QPushButton("Step (1 Iter)")
        self.btn_step.clicked.connect(self.step_optimization)
        top_layout.addWidget(self.btn_step)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.toggle_animation)
        top_layout.addWidget(self.btn_play)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset)
        top_layout.addWidget(self.btn_reset)

        top_layout.addStretch(1)

        top_layout.addWidget(QLabel("Population:"))
        self.spin_pop = QSpinBox()
        self.spin_pop.setRange(5, 100)
        self.spin_pop.setValue(20)
        top_layout.addWidget(self.spin_pop)

        top_layout.addWidget(QLabel("Max Iter:"))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(10, 2000)
        self.spin_iter.setValue(100)
        top_layout.addWidget(self.spin_iter)

        layout.addLayout(top_layout)

        speed_group = QGroupBox("Simulation Speed")
        speed_layout = QHBoxLayout()

        speed_layout.addWidget(QLabel("Fast"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 1000)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_slider)

        speed_layout.addWidget(QLabel("Slow"))
        self.speed_label = QLabel("100 ms")
        speed_layout.addWidget(self.speed_label)

        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)

        preset_layout = QHBoxLayout()

        slow_btn = QPushButton("Slow (500ms)")
        slow_btn.clicked.connect(lambda: self.set_speed(500))
        preset_layout.addWidget(slow_btn)

        medium_btn = QPushButton("Medium (100ms)")
        medium_btn.clicked.connect(lambda: self.set_speed(100))
        preset_layout.addWidget(medium_btn)

        fast_btn = QPushButton("Fast (50ms)")
        fast_btn.clicked.connect(lambda: self.set_speed(50))
        preset_layout.addWidget(fast_btn)

        very_fast_btn = QPushButton("Very Fast (10ms)")
        very_fast_btn.clicked.connect(lambda: self.set_speed(10))
        preset_layout.addWidget(very_fast_btn)

        layout.addLayout(preset_layout)

        plot_layout = QHBoxLayout()

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Y')
        self.plot_widget.setLabel('bottom', 'X')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_layout.addWidget(self.plot_widget, 2)

        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)

        self.iter_label = QLabel("Iteration: 0/0")
        self.iter_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.iter_label)

        self.best_label = QLabel("Best Fitness: N/A")
        info_layout.addWidget(self.best_label)

        self.solution_label = QLabel("Best Solution: N/A")
        info_layout.addWidget(self.solution_label)

        self.state_label = QLabel("Phase: Initialized")
        info_layout.addWidget(self.state_label)

        self.intensity_label = QLabel("Avg Intensity: N/A")
        info_layout.addWidget(self.intensity_label)

        info_layout.addStretch(1)
        plot_layout.addWidget(info_widget, 1)

        layout.addLayout(plot_layout)

        self.convergence_plot = pg.PlotWidget()
        self.convergence_plot.setBackground('w')
        self.convergence_plot.setLabel('left', 'Best Fitness')
        self.convergence_plot.setLabel('bottom', 'Iteration')
        self.convergence_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.convergence_plot)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def set_speed(self, speed_ms: int):
        self.speed_slider.setValue(speed_ms)
        self.update_speed()

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

        self.trail_lines = []

    def setup_algorithm(self):
        params = HBAParams(
            pop_size=self.spin_pop.value(),
            max_iter=self.spin_iter.value()
        )
        self.hba = HoneyBadgerAlgorithm(params)
        self.hba.set_optimization_problem(self.func, 2, self.bounds)  # Use new method
        self.update_info()
        self.update_population_plot()

    def update_population_plot(self):
        if self.hba.population is not None:
            x = self.hba.population[:, 0]
            y = self.hba.population[:, 1]
            self.population_scatter.setData(x, y)

            if hasattr(self.hba, 'previous_population') and self.hba.previous_population is not None:
                for line in self.trail_lines:
                    self.plot_widget.removeItem(line)
                self.trail_lines = []

                for i in range(self.hba.population.shape[0]):
                    x_old = self.hba.previous_population[i, 0]
                    y_old = self.hba.previous_population[i, 1]
                    x_new = x[i]
                    y_new = y[i]

                    if np.sqrt((x_new - x_old) ** 2 + (y_new - y_old) ** 2) > 0.01:
                        line = pg.PlotDataItem(
                            [x_old, x_new], [y_old, y_new],
                            pen=pg.mkPen(color=(255, 0, 0, 100), width=1, style=Qt.PenStyle.DotLine)
                        )
                        self.plot_widget.addItem(line)
                        self.trail_lines.append(line)

    def update_info(self):
        self.iter_label.setText(f"Iteration: {self.hba.current_iter}/{self.hba.params.max_iter}")
        self.best_label.setText(f"Best Fitness: {self.hba.best_fitness:.6f}")

        if self.hba.best_solution is not None:
            self.solution_label.setText(
                f"Best Solution: [{self.hba.best_solution[0]:.4f}, {self.hba.best_solution[1]:.4f}]"
            )

        if hasattr(self.hba, 'current_phase'):
            phase = self.hba.current_phase
        else:
            alpha = self.hba.update_density_factor()
            phase = "Digging (Exploration)" if alpha > 0.5 else "Honey (Exploitation)"
        self.state_label.setText(f"Phase: {phase}")

        if hasattr(self.hba, 'last_intensity') and self.hba.last_intensity is not None:
            avg_intensity = np.mean(self.hba.last_intensity)
            self.intensity_label.setText(f"Avg Intensity: {avg_intensity:.4e}")

    def step_optimization(self):
        """Run a single iteration of optimization"""
        if self.hba.current_iter >= self.hba.params.max_iter:
            self.status_label.setText("Optimization complete!")
            if self.is_animating:
                self.toggle_animation()
            return

        self.hba.previous_population = self.hba.population.copy()

        self.hba.run_one_iteration()

        self.update_population_plot()
        self.update_info()

        if len(self.hba.convergence_curve) > 0:
            self.convergence_plot.clear()
            self.convergence_plot.plot(
                self.hba.convergence_curve,
                pen=pg.mkPen(color='b', width=2)
            )

        self.status_label.setText(f"Step {self.hba.current_iter} complete")

    def run_full_optimization(self):
        self.btn_run_full.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_step.setEnabled(False)
        self.status_label.setText("Running full optimization...")

        remaining = self.hba.params.max_iter - self.hba.current_iter
        for i in range(remaining):
            self.hba.run_one_iteration()

        self.update_population_plot()
        self.update_info()
        self.plot_convergence()

        self.status_label.setText(f"Full optimization complete in {remaining} steps")
        self.btn_run_full.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_step.setEnabled(True)

    def toggle_animation(self):
        if not self.is_animating:
            self.is_animating = True
            self.btn_play.setText("⏸ Pause")
            self.btn_run_full.setEnabled(False)
            self.btn_step.setEnabled(False)
            self.animation_timer.start(self.speed_slider.value())
        else:
            self.is_animating = False
            self.btn_play.setText("▶ Play")
            self.btn_run_full.setEnabled(True)
            self.btn_step.setEnabled(True)
            self.animation_timer.stop()

    def update_speed(self):
        speed = self.speed_slider.value()
        self.speed_label.setText(f"{speed} ms")
        if self.is_animating:
            self.animation_timer.setInterval(speed)

        if speed > 0:
            fps = 1000 / speed
            self.speed_slider.setToolTip(f"Speed: {speed}ms ({fps:.1f} FPS)")

    def plot_convergence(self):
        if len(self.hba.convergence_curve) > 0:
            self.convergence_plot.clear()
            self.convergence_plot.plot(
                self.hba.convergence_curve,
                pen=pg.mkPen(color='b', width=2)
            )

    def reset(self):
        if self.is_animating:
            self.toggle_animation()

        self.setup_algorithm()
        self.convergence_plot.clear()

        for line in self.trail_lines:
            self.plot_widget.removeItem(line)
        self.trail_lines = []

        self.status_label.setText("Reset complete")
