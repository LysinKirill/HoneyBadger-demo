from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox, QSlider,
                             QGroupBox, QComboBox, QGridLayout, QSplitter)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

from core.export_data import DataExporter
from core.functions import TEST_FUNCTIONS_2D, get_function_2d_grid
from core.honey_badger import HoneyBadgerAlgorithm, HBAParams


class Tab2D(QWidget):
    def __init__(self):
        super().__init__()
        self.func = None
        self.bounds = None
        self.optimum = None
        self.func_name = None
        self.hba = None
        self.is_animating = False
        self.trail_lines = []

        self.init_ui()
        self.setup_default_function()

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.step_optimization)

    def init_ui(self):
        main_layout = QVBoxLayout()

        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Y')
        self.plot_widget.setLabel('bottom', 'X')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        left_layout.addWidget(self.plot_widget)

        right_panel = self.create_info_panel()

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 300])

        main_layout.addWidget(splitter)

        self.convergence_plot = pg.PlotWidget()
        self.convergence_plot.setBackground('w')
        self.convergence_plot.setLabel('left', 'Best Fitness')
        self.convergence_plot.setLabel('bottom', 'Iteration')
        self.convergence_plot.showGrid(x=True, y=True, alpha=0.3)
        self.convergence_plot.setMaximumHeight(200)
        main_layout.addWidget(self.convergence_plot)

        self.setLayout(main_layout)

    def create_control_panel(self):
        panel = QGroupBox("Controls")
        layout = QGridLayout()

        layout.addWidget(QLabel("Function:"), 0, 0)
        self.func_combo = QComboBox()
        self.func_combo.addItems(TEST_FUNCTIONS_2D.keys())
        self.func_combo.currentTextChanged.connect(self.change_function)
        layout.addWidget(self.func_combo, 0, 1)

        layout.addWidget(QLabel("Population:"), 0, 2)
        self.spin_pop = QSpinBox()
        self.spin_pop.setRange(5, 100)
        self.spin_pop.setValue(20)
        layout.addWidget(self.spin_pop, 0, 3)

        layout.addWidget(QLabel("Max Iter:"), 0, 4)
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(10, 2000)
        self.spin_iter.setValue(100)
        layout.addWidget(self.spin_iter, 0, 5)

        btn_row = 1
        self.btn_init = QPushButton("Initialize")
        self.btn_init.clicked.connect(self.initialize)
        layout.addWidget(self.btn_init, btn_row, 0)

        self.btn_step = QPushButton("Step")
        self.btn_step.clicked.connect(self.step_optimization)
        self.btn_step.setEnabled(False)
        layout.addWidget(self.btn_step, btn_row, 1)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.toggle_animation)
        self.btn_play.setEnabled(False)
        layout.addWidget(self.btn_play, btn_row, 2)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset)
        layout.addWidget(self.btn_reset, btn_row, 3)

        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setEnabled(False)
        layout.addWidget(self.btn_export, btn_row, 4)

        layout.addWidget(QLabel("Speed:"), btn_row, 5)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(10, 500)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self.update_speed)
        layout.addWidget(self.speed_slider, btn_row, 6)

        panel.setLayout(layout)
        return panel

    def create_info_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        info_group = QGroupBox("Status")
        info_layout = QVBoxLayout()

        self.iter_label = QLabel("Iteration: 0/0")
        self.iter_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.iter_label)

        self.best_label = QLabel("Best Fitness: N/A")
        info_layout.addWidget(self.best_label)

        self.solution_label = QLabel("Best Solution: N/A")
        info_layout.addWidget(self.solution_label)

        self.phase_label = QLabel("Phase: Not started")
        info_layout.addWidget(self.phase_label)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        opt_group = QGroupBox("Known Optimum")
        opt_layout = QVBoxLayout()
        self.opt_label = QLabel("Loading...")
        opt_layout.addWidget(self.opt_label)
        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)

        layout.addStretch()
        return panel

    def setup_default_function(self):
        self.func_name = list(TEST_FUNCTIONS_2D.keys())[0]
        self.func, self.bounds, self.optimum = TEST_FUNCTIONS_2D[self.func_name]
        self.plot_function_surface()
        self.update_optimum_label()

    def change_function(self, func_name):
        self.func_name = func_name
        self.func, self.bounds, self.optimum = TEST_FUNCTIONS_2D[func_name]
        self.plot_function_surface()
        self.update_optimum_label()
        self.reset()

    def plot_function_surface(self):
        self.plot_widget.clear()

        X, Y, Z = get_function_2d_grid(self.func, self.bounds, resolution=100)

        img = pg.ImageItem(Z.T)
        img.setRect(self.bounds[0], self.bounds[0],
                    self.bounds[1] - self.bounds[0],
                    self.bounds[1] - self.bounds[0])
        self.plot_widget.addItem(img)

        cmap = pg.colormap.get('viridis')
        bar = pg.ColorBarItem(colorMap=cmap, values=(Z.min(), Z.max()))
        bar.setImageItem(img)

        self.optimum_point = pg.ScatterPlotItem(
            [self.optimum[0]], [self.optimum[1]],
            pen=pg.mkPen('r', width=2), brush=pg.mkBrush('r'),
            size=12, symbol='x'
        )
        self.plot_widget.addItem(self.optimum_point)

        self.population_scatter = pg.ScatterPlotItem(
            pen=pg.mkPen('b'), brush=pg.mkBrush('b'),
            size=10, symbol='o', pxMode=True
        )
        self.plot_widget.addItem(self.population_scatter)

    def update_optimum_label(self):
        self.opt_label.setText(f"f({self.optimum[0]:.3f}, {self.optimum[1]:.3f}) = {self.func(self.optimum):.3f}")

    def initialize(self):
        params = HBAParams(
            pop_size=self.spin_pop.value(),
            max_iter=self.spin_iter.value()
        )
        self.hba = HoneyBadgerAlgorithm(params)
        self.hba.set_optimization_problem(self.func, 2, self.bounds)

        self.btn_step.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_init.setEnabled(False)

        self.convergence_plot.clear()
        for line in self.trail_lines:
            self.plot_widget.removeItem(line)
        self.trail_lines = []

        self.update_display()
        self.update_population_plot()

    def step_optimization(self):
        if self.hba.current_iter >= self.hba.params.max_iter:
            if self.is_animating:
                self.toggle_animation()
            return

        if hasattr(self.hba, 'population'):
            self.hba.previous_population = self.hba.population.copy()

        self.hba.run_one_iteration()
        self.update_population_plot()
        self.update_display()
        self.plot_convergence()

    def update_population_plot(self):
        if self.hba.population is None:
            return

        x = self.hba.population[:, 0]
        y = self.hba.population[:, 1]
        self.population_scatter.setData(x, y)

        if hasattr(self.hba, 'previous_population') and self.hba.previous_population is not None:
            for line in self.trail_lines:
                self.plot_widget.removeItem(line)
            self.trail_lines = []

            for i in range(len(x)):
                x_old = self.hba.previous_population[i, 0]
                y_old = self.hba.previous_population[i, 1]
                if abs(x[i] - x_old) > 0.01 or abs(y[i] - y_old) > 0.01:
                    line = pg.PlotDataItem(
                        [x_old, x[i]], [y_old, y[i]],
                        pen=pg.mkPen(color=(255, 0, 0, 100), width=1)
                    )
                    self.plot_widget.addItem(line)
                    self.trail_lines.append(line)

    def update_display(self):
        self.iter_label.setText(f"Iteration: {self.hba.current_iter}/{self.hba.params.max_iter}")
        self.best_label.setText(f"Best Fitness: {self.hba.best_fitness:.6f}")

        if self.hba.best_solution is not None:
            self.solution_label.setText(
                f"Best Solution: [{self.hba.best_solution[0]:.4f}, {self.hba.best_solution[1]:.4f}]"
            )

        phase = getattr(self.hba, 'current_phase', 'Unknown')
        self.phase_label.setText(f"Phase: {phase}")

    def plot_convergence(self):
        if len(self.hba.convergence_curve) > 0:
            self.convergence_plot.clear()
            self.convergence_plot.plot(self.hba.convergence_curve, pen=pg.mkPen('b', width=2))

    def toggle_animation(self):
        if not self.is_animating:
            self.is_animating = True
            self.btn_play.setText("⏸ Pause")
            self.btn_step.setEnabled(False)
            self.animation_timer.start(self.speed_slider.value())
        else:
            self.is_animating = False
            self.btn_play.setText("▶ Play")
            self.btn_step.setEnabled(True)
            self.animation_timer.stop()

    def update_speed(self):
        if self.is_animating:
            self.animation_timer.setInterval(self.speed_slider.value())

    def reset(self):
        if self.is_animating:
            self.toggle_animation()

        self.hba = None
        self.btn_step.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_init.setEnabled(True)

        self.population_scatter.setData([], [])
        for line in self.trail_lines:
            self.plot_widget.removeItem(line)
        self.trail_lines = []
        self.convergence_plot.clear()

        self.iter_label.setText("Iteration: 0/0")
        self.best_label.setText("Best Fitness: N/A")
        self.solution_label.setText("Best Solution: N/A")
        self.phase_label.setText("Phase: Not started")

    def export_results(self):
        if self.hba is None:
            return

        export_data = {
            'function_name': self.func_name,
            'bounds': self.bounds,
            'optimum': self.optimum.tolist(),
            'optimization_results': self.hba.get_optimization_history(),
            'timestamp': datetime.now().isoformat()
        }

        filename = f"{self.func_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        DataExporter.export_to_json(export_data, filename, subfolder="benchmark_2d")

    def on_tab_selected(self):
        pass