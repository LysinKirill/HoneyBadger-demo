from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox, QSlider,
                             QGroupBox, QComboBox, QGridLayout, QSplitter)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import pyqtgraph.opengl as gl

from core.export_data import DataExporter
from core.functions import TEST_FUNCTIONS_3D
from core.honey_badger import HoneyBadgerAlgorithm, HBAParams


class Tab3D(QWidget):
    def __init__(self):
        super().__init__()
        self.func = None
        self.bounds = None
        self.optimum = None
        self.func_name = None
        self.hba = None
        self.is_animating = False
        self.trajectory_history = []

        self.init_ui()
        self.setup_default_function()

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.step_optimization)

    def init_ui(self):
        main_layout = QVBoxLayout()

        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor('w')
        self.gl_widget.setCameraPosition(distance=80, elevation=30, azimuth=45)

        right_panel = self.create_info_panel()

        splitter.addWidget(self.gl_widget)
        splitter.addWidget(right_panel)
        splitter.setSizes([900, 300])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def create_control_panel(self):
        panel = QGroupBox("Controls")
        layout = QGridLayout()

        layout.addWidget(QLabel("Function:"), 0, 0)
        self.func_combo = QComboBox()
        self.func_combo.addItems(TEST_FUNCTIONS_3D.keys())
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

        cam_group = QGroupBox("Camera")
        cam_layout = QVBoxLayout()

        self.btn_reset_cam = QPushButton("Reset View")
        self.btn_reset_cam.clicked.connect(self.reset_camera)
        cam_layout.addWidget(self.btn_reset_cam)

        self.btn_top = QPushButton("Top")
        self.btn_top.clicked.connect(lambda: self.set_camera('top'))
        cam_layout.addWidget(self.btn_top)

        self.btn_iso = QPushButton("Isometric")
        self.btn_iso.clicked.connect(lambda: self.set_camera('iso'))
        cam_layout.addWidget(self.btn_iso)

        cam_group.setLayout(cam_layout)
        layout.addWidget(cam_group)

        layout.addStretch()
        return panel

    def setup_default_function(self):
        self.func_name = list(TEST_FUNCTIONS_3D.keys())[0]
        self.func, self.bounds, self.optimum = TEST_FUNCTIONS_3D[self.func_name]
        self.plot_function_surface()

    def change_function(self, func_name):
        self.func_name = func_name
        self.func, self.bounds, self.optimum = TEST_FUNCTIONS_3D[func_name]
        self.plot_function_surface()
        self.reset()

    def plot_function_surface(self):
        self.gl_widget.clear()

        resolution = 50
        x = np.linspace(self.bounds[0], self.bounds[1], resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = self.func(np.array([X[i, j], Y[i, j], 0]))

        z_min, z_max = Z.min(), Z.max()

        verts = []
        faces = []
        colors = []

        for i in range(resolution - 1):
            for j in range(resolution - 1):
                p1 = [X[i, j], Y[i, j], Z[i, j]]
                p2 = [X[i + 1, j], Y[i + 1, j], Z[i + 1, j]]
                p3 = [X[i, j + 1], Y[i, j + 1], Z[i, j + 1]]
                p4 = [X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]]

                idx = len(verts)
                verts.extend([p1, p2, p3, p4])

                c = (0.2 + 0.8 * (Z[i, j] - z_min) / (z_max - z_min) if z_max > z_min else 0.6,
                     0.5 - 0.5 * (Z[i, j] - z_min) / (z_max - z_min) if z_max > z_min else 0.4,
                     1.0 - 0.8 * (Z[i, j] - z_min) / (z_max - z_min) if z_max > z_min else 0.2,
                     1.0)
                colors.extend([c, c, c, c])

                faces.append([idx, idx + 1, idx + 2])
                faces.append([idx + 1, idx + 3, idx + 2])

        mesh = gl.GLMeshItem(
            vertexes=np.array(verts),
            faces=np.array(faces),
            vertexColors=np.array(colors),
            smooth=True,
            drawEdges=False,
            drawFaces=True,
            shader='shaded'
        )
        self.gl_widget.addItem(mesh)

        self.population_scatter = gl.GLScatterPlotItem(
            size=15, color=(1, 0, 0, 1), pxMode=True
        )
        self.gl_widget.addItem(self.population_scatter)

        self.best_scatter = gl.GLScatterPlotItem(
            size=25, color=(0, 1, 0, 1), pxMode=True
        )
        self.gl_widget.addItem(self.best_scatter)

    def initialize(self):
        params = HBAParams(
            pop_size=self.spin_pop.value(),
            max_iter=self.spin_iter.value()
        )
        self.hba = HoneyBadgerAlgorithm(params)
        self.hba.set_optimization_problem(self.func, 3, self.bounds)

        self.btn_step.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_init.setEnabled(False)

        for lines in self.trajectory_history:
            for line in lines:
                self.gl_widget.removeItem(line)
        self.trajectory_history = []

        self.update_display()
        self.update_3d_plot()

    def step_optimization(self):
        if self.hba.current_iter >= self.hba.params.max_iter:
            if self.is_animating:
                self.toggle_animation()
            return

        if hasattr(self.hba, 'population'):
            self.hba.previous_population = self.hba.population.copy()

        self.hba.run_one_iteration()
        self.update_3d_plot()
        self.update_display()

    def update_3d_plot(self):
        if self.hba.population is None:
            return

        pop = self.hba.population
        fitness = np.array([self.func(ind) for ind in pop])

        pos = np.zeros((pop.shape[0], 3))
        pos[:, :2] = pop[:, :2]
        pos[:, 2] = fitness

        self.population_scatter.setData(pos=pos)

        if self.hba.best_solution is not None:
            best = self.hba.best_solution
            best_pos = np.zeros((1, 3))
            best_pos[0, :2] = best[:2]
            best_pos[0, 2] = self.func(best)
            self.best_scatter.setData(pos=best_pos)

    def update_display(self):
        self.iter_label.setText(f"Iteration: {self.hba.current_iter}/{self.hba.params.max_iter}")
        self.best_label.setText(f"Best Fitness: {self.hba.best_fitness:.6f}")

        if self.hba.best_solution is not None:
            sol = self.hba.best_solution
            self.solution_label.setText(
                f"Best Solution: [{sol[0]:.4f}, {sol[1]:.4f}, {sol[2]:.4f}]"
            )

        phase = getattr(self.hba, 'current_phase', 'Unknown')
        self.phase_label.setText(f"Phase: {phase}")

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

    def reset_camera(self):
        self.gl_widget.setCameraPosition(distance=80, elevation=30, azimuth=45)

    def set_camera(self, mode):
        if mode == 'top':
            self.gl_widget.setCameraPosition(distance=80, elevation=90, azimuth=0)
        elif mode == 'iso':
            self.gl_widget.setCameraPosition(distance=80, elevation=30, azimuth=45)

    def reset(self):
        if self.is_animating:
            self.toggle_animation()

        self.hba = None
        self.btn_step.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_init.setEnabled(True)

        for lines in self.trajectory_history:
            for line in lines:
                self.gl_widget.removeItem(line)
        self.trajectory_history = []

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
        DataExporter.export_to_json(export_data, filename, subfolder="benchmark_3d")

    def on_tab_selected(self):
        pass