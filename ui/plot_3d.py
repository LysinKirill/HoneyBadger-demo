from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox, QSlider,
                             QGroupBox)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph.opengl as gl
from core.honey_badger import HoneyBadgerAlgorithm, HBAParams
from core.export_data import DataExporter
from datetime import datetime


class Plot3DWindow(QWidget):
    def __init__(self, func, bounds, optimum, func_name):
        super().__init__()
        self.func = func
        self.bounds = bounds
        self.optimum = optimum
        self.func_name = func_name

        self.setWindowTitle(f"3D HBA - {func_name}")
        self.setGeometry(150, 150, 1000, 800)

        self.init_ui()
        self.setup_3d_plot()
        self.setup_algorithm()

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.step_optimization)
        self.is_animating = False
        self.trajectory_history = []

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

        self.btn_export = QPushButton("Export Results")
        self.btn_export.clicked.connect(self.export_results)
        top_layout.addWidget(self.btn_export)

        layout.addLayout(top_layout)

        speed_group = QGroupBox("Simulation Speed")
        speed_layout = QVBoxLayout()

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Fast"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 1000)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self.update_speed)
        slider_row.addWidget(self.speed_slider)
        slider_row.addWidget(QLabel("Slow"))
        speed_layout.addLayout(slider_row)

        label_row = QHBoxLayout()
        label_row.addStretch(1)
        self.speed_label = QLabel("100 ms")
        self.speed_label.setStyleSheet("font-weight: bold;")
        label_row.addWidget(self.speed_label)
        label_row.addStretch(1)
        speed_layout.addLayout(label_row)

        preset_row = QHBoxLayout()
        preset_row.addStretch(1)

        presets = [
            ("Turbo", 10),
            ("Fast", 50),
            ("Normal", 100),
            ("Slow", 250),
            ("Step-by-Step", 1000),
        ]

        for label, speed in presets:
            btn = QPushButton(label)
            btn.setMaximumWidth(80)
            btn.clicked.connect(lambda checked, s=speed: self.set_speed(s))
            preset_row.addWidget(btn)

        preset_row.addStretch(1)
        speed_layout.addLayout(preset_row)

        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)

        plot_layout = QHBoxLayout()

        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor('w')
        self.gl_widget.opts['distance'] = 80
        self.gl_widget.opts['elevation'] = 30
        self.gl_widget.opts['azimuth'] = 45
        self.gl_widget.setCameraPosition(distance=80, elevation=30, azimuth=45)
        plot_layout.addWidget(self.gl_widget, 2)

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

        cam_group = QGroupBox("Camera Controls")
        cam_layout = QVBoxLayout()

        reset_cam_btn = QPushButton("Reset Camera")
        reset_cam_btn.clicked.connect(self.reset_camera)
        cam_layout.addWidget(reset_cam_btn)

        top_view_btn = QPushButton("Top View")
        top_view_btn.clicked.connect(lambda: self.set_camera_view('top'))
        cam_layout.addWidget(top_view_btn)

        side_view_btn = QPushButton("Side View")
        side_view_btn.clicked.connect(lambda: self.set_camera_view('side'))
        cam_layout.addWidget(side_view_btn)

        isometric_btn = QPushButton("Isometric")
        isometric_btn.clicked.connect(lambda: self.set_camera_view('iso'))
        cam_layout.addWidget(isometric_btn)

        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(self.zoom_in)
        cam_layout.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(self.zoom_out)
        cam_layout.addWidget(zoom_out_btn)

        cam_group.setLayout(cam_layout)
        info_layout.addWidget(cam_group)

        info_layout.addStretch(1)
        plot_layout.addWidget(info_widget, 1)

        layout.addLayout(plot_layout)

        self.status_label = QLabel("Ready for 3D optimization")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def setup_3d_plot(self):
        resolution = 300
        x = np.linspace(self.bounds[0], self.bounds[1], resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                if len(self.optimum) == 3:
                    Z[i, j] = self.func(np.array([X[i, j], Y[i, j], 0]))
                else:
                    Z[i, j] = self.func(np.array([X[i, j], Y[i, j]]))

        z_min, z_max = Z.min(), Z.max()

        self.surface = gl.GLMeshItem(
            vertexes=np.array([X.flatten(), Y.flatten(), Z.flatten()]).T,
            faces=np.array([
                               [i, i + 1, i + resolution + 1]
                               for i in range(0, resolution * (resolution - 1) - 1)
                               if (i + 1) % resolution != 0
                           ] + [
                               [i + 1, i + resolution + 1, i + resolution]
                               for i in range(0, resolution * (resolution - 1) - 1)
                               if (i + 1) % resolution != 0
                           ]),
            vertexColors=np.array([
                (0.2 + 0.8 * (z - z_min) / (z_max - z_min) if z_max > z_min else 0.6,
                 0.5 - 0.5 * (z - z_min) / (z_max - z_min) if z_max > z_min else 0.4,
                 1.0 - 0.8 * (z - z_min) / (z_max - z_min) if z_max > z_min else 0.2,
                 1.0)
                for z in Z.flatten()
            ]),
            smooth=True,
            drawEdges=False,
            drawFaces=True,
            shader='shaded'
        )
        self.gl_widget.addItem(self.surface)

        bounds_range = self.bounds[1] - self.bounds[0]
        axis_length = bounds_range * 0.6

        x_axis = gl.GLLinePlotItem(
            pos=np.array([[-axis_length, 0, 0], [axis_length, 0, 0]]),
            color=(1, 0, 0, 1),
            width=3,
            glOptions='opaque'
        )
        self.gl_widget.addItem(x_axis)

        y_axis = gl.GLLinePlotItem(
            pos=np.array([[0, -axis_length, 0], [0, axis_length, 0]]),
            color=(0, 1, 0, 1),
            width=3,
            glOptions='opaque'
        )
        self.gl_widget.addItem(y_axis)

        z_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, z_min], [0, 0, z_max]]),
            color=(0, 0, 1, 1),
            width=3,
            glOptions='opaque'
        )
        self.gl_widget.addItem(z_axis)

        self.population_scatter = gl.GLScatterPlotItem(
            size=15,
            color=(1, 0, 0, 1),
            pxMode=True,
            glOptions='opaque'
        )
        self.gl_widget.addItem(self.population_scatter)

        self.best_scatter = gl.GLScatterPlotItem(
            size=25,
            color=(0, 1, 0, 1),
            pxMode=True,
            glOptions='opaque'
        )
        self.gl_widget.addItem(self.best_scatter)

    def setup_algorithm(self):
        params = HBAParams(
            pop_size=self.spin_pop.value(),
            max_iter=self.spin_iter.value()
        )
        self.hba = HoneyBadgerAlgorithm(params)

        if len(self.optimum) == 3:
            dim = 3
        else:
            dim = 2

        self.hba.set_optimization_problem(self.func, dim, self.bounds)
        self.update_info()
        self.update_3d_plot()

    def update_3d_plot(self):
        if self.hba.population is None:
            return

        if hasattr(self.hba, 'previous_population') and self.hba.previous_population is not None:
            self.update_trajectories()

        pop = self.hba.population
        fitness_values = np.array([self.func(ind) for ind in pop])

        pop_positions = np.zeros((pop.shape[0], 3))

        if pop.shape[1] == 3:
            pop_positions = pop.copy()
        else:
            pop_positions[:, :2] = pop[:, :2]
            pop_positions[:, 2] = fitness_values

        self.population_scatter.setData(
            pos=pop_positions,
            color=(1, 0, 0, 1),
            size=12
        )

        if self.hba.best_solution is not None:
            best = self.hba.best_solution
            best_fitness = self.func(best)

            best_pos = np.zeros((1, 3))
            if len(best) == 3:
                best_pos[0] = best
            else:
                best_pos[0, :2] = best[:2]
                best_pos[0, 2] = best_fitness

            self.best_scatter.setData(
                pos=best_pos,
                color=(0, 1, 0, 1),
                size=20
            )

    def update_trajectories(self):
        if len(self.trajectory_history) > 10:
            for line in self.trajectory_history[0]:
                self.gl_widget.removeItem(line)
            self.trajectory_history.pop(0)

        new_lines = []
        pop = self.hba.population
        prev_pop = self.hba.previous_population

        for i in range(pop.shape[0]):
            if len(pop[i]) == 3:
                start = prev_pop[i]
                end = pop[i]
            else:
                start = np.zeros(3)
                end = np.zeros(3)
                start[:2] = prev_pop[i, :2]
                start[2] = self.func(prev_pop[i])
                end[:2] = pop[i, :2]
                end[2] = self.func(pop[i])

            if np.linalg.norm(end - start) > 0.001:
                line = gl.GLLinePlotItem(
                    pos=np.array([start, end]),
                    color=(0, 0, 1, 0.4),
                    width=2,
                    antialias=True
                )
                self.gl_widget.addItem(line)
                new_lines.append(line)

        self.trajectory_history.append(new_lines)

    def update_info(self):
        self.iter_label.setText(f"Iteration: {self.hba.current_iter}/{self.hba.params.max_iter}")
        self.best_label.setText(f"Best Fitness: {self.hba.best_fitness:.6f}")

        if self.hba.best_solution is not None:
            sol = self.hba.best_solution
            if len(sol) == 3:
                self.solution_label.setText(
                    f"Best Solution: [{sol[0]:.4f}, {sol[1]:.4f}, {sol[2]:.4f}]"
                )
            else:
                self.solution_label.setText(
                    f"Best Solution: [{sol[0]:.4f}, {sol[1]:.4f}]"
                )

        if hasattr(self.hba, 'current_phase'):
            phase = self.hba.current_phase
        else:
            alpha = self.hba.update_density_factor()
            phase = "Digging" if alpha > 0.5 else "Honey"
        self.state_label.setText(f"Phase: {phase}")

        if hasattr(self.hba, 'last_intensity') and self.hba.last_intensity is not None:
            avg_intensity = np.mean(self.hba.last_intensity)
            self.intensity_label.setText(f"Avg Intensity: {avg_intensity:.4e}")

    def step_optimization(self):
        if self.hba.current_iter >= self.hba.params.max_iter:
            self.status_label.setText("Optimization complete!")
            if self.is_animating:
                self.toggle_animation()
            return

        self.hba.previous_population = self.hba.population.copy()
        self.hba.run_one_iteration()
        self.update_3d_plot()
        self.update_info()
        self.status_label.setText(f"Step {self.hba.current_iter} complete")

    def run_full_optimization(self):
        self.btn_run_full.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_step.setEnabled(False)
        self.status_label.setText("Running full optimization...")

        remaining = self.hba.params.max_iter - self.hba.current_iter
        for i in range(remaining):
            self.hba.previous_population = self.hba.population.copy()
            self.hba.run_one_iteration()

        self.update_3d_plot()
        self.update_info()
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

    def set_speed(self, speed_ms: int):
        self.speed_slider.setValue(speed_ms)
        self.update_speed()

    def reset_camera(self):
        self.gl_widget.setCameraPosition(distance=80, elevation=30, azimuth=45)

    def set_camera_view(self, view_type: str):
        if view_type == 'top':
            self.gl_widget.setCameraPosition(distance=80, elevation=90, azimuth=0)
        elif view_type == 'side':
            self.gl_widget.setCameraPosition(distance=80, elevation=0, azimuth=90)
        elif view_type == 'iso':
            self.gl_widget.setCameraPosition(distance=80, elevation=30, azimuth=45)

    def zoom_in(self):
        current_distance = self.gl_widget.opts['distance']
        self.gl_widget.setCameraPosition(distance=max(10, current_distance * 0.8))

    def zoom_out(self):
        current_distance = self.gl_widget.opts['distance']
        self.gl_widget.setCameraPosition(distance=min(200, current_distance * 1.2))

    def reset(self):
        if self.is_animating:
            self.toggle_animation()

        for step_lines in self.trajectory_history:
            for line in step_lines:
                self.gl_widget.removeItem(line)
        self.trajectory_history = []

        self.setup_algorithm()
        self.reset_camera()
        self.status_label.setText("Reset complete")

    def export_results(self):
        export_data = {
            'function_name': self.func_name,
            'bounds': self.bounds,
            'optimum': self.optimum.tolist(),
            'optimization_results': self.hba.get_optimization_history(),
            'timestamp': datetime.now().isoformat(),
            'type': 'benchmark_3d'
        }

        filename = f"{self.func_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = DataExporter.export_to_json(export_data, filename, subfolder="benchmark_3d")

        self.status_label.setText(f"Exported to io/exports/benchmark_3d/{Path(filepath).name}")