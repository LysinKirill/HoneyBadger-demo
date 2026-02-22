from datetime import datetime

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox, QSlider,
                             QGroupBox, QComboBox, QGridLayout, QSplitter,
                             QCheckBox, QLineEdit, QScrollArea, QFileDialog)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import pyqtgraph.opengl as gl

from core.export_data import DataExporter
from core.functions import TEST_FUNCTIONS_2D, TEST_FUNCTIONS_3D, TEST_FUNCTIONS_5D
from core.honey_badger import HoneyBadgerAlgorithm, HBAParams
from core.recording import OptimizationRecording


class VariableSlider3D(QWidget):
    def __init__(self, index, name, min_val, max_val, initial_val, parent_tab):
        super().__init__()
        self.index = index
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.parent_tab = parent_tab
        self.updating = False

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.checkbox = QCheckBox(f"x{index + 1}")
        self.checkbox.stateChanged.connect(self.on_checkbox_changed)
        layout.addWidget(self.checkbox)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(int((initial_val - min_val) / (max_val - min_val) * 1000))
        self.slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.slider)

        self.textbox = QLineEdit()
        self.textbox.setText(f"{initial_val:.3f}")
        self.textbox.setFixedWidth(70)
        self.textbox.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.textbox)

        self.setLayout(layout)
        self.set_active(True)

    def on_checkbox_changed(self, state):
        if not self.updating:
            self.parent_tab.update_selected_variables()

    def on_slider_changed(self, value):
        if self.updating:
            return

        self.updating = True
        val = self.min_val + (value / 1000.0) * (self.max_val - self.min_val)
        self.textbox.setText(f"{val:.3f}")
        self.updating = False
        self.parent_tab.on_variable_changed()

    def on_text_changed(self, text):
        if self.updating:
            return

        try:
            val = float(text)
            if self.min_val <= val <= self.max_val:
                self.updating = True
                slider_val = int((val - self.min_val) / (self.max_val - self.min_val) * 1000)
                self.slider.setValue(slider_val)
                self.updating = False
                self.parent_tab.on_variable_changed()
        except ValueError:
            pass

    def get_value(self):
        return float(self.textbox.text())

    def set_active(self, active):
        self.slider.setEnabled(active)
        self.textbox.setEnabled(active)


class Tab3D(QWidget):
    def __init__(self):
        super().__init__()
        self.func = None
        self.bounds = None
        self.optimum = None
        self.func_name = None
        self.func_dim = 3
        self.hba = None
        self.is_animating = False
        self.trajectory_history = []
        self.variable_sliders = []
        self.fixed_values = []
        self.selected_vars = [0, 1, 2]

        self.recording = None
        self.is_recording = False
        self.playback_mode = False
        self.current_playback_step = 0

        self.init_ui()
        self.setup_default_function()

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.step_optimization)

    def init_ui(self):
        main_layout = QVBoxLayout()

        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        self.var_panel = self.create_variable_panel()
        main_layout.addWidget(self.var_panel)

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

        all_functions = {**TEST_FUNCTIONS_2D, **TEST_FUNCTIONS_3D, **TEST_FUNCTIONS_5D}
        self.func_combo.addItems(all_functions.keys())
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

        playback_row = 2
        self.add_playback_controls(layout, playback_row, 0)

        panel.setLayout(layout)
        return panel

    def add_playback_controls(self, layout, row, col):
        self.btn_record = QPushButton("● Record")
        self.btn_record.setStyleSheet("color: red; font-weight: bold;")
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_record.setEnabled(False)
        layout.addWidget(self.btn_record, row, col)
        col += 1

        self.btn_save_recording = QPushButton("💾 Save")
        self.btn_save_recording.clicked.connect(self.save_recording)
        self.btn_save_recording.setEnabled(False)
        layout.addWidget(self.btn_save_recording, row, col)
        col += 1

        self.btn_load_recording = QPushButton("📂 Load")
        self.btn_load_recording.clicked.connect(self.load_recording)
        layout.addWidget(self.btn_load_recording, row, col)
        col += 1

        self.btn_prev = QPushButton("◀ Prev")
        self.btn_prev.clicked.connect(self.prev_step)
        self.btn_prev.setEnabled(False)
        layout.addWidget(self.btn_prev, row, col)
        col += 1

        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self.next_step)
        self.btn_next.setEnabled(False)
        layout.addWidget(self.btn_next, row, col)
        col += 1

        self.step_label = QLabel("Step: 0/0")
        layout.addWidget(self.step_label, row, col)
        col += 1

    def create_variable_panel(self):
        panel = QGroupBox("Variable Controls (Select 3 variables for 3D plot)")
        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(150)

        scroll_widget = QWidget()
        self.var_layout = QVBoxLayout(scroll_widget)
        scroll.setWidget(scroll_widget)

        layout.addWidget(scroll)
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

        constants_group = QGroupBox("Fixed Variables")
        constants_layout = QVBoxLayout()
        self.constants_label = QLabel("Fixed variables: None")
        self.constants_label.setWordWrap(True)
        constants_layout.addWidget(self.constants_label)
        constants_group.setLayout(constants_layout)
        layout.addWidget(constants_group)

        layout.addStretch()
        return panel

    def setup_variable_controls(self):
        for slider in self.variable_sliders:
            slider.setParent(None)
            slider.deleteLater()
        self.variable_sliders.clear()

        while self.var_layout.count():
            item = self.var_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if "5D" in self.func_name:
            self.func_dim = 5
        elif "3D" in self.func_name:
            self.func_dim = 3
        else:
            self.func_dim = 2
        bounds_list = [self.bounds] * self.func_dim

        self.fixed_values = [0.0] * self.func_dim

        for i in range(self.func_dim):
            min_val = bounds_list[i][0]
            max_val = bounds_list[i][1]
            initial = (min_val + max_val) / 2

            slider = VariableSlider3D(i, f"x{i + 1}", min_val, max_val, initial, self)
            self.var_layout.addWidget(slider)
            self.variable_sliders.append(slider)
            self.fixed_values[i] = initial

        num_to_select = min(3, len(self.variable_sliders))
        for i in range(num_to_select):
            self.variable_sliders[i].checkbox.setChecked(True)

    def update_selected_variables(self):
        selected = []
        for i, slider in enumerate(self.variable_sliders):
            if slider.checkbox.isChecked():
                selected.append(i)

        if len(selected) > 3:
            for i in range(len(self.variable_sliders) - 1, -1, -1):
                if self.variable_sliders[i].checkbox.isChecked():
                    self.variable_sliders[i].checkbox.setChecked(False)
                    break

        self.selected_vars = []
        for i, slider in enumerate(self.variable_sliders):
            if slider.checkbox.isChecked():
                self.selected_vars.append(i)
                slider.set_active(False)
            else:
                slider.set_active(True)

        if len(self.selected_vars) < 3:
            needed = 3 - len(self.selected_vars)
            for i in range(len(self.variable_sliders)):
                if i not in self.selected_vars and needed > 0:
                    self.variable_sliders[i].checkbox.setChecked(True)
                    self.selected_vars.append(i)
                    self.variable_sliders[i].set_active(False)
                    needed -= 1

        self.update_constants_display()
        self.on_variable_changed()

    def update_constants_display(self):
        fixed_vars = []
        for i, slider in enumerate(self.variable_sliders):
            if not slider.checkbox.isChecked():
                fixed_vars.append(f"x{i + 1}={slider.get_value():.3f}")

        if fixed_vars:
            self.constants_label.setText("Fixed: " + ", ".join(fixed_vars))
        else:
            self.constants_label.setText("Fixed variables: None")

    def on_variable_changed(self):
        for i, slider in enumerate(self.variable_sliders):
            if not slider.checkbox.isChecked():
                self.fixed_values[i] = slider.get_value()

        self.update_constants_display()
        self.plot_function_surface()

    def create_reduced_function(self):
        def reduced_func(x, y, z):
            full_input = self.fixed_values.copy()
            while len(full_input) < self.func_dim:
                full_input.append(0)
            full_input = full_input[:self.func_dim]
            if len(self.selected_vars) >= 3:
                full_input[self.selected_vars[0]] = x
                full_input[self.selected_vars[1]] = y
                full_input[self.selected_vars[2]] = z
            elif len(self.selected_vars) == 2:
                full_input[self.selected_vars[0]] = x
                full_input[self.selected_vars[1]] = y
            elif len(self.selected_vars) == 1:
                full_input[self.selected_vars[0]] = x

            return self.func(np.array(full_input))

        return reduced_func

    def setup_default_function(self):
        self.func_name = list(TEST_FUNCTIONS_3D.keys())[0]
        self.func, self.bounds, self.optimum = TEST_FUNCTIONS_3D[self.func_name]
        self.setup_variable_controls()
        self.plot_function_surface()

    def change_function(self, func_name):
        self.func_name = func_name

        if func_name in TEST_FUNCTIONS_2D:
            self.func, self.bounds, self.optimum = TEST_FUNCTIONS_2D[func_name]
        elif func_name in TEST_FUNCTIONS_3D:
            self.func, self.bounds, self.optimum = TEST_FUNCTIONS_3D[func_name]
        else:
            self.func, self.bounds, self.optimum = TEST_FUNCTIONS_5D[func_name]

        self.setup_variable_controls()
        self.plot_function_surface()
        self.reset()

    def plot_function_surface(self):
        self.gl_widget.clear()

        if len(self.selected_vars) < 3:
            text = gl.GLTextItem(pos=(0, 0, 0), text="Select 3 variables for 3D plot")
            self.gl_widget.addItem(text)
            return

        plot_func = self.create_reduced_function()

        x_min = self.variable_sliders[self.selected_vars[0]].min_val
        x_max = self.variable_sliders[self.selected_vars[0]].max_val
        y_min = self.variable_sliders[self.selected_vars[1]].min_val
        y_max = self.variable_sliders[self.selected_vars[1]].max_val
        z_min = self.variable_sliders[self.selected_vars[2]].min_val
        z_max = self.variable_sliders[self.selected_vars[2]].max_val
        fixed_z = self.fixed_values[self.selected_vars[2]]

        resolution = 30
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = plot_func(X[i, j], Y[i, j], fixed_z)

        z_vals = Z.flatten()
        z_min_val, z_max_val = z_vals.min(), z_vals.max()

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

                if z_max_val > z_min_val:
                    norm_z = (Z[i, j] - z_min_val) / (z_max_val - z_min_val)
                else:
                    norm_z = 0.5

                c = (norm_z, 0.5, 1.0 - norm_z, 1.0)
                colors.extend([c, c, c, c])

                faces.append([idx, idx + 1, idx + 2])
                faces.append([idx + 1, idx + 3, idx + 2])

        if verts:
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

        axis_length = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.3
        x_axis = gl.GLLinePlotItem(
            pos=np.array([[-axis_length, 0, 0], [axis_length, 0, 0]]),
            color=(1, 0, 0, 1), width=2
        )
        self.gl_widget.addItem(x_axis)

        y_axis = gl.GLLinePlotItem(
            pos=np.array([[0, -axis_length, 0], [0, axis_length, 0]]),
            color=(0, 1, 0, 1), width=2
        )
        self.gl_widget.addItem(y_axis)

        z_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, -axis_length], [0, 0, axis_length]]),
            color=(0, 0, 1, 1), width=2
        )
        self.gl_widget.addItem(z_axis)

        self.population_scatter = gl.GLScatterPlotItem(
            size=15, color=(1, 0, 0, 1), pxMode=True
        )
        self.gl_widget.addItem(self.population_scatter)

        self.best_scatter = gl.GLScatterPlotItem(
            size=25, color=(0, 1, 0, 1), pxMode=True
        )
        self.gl_widget.addItem(self.best_scatter)

        self.gl_widget.addItem(gl.GLTextItem(pos=(axis_length * 1.2, 0, 0), text=f"x{self.selected_vars[0] + 1}"))
        self.gl_widget.addItem(gl.GLTextItem(pos=(0, axis_length * 1.2, 0), text=f"x{self.selected_vars[1] + 1}"))
        self.gl_widget.addItem(gl.GLTextItem(pos=(0, 0, axis_length * 1.2), text=f"x{self.selected_vars[2] + 1}"))

    def initialize(self):
        params = HBAParams(
            pop_size=self.spin_pop.value(),
            max_iter=self.spin_iter.value()
        )
        self.hba = HoneyBadgerAlgorithm(params)
        self.hba.set_optimization_problem(self.func, self.func_dim, self.bounds)

        self.btn_step.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.btn_init.setEnabled(False)

        self.playback_mode = False

        for lines in self.trajectory_history:
            for line in lines:
                self.gl_widget.removeItem(line)
        self.trajectory_history = []

        self.update_display()
        self.update_3d_plot()

    def step_optimization(self):
        if self.hba is None or self.playback_mode:
            return

        if self.hba.current_iter >= self.hba.params.max_iter:
            if self.is_animating:
                self.toggle_animation()
            return

        if hasattr(self.hba, 'population'):
            self.hba.previous_population = self.hba.population.copy()

        self.hba.run_one_iteration()

        if self.is_recording and self.recording:
            self.recording.record_step(
                self.hba.population,
                self.hba.best_solution,
                self.hba.best_fitness
            )

        self.update_3d_plot()
        self.update_display()

    def update_3d_plot(self):
        if self.hba is None or self.hba.population is None or len(self.selected_vars) < 3:
            return

        x = self.hba.population[:, self.selected_vars[0]]
        y = self.hba.population[:, self.selected_vars[1]]
        z = self.hba.population[:, self.selected_vars[2]]

        pos = np.column_stack([x, y, z])
        self.population_scatter.setData(pos=pos)

        if self.hba.best_solution is not None:
            best_pos = np.array([[
                self.hba.best_solution[self.selected_vars[0]],
                self.hba.best_solution[self.selected_vars[1]],
                self.hba.best_solution[self.selected_vars[2]]
            ]])
            self.best_scatter.setData(pos=best_pos)

        if hasattr(self.hba, 'previous_population') and self.hba.previous_population is not None:
            for lines in self.trajectory_history:
                for line in lines:
                    self.gl_widget.removeItem(line)
            self.trajectory_history = []

            new_lines = []
            for i in range(len(x)):
                x_old = self.hba.previous_population[i, self.selected_vars[0]]
                y_old = self.hba.previous_population[i, self.selected_vars[1]]
                z_old = self.hba.previous_population[i, self.selected_vars[2]]

                if (abs(x[i] - x_old) > 0.01 or
                        abs(y[i] - y_old) > 0.01 or
                        abs(z[i] - z_old) > 0.01):
                    line = gl.GLLinePlotItem(
                        pos=np.array([[x_old, y_old, z_old], [x[i], y[i], z[i]]]),
                        color=(0, 0, 1, 0.3),
                        width=1,
                        antialias=True
                    )
                    self.gl_widget.addItem(line)
                    new_lines.append(line)

            if new_lines:
                self.trajectory_history.append(new_lines)

    def update_display(self):
        self.iter_label.setText(f"Iteration: {self.hba.current_iter}/{self.hba.params.max_iter}")
        self.best_label.setText(f"Best Fitness: {self.hba.best_fitness:.6f}")

        if self.hba.best_solution is not None:
            sol_str = ", ".join([f"{v:.4f}" for v in self.hba.best_solution[:5]])
            self.solution_label.setText(f"Best Solution: [{sol_str}]")

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

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.btn_record.setText("◼ Stop")
            self.btn_record.setStyleSheet("color: black; font-weight: bold;")
            self.recording = OptimizationRecording()
            self.recording.function_name = self.func_name
            self.recording.function_dim = self.func_dim
            self.recording.bounds = self.bounds
            self.recording.selected_vars = self.selected_vars.copy()
            self.recording.fixed_values = self.fixed_values.copy()
            self.recording.timestamp = datetime.now().isoformat()

            if self.hba and self.hba.population is not None:
                self.recording.record_step(
                    self.hba.population,
                    self.hba.best_solution,
                    self.hba.best_fitness
                )
        else:
            self.is_recording = False
            self.btn_record.setText("● Record")
            self.btn_record.setStyleSheet("color: red; font-weight: bold;")
            self.btn_save_recording.setEnabled(True)

    def save_recording(self):
        if self.recording and self.recording.total_steps > 0:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Recording",
                f"recording_{self.func_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "JSON Files (*.json)"
            )
            if filename:
                self.recording.save(filename)
                self.iter_label.setText(f"Recording saved to {filename}")

    def load_recording(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Recording", "", "JSON Files (*.json)"
        )
        if filename:
            self.recording = OptimizationRecording().load(filename)
            self.playback_mode = True
            self.current_playback_step = 0
            self.btn_prev.setEnabled(True)
            self.btn_next.setEnabled(True)
            self.btn_step.setEnabled(False)
            self.btn_play.setEnabled(False)
            self.btn_init.setEnabled(False)
            self.btn_record.setEnabled(False)

            self.show_playback_step(0)
            self.iter_label.setText(f"Loaded recording: {self.recording.function_name}")

    def show_playback_step(self, step):
        if not self.recording or step < 0 or step >= self.recording.total_steps:
            return

        data = self.recording.get_step(step)
        if data:
            self.current_playback_step = step
            self.step_label.setText(f"Step: {step + 1}/{self.recording.total_steps}")
            self.update_playback_display(data)

    def update_playback_display(self, data):
        if len(self.selected_vars) < 3:
            return

        pop = data['population']
        pos = np.zeros((pop.shape[0], 3))
        pos[:, 0] = pop[:, self.selected_vars[0]]
        pos[:, 1] = pop[:, self.selected_vars[1]]
        pos[:, 2] = pop[:, self.selected_vars[2]]
        self.population_scatter.setData(pos=pos)

        best = data['best_solution']
        best_pos = np.array([[
            best[self.selected_vars[0]],
            best[self.selected_vars[1]],
            best[self.selected_vars[2]]
        ]])
        self.best_scatter.setData(pos=best_pos)

        self.iter_label.setText(f"Playback - Step: {data['step'] + 1}/{data['total_steps']}")
        self.best_label.setText(f"Best Fitness: {data['best_fitness']:.6f}")

        sol_str = ", ".join([f"{v:.4f}" for v in best[:5]])
        self.solution_label.setText(f"Best Solution: [{sol_str}]")

    def prev_step(self):
        if self.playback_mode and self.current_playback_step > 0:
            self.show_playback_step(self.current_playback_step - 1)

    def next_step(self):
        if self.playback_mode and self.current_playback_step < self.recording.total_steps - 1:
            self.show_playback_step(self.current_playback_step + 1)

    def exit_playback_mode(self):
        self.playback_mode = False
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.btn_step.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_init.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.step_label.setText("Step: 0/0")

        if self.hba:
            self.update_3d_plot()
            self.update_display()

    def reset(self):
        if self.is_animating:
            self.toggle_animation()

        self.hba = None
        self.playback_mode = False
        self.is_recording = False
        self.btn_record.setText("● Record")
        self.btn_record.setStyleSheet("color: red; font-weight: bold;")
        self.btn_record.setEnabled(False)
        self.btn_save_recording.setEnabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.btn_step.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_init.setEnabled(True)
        self.step_label.setText("Step: 0/0")

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
            'timestamp': datetime.now().isoformat(),
            'selected_variables': self.selected_vars,
            'fixed_values': self.fixed_values
        }

        filename = f"{self.func_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        DataExporter.export_to_json(export_data, filename, subfolder="benchmark_3d")

    def on_tab_selected(self):
        pass