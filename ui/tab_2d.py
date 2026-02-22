from datetime import datetime

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox, QSlider,
                             QGroupBox, QComboBox, QGridLayout, QSplitter,
                             QCheckBox, QLineEdit, QScrollArea)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

from core.export_data import DataExporter
from core.functions import TEST_FUNCTIONS_2D, TEST_FUNCTIONS_3D, TEST_FUNCTIONS_5D
from core.honey_badger import HoneyBadgerAlgorithm, HBAParams


class VariableSlider(QWidget):
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


class Tab2D(QWidget):
    def __init__(self):
        super().__init__()
        self.func = None
        self.bounds = None
        self.optimum = None
        self.func_name = None
        self.func_dim = 2
        self.hba = None
        self.is_animating = False
        self.trail_lines = []
        self.variable_sliders = []
        self.fixed_values = []
        self.selected_vars = [0, 1]

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

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'Y')
        self.plot_widget.setLabel('bottom', 'X')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.mouse_label = pg.TextItem("", color='k', anchor=(0, 1))
        self.plot_widget.addItem(self.mouse_label)

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

        panel.setLayout(layout)
        return panel

    def create_variable_panel(self):
        panel = QGroupBox("Variable Controls (Select 2 variables for 2D plot)")
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

        opt_group = QGroupBox("Known Optimum")
        opt_layout = QVBoxLayout()
        self.opt_label = QLabel("Loading...")
        opt_layout.addWidget(self.opt_label)
        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)

        cursor_group = QGroupBox("Cursor Position")
        cursor_layout = QVBoxLayout()
        self.cursor_label = QLabel("Move mouse over plot")
        cursor_layout.addWidget(self.cursor_label)
        cursor_group.setLayout(cursor_layout)
        layout.addWidget(cursor_group)

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

            slider = VariableSlider(i, f"x{i + 1}", min_val, max_val, initial, self)
            self.var_layout.addWidget(slider)
            self.variable_sliders.append(slider)
            self.fixed_values[i] = initial

        if len(self.variable_sliders) >= 2:
            self.variable_sliders[0].checkbox.setChecked(True)
            self.variable_sliders[1].checkbox.setChecked(True)

    def update_selected_variables(self):
        selected = []
        for i, slider in enumerate(self.variable_sliders):
            if slider.checkbox.isChecked():
                selected.append(i)

        if len(selected) > 2:
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

        if len(self.selected_vars) < 2:
            for i in range(min(2, len(self.variable_sliders))):
                if i not in self.selected_vars:
                    self.variable_sliders[i].checkbox.setChecked(True)
                    self.selected_vars.append(i)
                    self.variable_sliders[i].set_active(False)

        self.on_variable_changed()

    def on_variable_changed(self):
        for i, slider in enumerate(self.variable_sliders):
            if not slider.checkbox.isChecked():
                self.fixed_values[i] = slider.get_value()

        self.plot_function_surface()

    def create_reduced_function(self):

        def reduced_func(x, y):
            full_input = self.fixed_values.copy()
            while len(full_input) < self.func_dim:
                full_input.append(0)
            full_input = full_input[:self.func_dim]

            if len(self.selected_vars) >= 2:
                full_input[self.selected_vars[0]] = x
                full_input[self.selected_vars[1]] = y

            return self.func(np.array(full_input))

        return reduced_func
    def setup_default_function(self):
        self.func_name = list(TEST_FUNCTIONS_2D.keys())[0]
        self.func, self.bounds, self.optimum = TEST_FUNCTIONS_2D[self.func_name]
        self.setup_variable_controls()
        self.plot_function_surface()
        self.update_optimum_label()

    def change_function(self, func_name):
        self.func_name = func_name

        if func_name in TEST_FUNCTIONS_2D:
            self.func, self.bounds, self.optimum = TEST_FUNCTIONS_2D[func_name]
            self.func_dim = 2
        elif func_name in TEST_FUNCTIONS_3D:
            self.func, self.bounds, self.optimum = TEST_FUNCTIONS_3D[func_name]
            self.func_dim = 3
        else:
            self.func, self.bounds, self.optimum = TEST_FUNCTIONS_5D[func_name]
            self.func_dim = 5

        self.setup_variable_controls()
        self.plot_function_surface()
        self.update_optimum_label()
        self.reset()

    def plot_function_surface(self):
        self.plot_widget.clear()

        if len(self.selected_vars) < 2:
            return

        plot_func = self.create_reduced_function()

        x_min = self.variable_sliders[self.selected_vars[0]].min_val
        x_max = self.variable_sliders[self.selected_vars[0]].max_val
        y_min = self.variable_sliders[self.selected_vars[1]].min_val
        y_max = self.variable_sliders[self.selected_vars[1]].max_val

        resolution = 100
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = plot_func(X[i, j], Y[i, j])

        img = pg.ImageItem(Z.T)
        img.setRect(x_min, y_min, x_max - x_min, y_max - y_min)
        self.plot_widget.addItem(img)

        cmap = pg.colormap.get('viridis')
        bar = pg.ColorBarItem(colorMap=cmap, values=(Z.min(), Z.max()))
        bar.setImageItem(img)

        self.plot_widget.setLabel('bottom', f'x{self.selected_vars[0] + 1}')
        self.plot_widget.setLabel('left', f'x{self.selected_vars[1] + 1}')

        self.population_scatter = pg.ScatterPlotItem(
            pen=pg.mkPen('b'), brush=pg.mkBrush('b'),
            size=10, symbol='o', pxMode=True
        )
        self.plot_widget.addItem(self.population_scatter)

    def on_mouse_moved(self, pos):
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            if len(self.selected_vars) >= 2:
                try:
                    plot_func = self.create_reduced_function()
                    z = plot_func(x, y)
                    self.cursor_label.setText(
                        f"x{self.selected_vars[0] + 1}={x:.3f}, "
                        f"x{self.selected_vars[1] + 1}={y:.3f}, "
                        f"f={z:.3f}"
                    )
                except:
                    pass

    def update_optimum_label(self):
        if len(self.optimum) >= 2:
            opt_str = ", ".join([f"{v:.3f}" for v in self.optimum[:3]])
            self.opt_label.setText(f"f({opt_str}) = {self.func(self.optimum):.3f}")

    def initialize(self):
        if self.func_dim == 2:
            opt_bounds = self.bounds
        else:
            opt_bounds = self.bounds

        params = HBAParams(
            pop_size=self.spin_pop.value(),
            max_iter=self.spin_iter.value()
        )
        self.hba = HoneyBadgerAlgorithm(params)
        self.hba.set_optimization_problem(self.func, self.func_dim, opt_bounds)

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

    def update_population_plot(self):
        if self.hba is None or self.hba.population is None or len(self.selected_vars) < 2:
            return

        x = self.hba.population[:, self.selected_vars[0]]
        y = self.hba.population[:, self.selected_vars[1]]
        self.population_scatter.setData(x, y)

        if hasattr(self.hba, 'previous_population') and self.hba.previous_population is not None:
            for line in self.trail_lines:
                self.plot_widget.removeItem(line)
            self.trail_lines = []

            for i in range(len(x)):
                x_old = self.hba.previous_population[i, self.selected_vars[0]]
                y_old = self.hba.previous_population[i, self.selected_vars[1]]
                if abs(x[i] - x_old) > 0.01 or abs(y[i] - y_old) > 0.01:
                    line = pg.PlotDataItem(
                        [x_old, x[i]], [y_old, y[i]],
                        pen=pg.mkPen(color=(255, 0, 0, 100), width=1, style=Qt.PenStyle.DashLine)
                    )
                    self.plot_widget.addItem(line)
                    self.trail_lines.append(line)

    def update_display(self):
        if self.hba is None:
            return

        self.iter_label.setText(f"Iteration: {self.hba.current_iter}/{self.hba.params.max_iter}")
        self.best_label.setText(f"Best Fitness: {self.hba.best_fitness:.6f}")

        if self.hba.best_solution is not None:
            sol_str = ", ".join([f"{v:.4f}" for v in self.hba.best_solution[:3]])
            self.solution_label.setText(f"Best Solution: [{sol_str}]")

        phase = getattr(self.hba, 'current_phase', 'Unknown')
        self.phase_label.setText(f"Phase: {phase}")

    def step_optimization(self):
        if self.hba is None:
            return

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

    def plot_convergence(self):
        if self.hba and len(self.hba.convergence_curve) > 0:
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

        if hasattr(self, 'population_scatter'):
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
            'timestamp': datetime.now().isoformat(),
            'selected_variables': self.selected_vars,
            'fixed_values': self.fixed_values
        }

        filename = f"{self.func_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        DataExporter.export_to_json(export_data, filename, subfolder="benchmark_2d")

    def on_tab_selected(self):
        pass