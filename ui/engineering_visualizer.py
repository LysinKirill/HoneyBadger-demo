import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
QPushButton, QLabel, QSpinBox, QSlider,
QGroupBox, QTableWidget, QTableWidgetItem,
QHeaderView)

from core.honey_badger import HoneyBadgerAlgorithm, HBAParams

class EngineeringVisualizer(QWidget):
    update_signal = pyqtSignal()

    def __init__(self, problem_data: dict, problem_name: str):
        super().__init__()
        self.problem_data = problem_data
        self.problem_name = problem_name
        self.problem = problem_data['problem']
        self.func = problem_data['function']
        self.objective_func = problem_data['objective']
        self.bounds = problem_data['bounds']
        self.dim = problem_data['dim']

        self.population_history = []
        self.current_population_scatter = None
        self.best_solution_scatter = None
        self.previous_population = None
        self.convergence_curve = []
        self.history = []
        self.trail_lines = []
        self.x_range = None
        self.y_range = None

        self.setWindowTitle(f"HBA - {problem_name}")
        self.setGeometry(200, 200, 1200, 800)

        self.init_ui()
        self.setup_algorithm()

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.step_optimization)
        self.is_animating = False
        self.update_signal.connect(self.update_display)

    def init_ui(self):
        layout = QHBoxLayout()
        left_panel = QVBoxLayout()

        title = QLabel(f"{self.problem_name}")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_panel.addWidget(title)

        if self.problem:
            desc = QLabel(self.problem.description)
            desc.setWordWrap(True)
            left_panel.addWidget(desc)
            diagram = QLabel(self.problem.diagram)
            diagram.setFont(QFont("Courier", 10))
            diagram.setAlignment(Qt.AlignmentFlag.AlignCenter)
            left_panel.addWidget(diagram)

        left_panel.addStretch(1)
        control_group = QGroupBox("Optimization Controls")
        control_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()

        self.btn_run = QPushButton("Initialize")
        self.btn_run.clicked.connect(self.initialize_optimization)
        btn_layout.addWidget(self.btn_run)

        self.btn_step = QPushButton("Step")
        self.btn_step.clicked.connect(self.step_optimization)
        self.btn_step.setEnabled(False)
        btn_layout.addWidget(self.btn_step)

        self.btn_play = QPushButton("▶ Run")
        self.btn_play.clicked.connect(self.toggle_animation)
        self.btn_play.setEnabled(False)
        btn_layout.addWidget(self.btn_play)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset)
        btn_layout.addWidget(self.btn_reset)
        control_layout.addLayout(btn_layout)

        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Population:"))
        self.spin_pop = QSpinBox()
        self.spin_pop.setRange(10, 100)
        self.spin_pop.setValue(30)
        param_layout.addWidget(self.spin_pop)
        param_layout.addWidget(QLabel("Max Iter:"))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(10, 2000)
        self.spin_iter.setValue(200)
        param_layout.addWidget(self.spin_iter)
        control_layout.addLayout(param_layout)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(10, 1000)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("100 ms")
        speed_layout.addWidget(self.speed_label)
        control_layout.addLayout(speed_layout)
        control_group.setLayout(control_layout)
        left_panel.addWidget(control_group)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        layout.addWidget(left_widget, 1)

        center_panel = QVBoxLayout()
        self.convergence_plot = pg.PlotWidget()
        self.convergence_plot.setBackground('w')
        self.convergence_plot.setLabel('left', 'Objective Value')
        self.convergence_plot.setLabel('bottom', 'Iteration')
        self.convergence_plot.showGrid(x=True, y=True, alpha=0.3)
        self.convergence_plot.setMinimumHeight(300)
        center_panel.addWidget(self.convergence_plot)

        self.population_plot = pg.PlotWidget()
        self.population_plot.setBackground('w')
        self.population_plot.showGrid(x=True, y=True, alpha=0.3)
        self.population_plot.setMinimumHeight(300)
        self.init_plot_ranges()
        center_panel.addWidget(self.population_plot)

        info_group = QGroupBox("Current Status")
        info_layout = QVBoxLayout()
        self.iter_label = QLabel("Iteration: 0/0")
        info_layout.addWidget(self.iter_label)
        self.objective_label = QLabel("Objective (Weight/Cost): N/A")
        info_layout.addWidget(self.objective_label)
        self.best_label = QLabel("Best Found: N/A")
        info_layout.addWidget(self.best_label)
        self.status_label = QLabel("Status: Initialized")
        info_layout.addWidget(self.status_label)
        info_group.setLayout(info_layout)
        center_panel.addWidget(info_group)

        center_widget = QWidget()
        center_widget.setLayout(center_panel)
        layout.addWidget(center_widget, 2)

        right_panel = QVBoxLayout()
        variables_group = QGroupBox("Design Variables")
        variables_layout = QVBoxLayout()
        self.variable_table = QTableWidget()
        self.variable_table.setColumnCount(4)
        self.variable_table.setHorizontalHeaderLabels(["Variable", "Value", "Min", "Max"])
        self.variable_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        variables_layout.addWidget(self.variable_table)
        variables_group.setLayout(variables_layout)
        right_panel.addWidget(variables_group, 1)

        constraints_group = QGroupBox("Constraints")
        constraints_layout = QVBoxLayout()
        self.constraint_table = QTableWidget()
        self.constraint_table.setColumnCount(3)
        self.constraint_table.setHorizontalHeaderLabels(["Constraint", "Status", "Value"])
        self.constraint_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        constraints_layout.addWidget(self.constraint_table)
        constraints_group.setLayout(constraints_layout)
        right_panel.addWidget(constraints_group, 1)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        layout.addWidget(right_widget, 1)
        self.setLayout(layout)
        self.setup_tables()

    def init_plot_ranges(self):
        if self.dim >= 2:
            x_min, x_max = self.bounds[0]
            y_min, y_max = self.bounds[1]
            x_padding = (x_max - x_min) * 0.2
            y_padding = (y_max - y_min) * 0.2
            self.x_range = (x_min - x_padding, x_max + x_padding)
            self.y_range = (y_min - y_padding, y_max + y_padding)
            if self.problem and len(self.problem.variables) >= 2:
                var1 = self.problem.variables[0]['name']
                var2 = self.problem.variables[1]['name']
                self.population_plot.setLabel('bottom', var1)
                self.population_plot.setLabel('left', var2)
            else:
                self.population_plot.setLabel('bottom', 'Variable 1')
                self.population_plot.setLabel('left', 'Variable 2')
        else:
            self.x_range = (0, 1)
            self.y_range = (0, 1)
            self.population_plot.setLabel('bottom', 'N/A')
            self.population_plot.setLabel('left', 'N/A')

    def setup_tables(self):
        if self.problem:
            self.variable_table.setRowCount(self.dim)
            for i in range(self.dim):
                var_info = self.problem.get_variable_info(i)
                self.variable_table.setItem(i, 0, QTableWidgetItem(var_info['name']))
                self.variable_table.setItem(i, 1, QTableWidgetItem("N/A"))
                self.variable_table.setItem(i, 2, QTableWidgetItem(f"{self.bounds[i][0]:.3f}"))
                self.variable_table.setItem(i, 3, QTableWidgetItem(f"{self.bounds[i][1]:.3f}"))

            self.constraint_table.setRowCount(len(self.problem.constraints))
            for i in range(len(self.problem.constraints)):
                const_info = self.problem.get_constraint_info(i)
                self.constraint_table.setItem(i, 0, QTableWidgetItem(const_info['name']))
                self.constraint_table.setItem(i, 1, QTableWidgetItem("N/A"))
                self.constraint_table.setItem(i, 2, QTableWidgetItem("N/A"))

    def setup_algorithm(self):
        params = HBAParams(
            pop_size=self.spin_pop.value(),
            max_iter=self.spin_iter.value()
        )
        self.hba = HoneyBadgerAlgorithm(params)
        lower_bounds = [b[0] for b in self.bounds]
        upper_bounds = [b[1] for b in self.bounds]
        bounds_tuple = (min(lower_bounds), max(upper_bounds))
        self.hba.set_optimization_problem(self.objective_func, self.dim, bounds_tuple)
        self.convergence_curve = []

    def initialize_optimization(self):
        self.btn_run.setEnabled(False)
        self.btn_step.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.setup_algorithm()
        self.clear_optimization_data()
        self.convergence_plot.clear()
        self.population_plot.clear()
        self.set_plot_ranges()
        self.update_display()
        self.update_visualization()
        self.status_label.setText("Status: Initialized - Ready to optimize")
        self.btn_run.setText("Reinitialize")
        self.btn_run.setEnabled(True)

    def clear_optimization_data(self):
        self.history = []
        self.convergence_curve = []
        self.population_history = []
        self.trail_lines = []
        self.previous_population = None

    def set_plot_ranges(self):
        if self.dim >= 2 and self.x_range and self.y_range:
            self.population_plot.setXRange(self.x_range[0], self.x_range[1])
            self.population_plot.setYRange(self.y_range[0], self.y_range[1])

    def step_optimization(self):
        if self.hba.current_iter >= self.hba.params.max_iter:
            self.status_label.setText("Optimization complete!")
            if self.is_animating:
                self.toggle_animation()
            return

        if hasattr(self.hba, 'population') and self.hba.population is not None:
            self.previous_population = self.hba.population.copy()

        self.hba.run_one_iteration()
        self.convergence_curve.append(self.hba.best_fitness)
        self.history.append(self.hba.best_solution.copy() if self.hba.best_solution is not None else None)

        if self.hba.population is not None:
            self.population_history.append(self.hba.population.copy())

        self.update_display()
        self.update_visualization()
        self.status_label.setText(f"Status: Iteration {self.hba.current_iter} complete")

    def update_visualization(self):
        self.update_convergence_plot()
        if self.dim >= 2:
            self.update_population_plot()
        else:
            self.clear_population_plot()

    def update_convergence_plot(self):
        if len(self.convergence_curve) > 0:
            self.convergence_plot.clear()
            self.convergence_plot.plot(self.convergence_curve, pen=pg.mkPen('b', width=2))
            current_point = pg.ScatterPlotItem(
                [len(self.convergence_curve) - 1],
                [self.convergence_curve[-1]],
                pen=pg.mkPen('r'),
                brush=pg.mkBrush('r'),
                size=10
            )
            self.convergence_plot.addItem(current_point)

    def update_population_plot(self):
        if self.hba.population is None or len(self.hba.population) == 0:
            return

        self.remove_old_scatter_plots()
        self.clear_trail_lines()

        x = self.hba.population[:, 0]
        y = self.hba.population[:, 1]

        self.plot_trails(x, y)
        self.plot_current_population(x, y)
        self.plot_best_solution()

    def remove_old_scatter_plots(self):
        if self.current_population_scatter:
            self.population_plot.removeItem(self.current_population_scatter)
            self.current_population_scatter = None
        if self.best_solution_scatter:
            self.population_plot.removeItem(self.best_solution_scatter)
            self.best_solution_scatter = None

    def clear_trail_lines(self):
        for line in self.trail_lines:
            self.population_plot.removeItem(line)
        self.trail_lines = []

    def plot_trails(self, x, y):
        if self.previous_population is None or len(self.previous_population) != len(x):
            return

        for i in range(len(x)):
            x_old = self.previous_population[i, 0]
            y_old = self.previous_population[i, 1]
            if abs(x[i] - x_old) > 0.001 or abs(y[i] - y_old) > 0.001:
                line = pg.PlotDataItem(
                    [x_old, x[i]], [y_old, y[i]],
                    pen=pg.mkPen(color=(255, 0, 0, 100), width=1)
                )
                self.population_plot.addItem(line)
                self.trail_lines.append(line)

    def plot_current_population(self, x, y):
        self.current_population_scatter = pg.ScatterPlotItem(
            x, y,
            pen=pg.mkPen('b', width=1),
            brush=pg.mkBrush('b'),
            size=12,
            symbol='o',
            pxMode=True
        )
        self.population_plot.addItem(self.current_population_scatter)

    def plot_best_solution(self):
        if self.hba.best_solution is not None and len(self.hba.best_solution) >= 2:
            self.best_solution_scatter = pg.ScatterPlotItem(
                [self.hba.best_solution[0]], [self.hba.best_solution[1]],
                pen=pg.mkPen('g', width=3),
                brush=pg.mkBrush('g'),
                size=20,
                symbol='x',
                pxMode=True
            )
            self.population_plot.addItem(self.best_solution_scatter)

    def clear_population_plot(self):
        self.population_plot.clear()
        self.population_plot.setLabel('left', 'N/A')
        self.population_plot.setLabel('bottom', 'N/A')
        self.population_plot.addItem(pg.TextItem("Population visualization\navailable for 2D+ problems only",
                                                 color='k', anchor=(0.5, 0.5)))

    def update_display(self):
        self.iter_label.setText(f"Iteration: {self.hba.current_iter}/{self.hba.params.max_iter}")

        if self.hba.best_solution is not None:
            obj_value, constraints, satisfied = self.func(self.hba.best_solution)
            self.objective_label.setText(f"Objective: {obj_value:.6f}")
            self.best_label.setText(f"Best Found: {self.hba.best_fitness:.6f}")
            self.update_variable_table(obj_value, constraints, satisfied)

    def update_variable_table(self, obj_value, constraints, satisfied):
        for i in range(self.dim):
            if i < len(self.hba.best_solution):
                value = self.hba.best_solution[i]
                item = QTableWidgetItem(f"{value:.4f}")
                if value < self.bounds[i][0] or value > self.bounds[i][1]:
                    item.setBackground(QColor(255, 200, 200))
                self.variable_table.setItem(i, 1, item)

        for i in range(len(constraints)):
            status = "✓ Satisfied" if satisfied[i] else "✗ Violated"
            value = f"{constraints[i]:.4f}"
            status_item = QTableWidgetItem(status)
            value_item = QTableWidgetItem(value)
            if satisfied[i]:
                status_item.setBackground(QColor(200, 255, 200))
            else:
                status_item.setBackground(QColor(255, 200, 200))
            self.constraint_table.setItem(i, 1, status_item)
            self.constraint_table.setItem(i, 2, value_item)

    def toggle_animation(self):
        if not self.is_animating:
            self.is_animating = True
            self.btn_play.setText("⏸ Pause")
            self.btn_step.setEnabled(False)
            self.btn_run.setEnabled(False)
            self.animation_timer.start(self.speed_slider.value())
            self.status_label.setText("Status: Running animation...")
        else:
            self.is_animating = False
            self.btn_play.setText("▶ Run")
            self.btn_step.setEnabled(True)
            self.btn_run.setEnabled(True)
            self.animation_timer.stop()
            self.status_label.setText("Status: Animation paused")

    def update_speed(self):
        speed = self.speed_slider.value()
        self.speed_label.setText(f"{speed} ms")
        if self.is_animating:
            self.animation_timer.setInterval(speed)

    def reset(self):
        if self.is_animating:
            self.toggle_animation()

        self.setup_algorithm()
        self.clear_optimization_data()
        self.convergence_plot.clear()
        self.population_plot.clear()
        self.init_plot_ranges()
        self.set_plot_ranges()

        for i in range(self.dim):
            self.variable_table.setItem(i, 1, QTableWidgetItem("N/A"))

        for i in range(self.constraint_table.rowCount()):
            self.constraint_table.setItem(i, 1, QTableWidgetItem("N/A"))
            self.constraint_table.setItem(i, 2, QTableWidgetItem("N/A"))

        self.iter_label.setText("Iteration: 0/0")
        self.objective_label.setText("Objective (Weight/Cost): N/A")
        self.best_label.setText("Best Found: N/A")
        self.status_label.setText("Status: Reset complete")