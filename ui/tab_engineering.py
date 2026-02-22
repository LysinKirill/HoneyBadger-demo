from datetime import datetime
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSpinBox, QSlider,
                             QGroupBox, QComboBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QGridLayout,
                             QSplitter)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
import pyqtgraph as pg

from core.engineering import ENGINEERING_PROBLEMS
from core.honey_badger import HoneyBadgerAlgorithm, HBAParams
from core.export_data import DataExporter


class TabEngineering(QWidget):
    def __init__(self):
        super().__init__()
        self.problem_data = None
        self.problem_name = None
        self.problem = None
        self.func = None
        self.objective_func = None
        self.bounds = None
        self.dim = None
        self.hba = None
        self.is_animating = False
        self.convergence_curve = []

        self.init_ui()
        self.setup_default_problem()

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.step_optimization)

    def init_ui(self):
        main_layout = QVBoxLayout()

        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        info_group = QGroupBox("Problem Description")
        info_layout = QVBoxLayout()
        self.desc_label = QLabel()
        self.desc_label.setWordWrap(True)
        info_layout.addWidget(self.desc_label)

        self.diagram_label = QLabel()
        self.diagram_label.setFont(QFont("Courier", 9))
        self.diagram_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(self.diagram_label)
        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)

        self.convergence_plot = pg.PlotWidget()
        self.convergence_plot.setBackground('w')
        self.convergence_plot.setLabel('left', 'Objective Value')
        self.convergence_plot.setLabel('bottom', 'Iteration')
        self.convergence_plot.showGrid(x=True, y=True, alpha=0.3)
        left_layout.addWidget(self.convergence_plot)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        vars_group = QGroupBox("Design Variables")
        vars_layout = QVBoxLayout()
        self.variable_table = QTableWidget()
        self.variable_table.setColumnCount(4)
        self.variable_table.setHorizontalHeaderLabels(["Variable", "Value", "Min", "Max"])
        self.variable_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        vars_layout.addWidget(self.variable_table)
        vars_group.setLayout(vars_layout)
        right_layout.addWidget(vars_group)

        cons_group = QGroupBox("Constraints")
        cons_layout = QVBoxLayout()
        self.constraint_table = QTableWidget()
        self.constraint_table.setColumnCount(3)
        self.constraint_table.setHorizontalHeaderLabels(["Constraint", "Status", "Value"])
        self.constraint_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        cons_layout.addWidget(self.constraint_table)
        cons_group.setLayout(cons_layout)
        right_layout.addWidget(cons_group)

        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        self.iter_label = QLabel("Iteration: 0/0")
        status_layout.addWidget(self.iter_label)
        self.objective_label = QLabel("Objective: N/A")
        status_layout.addWidget(self.objective_label)
        self.feasible_label = QLabel("Feasible: N/A")
        status_layout.addWidget(self.feasible_label)
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 400])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def create_control_panel(self):
        panel = QGroupBox("Controls")
        layout = QGridLayout()

        layout.addWidget(QLabel("Problem:"), 0, 0)
        self.problem_combo = QComboBox()
        self.problem_combo.addItems(ENGINEERING_PROBLEMS.keys())
        self.problem_combo.currentTextChanged.connect(self.change_problem)
        layout.addWidget(self.problem_combo, 0, 1)

        layout.addWidget(QLabel("Population:"), 0, 2)
        self.spin_pop = QSpinBox()
        self.spin_pop.setRange(10, 100)
        self.spin_pop.setValue(30)
        layout.addWidget(self.spin_pop, 0, 3)

        layout.addWidget(QLabel("Max Iter:"), 0, 4)
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(10, 2000)
        self.spin_iter.setValue(200)
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

    def setup_default_problem(self):
        self.change_problem(list(ENGINEERING_PROBLEMS.keys())[0])

    def change_problem(self, problem_name):
        self.problem_name = problem_name
        self.problem_data = ENGINEERING_PROBLEMS[problem_name]
        self.problem = self.problem_data['problem']
        self.func = self.problem_data['function']
        self.objective_func = self.problem_data['objective']
        self.bounds = self.problem_data['bounds']
        self.dim = self.problem_data['dim']

        self.desc_label.setText(self.problem.description)
        self.diagram_label.setText(self.problem.diagram)

        self.setup_tables()
        self.reset()

    def setup_tables(self):
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

    def initialize(self):
        lower_bounds = [b[0] for b in self.bounds]
        upper_bounds = [b[1] for b in self.bounds]
        bounds_tuple = (min(lower_bounds), max(upper_bounds))

        params = HBAParams(
            pop_size=self.spin_pop.value(),
            max_iter=self.spin_iter.value()
        )
        self.hba = HoneyBadgerAlgorithm(params)
        self.hba.set_optimization_problem(self.objective_func, self.dim, bounds_tuple)

        self.convergence_curve = []
        self.convergence_plot.clear()

        self.btn_step.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_init.setEnabled(False)

        self.update_display()

    def step_optimization(self):
        if self.hba.current_iter >= self.hba.params.max_iter:
            if self.is_animating:
                self.toggle_animation()
            return

        if hasattr(self.hba, 'population'):
            self.hba.previous_population = self.hba.population.copy()

        self.hba.run_one_iteration()
        self.convergence_curve.append(self.hba.best_fitness)
        self.update_display()
        self.plot_convergence()

    def update_display(self):
        self.iter_label.setText(f"Iteration: {self.hba.current_iter}/{self.hba.params.max_iter}")

        if self.hba.best_solution is not None:
            obj_value, constraints, satisfied = self.func(self.hba.best_solution)
            self.objective_label.setText(f"Objective: {obj_value:.6f}")
            self.feasible_label.setText(f"Feasible: {'Yes' if all(satisfied) else 'No'}")

            for i in range(self.dim):
                item = QTableWidgetItem(f"{self.hba.best_solution[i]:.4f}")
                if self.hba.best_solution[i] < self.bounds[i][0] or self.hba.best_solution[i] > self.bounds[i][1]:
                    item.setBackground(QColor(255, 200, 200))
                self.variable_table.setItem(i, 1, item)

            for i in range(len(constraints)):
                status = "✓" if satisfied[i] else "✗"
                status_item = QTableWidgetItem(status)
                value_item = QTableWidgetItem(f"{constraints[i]:.4f}")

                if satisfied[i]:
                    status_item.setBackground(QColor(200, 255, 200))
                else:
                    status_item.setBackground(QColor(255, 200, 200))

                self.constraint_table.setItem(i, 1, status_item)
                self.constraint_table.setItem(i, 2, value_item)

    def plot_convergence(self):
        if len(self.convergence_curve) > 0:
            self.convergence_plot.clear()
            self.convergence_plot.plot(self.convergence_curve, pen=pg.mkPen('b', width=2))

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
        self.convergence_curve = []
        self.convergence_plot.clear()

        self.btn_step.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_init.setEnabled(True)

        self.iter_label.setText("Iteration: 0/0")
        self.objective_label.setText("Objective: N/A")
        self.feasible_label.setText("Feasible: N/A")

        for i in range(self.dim):
            self.variable_table.setItem(i, 1, QTableWidgetItem("N/A"))

        for i in range(self.constraint_table.rowCount()):
            self.constraint_table.setItem(i, 1, QTableWidgetItem("N/A"))
            self.constraint_table.setItem(i, 2, QTableWidgetItem("N/A"))

    def export_results(self):
        if self.hba is None or self.hba.best_solution is None:
            return

        export_data = {
            'problem': self.problem_name,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'pop_size': self.hba.params.pop_size,
                'max_iter': self.hba.params.max_iter,
                'iterations_completed': self.hba.current_iter
            },
            'solution': {
                'variables': self.hba.best_solution.tolist(),
                'variable_names': [v['name'] for v in self.problem.variables] if self.problem else [],
                'objective_value': float(self.hba.best_fitness)
            },
            'convergence_data': self.convergence_curve
        }

        filename = f"{self.problem_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        DataExporter.export_to_json(export_data, filename, subfolder="engineering")

    def on_tab_selected(self):
        pass