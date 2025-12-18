from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel,
                             QComboBox, QGroupBox, QDialog,
                             QTextEdit, QDialogButtonBox)
from PyQt6.QtCore import Qt
from ui.plot_2d import Plot2DWindow
from ui.plot_3d import Plot3DWindow
from core.functions import TEST_FUNCTIONS_2D, TEST_FUNCTIONS_3D
from core.engineering import ENGINEERING_PROBLEMS
from core.honey_badger import HoneyBadgerAlgorithm, HBAParams


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Honey Badger Algorithm Visualizer")
        self.setGeometry(100, 100, 500, 400)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        title = QLabel("Honey Badger Algorithm (HBA)")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)

        desc = QLabel(
            "Original HBA implementation based on:\n"
            "Hashim et al. (2022) - Mathematics and Computers in Simulation\n\n"
            "Select optimization mode:"
        )
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        group_2d = QGroupBox("2D Benchmark Functions")
        layout_2d = QVBoxLayout()

        self.combo_2d = QComboBox()
        self.combo_2d.addItems(TEST_FUNCTIONS_2D.keys())
        layout_2d.addWidget(QLabel("Select test function:"))
        layout_2d.addWidget(self.combo_2d)

        btn_2d = QPushButton("Open 2D Visualizer")
        btn_2d.clicked.connect(self.open_2d)
        layout_2d.addWidget(btn_2d)

        group_2d.setLayout(layout_2d)
        layout.addWidget(group_2d)

        group_3d = QGroupBox("3D Benchmark Functions")
        layout_3d = QVBoxLayout()

        self.combo_3d = QComboBox()
        self.combo_3d.addItems(TEST_FUNCTIONS_3D.keys())
        layout_3d.addWidget(QLabel("Select test function:"))
        layout_3d.addWidget(self.combo_3d)

        btn_3d = QPushButton("Open 3D Visualizer")
        btn_3d.clicked.connect(self.open_3d)
        layout_3d.addWidget(btn_3d)

        group_3d.setLayout(layout_3d)
        layout.addWidget(group_3d)

        group_eng = QGroupBox("Engineering Design Problems")
        layout_eng = QVBoxLayout()

        self.combo_eng = QComboBox()
        self.combo_eng.addItems(ENGINEERING_PROBLEMS.keys())
        layout_eng.addWidget(QLabel("Select engineering problem:"))
        layout_eng.addWidget(self.combo_eng)

        btn_eng = QPushButton("Solve Engineering Problem")
        btn_eng.clicked.connect(self.solve_engineering)
        layout_eng.addWidget(btn_eng)

        group_eng.setLayout(layout_eng)
        layout.addWidget(group_eng)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

    def open_2d(self):
        func_name = self.combo_2d.currentText()
        func, bounds, optimum = TEST_FUNCTIONS_2D[func_name]

        self.plot_2d = Plot2DWindow(func, bounds, optimum, func_name)
        self.plot_2d.show()
        self.status_label.setText(f"2D mode: {func_name}")

    def open_3d(self):
        func_name = self.combo_3d.currentText()
        func, bounds, optimum = TEST_FUNCTIONS_3D[func_name]

        self.plot_3d = Plot3DWindow(func, bounds, optimum, func_name)
        self.plot_3d.show()
        self.status_label.setText(f"3D mode: {func_name}")

    def solve_engineering(self):
        problem_name = self.combo_eng.currentText()
        problem = ENGINEERING_PROBLEMS[problem_name]

        dialog = EngineeringDialog(problem, problem_name, self)
        dialog.exec()

        self.status_label.setText(f"Solved: {problem_name}")


class EngineeringDialog(QDialog):
    def __init__(self, problem: dict, problem_name: str, parent=None):
        super().__init__(parent)
        self.problem = problem
        self.problem_name = problem_name

        self.setWindowTitle(f"Engineering: {problem_name}")
        self.setGeometry(200, 200, 600, 500)

        self.init_ui()
        self.solve_problem()

    def init_ui(self):
        layout = QVBoxLayout()

        desc = QLabel(f"<h3>{self.problem_name}</h3>")
        desc.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(desc)

        details = QLabel(f"<b>Description:</b> {self.problem['description']}<br>"
                         f"<b>Dimensions:</b> {self.problem['dim']}<br>"
                         f"<b>Known Optimum:</b> {self.problem['optimal']:.6f}")
        details.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(details)

        self.btn_run = QPushButton("Run HBA Optimization")
        self.btn_run.clicked.connect(self.solve_problem)
        layout.addWidget(self.btn_run)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(300)
        layout.addWidget(self.results_text)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def solve_problem(self):
        self.btn_run.setEnabled(False)
        self.results_text.setText("Optimizing...\n")

        params = HBAParams(
            pop_size=50,
            max_iter=1000,
            seed=42
        )
        hba = HoneyBadgerAlgorithm(params)

        func = self.problem['function']
        bounds = self.problem['bounds']
        dim = self.problem['dim']
        best_solution, best_fitness = hba.optimize(func, dim, bounds[0])

        result_text = f"""
=== {self.problem_name} ===
Optimization Results:
• Best fitness (objective value): {best_fitness:.6f}
• Known optimum from literature: {self.problem['optimal']:.6f}
• Error relative to optimum: {abs(best_fitness - self.problem['optimal']):.6e}

Best Solution:
"""
        for i, value in enumerate(best_solution):
            var_name = self.get_variable_name(i)
            bounds_str = f"[{bounds[i][0]:.3f}, {bounds[i][1]:.3f}]"
            result_text += f"  {var_name}: {value:.6f} {bounds_str}\n"

        result_text += f"\nConvergence: {len(hba.convergence_curve)} iterations"
        result_text += f"\nFinal fitness: {hba.convergence_curve[-1]:.6f}"

        self.results_text.setText(result_text)
        self.btn_run.setEnabled(True)

    def get_variable_name(self, idx: int) -> str:
        """Get variable name based on problem type"""
        if self.problem_name == "Spring Design":
            names = ["d (wire diameter)", "D (coil diameter)", "N (active coils)"]
        elif self.problem_name == "Pressure Vessel":
            names = ["Ts (shell thickness)", "Th (head thickness)",
                     "R (inner radius)", "L (length)"]
        elif self.problem_name == "Welded Beam":
            names = ["h (weld thickness)", "l (weld length)",
                     "t (beam thickness)", "b (beam width)"]
        elif self.problem_name == "Speed Reducer":
            names = ["b (face width)", "m (module)", "z (teeth number)",
                     "l1 (length 1)", "l2 (length 2)", "d1 (diameter 1)",
                     "d2 (diameter 2)"]
        else:
            names = [f"x{i + 1}" for i in range(10)]

        return names[idx] if idx < len(names) else f"x{idx + 1}"
