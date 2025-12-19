from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QLabel,
                             QComboBox, QGroupBox)

from core.engineering import ENGINEERING_PROBLEMS
from core.functions import TEST_FUNCTIONS_2D, TEST_FUNCTIONS_3D
from ui.engineering_visualizer import EngineeringVisualizer
from ui.plot_2d import Plot2DWindow
from ui.plot_3d import Plot3DWindow


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

        self.engineering_visualizer = EngineeringVisualizer(problem, problem_name)
        self.engineering_visualizer.show()

        self.status_label.setText(f"Opened: {problem_name}")
