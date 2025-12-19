import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow
from pathlib import Path


def setup_directories():
    directories = [
        "iodata/exports/benchmark_2d",
        "iodata/exports/benchmark_3d",
        "iodata/exports/engineering",
        "iodata/configs",
        "iodata/logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print(f"Created directory structure in: {Path('iodata').absolute()}")


def main():
    setup_directories()

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
