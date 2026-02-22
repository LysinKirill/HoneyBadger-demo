from PyQt6.QtWidgets import QTabWidget
from ui.tab_2d import Tab2D
from ui.tab_3d import Tab3D
from ui.tab_engineering import TabEngineering


class VisualizationTabs(QTabWidget):
    def __init__(self):
        super().__init__()

        self.tab_2d = Tab2D()
        self.tab_3d = Tab3D()
        self.tab_engineering = TabEngineering()

        self.addTab(self.tab_2d, "2D Benchmark Functions")
        self.addTab(self.tab_3d, "3D Benchmark Functions")
        self.addTab(self.tab_engineering, "Engineering Problems")

        self.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        current_tab = self.widget(index)
        if hasattr(current_tab, 'on_tab_selected'):
            current_tab.on_tab_selected()
