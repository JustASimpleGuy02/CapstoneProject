import sys
import os

sys.path.append(os.path.join(__file__, "..", ".."))

from PyQt5.QtWidgets import QApplication
from App.main_window import MainWindow

if __name__ == "__main__":
    app=QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())