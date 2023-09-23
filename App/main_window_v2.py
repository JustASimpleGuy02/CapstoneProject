import sys
import os
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from App.worker.worker import Worker
from App.controller import search_controller

IMAGE_FOLDER = os.path.abspath(os.path.join(__file__, "..", "images"))
QSS_FOLDER = os.path.abspath(os.path.join(__file__, "..", "qss_files"))


def rescale_image(image):
    h, w = image.shape[:2]
    max_dim = max(h, w)

    scale_factor = 500 / max_dim

    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    resized_image = cv2.resize(
        image, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC
    )

    return resized_image


def numpy_to_qpixmap(image):
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    qimage = QImage(
        image.data, width, height, bytes_per_line, QImage.Format_RGB888
    )
    qpixmap = QPixmap.fromImage(qimage.rgbSwapped())
    return qpixmap


class MainWindow(QMainWindow):
    search_done = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setup_UI()
        self.threadpool = QThreadPool.globalInstance()
        self.search_done.connect(self.showing_result)

    def setup_UI(self):
        self.setWindowIcon(QIcon(os.path.join(IMAGE_FOLDER, "hanger.png")))
        self.setWindowTitle("Demo App")

        self.main_frame = QFrame()
        self.main_layout = QHBoxLayout()
        self.function_layout = QVBoxLayout()

        # Side bar
        self.side_bar = QFrame()
        self.side_bar.setStyleSheet(
            open(os.path.join(QSS_FOLDER, "side_bar.qss")).read()
        )
        self.side_bar_layout = QVBoxLayout()
        # self.side_bar_layout.setSpacing(0)
        self.side_bar_layout.setContentsMargins(5, 5, 5, 0)
        self.side_bar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.search_tab_button = QPushButton()
        self.search_tab_button.setFixedSize(QSize(50, 50))
        self.search_tab_button.setIcon(
            QIcon(os.path.join(IMAGE_FOLDER, "search_tab.png"))
        )
        self.search_tab_button.setStyleSheet(
            open(os.path.join(QSS_FOLDER, "button.qss")).read()
        )

        self.view_tab_button = QPushButton()
        self.view_tab_button.setFixedSize(QSize(50, 50))
        self.view_tab_button.setIcon(
            QIcon(os.path.join(IMAGE_FOLDER, "view_tab.png"))
        )
        self.view_tab_button.setStyleSheet(
            open(os.path.join(QSS_FOLDER, "button.qss")).read()
        )

        self.side_bar_layout.addWidget(
            self.search_tab_button, 0, Qt.AlignmentFlag.AlignTop
        )
        self.side_bar_layout.addWidget(
            self.view_tab_button, 0, Qt.AlignmentFlag.AlignTop
        )
        self.side_bar.setLayout(self.side_bar_layout)

        # Search frame
        self.search_frame = QFrame()
        self.search_layout = QHBoxLayout()
        self.search_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Enter prompt...")

        self.font = QFont()
        self.font.setPointSize(18)
        self.font.setBold(True)
        self.search_box.setFont(self.font)
        self.search_box.setMinimumHeight(50)
        self.search_box.setStyleSheet(
            open(os.path.join(QSS_FOLDER, "search_box.qss")).read()
        )

        self.search_button = QPushButton()
        self.search_button.setFixedSize(QSize(50, 50))
        self.search_button.setIcon(
            QIcon(os.path.join(IMAGE_FOLDER, "magnifying_glass.png"))
        )
        self.search_button.setStyleSheet(
            open(os.path.join(QSS_FOLDER, "button.qss")).read()
        )

        self.clear_output_button = QPushButton()
        self.clear_output_button.setFixedSize(QSize(50, 50))
        self.clear_output_button.setIcon(
            QIcon(os.path.join(IMAGE_FOLDER, "clear.png"))
        )
        self.clear_output_button.setStyleSheet(
            open(os.path.join(QSS_FOLDER, "button.qss")).read()
        )

        self.search_layout.addWidget(self.search_box)
        self.search_layout.addWidget(self.search_button)
        self.search_layout.addWidget(self.clear_output_button)
        self.search_frame.setLayout(self.search_layout)

        # Result frame
        self.result_frame = QFrame()
        self.result_frame.setMinimumSize(700, 700)
        #        self.result_layout = QStackedLayout()
        self.result_layout = QHBoxLayout()
        self.result_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.result_screen = QLabel()
        self.result_screen.setFont(self.font)
        self.result_screen.setPixmap(QPixmap())
        self.result_screen.setText("<b>Empty</b>")
        self.result_screen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_layout.addWidget(self.result_screen)

        self.result_frame.setLayout(self.result_layout)

        self.function_layout.addWidget(self.search_frame)
        self.function_layout.addWidget(self.result_frame)
        self.main_layout.addWidget(self.side_bar)
        self.main_layout.addLayout(self.function_layout)
        self.main_frame.setLayout(self.main_layout)
        self.setCentralWidget(self.main_frame)

        self.search_button.clicked.connect(self.search_function)
        self.clear_output_button.clicked.connect(self.clear_output_function)

    def search_function(self):
        self.clear_output_function()
        input_prompt = self.search_box.text()

        if len(input_prompt) == 0:
            self.result_screen.setText("<b>Input prompt empty!</b>")
        else:
            self.result_screen.setText("<b>Searching...</b>")
            # print(input_prompt)

            self.threadpool.start(
                Worker(
                    search_controller.search,
                    prompt=input_prompt,
                    search_done=self.search_done,
                    n_sample=-1,
                )
            )

    def showing_result(self, images):
        self.result_screen.setText("")

        for image in images:
            image = rescale_image(image)
            self.result_screen = QLabel()
            self.result_screen.setFont(self.font)
            self.result_screen.setPixmap(numpy_to_qpixmap(image))
            self.result_screen.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.result_layout.addWidget(self.result_screen)

    def clear_output_function(self):
        for i in reversed(range(self.result_layout.count())):
            self.result_layout.itemAt(i).widget().setParent(None)
        self.result_screen.setPixmap(QPixmap())
        self.result_screen.setText("<b>Empty</b>")
        self.result_layout.addWidget(self.result_screen)

    def add_item_function(self):
        pass
