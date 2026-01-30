import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout, QMainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import AppLogger
import logging
from WiredConnection import SerialConnection


class MyWindow(QWidget):
    serial_connection = SerialConnection()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3DGS Accelerator Graphic User Interface - CMU EECS Group")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        connect_button = QPushButton("Connect")
        connect_button.clicked.connect(self.on_connect_button_click)
        disconnect_button = QPushButton("Disconnect")
        disconnect_button.clicked.connect(self.on_disconnect_button_click)
        send_button = QPushButton("Send Data")
        send_button.clicked.connect(self.on_send_button_click)

        dlg = AppLogger.MyDialog()

        layout.addWidget(connect_button)
        layout.addWidget(disconnect_button)
        layout.addWidget(send_button)
        layout.addWidget(dlg)

        self.setLayout(layout)

        eecs_pixmap = QPixmap('./eecs_group_logo_merged.png')
        eecs_pixmap = eecs_pixmap.scaledToHeight(100, Qt.SmoothTransformation)

        eecs_logo_label = QLabel()
        eecs_logo_label.setPixmap(eecs_pixmap)
        eecs_logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(eecs_logo_label)


    def on_connect_button_click(self):
        logging.info("connecting to FPGA")
        self.serial_connection.connect()


    def on_disconnect_button_click(self):
        logging.info("disconnecting...")
        self.serial_connection.disconnect()

    def on_send_button_click(self, serial):
        logging.info("sending data to FPGA")


if __name__ == "__main__": 
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    logging.info("Application setup complete")

    sys.exit(app.exec())
        