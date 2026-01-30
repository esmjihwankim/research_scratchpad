import serial
import logging

class SerialConnection:
    ser = serial.Serial()

    def connect(self):
        self.ser = serial.Serial('COM8', 115200)
        if self.ser.is_open:
            logging.info("serial port is open")
        else:
            logging.info("serial port is not open")

    def disconnect(self):
        self.ser.close()
        logging.info("serial port is closed")
