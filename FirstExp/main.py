from sys import path,argv,exit
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIntValidator

from UI_packages import Ui_MainWindow as MW


class MAIN_UI(QtWidgets.QMainWindow):
    def __init__(self):
        super(MAIN_UI,self).__init__()
        self.ui = MW.Ui_MainWindow()
        self.ui.setupUi(self)#* 建立界面
        self.show()
        
        
        
if __name__ == '__main__':
    app=QApplication(argv)
    win1=MAIN_UI()
    exit(app.exec())