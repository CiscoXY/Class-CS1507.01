from sys import path,argv,exit
import os
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIntValidator , QPixmap

from UI_packages import Ui_MainWindow as MW


class MAIN_UI(QtWidgets.QMainWindow):
    def __init__(self):
        super(MAIN_UI,self).__init__()
        self.ui = MW.Ui_MainWindow()
        self.ui.setupUi(self)#* 建立界面
        self.show()
        
        #*创建图片对象
        self.paiming_pingfen_scatter = QPixmap(os.path.abspath('images/describe/排名_评分.jpg')).scaled(990, 990)
        self.paiming_pingjiarenshu_scatter = QPixmap(os.path.abspath('images/describe/排名_评价人数.jpg')).scaled(990, 990)
        self.pingfen_pingjiarenshu_scatter = QPixmap(os.path.abspath('images/describe/评分_评价人数.jpg')).scaled(990, 990)
        
        
        self.pingfen_pingjiarenshu_hist = QPixmap(os.path.abspath('images/describe/评分_评价人数_hist.jpg')).scaled(990, 990)
        self.yingpianzhonglei_hist = QPixmap(os.path.abspath('images/describe/影片种类占比.jpg')).scaled(990, 990)
        self.shangyingshijian_hist = QPixmap(os.path.abspath('images/describe/上映时期(年)占比.jpg')).scaled(990, 990)
        self.zhizuoguojia_hist = QPixmap(os.path.abspath('images/describe/制片国家_地区占比.jpg')).scaled(990, 990)
        self.yingpianyuyan_hist = QPixmap(os.path.abspath('images/describe/影片语言占比.jpg')).scaled(990, 990)
        self.pianchang = QPixmap(os.path.abspath('images/describe/片长的直方图和密度图.jpg')).scaled(990, 990)
        
        self.guojia_QQ = QPixmap(os.path.abspath('images/analysis/国家对应标准正态Q-Q图.png')).scaled(990 , 990)
        self.pingfen_QQ = QPixmap(os.path.abspath('images/analysis/评分对应标准正态Q-Q图.png')).scaled(990 , 990)
        self.pingjiarenshu_QQ = QPixmap(os.path.abspath('images/analysis/评价人数对应标准正态Q-Q图.png')).scaled(990 , 990)
        self.shangyingshijian_QQ = QPixmap(os.path.abspath('images/analysis/上映对应标准正态Q-Q图.png')).scaled(990 , 990)
        self.yingpianleixing_QQ = QPixmap(os.path.abspath('images/analysis/影片类型对应标准正态Q-Q图.png')).scaled(990 , 990)
        self.yuyan_QQ = QPixmap(os.path.abspath('images/analysis/语言类型对应标准正态Q-Q图.png')).scaled(990 , 990)
        
        self.ui.Rank_mark.clicked.connect(self.describe_Rank_mark)
        self.ui.Rank_number.clicked.connect(self.describe_Rank_number)
        self.ui.Mark_number.clicked.connect(self.describe_Mark_number)
        self.ui.Mark_number_hist.clicked.connect(self.describe_Mark_number_hist)
        self.ui.types_hist.clicked.connect(self.describe_types_hist)
        self.ui.countries_hist.clicked.connect(self.describe_countries_hist)
        self.ui.languages_hist.clicked.connect(self.describe_languages_hist)
        self.ui.time_hist.clicked.connect(self.describe_time_hist)
        self.ui.long_hist.clicked.connect(self.describe_long_hist)
        
        self.ui.Mark_QQ.clicked.connect(self.analysis_Mark_QQ)
        self.ui.Number_QQ.clicked.connect(self.analysis_Number_QQ)
        self.ui.Countries_QQ.clicked.connect(self.analysis_Countries_QQ)
        self.ui.Types_QQ.clicked.connect(self.analysis_Types_QQ)
        self.ui.Languages_QQ.clicked.connect(self.analysis_Languages_QQ)
        self.ui.Time_QQ.clicked.connect(self.analysis_Time_QQ)
        
    def describe_Rank_mark(self):
        self.ui.fig_show.setPixmap(self.paiming_pingfen_scatter)
    def describe_Rank_number(self):
        self.ui.fig_show.setPixmap(self.paiming_pingjiarenshu_scatter)
    def describe_Mark_number(self):
        self.ui.fig_show.setPixmap(self.pingfen_pingjiarenshu_scatter)
    def describe_Mark_number_hist(self):
        self.ui.fig_show.setPixmap(self.pingfen_pingjiarenshu_hist)
    def describe_types_hist(self):
        self.ui.fig_show.setPixmap(self.yingpianzhonglei_hist)
    def describe_countries_hist(self):
        self.ui.fig_show.setPixmap(self.zhizuoguojia_hist)
    def describe_languages_hist(self):
        self.ui.fig_show.setPixmap(self.yingpianyuyan_hist)        
    def describe_time_hist(self):
        self.ui.fig_show.setPixmap(self.shangyingshijian_hist)
    def describe_long_hist(self):
        self.ui.fig_show.setPixmap(self.pianchang)
            
    def analysis_Mark_QQ(self):
        self.ui.fig_show.setPixmap(self.pingfen_QQ)
    def analysis_Number_QQ(self):
        self.ui.fig_show.setPixmap(self.pingjiarenshu_QQ)
    def analysis_Countries_QQ(self):
        self.ui.fig_show.setPixmap(self.guojia_QQ)
    def analysis_Types_QQ(self):
        self.ui.fig_show.setPixmap(self.yingpianleixing_QQ)
    def analysis_Languages_QQ(self):
        self.ui.fig_show.setPixmap(self.yuyan_QQ)
    def analysis_Time_QQ(self):
        self.ui.fig_show.setPixmap(self.shangyingshijian_QQ)
if __name__ == '__main__':
    app=QApplication(argv)
    window=MAIN_UI()
    exit(app.exec())