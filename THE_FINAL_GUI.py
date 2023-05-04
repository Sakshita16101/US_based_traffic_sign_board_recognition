# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/sakshita/PycharmProjects/GUI/FINAL_DESIGN.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import os
import cv2

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1061, 750)
        self.image_id = 0
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.mdiArea_2 = QtWidgets.QMdiArea(self.centralwidget)
        self.mdiArea_2.setGeometry(QtCore.QRect(530, 10, 511, 461))
        self.mdiArea_2.setObjectName("mdiArea_2")
        self.cutouts = QLabel(self.centralwidget)
        self.cutouts.setGeometry(QtCore.QRect(530, 10, 511, 461))
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(540, 10, 81, 31))
        self.label_2.setObjectName("label_2")
        self.mdiArea_3 = QtWidgets.QMdiArea(self.centralwidget)
        self.mdiArea_3.setGeometry(QtCore.QRect(10, 510, 201, 341))
        self.mdiArea_3.setObjectName("mdiArea_3")
        self.mdiArea_4 = QtWidgets.QMdiArea(self.centralwidget)
        self.mdiArea_4.setGeometry(QtCore.QRect(220, 510, 201, 361))
        self.mdiArea_4.setObjectName("mdiArea_4")
        self.mdiArea_5 = QtWidgets.QMdiArea(self.centralwidget)
        self.mdiArea_5.setGeometry(QtCore.QRect(430, 510, 191, 301))
        self.mdiArea_5.setObjectName("mdiArea_5")
        self.mdiArea_6 = QtWidgets.QMdiArea(self.centralwidget)
        self.mdiArea_6.setGeometry(QtCore.QRect(630, 510, 201, 351))
        self.mdiArea_6.setObjectName("mdiArea_6")
        self.mdiArea_7 = QtWidgets.QMdiArea(self.centralwidget)
        self.mdiArea_7.setGeometry(QtCore.QRect(840, 510, 201, 361))
        self.mdiArea_7.setObjectName("mdiArea_7")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(230, 511, 101, 31))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(440, 510, 111, 31))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(640, 510, 67, 31))
        self.label_5.setObjectName("label_5")
        self.mdiArea = QtWidgets.QMdiArea(self.centralwidget)
        self.mdiArea.setGeometry(QtCore.QRect(10, 9, 511, 461))
        self.mdiArea.setObjectName("mdiArea")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(16, 11, 101, 21))

        self.label.setObjectName("label")
        self.photo=QLabel(self.centralwidget)
        self.photo.setGeometry(QtCore.QRect(10, 10, 511, 461))
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(8, 444, 111, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.prev)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(140, 444, 111, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.next)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(410, 444, 111, 31))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.GetCutouts)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(20, 511, 101, 31))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(840, 490, 201, 71))
        self.label_7.setObjectName("label_7")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1061, 22))
        self.menubar.setObjectName("menubar")
        self.menuFILE = QtWidgets.QMenu(self.menubar)
        self.menuFILE.setObjectName("menuFILE")

        MainWindow.setMenuBar(self.menubar)
        #bar = self.menuBar()
        file = self.menubar.addMenu("input folder")
        file.addAction("Browse Input folder")

        file.triggered.connect(self.DirBrowse)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pipeline"))

        self.label_2.setText(_translate("MainWindow", "Cutouts"))

        self.label_3.setText(_translate("MainWindow", "TextBox++"))
        self.label_4.setText(_translate("MainWindow", "Segmentation"))
        self.label_5.setText(_translate("MainWindow", "OCR"))
        self.label.setText(_translate("MainWindow", "Image Viewer"))
        self.pushButton.setText(_translate("MainWindow", "PREVIOUS"))
        self.pushButton_2.setText(_translate("MainWindow", "NEXT"))


        self.pushButton_3.setText(_translate("MainWindow", "PROCESS"))
        self.label_6.setText(_translate("MainWindow", "Neuromotive"))
        self.label_7.setText(_translate("MainWindow", "OCR Correction & Prediction"))
        self.menuFILE.setTitle(_translate("MainWindow", "FILE"))
        #self.actionBrowse_Input_Folder.setText(_translate("MainWindow", "Browse Input Folder"))
    def DirBrowse(self):
        self.filePaths = QFileDialog.getExistingDirectory(None,
                                                       'Select  input folder:')
        #print(os.listdir(filePaths))
        #self.setWindowTitle("title")

        self.filename = os.listdir(self.filePaths)
        print(self.filename)
        self.num_images=len(self.filename)



        pixmap = QPixmap(os.path.join(self.filePaths, self.filename[self.image_id]))



        print(pixmap.height())
        if not pixmap.isNull():
            #self.label= QtWidgets.QLabel(self)






            self.photo.setAlignment(Qt.AlignCenter)
            pixmap = (QPixmap(pixmap))
            #print(sub.width())
            pixmap = pixmap.scaled(400, 400)
            # self.showPicture(pixmap)
            self.photo.setPixmap(pixmap)


    def prev(self):
        print("previous")
        print(self.image_id)
        self.image_id=self.image_id-1
        if self.image_id<0:
            print("no previous")
        else:
            pixmap = QPixmap(os.path.join(self.filePaths, self.filename[self.image_id]))


            print(pixmap.height())
            if not pixmap.isNull():
                # self.label= QtWidgets.QLabel(self)

                self.photo.setAlignment(Qt.AlignCenter)
                pixmap = (QPixmap(pixmap))
                # print(sub.width())
                pixmap = pixmap.scaled(400, 400)
                # self.showPicture(pixmap)
                self.photo.setPixmap(pixmap)


    def next(self):
        print("next")
        print(self.image_id)
        self.image_id = self.image_id + 1

        if self.image_id > (self.num_images-1):
            #print("no next")
           pass

        else:

            pixmap = QPixmap(os.path.join(self.filePaths, self.filename[self.image_id]))


            print(pixmap.height())
            if not pixmap.isNull():
                # self.label= QtWidgets.QLabel(self)

                self.photo.setAlignment(Qt.AlignCenter)
                pixmap = (QPixmap(pixmap))
                # print(sub.width())
                pixmap = pixmap.scaled(400, 400)
                # self.showPicture(pixmap)
                self.photo.setPixmap(pixmap)


    def GetCutouts(self):

        path= "/home/himanshi/data/IIT MANDI PROJECT/OCR_NLP/image.jpg"
        #image=cv2.imread("/home/himanshi/data/IIT MANDI PROJECT/OCR_NLP/image.png")
        pixmap1 = QPixmap(path)

        if not pixmap1.isNull():
            self.cutouts.setAlignment(Qt.AlignCenter)
            pixmap1 = (QPixmap(pixmap1))
            # print(sub.width())
            pixmap1 = pixmap1.scaled(400, 400)
            # self.showPicture(pixmap)
            self.cutouts.setPixmap(pixmap1)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

