from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import numpy as np
import cv2
import random
from scipy import ndimage
from scipy import signal
import math


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1260, 911)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.imgcLabel = QtWidgets.QLabel(self.centralwidget)
        self.imgcLabel.setGeometry(QtCore.QRect(410, 180, 321, 221))
        self.imgcLabel.setText("")
        self.imgcLabel.setObjectName("imgcLabel")
        self.imgcLabel.setScaledContents(True)
        self.hisBtn = QtWidgets.QPushButton(self.centralwidget)
        self.hisBtn.setGeometry(QtCore.QRect(40, 100, 271, 51))
        self.hisBtn.setObjectName("hisBtn")
        self.filBtn = QtWidgets.QPushButton(self.centralwidget)
        self.filBtn.setGeometry(QtCore.QRect(40, 200, 271, 51))
        self.filBtn.setObjectName("filBtn")
        self.res2Btn = QtWidgets.QPushButton(self.centralwidget)
        self.res2Btn.setGeometry(QtCore.QRect(40, 600, 271, 51))
        self.res2Btn.setObjectName("res2Btn")
        self.EdgBtn = QtWidgets.QPushButton(self.centralwidget)
        self.EdgBtn.setGeometry(QtCore.QRect(40, 700, 271, 51))
        self.EdgBtn.setObjectName("EdgBtn")
        self.SsBtn = QtWidgets.QPushButton(self.centralwidget)
        self.SsBtn.setGeometry(QtCore.QRect(40, 400, 271, 51))
        self.SsBtn.setObjectName("SsBtn")
        self.res1Btn = QtWidgets.QPushButton(self.centralwidget)
        self.res1Btn.setGeometry(QtCore.QRect(40, 500, 271, 51))
        self.res1Btn.setObjectName("res1Btn")
        self.moiBtn = QtWidgets.QPushButton(self.centralwidget)
        self.moiBtn.setGeometry(QtCore.QRect(40, 300, 271, 51))
        self.moiBtn.setObjectName("moiBtn")
        self.imgLabel1 = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel1.setGeometry(QtCore.QRect(840, 180, 321, 221))
        self.imgLabel1.setText("")
        self.imgLabel1.setObjectName("imgLabel1")
        self.imgLabel1.setScaledContents(True)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(30, 30, 121, 31))
        self.pushButton.setObjectName("pushButton")
        self.imgLabel2 = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel2.setGeometry(QtCore.QRect(410, 520, 321, 221))
        self.imgLabel2.setText("")
        self.imgLabel2.setObjectName("imgLabel2")
        self.imgLabel2.setScaledContents(True)
        self.imgLabel3 = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel3.setGeometry(QtCore.QRect(840, 520, 321, 221))
        self.imgLabel3.setText("")
        self.imgLabel3.setObjectName("imgLabel3")
        self.imgLabel3.setScaledContents(True)
        self.imgViewLabel = QtWidgets.QLabel(self.centralwidget)
        self.imgViewLabel.setGeometry(MainWindow.geometry())
        self.imgViewLabel.setObjectName("imgViewLabel")
        self.imgViewLabel.hide()
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(70, 160, 171, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.hide()
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(70, 250, 171, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.hide()
        self.txtcLabel1 = QtWidgets.QLabel(self.centralwidget)
        self.txtcLabel1.setGeometry(QtCore.QRect(470, 430, 261, 41))
        self.txtcLabel1.setText("")
        self.txtcLabel1.setObjectName("txtcLabel1")
        self.txtcLabel1.setScaledContents(True)
        self.txtcLabel1.setAlignment(QtCore.Qt.AlignVCenter)
        self.txtLabel1 = QtWidgets.QLabel(self.centralwidget)
        self.txtLabel1.setGeometry(QtCore.QRect(940, 430, 261, 41))
        self.txtLabel1.setText("")
        self.txtLabel1.setObjectName("txtLabel1")
        self.txtLabel1.setScaledContents(True)
        self.txtLabel1.setAlignment(QtCore.Qt.AlignVCenter)
        self.txtLabel2 = QtWidgets.QLabel(self.centralwidget)
        self.txtLabel2.setGeometry(QtCore.QRect(470, 750, 261, 41))
        self.txtLabel2.setText("")
        self.txtLabel2.setObjectName("txtLabel2")
        self.txtLabel2.setScaledContents(True)
        self.txtLabel2.setAlignment(QtCore.Qt.AlignVCenter)
        self.txtLabel3 = QtWidgets.QLabel(self.centralwidget)
        self.txtLabel3.setGeometry(QtCore.QRect(940, 750, 261, 41))
        self.txtLabel3.setText("")
        self.txtLabel3.setObjectName("txtLabel3")
        self.txtLabel3.setScaledContents(True)
        self.txtLabel3.setAlignment(QtCore.Qt.AlignVCenter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1260, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionClose = QtWidgets.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionClose)
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.actionZoomIn = QtWidgets.QAction(MainWindow)
        self.actionZoomIn.setObjectName("actionZoomIn")
        self.actionZoomOut = QtWidgets.QAction(MainWindow)
        self.actionZoomOut.setObjectName("actionZoomOut")
        self.actionNormalSize = QtWidgets.QAction(MainWindow)
        self.actionNormalSize.setObjectName("actionNormalSize")
        self.actionFitWindow = QtWidgets.QAction(MainWindow)
        self.actionFitWindow.setObjectName("actionFitWindow")
        self.menubar.addAction(self.menuView.menuAction())
        self.menuView.addAction(self.actionZoomIn)
        self.menuView.addAction(self.actionZoomOut)
        self.menuView.addAction(self.actionNormalSize)
        self.menuView.addAction(self.actionFitWindow)
        self.scale_factor = 1.0
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(lambda: self.on_click_choose_image())
        self.hisBtn.clicked.connect(lambda: self.on_click_his_equalize())
        self.filBtn.clicked.connect(lambda: self.on_click_filter())
        self.pushButton_2.clicked.connect(lambda: self.on_click_DF())
        self.pushButton_3.clicked.connect(lambda: self.on_click_MD())
        self.SsBtn.clicked.connect(lambda: self.on_click_blur_sharpen())
        self.moiBtn.clicked.connect(lambda: self.on_click_moire_pattern())
        self.res1Btn.clicked.connect(lambda: self.on_click_image_restoration1())
        self.res2Btn.clicked.connect(lambda: self.on_click_image_restoration2())
        self.EdgBtn.clicked.connect(lambda: self.on_click_simple_edge_detection())
        self.actionOpen.triggered.connect(lambda: self.action_choose_image())
        self.actionOpen.setShortcut("Ctrl+O")
        self.actionClose.triggered.connect(lambda: self.close())
        self.actionClose.setShortcut("Ctrl+C")
        self.actionZoomIn.triggered.connect(lambda: self.zoomIn())
        self.actionZoomIn.setShortcut("Ctrl++")
        self.actionZoomOut.triggered.connect(lambda: self.zoomOut())
        self.actionZoomOut.setShortcut("Ctrl+-")
        self.actionNormalSize.triggered.connect(lambda: self.normalSize())
        self.actionNormalSize.setShortcut("Ctrl+N")
        self.actionFitWindow.triggered.connect(lambda: self.fitToWindow())
        self.actionFitWindow.setShortcut("Ctrl+F")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.hisBtn.setText(_translate("MainWindow", "Histogram Equalize"))
        self.filBtn.setText(_translate("MainWindow", "Filter Threshold"))
        self.res2Btn.setText(_translate("MainWindow", "Image Restoration 2"))
        self.EdgBtn.setText(_translate("MainWindow", "Edge Detection"))
        self.SsBtn.setText(_translate("MainWindow", "Smoothing And Sharpening"))
        self.res1Btn.setText(_translate("MainWindow", "Image Restoration 1"))
        self.moiBtn.setText(_translate("MainWindow", "Moire Pattern"))
        self.pushButton.setText(_translate("MainWindow", "Choose Image"))
        self.pushButton_2.setText(_translate("MainWindow", "Directional Filtering"))
        self.pushButton_3.setText(_translate("MainWindow", "Median Threshold"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionClose.setText(_translate("MainWindow", "Close"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.actionZoomIn.setText(_translate("MainWindow", "Zoom In"))
        self.actionZoomOut.setText(_translate("MainWindow", "Zoom Out"))
        self.actionNormalSize.setText(_translate("MainWindow", "Normal Size"))
        self.actionFitWindow.setText(_translate("MainWindow", "Fit to Window"))

    def action_choose_image(self):
        Tk().withdraw()
        filename = askopenfilename()
        file = open("outputs/choose_view_image.txt", "w")
        file.write(filename)
        file.close()
        self.imgViewLabel.setPixmap(QtGui.QPixmap(filename))
        self.imgViewLabel.show()
        self.imgcLabel.hide()
        self.imgLabel1.hide()
        self.imgLabel2.hide()
        self.imgLabel3.hide()
        self.pushButton.hide()
        self.hisBtn.hide()
        self.SsBtn.hide()
        self.moiBtn.hide()
        self.filBtn.hide()
        self.res1Btn.hide()
        self.res2Btn.hide()
        self.EdgBtn.hide()
        self.txtcLabel1.hide()
        self.txtLabel1.hide()
        self.txtLabel2.hide()
        self.txtLabel3.hide()

    def close(self):
        if self.imgViewLabel.isHidden() == False:
            self.imgViewLabel.hide()
            self.imgcLabel.show()
            self.imgLabel1.show()
            self.imgLabel2.show()
            self.imgLabel3.show()
            self.pushButton.show()
            self.hisBtn.show()
            self.SsBtn.show()
            self.moiBtn.show()
            self.filBtn.show()
            self.res1Btn.show()
            self.res2Btn.show()
            self.EdgBtn.show()
            self.txtcLabel1.show()
            self.txtLabel1.show()
            self.txtLabel2.show()
            self.txtLabel3.show()
        else:
            MainWindow.close()
    def zoomIn(self):
        self.scale_factor *= 1.25
        filename = open("outputs/choose_view_image.txt", "r")
        img = cv2.imread(filename.read())
        filename.close()

        col, row, _ = np.shape(img)
        col = int(col * self.scale_factor)
        row = int(row * self.scale_factor)
        img = cv2.resize(img, (row, col))
        cv2.imwrite("outputs/zoom.png", img)
        self.imgViewLabel.setPixmap(QtGui.QPixmap("outputs/zoom.png"))
    def zoomOut(self):
        self.scale_factor *= 0.8

        filename = open("outputs/choose_view_image.txt", "r")
        img = cv2.imread(filename.read())
        filename.close()
        col, row, _ = np.shape(img)
        col = int(col * self.scale_factor)
        row = int(row * self.scale_factor)
        img = cv2.resize(img, (row, col))
        cv2.imwrite("outputs/zoom.png", img)
        self.imgViewLabel.setPixmap(QtGui.QPixmap("outputs/zoom.png"))
    def normalSize(self):
        self.imgViewLabel.setScaledContents(False)
        filename = open("outputs/choose_view_image.txt", "r")
        self.imgViewLabel.setPixmap(QtGui.QPixmap(filename.read()))
        filename.close()


    def fitToWindow(self):
        self.imgViewLabel.setScaledContents(True)

    def on_click_choose_image(self):
        Tk().withdraw()
        filename = askopenfilename()
        file = open("outputs/choose_image.txt", "w")
        file.write(filename)
        file.close()
        self.imgcLabel.setPixmap(QtGui.QPixmap(filename))
        self.imgLabel1.hide()
        self.imgLabel2.hide()
        self.imgLabel3.hide()
        self.txtLabel1.hide()
        self.txtLabel2.hide()
        self.txtLabel3.hide()
    def show_labels_image(self):
        file = open("outputs/choose_image.txt", "r")
        image = cv2.imread(file.read())
        cv2.imwrite("outputs/choose_image.png", image)
        # image = cv2.imread(file)
        self.imgcLabel.setPixmap(QtGui.QPixmap("outputs/choose_image.png"))
        file.close()
        if self.imgLabel1.isHidden()==True:
            self.imgLabel1.show()
        if self.imgLabel2.isHidden()==True:
            self.imgLabel2.show()
        if self.imgLabel3.isHidden()==True:
            self.imgLabel3.show()
        if self.txtLabel1.isHidden()==True:
            self.txtLabel1.show()
        if self.txtLabel2.isHidden() == True:
            self.txtLabel2.show()
        if self.txtLabel3.isHidden() == True:
            self.txtLabel3.show()


    def on_click_his_equalize(self):
        # filename = histogram_equalize(self.imgcLabel.pixmap())
        # self.imgLabel1.setPixmap(QtGui.QPixmap(histogram_equalize(filename)))
        # filename = askopenfilename()
        #self.imgcLabel.setPixmap(QtGui.QPixmap(filename))
        self.histogram_equalize()
        self.show_labels_image()
        self.imgLabel1.setPixmap(QtGui.QPixmap("outputs/Histogram_Equalize.png"))
        self.imgLabel2.setPixmap(QtGui.QPixmap("outputs/input_histogram.png"))
        self.imgLabel3.setPixmap(QtGui.QPixmap("outputs/output_histogram.png"))
        self.txtcLabel1.setText("Original Image")
        self.txtLabel1.setText("Histogram Equalize Image")
        self.txtLabel2.setText("Histogram Original Image")
        self.txtLabel3.setText("Histogram Equalize")
    def on_click_filter(self):
        self.pushButton_2.show()
        self.pushButton_3.show()
        self.hisBtn.hide()
        self.filBtn.hide()
        self.moiBtn.hide()
        self.SsBtn.hide()
        self.res1Btn.hide()
        self.res2Btn.hide()
        self.EdgBtn.hide()
    def on_click_DF(self):
        self.directional_filter()
        self.show_labels_image()
        self.imgLabel1.setPixmap(QtGui.QPixmap("outputs/randn_df.png"))
        self.imgLabel2.setPixmap(QtGui.QPixmap("outputs/img1_df.png"))
        self.imgLabel3.setPixmap(QtGui.QPixmap("outputs/img2_df.png"))
        self.txtcLabel1.setText("Original Image")
        self.txtLabel1.setText("Random noise")
        self.txtLabel2.setText("Image with Filter1")
        self.txtLabel3.setText("Image with Filter2")
        self.pushButton_2.hide()
        self.pushButton_3.hide()
        self.hisBtn.show()
        self.filBtn.show()
        self.moiBtn.show()
        self.SsBtn.show()
        self.res1Btn.show()
        self.res2Btn.show()
        self.EdgBtn.show()
    def on_click_MD(self):
        self.median_threshold()
        self.show_labels_image()
        self.imgLabel1.setPixmap(QtGui.QPixmap("outputs/noise_mt.png"))
        self.imgLabel2.setPixmap(QtGui.QPixmap("outputs/median_filter_threshold.png"))
        self.imgLabel3.hide()
        self.txtcLabel1.setText("Original Image")
        self.txtLabel1.setText("Noise Image")
        self.txtLabel2.setText("Median Filter Threshold")
        self.txtLabel3.hide()
        self.pushButton_2.hide()
        self.pushButton_3.hide()
        self.hisBtn.show()
        self.filBtn.show()
        self.moiBtn.show()
        self.SsBtn.show()
        self.res1Btn.show()
        self.res2Btn.show()
        self.EdgBtn.show()
    def on_click_blur_sharpen(self):
        self.smooth_and_sharpen()
        self.show_labels_image()
        self.imgLabel1.setPixmap(QtGui.QPixmap("outputs/blur.png"))
        self.imgLabel2.setPixmap(QtGui.QPixmap("outputs/gaussian_blur.png"))
        self.imgLabel3.setPixmap(QtGui.QPixmap("outputs/sharpen_image.png"))
        self.txtcLabel1.setText("Original Image")
        self.txtLabel1.setText("Blur Image")
        self.txtLabel2.setText("Gaussian Blur Image")
        self.txtLabel3.setText("Sharpen Image")
    def on_click_moire_pattern(self):
        self.Moire_pattern()
        self.show_labels_image()
        self.imgLabel1.setPixmap(QtGui.QPixmap("outputs/dft2d.png"))
        self.imgLabel2.setPixmap(QtGui.QPixmap("outputs/image_media.png"))
        self.imgLabel3.setPixmap(QtGui.QPixmap("outputs/image_butterworth.png"))
        self.txtcLabel1.setText("Original Image")
        self.txtLabel1.setText("DFT2D")
        self.txtLabel2.setText("Image Media")
        self.txtLabel3.setText("Image Butterworth")
    def on_click_image_restoration1(self):
        self.image_restoration1()
        self.show_labels_image()
        self.imgcLabel.setPixmap(QtGui.QPixmap("outputs/noise_r1.png"))
        self.imgLabel1.setPixmap(QtGui.QPixmap("outputs/blurred_r1.png"))
        self.imgLabel2.setPixmap(QtGui.QPixmap("outputs/median_r1.png"))
        self.imgLabel3.setPixmap(QtGui.QPixmap("outputs/wiener_r1.png"))
        self.txtcLabel1.setText("Noise Image")
        self.txtLabel1.setText("Blurred restoration")
        self.txtLabel2.setText("Median restoration")
        self.txtLabel3.setText("Wiener restoration")
    def on_click_image_restoration2(self):
        self.image_restoration2()
        self.show_labels_image()
        self.imgLabel1.setPixmap(QtGui.QPixmap("outputs/restoration2.png"))
        self.imgLabel2.hide()
        self.imgLabel3.hide()
        self.txtcLabel1.setText("Original Image")
        self.txtLabel1.setText("Restoration Image")
        self.txtLabel2.hide()
        self.txtLabel3.hide()
    def on_click_simple_edge_detection(self):
        self.simple_edge_detection()
        self.show_labels_image()
        self.imgLabel1.setPixmap(QtGui.QPixmap("outputs/canny_sed.png"))
        self.imgLabel2.setPixmap(QtGui.QPixmap("outputs/hough_transform_sed.png"))
        self.imgLabel3.hide()
        self.txtcLabel1.setText("Original Image")
        self.txtLabel1.setText("Canny detection")
        self.txtLabel2.setText("Hough Transform")
        self.txtLabel3.hide()
    def histogram_equalize(self):
        # loading an image
        file = open("outputs/choose_image.txt", "r")
        rfile = file.read()
        file.close()
        image = Image.open(rfile)
        row, col, ch = np.shape(image)
        # changing image to bytes so as to get pixel intesities
        image_to_float = image.tobytes()
        pixel_intensities = [image_to_float[i] for i in range(len(image_to_float))]

        img = np.array(pixel_intensities).reshape((row, col, ch))

        hist, bins = np.histogram(img.flatten(), 256, [0, 256])

        # cumulative distribution function
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = ((cdf_m - cdf_m.min()) * 255) / (cdf_m.max() - cdf_m.min())
        cdf_scaled = np.ma.filled(cdf_m, 0).astype('uint8')
        img2 = cdf_scaled[img]
        cv2.imwrite('outputs/Histogram_Equalize.png', img2)

        plt.plot()
        plt.title("Given Image cumulative distribution function")
        plt.plot(cdf_normalized, color='b')
        plt.hist(img.flatten(), 256, [0, 256], color='r')
        plt.savefig('outputs/input_histogram.png')

        plt.plot()
        plt.title("After Global Histogram Equalization")
        plt.plot(cdf_normalized, color='b')
        plt.hist(img2.flatten(), 256, [0, 256], color='r')
        plt.savefig('outputs/output_histogram.png')

    # def histequal(self,img):
    #     equ=cv2.equaflizeHist(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    #     equ=cv2.cvtColor((equ,cv2.COLOR_BGR2GRAY))
    #     return equ

    def directional_filter(self):
        file = open("outputs/choose_image.txt", "r")
        image = cv2.imread(file.read())
        file.close()
        a = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]) / 3
        b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) / 3
        c = np.rot90(a)
        d = np.rot90(b)

        dst = np.empty_like(image)
        noise = cv2.randn(dst, (0, 0, 0), (30, 30, 30))
        noise_image = cv2.addWeighted(image, 0.5, noise, 0.5, 30)

        img1 = cv2.filter2D(noise_image, -1, a)
        img2 = cv2.filter2D(noise_image, -1, b)
        img3 = cv2.filter2D(noise_image, -1, c)
        img4 = cv2.filter2D(noise_image, -1, d)

        # plt.plot(), plt.imshow(noise_image), plt.title('Noise Image'), plt.savefig("outputs/randn_df.png")
        # plt.plot(), plt.imshow(img1), plt.title('Image with Filter 1'), plt.savefig("outputs/img1_df.png")
        # plt.plot(), plt.imshow(img2), plt.title('Image with Filter 2'), plt.savefig("outputs/img2_df.png")
        # plt.subplot(3, 2, 5), plt.imshow(img3), plt.title('Image with Filter 3')
        # plt.subplot(3, 2, 6), plt.imshow(img4), plt.title('Image with Filter 4')

        cv2.imwrite("outputs/randn_df.png", noise_image)
        cv2.imwrite("outputs/img1_df.png", img1)
        cv2.imwrite("outputs/img2_df.png", img2)
    def median_threshold(self):
        def noise(image, probability):
            out = np.zeros(image.shape, np.uint8)
            thres = 1 - probability

            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    rnd = random.random()
                    if rnd < probability:
                        out[i][j] = 0
                    elif rnd > probability:
                        out[i][j] = 255
                    else:
                        out[i][j] = image[i][j]
            return out

        def med_threshold(image, n, thresh):
            io, jo, chanel = image.shape
            filter = cv2.medianBlur(image, ksize=n)
            for i in range(io):
                for j in range(jo):
                    if np.abs(image[i][j] - filter[i][j]).all() <= thresh:
                        filter[i][j] = image[i][j]
            return filter
        file = open("outputs/choose_image.txt", "r")
        image = cv2.imread(file.read())
        file.close()
        noise_image = noise(image, 0.05)
        new = med_threshold(image, 3, 255)
        # plt.plot(), plt.imshow(image), plt.title('Original Image')
        # plt.plot(), plt.imshow(noise_image), plt.title('Noise'), plt.savefig("outputs/noise_mt.png")
        # plt.plot(), plt.imshow(new), plt.title('Median Filter Threshold'), plt.savefig("outputs/median_filter_threshold.png")
        cv2.imwrite("outputs/noise_mt.png", noise_image)
        cv2.imwrite("outputs/median_filter_threshold.png", new)

    def smooth_and_sharpen(self):
        file = open("outputs/choose_image.txt", "r")
        image = cv2.imread(file.read())
        file.close()

        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur = cv2.blur(img, (13, 13))
        gaussian_blur = cv2.GaussianBlur(img, (37, 37), 0)

        # plt.plot(), plt.imshow(blur), plt.title("Blur Image"), plt.savefig("outputs/blur.png")
        cv2.imwrite("outputs/blur.png", blur)
        # plt.plot(), plt.title(gaussian_blur), plt.title("Gaussian Blur Image"), plt.savefig("outputs/gaussian_blur.png")
        cv2.imwrite("outputs/gaussian_blur.png", gaussian_blur)

        def corr(img, mask):
            row, col = img.shape
            m, n = mask.shape
            new = np.zeros((row + m - 1, col + n - 1))
            n = n // 2
            m = m // 2
            filtering_img = np.zeros(img.shape)
            new[m:new.shape[0] - m, n:new.shape[1] - n] = img
            for i in range(m, new.shape[0] - m):
                for j in range(n, new.shape[1] - n):
                    temp = new[i - m:i + m + 1, j - m: j + m + 1]
                    # print(temp)
                    result = temp * mask
                    filtering_img[i - m, j - n] - result.sum()
            return filtering_img

        def gaussian(m, n, sigma):
            gaussian = np.zeros((m, n))
            m = m // 2
            n = n // 2
            for x in range(-m, m + 1):
                for y in range(-n, n + 1):
                    x1 = sigma * (2 * np.pi) ** 2
                    x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
                    gaussian[x + m, y + n] = (1 / x1) * x2
            return gaussian

        # file = open("outputs/choose_image.txt", 'r')
        # img = skimage.io.imread(file.read())
        # file.close()
        # img = skimage.color.rgb2gray(img)
        file = open("outputs/choose_image.txt", "r")
        image = cv2.imread(file.read())
        file.close()
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        g = gaussian(5, 5, 2)
        Ig1 = corr(img, g)
        g = gaussian(5, 5, 5)
        Ig2 = corr(img, g)
        edg = Ig1 - Ig2
        alpha = 30
        sharped = img + edg * alpha
        # plt.plot(), plt.imshow(sharped), plt.title('Sharpen Image'), plt.savefig("outputs/sharpen.png"), plt.xticks(), plt.yticks()
        cv2.imwrite("outputs/sharpen_image.png", sharped)
    def Moire_pattern(self):
        file = open("outputs/choose_image.txt", "r")
        image = cv2.imread(file.read(), 0)
        file.close()
        def butterworthlp(sx, sy, D0, n):
            hr = np.arange(-sx / 2, sx / 2, 1)
            hc = np.arange(-sy / 2, sy / 2, 1)
            x, y = np.meshgrid(hc, hr)
            mg = np.sqrt(x ** 2 + y ** 2)
            H = 1 / (1 + (mg / D0) ** (2 * n))
            return H

        def DFT2D(image):
            f = np.fft.fft2(image)
            fshipt = np.fft.fftshift(f)
            mag = 20 * np.log(np.abs(fshipt))
            mag = np.asarray(mag, dtype=np.uint8)
            return mag

        def images(image):
            image = cv2.resize(image, (500, 500))
            dft2d = DFT2D(image)
            img_media = cv2.medianBlur(image, 7)
            [sx, sy] = np.shape(image)
            H = butterworthlp(sx, sy, 15, 0.5)
            G = np.fft.fftshift(np.fft.fft2(image))
            Ip = H * G
            image_butterworth = np.abs(np.fft.ifft2(np.fft.ifftshift(Ip))).astype('uint8')
            return [dft2d, img_media, image_butterworth]
        dft2d, img_media, image_butterworth = images(image)
        # plt.figure()
        # plt.plot(), plt.imshow(dft2d, 'gray'), plt.title("dft2d"), plt.savefig("outputs/dft2d.png")
        # plt.plot(), plt.imshow(img_media, 'gray'), plt.title("img_media"), plt.savefig("outputs/img_media.png")
        # plt.plot(), plt.imshow(image_butterworth, 'gray'), plt.title("image_butterworth"), plt.savefig("outputs/image_butterworth.png")
        # cv2.cvtColor(dft2d, cv2.COLOR_BGR2GRAY)
        # cv2.cvtColor(img_media, cv2.COLOR_BGR2GRAY)
        # cv2.cvtColor(image_butterworth, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("outputs/dft2d.png", dft2d)
        cv2.imwrite("outputs/image_media.png", img_media)
        cv2.imwrite("outputs/image_butterworth.png", image_butterworth)
    def image_restoration1(self):
        file = open("outputs/choose_image.txt", "r")
        img = cv2.imread(file.read(), 0)
        file.close()
        noise_r1 = np.copy(img).astype(np.float64)
        noise_r1 += img.std() * 0.5 * np.random.standard_normal(img.shape)
        # noise_r1 = cv2.cvtColor(noise_r1, cv2.GRAY2BGR)
        blurred_r1 = ndimage.gaussian_filter(noise_r1, sigma=3)
        median_r1 = ndimage.median_filter(noise_r1, size=5)
        wiener_r1 = signal.wiener(noise_r1, (5, 5))
        # plt.plot(), plt.imshow(noise_r1), plt.title("Noise"), plt.savefig("outputs/noise_r1.png")
        # plt.plot(), plt.imshow(blurred_r1), plt.title("Blurred"), plt.savefig("outputs/blurred_r1.png")
        # plt.plot(), plt.imshow(median_r1), plt.title("Median"), plt.savefig("outputs/median_r1.png")
        # plt.plot(), plt.imshow(wiener_r1), plt.title("Wiener"), plt.savefig("outputs/wiener_r1.png")
        cv2.imwrite("outputs/noise_r1.png", noise_r1)
        cv2.imwrite("outputs/blurred_r1.png", blurred_r1)
        cv2.imwrite("outputs/median_r1.png", median_r1)
        cv2.imwrite("outputs/wiener_r1.png", wiener_r1)
    def image_restoration2(self):
        file = open("outputs/choose_image.txt", "r")
        img = cv2.imread(file.read())
        row, col, ch = np.shape(img)
        img = cv2.resize(img, (row, row))
        file.close()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        xSize, ySize = gray_img.shape

        def applyFilter(img, func):
            image = np.copy(img)

            # Construct image from blurring function
            for u in range(0, ySize):
                for v in range(0, xSize):
                    image[u, v] = func(u, v)

            # Performe the actual blurring of the image. Not working as expected
            return image * img

        def blurr(y, x):
            a = 0.05
            b = 0.05
            T = 1
            C = math.pi * (a * y + b * x)

            if (C == 0):
                return 1

            return (T / C) * math.sin(C) * math.e ** (-1j * C)

        def toReal(img):
            realImg = np.zeros(img.shape)

            for i in range(0, img.shape[0]):
                for j in range(0, img.shape[1]):
                    realImg[i, j] = np.absolute(img[i, j])

            return realImg

        def normalize(image):
            img = image.copy()
            img = toReal(img)
            img -= img.min()
            img *= 255.0 / img.max()
            return img.astype(np.uint8)

        f = np.fft.fft2(gray_img.astype(np.int32))
        fft_img = np.fft.fftshift(f)

        # Apply the blurring filter
        filtered_fft = applyFilter(fft_img, blurr)

        f_fft_img = np.fft.ifftshift(filtered_fft)
        filtered_img = np.fft.ifft2(f_fft_img)

        filtered_img = normalize(filtered_img)
        cv2.imwrite("outputs/restoration2.png", filtered_img)
    def simple_edge_detection(self):
        file = open("outputs/choose_image.txt", "r")
        img = cv2.imread(file.read(), cv2.IMREAD_GRAYSCALE)
        file.close()
        def auto_canny(image, sigma=0.33):
            v = np.median(image)
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edged = cv2.Canny(image, lower, upper)
            return edged

        def clip(idx):
            return int(max(idx, 0))

        def hough_peaks(H, numpeaks=1, threshold=100, nhood_size=5):
            peaks = np.zeros((numpeaks, 2), dtype=np.uint64)
            temp_H = np.copy(H)
            for i in range(numpeaks):
                _, max_val, _, max_loc = cv2.minMaxLoc(temp_H)  # find maximum peaks
                if max_val > threshold:
                    peaks[i] = max_loc
                    (c, r) = max_loc
                    t = nhood_size // 2.0
                    temp_H[clip(r - t):int(r + t + 1), clip(c - t):int(c + t + 1)] = 0
                else:
                    peaks = peaks[:i]
            return peaks[:, ::-1]

        def hough_lines_acc(img, rho_res=1, thetas=np.arange(-90, 90, 1)):
            rho_max = int(np.linalg.norm(img.shape - np.array([1, 1]), 2))
            rhos = np.arange(-rho_max, rho_max, rho_res)
            thetas -= min(min(thetas), 0)
            accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
            yis, xis = np.nonzero(img)  # use only edge points
            for idx in range(len(xis)):
                x = xis[idx]
                y = yis[idx]
                temp_rhos = x * np.cos(np.deg2rad(thetas)) + y * np.sin(np.deg2rad(thetas))
                temp_rhos = temp_rhos / rho_res + rho_max
                m, n = accumulator.shape
                valid_idxs = np.nonzero((temp_rhos < m) & (thetas < n))
                temp_rhos = temp_rhos[valid_idxs]
                temp_thetas = thetas[valid_idxs]
                c = np.stack([temp_rhos, temp_thetas], 1)
                cc = np.ascontiguousarray(c).view(np.dtype((np.void, c.dtype.itemsize * c.shape[1])))
                _, idxs, counts = np.unique(cc, return_index=True, return_counts=True)
                uc = c[idxs].astype(np.uint)
                accumulator[uc[:, 0], uc[:, 1]] += counts.astype(np.uint)
            accumulator = cv2.normalize(accumulator, accumulator, 0, 255,
                                        cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            return accumulator, thetas, rhos

        def hough_lines_draw(img, peaks, rhos, thetas):
            for peak in peaks:
                rho = rhos[peak[0]]
                theta = thetas[peak[1]] * np.pi / 180.0
                a = np.cos(theta);
                b = np.sin(theta)
                pt0 = rho * np.array([a, b])
                pt1 = tuple((pt0 + 1000 * np.array([-b, a])).astype(int))
                pt2 = tuple((pt0 - 1000 * np.array([-b, a])).astype(int))
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            return img
        edge_image = auto_canny(img)
        H, thetas, rhos = hough_lines_acc(edge_image)
        peaks = hough_peaks(H, numpeaks=10, threshold=100, nhood_size=50)
        H_peaks = H.copy()
        for peak in peaks:
            cv2.circle(H_peaks, tuple(peak[::-1]), 5, (255, 255, 255), -1)
            color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            draw = hough_lines_draw(color_img, peaks, rhos, thetas)

        cv2.imwrite("outputs/canny_sed.png", edge_image)
        cv2.imwrite("outputs/hough_transform_sed.png", draw)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())