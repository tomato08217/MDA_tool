import sys
from PyQt5 import QtCore, QtGui, uic,QtWidgets
import pandas as  pd
from DataContainer import DataContainer
from sklearn.svm import l1_min_c
import numpy as np
from time import time
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from DataSeparate import DataSeparate
from copy import deepcopy
import os
import math
from PyQt5.QtWidgets import QTableWidget,QHeaderView,QTableWidgetItem,QAbstractItemView,QCheckBox,QHBoxLayout,QWidget
from PyQt5.QtCore import Qt
import ctypes
from radiomics import featureextractor
import csv
from lifelines import CoxPHFitter
from pickle import load, dump
qtCreatorFile = sys.path[0] + '\\GUI\\FeatureCalculation.ui' # Enter file here.


Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.folder_Image = ""
        self.folder_Segmentation = ""
        self.files_Image = []
        self.files_Segmenatation = []
        self.pushButtonLoad.clicked.connect(self.Load)
    
        self.tableWidget.setColumnCount(8)
        # self.tableWidget.horizontalHeader().setSectionResizeMode(2,QHeaderView.Stretch)#设置第3列宽度自动调整，充满屏幕
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setStretchLastSection(True) ##设置最后一列拉伸至最大
        self.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection) #设置只可以单选，可以使用ExtendedSelection进行多选
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows) #设置 不可选择单个单元格，只可选择一行。
        #self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers) #设置表格不可更改
    
        
        

        self.pushButtonSelect_All.setEnabled(False)
        self.pushButtonSelect_None.setEnabled(False)
        self.pushButton_CalculateFeature.setEnabled(False)
        self.progressBar.setEnabled(False)

        self.pushButtonSelect_All.clicked.connect(self.SelectAll)
        self.pushButtonSelect_None.clicked.connect(self.SelectNone)     
        self.pushButton_CalculateFeature.clicked.connect(self.Run)
        self.progressBar.reset()

        self.count = 0

        
    def Load(self):
        try:
            self.folder = QtWidgets.QFileDialog.getExistingDirectory(self,"Select Image Folder")
            self.folderName.setText(self.folder)
            
            self.folder_Image = self.folder+'/Image'
            self.folder_Segmentation = self.folder+'/Segmentation'

            temp_Imagefiles = os.listdir(self.folder_Image)
            temp_Segmentationfiles = os.listdir(self.folder_Segmentation)
            
            for file in temp_Imagefiles:
                filename, extension  = os.path.splitext(file)
                if extension == '.nrrd':
                    self.files_Image.append(filename)

            for file in temp_Segmentationfiles:
                filename, extension  = os.path.splitext(file)
                if extension == '.nrrd':
                    self.files_Segmenatation.append(filename)

            
            self.tableWidget.setRowCount(len(self.files_Image))
            for i in range(len(self.files_Image)):
                checkbox = QTableWidgetItem()
                checkbox.setCheckState(Qt.Checked)
                self.tableWidget.setItem(i,1,QTableWidgetItem(self.files_Image[i]))
                self.tableWidget.item(i,1).setFlags(QtCore.Qt.ItemIsEnabled)
                self.tableWidget.setItem(i,0,checkbox)
            for i in range(len(self.files_Segmenatation)):
                self.tableWidget.setItem(i,2,QTableWidgetItem(self.files_Segmenatation[i]))
                self.tableWidget.item(i,2).setFlags(QtCore.Qt.ItemIsEnabled)
            for i in range(self.tableWidget.rowCount()):
                    if self.tableWidget.item(i,0).checkState()==Qt.Checked:
                        if self.tableWidget.item(i,1).text() != self.tableWidget.item(i,2).text(): 
                            raise RuntimeError('testError')

            self.pushButtonSelect_All.setEnabled(True)
            self.pushButtonSelect_None.setEnabled(True)
            self.pushButton_CalculateFeature.setEnabled(True)
            self.progressBar.setEnabled(True)

        except  RuntimeError:
            ctypes.windll.user32.MessageBoxW(0, "Data not march!", "INPUT ERROR", 1)

        except FileNotFoundError:
            ctypes.windll.user32.MessageBoxW(0, "Input data not found!", "INPUT ERROR", 1)


    def SelectAll(self):
        for i in range(self.tableWidget.rowCount()):
            self.tableWidget.item(i,0).setCheckState(Qt.Checked) 

    def SelectNone(self):
        for i in range(self.tableWidget.rowCount()):
            self.tableWidget.item(i,0).setCheckState(Qt.Unchecked) 

    def ReturnPredictionValue(self,original_firstorder_Skewness:float, original_shape_Flatness:float, wavelet_HLL_glszm_LargeAreaHighGrayLevelEmphasis:float, wavelet_LLL_gldm_LargeDependenceHighGrayLevelEmphasis:float, age:float, FVC:float,LDH_rate:float ):
        ##############################################################################
        ##already got 4 features value
        # original_firstorder_Skewness = 2.19359
        # original_shape_Flatness = 0.526696
        # wavelet_HLL_glszm_LargeAreaHighGrayLevelEmphasis = 3192030000
        # wavelet_LLL_gldm_LargeDependenceHighGrayLevelEmphasis = 13654.9
        ##############################################################################

        ##scale with training cohort
        a = (original_firstorder_Skewness - 1.880817)/0.7693646
        b = (original_shape_Flatness - 0.5360677)/0.04865724
        c = (wavelet_HLL_glszm_LargeAreaHighGrayLevelEmphasis - 12543870000)/14860100000
        d = (wavelet_LLL_gldm_LargeDependenceHighGrayLevelEmphasis - 27874.14)/11933.33

        ##rad_score calculation
        rad_score = a * (-1.3008) + b * 0.6083 + c * (-0.4295) + d * 0.3595

        ##form datafram for test patient imformation
        test_patient = pd.DataFrame([(rad_score,age,FVC,LDH_rate)])
        test_patient.columns = ('rad_score','age','FVC<50','LDH_rate')

        ## form cox model
        # train = pd.read_csv('train.plus.rad_score_renew+HRCTscore.csv')
        # cph = CoxPHFitter()

        # #data reorgination
        # feature_tr = train[['Survival','CustomLabel','rad_score','age','FVC<50','LDH_rate']]

        # cph.fit(feature_tr, duration_col='Survival', event_col='CustomLabel')

        filename = 'cph_model.sav'
        # dump(cph, open(filename, 'wb'))
        cph = load(open(filename, 'rb'))
        
        predict = cph.predict_survival_function(test_patient,24) #test predict
        return round(rad_score,3), round(predict.iat[0,0],3)

        #find baseline hazard fuction
        # cph.baseline_hazard_
        # cph.baseline_cumulative_hazard_
        # cph.baseline_survival_

    def Run(self):
        try:
            self.progressBar.setValue(0)
            index = 0
            keys, values= [], []
            self.count = 0
            for i in range(self.tableWidget.rowCount()):
                if self.tableWidget.item(i,0).checkState()==Qt.Checked:
                    if self.tableWidget.item(i,1).text() == self.tableWidget.item(i,2).text(): 
                        self.count = self.count + 1
                                    
            for i in range(self.tableWidget.rowCount()):
                if self.tableWidget.item(i,0).checkState()==Qt.Checked:
                    if self.tableWidget.item(i,1).text() != self.tableWidget.item(i,2).text() or self.tableWidget.item(i,3)== None or self.tableWidget.item(i,4) == None or self.tableWidget.item(i,5)== None : 
                        raise RuntimeError('testError')
            
            for i in range(self.tableWidget.rowCount()):
                if self.tableWidget.item(i,0).checkState()==Qt.Checked:
                        # feature = CalculateFeature(self.folder_Image+self.tableWidget.item(i,1).text(), self.tableWidget.item(i,2).text())
                        file_image = self.folder+'/Image/'+self.tableWidget.item(i,1).text()+'.nrrd'
                        file_segmentation = self.folder+'/Segmentation/'+self.tableWidget.item(i,2).text()+'.nrrd'
                        result = self.CalculateFeature(file_image, file_segmentation)
                        index = index+1
                        values = []
                        for key, value in result.items():
                            values.append(str(value))       

                        keys = []
                        for key, value in result.items():
                                keys.append(key)

                        result_csv=pd.DataFrame(columns=keys,index=[index], data=[values])
                        if index == 1:
                            result_csv.to_csv(self.folder+'/result.csv',mode='w', encoding='gbk',header=True)
                        else:
                            result_csv.to_csv(self.folder+'/result.csv',mode='a', encoding='gbk',header=False)
                        
                        age = float(self.tableWidget.item(i,3).text())
                        FVC = float(self.tableWidget.item(i,4).text())
                        LDH_rate = float(self.tableWidget.item(i,5).text())

                        feature1 = float(result_csv.at[index,'original_firstorder_Skewness'])
                        feature2 = float(result_csv.at[index,'original_shape_Flatness'])
                        feature3 = float(result_csv.at[index,'wavelet-HLL_glszm_LargeAreaHighGrayLevelEmphasis'])
                        feature4 = float(result_csv.at[index,'wavelet-LLL_gldm_LargeDependenceHighGrayLevelEmphasis'])
                        radscore, prediction = self.ReturnPredictionValue(feature1,feature2,feature3,feature4,age, FVC, LDH_rate)
                        self.tableWidget.setItem(i,6,QTableWidgetItem(str(radscore)))
                        self.tableWidget.item(i,6).setFlags(QtCore.Qt.ItemIsEnabled)
                        self.tableWidget.setItem(i,7,QTableWidgetItem(str(prediction)))
                        self.tableWidget.item(i,7).setFlags(QtCore.Qt.ItemIsEnabled)
                        prediction_csv=pd.DataFrame([[self.tableWidget.item(i,1).text(),prediction]],columns=['Name','Prediction'])
                        if index == 1:
                            prediction_csv.to_csv(self.folder+'/prediction.csv',mode='w', encoding='gbk',header=True)
                        else:
                            prediction_csv.to_csv(self.folder+'/prediction.csv',mode='a', encoding='gbk',header=False)

                        self.progressBar.setValue(round(index*100/self.count))
            
            print("Done!")
        except  RuntimeError:
            ctypes.windll.user32.MessageBoxW(0, "Image/segmentation mismarch error, please correct!", "INPUT ERROR", 1)


    def CalculateFeature(self, imagePath, labelPath):
        paramPath =  sys.path[0] + '\\Params.yaml'
        print (paramPath, imagePath, labelPath)
        extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
        result = extractor.execute(imagePath, labelPath)
        return result




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())