# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:22:09 2020

@author: z003w4bu
"""

##############################################################################
import pandas as pd
## 3 clinical factor already got
age = 45
FVC = 0
LDH_rate  = 0

##already got 4 features value
original_firstorder_Skewness = 2.19359
original_shape_Flatness = 0.526696
wavelet_HLL_glszm_LargeAreaHighGrayLevelEmphasis = 3192030000
wavelet_LLL_gldm_LargeDependenceHighGrayLevelEmphasis = 13654.9
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
train = pd.read_csv('train.plus.rad_score_renew+HRCTscore.csv')
from lifelines import CoxPHFitter
cph = CoxPHFitter()
#data reorgination
feature_tr = train[['Survival','CustomLabel','rad_score','age','FVC<50','LDH_rate']]
cph.fit(feature_tr, duration_col='Survival', event_col='CustomLabel')

#cph.plot(hazard_ratios=True)
#cph.predict_median(feature_te)
cph.predict_survival_function(test_patient,24)#test predict

#find baseline hazard fuction
cph.baseline_hazard_
cph.baseline_cumulative_hazard_
cph.baseline_survival_



