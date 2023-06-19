#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:14:09 2021

@author: ubuntu
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import time

from ast import Interactive
from sklearn import linear_model

#%%
def regression(fileflujos,filepresiones,filetiempos,filevolumenes,name,Y_data,i):
    flujos=np.asarray((fileflujos))*1/60
    presiones=10.2*np.asarray((filepresiones))
    tiempos=np.asarray((filetiempos))
    volumenes=np.asarray((filevolumenes)) 
    volumenes=volumenes
    
    
    FV=np.zeros((flujos.shape[0],2))
    FV[:,0]=flujos
    FV[:,1]=volumenes
    reg = linear_model.LinearRegression()
    reg.fit(FV,presiones)
    R,E=reg.coef_
    print(name)
    print('Resistance(R)=',(R,2), 'cm H2O L/S, ')
    print('Compliance (Crs)=',(1000*1/E), 'ml/cm H2O')

    print('---------------')
    P=E*volumenes+R*flujos
    erre = (R)
    ce = (1000*1/E)
    Y_data[i,0] = erre
    Y_data[i,1] = ce

    return