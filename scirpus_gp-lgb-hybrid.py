from statsmodels.robust import mad

import matplotlib.pyplot as plt

from scipy.signal import butter

from scipy import signal

from scipy.signal import savgol_filter

import seaborn as sns

from sklearn import *

import pandas as pd 

import numpy as np

import warnings

import scipy

import pywt

import os

import gc

from sklearn.metrics import f1_score

plt.style.use('ggplot')

sns.set_style('darkgrid')

class GP:

    def __init__(self):

        self.classes = 11

        self.class_names = [ 'class_0',

                             'class_1',

                             'class_2',

                             'class_3',

                             'class_4',

                             'class_5',

                             'class_6',

                             'class_7',

                             'class_8',

                             'class_9',

                             'class_10']





    def GrabPredictions(self, data):

        oof_preds = np.zeros((len(data), len(self.class_names)))

        oof_preds[:,0] = self.GP_class_0(data)

        oof_preds[:,1] = self.GP_class_1(data)

        oof_preds[:,2] = self.GP_class_2(data)

        oof_preds[:,3] = self.GP_class_3(data)

        oof_preds[:,4] = self.GP_class_4(data)

        oof_preds[:,5] = self.GP_class_5(data)

        oof_preds[:,6] = self.GP_class_6(data)

        oof_preds[:,7] = self.GP_class_7(data)

        oof_preds[:,8] = self.GP_class_8(data)

        oof_preds[:,9] = self.GP_class_9(data)

        oof_preds[:,10] = self.GP_class_10(data)

        oof_df = pd.DataFrame(oof_preds, columns=self.class_names)

        oof_df =oof_df.div(oof_df.sum(axis=1), axis=0)

        return oof_df



    def Output(self, p):

        return 1.0/(1.0+np.exp(-p))

   

    def GP_class_0(self,data):

        return self.Output( -1.394145 +

                            0.100000*np.tanh(((np.where(((((np.where((-1.0) > -998, (((data["meanbatch_slices2"]) + (data["rangebatch_slices2"]))/2.0), (((2.0)) + (data["minbatch_msignal"])) )) - (data["meanbatch_slices2"]))) - ((4.64834403991699219))) > -998, data["abs_minbatch_slices2_msignal"], data["abs_minbatch_msignal"] )) + (data["minbatch_msignal"]))) +

                            0.100000*np.tanh(np.where(((data["mean_abs_chgbatch_slices2_msignal"]) + (((np.sin((np.cos((((data["abs_minbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_msignal"]))))))) / 2.0))) <= -998, data["maxbatch_slices2"], ((data["minbatch_msignal"]) + (data["mean_abs_chgbatch_msignal"])) )) +

                            0.100000*np.tanh((((-(((-((data["minbatch_slices2_msignal"]))))))) - (np.where((-(((-((np.tanh((data["signal_shift_-1"])))))))) > -998, data["signal"], data["abs_minbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh((((data["abs_avgbatch_msignal"]) + (((np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["minbatch_slices2_msignal"], np.where(data["medianbatch_slices2_msignal"] <= -998, data["minbatch_slices2_msignal"], (((3.0)) * (np.where((((data["rangebatch_msignal"]) + ((((9.0)) + (data["abs_minbatch_msignal"]))))/2.0) <= -998, data["minbatch_slices2_msignal"], np.cos((np.where((-1.0) > -998, data["minbatch_slices2_msignal"], data["abs_minbatch_slices2_msignal"] ))) ))) ) )) * 2.0)))/2.0)) +

                            0.100000*np.tanh(((((data["abs_avgbatch_msignal"]) + (data["minbatch_slices2_msignal"]))) + (data["minbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(((data["minbatch_slices2_msignal"]) + (((data["abs_minbatch_slices2_msignal"]) + (np.where((((data["abs_minbatch_slices2_msignal"]) + (np.tanh((((data["signal_shift_+1_msignal"]) - (np.cos((data["rangebatch_msignal"]))))))))/2.0) > -998, data["minbatch_slices2_msignal"], (((3.0)) * 2.0) )))))) +

                            0.100000*np.tanh(((((data["minbatch_msignal"]) - (((np.where(np.where(np.tanh((data["abs_minbatch_msignal"])) > -998, data["mean_abs_chgbatch_msignal"], ((np.where((1.0) > -998, (-(((3.0)))), (((3.0)) * 2.0) )) / 2.0) ) <= -998, (3.0), np.cos(((3.0))) )) * 2.0)))) * 2.0)) +

                            0.100000*np.tanh(((data["minbatch"]) + (((np.tanh((data["meanbatch_slices2_msignal"]))) + (np.where(np.tanh((data["abs_minbatch_slices2_msignal"])) <= -998, np.where(data["mean_abs_chgbatch_msignal"] <= -998, data["mean_abs_chgbatch_msignal"], np.tanh((np.sin((data["abs_avgbatch_slices2_msignal"])))) ), ((data["abs_minbatch_msignal"]) * 2.0) )))))) +

                            0.100000*np.tanh(((((np.where(np.where(data["abs_maxbatch"] > -998, (-3.0), data["abs_avgbatch_slices2_msignal"] ) > -998, ((((data["minbatch_msignal"]) * 2.0)) - ((-3.0))), ((data["minbatch_msignal"]) + ((-3.0))) )) + (((data["abs_minbatch_msignal"]) - ((-3.0)))))) * 2.0)) +

                            0.100000*np.tanh(((data["maxtominbatch_slices2_msignal"]) + ((((data["minbatch"]) + (((data["minbatch"]) + (data["abs_minbatch_slices2_msignal"]))))/2.0)))) +

                            0.100000*np.tanh(((data["rangebatch_msignal"]) - (np.where(data["rangebatch_msignal"] > -998, ((((((((data["rangebatch_msignal"]) * ((-((data["minbatch_slices2_msignal"])))))) - (data["stdbatch_msignal"]))) * ((-((data["minbatch_slices2_msignal"])))))) - ((3.0))), ((data["maxtominbatch"]) - (data["meanbatch_slices2"])) )))) +

                            0.100000*np.tanh(np.where((-3.0) <= -998, ((((data["abs_minbatch_slices2_msignal"]) / 2.0)) + (((data["stdbatch_slices2_msignal"]) * (data["medianbatch_msignal"])))), ((data["minbatch_msignal"]) + (((((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0)) * 2.0))) )) +

                            0.100000*np.tanh(np.where(((data["stdbatch_slices2_msignal"]) / 2.0) > -998, ((np.where(data["signal_shift_+1_msignal"] > -998, data["minbatch_msignal"], ((data["maxbatch_slices2_msignal"]) / 2.0) )) + ((((data["abs_avgbatch_slices2_msignal"]) + (data["maxbatch_msignal"]))/2.0))), ((((data["minbatch_slices2"]) + ((((np.sin((np.where(data["signal_shift_+1_msignal"] > -998, data["minbatch_msignal"], data["meanbatch_slices2_msignal"] )))) + (data["stdbatch_slices2_msignal"]))/2.0)))) + (data["mean_abs_chgbatch_slices2_msignal"])) )) +

                            0.100000*np.tanh(np.where(((np.where(data["mean_abs_chgbatch_msignal"] > -998, data["stdbatch_slices2_msignal"], data["mean_abs_chgbatch_msignal"] )) - (((data["signal"]) + (data["minbatch_slices2"])))) <= -998, data["stdbatch_slices2_msignal"], ((((data["abs_minbatch_slices2_msignal"]) + (data["minbatch_slices2"]))) + (((np.where(((data["abs_minbatch_msignal"]) + (data["abs_minbatch_msignal"])) > -998, data["maxbatch_msignal"], ((data["mean_abs_chgbatch_msignal"]) * (data["minbatch_msignal"])) )) / 2.0))) )) +

                            0.100000*np.tanh(((((((5.51651144027709961)) + (((data["abs_maxbatch"]) + ((((((5.51651144027709961)) * (data["minbatch_slices2_msignal"]))) - (np.cos((np.where(data["maxtominbatch_slices2_msignal"] > -998, (0.0), (5.51651144027709961) )))))))))/2.0)) * 2.0)) +

                            0.100000*np.tanh(((np.where(np.sin((np.cos(((((((data["minbatch_slices2"]) + (np.cos((data["abs_minbatch_slices2_msignal"]))))/2.0)) + (data["abs_minbatch_msignal"])))))) > -998, data["maxtominbatch_slices2_msignal"], np.sin((data["maxtominbatch_slices2"])) )) + (data["minbatch_slices2_msignal"]))) +

                            0.100000*np.tanh((((3.0)) + (((data["minbatch_slices2_msignal"]) + ((((2.0)) * (((data["minbatch_slices2_msignal"]) + ((((2.0)) * (((data["minbatch_slices2_msignal"]) + ((((3.0)) + (np.where(((((data["minbatch_msignal"]) / 2.0)) / 2.0) <= -998, np.where((-((data["mean_abs_chgbatch_slices2"]))) > -998, (3.0), data["stdbatch_slices2_msignal"] ), data["minbatch_slices2_msignal"] )))))))))))))))) +

                            0.100000*np.tanh(((data["abs_minbatch_slices2_msignal"]) + (np.where(np.tanh((data["maxtominbatch_slices2_msignal"])) > -998, (1.0), data["minbatch_msignal"] )))) +

                            0.100000*np.tanh(((np.where(np.cos((((np.tanh((np.cos((data["abs_minbatch_msignal"]))))) * (np.tanh((np.cos((np.tanh((data["maxtominbatch_slices2_msignal"])))))))))) > -998, (((2.0)) + (data["minbatch_msignal"])), data["abs_minbatch_msignal"] )) * 2.0)) +

                            0.100000*np.tanh(np.where(data["maxbatch_slices2"] > -998, data["abs_minbatch_slices2_msignal"], data["abs_minbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(((((((np.cos((np.where((1.03379869461059570) > -998, data["minbatch_msignal"], np.cos((data["minbatch_msignal"])) )))) * 2.0)) * 2.0)) - (np.where((1.03379869461059570) > -998, ((data["maxbatch_slices2"]) / 2.0), np.cos(((((((2.0)) - (np.cos((((np.cos((data["minbatch_msignal"]))) * 2.0)))))) * 2.0))) )))) +

                            0.100000*np.tanh(((data["abs_minbatch_slices2_msignal"]) + (data["meanbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(((data["minbatch_msignal"]) - (np.where(((data["minbatch_msignal"]) - (np.where((((-3.0)) * 2.0) > -998, (-3.0), data["minbatch_msignal"] ))) > -998, (-3.0), np.where(((np.cos((np.cos((((((data["minbatch_msignal"]) - (data["maxtominbatch_slices2"]))) / 2.0)))))) * 2.0) > -998, (-3.0), ((data["signal_shift_+1_msignal"]) * 2.0) ) )))) +

                            0.100000*np.tanh((((((7.0)) + (data["minbatch_slices2_msignal"]))) + (np.where(data["rangebatch_msignal"] <= -998, ((np.cos(((-((((((data["minbatch_slices2_msignal"]) * 2.0)) - (((((data["minbatch_slices2_msignal"]) * (((((((data["minbatch_slices2_msignal"]) * 2.0)) * 2.0)) * 2.0)))) / 2.0))))))))) / 2.0), ((((data["minbatch_slices2_msignal"]) * 2.0)) * 2.0) )))) +

                            0.100000*np.tanh(((((((((((((((np.sin((data["abs_avgbatch_slices2_msignal"]))) + (((data["abs_avgbatch_msignal"]) * 2.0)))/2.0)) / 2.0)) + (np.where(((((((3.0)) + (data["stdbatch_msignal"]))/2.0)) / 2.0) <= -998, data["abs_avgbatch_slices2_msignal"], np.cos((data["abs_avgbatch_slices2_msignal"])) )))) + (np.where(data["signal_shift_+1_msignal"] > -998, data["minbatch_msignal"], np.tanh((data["maxtominbatch_msignal"])) )))/2.0)) + (data["abs_avgbatch_msignal"]))) * 2.0)) +

                            0.100000*np.tanh((((3.0)) + (np.where(((data["minbatch_msignal"]) + ((3.0))) <= -998, data["stdbatch_slices2"], data["minbatch_msignal"] )))) +

                            0.100000*np.tanh((((14.12892627716064453)) * ((((((14.12892627716064453)) * (((np.cos((np.where((14.12892627716064453) <= -998, np.where(data["minbatch_slices2_msignal"] > -998, data["abs_avgbatch_msignal"], data["abs_maxbatch_slices2"] ), np.where(np.tanh((data["minbatch"])) <= -998, (2.0), data["minbatch_slices2_msignal"] ) )))) / 2.0)))) / 2.0)))) +

                            0.100000*np.tanh((-(((((((((np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["rangebatch_slices2"], (-(((((data["rangebatch_slices2"]) + (data["signal_shift_-1"]))/2.0)))) )) + (data["stdbatch_slices2"]))) + (data["rangebatch_slices2"]))/2.0)) - ((4.0))))))) +

                            0.100000*np.tanh(((data["abs_avgbatch_slices2_msignal"]) + (np.where(data["maxbatch_slices2_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], data["maxbatch_msignal"] )))) +

                            0.100000*np.tanh((((((((data["minbatch_slices2_msignal"]) + ((((8.0)) * (np.where(((np.cos((np.where(data["rangebatch_slices2"] > -998, data["minbatch_slices2_msignal"], ((data["abs_avgbatch_slices2_msignal"]) * 2.0) )))) / 2.0) > -998, data["minbatch_slices2_msignal"], np.cos((data["maxtominbatch_msignal"])) )))))) + ((((-1.0)) + ((12.81776809692382812)))))/2.0)) + ((((8.0)) * (np.cos((data["minbatch_slices2_msignal"]))))))) +

                            0.100000*np.tanh((((3.0)) + (np.where(np.tanh((data["maxtominbatch_slices2"])) <= -998, (1.0), np.where(data["stdbatch_msignal"] <= -998, (3.0), (((3.0)) + (((data["minbatch_msignal"]) * (np.where((3.0) > -998, (3.0), (-(((((3.0)) + (((data["minbatch_msignal"]) * (((data["stdbatch_slices2_msignal"]) * 2.0)))))))) ))))) ) )))) +

                            0.100000*np.tanh(np.where((-((np.where(data["minbatch_msignal"] <= -998, np.where(data["stdbatch_slices2"] > -998, (3.0), data["mean_abs_chgbatch_slices2_msignal"] ), ((data["meanbatch_msignal"]) + (data["minbatch_msignal"])) )))) <= -998, data["minbatch_msignal"], np.where(data["minbatch_msignal"] <= -998, data["minbatch_msignal"], ((data["minbatch_msignal"]) + ((3.0))) ) )) +

                            0.100000*np.tanh(np.where(data["signal_shift_-1_msignal"] > -998, (((((4.0)) - (((data["rangebatch_slices2"]) + (np.sin(((((-((np.tanh((((np.cos((np.tanh((((np.tanh(((3.81548619270324707)))) * 2.0)))))) * (np.sin(((2.0))))))))))) / 2.0)))))))) * 2.0), np.where(np.tanh(((4.0))) > -998, (3.81548619270324707), (4.0) ) )) +

                            0.100000*np.tanh(np.where(np.cos((((np.tanh((((np.sin((data["abs_maxbatch_msignal"]))) + (data["abs_avgbatch_msignal"]))))) * 2.0))) > -998, ((((data["minbatch_slices2"]) + (data["abs_avgbatch_msignal"]))) * 2.0), np.where(np.where((((data["mean_abs_chgbatch_msignal"]) + ((-1.0)))/2.0) > -998, np.sin((data["abs_maxbatch_msignal"])), data["medianbatch_slices2_msignal"] ) <= -998, data["signal_shift_-1"], data["abs_maxbatch_slices2_msignal"] ) )) +

                            0.100000*np.tanh(np.where((8.0) <= -998, data["minbatch"], (((2.0)) + (data["minbatch_slices2_msignal"])) )) +

                            0.100000*np.tanh(((((((np.sin((data["abs_maxbatch_msignal"]))) + ((((data["abs_avgbatch_msignal"]) + (data["minbatch_slices2_msignal"]))/2.0)))) + (np.cos((data["minbatch_slices2_msignal"]))))) + (np.where((-(((-1.0)))) > -998, ((data["abs_maxbatch_slices2_msignal"]) + ((2.0))), np.cos((data["rangebatch_slices2"])) )))) +

                            0.100000*np.tanh(((np.sin((data["abs_maxbatch_slices2_msignal"]))) * (((((data["abs_maxbatch_msignal"]) * 2.0)) - (np.sin((((np.where(data["meanbatch_slices2_msignal"] > -998, data["abs_maxbatch_msignal"], data["abs_maxbatch_slices2_msignal"] )) + (data["rangebatch_slices2"]))))))))) +

                            0.100000*np.tanh(((((data["minbatch_msignal"]) + ((((7.0)) - (np.tanh(((-(((7.0))))))))))) - (((data["rangebatch_slices2"]) - (((data["maxtominbatch_slices2"]) - (((data["rangebatch_slices2"]) - (data["abs_maxbatch"]))))))))) +

                            0.100000*np.tanh(((((np.where(np.cos((data["stdbatch_msignal"])) <= -998, data["signal"], np.where(data["abs_avgbatch_msignal"] <= -998, np.where(data["medianbatch_slices2"] <= -998, data["meanbatch_msignal"], data["meanbatch_slices2_msignal"] ), data["abs_avgbatch_slices2_msignal"] ) )) + (np.tanh(((3.0)))))) / 2.0)) +

                            0.100000*np.tanh(data["abs_maxbatch_msignal"]) +

                            0.100000*np.tanh(np.where((-1.0) <= -998, np.sin((np.sin((data["maxtominbatch_slices2"])))), np.where(data["minbatch_msignal"] <= -998, data["maxbatch_slices2_msignal"], ((data["minbatch_slices2_msignal"]) + (((np.sin((((((np.cos((data["minbatch_msignal"]))) * 2.0)) - ((0.0)))))) * 2.0))) ) )) +

                            0.100000*np.tanh(np.cos((np.where(data["maxbatch_slices2_msignal"] > -998, ((np.sin(((9.0)))) + (data["abs_minbatch_slices2_msignal"])), data["mean_abs_chgbatch_slices2"] )))) +

                            0.100000*np.tanh(((((data["stdbatch_slices2_msignal"]) + (((data["abs_minbatch_msignal"]) + (((((((np.where((3.0) > -998, np.sin((data["abs_maxbatch_msignal"])), data["signal_shift_-1_msignal"] )) * 2.0)) * 2.0)) * 2.0)))))) + (data["signal_shift_-1_msignal"]))) +

                            0.100000*np.tanh(np.where(np.tanh(((((2.0)) + (((np.cos((data["minbatch_msignal"]))) * 2.0))))) > -998, ((np.tanh((((((data["maxbatch_slices2_msignal"]) * (np.cos(((2.0)))))) + ((((((((2.0)) + (data["minbatch_msignal"]))) * 2.0)) * 2.0)))))) * 2.0), (-((data["minbatch_msignal"]))) )) +

                            0.100000*np.tanh(np.where(np.where((12.28963756561279297) <= -998, data["abs_avgbatch_msignal"], (3.0) ) <= -998, data["maxbatch_slices2_msignal"], data["abs_avgbatch_msignal"] )) +

                            0.100000*np.tanh(((((6.15842866897583008)) + (((data["abs_maxbatch_slices2_msignal"]) * (np.where(data["stdbatch_msignal"] <= -998, ((((3.0)) + ((((((np.tanh((np.cos((np.where(np.sin((((data["medianbatch_slices2_msignal"]) * 2.0))) <= -998, data["abs_minbatch_slices2_msignal"], data["stdbatch_msignal"] )))))) + (data["minbatch_msignal"]))/2.0)) + (((((2.0)) + ((2.0)))/2.0)))))/2.0), data["minbatch_slices2_msignal"] )))))/2.0)) +

                            0.100000*np.tanh(np.where((-((data["abs_avgbatch_msignal"]))) > -998, (((4.0)) * (np.cos((np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, data["minbatch_slices2_msignal"], ((data["abs_avgbatch_msignal"]) * (data["minbatch_slices2"])) ))))), data["abs_minbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(((np.sin((data["rangebatch_slices2"]))) + (data["maxtominbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(((np.where(np.where((2.0) <= -998, (-((((np.where(data["rangebatch_slices2_msignal"] > -998, (2.0), data["signal"] )) - (((data["minbatch_slices2_msignal"]) + ((0.0)))))))), data["minbatch_msignal"] ) <= -998, ((data["mean_abs_chgbatch_slices2_msignal"]) - (((data["mean_abs_chgbatch_slices2_msignal"]) + ((1.0))))), (3.0) )) - (data["maxbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(((((((3.0)) / 2.0)) + ((((3.0)) + (((data["minbatch_slices2_msignal"]) * 2.0)))))/2.0)) +

                            0.100000*np.tanh(np.where(data["abs_avgbatch_msignal"] > -998, (((((((data["abs_minbatch_slices2"]) + (data["maxbatch_msignal"]))/2.0)) - (data["maxbatch_msignal"]))) / 2.0), data["maxtominbatch_slices2"] )) +

                            0.100000*np.tanh(np.tanh((((np.where(((np.where((((9.0)) / 2.0) <= -998, data["stdbatch_msignal"], ((np.sin((data["minbatch_msignal"]))) + (data["maxbatch_msignal"])) )) + (np.cos((data["rangebatch_slices2_msignal"])))) <= -998, data["abs_maxbatch"], np.sin((data["rangebatch_slices2"])) )) * 2.0)))) +

                            0.100000*np.tanh((((5.0)) - (np.where(np.where(((((np.tanh((data["abs_maxbatch_msignal"]))) - (np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["rangebatch_msignal"], (5.0) )))) * 2.0) <= -998, data["abs_maxbatch_msignal"], ((data["abs_avgbatch_slices2"]) / 2.0) ) <= -998, ((data["rangebatch_slices2"]) - (data["medianbatch_slices2"])), data["rangebatch_slices2"] )))) +

                            0.100000*np.tanh(((((np.where(data["stdbatch_msignal"] <= -998, data["abs_maxbatch_slices2"], data["abs_avgbatch_msignal"] )) + ((1.0)))) * (np.sin((np.where(data["medianbatch_slices2_msignal"] > -998, data["maxbatch_msignal"], ((((data["abs_maxbatch_slices2_msignal"]) * (((np.cos((np.tanh((data["mean_abs_chgbatch_slices2"]))))) * 2.0)))) * 2.0) )))))) +

                            0.100000*np.tanh(((np.where(data["maxbatch_slices2_msignal"] <= -998, data["abs_avgbatch_slices2"], np.cos((np.where(np.tanh((np.cos((data["medianbatch_msignal"])))) <= -998, data["minbatch"], data["minbatch_slices2_msignal"] ))) )) * 2.0)) +

                            0.100000*np.tanh(((((((5.0)) + (((((data["minbatch_msignal"]) - (np.where(data["signal_shift_+1_msignal"] <= -998, ((((5.0)) + (((data["minbatch"]) * 2.0)))/2.0), ((data["signal_shift_+1_msignal"]) * (data["rangebatch_msignal"])) )))) * 2.0)))/2.0)) * 2.0)) +

                            0.100000*np.tanh(((np.where(((data["meanbatch_slices2_msignal"]) / 2.0) > -998, ((data["abs_maxbatch_slices2_msignal"]) * 2.0), (((3.0)) / 2.0) )) + (data["stdbatch_msignal"]))) +

                            0.100000*np.tanh(np.where((2.0) > -998, (2.0), ((data["abs_minbatch_slices2"]) * 2.0) )) +

                            0.100000*np.tanh(((np.cos((((data["signal_shift_-1"]) - (np.where(data["abs_minbatch_msignal"] > -998, data["minbatch_slices2"], ((((data["abs_avgbatch_slices2"]) * (((data["signal_shift_-1"]) + (((np.cos((((data["signal_shift_-1"]) - (data["stdbatch_msignal"]))))) * 2.0)))))) * ((-((data["signal_shift_-1"]))))) )))))) * 2.0)) +

                            0.100000*np.tanh(((((((np.cos((((data["signal_shift_+1_msignal"]) - (np.where(data["abs_minbatch_slices2_msignal"] > -998, data["minbatch_slices2_msignal"], np.tanh((((data["abs_maxbatch_msignal"]) - (data["minbatch_slices2_msignal"])))) )))))) * 2.0)) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(((data["maxbatch_msignal"]) * (((((np.where(data["rangebatch_msignal"] <= -998, data["maxtominbatch_msignal"], data["maxbatch_msignal"] )) * 2.0)) * (np.sin((np.where((-((data["stdbatch_msignal"]))) > -998, data["abs_maxbatch_slices2_msignal"], data["abs_maxbatch_slices2_msignal"] )))))))) +

                            0.100000*np.tanh(((data["abs_maxbatch_msignal"]) - (((data["abs_maxbatch_msignal"]) * (((data["abs_maxbatch_msignal"]) * (data["stdbatch_slices2_msignal"]))))))) +

                            0.100000*np.tanh(((np.where(((((data["abs_minbatch_msignal"]) - ((1.69130122661590576)))) - ((((data["maxtominbatch"]) + (data["rangebatch_slices2"]))/2.0))) <= -998, data["rangebatch_slices2"], ((((((1.69130122661590576)) + (((data["minbatch_slices2_msignal"]) + (np.where(data["mean_abs_chgbatch_msignal"] > -998, (0.0), np.sin((data["maxbatch_msignal"])) )))))/2.0)) * 2.0) )) * 2.0)) +

                            0.100000*np.tanh(((((np.sin((data["maxbatch_slices2"]))) + (((data["abs_avgbatch_slices2_msignal"]) + (np.where(((data["meanbatch_msignal"]) - (((data["signal_shift_-1_msignal"]) * 2.0))) <= -998, data["mean_abs_chgbatch_slices2"], data["minbatch_msignal"] )))))) + (np.tanh((((data["rangebatch_msignal"]) * ((2.0)))))))) +

                            0.100000*np.tanh(np.where(data["rangebatch_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], (1.0) )) +

                            0.100000*np.tanh((-((((np.where((((((9.0)) * 2.0)) * 2.0) <= -998, np.sin((np.cos((((data["signal_shift_-1_msignal"]) * 2.0))))), (((1.0)) * (((data["rangebatch_slices2_msignal"]) * 2.0))) )) * (((data["signal_shift_-1_msignal"]) * 2.0))))))) +

                            0.100000*np.tanh(((((np.cos((((data["stdbatch_slices2"]) * (data["minbatch"]))))) * 2.0)) * (np.where(np.sin((np.sin((data["maxtominbatch_slices2_msignal"])))) <= -998, np.sin((data["abs_avgbatch_slices2_msignal"])), np.where(((np.cos((((data["stdbatch_slices2"]) * (data["minbatch"]))))) * 2.0) <= -998, (((-1.0)) / 2.0), np.tanh((data["abs_avgbatch_slices2"])) ) )))) +

                            0.100000*np.tanh((((((data["minbatch_slices2_msignal"]) + (np.where((((data["stdbatch_slices2_msignal"]) + (np.tanh(((2.0)))))/2.0) <= -998, data["medianbatch_msignal"], ((data["maxbatch_msignal"]) - (((data["stdbatch_msignal"]) * (((data["abs_avgbatch_slices2"]) * (data["medianbatch_msignal"])))))) )))/2.0)) * 2.0)) +

                            0.100000*np.tanh(np.where((((5.02276659011840820)) * (data["rangebatch_slices2_msignal"])) <= -998, np.where(np.sin((data["rangebatch_msignal"])) > -998, data["abs_maxbatch_slices2"], data["maxbatch_slices2"] ), data["rangebatch_slices2_msignal"] )) +

                            0.100000*np.tanh(((np.sin((np.where(data["abs_maxbatch_slices2"] > -998, data["maxbatch_msignal"], np.tanh(((0.0))) )))) + ((((((data["maxtominbatch_msignal"]) * (np.sin((data["abs_minbatch_slices2_msignal"]))))) + (data["maxbatch_msignal"]))/2.0)))) +

                            0.100000*np.tanh(np.where(data["maxbatch_slices2_msignal"] > -998, ((data["stdbatch_msignal"]) - (((data["abs_avgbatch_msignal"]) * (((data["mean_abs_chgbatch_slices2_msignal"]) * ((((3.0)) - (data["abs_minbatch_msignal"])))))))), ((((np.sin((np.tanh((((((np.tanh((data["abs_minbatch_msignal"]))) * 2.0)) * 2.0)))))) / 2.0)) + (data["minbatch_msignal"])) )) +

                            0.100000*np.tanh(np.sin((np.where(np.sin((np.where((0.0) > -998, data["maxbatch_slices2_msignal"], data["medianbatch_slices2"] ))) > -998, data["abs_maxbatch_msignal"], data["medianbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh((((-((data["signal_shift_-1_msignal"])))) * (np.where(np.where(data["meanbatch_slices2"] > -998, data["maxtominbatch_slices2_msignal"], data["signal_shift_-1_msignal"] ) > -998, np.where(np.tanh((data["meanbatch_msignal"])) > -998, data["abs_maxbatch_msignal"], np.sin((np.cos(((10.0))))) ), np.where(((data["abs_avgbatch_slices2_msignal"]) / 2.0) > -998, ((np.sin((data["maxbatch_msignal"]))) * 2.0), data["stdbatch_slices2_msignal"] ) )))) +

                            0.100000*np.tanh(np.cos(((((np.cos((data["abs_maxbatch_slices2"]))) + ((((data["maxtominbatch_msignal"]) + (data["abs_maxbatch_slices2"]))/2.0)))/2.0)))) +

                            0.100000*np.tanh(((np.where(np.sin((data["signal_shift_-1_msignal"])) <= -998, (((np.cos((data["minbatch_slices2_msignal"]))) + (data["medianbatch_slices2_msignal"]))/2.0), ((np.where(data["abs_minbatch_msignal"] <= -998, (2.0), np.where(np.where(data["maxtominbatch_slices2"] <= -998, data["medianbatch_slices2"], data["signal_shift_+1_msignal"] ) > -998, np.cos((data["minbatch_slices2_msignal"])), np.sin((np.cos(((-((data["minbatch_slices2_msignal"]))))))) ) )) * 2.0) )) * 2.0)) +

                            0.100000*np.tanh(((data["abs_maxbatch_msignal"]) - (((np.cos((data["abs_maxbatch"]))) - (((data["minbatch_msignal"]) - (((((((((((np.sin((((((data["minbatch_msignal"]) / 2.0)) / 2.0)))) * 2.0)) * 2.0)) / 2.0)) / 2.0)) / 2.0)))))))) +

                            0.100000*np.tanh(np.sin((np.where(((np.sin((np.sin((((np.where(np.sin((data["abs_maxbatch_msignal"])) > -998, data["abs_maxbatch_msignal"], data["medianbatch_slices2_msignal"] )) / 2.0)))))) * 2.0) > -998, data["meanbatch_slices2_msignal"], ((np.sin((np.where(np.where(((data["maxtominbatch_slices2_msignal"]) * 2.0) > -998, data["abs_maxbatch_msignal"], (0.0) ) > -998, data["meanbatch_slices2_msignal"], data["signal"] )))) * 2.0) )))) +

                            0.100000*np.tanh(((np.tanh((((data["meanbatch_slices2_msignal"]) * (np.where(np.sin((((data["maxtominbatch_slices2"]) - (data["signal_shift_+1"])))) <= -998, data["signal_shift_+1"], ((data["signal"]) * (data["medianbatch_msignal"])) )))))) + ((((((9.29164600372314453)) * (np.cos((data["minbatch_msignal"]))))) - (data["medianbatch_msignal"]))))) +

                            0.100000*np.tanh(np.tanh(((((3.0)) - (np.where(((data["signal_shift_+1_msignal"]) * (data["minbatch"])) <= -998, ((data["minbatch"]) * ((1.0))), np.where(data["medianbatch_msignal"] > -998, data["meanbatch_msignal"], np.where(data["stdbatch_slices2_msignal"] <= -998, np.where((3.0) <= -998, (3.0), (((data["meanbatch_msignal"]) + (((data["minbatch"]) - (data["abs_maxbatch_slices2_msignal"]))))/2.0) ), data["meanbatch_msignal"] ) ) )))))) +

                            0.100000*np.tanh(((data["maxtominbatch_slices2"]) * ((-((np.cos((np.where(((data["abs_avgbatch_slices2"]) * (data["maxtominbatch_slices2"])) > -998, data["abs_avgbatch_slices2"], ((((data["maxtominbatch_slices2_msignal"]) * ((3.0)))) - (data["abs_minbatch_slices2_msignal"])) ))))))))) +

                            0.100000*np.tanh(((data["maxbatch_msignal"]) + (np.where((((3.0)) + ((((data["maxbatch_msignal"]) + (data["abs_minbatch_slices2"]))/2.0))) > -998, data["minbatch_slices2_msignal"], ((np.where(data["maxbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_msignal"], data["rangebatch_slices2_msignal"] )) - (((data["rangebatch_slices2"]) + ((((data["minbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))/2.0))))) )))) +

                            0.100000*np.tanh((((-2.0)) * (np.tanh((((np.where((-((np.where(((((data["abs_maxbatch_slices2"]) / 2.0)) / 2.0) <= -998, data["medianbatch_slices2_msignal"], ((data["minbatch"]) + ((-2.0))) )))) > -998, data["signal_shift_-1_msignal"], data["signal_shift_-1_msignal"] )) * 2.0)))))) +

                            0.100000*np.tanh(np.where(np.where(data["signal_shift_-1_msignal"] <= -998, (3.0), (-2.0) ) <= -998, data["mean_abs_chgbatch_slices2_msignal"], np.where(data["abs_minbatch_msignal"] > -998, ((np.sin((data["medianbatch_slices2_msignal"]))) * (((data["medianbatch_slices2"]) * 2.0))), ((data["meanbatch_slices2"]) * (np.sin(((((np.sin((np.sin((np.sin((data["medianbatch_slices2_msignal"]))))))) + (data["signal_shift_+1"]))/2.0))))) ) )) +

                            0.100000*np.tanh((((((((((3.73540258407592773)) - (data["rangebatch_slices2"]))) + (((((np.cos(((((((3.73540258407592773)) - (data["abs_maxbatch"]))) - (data["mean_abs_chgbatch_slices2"]))))) / 2.0)) * (data["abs_avgbatch_msignal"]))))) - (data["rangebatch_slices2"]))) + ((1.62539756298065186)))) +

                            0.100000*np.tanh(((((data["maxbatch_slices2"]) * (np.sin((((data["rangebatch_slices2"]) - (np.where(data["abs_avgbatch_msignal"] > -998, data["medianbatch_msignal"], ((data["maxbatch_slices2"]) / 2.0) )))))))) + (((data["abs_avgbatch_slices2_msignal"]) + (((data["rangebatch_slices2"]) * (np.sin((((data["rangebatch_slices2"]) - (data["abs_avgbatch_msignal"]))))))))))) +

                            0.100000*np.tanh(((data["abs_avgbatch_msignal"]) * ((((((13.46102237701416016)) - (np.tanh((data["rangebatch_msignal"]))))) * ((-((data["signal_shift_-1_msignal"])))))))) +

                            0.100000*np.tanh(((data["abs_avgbatch_msignal"]) * 2.0)) +

                            0.100000*np.tanh(np.where(((np.where(data["maxbatch_msignal"] > -998, data["minbatch_msignal"], data["abs_avgbatch_slices2_msignal"] )) + (data["maxbatch_slices2_msignal"])) > -998, np.sin((data["maxbatch_msignal"])), (3.0) )) +

                            0.100000*np.tanh((3.0)) +

                            0.100000*np.tanh(((data["minbatch"]) - (np.where(((data["minbatch_slices2_msignal"]) / 2.0) > -998, ((data["mean_abs_chgbatch_slices2_msignal"]) * (data["signal_shift_+1"])), (((data["abs_maxbatch_slices2_msignal"]) + (np.where(np.where(((data["mean_abs_chgbatch_slices2_msignal"]) * (data["signal_shift_-1"])) <= -998, data["abs_avgbatch_msignal"], np.cos((np.tanh((data["maxtominbatch_msignal"])))) ) > -998, ((((data["stdbatch_slices2"]) * 2.0)) - (data["rangebatch_slices2_msignal"])), (0.0) )))/2.0) )))) +

                            0.100000*np.tanh(np.where(data["abs_minbatch_slices2"] <= -998, data["abs_minbatch_slices2_msignal"], np.where(np.sin((data["abs_avgbatch_slices2_msignal"])) > -998, (((np.where(data["mean_abs_chgbatch_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], data["rangebatch_slices2"] )) + (data["abs_avgbatch_slices2_msignal"]))/2.0), (3.0) ) )) +

                            0.100000*np.tanh((((((2.0)) * ((((((np.where((((2.0)) - (data["signal_shift_+1"])) <= -998, ((np.where(data["maxbatch_slices2"] > -998, data["maxbatch_slices2"], ((data["minbatch_slices2"]) / 2.0) )) / 2.0), (0.0) )) + (((data["minbatch_slices2"]) + (np.where(data["rangebatch_msignal"] <= -998, data["stdbatch_slices2_msignal"], data["maxbatch_slices2"] )))))/2.0)) - (data["signal_shift_+1"]))))) * 2.0)) +

                            0.100000*np.tanh(np.where(data["signal_shift_+1_msignal"] > -998, np.where(data["maxbatch_slices2"] > -998, ((((data["abs_avgbatch_slices2_msignal"]) * 2.0)) * (((np.sin((data["abs_maxbatch_slices2_msignal"]))) * (((data["maxbatch_slices2"]) - (((((data["abs_minbatch_slices2"]) * 2.0)) - ((3.0))))))))), np.where(data["abs_avgbatch_slices2_msignal"] <= -998, data["abs_maxbatch_slices2_msignal"], data["signal_shift_+1_msignal"] ) ), data["abs_maxbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(np.where(data["meanbatch_msignal"] > -998, np.where(np.where(data["signal_shift_-1_msignal"] > -998, (((-3.0)) * ((-3.0))), data["rangebatch_msignal"] ) > -998, (((-3.0)) * (data["signal_shift_-1_msignal"])), ((data["abs_avgbatch_slices2"]) - (data["meanbatch_slices2_msignal"])) ), data["medianbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(((np.cos((((((data["meanbatch_slices2_msignal"]) - (np.cos((np.where(np.cos((data["meanbatch_slices2"])) > -998, data["mean_abs_chgbatch_slices2"], np.sin((data["signal_shift_-1_msignal"])) )))))) / 2.0)))) * (((data["mean_abs_chgbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh(np.where(np.where(data["abs_avgbatch_slices2_msignal"] <= -998, np.where((4.0) <= -998, data["medianbatch_msignal"], (((data["abs_avgbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))/2.0) ), (((4.0)) - (data["medianbatch_msignal"])) ) > -998, ((np.where(data["abs_maxbatch_slices2"] > -998, (4.0), np.sin((np.sin((data["maxtominbatch_slices2"])))) )) - (data["medianbatch_msignal"])), (4.0) )) +

                            0.100000*np.tanh(np.where(((np.where((((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, (0.41478762030601501), data["minbatch_slices2"] )) + (data["meanbatch_slices2_msignal"]))/2.0) <= -998, data["maxbatch_msignal"], data["abs_minbatch_slices2"] )) / 2.0) > -998, np.sin((np.sin((data["meanbatch_slices2_msignal"])))), (((((data["maxbatch_msignal"]) / 2.0)) + (np.where(np.sin((data["maxtominbatch_slices2_msignal"])) <= -998, data["minbatch_slices2_msignal"], np.sin((data["meanbatch_slices2_msignal"])) )))/2.0) )) +

                            0.100000*np.tanh(np.where(data["medianbatch_msignal"] > -998, ((data["medianbatch_slices2"]) * (np.sin((data["medianbatch_msignal"])))), ((data["medianbatch_slices2"]) * (data["medianbatch_msignal"])) )) +

                            0.100000*np.tanh(((np.where(data["abs_avgbatch_msignal"] > -998, np.where((((5.28550529479980469)) + (((data["abs_maxbatch_slices2"]) * ((3.0))))) > -998, np.cos((data["stdbatch_slices2"])), ((((-((data["abs_avgbatch_msignal"])))) + (data["rangebatch_msignal"]))/2.0) ), data["stdbatch_slices2"] )) * 2.0)) +

                            0.100000*np.tanh(np.where(data["abs_maxbatch_msignal"] <= -998, ((((data["rangebatch_slices2"]) + ((2.0)))) - (((data["abs_minbatch_slices2"]) * (data["signal_shift_+1"])))), ((np.where(((data["rangebatch_slices2"]) * 2.0) > -998, ((np.where(np.sin((data["maxbatch_msignal"])) <= -998, data["abs_maxbatch_slices2_msignal"], data["abs_avgbatch_slices2"] )) - (data["signal_shift_+1"])), ((np.tanh(((1.0)))) * (data["abs_maxbatch"])) )) * 2.0) )) +

                            0.100000*np.tanh(((data["maxbatch_slices2_msignal"]) * (np.where(data["meanbatch_slices2_msignal"] <= -998, data["stdbatch_slices2"], np.where(np.cos((data["stdbatch_slices2_msignal"])) <= -998, data["stdbatch_slices2_msignal"], np.where(((data["maxbatch_slices2_msignal"]) * (np.sin((data["stdbatch_slices2_msignal"])))) <= -998, ((data["abs_avgbatch_slices2_msignal"]) - (np.where((5.0) <= -998, data["maxtominbatch_msignal"], data["maxbatch_slices2_msignal"] ))), np.sin((data["abs_avgbatch_slices2_msignal"])) ) ) )))) +

                            0.100000*np.tanh(np.sin((np.where(((data["maxbatch_msignal"]) - ((((-2.0)) * (((np.cos((data["stdbatch_slices2"]))) * (data["minbatch_msignal"])))))) <= -998, (((((3.0)) * (((data["stdbatch_slices2"]) / 2.0)))) * (((((((data["minbatch_msignal"]) - (data["maxtominbatch"]))) * 2.0)) * 2.0))), data["abs_maxbatch_msignal"] )))) +

                            0.100000*np.tanh(np.where(data["signal_shift_-1"] > -998, (((((((data["abs_minbatch_slices2_msignal"]) + (((data["minbatch"]) + ((2.0)))))/2.0)) * 2.0)) / 2.0), ((data["maxbatch_slices2_msignal"]) + (np.sin((((data["rangebatch_slices2_msignal"]) + (np.tanh((((data["minbatch_slices2"]) * 2.0))))))))) )) +

                            0.100000*np.tanh(np.where(np.where(np.where(((data["maxbatch_msignal"]) * 2.0) > -998, data["stdbatch_slices2"], np.tanh((data["maxbatch_msignal"])) ) <= -998, data["maxbatch_msignal"], np.sin((data["maxbatch_msignal"])) ) <= -998, data["maxbatch_msignal"], ((data["maxbatch_msignal"]) * (np.sin((data["maxbatch_msignal"])))) )) +

                            0.100000*np.tanh(((((data["medianbatch_slices2_msignal"]) - (((data["signal_shift_-1"]) + (np.cos((np.where(data["abs_avgbatch_slices2"] > -998, data["maxbatch_slices2"], data["abs_minbatch_msignal"] )))))))) / 2.0)) +

                            0.100000*np.tanh(((np.where(np.where(np.cos((data["minbatch_slices2_msignal"])) <= -998, (-(((((((-3.0)) + ((((-1.0)) * 2.0)))) - (((np.cos((data["minbatch_slices2_msignal"]))) * 2.0)))))), np.cos((data["abs_maxbatch_slices2"])) ) <= -998, data["abs_minbatch_slices2_msignal"], np.cos((((np.cos((data["minbatch_slices2_msignal"]))) - (((data["abs_maxbatch_slices2"]) / 2.0))))) )) * 2.0)) +

                            0.100000*np.tanh((((3.0)) + (((data["minbatch_msignal"]) + (((np.where(data["meanbatch_msignal"] <= -998, data["meanbatch_msignal"], ((np.sin((data["abs_maxbatch_msignal"]))) * 2.0) )) - (np.tanh((np.where((3.0) > -998, data["signal_shift_-1_msignal"], (1.0) )))))))))) +

                            0.100000*np.tanh(((np.sin((np.where(data["medianbatch_slices2"] > -998, data["abs_maxbatch_msignal"], np.where(data["meanbatch_slices2_msignal"] > -998, np.tanh((np.sin((np.where(data["stdbatch_slices2"] > -998, data["abs_maxbatch_msignal"], data["abs_maxbatch_slices2_msignal"] ))))), data["abs_avgbatch_slices2"] ) )))) * 2.0)) +

                            0.100000*np.tanh(((data["rangebatch_slices2_msignal"]) + (data["abs_avgbatch_msignal"]))) +

                            0.100000*np.tanh(((data["abs_minbatch_msignal"]) - ((-((((data["abs_minbatch_msignal"]) - (data["maxtominbatch_slices2"])))))))) +

                            0.100000*np.tanh(((((data["maxbatch_msignal"]) - (np.where((((data["abs_minbatch_slices2"]) + (np.sin((data["stdbatch_slices2"]))))/2.0) <= -998, ((np.cos((data["stdbatch_slices2"]))) * (data["stdbatch_slices2_msignal"])), np.cos(((((data["meanbatch_slices2_msignal"]) + (np.sin(((2.38480615615844727)))))/2.0))) )))) * (np.cos((data["stdbatch_slices2"]))))) +

                            0.100000*np.tanh(((data["medianbatch_slices2"]) + (data["minbatch_slices2"]))) +

                            0.100000*np.tanh((((-1.0)) + ((((-1.0)) + ((-((((np.where(data["abs_maxbatch_msignal"] > -998, data["maxtominbatch_slices2"], (((-1.0)) + (((data["medianbatch_slices2_msignal"]) + (((((np.sin((data["maxtominbatch_msignal"]))) + (((data["meanbatch_slices2"]) * 2.0)))) * 2.0))))) )) + (((data["meanbatch_slices2"]) * 2.0))))))))))) +

                            0.100000*np.tanh((((9.0)) * (((((np.tanh((data["maxbatch_msignal"]))) + ((((-((data["minbatch_slices2_msignal"])))) + (np.where((((np.cos((data["abs_minbatch_slices2_msignal"]))) + (np.tanh(((2.0)))))/2.0) > -998, data["minbatch_msignal"], (2.0) )))))) * 2.0)))) +

                            0.100000*np.tanh(((data["abs_avgbatch_slices2"]) + (np.where(data["abs_avgbatch_slices2"] > -998, data["minbatch_msignal"], ((np.where(data["abs_avgbatch_slices2"] > -998, ((data["maxbatch_msignal"]) + (data["maxbatch_msignal"])), np.cos((((data["abs_avgbatch_slices2"]) + (data["minbatch_msignal"])))) )) - ((-2.0))) )))) +

                            0.100000*np.tanh(((np.where(data["signal_shift_+1"] <= -998, np.where(((data["rangebatch_slices2"]) * 2.0) <= -998, np.cos((data["rangebatch_msignal"])), ((((data["abs_minbatch_slices2_msignal"]) * 2.0)) * 2.0) ), np.cos((((data["abs_maxbatch_slices2"]) + (np.where(np.where(data["meanbatch_msignal"] > -998, data["signal_shift_+1"], data["minbatch_slices2_msignal"] ) > -998, data["signal_shift_+1"], (1.0) ))))) )) * 2.0)) +

                            0.100000*np.tanh(((((((data["abs_avgbatch_slices2_msignal"]) - (np.where(data["maxbatch_slices2_msignal"] > -998, data["signal_shift_-1_msignal"], ((((data["abs_minbatch_slices2"]) - (data["stdbatch_slices2"]))) * 2.0) )))) - (np.where((-2.0) > -998, data["abs_avgbatch_slices2_msignal"], ((data["abs_avgbatch_slices2_msignal"]) - ((0.20605091750621796))) )))) * 2.0)) +

                            0.100000*np.tanh(np.where(data["abs_maxbatch"] <= -998, data["abs_avgbatch_slices2_msignal"], ((data["medianbatch_msignal"]) * (((data["medianbatch_msignal"]) - (((data["mean_abs_chgbatch_msignal"]) + ((((-(((-((data["mean_abs_chgbatch_msignal"]))))))) * 2.0))))))) )) +

                            0.100000*np.tanh(np.where(data["abs_maxbatch_msignal"] <= -998, data["rangebatch_msignal"], data["abs_maxbatch_slices2"] )) +

                            0.100000*np.tanh(((data["maxbatch_msignal"]) * (np.cos((np.where(data["stdbatch_msignal"] > -998, data["stdbatch_slices2"], data["rangebatch_msignal"] )))))) +

                            0.100000*np.tanh(((data["maxtominbatch"]) * (np.where((((((-1.0)) * (((data["abs_avgbatch_msignal"]) / 2.0)))) / 2.0) <= -998, data["mean_abs_chgbatch_msignal"], np.cos((data["abs_avgbatch_msignal"])) )))) +

                            0.100000*np.tanh(((((np.sin((np.sin((np.where(data["abs_maxbatch_msignal"] <= -998, data["maxbatch_slices2"], data["abs_maxbatch_msignal"] )))))) * (np.where(data["abs_avgbatch_slices2"] <= -998, ((data["minbatch_msignal"]) - (((np.where(np.cos((data["signal_shift_-1_msignal"])) <= -998, np.tanh((data["minbatch_slices2_msignal"])), data["abs_maxbatch_msignal"] )) / 2.0))), (((3.0)) / 2.0) )))) * 2.0)) +

                            0.100000*np.tanh(np.sin((np.where(np.where(data["mean_abs_chgbatch_slices2"] > -998, (((6.42563724517822266)) + (np.tanh((np.where(data["abs_avgbatch_slices2"] > -998, data["abs_avgbatch_msignal"], data["abs_maxbatch"] ))))), data["maxbatch_slices2_msignal"] ) > -998, (-((data["minbatch_msignal"]))), np.tanh((((data["rangebatch_msignal"]) - (np.sin((data["rangebatch_msignal"])))))) )))) +

                            0.100000*np.tanh(np.cos((np.where(((data["signal_shift_+1_msignal"]) - (np.tanh(((3.0))))) > -998, (-((((data["abs_maxbatch_slices2"]) + (data["signal_shift_-1"]))))), ((np.tanh((data["abs_avgbatch_slices2_msignal"]))) * ((3.0))) )))) +

                            0.100000*np.tanh(((((((data["mean_abs_chgbatch_msignal"]) + (((np.where(data["rangebatch_msignal"] <= -998, ((((data["mean_abs_chgbatch_msignal"]) + (data["abs_minbatch_msignal"]))) - ((-((((data["minbatch"]) + (data["meanbatch_msignal"]))))))), data["abs_avgbatch_slices2_msignal"] )) / 2.0)))) - ((-((((data["minbatch"]) + (((((data["abs_avgbatch_msignal"]) * 2.0)) / 2.0))))))))) * (data["rangebatch_msignal"]))) +

                            0.100000*np.tanh(np.where((1.0) <= -998, (0.0), (((8.82649421691894531)) - ((((data["maxbatch_slices2"]) + ((-(((-((data["rangebatch_msignal"]))))))))/2.0))) )) +

                            0.100000*np.tanh(np.where(data["signal_shift_-1_msignal"] <= -998, (-1.0), data["abs_maxbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(((data["mean_abs_chgbatch_msignal"]) - (((((((data["rangebatch_slices2_msignal"]) + ((((((data["rangebatch_slices2_msignal"]) - (np.cos((((np.cos((data["maxbatch_slices2_msignal"]))) / 2.0)))))) + (np.cos(((-((np.cos((data["abs_avgbatch_slices2"])))))))))/2.0)))) + (((data["maxtominbatch_slices2"]) + (data["mean_abs_chgbatch_msignal"]))))) / 2.0)))) +

                            0.100000*np.tanh(((((np.sin((data["meanbatch_slices2_msignal"]))) * (np.where(((data["maxbatch_slices2_msignal"]) / 2.0) > -998, data["maxbatch_slices2_msignal"], (((np.where(data["rangebatch_msignal"] <= -998, data["rangebatch_msignal"], (13.86034679412841797) )) + (np.sin((data["meanbatch_slices2_msignal"]))))/2.0) )))) * (np.where(np.sin((np.sin((data["maxbatch_slices2_msignal"])))) > -998, data["maxbatch_slices2_msignal"], (13.86034679412841797) )))) +

                            0.100000*np.tanh(((data["minbatch_slices2_msignal"]) - ((-((((np.where(((((data["minbatch_slices2_msignal"]) + ((14.98029136657714844)))) / 2.0) > -998, data["abs_maxbatch_msignal"], data["abs_maxbatch"] )) / 2.0))))))) +

                            0.100000*np.tanh(((np.sin((np.where(np.sin((data["maxbatch_msignal"])) <= -998, np.where((2.0) > -998, data["abs_minbatch_slices2"], data["abs_avgbatch_msignal"] ), np.sin((data["abs_maxbatch_msignal"])) )))) * 2.0)) +

                            0.100000*np.tanh(((data["minbatch_slices2_msignal"]) - ((((-2.0)) + (((((data["signal_shift_-1_msignal"]) + (((np.tanh((((np.where(((data["signal_shift_+1_msignal"]) * 2.0) <= -998, data["maxbatch_msignal"], ((data["signal_shift_-1_msignal"]) + (((np.tanh((data["signal_shift_-1_msignal"]))) / 2.0))) )) + (np.tanh((data["stdbatch_slices2"]))))))) / 2.0)))) / 2.0)))))) +

                            0.100000*np.tanh(((np.tanh((data["signal_shift_-1_msignal"]))) - (((np.cos((data["signal_shift_-1_msignal"]))) * (np.where((-((((data["abs_maxbatch_slices2_msignal"]) / 2.0)))) > -998, data["maxtominbatch"], np.where((-3.0) > -998, (((data["minbatch_slices2"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0), np.tanh((data["abs_avgbatch_slices2"])) ) )))))) +

                            0.100000*np.tanh(np.sin((((np.where(np.sin((data["rangebatch_slices2_msignal"])) <= -998, ((data["abs_avgbatch_slices2"]) * 2.0), np.sin(((((((data["abs_avgbatch_slices2"]) / 2.0)) + (data["abs_avgbatch_slices2_msignal"]))/2.0))) )) * (((np.tanh((data["mean_abs_chgbatch_slices2"]))) * (data["meanbatch_slices2_msignal"]))))))) +

                            0.100000*np.tanh(np.cos((data["stdbatch_slices2"]))) +

                            0.100000*np.tanh((((-(((-((np.cos((np.where(((data["maxbatch_slices2_msignal"]) * 2.0) <= -998, np.cos((np.sin((data["meanbatch_slices2_msignal"])))), ((np.where(np.tanh(((9.80403041839599609))) <= -998, np.sin((np.cos(((-2.0))))), data["maxbatch_slices2_msignal"] )) * (data["rangebatch_slices2"])) )))))))))) * 2.0)) +

                            0.100000*np.tanh(((np.where(data["abs_maxbatch_slices2_msignal"] > -998, (((np.sin((data["signal_shift_+1"]))) + ((7.59010982513427734)))/2.0), ((data["maxtominbatch_slices2"]) * 2.0) )) - (((data["abs_avgbatch_msignal"]) / 2.0)))) +

                            0.100000*np.tanh((((-((((((data["maxtominbatch_slices2_msignal"]) + ((((-2.0)) - (data["medianbatch_msignal"]))))) + (((data["stdbatch_slices2"]) * (data["medianbatch_msignal"])))))))) - (((((np.where(((np.cos(((10.80713462829589844)))) + (data["stdbatch_slices2_msignal"])) <= -998, data["signal"], data["signal_shift_+1_msignal"] )) * ((10.80713462829589844)))) * 2.0)))) +

                            0.100000*np.tanh(np.sin(((((((np.cos((np.cos((np.tanh((((data["abs_avgbatch_slices2_msignal"]) / 2.0)))))))) * 2.0)) + (np.where(data["mean_abs_chgbatch_msignal"] > -998, data["minbatch_msignal"], np.where(((data["maxbatch_slices2_msignal"]) + (((data["minbatch_msignal"]) + (data["minbatch_msignal"])))) <= -998, data["maxbatch_slices2"], np.cos((np.cos((((data["abs_maxbatch_msignal"]) + (data["medianbatch_slices2_msignal"])))))) ) )))/2.0)))) +

                            0.100000*np.tanh(((data["maxbatch_msignal"]) * (np.where((-1.0) <= -998, ((np.where((0.0) > -998, data["stdbatch_slices2"], data["stdbatch_slices2"] )) * 2.0), np.cos((np.where(data["abs_minbatch_msignal"] <= -998, data["stdbatch_slices2"], np.where(data["stdbatch_slices2"] > -998, data["stdbatch_slices2"], data["abs_avgbatch_msignal"] ) ))) )))) +

                            0.100000*np.tanh(np.cos((np.where(data["maxbatch_msignal"] <= -998, ((np.cos(((((2.0)) + (np.tanh((data["stdbatch_slices2_msignal"]))))))) + ((((3.0)) - ((8.0))))), (((((np.where(data["maxbatch_msignal"] > -998, data["signal_shift_-1"], data["stdbatch_slices2_msignal"] )) + ((-((data["minbatch_slices2"])))))/2.0)) * 2.0) )))) +

                            0.100000*np.tanh(((((np.sin((data["meanbatch_slices2_msignal"]))) / 2.0)) * (((data["abs_avgbatch_slices2_msignal"]) + (((data["maxbatch_slices2"]) * 2.0)))))) +

                            0.100000*np.tanh(((np.cos((((data["minbatch_slices2_msignal"]) * (data["minbatch_slices2_msignal"]))))) * (data["medianbatch_slices2_msignal"]))) +

                            0.100000*np.tanh((((2.0)) * (np.sin(((-((np.where((-((np.cos((np.sin((data["meanbatch_slices2"]))))))) <= -998, ((data["maxbatch_msignal"]) * 2.0), ((((data["maxbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2"]))) * 2.0) ))))))))) +

                            0.100000*np.tanh(((data["abs_maxbatch_msignal"]) - (np.where(data["abs_maxbatch_msignal"] <= -998, ((data["mean_abs_chgbatch_slices2"]) * (data["rangebatch_slices2"])), ((data["rangebatch_msignal"]) * (np.where(data["abs_maxbatch_msignal"] > -998, data["mean_abs_chgbatch_slices2"], data["rangebatch_msignal"] ))) )))) +

                            0.100000*np.tanh(((((((((((((-2.0)) / 2.0)) + (((data["abs_maxbatch"]) * ((-3.0)))))/2.0)) + (((((((((((4.0)) + ((1.0)))/2.0)) * 2.0)) * (((data["meanbatch_slices2_msignal"]) / 2.0)))) * (np.where(data["minbatch_slices2_msignal"] > -998, data["abs_maxbatch"], (((2.0)) / 2.0) )))))) * (np.sin((data["signal_shift_-1_msignal"]))))) * 2.0)) +

                            0.100000*np.tanh(np.where(np.where(((((data["abs_maxbatch_slices2_msignal"]) + (np.cos((np.sin((((np.cos((data["signal_shift_+1_msignal"]))) * 2.0)))))))) - (((((0.0)) + (data["signal_shift_-1"]))/2.0))) <= -998, (2.0), data["abs_avgbatch_msignal"] ) <= -998, data["abs_avgbatch_msignal"], np.sin((((data["meanbatch_slices2_msignal"]) + ((-((data["signal_shift_+1_msignal"]))))))) )) +

                            0.100000*np.tanh(np.where((1.0) > -998, np.sin((np.where(data["abs_minbatch_slices2"] <= -998, np.where(data["signal_shift_-1_msignal"] <= -998, np.sin((((data["rangebatch_msignal"]) + (np.cos((data["mean_abs_chgbatch_msignal"])))))), data["maxtominbatch"] ), data["maxbatch_msignal"] ))), (((data["maxbatch_slices2_msignal"]) + (((data["rangebatch_msignal"]) / 2.0)))/2.0) )) +

                            0.100000*np.tanh(np.where(data["maxbatch_msignal"] <= -998, data["maxbatch_msignal"], np.sin((data["maxbatch_msignal"])) )) +

                            0.100000*np.tanh(((np.cos((np.tanh((np.tanh((np.where(np.sin((data["abs_maxbatch_msignal"])) <= -998, (8.46001815795898438), np.sin((data["rangebatch_msignal"])) )))))))) + (np.cos((np.cos((np.sin(((-((data["maxbatch_slices2_msignal"])))))))))))) +

                            0.100000*np.tanh((-((((data["medianbatch_slices2"]) + (((data["medianbatch_slices2"]) + ((((((np.cos((((((((data["meanbatch_msignal"]) * 2.0)) * 2.0)) / 2.0)))) + (data["abs_minbatch_slices2_msignal"]))) + ((((data["maxtominbatch_slices2_msignal"]) + (np.sin((np.tanh((((data["signal_shift_+1"]) * (((data["maxtominbatch_slices2_msignal"]) / 2.0)))))))))/2.0)))/2.0))))))))) +

                            0.100000*np.tanh((((np.sin(((-(((-((data["rangebatch_msignal"]))))))))) + ((((((2.0)) * (np.where((-3.0) <= -998, (0.0), np.cos((data["minbatch_slices2_msignal"])) )))) / 2.0)))/2.0)) +

                            0.100000*np.tanh(np.where(np.sin((data["abs_avgbatch_slices2_msignal"])) <= -998, data["abs_avgbatch_slices2_msignal"], ((((data["abs_maxbatch_msignal"]) - (data["minbatch_slices2"]))) * 2.0) )) +

                            0.100000*np.tanh(((np.sin((np.where((((((np.where(data["minbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2"], data["minbatch_slices2_msignal"] )) * 2.0)) + (((data["minbatch_slices2_msignal"]) * 2.0)))/2.0) > -998, (((data["maxbatch_msignal"]) + (((data["minbatch_slices2_msignal"]) * 2.0)))/2.0), data["maxbatch_msignal"] )))) * 2.0)) +

                            0.100000*np.tanh(np.cos((((np.cos((data["stdbatch_msignal"]))) + ((-((((data["signal_shift_-1_msignal"]) - ((((np.sin((data["signal_shift_-1"]))) + ((-((((np.sin((data["meanbatch_slices2_msignal"]))) / 2.0))))))/2.0))))))))))) +

                            0.100000*np.tanh(np.where(data["abs_minbatch_msignal"] <= -998, data["maxtominbatch_msignal"], np.cos(((-((np.cos((np.cos(((-(((-((data["stdbatch_msignal"]))))))))))))))) )) +

                            0.100000*np.tanh((-((np.sin((np.cos((data["minbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh(np.where((0.0) <= -998, np.cos((data["signal_shift_-1_msignal"])), np.where((-1.0) <= -998, ((data["abs_maxbatch_slices2"]) - (data["signal_shift_-1_msignal"])), ((((data["abs_avgbatch_slices2_msignal"]) - (data["signal_shift_-1_msignal"]))) - (np.where(data["abs_avgbatch_slices2"] <= -998, data["abs_avgbatch_slices2_msignal"], (0.0) ))) ) )) +

                            0.100000*np.tanh(np.sin((np.where(((data["minbatch_msignal"]) - (((data["abs_avgbatch_msignal"]) * 2.0))) > -998, ((data["abs_avgbatch_slices2"]) - (data["minbatch_slices2_msignal"])), np.sin((np.where(data["maxbatch_slices2"] > -998, data["maxbatch_msignal"], ((np.sin(((1.0)))) / 2.0) ))) )))) +

                            0.100000*np.tanh(data["abs_maxbatch_slices2_msignal"]) +

                            0.100000*np.tanh(np.cos((np.where(((np.where(np.cos((data["stdbatch_slices2"])) > -998, ((data["rangebatch_slices2_msignal"]) * 2.0), ((data["mean_abs_chgbatch_slices2"]) * 2.0) )) * 2.0) > -998, ((data["mean_abs_chgbatch_slices2"]) * 2.0), np.cos((np.where(data["medianbatch_msignal"] > -998, ((data["mean_abs_chgbatch_slices2"]) * 2.0), ((data["mean_abs_chgbatch_slices2"]) * 2.0) ))) )))) +

                            0.100000*np.tanh(((data["medianbatch_slices2_msignal"]) * (np.sin((data["maxbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh((((((((((data["medianbatch_slices2_msignal"]) + (((data["meanbatch_slices2"]) * (data["meanbatch_slices2"]))))/2.0)) + (data["stdbatch_msignal"]))/2.0)) + (data["meanbatch_slices2"]))/2.0)) +

                            0.100000*np.tanh(((np.sin((np.where(np.where(data["meanbatch_slices2_msignal"] <= -998, (-1.0), np.sin((np.sin((data["meanbatch_slices2"])))) ) > -998, ((data["signal_shift_+1_msignal"]) - (((((data["signal_shift_+1_msignal"]) * 2.0)) * 2.0))), (((-2.0)) * 2.0) )))) * 2.0)) +

                            0.100000*np.tanh((-(((((((np.cos(((2.0)))) + (data["stdbatch_slices2"]))/2.0)) * 2.0))))) +

                            0.100000*np.tanh(np.where((3.0) <= -998, data["abs_minbatch_msignal"], data["abs_avgbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(((np.where(data["maxtominbatch"] <= -998, ((data["rangebatch_msignal"]) - (data["meanbatch_slices2_msignal"])), data["abs_minbatch_slices2_msignal"] )) - (data["meanbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(((np.where(data["mean_abs_chgbatch_msignal"] <= -998, data["medianbatch_slices2_msignal"], ((data["medianbatch_slices2_msignal"]) * 2.0) )) * (data["maxbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(((data["abs_avgbatch_msignal"]) - ((((data["signal_shift_-1_msignal"]) + (np.where(data["rangebatch_msignal"] > -998, np.where(data["signal_shift_-1_msignal"] <= -998, (((data["abs_maxbatch_slices2_msignal"]) + ((1.0)))/2.0), (1.0) ), np.tanh((data["maxbatch_msignal"])) )))/2.0)))) +

                            0.100000*np.tanh(np.sin((np.where(data["rangebatch_slices2_msignal"] <= -998, ((data["stdbatch_slices2"]) * 2.0), np.where(((data["maxtominbatch_slices2"]) * 2.0) <= -998, np.cos((data["minbatch"])), data["rangebatch_msignal"] ) )))) +

                            0.100000*np.tanh(((np.sin((((((2.0)) + (data["signal_shift_-1_msignal"]))/2.0)))) + (((((data["minbatch_msignal"]) + (((((((((2.0)) + (np.tanh(((1.0)))))/2.0)) * 2.0)) - (((data["signal_shift_+1_msignal"]) - (np.cos((((data["signal_shift_+1"]) * 2.0)))))))))) * 2.0)))) +

                            0.100000*np.tanh(np.cos((((((data["abs_maxbatch_slices2_msignal"]) * 2.0)) * 2.0)))) +

                            0.100000*np.tanh(np.where(np.tanh((((np.tanh((data["rangebatch_msignal"]))) * 2.0))) <= -998, np.sin((((data["rangebatch_msignal"]) - (np.cos((((data["maxtominbatch_msignal"]) * ((-(((1.0)))))))))))), (((9.77249813079833984)) / 2.0) )) +

                            0.099609*np.tanh(np.cos((np.where(np.sin((np.sin((((((((np.sin(((-(((-1.0))))))) / 2.0)) * (data["maxtominbatch_msignal"]))) / 2.0))))) > -998, data["minbatch_slices2"], ((data["abs_maxbatch_slices2_msignal"]) * (data["maxbatch_slices2"])) )))) +

                            0.100000*np.tanh(np.sin((((((3.0)) + (np.where(np.where(data["minbatch"] <= -998, np.sin((data["meanbatch_slices2"])), (-((np.where(np.cos(((3.0))) > -998, (13.42668724060058594), data["abs_maxbatch_slices2"] )))) ) > -998, data["meanbatch_slices2"], np.sin(((((3.0)) + (np.sin((((((np.tanh((data["meanbatch_msignal"]))) * 2.0)) * 2.0))))))) )))/2.0)))) +

                            0.100000*np.tanh(np.sin((((((np.where(((np.sin((((data["stdbatch_slices2"]) + (np.cos((data["stdbatch_slices2"]))))))) / 2.0) > -998, (3.0), (3.0) )) * (data["abs_maxbatch"]))) * (data["stdbatch_slices2"]))))) +

                            0.100000*np.tanh(np.sin((((np.tanh((((np.where(data["abs_maxbatch_msignal"] <= -998, np.sin((data["abs_minbatch_slices2"])), (1.0) )) / 2.0)))) * 2.0)))) +

                            0.100000*np.tanh(np.cos((((((-((((data["maxbatch_slices2"]) + (np.sin((((np.cos((np.sin((np.sin((np.cos(((-2.0)))))))))) * ((-((np.tanh((np.sin(((2.03653025627136230)))))))))))))))))) + (np.tanh((((data["maxbatch_slices2"]) + (np.where(((data["minbatch_slices2_msignal"]) / 2.0) <= -998, (2.03653025627136230), data["signal_shift_+1"] )))))))/2.0)))) +

                            0.100000*np.tanh(((((np.cos((np.cos(((((((data["minbatch_msignal"]) + (np.cos((np.cos((((((((data["medianbatch_slices2_msignal"]) - (data["maxbatch_msignal"]))) * 2.0)) * 2.0)))))))/2.0)) * 2.0)))))) - (((np.cos((((((((data["medianbatch_slices2_msignal"]) - (data["maxbatch_msignal"]))) * 2.0)) * 2.0)))) * 2.0)))) * 2.0)) +

                            0.100000*np.tanh(np.tanh((((data["abs_avgbatch_slices2"]) * (np.cos((((np.cos((np.where(np.where(data["minbatch_slices2_msignal"] <= -998, data["mean_abs_chgbatch_slices2"], np.sin(((2.0))) ) > -998, data["abs_maxbatch_slices2_msignal"], np.cos((data["rangebatch_slices2"])) )))) * 2.0)))))))) +

                            0.100000*np.tanh(((((3.0)) + (data["minbatch"]))/2.0)) +

                            0.100000*np.tanh(((np.cos((np.where((((((data["minbatch_msignal"]) + (((np.sin((data["minbatch_msignal"]))) / 2.0)))/2.0)) * 2.0) <= -998, np.sin((data["mean_abs_chgbatch_msignal"])), data["stdbatch_slices2"] )))) + (((data["mean_abs_chgbatch_msignal"]) / 2.0)))) +

                            0.100000*np.tanh(np.where(((data["signal_shift_-1"]) + (((data["abs_maxbatch"]) * ((-((data["signal_shift_+1_msignal"]))))))) > -998, ((np.sin((data["rangebatch_slices2_msignal"]))) * ((-((np.where(np.cos(((((0.0)) * ((-2.0))))) > -998, data["stdbatch_msignal"], data["rangebatch_slices2_msignal"] )))))), np.tanh((((np.cos((data["maxbatch_slices2_msignal"]))) * 2.0))) )) +

                            0.100000*np.tanh(((np.where((((data["minbatch"]) + ((3.0)))/2.0) <= -998, data["meanbatch_slices2_msignal"], ((np.sin(((-((((data["meanbatch_slices2_msignal"]) - (data["signal_shift_-1_msignal"])))))))) * (((data["minbatch_msignal"]) * 2.0))) )) - (((np.sin((((((data["medianbatch_msignal"]) * 2.0)) * 2.0)))) * (((data["meanbatch_slices2_msignal"]) - ((0.0)))))))) +

                            0.089932*np.tanh((6.0)) +

                            0.100000*np.tanh(np.where(((data["maxbatch_slices2_msignal"]) * (data["rangebatch_msignal"])) <= -998, (((-((data["rangebatch_slices2"])))) * (((np.tanh((np.sin((np.sin(((((-((data["rangebatch_slices2"])))) * (np.sin(((((-((data["rangebatch_slices2"])))) * (data["signal_shift_+1_msignal"]))))))))))))) * 2.0))), np.sin(((((-((data["rangebatch_slices2"])))) * (data["signal_shift_+1_msignal"])))) )) +

                            0.098436*np.tanh(np.where(np.where((-((np.tanh((np.sin((data["abs_avgbatch_msignal"]))))))) <= -998, data["abs_maxbatch_msignal"], data["rangebatch_slices2_msignal"] ) > -998, (12.92754554748535156), data["maxtominbatch"] )) +

                            0.100000*np.tanh(np.cos((((data["maxbatch_slices2"]) / 2.0)))) +

                            0.100000*np.tanh(np.sin((((data["abs_maxbatch_slices2_msignal"]) - ((6.0)))))) +

                            0.100000*np.tanh(data["rangebatch_slices2_msignal"]) +

                            0.100000*np.tanh(((np.cos((np.where(data["minbatch_slices2"] <= -998, data["meanbatch_slices2"], data["stdbatch_slices2"] )))) * (data["meanbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(np.cos((np.where((0.0) <= -998, (((3.0)) * ((-((data["minbatch_slices2_msignal"]))))), data["minbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh(((((((((3.0)) + ((-((np.sin((data["minbatch_msignal"])))))))) + ((((((((np.cos((((data["rangebatch_slices2_msignal"]) / 2.0)))) + ((-((((data["maxtominbatch_msignal"]) * 2.0))))))/2.0)) + (data["mean_abs_chgbatch_msignal"]))) * ((((data["rangebatch_slices2"]) + (data["stdbatch_slices2"]))/2.0)))))) + (data["minbatch_msignal"]))/2.0)) +

                            0.100000*np.tanh(data["maxbatch_slices2_msignal"]) +

                            0.093744*np.tanh(np.sin((np.where(((((np.sin((np.cos((((data["minbatch_slices2"]) * 2.0)))))) + (((np.sin((np.tanh((np.sin((data["maxtominbatch_msignal"]))))))) - (((((-3.0)) + (data["stdbatch_msignal"]))/2.0)))))) * (np.sin((np.cos((data["signal_shift_+1_msignal"])))))) <= -998, data["signal_shift_+1_msignal"], (((-3.0)) * (data["signal_shift_+1_msignal"])) )))) +

                            0.100000*np.tanh(np.where(data["maxbatch_msignal"] > -998, np.sin(((((data["maxbatch_msignal"]) + (data["minbatch_slices2_msignal"]))/2.0))), (-((((data["meanbatch_slices2_msignal"]) + (np.cos((((data["maxbatch_msignal"]) / 2.0)))))))) )) +

                            0.099707*np.tanh(np.cos((np.where(np.sin((np.sin((data["mean_abs_chgbatch_slices2"])))) <= -998, (3.0), np.where(np.tanh((((data["abs_avgbatch_slices2"]) * (np.tanh((np.tanh((data["minbatch_slices2_msignal"])))))))) > -998, np.where(np.sin((np.tanh((data["meanbatch_msignal"])))) <= -998, data["minbatch"], data["minbatch_slices2_msignal"] ), data["meanbatch_slices2"] ) )))) +

                            0.100000*np.tanh(np.where(((data["abs_maxbatch"]) - ((-((((data["abs_minbatch_slices2"]) + (data["maxbatch_msignal"]))))))) > -998, np.cos(((((((data["signal_shift_+1_msignal"]) - ((0.0)))) + (data["mean_abs_chgbatch_msignal"]))/2.0))), ((data["stdbatch_slices2_msignal"]) + (data["minbatch_slices2_msignal"])) )) +

                            0.100000*np.tanh(np.sin((((((((np.sin((data["maxtominbatch_msignal"]))) + ((((((0.0)) + (data["maxbatch_msignal"]))) - (np.tanh(((((((np.cos((np.tanh((data["maxtominbatch"]))))) - (data["minbatch"]))) + (np.tanh((((data["meanbatch_msignal"]) * 2.0)))))/2.0)))))))/2.0)) + (np.where((0.0) <= -998, np.sin((data["minbatch_msignal"])), data["mean_abs_chgbatch_slices2"] )))/2.0)))) +

                            0.100000*np.tanh(np.where(np.where((((-((data["minbatch_slices2_msignal"])))) / 2.0) <= -998, data["maxbatch_slices2"], (-((data["maxtominbatch"]))) ) > -998, np.tanh(((-((np.tanh((data["medianbatch_msignal"]))))))), np.tanh(((-((((data["signal_shift_-1"]) - (data["abs_minbatch_slices2"]))))))) )))  

    

    def GP_class_1(self,data):

        return self.Output( -1.623856 +

                            0.100000*np.tanh(np.where(data["minbatch_msignal"] > -998, ((((np.tanh(((((data["mean_abs_chgbatch_slices2_msignal"]) + (np.tanh((np.tanh(((-((data["rangebatch_slices2"])))))))))/2.0)))) * 2.0)) + (((data["rangebatch_slices2"]) * (((np.sin((((data["minbatch_msignal"]) * 2.0)))) * 2.0))))), data["maxtominbatch_slices2"] )) +

                            0.100000*np.tanh(((((((((((data["minbatch_msignal"]) + ((3.0)))) * 2.0)) + (((np.where((((3.0)) / 2.0) <= -998, (2.0), (((3.0)) / 2.0) )) / 2.0)))) * 2.0)) * (((np.tanh((np.where(data["maxbatch_msignal"] <= -998, data["signal"], ((data["signal"]) + ((2.0))) )))) * 2.0)))) +

                            0.100000*np.tanh(np.where(((data["maxtominbatch_slices2_msignal"]) / 2.0) > -998, ((((((np.sin((((data["maxtominbatch_slices2_msignal"]) + ((-1.0)))))) / 2.0)) - ((((-3.0)) * ((((data["minbatch_slices2_msignal"]) + ((3.0)))/2.0)))))) * 2.0), np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["maxtominbatch_msignal"], np.cos(((((-3.0)) * ((((data["minbatch_slices2_msignal"]) + ((3.0)))/2.0))))) ) )) +

                            0.100000*np.tanh(((data["abs_maxbatch"]) * (np.cos((np.where(data["maxbatch_msignal"] > -998, data["maxbatch_msignal"], np.where(data["maxbatch_msignal"] > -998, data["maxbatch_msignal"], ((((((((-((np.tanh((data["abs_avgbatch_slices2_msignal"])))))) + (data["maxtominbatch_msignal"]))/2.0)) * 2.0)) - (((data["abs_maxbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2_msignal"])))) ) )))))) +

                            0.100000*np.tanh(((((5.0)) + (((((data["maxtominbatch_slices2_msignal"]) * (data["signal_shift_-1"]))) - (((((data["meanbatch_msignal"]) * (((np.tanh(((-3.0)))) + (data["maxtominbatch_slices2_msignal"]))))) * 2.0)))))/2.0)) +

                            0.100000*np.tanh(((np.sin((data["abs_minbatch_slices2_msignal"]))) * (np.where((((((-2.0)) * ((((data["maxtominbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"]))/2.0)))) / 2.0) > -998, np.where((((((((((1.0)) * 2.0)) * 2.0)) * 2.0)) / 2.0) > -998, data["abs_minbatch_slices2_msignal"], data["rangebatch_slices2_msignal"] ), data["rangebatch_slices2_msignal"] )))) +

                            0.100000*np.tanh(np.where(((((data["maxtominbatch_msignal"]) * 2.0)) * 2.0) <= -998, np.where(data["maxtominbatch_msignal"] > -998, data["mean_abs_chgbatch_msignal"], data["medianbatch_slices2_msignal"] ), np.where((((np.tanh((data["maxtominbatch_slices2_msignal"]))) + (((data["maxtominbatch_msignal"]) * 2.0)))/2.0) > -998, ((data["maxtominbatch_msignal"]) * 2.0), ((data["rangebatch_slices2_msignal"]) * (np.sin((np.where(data["maxtominbatch_msignal"] > -998, data["maxtominbatch_slices2_msignal"], data["medianbatch_slices2_msignal"] ))))) ) )) +

                            0.100000*np.tanh(((data["maxtominbatch_slices2_msignal"]) * ((((((2.0)) - (np.where(data["minbatch_msignal"] > -998, data["maxtominbatch_slices2_msignal"], data["maxbatch_msignal"] )))) - (data["minbatch_msignal"]))))) +

                            0.100000*np.tanh(((np.sin((data["abs_minbatch_msignal"]))) * ((-((((((((data["minbatch_slices2"]) / 2.0)) - (data["abs_minbatch_msignal"]))) + (((data["abs_minbatch_slices2_msignal"]) - (data["stdbatch_slices2_msignal"])))))))))) +

                            0.100000*np.tanh(((np.where(((data["minbatch_slices2_msignal"]) * 2.0) <= -998, data["minbatch_slices2_msignal"], data["abs_maxbatch_slices2"] )) * (np.cos((np.where((-((np.where(((data["abs_maxbatch_msignal"]) / 2.0) > -998, (((data["minbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))/2.0), data["minbatch_slices2_msignal"] )))) > -998, (((data["minbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))/2.0), data["abs_maxbatch_msignal"] )))))) +

                            0.100000*np.tanh((((((0.15674118697643280)) + (data["minbatch_msignal"]))) + (((np.where(data["maxtominbatch_msignal"] > -998, (2.0), np.where(((((((data["maxtominbatch_msignal"]) + ((2.0)))) + (data["minbatch_msignal"]))) / 2.0) > -998, np.cos((np.where((0.15674118697643280) > -998, data["maxbatch_msignal"], data["stdbatch_msignal"] ))), data["maxtominbatch_msignal"] ) )) * 2.0)))) +

                            0.100000*np.tanh(np.where(((data["abs_minbatch_slices2_msignal"]) * 2.0) <= -998, (-((data["abs_maxbatch_slices2_msignal"]))), ((((np.cos(((-((((data["abs_minbatch_slices2_msignal"]) / 2.0))))))) * 2.0)) * 2.0) )) +

                            0.100000*np.tanh(np.where(np.where((2.0) > -998, ((data["maxbatch_slices2_msignal"]) - (((np.sin((data["maxtominbatch_msignal"]))) * 2.0))), np.sin((np.where(data["abs_avgbatch_msignal"] <= -998, data["rangebatch_msignal"], np.sin((data["maxbatch_msignal"])) ))) ) > -998, ((np.where(data["abs_minbatch_msignal"] <= -998, data["meanbatch_slices2"], (2.0) )) - (data["maxbatch_msignal"])), data["maxbatch_msignal"] )) +

                            0.100000*np.tanh(((data["minbatch_msignal"]) + (((((7.73354005813598633)) + (np.sin((((((7.73354005813598633)) + (((data["minbatch_msignal"]) + ((7.73354005813598633)))))/2.0)))))/2.0)))) +

                            0.100000*np.tanh((((3.0)) + (np.where((((3.0)) + (np.where(data["abs_minbatch_slices2_msignal"] > -998, data["minbatch_slices2_msignal"], data["rangebatch_msignal"] ))) > -998, data["minbatch_slices2_msignal"], (2.0) )))) +

                            0.100000*np.tanh(np.cos((np.where(np.where(np.cos((data["rangebatch_msignal"])) > -998, data["maxbatch_msignal"], data["maxtominbatch_slices2"] ) > -998, (((3.0)) + (data["minbatch_msignal"])), (-((((data["minbatch_slices2"]) + (np.tanh((np.sin((((((data["stdbatch_slices2_msignal"]) / 2.0)) + (data["maxbatch_msignal"]))))))))))) )))) +

                            0.100000*np.tanh(((data["signal"]) * (np.where((((data["maxtominbatch"]) + ((-(((0.0))))))/2.0) <= -998, np.tanh((np.where(data["maxtominbatch_msignal"] <= -998, np.tanh((((data["maxtominbatch_msignal"]) * 2.0))), np.cos((((data["maxtominbatch_msignal"]) * 2.0))) ))), np.cos((((data["maxtominbatch_msignal"]) * 2.0))) )))) +

                            0.100000*np.tanh(np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["mean_abs_chgbatch_slices2_msignal"], np.cos((((data["abs_maxbatch_slices2_msignal"]) + (data["maxtominbatch_msignal"])))) )) +

                            0.100000*np.tanh(((((np.sin((((np.where(np.where(((data["minbatch"]) * 2.0) <= -998, ((data["maxbatch_msignal"]) + (data["minbatch_msignal"])), np.sin((np.where(data["minbatch_msignal"] <= -998, data["mean_abs_chgbatch_slices2_msignal"], ((np.where(((((data["minbatch"]) * 2.0)) * 2.0) <= -998, (-2.0), data["maxbatch_msignal"] )) * 2.0) ))) ) <= -998, data["minbatch_msignal"], data["maxbatch_msignal"] )) * 2.0)))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(((((np.where(np.cos((data["mean_abs_chgbatch_slices2_msignal"])) > -998, np.cos((data["maxbatch_msignal"])), np.where(data["maxtominbatch_slices2_msignal"] > -998, np.cos((data["maxbatch_msignal"])), data["maxbatch_msignal"] ) )) * 2.0)) * 2.0)) +

                            0.100000*np.tanh((((data["minbatch_msignal"]) + (((data["minbatch_msignal"]) + ((((8.55910587310791016)) + (((np.cos((data["minbatch_slices2_msignal"]))) - (np.cos(((((8.55910587310791016)) + ((((data["minbatch_slices2_msignal"]) + ((((((((data["minbatch_slices2_msignal"]) + (data["minbatch_msignal"]))/2.0)) * 2.0)) + ((((8.55910587310791016)) + ((8.55910587310791016)))))))/2.0)))))))))))))/2.0)) +

                            0.100000*np.tanh((((((((((data["rangebatch_slices2"]) + (((((((((7.0)) - (data["maxbatch_slices2"]))) - (np.sin((np.tanh((np.where(data["stdbatch_slices2_msignal"] > -998, data["signal_shift_-1_msignal"], np.cos(((((data["abs_avgbatch_msignal"]) + (np.cos((data["mean_abs_chgbatch_slices2_msignal"]))))/2.0))) )))))))) + (data["abs_avgbatch_msignal"]))/2.0)))/2.0)) / 2.0)) * (np.cos((data["maxbatch_msignal"]))))) * 2.0)) +

                            0.100000*np.tanh((((3.0)) * (np.where(np.tanh((data["maxtominbatch_msignal"])) > -998, ((data["stdbatch_slices2_msignal"]) * (np.sin((data["abs_minbatch_msignal"])))), np.where(((np.sin((np.where(((data["maxbatch_slices2"]) * 2.0) > -998, data["stdbatch_msignal"], data["abs_minbatch_slices2"] )))) / 2.0) <= -998, data["abs_minbatch_slices2"], data["maxtominbatch_msignal"] ) )))) +

                            0.100000*np.tanh((((((3.0)) - (((np.where((((data["meanbatch_slices2"]) + ((((((-2.0)) + ((-2.0)))) * 2.0)))/2.0) > -998, data["rangebatch_slices2"], (((3.0)) * 2.0) )) + ((-2.0)))))) - (((np.where((3.0) > -998, data["rangebatch_slices2"], data["maxbatch_slices2"] )) + ((-2.0)))))) +

                            0.100000*np.tanh((((6.0)) - (np.where(((data["rangebatch_slices2"]) * (np.cos((data["rangebatch_slices2"])))) <= -998, (6.0), data["rangebatch_slices2"] )))) +

                            0.100000*np.tanh(((((((data["minbatch_msignal"]) + ((3.0)))) + ((((((3.0)) + (data["minbatch_msignal"]))) * 2.0)))) + ((((((3.0)) + (data["minbatch_msignal"]))) * 2.0)))) +

                            0.100000*np.tanh(np.where(np.sin((np.cos((np.tanh(((((data["meanbatch_slices2_msignal"]) + ((((2.0)) * (np.cos((data["abs_avgbatch_slices2_msignal"]))))))/2.0))))))) <= -998, np.cos((data["meanbatch_slices2"])), np.sin((((data["minbatch_msignal"]) * 2.0))) )) +

                            0.100000*np.tanh(np.where(np.where(((data["maxbatch_msignal"]) * 2.0) > -998, np.sin((np.sin((((data["maxbatch_msignal"]) * 2.0))))), np.sin((data["signal"])) ) > -998, ((((np.sin((((data["maxbatch_msignal"]) * 2.0)))) * 2.0)) * 2.0), np.sin((((data["maxbatch_msignal"]) * 2.0))) )) +

                            0.100000*np.tanh(((data["medianbatch_msignal"]) - (((np.where(((data["minbatch_msignal"]) - (data["minbatch_msignal"])) > -998, ((((np.where(np.cos((data["minbatch_msignal"])) > -998, np.cos((data["minbatch_msignal"])), (-1.0) )) * 2.0)) * 2.0), ((np.cos((data["minbatch_msignal"]))) + ((((-1.0)) + (np.cos((data["rangebatch_slices2"])))))) )) * 2.0)))) +

                            0.100000*np.tanh(((np.where((-((data["minbatch_msignal"]))) <= -998, np.sin(((((data["minbatch_msignal"]) + (((data["minbatch_msignal"]) * 2.0)))/2.0))), ((data["minbatch_msignal"]) * 2.0) )) + ((((10.99531459808349609)) * (np.sin((((data["minbatch_msignal"]) * 2.0)))))))) +

                            0.100000*np.tanh(((((((((np.sin((data["minbatch_msignal"]))) * (np.where(np.where((-3.0) > -998, data["abs_minbatch_slices2_msignal"], np.cos((np.sin((np.sin((((data["minbatch_msignal"]) * 2.0))))))) ) > -998, data["minbatch_msignal"], np.sin(((10.0))) )))) * 2.0)) * 2.0)) * (np.sin((((data["minbatch_msignal"]) * 2.0)))))) +

                            0.100000*np.tanh(np.where(((((4.34722900390625000)) + (data["minbatch_msignal"]))/2.0) <= -998, np.sin((data["stdbatch_slices2_msignal"])), ((((4.34722900390625000)) + (data["minbatch_msignal"]))/2.0) )) +

                            0.100000*np.tanh(((((((((((data["minbatch_msignal"]) + ((3.0)))) * 2.0)) * ((3.0)))) * 2.0)) - (np.where(data["maxbatch_msignal"] > -998, data["maxbatch_msignal"], ((data["minbatch_msignal"]) + ((3.0))) )))) +

                            0.100000*np.tanh(((((data["rangebatch_slices2"]) * (np.where((1.0) > -998, np.cos((data["abs_maxbatch_msignal"])), np.where(data["maxtominbatch"] > -998, np.cos((data["abs_maxbatch_msignal"])), np.where(data["abs_maxbatch_msignal"] > -998, (-((data["abs_avgbatch_msignal"]))), data["abs_maxbatch_slices2"] ) ) )))) * 2.0)) +

                            0.100000*np.tanh(data["medianbatch_slices2_msignal"]) +

                            0.100000*np.tanh(((data["abs_maxbatch"]) * ((((((9.0)) * (np.cos((np.where(np.where(np.tanh((np.cos((np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["meanbatch_slices2"], np.sin((np.sin((np.tanh((data["abs_maxbatch_slices2_msignal"])))))) ))))) <= -998, np.cos((data["abs_maxbatch"])), np.cos((data["stdbatch_slices2"])) ) > -998, data["abs_maxbatch_slices2_msignal"], data["maxbatch_msignal"] )))))) * 2.0)))) +

                            0.100000*np.tanh(((((data["rangebatch_slices2"]) + (np.where(data["signal"] > -998, data["signal_shift_-1"], (((((((((3.0)) * 2.0)) + (data["mean_abs_chgbatch_msignal"]))) * (((data["rangebatch_slices2"]) * 2.0)))) * 2.0) )))) * 2.0)) +

                            0.100000*np.tanh(np.where(data["minbatch_slices2_msignal"] > -998, ((np.cos((((data["minbatch_msignal"]) + ((3.0)))))) * 2.0), np.where(data["minbatch_slices2_msignal"] <= -998, np.tanh((data["maxtominbatch"])), (-((np.where(data["minbatch_msignal"] > -998, data["stdbatch_slices2"], ((data["minbatch_slices2_msignal"]) * 2.0) )))) ) )) +

                            0.100000*np.tanh(np.sin((((np.sin((((np.where(np.where(data["rangebatch_msignal"] > -998, data["maxbatch_msignal"], ((data["abs_minbatch_slices2"]) * (np.where(data["rangebatch_slices2"] <= -998, ((data["rangebatch_slices2_msignal"]) - ((-2.0))), (((((1.0)) + (data["mean_abs_chgbatch_slices2_msignal"]))) * 2.0) ))) ) > -998, data["maxbatch_msignal"], data["mean_abs_chgbatch_slices2_msignal"] )) * 2.0)))) * 2.0)))) +

                            0.100000*np.tanh(np.where(data["abs_maxbatch"] <= -998, (-((data["maxbatch_slices2_msignal"]))), np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, data["meanbatch_msignal"], (((7.0)) * (((((data["meanbatch_msignal"]) + (np.cos((data["maxbatch_slices2_msignal"]))))) * 2.0))) ) )) +

                            0.100000*np.tanh(np.sin((np.where((((0.60296070575714111)) * (data["abs_avgbatch_slices2_msignal"])) <= -998, ((((data["mean_abs_chgbatch_slices2"]) - (np.sin((((((data["maxbatch_msignal"]) * 2.0)) * 2.0)))))) / 2.0), np.where(((data["maxbatch_msignal"]) * 2.0) <= -998, np.where(data["meanbatch_msignal"] <= -998, (1.0), data["maxbatch_msignal"] ), ((data["maxbatch_msignal"]) * 2.0) ) )))) +

                            0.100000*np.tanh((((((3.0)) + (np.where(((data["minbatch_slices2_msignal"]) - (data["rangebatch_slices2"])) > -998, data["minbatch_slices2_msignal"], (-2.0) )))) - (((((np.cos((((((data["maxbatch_slices2_msignal"]) - (np.where(data["abs_avgbatch_msignal"] > -998, (((-2.0)) / 2.0), data["abs_minbatch_slices2_msignal"] )))) * 2.0)))) * 2.0)) * 2.0)))) +

                            0.100000*np.tanh(np.where(np.where((-2.0) <= -998, ((data["meanbatch_msignal"]) * (data["stdbatch_msignal"])), ((data["medianbatch_slices2_msignal"]) - (np.tanh((np.tanh((((data["mean_abs_chgbatch_msignal"]) + ((3.0))))))))) ) > -998, ((data["mean_abs_chgbatch_msignal"]) + ((3.0))), np.sin((np.cos((data["medianbatch_slices2_msignal"])))) )) +

                            0.100000*np.tanh(((data["maxtominbatch_slices2_msignal"]) * ((((((data["stdbatch_msignal"]) - (data["minbatch_msignal"]))) + (np.where(((data["abs_maxbatch"]) * 2.0) <= -998, data["minbatch_slices2_msignal"], np.where(data["minbatch_slices2_msignal"] > -998, data["maxtominbatch_msignal"], ((data["maxtominbatch_slices2_msignal"]) * (np.cos((data["medianbatch_slices2_msignal"])))) ) )))/2.0)))) +

                            0.100000*np.tanh(((((data["minbatch_msignal"]) + (np.where(np.where(np.cos((data["stdbatch_slices2_msignal"])) > -998, ((np.cos((np.cos((data["maxbatch_slices2_msignal"]))))) + (np.cos((data["maxbatch_slices2_msignal"])))), (((4.13756561279296875)) * (np.cos(((((((data["minbatch_msignal"]) * 2.0)) + ((2.28219199180603027)))/2.0))))) ) > -998, (4.13756561279296875), np.cos((data["maxbatch_slices2_msignal"])) )))) * 2.0)) +

                            0.100000*np.tanh(((data["abs_maxbatch_slices2"]) * ((((np.sin((data["medianbatch_slices2_msignal"]))) + (np.cos(((((data["abs_minbatch_slices2_msignal"]) + (((data["abs_maxbatch_slices2_msignal"]) * ((((((data["medianbatch_slices2_msignal"]) + ((3.0)))/2.0)) / 2.0)))))/2.0)))))/2.0)))) +

                            0.100000*np.tanh(np.tanh((np.where(np.cos((data["abs_maxbatch_slices2_msignal"])) > -998, np.cos((data["maxtominbatch_msignal"])), data["stdbatch_slices2"] )))) +

                            0.100000*np.tanh(((data["abs_maxbatch_slices2"]) * (np.cos((np.where(np.where(data["maxbatch_slices2"] <= -998, (2.0), data["abs_maxbatch_slices2"] ) > -998, data["abs_maxbatch_msignal"], (8.0) )))))) +

                            0.100000*np.tanh(((((np.cos((np.where(np.cos((np.where(data["abs_maxbatch_msignal"] > -998, data["abs_maxbatch_msignal"], data["abs_maxbatch_msignal"] ))) > -998, data["abs_maxbatch_msignal"], ((((np.where((-3.0) > -998, data["abs_maxbatch_msignal"], data["abs_maxbatch_msignal"] )) * 2.0)) * 2.0) )))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(((((((np.cos((((data["maxbatch_msignal"]) * 2.0)))) - (((data["minbatch_slices2"]) * ((((1.0)) - (((data["minbatch_msignal"]) * (np.cos(((-((data["medianbatch_slices2"])))))))))))))) * 2.0)) + (data["medianbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(((((np.sin((((np.where(data["maxbatch_msignal"] > -998, data["maxbatch_msignal"], (-((((np.where(data["abs_avgbatch_slices2"] > -998, data["maxbatch_msignal"], data["medianbatch_slices2"] )) * 2.0)))) )) * 2.0)))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh((((6.0)) * (np.cos(((-((((np.where(((data["rangebatch_slices2"]) + ((6.0))) <= -998, (6.0), np.where(data["rangebatch_slices2_msignal"] <= -998, ((data["maxbatch_slices2_msignal"]) * (((np.cos((data["minbatch_slices2_msignal"]))) / 2.0))), data["rangebatch_slices2"] ) )) * 2.0))))))))) +

                            0.100000*np.tanh((((((7.0)) + ((((data["medianbatch_msignal"]) + (((np.sin((data["stdbatch_slices2"]))) * 2.0)))/2.0)))) * (np.sin(((((((1.0)) * (data["stdbatch_slices2"]))) + (np.where((2.0) > -998, data["medianbatch_msignal"], (12.12480354309082031) )))))))) +

                            0.100000*np.tanh((((-(((14.63042068481445312))))) * (np.cos((np.where(np.cos((data["minbatch_msignal"])) > -998, data["minbatch_msignal"], (-((np.cos((np.where((((-((data["minbatch_msignal"])))) * (np.sin((data["minbatch_slices2_msignal"])))) > -998, data["minbatch_msignal"], (((-(((3.0))))) * (data["maxbatch_msignal"])) )))))) )))))) +

                            0.100000*np.tanh((((((-((np.sin((((data["abs_maxbatch"]) - ((((data["abs_maxbatch"]) + ((-((((data["abs_maxbatch_slices2"]) * (np.where(np.where(data["abs_minbatch_slices2"] > -998, (1.0), (-((data["abs_maxbatch"]))) ) > -998, data["minbatch_slices2_msignal"], data["abs_maxbatch_slices2"] ))))))))/2.0))))))))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(((np.cos(((((data["minbatch_msignal"]) + (np.where(data["stdbatch_slices2"] <= -998, data["minbatch_msignal"], np.cos((((data["signal_shift_-1_msignal"]) * (((data["minbatch_msignal"]) * (np.sin((((data["minbatch"]) * 2.0))))))))) )))/2.0)))) * 2.0)) +

                            0.100000*np.tanh(np.where(data["medianbatch_msignal"] <= -998, np.where(data["signal_shift_-1"] <= -998, data["maxbatch_msignal"], data["maxbatch_msignal"] ), ((((np.cos(((((((data["maxbatch_msignal"]) * 2.0)) + (np.where(data["signal_shift_-1"] <= -998, data["signal"], np.tanh((((data["medianbatch_msignal"]) * (((data["signal"]) * (data["mean_abs_chgbatch_msignal"])))))) )))/2.0)))) * 2.0)) * 2.0) )) +

                            0.100000*np.tanh(((np.sin((((np.where(np.sin((data["maxbatch_msignal"])) > -998, data["maxbatch_msignal"], np.where((10.89435291290283203) <= -998, (((2.0)) - (data["maxbatch_msignal"])), np.cos((((np.where(data["maxbatch_msignal"] <= -998, data["medianbatch_msignal"], ((data["abs_maxbatch"]) * 2.0) )) * 2.0))) ) )) * 2.0)))) * 2.0)) +

                            0.100000*np.tanh(((data["rangebatch_slices2"]) * (((((((((((data["meanbatch_slices2_msignal"]) - (np.tanh((data["medianbatch_msignal"]))))) + (data["stdbatch_slices2"]))) * 2.0)) + (np.cos((np.tanh((((data["rangebatch_slices2"]) * (((data["abs_avgbatch_msignal"]) - (np.cos((data["rangebatch_slices2"]))))))))))))) * (((data["meanbatch_slices2_msignal"]) - (np.cos((data["rangebatch_slices2"]))))))))) +

                            0.100000*np.tanh(((((((14.05354499816894531)) + (((data["medianbatch_msignal"]) - (np.tanh((data["medianbatch_msignal"]))))))/2.0)) * (np.where(data["medianbatch_msignal"] > -998, np.sin((((np.tanh((np.sin((((data["medianbatch_msignal"]) * 2.0)))))) * 2.0))), ((data["medianbatch_msignal"]) / 2.0) )))) +

                            0.100000*np.tanh(((((-((np.where(data["abs_avgbatch_msignal"] > -998, (((data["maxtominbatch"]) + ((-((data["minbatch_slices2_msignal"])))))/2.0), data["maxtominbatch"] ))))) + (data["abs_minbatch_msignal"]))/2.0)) +

                            0.100000*np.tanh(np.cos((np.where(np.where(((np.cos((data["minbatch_slices2"]))) / 2.0) > -998, ((((data["signal_shift_+1_msignal"]) - (((data["maxbatch_msignal"]) - (data["abs_minbatch_slices2"]))))) - (data["maxbatch_msignal"])), data["maxbatch_msignal"] ) > -998, data["maxbatch_msignal"], (((1.0)) / 2.0) )))) +

                            0.100000*np.tanh((((((((7.0)) * 2.0)) * (((np.cos((((data["maxbatch_slices2"]) * 2.0)))) * 2.0)))) * 2.0)) +

                            0.100000*np.tanh(np.where(((np.tanh((data["maxtominbatch"]))) - (((((data["medianbatch_slices2_msignal"]) + (np.tanh(((6.0)))))) + (data["meanbatch_slices2"])))) <= -998, data["medianbatch_slices2_msignal"], (((((data["minbatch"]) * ((((-((data["abs_minbatch_slices2"])))) + (data["meanbatch_slices2"]))))) + ((6.0)))/2.0) )) +

                            0.100000*np.tanh((((4.0)) + (np.where(np.sin((data["medianbatch_msignal"])) <= -998, np.tanh(((1.0))), ((data["minbatch_msignal"]) + (np.sin((((data["abs_avgbatch_slices2_msignal"]) * 2.0))))) )))) +

                            0.100000*np.tanh(np.cos((np.where((11.96294879913330078) > -998, np.where(data["abs_maxbatch_slices2"] > -998, data["maxbatch_msignal"], np.tanh((np.where(data["minbatch_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], np.where(((data["minbatch_slices2_msignal"]) * 2.0) <= -998, data["maxbatch_slices2"], data["rangebatch_slices2_msignal"] ) ))) ), np.where((-(((11.96294879913330078)))) > -998, data["maxbatch_msignal"], data["rangebatch_slices2"] ) )))) +

                            0.100000*np.tanh(np.where(np.cos((((data["stdbatch_slices2"]) * 2.0))) <= -998, np.cos((((((data["stdbatch_slices2"]) * 2.0)) * 2.0))), (((7.0)) * ((-((np.cos((((((data["stdbatch_slices2"]) * 2.0)) * 2.0)))))))) )) +

                            0.100000*np.tanh(((((data["maxbatch_slices2"]) + (data["maxbatch_slices2"]))) * (np.where(data["abs_maxbatch"] <= -998, np.cos((data["maxbatch_slices2_msignal"])), np.cos((((data["maxbatch_slices2_msignal"]) * 2.0))) )))) +

                            0.100000*np.tanh(((np.where(data["medianbatch_msignal"] > -998, (((-(((((data["maxtominbatch"]) + (data["meanbatch_slices2"]))/2.0))))) * ((((data["maxbatch_msignal"]) + (np.sin((data["medianbatch_msignal"]))))/2.0))), np.where(data["abs_avgbatch_msignal"] <= -998, data["medianbatch_slices2_msignal"], (((data["abs_avgbatch_msignal"]) + (np.tanh((data["rangebatch_slices2"]))))/2.0) ) )) * 2.0)) +

                            0.100000*np.tanh(((data["abs_maxbatch_msignal"]) * (np.where((((np.cos((data["abs_minbatch_slices2_msignal"]))) + (data["maxbatch_msignal"]))/2.0) <= -998, data["abs_maxbatch_msignal"], np.cos((((data["maxbatch_slices2_msignal"]) * 2.0))) )))) +

                            0.100000*np.tanh(((((((((8.0)) * (data["maxbatch_slices2_msignal"]))) * (data["medianbatch_slices2_msignal"]))) + (np.where(np.cos((np.sin(((-((data["abs_avgbatch_msignal"]))))))) <= -998, ((np.where((0.0) > -998, data["meanbatch_slices2"], np.sin((np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["abs_avgbatch_msignal"], (-((data["abs_maxbatch_slices2_msignal"]))) ))) )) / 2.0), (0.05662323534488678) )))/2.0)) +

                            0.100000*np.tanh(np.where(((data["rangebatch_slices2_msignal"]) * 2.0) > -998, ((((np.cos((data["maxbatch_slices2_msignal"]))) * 2.0)) * 2.0), ((data["maxbatch_slices2_msignal"]) + (((data["maxbatch_slices2_msignal"]) / 2.0))) )) +

                            0.100000*np.tanh(np.where(data["maxbatch_slices2_msignal"] <= -998, data["signal_shift_-1"], ((data["medianbatch_slices2"]) * (np.cos((((((data["abs_avgbatch_slices2_msignal"]) - ((-((np.cos((data["abs_maxbatch_msignal"])))))))) * 2.0))))) )) +

                            0.100000*np.tanh(((np.sin((((np.where(np.where((1.0) > -998, np.cos((data["maxbatch_slices2_msignal"])), (1.0) ) <= -998, (1.0), ((np.cos((np.where(((data["signal_shift_+1_msignal"]) * (data["abs_avgbatch_msignal"])) <= -998, np.sin(((1.0))), data["maxbatch_slices2_msignal"] )))) * 2.0) )) - (np.where(np.cos((data["maxbatch_slices2_msignal"])) > -998, data["meanbatch_msignal"], (1.0) )))))) * 2.0)) +

                            0.100000*np.tanh((-((((np.cos((((data["medianbatch_slices2_msignal"]) - (np.where(((data["medianbatch_slices2_msignal"]) / 2.0) > -998, data["rangebatch_slices2"], ((((((6.0)) - (((np.where((6.0) > -998, data["rangebatch_slices2"], (((-((data["medianbatch_slices2_msignal"])))) * 2.0) )) * 2.0)))) + (data["medianbatch_slices2_msignal"]))/2.0) )))))) * (data["rangebatch_slices2"])))))) +

                            0.100000*np.tanh(((((((data["medianbatch_msignal"]) * 2.0)) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.where(np.cos((data["maxbatch_slices2_msignal"])) > -998, ((np.cos((((data["maxbatch_msignal"]) + (np.tanh((((data["signal_shift_+1_msignal"]) * ((((-3.0)) - (data["maxbatch_msignal"]))))))))))) * 2.0), np.cos((data["maxbatch_msignal"])) )) +

                            0.100000*np.tanh(((np.where(data["maxtominbatch"] <= -998, (-2.0), ((data["minbatch_slices2_msignal"]) / 2.0) )) - ((((((((-((data["minbatch_slices2_msignal"])))) / 2.0)) * ((((data["maxtominbatch"]) + (data["maxbatch_slices2"]))/2.0)))) + (((((np.sin((np.sin((data["minbatch_slices2_msignal"]))))) * 2.0)) * 2.0)))))) +

                            0.100000*np.tanh(((data["minbatch_slices2_msignal"]) * ((-((np.cos((np.where(np.where(((data["minbatch_slices2_msignal"]) * ((-((np.cos(((-((data["abs_avgbatch_slices2"])))))))))) <= -998, data["maxbatch_msignal"], data["stdbatch_slices2"] ) > -998, data["stdbatch_slices2"], data["minbatch_slices2_msignal"] ))))))))) +

                            0.100000*np.tanh(np.where(((((data["abs_maxbatch_slices2_msignal"]) - (data["rangebatch_slices2"]))) * 2.0) > -998, np.cos((((data["maxbatch_slices2_msignal"]) * 2.0))), ((((data["maxbatch_slices2_msignal"]) * 2.0)) + (np.sin((((data["maxbatch_slices2_msignal"]) * 2.0))))) )) +

                            0.100000*np.tanh(np.sin((np.where(np.cos((((data["medianbatch_msignal"]) * (np.where(data["minbatch"] > -998, data["medianbatch_slices2_msignal"], data["abs_avgbatch_slices2_msignal"] ))))) > -998, ((data["medianbatch_msignal"]) * 2.0), np.cos((np.where(((((((data["medianbatch_slices2_msignal"]) - (np.tanh(((6.0)))))) / 2.0)) * 2.0) > -998, data["abs_avgbatch_slices2_msignal"], ((np.sin((np.cos((data["maxbatch_slices2_msignal"]))))) * 2.0) ))) )))) +

                            0.100000*np.tanh(((data["abs_maxbatch"]) * (np.sin((np.sin((np.cos((((data["maxbatch_slices2_msignal"]) * (((np.cos((np.cos((((((np.cos((np.tanh((np.sin((((data["maxbatch_slices2_msignal"]) * 2.0)))))))) * 2.0)) + (np.sin((((data["stdbatch_slices2"]) * 2.0)))))))))) + (np.where(data["medianbatch_msignal"] <= -998, data["abs_maxbatch_msignal"], data["stdbatch_slices2"] )))))))))))))) +

                            0.100000*np.tanh(np.cos((np.where(data["abs_maxbatch"] <= -998, np.where(((data["abs_maxbatch_slices2"]) / 2.0) > -998, np.cos((np.where(np.cos((data["signal_shift_-1_msignal"])) > -998, np.tanh((np.cos((data["stdbatch_slices2_msignal"])))), data["maxbatch_slices2"] ))), np.cos((data["meanbatch_slices2"])) ), data["maxbatch_msignal"] )))) +

                            0.100000*np.tanh(np.cos((np.where((((-((data["maxbatch_slices2_msignal"])))) * 2.0) <= -998, data["maxbatch_slices2_msignal"], ((data["maxbatch_slices2_msignal"]) * 2.0) )))) +

                            0.100000*np.tanh(((((np.cos((np.where((6.0) > -998, ((data["maxbatch_slices2_msignal"]) * 2.0), np.where((4.52745151519775391) > -998, (((((((4.52745151519775391)) + (np.where((4.52745151519775391) > -998, (4.52745151519775391), data["abs_minbatch_slices2"] )))/2.0)) + (np.cos(((4.52745151519775391)))))/2.0), data["medianbatch_msignal"] ) )))) - (np.cos(((((((data["minbatch_msignal"]) + ((4.52745151519775391)))/2.0)) * 2.0)))))) * 2.0)) +

                            0.100000*np.tanh(((data["minbatch_msignal"]) + (((np.cos((data["maxbatch_msignal"]))) - ((((((-3.0)) + ((-3.0)))) + ((-((((data["minbatch_msignal"]) + (((np.cos((data["maxbatch_msignal"]))) - (((np.cos(((-3.0)))) + ((-((np.cos((((data["abs_avgbatch_slices2_msignal"]) + (data["medianbatch_msignal"]))))))))))))))))))))))) +

                            0.100000*np.tanh(np.cos((((data["mean_abs_chgbatch_slices2"]) * 2.0)))) +

                            0.100000*np.tanh(((data["minbatch_msignal"]) - ((((((data["medianbatch_msignal"]) + ((-2.0)))/2.0)) - (((np.where(np.where(((data["minbatch_msignal"]) - (data["maxtominbatch_msignal"])) > -998, (9.0), data["maxtominbatch_msignal"] ) > -998, (9.0), data["minbatch_msignal"] )) / 2.0)))))) +

                            0.100000*np.tanh(np.sin((((((np.where(((data["signal_shift_-1_msignal"]) - (np.sin((np.where((-((((np.cos((data["mean_abs_chgbatch_msignal"]))) / 2.0)))) > -998, data["minbatch_slices2_msignal"], ((data["medianbatch_msignal"]) - (((data["mean_abs_chgbatch_msignal"]) * 2.0))) ))))) <= -998, (3.0), np.sin((np.where(data["stdbatch_slices2_msignal"] > -998, data["minbatch_slices2_msignal"], data["medianbatch_slices2"] ))) )) * 2.0)) * 2.0)))) +

                            0.100000*np.tanh(((((((((data["abs_maxbatch"]) + (((data["signal_shift_-1"]) * (data["mean_abs_chgbatch_msignal"]))))) - ((-3.0)))) + (((data["signal_shift_+1"]) / 2.0)))) - (data["minbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(((np.tanh((np.cos((((data["maxbatch_slices2_msignal"]) * 2.0)))))) * (np.where(np.tanh((((data["medianbatch_slices2_msignal"]) / 2.0))) > -998, ((data["maxbatch_slices2_msignal"]) * 2.0), ((np.cos((((data["minbatch_msignal"]) * (data["medianbatch_slices2"]))))) * 2.0) )))) +

                            0.100000*np.tanh(np.where(np.cos((data["minbatch_msignal"])) <= -998, data["maxbatch_slices2_msignal"], ((data["medianbatch_msignal"]) - (((np.cos((data["minbatch_msignal"]))) * (np.where(np.tanh((((data["medianbatch_msignal"]) + (np.cos((data["rangebatch_msignal"])))))) <= -998, np.cos((data["minbatch_msignal"])), np.where(((data["rangebatch_msignal"]) * 2.0) > -998, data["rangebatch_slices2"], data["abs_maxbatch"] ) ))))) )) +

                            0.100000*np.tanh(((((data["medianbatch_slices2_msignal"]) * (((((data["maxbatch_slices2_msignal"]) * 2.0)) - (data["meanbatch_msignal"]))))) * (((((np.tanh(((((3.0)) * 2.0)))) * 2.0)) - (np.where(np.cos(((-((np.tanh((data["rangebatch_slices2"]))))))) > -998, data["medianbatch_slices2_msignal"], data["medianbatch_slices2"] )))))) +

                            0.100000*np.tanh(((np.sin(((((0.0)) - ((((data["maxbatch_msignal"]) + (((((data["minbatch_slices2_msignal"]) * (np.where(np.where(data["meanbatch_msignal"] > -998, data["minbatch_slices2_msignal"], data["maxbatch_msignal"] ) > -998, (-2.0), (((-2.0)) * 2.0) )))) * 2.0)))/2.0)))))) * 2.0)) +

                            0.100000*np.tanh(np.where(((data["maxbatch_msignal"]) * 2.0) > -998, np.sin((((np.where(np.sin((data["abs_avgbatch_msignal"])) <= -998, ((data["maxbatch_msignal"]) * (np.cos((data["maxbatch_msignal"])))), ((data["medianbatch_msignal"]) * 2.0) )) + ((((1.0)) * (np.sin((data["rangebatch_slices2"])))))))), ((data["medianbatch_slices2_msignal"]) * 2.0) )) +

                            0.100000*np.tanh((-((((np.where(data["signal_shift_-1_msignal"] > -998, data["signal_shift_-1_msignal"], data["maxtominbatch_slices2"] )) * (((np.sin(((((-((((data["maxtominbatch_slices2"]) / 2.0))))) * (data["signal_shift_-1_msignal"]))))) + (data["maxbatch_slices2"])))))))) +

                            0.100000*np.tanh(np.where(((data["maxbatch_msignal"]) * 2.0) > -998, ((((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, np.where(data["mean_abs_chgbatch_slices2"] > -998, data["maxbatch_msignal"], np.where(data["maxbatch_slices2"] > -998, data["maxtominbatch"], (6.96649932861328125) ) ), (((6.96649932861328125)) - (data["maxbatch_msignal"])) )) * ((6.96649932861328125)))) * 2.0), (((6.96649932861328125)) - (data["stdbatch_slices2"])) )) +

                            0.100000*np.tanh(((((data["rangebatch_slices2"]) - (((((((2.0)) + ((((data["abs_minbatch_slices2_msignal"]) + (data["stdbatch_slices2"]))/2.0)))/2.0)) / 2.0)))) * (np.tanh((np.cos((data["stdbatch_slices2"]))))))) +

                            0.100000*np.tanh(np.cos((((np.where((-((np.sin((np.where((6.0) > -998, np.tanh(((-((data["meanbatch_slices2_msignal"]))))), data["rangebatch_slices2"] )))))) <= -998, ((np.where(((data["rangebatch_slices2_msignal"]) - ((5.27865314483642578))) > -998, data["abs_maxbatch"], (((((data["minbatch"]) / 2.0)) + (np.cos(((5.27865314483642578)))))/2.0) )) * 2.0), ((data["meanbatch_slices2_msignal"]) * 2.0) )) * 2.0)))) +

                            0.100000*np.tanh((((((((data["signal_shift_+1"]) * 2.0)) + (data["abs_maxbatch"]))/2.0)) * (((((np.cos((data["stdbatch_slices2"]))) * (((data["abs_maxbatch_slices2_msignal"]) - ((-((((np.cos((((data["stdbatch_slices2"]) - ((-((np.cos((np.sin((((np.cos((data["stdbatch_slices2"]))) * (data["rangebatch_slices2"])))))))))))))) * ((3.29388928413391113))))))))))) * 2.0)))) +

                            0.100000*np.tanh(((np.cos((((((np.where(((data["maxtominbatch_msignal"]) - (((data["maxbatch_slices2_msignal"]) - (np.cos((((data["maxtominbatch_msignal"]) - (np.sin((data["stdbatch_slices2"])))))))))) > -998, data["medianbatch_msignal"], data["maxtominbatch"] )) - (data["maxbatch_slices2_msignal"]))) * (data["maxbatch_slices2_msignal"]))))) * 2.0)) +

                            0.100000*np.tanh(np.cos((np.where((2.0) <= -998, data["maxbatch_slices2_msignal"], ((data["rangebatch_slices2"]) + (np.where(data["maxbatch_slices2_msignal"] > -998, ((np.cos((data["stdbatch_slices2"]))) + (((data["maxbatch_slices2_msignal"]) * 2.0))), ((data["abs_minbatch_slices2"]) / 2.0) ))) )))) +

                            0.100000*np.tanh(((np.cos((data["abs_maxbatch_msignal"]))) * (((np.where(data["abs_maxbatch_msignal"] <= -998, np.cos((data["minbatch_slices2"])), data["medianbatch_msignal"] )) * (data["abs_minbatch_slices2"]))))) +

                            0.100000*np.tanh(np.cos((np.where(np.where(np.cos((data["stdbatch_slices2"])) > -998, data["maxbatch_msignal"], np.cos((data["signal_shift_+1"])) ) <= -998, ((np.cos((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, ((data["stdbatch_slices2"]) + ((5.56591081619262695))), data["stdbatch_slices2"] )))) * 2.0), data["stdbatch_slices2"] )))) +

                            0.100000*np.tanh(np.where(data["signal_shift_+1"] <= -998, data["abs_maxbatch_msignal"], np.cos(((((np.cos(((0.0)))) + (np.where(data["abs_maxbatch_msignal"] <= -998, np.cos(((2.0))), ((((data["minbatch_msignal"]) * ((-2.0)))) / 2.0) )))/2.0))) )) +

                            0.100000*np.tanh(((data["maxbatch_msignal"]) * (np.cos((((np.cos((((((data["signal_shift_-1"]) * (np.where(((data["signal_shift_-1_msignal"]) - (data["maxbatch_msignal"])) > -998, data["signal_shift_+1"], ((np.cos((((((data["signal_shift_-1"]) * (np.where(data["maxbatch_msignal"] > -998, data["maxbatch_msignal"], data["maxbatch_msignal"] )))) / 2.0)))) - (data["maxbatch_msignal"])) )))) / 2.0)))) - (data["maxbatch_msignal"]))))))) +

                            0.100000*np.tanh(np.where((1.0) <= -998, np.cos((data["meanbatch_msignal"])), np.cos(((((((data["minbatch_msignal"]) + (np.where((3.0) > -998, data["abs_maxbatch_slices2_msignal"], np.sin((np.tanh((data["meanbatch_slices2_msignal"])))) )))) + (np.tanh(((((-((data["stdbatch_slices2"])))) - ((3.0)))))))/2.0))) )) +

                            0.100000*np.tanh(((((data["abs_avgbatch_slices2_msignal"]) * 2.0)) + (((np.where(data["medianbatch_slices2_msignal"] > -998, np.sin((((data["medianbatch_msignal"]) * 2.0))), np.where(data["abs_avgbatch_slices2_msignal"] > -998, (0.0), ((((data["rangebatch_msignal"]) * 2.0)) + (data["rangebatch_slices2"])) ) )) * (((((((data["rangebatch_msignal"]) * 2.0)) + (((data["medianbatch_msignal"]) * 2.0)))) * 2.0)))))) +

                            0.100000*np.tanh(((((np.sin((np.where(data["mean_abs_chgbatch_slices2"] <= -998, data["abs_maxbatch"], ((((data["mean_abs_chgbatch_slices2"]) * 2.0)) * 2.0) )))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.where(((((data["abs_minbatch_msignal"]) * (data["rangebatch_slices2"]))) + (data["maxbatch_slices2"])) <= -998, ((data["maxbatch_slices2"]) + (((data["abs_minbatch_msignal"]) + (((data["abs_minbatch_msignal"]) * (data["minbatch_slices2"])))))), ((data["maxbatch_slices2"]) + ((((3.0)) + (((data["abs_minbatch_msignal"]) * (data["rangebatch_slices2"])))))) )) +

                            0.100000*np.tanh(np.cos((((data["minbatch"]) * (np.where(((((np.tanh((((np.tanh((data["abs_maxbatch_slices2"]))) * 2.0)))) / 2.0)) / 2.0) <= -998, ((data["minbatch"]) * (np.where(data["medianbatch_slices2_msignal"] <= -998, ((np.tanh((((np.tanh((data["medianbatch_slices2_msignal"]))) * 2.0)))) / 2.0), np.cos(((2.0))) ))), data["signal"] )))))) +

                            0.100000*np.tanh(((np.cos((np.where(data["maxbatch_msignal"] > -998, ((np.where((-1.0) > -998, data["maxbatch_slices2_msignal"], np.cos((((((np.where((-((data["mean_abs_chgbatch_msignal"]))) > -998, data["maxbatch_slices2_msignal"], data["maxtominbatch_slices2_msignal"] )) * 2.0)) * 2.0))) )) * 2.0), (-((data["stdbatch_slices2"]))) )))) * 2.0)) +

                            0.100000*np.tanh(np.sin((((np.where(data["maxtominbatch_msignal"] > -998, data["medianbatch_msignal"], ((data["medianbatch_msignal"]) - (((np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["medianbatch_msignal"], np.sin((np.where((-3.0) > -998, (((np.where(data["medianbatch_msignal"] <= -998, data["abs_avgbatch_slices2"], (-(((3.0)))) )) + (data["medianbatch_msignal"]))/2.0), data["abs_maxbatch_msignal"] ))) )) * 2.0))) )) * 2.0)))) +

                            0.100000*np.tanh(np.where(((((data["meanbatch_slices2"]) * (data["mean_abs_chgbatch_msignal"]))) * (data["mean_abs_chgbatch_msignal"])) <= -998, data["maxtominbatch_slices2"], ((np.sin((((((data["minbatch_msignal"]) - (np.where(((data["minbatch_msignal"]) - (np.where(data["mean_abs_chgbatch_msignal"] <= -998, ((data["medianbatch_slices2_msignal"]) * 2.0), data["mean_abs_chgbatch_msignal"] ))) <= -998, data["mean_abs_chgbatch_msignal"], ((data["medianbatch_msignal"]) * 2.0) )))) * 2.0)))) * 2.0) )) +

                            0.100000*np.tanh(((np.sin(((-((((data["abs_maxbatch"]) + (np.cos((np.tanh(((((((data["abs_avgbatch_slices2_msignal"]) + (data["maxbatch_msignal"]))/2.0)) - (np.tanh((data["stdbatch_slices2"])))))))))))))))) * 2.0)) +

                            0.100000*np.tanh(np.where(np.where((((np.tanh((data["meanbatch_slices2_msignal"]))) + (data["abs_maxbatch_slices2_msignal"]))/2.0) <= -998, data["rangebatch_slices2"], (((((np.cos(((((data["mean_abs_chgbatch_slices2"]) + (data["abs_maxbatch_slices2_msignal"]))/2.0)))) + (data["abs_maxbatch_slices2_msignal"]))) + (data["meanbatch_slices2_msignal"]))/2.0) ) <= -998, data["meanbatch_slices2_msignal"], np.cos((data["maxbatch_msignal"])) )) +

                            0.100000*np.tanh((((3.0)) * (np.cos((((((np.sin((data["medianbatch_msignal"]))) + (data["minbatch_msignal"]))) / 2.0)))))) +

                            0.100000*np.tanh(np.sin((((((data["signal_shift_+1"]) + (np.cos((data["minbatch_slices2"]))))) * 2.0)))) +

                            0.100000*np.tanh((((((((3.0)) + (data["minbatch"]))) - (np.sin((data["rangebatch_msignal"]))))) * ((((3.0)) + (np.sin(((-((np.cos((((data["minbatch_slices2"]) * (data["signal"])))))))))))))) +

                            0.100000*np.tanh(np.sin((np.sin(((((data["maxbatch_msignal"]) + (np.where(np.sin((data["abs_minbatch_msignal"])) > -998, np.sin(((-2.0))), np.sin(((((np.sin((data["maxbatch_slices2"]))) + (np.where(data["signal_shift_+1_msignal"] <= -998, (((11.92518615722656250)) * (np.sin(((11.92518615722656250))))), (((((np.cos((data["abs_maxbatch_slices2_msignal"]))) / 2.0)) + ((-1.0)))/2.0) )))/2.0))) )))/2.0)))))) +

                            0.100000*np.tanh(((np.cos((np.where((((data["abs_minbatch_slices2"]) + (((np.cos(((((data["maxtominbatch_slices2_msignal"]) + (data["rangebatch_slices2_msignal"]))/2.0)))) * ((-3.0)))))/2.0) > -998, ((data["rangebatch_slices2_msignal"]) * 2.0), ((((data["abs_minbatch_slices2"]) * 2.0)) + (np.cos((((data["mean_abs_chgbatch_msignal"]) * 2.0))))) )))) * 2.0)) +

                            0.100000*np.tanh(((((((((((((np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["maxtominbatch_msignal"], np.tanh((data["maxtominbatch_msignal"])) )) - (data["rangebatch_msignal"]))) + ((6.0)))/2.0)) + (np.sin(((6.0)))))) - (data["rangebatch_msignal"]))) + ((6.0)))/2.0)) +

                            0.100000*np.tanh(data["medianbatch_msignal"]) +

                            0.100000*np.tanh((((np.where(data["stdbatch_msignal"] > -998, np.sin(((((1.0)) * (data["abs_avgbatch_slices2_msignal"])))), (-((np.where(np.sin((((np.tanh((data["mean_abs_chgbatch_slices2_msignal"]))) / 2.0))) <= -998, ((data["stdbatch_msignal"]) / 2.0), data["minbatch_slices2"] )))) )) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)) +

                            0.100000*np.tanh(((data["minbatch_slices2_msignal"]) + (((((8.0)) + (((data["minbatch_slices2_msignal"]) + (((((8.0)) + (np.tanh((np.where(data["stdbatch_slices2_msignal"] > -998, np.where((8.0) <= -998, data["minbatch_slices2_msignal"], (8.0) ), (8.0) )))))/2.0)))))/2.0)))) +

                            0.100000*np.tanh(np.sin((((data["abs_minbatch_slices2_msignal"]) + (((np.where(data["rangebatch_slices2_msignal"] <= -998, (((-3.0)) - (((data["minbatch_slices2"]) - (np.sin((((data["abs_avgbatch_msignal"]) - (np.sin(((-((data["signal"]))))))))))))), data["rangebatch_msignal"] )) + (data["meanbatch_slices2"]))))))) +

                            0.100000*np.tanh((-((((np.cos((np.where(data["abs_minbatch_slices2"] > -998, data["stdbatch_slices2_msignal"], data["maxtominbatch_slices2_msignal"] )))) * 2.0))))) +

                            0.100000*np.tanh((((((((4.0)) + ((((((data["minbatch_slices2_msignal"]) * 2.0)) + (np.where((-(((((4.0)) + ((((((data["minbatch_slices2_msignal"]) * 2.0)) + ((4.0)))/2.0)))))) <= -998, np.sin(((-((np.sin(((-((data["minbatch_slices2_msignal"])))))))))), np.cos(((4.0))) )))/2.0)))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.where(np.sin((np.sin((data["signal_shift_-1"])))) <= -998, np.sin((np.sin(((((-1.0)) * (np.sin((((np.sin((np.cos((data["signal_shift_-1"]))))) * 2.0))))))))), ((np.sin((((np.sin((((data["abs_maxbatch"]) * (np.sin((data["signal_shift_-1"]))))))) * 2.0)))) * 2.0) )) +

                            0.100000*np.tanh(((data["maxbatch_slices2_msignal"]) * (np.cos((np.where(data["maxbatch_slices2_msignal"] <= -998, (-((((data["maxbatch_slices2_msignal"]) * 2.0)))), ((data["maxbatch_slices2_msignal"]) * 2.0) )))))) +

                            0.100000*np.tanh(np.tanh((np.where(np.tanh((((np.cos((np.where((-1.0) <= -998, data["mean_abs_chgbatch_slices2_msignal"], data["maxbatch_msignal"] )))) / 2.0))) <= -998, data["abs_maxbatch_slices2_msignal"], np.sin((((data["medianbatch_msignal"]) * 2.0))) )))) +

                            0.100000*np.tanh((-((np.sin((np.where(np.cos((data["rangebatch_slices2"])) > -998, data["minbatch_msignal"], np.cos((np.cos((np.where(data["maxtominbatch"] > -998, data["abs_minbatch_slices2"], ((data["signal_shift_-1_msignal"]) / 2.0) ))))) ))))))) +

                            0.100000*np.tanh(((((np.where(data["medianbatch_slices2_msignal"] <= -998, data["signal"], ((np.where(data["signal_shift_-1_msignal"] <= -998, np.where(data["maxbatch_slices2_msignal"] <= -998, data["minbatch_slices2"], (-2.0) ), ((data["meanbatch_msignal"]) - (((np.cos((((data["signal_shift_-1_msignal"]) - (data["minbatch_msignal"]))))) * 2.0))) )) - (data["signal_shift_-1_msignal"])) )) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.sin((((np.where(((data["minbatch"]) * 2.0) <= -998, ((data["medianbatch_slices2"]) * 2.0), ((((2.0)) + (data["minbatch"]))/2.0) )) * 2.0)))) +

                            0.100000*np.tanh(np.where(np.tanh((((data["abs_maxbatch_msignal"]) - (np.where(data["signal_shift_-1_msignal"] <= -998, data["medianbatch_msignal"], data["signal_shift_-1_msignal"] ))))) <= -998, data["abs_maxbatch_msignal"], np.cos((((data["abs_maxbatch_msignal"]) - (np.where(data["abs_maxbatch_msignal"] <= -998, np.tanh((data["signal_shift_+1"])), data["signal_shift_-1_msignal"] ))))) )) +

                            0.100000*np.tanh(((data["abs_maxbatch"]) * (((((data["maxbatch_slices2_msignal"]) / 2.0)) * (np.sin((np.where((((np.cos((data["minbatch_slices2"]))) + (((data["abs_maxbatch"]) / 2.0)))/2.0) > -998, ((data["abs_maxbatch_slices2_msignal"]) / 2.0), ((data["abs_avgbatch_msignal"]) * (data["medianbatch_msignal"])) )))))))) +

                            0.100000*np.tanh(((np.where(data["abs_avgbatch_msignal"] > -998, data["abs_maxbatch"], data["maxbatch_slices2_msignal"] )) + (((np.where(data["maxbatch_msignal"] > -998, ((data["maxbatch_msignal"]) * (np.sin((((data["maxbatch_slices2_msignal"]) * ((-3.0))))))), (2.0) )) * (data["abs_maxbatch"]))))) +

                            0.100000*np.tanh(((data["maxbatch_slices2_msignal"]) - (((data["signal_shift_+1"]) + (np.cos((np.where(np.cos((data["maxbatch_msignal"])) > -998, np.where(data["mean_abs_chgbatch_slices2"] > -998, (((-3.0)) - (data["abs_maxbatch_slices2"])), data["stdbatch_slices2"] ), np.where(data["abs_minbatch_slices2_msignal"] > -998, data["abs_avgbatch_msignal"], data["medianbatch_slices2_msignal"] ) )))))))) +

                            0.100000*np.tanh(((((np.sin((np.where(np.where(((data["maxbatch_slices2_msignal"]) * 2.0) > -998, ((data["maxbatch_slices2_msignal"]) * 2.0), ((data["abs_avgbatch_slices2"]) + ((-1.0))) ) > -998, ((data["maxbatch_slices2_msignal"]) * 2.0), data["signal"] )))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.where(((data["maxbatch_msignal"]) * 2.0) <= -998, data["abs_avgbatch_msignal"], ((data["abs_avgbatch_slices2_msignal"]) * (((data["rangebatch_slices2_msignal"]) * (np.where(((data["maxbatch_msignal"]) * 2.0) <= -998, data["medianbatch_msignal"], np.tanh((np.cos((np.where(np.tanh((np.where((-1.0) > -998, data["abs_minbatch_msignal"], ((data["maxbatch_msignal"]) * 2.0) ))) <= -998, data["medianbatch_msignal"], ((data["maxbatch_msignal"]) * 2.0) ))))) ))))) )) +

                            0.100000*np.tanh(np.cos((np.where(data["signal_shift_+1"] > -998, data["stdbatch_slices2"], (-((((np.tanh((np.cos((data["maxbatch_msignal"]))))) * 2.0)))) )))) +

                            0.100000*np.tanh(((np.where(data["stdbatch_slices2_msignal"] <= -998, np.tanh((data["maxbatch_slices2_msignal"])), np.sin((((np.tanh((np.cos((data["maxbatch_slices2_msignal"]))))) + (((np.where(np.cos((data["abs_maxbatch_msignal"])) <= -998, (-((data["mean_abs_chgbatch_slices2"]))), ((data["medianbatch_msignal"]) - ((((0.0)) / 2.0))) )) * 2.0))))) )) * 2.0)) +

                            0.100000*np.tanh(np.cos((np.where((((((np.cos((data["maxbatch_msignal"]))) + ((0.0)))) + (np.cos((np.where(data["abs_maxbatch_msignal"] > -998, data["maxbatch_slices2"], np.tanh((data["rangebatch_slices2"])) )))))/2.0) > -998, np.where(data["meanbatch_slices2_msignal"] > -998, data["maxbatch_msignal"], (((((data["rangebatch_slices2_msignal"]) + (data["maxbatch_msignal"]))/2.0)) + ((9.0))) ), data["rangebatch_slices2"] )))) +

                            0.100000*np.tanh(((np.where(np.cos((data["signal_shift_+1_msignal"])) > -998, data["abs_minbatch_msignal"], np.sin((np.tanh(((-3.0))))) )) * (np.where(np.cos((data["minbatch"])) > -998, np.cos((np.cos((((data["medianbatch_msignal"]) / 2.0))))), np.cos((data["medianbatch_slices2_msignal"])) )))) +

                            0.100000*np.tanh(np.where(data["signal_shift_-1"] > -998, (((data["abs_maxbatch_slices2"]) + (data["medianbatch_slices2"]))/2.0), data["medianbatch_slices2"] )) +

                            0.100000*np.tanh(np.where(((data["rangebatch_msignal"]) + ((2.0))) > -998, (((10.0)) * (((data["maxbatch_slices2_msignal"]) * (np.sin(((-((((data["rangebatch_msignal"]) + (data["signal_shift_-1"]))))))))))), ((data["rangebatch_msignal"]) + (data["signal_shift_+1"])) )) +

                            0.100000*np.tanh(((((np.where(data["signal_shift_-1_msignal"] <= -998, (-((data["abs_maxbatch"]))), np.cos((np.where(data["signal_shift_+1_msignal"] <= -998, np.cos((data["meanbatch_slices2_msignal"])), ((data["signal_shift_+1_msignal"]) - (data["maxbatch_msignal"])) ))) )) * 2.0)) * 2.0)) +

                            0.100000*np.tanh((-((np.sin((((data["minbatch_msignal"]) + (((np.tanh((((np.cos((np.sin((((np.where(data["mean_abs_chgbatch_msignal"] > -998, data["medianbatch_slices2"], data["abs_avgbatch_slices2_msignal"] )) * 2.0)))))) - (data["signal_shift_-1_msignal"]))))) * ((((1.0)) * 2.0))))))))))) +

                            0.100000*np.tanh(np.sin((((data["rangebatch_slices2"]) * ((((data["medianbatch_slices2_msignal"]) + ((-(((((-1.0)) - (np.tanh((np.where(data["stdbatch_msignal"] <= -998, ((np.sin((((data["meanbatch_slices2"]) + (data["medianbatch_slices2_msignal"]))))) / 2.0), np.sin((((data["rangebatch_slices2"]) / 2.0))) ))))))))))/2.0)))))) +

                            0.100000*np.tanh(((np.cos((((data["meanbatch_slices2_msignal"]) * (np.sin((np.sin((np.cos((((data["medianbatch_msignal"]) / 2.0)))))))))))) + (np.cos((np.tanh((np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) + (np.cos(((((data["abs_maxbatch_slices2_msignal"]) + ((((1.0)) * (np.tanh((np.tanh((((np.cos(((3.0)))) / 2.0)))))))))/2.0)))))))))))))) +

                            0.100000*np.tanh(((np.cos((((((data["minbatch_slices2"]) * ((5.0)))) + (np.cos((((data["abs_avgbatch_slices2_msignal"]) - (np.cos((np.where(data["stdbatch_slices2_msignal"] <= -998, np.cos(((((data["stdbatch_slices2_msignal"]) + (np.cos((((data["abs_maxbatch_msignal"]) / 2.0)))))/2.0))), np.cos(((((data["stdbatch_slices2_msignal"]) + (np.cos((((data["abs_maxbatch_msignal"]) / 2.0)))))/2.0))) )))))))))))) * 2.0)) +

                            0.100000*np.tanh((((8.15774059295654297)) * (((np.sin((data["minbatch_slices2_msignal"]))) * (((np.cos((((data["abs_maxbatch"]) * (np.where(np.where(data["abs_maxbatch"] <= -998, (1.0), data["abs_maxbatch"] ) > -998, (8.15774059295654297), np.where(data["abs_maxbatch"] > -998, data["abs_maxbatch_msignal"], data["abs_maxbatch"] ) )))))) * 2.0)))))) +

                            0.100000*np.tanh((((-((np.sin((((((-((data["maxtominbatch_slices2"])))) + (np.cos(((-2.0)))))/2.0))))))) / 2.0)) +

                            0.100000*np.tanh(((np.tanh((np.cos((((((((1.0)) + (data["minbatch_msignal"]))) + (np.where(data["abs_maxbatch"] > -998, data["abs_avgbatch_slices2_msignal"], ((((np.tanh((np.cos(((((data["minbatch_slices2_msignal"]) + (np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], (((data["maxbatch_msignal"]) + ((3.0)))/2.0) )))/2.0)))))) * 2.0)) / 2.0) )))/2.0)))))) * 2.0)) +

                            0.100000*np.tanh((((((14.40959262847900391)) / 2.0)) * 2.0)) +

                            0.100000*np.tanh(((data["abs_avgbatch_slices2_msignal"]) * (np.cos((((((data["medianbatch_msignal"]) + (np.where((1.0) > -998, data["medianbatch_msignal"], np.where((1.0) > -998, data["medianbatch_msignal"], ((data["meanbatch_msignal"]) / 2.0) ) )))) * 2.0)))))) +

                            0.100000*np.tanh(np.where(data["minbatch_slices2_msignal"] <= -998, data["medianbatch_msignal"], np.tanh((np.sin((((np.tanh((np.sin((((np.sin((((np.where((((-((data["maxtominbatch_slices2"])))) * 2.0) > -998, data["abs_maxbatch_slices2_msignal"], (-((data["medianbatch_msignal"]))) )) / 2.0)))) / 2.0)))))) - (np.tanh((data["minbatch_slices2_msignal"])))))))) )) +

                            0.100000*np.tanh(np.cos((((((np.where(data["maxtominbatch_slices2_msignal"] <= -998, ((np.cos((data["abs_minbatch_slices2_msignal"]))) + (data["abs_minbatch_slices2_msignal"])), np.cos((((data["abs_minbatch_msignal"]) * (data["abs_minbatch_slices2_msignal"])))) )) - (((data["abs_minbatch_slices2_msignal"]) * ((((data["abs_minbatch_slices2_msignal"]) + (((data["medianbatch_slices2"]) / 2.0)))/2.0)))))) / 2.0)))) +

                            0.100000*np.tanh((-(((((data["signal_shift_-1"]) + ((((np.where((((0.0)) / 2.0) > -998, data["maxtominbatch_slices2_msignal"], np.tanh((np.cos((np.cos((data["signal_shift_-1"])))))) )) + (data["maxbatch_slices2"]))/2.0)))/2.0))))) +

                            0.100000*np.tanh((((7.0)) - (np.where(data["maxbatch_slices2_msignal"] > -998, data["maxbatch_slices2_msignal"], data["maxbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh(((((np.where(data["medianbatch_slices2"] > -998, data["medianbatch_msignal"], (((0.0)) - (data["abs_minbatch_slices2"])) )) + (((data["medianbatch_slices2"]) - (np.where((2.0) <= -998, data["maxbatch_msignal"], np.tanh((data["meanbatch_slices2"])) )))))) + (data["mean_abs_chgbatch_slices2"]))) +

                            0.100000*np.tanh(((((((np.sin((data["maxtominbatch"]))) * 2.0)) / 2.0)) / 2.0)) +

                            0.100000*np.tanh(((((np.sin((np.where(data["signal_shift_-1_msignal"] <= -998, data["medianbatch_slices2"], ((((data["minbatch_slices2_msignal"]) - ((((np.tanh((np.sin((data["signal_shift_-1_msignal"]))))) + ((((data["stdbatch_msignal"]) + (np.where(data["signal_shift_-1_msignal"] <= -998, data["minbatch_slices2_msignal"], data["medianbatch_slices2"] )))/2.0)))/2.0)))) * 2.0) )))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.where(((np.tanh((np.cos(((2.0)))))) - (((np.cos(((2.0)))) * 2.0))) > -998, ((np.cos((((np.sin(((-1.0)))) * ((((-((data["minbatch_msignal"])))) / 2.0)))))) * 2.0), (((-1.0)) - (data["minbatch_slices2_msignal"])) )) +

                            0.100000*np.tanh(((np.where(np.cos((((data["medianbatch_slices2_msignal"]) - ((1.0))))) > -998, ((np.cos(((((data["medianbatch_slices2_msignal"]) + ((1.0)))/2.0)))) * 2.0), (1.0) )) * (((data["maxtominbatch_msignal"]) * (((data["maxtominbatch_slices2_msignal"]) - (data["meanbatch_slices2"]))))))) +

                            0.100000*np.tanh(((((np.sin((data["abs_maxbatch"]))) * 2.0)) * (((data["maxtominbatch"]) - (data["rangebatch_msignal"]))))) +

                            0.100000*np.tanh(np.cos((((((np.where(np.where(np.sin((data["minbatch"])) > -998, data["medianbatch_msignal"], ((data["medianbatch_msignal"]) * ((2.88659405708312988))) ) > -998, data["maxbatch_slices2_msignal"], ((np.cos((np.where(data["maxbatch_slices2_msignal"] > -998, np.cos((data["maxbatch_slices2_msignal"])), data["stdbatch_slices2"] )))) - (np.tanh((data["signal_shift_-1_msignal"])))) )) * 2.0)) - (data["signal_shift_-1_msignal"]))))) +

                            0.100000*np.tanh(((np.cos((data["meanbatch_msignal"]))) + (((((((np.tanh((np.cos((np.tanh((np.cos((data["signal"]))))))))) / 2.0)) * 2.0)) + (((((data["stdbatch_slices2_msignal"]) / 2.0)) / 2.0)))))) +

                            0.100000*np.tanh(np.cos(((((np.cos((np.where(((data["maxbatch_slices2_msignal"]) * 2.0) <= -998, data["minbatch_slices2_msignal"], data["minbatch_slices2"] )))) + ((-((data["meanbatch_msignal"])))))/2.0)))) +

                            0.100000*np.tanh(np.cos((np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["abs_minbatch_slices2_msignal"], np.sin((np.sin((np.sin((((data["rangebatch_slices2_msignal"]) * 2.0))))))) )))) +

                            0.100000*np.tanh(np.where(data["abs_minbatch_slices2"] > -998, np.where(np.cos((np.sin((data["mean_abs_chgbatch_slices2_msignal"])))) > -998, np.cos((((np.tanh((((np.sin((np.where(((data["signal"]) / 2.0) <= -998, np.cos((((data["signal_shift_+1_msignal"]) / 2.0))), data["rangebatch_slices2"] )))) - (data["signal_shift_+1_msignal"]))))) * 2.0))), (2.0) ), (2.0) )) +

                            0.100000*np.tanh(((((3.0)) + (np.sin((((data["meanbatch_msignal"]) / 2.0)))))/2.0)) +

                            0.100000*np.tanh(np.tanh((((((2.0)) + (np.where(np.where(((((10.0)) + (data["meanbatch_msignal"]))/2.0) > -998, data["minbatch"], data["maxbatch_msignal"] ) > -998, data["minbatch"], np.where(np.sin((data["minbatch"])) <= -998, data["maxtominbatch_msignal"], np.cos((((((2.0)) + (data["minbatch_slices2_msignal"]))/2.0))) ) )))/2.0)))) +

                            0.099609*np.tanh(np.where(np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, (6.0), data["maxtominbatch_slices2_msignal"] ) > -998, np.where(np.where(data["abs_minbatch_msignal"] > -998, data["stdbatch_msignal"], (((np.cos((data["maxtominbatch_msignal"]))) + (data["abs_minbatch_msignal"]))/2.0) ) > -998, data["stdbatch_msignal"], data["abs_minbatch_msignal"] ), ((data["stdbatch_slices2_msignal"]) * (data["stdbatch_msignal"])) )) +

                            0.100000*np.tanh(((np.sin((np.cos((data["minbatch"]))))) * 2.0)) +

                            0.100000*np.tanh(np.where((-((data["maxtominbatch_msignal"]))) <= -998, data["maxtominbatch_msignal"], ((np.tanh((((((np.tanh((np.cos((((np.where((3.0) <= -998, data["abs_maxbatch"], data["maxtominbatch_slices2_msignal"] )) / 2.0)))))) - (np.cos((data["abs_maxbatch"]))))) / 2.0)))) - (np.cos((data["abs_maxbatch"])))) )) +

                            0.100000*np.tanh(np.cos(((((np.cos((data["rangebatch_slices2"]))) + (((((((((data["mean_abs_chgbatch_slices2"]) * 2.0)) * 2.0)) * (np.where((1.0) > -998, ((data["medianbatch_slices2_msignal"]) + (data["stdbatch_slices2"])), np.where((((((data["mean_abs_chgbatch_slices2"]) * 2.0)) + (data["mean_abs_chgbatch_slices2"]))/2.0) > -998, data["rangebatch_slices2"], data["rangebatch_slices2"] ) )))) * 2.0)))/2.0)))) +

                            0.100000*np.tanh(np.where((((9.33542633056640625)) * 2.0) > -998, np.tanh((data["abs_minbatch_slices2"])), (2.0) )) +

                            0.100000*np.tanh((((np.tanh((np.where(data["abs_maxbatch_slices2"] > -998, (3.0), (((np.where(data["maxbatch_msignal"] <= -998, data["rangebatch_slices2_msignal"], data["signal_shift_-1"] )) + (np.cos((data["mean_abs_chgbatch_slices2_msignal"]))))/2.0) )))) + (data["abs_maxbatch"]))/2.0)) +

                            0.100000*np.tanh(np.cos((((((np.where(np.cos((((((data["signal_shift_+1"]) * 2.0)) * 2.0))) > -998, data["signal_shift_+1"], np.sin((np.cos((data["minbatch_msignal"])))) )) * 2.0)) * 2.0)))) +

                            0.100000*np.tanh(np.where(np.where(data["maxbatch_slices2_msignal"] <= -998, np.cos((np.cos((data["signal_shift_-1_msignal"])))), np.where(data["minbatch"] > -998, np.cos((data["abs_minbatch_slices2_msignal"])), np.cos((data["abs_minbatch_slices2_msignal"])) ) ) > -998, np.cos((np.cos((data["abs_minbatch_slices2_msignal"])))), np.cos((((np.tanh((data["medianbatch_slices2_msignal"]))) / 2.0))) )) +

                            0.100000*np.tanh(np.cos(((((-((np.where(((np.cos((data["signal"]))) / 2.0) > -998, data["abs_maxbatch_slices2_msignal"], np.where(np.cos((np.sin((((data["signal"]) * (data["rangebatch_slices2"])))))) > -998, (-((data["medianbatch_msignal"]))), (2.0) ) ))))) * 2.0)))) +

                            0.100000*np.tanh(((np.cos((((((np.cos((((data["signal_shift_+1_msignal"]) / 2.0)))) - (data["maxbatch_slices2_msignal"]))) * 2.0)))) - (np.sin(((-((np.where(data["minbatch_slices2_msignal"] <= -998, data["stdbatch_msignal"], np.cos((((((((data["signal_shift_-1_msignal"]) * 2.0)) - (data["maxbatch_slices2_msignal"]))) * 2.0))) ))))))))) +

                            0.100000*np.tanh(np.where(np.sin((((data["abs_minbatch_slices2_msignal"]) + (((((np.tanh((data["abs_maxbatch_slices2"]))) / 2.0)) * (data["abs_minbatch_slices2_msignal"])))))) > -998, ((data["maxbatch_slices2"]) + (data["maxbatch_slices2_msignal"])), np.cos((np.where(((data["signal_shift_-1_msignal"]) * 2.0) <= -998, (-3.0), data["meanbatch_msignal"] ))) )) +

                            0.089932*np.tanh(np.sin((((np.sin((data["stdbatch_msignal"]))) * 2.0)))) +

                            0.100000*np.tanh((((((3.0)) - (((data["signal"]) / 2.0)))) + (((data["stdbatch_slices2"]) * (np.where(np.where((((data["meanbatch_msignal"]) + ((1.0)))/2.0) <= -998, data["minbatch"], data["abs_maxbatch_slices2"] ) > -998, data["minbatch"], ((data["minbatch"]) - (((data["signal"]) / 2.0))) )))))) +

                            0.098436*np.tanh(np.sin(((-((((data["abs_avgbatch_msignal"]) - (np.where(np.where(np.tanh((np.sin((((data["stdbatch_slices2"]) * 2.0))))) <= -998, data["mean_abs_chgbatch_slices2"], data["signal_shift_+1"] ) <= -998, data["signal_shift_+1"], np.where(data["rangebatch_slices2_msignal"] <= -998, data["maxbatch_slices2_msignal"], data["signal_shift_+1"] ) ))))))))) +

                            0.100000*np.tanh(np.cos((np.sin((((((((data["signal_shift_-1"]) + (data["medianbatch_slices2_msignal"]))) + ((-((data["abs_avgbatch_msignal"])))))) / 2.0)))))) +

                            0.100000*np.tanh(np.where((((((data["maxtominbatch_msignal"]) + (data["signal_shift_-1_msignal"]))/2.0)) / 2.0) <= -998, np.where(data["signal"] <= -998, (((-3.0)) * (data["medianbatch_msignal"])), ((data["signal"]) - (data["medianbatch_msignal"])) ), ((data["medianbatch_msignal"]) - (data["signal_shift_+1_msignal"])) )) +

                            0.100000*np.tanh(np.sin((((np.sin((np.cos((((np.where((8.33993053436279297) <= -998, (1.0), data["mean_abs_chgbatch_slices2"] )) / 2.0)))))) / 2.0)))) +

                            0.100000*np.tanh(((np.cos((((data["maxtominbatch_slices2_msignal"]) + (np.where(((np.cos((((data["maxtominbatch_slices2_msignal"]) + (data["signal_shift_+1_msignal"]))))) * 2.0) > -998, data["abs_maxbatch_msignal"], (2.53737401962280273) )))))) * 2.0)) +

                            0.100000*np.tanh(np.sin((((np.where(np.sin(((((((-2.0)) / 2.0)) / 2.0))) <= -998, np.sin((np.cos(((0.0))))), data["abs_maxbatch_slices2_msignal"] )) / 2.0)))) +

                            0.100000*np.tanh(np.where(np.where((-((np.sin((((np.where(data["maxbatch_slices2_msignal"] <= -998, np.tanh(((2.0))), np.tanh((((data["abs_minbatch_slices2_msignal"]) * (data["maxbatch_msignal"])))) )) * (((data["medianbatch_msignal"]) * 2.0)))))))) > -998, ((data["maxbatch_slices2_msignal"]) / 2.0), data["stdbatch_msignal"] ) <= -998, data["signal_shift_-1"], data["maxbatch_slices2_msignal"] )) +

                            0.100000*np.tanh((-((np.cos(((-((((data["maxbatch_slices2_msignal"]) + ((-((np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["minbatch_slices2_msignal"], (-((np.where(data["abs_maxbatch"] <= -998, ((data["signal_shift_-1_msignal"]) / 2.0), ((data["maxbatch_slices2_msignal"]) + ((((data["abs_minbatch_slices2"]) + (((data["rangebatch_slices2"]) + (((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0)))))/2.0))) )))) ))))))))))))))) +

                            0.093744*np.tanh(np.tanh((np.where(((data["maxbatch_msignal"]) - (((data["abs_maxbatch_slices2_msignal"]) / 2.0))) <= -998, np.cos(((9.28210449218750000))), (-((data["minbatch_msignal"]))) )))) +

                            0.100000*np.tanh(((((data["abs_minbatch_slices2_msignal"]) + (np.cos((np.where((((((data["signal_shift_-1_msignal"]) * 2.0)) + (np.where(np.cos((data["minbatch_slices2"])) <= -998, data["abs_avgbatch_msignal"], np.cos((data["signal_shift_-1_msignal"])) )))/2.0) <= -998, data["signal_shift_+1_msignal"], data["minbatch_slices2"] )))))) * (data["signal_shift_-1_msignal"]))) +

                            0.099707*np.tanh(((np.where(np.sin((np.sin((np.tanh((np.tanh((np.tanh((data["signal_shift_+1"])))))))))) <= -998, (((((data["medianbatch_msignal"]) * 2.0)) + (np.cos((data["maxbatch_slices2"]))))/2.0), data["medianbatch_msignal"] )) + (data["signal_shift_+1_msignal"]))) +

                            0.100000*np.tanh(np.cos((((data["minbatch_slices2_msignal"]) * ((((3.0)) - ((-((np.where(np.sin((data["minbatch_slices2_msignal"])) <= -998, np.where(data["maxbatch_slices2"] > -998, data["maxbatch_slices2"], (3.0) ), np.tanh((data["signal_shift_+1_msignal"])) ))))))))))) +

                            0.100000*np.tanh((((2.0)) / 2.0)) +

                            0.100000*np.tanh(np.sin((np.where((-3.0) <= -998, np.tanh((data["maxtominbatch"])), np.where((-((((np.tanh((np.sin((data["meanbatch_msignal"]))))) * 2.0)))) <= -998, data["abs_maxbatch_slices2_msignal"], np.sin((np.where(np.cos((np.tanh((np.sin((np.sin((data["mean_abs_chgbatch_msignal"])))))))) <= -998, np.sin((np.sin((((data["signal_shift_-1_msignal"]) * 2.0))))), data["mean_abs_chgbatch_msignal"] ))) ) )))))  

    

    def GP_class_2(self,data):

        return self.Output( -2.199090 +

                            0.100000*np.tanh(((((((np.sin((np.sin((((np.sin((data["abs_avgbatch_msignal"]))) * 2.0)))))) * 2.0)) + (data["meanbatch_msignal"]))) - (np.sin((np.tanh(((-((np.sin((((np.sin((data["abs_avgbatch_msignal"]))) * 2.0))))))))))))) +

                            0.100000*np.tanh(np.sin((np.where((-((data["medianbatch_slices2"]))) <= -998, ((data["mean_abs_chgbatch_slices2_msignal"]) + (((data["signal_shift_+1"]) * 2.0))), data["abs_avgbatch_msignal"] )))) +

                            0.100000*np.tanh(((((np.sin((np.where(np.sin((data["abs_avgbatch_slices2_msignal"])) > -998, data["abs_avgbatch_msignal"], np.where(data["maxtominbatch_msignal"] > -998, (1.0), ((data["maxbatch_slices2"]) / 2.0) ) )))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(((np.where((-((np.sin((np.sin((data["abs_avgbatch_slices2_msignal"]))))))) <= -998, np.sin((data["abs_avgbatch_slices2_msignal"])), data["abs_maxbatch_slices2"] )) * (((np.where((-(((0.0)))) <= -998, data["abs_maxbatch_msignal"], data["rangebatch_slices2"] )) * ((((np.where(data["signal_shift_-1_msignal"] > -998, np.sin((data["abs_avgbatch_slices2_msignal"])), (-(((0.0)))) )) + (data["meanbatch_slices2_msignal"]))/2.0)))))) +

                            0.100000*np.tanh(np.where(np.sin((np.where((1.0) > -998, ((data["maxbatch_slices2"]) * 2.0), np.sin((data["maxbatch_slices2"])) ))) <= -998, data["minbatch_msignal"], np.where(data["meanbatch_msignal"] <= -998, data["minbatch_slices2"], ((np.cos((((np.where(data["medianbatch_slices2"] > -998, data["minbatch_msignal"], np.cos((((np.cos((((data["minbatch_msignal"]) * 2.0)))) * 2.0))) )) * 2.0)))) * 2.0) ) )) +

                            0.100000*np.tanh(((data["abs_avgbatch_msignal"]) * (data["maxbatch_slices2"]))) +

                            0.100000*np.tanh(((((np.where((-((data["meanbatch_msignal"]))) > -998, ((((data["meanbatch_msignal"]) * 2.0)) + (np.tanh((np.tanh((data["maxbatch_slices2"])))))), data["abs_avgbatch_msignal"] )) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.where(data["minbatch_slices2"] <= -998, ((np.cos((np.cos((np.where(data["maxtominbatch_msignal"] <= -998, ((data["stdbatch_slices2"]) / 2.0), (-((data["maxtominbatch_slices2_msignal"]))) )))))) * 2.0), ((np.where(((np.sin((data["meanbatch_slices2_msignal"]))) / 2.0) <= -998, data["minbatch"], ((np.sin((np.where(data["meanbatch_slices2"] > -998, data["minbatch_msignal"], data["minbatch"] )))) * 2.0) )) * 2.0) )) +

                            0.100000*np.tanh(((np.where(np.where(data["maxbatch_slices2"] > -998, data["rangebatch_msignal"], (((((-1.0)) + (data["medianbatch_slices2_msignal"]))) * ((-((((np.cos(((((-1.0)) + (data["medianbatch_slices2"]))))) * 2.0)))))) ) > -998, data["rangebatch_msignal"], np.sin((data["abs_avgbatch_msignal"])) )) * (np.sin((data["abs_avgbatch_msignal"]))))) +

                            0.100000*np.tanh(np.where(np.cos((data["abs_avgbatch_msignal"])) <= -998, ((data["abs_maxbatch_msignal"]) - (data["mean_abs_chgbatch_msignal"])), ((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)) * (data["maxbatch_slices2"])) )) +

                            0.100000*np.tanh(((np.cos(((-((data["stdbatch_slices2"])))))) * (((data["meanbatch_slices2"]) + (((np.where(((np.sin((data["stdbatch_slices2"]))) / 2.0) > -998, data["meanbatch_msignal"], (-((np.cos((((data["abs_minbatch_slices2_msignal"]) + (((((data["minbatch_slices2_msignal"]) * 2.0)) / 2.0)))))))) )) / 2.0)))))) +

                            0.100000*np.tanh(((data["maxbatch_slices2_msignal"]) * (((data["minbatch_msignal"]) * (np.sin(((((((((data["minbatch_slices2_msignal"]) + (data["rangebatch_slices2_msignal"]))/2.0)) * ((0.0)))) + (np.sin((((((np.cos(((0.0)))) * (np.sin((data["abs_avgbatch_msignal"]))))) + (((data["maxbatch_slices2_msignal"]) * 2.0)))))))))))))) +

                            0.100000*np.tanh(((np.cos((((((data["signal"]) * 2.0)) * (np.tanh((np.tanh((((((data["abs_minbatch_msignal"]) * 2.0)) * 2.0)))))))))) * 2.0)) +

                            0.100000*np.tanh(((data["meanbatch_msignal"]) - (np.cos((((data["minbatch_msignal"]) + (np.where(((data["abs_minbatch_msignal"]) - (data["abs_minbatch_msignal"])) > -998, (6.92890548706054688), np.where(data["minbatch_msignal"] <= -998, ((((data["meanbatch_msignal"]) - (data["abs_avgbatch_slices2"]))) * 2.0), data["abs_minbatch_slices2_msignal"] ) )))))))) +

                            0.100000*np.tanh(((((((data["abs_minbatch_slices2_msignal"]) + ((((((np.where(data["maxtominbatch_slices2_msignal"] > -998, (-3.0), (((3.0)) / 2.0) )) + (data["abs_avgbatch_slices2_msignal"]))/2.0)) / 2.0)))/2.0)) + (((data["stdbatch_msignal"]) + ((((data["medianbatch_slices2"]) + (((data["stdbatch_msignal"]) + (((np.cos((data["medianbatch_slices2"]))) * ((3.0)))))))/2.0)))))/2.0)) +

                            0.100000*np.tanh(((((((data["mean_abs_chgbatch_msignal"]) * (np.cos((np.where(np.cos((np.where(data["mean_abs_chgbatch_msignal"] <= -998, data["signal"], data["stdbatch_slices2_msignal"] ))) <= -998, ((data["maxtominbatch"]) + (data["signal_shift_-1"])), data["mean_abs_chgbatch_msignal"] )))))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(((data["minbatch_msignal"]) * (np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) + (np.where(data["maxtominbatch"] <= -998, (((3.0)) * (data["mean_abs_chgbatch_slices2_msignal"])), data["minbatch"] )))))))) +

                            0.100000*np.tanh(((data["maxbatch_slices2"]) * (np.where(data["meanbatch_slices2_msignal"] <= -998, np.where(((((((np.cos(((((((data["stdbatch_slices2"]) + (data["mean_abs_chgbatch_msignal"]))/2.0)) * (data["abs_maxbatch_slices2"]))))) * (np.tanh((data["maxbatch_slices2"]))))) * (data["abs_minbatch_slices2_msignal"]))) * 2.0) <= -998, data["maxbatch_slices2"], np.cos((data["minbatch"])) ), np.cos((data["stdbatch_slices2"])) )))) +

                            0.100000*np.tanh(((((data["maxbatch_slices2"]) * (((((((((data["abs_avgbatch_msignal"]) + (((data["abs_avgbatch_msignal"]) + (data["maxbatch_slices2"]))))) + (np.tanh((((data["meanbatch_slices2"]) / 2.0)))))) + (((data["meanbatch_msignal"]) * 2.0)))) * 2.0)))) + (np.where((1.0) > -998, data["maxbatch_slices2"], ((np.sin((data["maxbatch_slices2"]))) * (data["maxbatch_slices2"])) )))) +

                            0.100000*np.tanh(np.where(((((data["rangebatch_slices2"]) + (data["medianbatch_slices2_msignal"]))) * (data["medianbatch_slices2_msignal"])) <= -998, ((data["rangebatch_slices2"]) * 2.0), ((data["rangebatch_msignal"]) * (((data["medianbatch_slices2_msignal"]) - (np.sin((data["rangebatch_slices2"])))))) )) +

                            0.100000*np.tanh((((-((np.sin((np.where(np.sin((data["signal_shift_+1"])) <= -998, ((data["abs_minbatch_slices2_msignal"]) + (data["medianbatch_slices2"])), np.cos((data["minbatch_msignal"])) ))))))) * 2.0)) +

                            0.100000*np.tanh((((6.71121358871459961)) * (np.sin((((data["abs_avgbatch_msignal"]) - (np.where(data["signal_shift_-1_msignal"] > -998, data["rangebatch_msignal"], np.sin((((data["abs_avgbatch_msignal"]) - (np.where(data["abs_avgbatch_msignal"] > -998, data["rangebatch_msignal"], data["rangebatch_msignal"] ))))) )))))))) +

                            0.100000*np.tanh(np.where(data["minbatch_msignal"] <= -998, ((data["abs_avgbatch_msignal"]) + (((((data["meanbatch_slices2_msignal"]) * 2.0)) / 2.0))), (((-((np.cos((np.where(data["maxtominbatch_msignal"] > -998, data["minbatch_msignal"], ((data["stdbatch_slices2_msignal"]) * 2.0) ))))))) * 2.0) )) +

                            0.100000*np.tanh(((np.sin(((((data["rangebatch_msignal"]) + ((-((np.cos((np.tanh(((-((np.sin((data["signal_shift_+1_msignal"]))))))))))))))/2.0)))) * (((data["abs_maxbatch_slices2"]) * 2.0)))) +

                            0.100000*np.tanh(((np.tanh((np.tanh((np.where(((data["rangebatch_msignal"]) - (np.where(data["signal_shift_+1"] <= -998, (-((((data["signal"]) / 2.0)))), data["signal"] ))) <= -998, data["mean_abs_chgbatch_slices2"], np.sin((data["minbatch_msignal"])) )))))) - (((np.cos((((data["minbatch_msignal"]) * ((3.0)))))) * 2.0)))) +

                            0.100000*np.tanh(np.cos((((data["minbatch_slices2_msignal"]) * (np.where(np.where(((np.cos((((np.cos((data["stdbatch_slices2"]))) * 2.0)))) / 2.0) > -998, (2.0), ((data["meanbatch_slices2_msignal"]) * ((((data["minbatch_slices2_msignal"]) + (data["stdbatch_slices2_msignal"]))/2.0))) ) > -998, (2.0), data["abs_avgbatch_slices2"] )))))) +

                            0.100000*np.tanh(np.cos((((np.where(np.where(((np.cos((((np.where(data["minbatch_msignal"] > -998, data["minbatch_msignal"], data["meanbatch_slices2"] )) * 2.0)))) * 2.0) > -998, data["minbatch_msignal"], ((np.cos((data["rangebatch_slices2"]))) * 2.0) ) > -998, data["minbatch_msignal"], (((((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) + (((data["maxtominbatch_slices2_msignal"]) * (data["minbatch_msignal"]))))/2.0)) * 2.0) )) * 2.0)))) +

                            0.100000*np.tanh(np.sin((np.where(np.where(data["abs_minbatch_msignal"] > -998, np.sin((np.where(data["stdbatch_msignal"] > -998, ((data["stdbatch_msignal"]) * (data["stdbatch_msignal"])), ((data["abs_maxbatch_slices2"]) * (data["stdbatch_msignal"])) ))), data["minbatch_slices2"] ) > -998, ((((data["minbatch"]) - (np.sin((data["maxtominbatch_slices2"]))))) * (data["maxbatch_slices2"])), data["minbatch"] )))) +

                            0.100000*np.tanh(np.where((((7.0)) - (np.where(np.sin(((((-((((np.sin(((((-((data["maxbatch_msignal"])))) - (((data["maxbatch_slices2_msignal"]) * 2.0)))))) / 2.0))))) - ((((-1.0)) * 2.0))))) <= -998, data["stdbatch_slices2_msignal"], data["maxbatch_msignal"] ))) <= -998, data["maxbatch_slices2_msignal"], np.sin(((((-((data["maxbatch_msignal"])))) - (data["maxbatch_msignal"])))) )) +

                            0.100000*np.tanh(((data["mean_abs_chgbatch_slices2_msignal"]) - (((np.where(data["stdbatch_slices2_msignal"] <= -998, (((data["maxbatch_slices2"]) + (np.where((((((data["medianbatch_slices2"]) * (data["stdbatch_slices2_msignal"]))) + (((data["stdbatch_msignal"]) + (data["maxtominbatch"]))))/2.0) <= -998, np.sin((data["stdbatch_slices2_msignal"])), data["signal_shift_-1"] )))/2.0), data["signal_shift_-1"] )) + (data["maxtominbatch"]))))) +

                            0.100000*np.tanh(np.where(((np.where(data["stdbatch_slices2_msignal"] <= -998, np.cos(((((-3.0)) / 2.0))), ((((data["minbatch_msignal"]) * 2.0)) - (data["abs_minbatch_slices2_msignal"])) )) / 2.0) <= -998, data["medianbatch_slices2"], ((np.cos((((data["medianbatch_msignal"]) * 2.0)))) * 2.0) )) +

                            0.100000*np.tanh(((np.tanh((np.sin((((((np.cos(((-2.0)))) - (data["maxbatch_slices2_msignal"]))) * 2.0)))))) * (((data["maxbatch_slices2_msignal"]) + (((data["maxbatch_slices2_msignal"]) * 2.0)))))) +

                            0.100000*np.tanh(((np.where(np.where(((data["rangebatch_slices2"]) + (((np.cos((data["minbatch_msignal"]))) * 2.0))) <= -998, data["minbatch_msignal"], np.sin((data["minbatch_msignal"])) ) <= -998, np.tanh((np.sin((data["rangebatch_slices2"])))), np.sin(((-((((data["minbatch_msignal"]) * (((((np.cos((data["rangebatch_slices2"]))) * 2.0)) / 2.0)))))))) )) * 2.0)) +

                            0.100000*np.tanh(((data["maxbatch_msignal"]) * (np.sin((np.where((((np.sin(((2.0)))) + (data["minbatch_msignal"]))/2.0) > -998, ((data["minbatch_msignal"]) - (np.cos((np.tanh((data["minbatch_msignal"])))))), data["mean_abs_chgbatch_slices2"] )))))) +

                            0.100000*np.tanh(((data["minbatch"]) + ((((3.0)) + (((data["minbatch"]) + ((((3.0)) + (((np.where(data["minbatch"] <= -998, ((data["minbatch_slices2"]) * (data["stdbatch_msignal"])), np.sin((np.where(((data["stdbatch_slices2_msignal"]) - ((((4.0)) * 2.0))) > -998, data["mean_abs_chgbatch_msignal"], data["stdbatch_slices2_msignal"] ))) )) * 2.0)))))))))) +

                            0.100000*np.tanh(np.where((((-1.0)) + (data["rangebatch_slices2"])) <= -998, data["abs_minbatch_slices2"], (((6.94248819351196289)) * (np.sin((((np.sin((((((np.cos((data["meanbatch_msignal"]))) - (((data["rangebatch_slices2"]) - ((-3.0)))))) / 2.0)))) * 2.0))))) )) +

                            0.100000*np.tanh(((np.where(np.where(((((np.sin((data["meanbatch_slices2_msignal"]))) * 2.0)) - (np.sin((((data["abs_maxbatch_msignal"]) * 2.0))))) <= -998, (2.0), data["abs_maxbatch_msignal"] ) <= -998, data["meanbatch_slices2_msignal"], ((((((((data["medianbatch_msignal"]) - (np.sin((data["rangebatch_slices2"]))))) / 2.0)) * 2.0)) * 2.0) )) * 2.0)) +

                            0.100000*np.tanh(((np.cos((((data["medianbatch_msignal"]) * ((((3.0)) * ((-1.0)))))))) * 2.0)) +

                            0.100000*np.tanh(((((np.cos((data["abs_maxbatch_msignal"]))) * (data["maxbatch_slices2_msignal"]))) - (((np.where(data["minbatch_msignal"] > -998, data["abs_maxbatch_msignal"], data["minbatch_slices2"] )) * (np.where(np.cos((data["minbatch_slices2"])) > -998, np.cos((data["minbatch_msignal"])), np.sin(((((data["meanbatch_slices2_msignal"]) + (np.where(data["meanbatch_msignal"] > -998, data["maxtominbatch"], data["abs_maxbatch_msignal"] )))/2.0))) )))))) +

                            0.100000*np.tanh(((np.sin((((data["minbatch"]) - (np.where(data["minbatch"] > -998, data["medianbatch_slices2"], ((data["mean_abs_chgbatch_slices2_msignal"]) - (((data["medianbatch_msignal"]) * (data["medianbatch_slices2"])))) )))))) * 2.0)) +

                            0.100000*np.tanh((((((((-((np.cos((np.where(data["maxbatch_slices2_msignal"] > -998, data["minbatch_msignal"], np.tanh((np.tanh(((((((((data["minbatch_msignal"]) / 2.0)) / 2.0)) + ((((data["minbatch_msignal"]) + (np.where(data["minbatch_msignal"] > -998, data["minbatch_msignal"], data["meanbatch_slices2_msignal"] )))/2.0)))/2.0))))) ))))))) * 2.0)) * 2.0)) + (data["minbatch"]))) +

                            0.100000*np.tanh(((data["rangebatch_slices2"]) * (((np.where(data["rangebatch_slices2"] <= -998, (-((np.cos((data["abs_maxbatch_msignal"]))))), (((data["abs_maxbatch_slices2_msignal"]) + (((data["medianbatch_msignal"]) * 2.0)))/2.0) )) * (np.sin((((((data["maxbatch_slices2_msignal"]) + (np.tanh(((-((np.where(np.cos(((1.0))) > -998, (1.0), (-((data["medianbatch_msignal"]))) ))))))))) * 2.0)))))))) +

                            0.100000*np.tanh(np.where(data["abs_minbatch_slices2_msignal"] > -998, (-(((((((-2.0)) * (np.sin((((data["minbatch_msignal"]) - (np.cos((np.cos((data["minbatch_msignal"]))))))))))) * 2.0)))), ((data["medianbatch_msignal"]) - (np.where(data["medianbatch_msignal"] <= -998, data["abs_minbatch_slices2"], np.sin((data["medianbatch_msignal"])) ))) )) +

                            0.100000*np.tanh(((((np.where(data["minbatch"] <= -998, np.where(np.tanh((data["stdbatch_msignal"])) <= -998, (((((9.29319095611572266)) * 2.0)) * 2.0), data["minbatch_msignal"] ), data["maxtominbatch_slices2_msignal"] )) * (np.sin((data["minbatch"]))))) - (np.sin((data["abs_maxbatch_slices2"]))))) +

                            0.100000*np.tanh(np.where((((12.03842926025390625)) * ((12.03842926025390625))) <= -998, (((-2.0)) * (data["rangebatch_msignal"])), ((data["abs_maxbatch_msignal"]) * (np.cos((((data["medianbatch_slices2_msignal"]) + (np.where(data["mean_abs_chgbatch_slices2"] > -998, (((12.03842926025390625)) * (data["mean_abs_chgbatch_slices2"])), data["rangebatch_msignal"] ))))))) )) +

                            0.100000*np.tanh(((np.where(data["stdbatch_slices2_msignal"] > -998, np.where(data["abs_minbatch_slices2_msignal"] > -998, (9.0), (((9.0)) + (((data["mean_abs_chgbatch_slices2_msignal"]) + (data["minbatch_msignal"])))) ), data["minbatch_msignal"] )) * (np.sin((np.sin((((data["abs_maxbatch"]) + (data["rangebatch_msignal"]))))))))) +

                            0.100000*np.tanh(np.where(data["maxtominbatch"] <= -998, (((7.88421916961669922)) * (np.where(data["rangebatch_msignal"] > -998, (-3.0), np.cos(((3.0))) ))), np.cos((((data["meanbatch_slices2_msignal"]) * (np.where(data["mean_abs_chgbatch_slices2"] > -998, (-3.0), np.cos((((data["meanbatch_slices2_msignal"]) * (np.where(data["rangebatch_msignal"] > -998, (-3.0), data["rangebatch_msignal"] ))))) ))))) )) +

                            0.100000*np.tanh(((data["minbatch_msignal"]) * (np.where(np.sin((data["minbatch"])) <= -998, data["abs_maxbatch_slices2_msignal"], np.sin((np.sin((np.where(data["meanbatch_slices2"] <= -998, data["abs_maxbatch_msignal"], data["abs_maxbatch_msignal"] ))))) )))) +

                            0.100000*np.tanh(np.sin(((-((np.where(np.where(np.where(np.where((3.0) <= -998, data["stdbatch_slices2"], data["maxbatch_slices2"] ) > -998, data["abs_minbatch_msignal"], ((data["abs_minbatch_msignal"]) * (data["maxbatch_slices2_msignal"])) ) <= -998, data["maxbatch_slices2"], data["signal"] ) > -998, ((data["maxbatch_slices2_msignal"]) * 2.0), data["mean_abs_chgbatch_slices2"] ))))))) +

                            0.100000*np.tanh(((((data["medianbatch_slices2"]) + (np.where((((-1.0)) * (data["maxbatch_slices2"])) > -998, data["minbatch"], np.where(data["abs_maxbatch_slices2"] > -998, data["minbatch"], ((data["maxbatch_slices2_msignal"]) + ((((1.0)) + (np.where(((data["mean_abs_chgbatch_slices2"]) + (np.sin((data["signal_shift_-1_msignal"])))) > -998, data["minbatch"], data["minbatch"] ))))) ) )))) * 2.0)) +

                            0.100000*np.tanh(((((((np.sin(((((((np.where(data["abs_maxbatch_msignal"] > -998, data["meanbatch_msignal"], np.sin((np.sin((np.cos((np.sin((data["meanbatch_msignal"])))))))) )) * 2.0)) + (np.cos((np.sin((np.sin((np.where(data["mean_abs_chgbatch_slices2"] <= -998, data["stdbatch_slices2"], data["meanbatch_slices2_msignal"] )))))))))/2.0)))) * 2.0)) / 2.0)) * 2.0)) +

                            0.100000*np.tanh((-((((np.sin((data["abs_maxbatch_slices2_msignal"]))) * (np.where(((np.where(data["maxbatch_slices2_msignal"] <= -998, (((((5.0)) / 2.0)) + (data["mean_abs_chgbatch_msignal"])), np.sin((data["abs_maxbatch_slices2_msignal"])) )) * 2.0) <= -998, data["abs_maxbatch_slices2_msignal"], data["maxbatch_slices2_msignal"] ))))))) +

                            0.100000*np.tanh(((((data["abs_minbatch_msignal"]) - (data["maxtominbatch_msignal"]))) - (((data["rangebatch_msignal"]) - (data["maxtominbatch_msignal"]))))) +

                            0.100000*np.tanh(((np.where(((np.cos((data["stdbatch_slices2"]))) + (data["meanbatch_slices2_msignal"])) <= -998, np.cos((np.tanh((np.tanh((data["stdbatch_slices2"])))))), ((data["rangebatch_slices2_msignal"]) * (np.where(data["rangebatch_msignal"] <= -998, np.sin((np.cos((((((-((data["minbatch_msignal"])))) + (data["medianbatch_slices2_msignal"]))/2.0))))), np.cos((data["stdbatch_slices2"])) ))) )) * 2.0)) +

                            0.100000*np.tanh((-((((data["abs_avgbatch_slices2_msignal"]) + (np.where(np.where(data["abs_minbatch_slices2"] > -998, data["minbatch"], (6.36868524551391602) ) <= -998, (-3.0), ((np.sin((np.where((-2.0) <= -998, np.sin((data["abs_maxbatch"])), data["abs_maxbatch_msignal"] )))) * (data["rangebatch_slices2"])) ))))))) +

                            0.100000*np.tanh((-((np.cos((((data["minbatch_msignal"]) - (np.sin((np.where(data["maxbatch_msignal"] > -998, data["abs_avgbatch_slices2"], data["maxbatch_msignal"] ))))))))))) +

                            0.100000*np.tanh(((data["rangebatch_msignal"]) * (np.sin((np.where(((data["maxbatch_msignal"]) * ((-((((data["maxbatch_msignal"]) * 2.0)))))) <= -998, ((data["maxbatch_msignal"]) * 2.0), np.where(data["abs_maxbatch"] <= -998, ((data["signal"]) * 2.0), np.where(data["abs_avgbatch_slices2"] <= -998, data["abs_avgbatch_msignal"], (-((((data["maxbatch_msignal"]) * 2.0)))) ) ) )))))) +

                            0.100000*np.tanh((((-((((((data["signal"]) * 2.0)) + (np.where(((np.where((0.36227354407310486) <= -998, ((data["abs_minbatch_slices2_msignal"]) / 2.0), (-3.0) )) * 2.0) > -998, data["maxtominbatch"], ((data["maxtominbatch"]) + (np.sin((np.tanh((np.where((0.36227354407310486) <= -998, data["abs_maxbatch_slices2"], data["maxtominbatch"] ))))))) ))))))) * 2.0)) +

                            0.100000*np.tanh(((data["abs_avgbatch_slices2"]) * (np.sin((np.sin((((((((-2.0)) + (np.where((-((data["maxbatch_slices2_msignal"]))) > -998, ((data["maxbatch_slices2_msignal"]) * 2.0), (-(((-((data["signal_shift_-1"])))))) )))/2.0)) * 2.0)))))))) +

                            0.100000*np.tanh((((8.0)) * (((np.where(((data["rangebatch_msignal"]) + ((8.0))) <= -998, data["rangebatch_slices2_msignal"], np.sin((((data["rangebatch_msignal"]) + (np.where(np.sin((np.tanh((data["abs_maxbatch_slices2"])))) > -998, data["abs_maxbatch_slices2"], data["abs_maxbatch_msignal"] ))))) )) * ((8.0)))))) +

                            0.100000*np.tanh((((-((np.cos((((((data["minbatch_msignal"]) - (((np.cos((data["minbatch_msignal"]))) - (np.cos((data["minbatch_msignal"]))))))) - (np.tanh(((-((data["abs_avgbatch_msignal"]))))))))))))) * 2.0)) +

                            0.100000*np.tanh(np.where(data["minbatch"] <= -998, data["abs_avgbatch_slices2_msignal"], np.where(np.cos((np.cos((np.sin(((2.0))))))) > -998, ((((((3.0)) + (data["minbatch_slices2"]))) + (((data["minbatch"]) * 2.0)))/2.0), data["abs_minbatch_msignal"] ) )) +

                            0.100000*np.tanh(np.cos(((((((data["abs_maxbatch_slices2_msignal"]) + (((data["meanbatch_slices2_msignal"]) - (np.tanh((np.tanh((np.tanh((np.cos((data["maxtominbatch_slices2_msignal"]))))))))))))/2.0)) * (data["minbatch"]))))) +

                            0.100000*np.tanh(np.sin((np.where((((1.0)) * (np.tanh(((((np.where(data["minbatch_msignal"] > -998, data["minbatch_msignal"], (-((data["maxtominbatch_slices2"]))) )) + (np.cos(((-3.0)))))/2.0))))) <= -998, (0.0), np.cos((((data["maxtominbatch_msignal"]) - (np.tanh(((((np.sin((data["minbatch_slices2_msignal"]))) + ((-3.0)))/2.0))))))) )))) +

                            0.100000*np.tanh(((((np.sin((np.where(data["minbatch_slices2_msignal"] <= -998, data["abs_maxbatch_slices2"], np.cos(((((data["abs_maxbatch_slices2_msignal"]) + (data["minbatch_slices2_msignal"]))/2.0))) )))) * (np.where(((np.sin((np.sin((((data["rangebatch_msignal"]) * 2.0)))))) * (data["minbatch_slices2_msignal"])) <= -998, data["abs_maxbatch_slices2_msignal"], data["maxbatch_slices2_msignal"] )))) * (((((np.cos(((-1.0)))) * 2.0)) * 2.0)))) +

                            0.100000*np.tanh((((-((np.sin((data["abs_maxbatch_msignal"])))))) * (((data["abs_avgbatch_slices2_msignal"]) - (np.where(np.sin((data["meanbatch_msignal"])) <= -998, np.sin((np.sin(((((-((data["abs_maxbatch_msignal"])))) * ((((-2.0)) + (data["abs_maxbatch_msignal"])))))))), ((data["abs_minbatch_slices2_msignal"]) * (data["abs_avgbatch_msignal"])) )))))) +

                            0.100000*np.tanh(np.cos((((np.where(data["medianbatch_msignal"] > -998, data["medianbatch_msignal"], np.where(((data["signal_shift_-1"]) * 2.0) > -998, data["maxtominbatch_slices2_msignal"], data["medianbatch_slices2_msignal"] ) )) * 2.0)))) +

                            0.100000*np.tanh(((data["meanbatch_slices2"]) * (np.sin((((data["maxbatch_msignal"]) * (np.sin((np.where(np.cos((data["abs_maxbatch_msignal"])) <= -998, data["rangebatch_slices2"], data["maxbatch_msignal"] )))))))))) +

                            0.100000*np.tanh((-((((np.cos((((((data["stdbatch_slices2"]) * 2.0)) * (np.where((-3.0) <= -998, ((((data["minbatch_msignal"]) * (np.cos((((data["minbatch_msignal"]) * (np.where(((np.cos((data["stdbatch_slices2"]))) * 2.0) <= -998, data["minbatch_msignal"], data["stdbatch_slices2"] )))))))) - (data["abs_minbatch_slices2_msignal"])), data["stdbatch_slices2"] )))))) * 2.0))))) +

                            0.100000*np.tanh(((np.sin((np.where(((data["abs_avgbatch_msignal"]) - (data["abs_avgbatch_msignal"])) > -998, data["minbatch_msignal"], ((np.tanh((np.cos((data["minbatch_msignal"]))))) * 2.0) )))) * 2.0)) +

                            0.100000*np.tanh((-((np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) - (np.where(data["medianbatch_slices2_msignal"] <= -998, data["mean_abs_chgbatch_msignal"], np.where((-3.0) > -998, data["maxbatch_msignal"], ((data["stdbatch_msignal"]) / 2.0) ) ))))))))) +

                            0.100000*np.tanh(np.where(((data["medianbatch_msignal"]) + (data["maxbatch_msignal"])) > -998, np.cos((((((((data["medianbatch_msignal"]) / 2.0)) - (data["minbatch_slices2_msignal"]))) * 2.0))), ((((((((data["medianbatch_msignal"]) / 2.0)) - (data["rangebatch_msignal"]))) * 2.0)) * (np.cos(((((np.cos(((-3.0)))) + (np.cos((((((data["maxbatch_msignal"]) * 2.0)) / 2.0)))))/2.0))))) )) +

                            0.100000*np.tanh(((((np.cos((np.where(np.sin((((((np.cos((((data["medianbatch_slices2"]) * 2.0)))) - ((-((np.sin((data["medianbatch_slices2_msignal"])))))))) + (((data["maxbatch_slices2_msignal"]) / 2.0))))) > -998, data["meanbatch_msignal"], data["abs_avgbatch_slices2_msignal"] )))) + (data["rangebatch_slices2"]))) * (np.cos((data["stdbatch_slices2"]))))) +

                            0.100000*np.tanh(np.where(((data["signal_shift_-1"]) * (np.sin((np.cos((data["medianbatch_slices2_msignal"])))))) > -998, np.cos((data["minbatch_slices2"])), data["abs_maxbatch"] )) +

                            0.100000*np.tanh((-((np.sin((((np.where(np.sin(((-3.0))) <= -998, np.where(data["abs_minbatch_msignal"] <= -998, np.sin(((-3.0))), np.where(data["rangebatch_slices2"] > -998, (-3.0), (-2.0) ) ), ((data["signal_shift_-1_msignal"]) - ((((np.tanh((np.sin((data["rangebatch_slices2"]))))) + (((data["maxbatch_slices2_msignal"]) * (data["rangebatch_slices2"]))))/2.0))) )) / 2.0))))))) +

                            0.100000*np.tanh(np.sin((((data["minbatch_msignal"]) + ((-((np.where((((2.39420104026794434)) * (data["abs_maxbatch_msignal"])) > -998, data["meanbatch_slices2_msignal"], (-((np.where(((((np.sin((np.cos((data["abs_avgbatch_msignal"]))))) - (data["meanbatch_slices2_msignal"]))) * 2.0) <= -998, data["stdbatch_slices2"], ((((data["maxtominbatch_slices2"]) / 2.0)) + (((data["minbatch_msignal"]) * 2.0))) )))) ))))))))) +

                            0.100000*np.tanh(np.where(((data["maxtominbatch"]) + (((np.cos((((data["minbatch"]) - (data["minbatch"]))))) * 2.0))) > -998, ((data["abs_minbatch_msignal"]) - (np.where(data["abs_maxbatch_slices2"] > -998, data["maxtominbatch_slices2"], data["signal_shift_+1_msignal"] ))), np.where(data["abs_minbatch_msignal"] <= -998, (3.0), data["maxbatch_slices2"] ) )) +

                            0.100000*np.tanh(np.where(data["maxbatch_msignal"] <= -998, data["abs_maxbatch_slices2_msignal"], np.cos((((np.where(data["signal_shift_+1_msignal"] > -998, data["minbatch_slices2_msignal"], np.cos((((np.where(data["minbatch"] > -998, data["minbatch_slices2_msignal"], np.tanh((data["minbatch"])) )) * 2.0))) )) * 2.0))) )) +

                            0.100000*np.tanh(np.cos((((data["minbatch_msignal"]) + (np.where(data["minbatch_msignal"] > -998, data["minbatch_slices2_msignal"], ((data["maxtominbatch_slices2_msignal"]) / 2.0) )))))) +

                            0.100000*np.tanh(np.sin(((-((((np.sin(((((1.0)) * (np.sin((np.cos((data["maxbatch_slices2"]))))))))) * 2.0))))))) +

                            0.100000*np.tanh(((np.sin((np.sin((((((np.cos((((data["maxbatch_slices2"]) * 2.0)))) * 2.0)) + (np.where(((((((data["maxbatch_slices2"]) * 2.0)) * ((0.0)))) * 2.0) <= -998, ((((data["rangebatch_slices2"]) + (((data["maxbatch_slices2"]) * 2.0)))) * 2.0), data["abs_avgbatch_msignal"] )))))))) / 2.0)) +

                            0.100000*np.tanh(((((np.cos((((((data["maxbatch_slices2"]) + (np.where((((data["maxbatch_slices2"]) + (np.tanh(((3.0)))))/2.0) <= -998, ((np.cos((data["abs_avgbatch_slices2_msignal"]))) * 2.0), data["stdbatch_slices2_msignal"] )))) * 2.0)))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(((np.cos((((data["medianbatch_msignal"]) * (np.where(np.where(np.cos((data["medianbatch_msignal"])) <= -998, data["abs_minbatch_slices2"], (-3.0) ) <= -998, (-3.0), (-3.0) )))))) * 2.0)) +

                            0.100000*np.tanh(((((((data["abs_minbatch_slices2_msignal"]) * (data["signal_shift_-1_msignal"]))) * (data["rangebatch_slices2"]))) + (((np.sin((data["meanbatch_slices2_msignal"]))) * ((((((((data["meanbatch_slices2_msignal"]) * ((((data["minbatch_slices2_msignal"]) + (data["stdbatch_slices2_msignal"]))/2.0)))) + (np.cos((np.sin((data["abs_avgbatch_slices2"]))))))/2.0)) * 2.0)))))) +

                            0.100000*np.tanh((((((((3.0)) + (np.where(data["rangebatch_slices2"] > -998, np.tanh((((((data["meanbatch_slices2_msignal"]) * ((-(((-((((np.cos((data["maxbatch_msignal"]))) / 2.0)))))))))) * 2.0))), ((data["abs_avgbatch_slices2_msignal"]) + ((3.0))) )))) * (data["signal_shift_-1_msignal"]))) * (((data["meanbatch_msignal"]) * 2.0)))) +

                            0.100000*np.tanh(((data["minbatch_msignal"]) * (np.sin((np.cos((((data["meanbatch_slices2"]) - (((data["minbatch_slices2"]) - (np.where(data["minbatch_slices2"] > -998, np.sin(((-((data["mean_abs_chgbatch_slices2_msignal"]))))), np.cos((data["minbatch_msignal"])) )))))))))))) +

                            0.100000*np.tanh(((np.where(data["mean_abs_chgbatch_slices2"] <= -998, (-((data["abs_maxbatch_msignal"]))), (((2.0)) * 2.0) )) * (((np.sin((data["minbatch"]))) * (np.cos((((data["meanbatch_slices2"]) + (np.tanh(((0.0)))))))))))) +

                            0.100000*np.tanh(np.where(data["abs_avgbatch_msignal"] <= -998, data["abs_maxbatch_msignal"], (((2.0)) * ((((-3.0)) * (np.sin((np.where(data["abs_maxbatch"] > -998, (((data["abs_avgbatch_msignal"]) + (data["minbatch_msignal"]))/2.0), data["abs_avgbatch_slices2_msignal"] ))))))) )) +

                            0.100000*np.tanh(((np.sin((data["maxbatch_msignal"]))) * ((-((np.where((((data["meanbatch_slices2"]) + ((-((data["minbatch_msignal"])))))/2.0) <= -998, np.tanh((np.sin(((((data["meanbatch_slices2"]) + (np.sin(((-((((np.sin((data["maxbatch_msignal"]))) + (data["meanbatch_slices2"])))))))))/2.0))))), data["meanbatch_slices2"] ))))))) +

                            0.100000*np.tanh(((np.cos((((data["abs_maxbatch"]) * 2.0)))) * (((((np.where(((np.where(data["abs_maxbatch"] > -998, data["abs_maxbatch"], (13.46337604522705078) )) * (((data["signal_shift_-1"]) * 2.0))) > -998, (2.0), np.cos((((data["abs_maxbatch"]) * 2.0))) )) * 2.0)) / 2.0)))) +

                            0.100000*np.tanh(np.cos(((((data["stdbatch_msignal"]) + (np.where(data["minbatch_slices2_msignal"] > -998, data["rangebatch_msignal"], ((np.sin((np.tanh(((((data["stdbatch_msignal"]) + (data["rangebatch_msignal"]))/2.0)))))) * (np.cos((data["abs_avgbatch_msignal"])))) )))/2.0)))) +

                            0.100000*np.tanh((((2.0)) + ((((-((data["signal_shift_-1"])))) * (np.where((-(((-((data["signal_shift_-1"])))))) <= -998, data["abs_maxbatch_slices2"], (-(((((data["minbatch"]) + ((((data["meanbatch_slices2_msignal"]) + (np.where(data["signal_shift_-1"] <= -998, ((((2.0)) + ((((data["signal_shift_-1"]) + ((2.0)))/2.0)))/2.0), data["meanbatch_slices2_msignal"] )))/2.0)))/2.0)))) )))))) +

                            0.100000*np.tanh(np.where((3.0) <= -998, data["rangebatch_slices2"], ((np.sin((np.cos(((((np.where(np.where(((((data["medianbatch_slices2"]) * (data["rangebatch_slices2"]))) * 2.0) <= -998, data["rangebatch_slices2"], ((data["meanbatch_msignal"]) * 2.0) ) <= -998, data["maxtominbatch_slices2"], data["medianbatch_slices2"] )) + (((data["rangebatch_slices2"]) * 2.0)))/2.0)))))) * 2.0) )) +

                            0.100000*np.tanh(np.where(((((np.cos((data["meanbatch_msignal"]))) + (np.tanh((data["maxbatch_msignal"]))))) * (((((np.cos((((data["abs_avgbatch_msignal"]) / 2.0)))) * 2.0)) / 2.0))) > -998, np.sin((data["meanbatch_slices2_msignal"])), (((data["rangebatch_slices2"]) + (data["abs_maxbatch_slices2"]))/2.0) )) +

                            0.100000*np.tanh(np.cos((data["maxtominbatch_slices2"]))) +

                            0.100000*np.tanh(np.where(data["abs_minbatch_slices2_msignal"] > -998, ((((((data["maxbatch_slices2_msignal"]) * (np.sin((((((data["maxbatch_slices2_msignal"]) * (np.sin((np.where((2.81497907638549805) <= -998, data["meanbatch_msignal"], data["rangebatch_slices2"] )))))) * 2.0)))))) * 2.0)) * 2.0), np.where((2.0) <= -998, (2.0), ((np.cos((data["signal_shift_+1_msignal"]))) * 2.0) ) )) +

                            0.100000*np.tanh(((np.sin(((((-((data["maxbatch_msignal"])))) / 2.0)))) * (np.cos(((((((data["minbatch_msignal"]) / 2.0)) + (np.where((9.0) > -998, data["minbatch_msignal"], data["abs_maxbatch"] )))/2.0)))))) +

                            0.100000*np.tanh(((np.cos((data["minbatch_msignal"]))) * ((-((np.where(data["signal_shift_+1"] <= -998, data["abs_maxbatch"], np.where(np.where(data["abs_maxbatch_slices2"] > -998, (-((data["minbatch_slices2"]))), ((((data["signal_shift_+1"]) * 2.0)) * 2.0) ) > -998, data["abs_maxbatch_slices2"], (-1.0) ) ))))))) +

                            0.100000*np.tanh(np.where(data["minbatch"] <= -998, data["abs_maxbatch"], ((((np.where(np.cos((((data["minbatch"]) - (data["meanbatch_slices2"])))) <= -998, ((((data["meanbatch_slices2"]) - (data["meanbatch_slices2"]))) * 2.0), ((((((((np.sin((((data["minbatch"]) - (data["meanbatch_slices2"]))))) * 2.0)) * 2.0)) / 2.0)) * 2.0) )) * 2.0)) * 2.0) )) +

                            0.100000*np.tanh(np.where(data["abs_maxbatch_msignal"] <= -998, ((np.cos((((np.sin((data["abs_avgbatch_msignal"]))) + (data["abs_maxbatch_msignal"]))))) / 2.0), np.cos((data["abs_avgbatch_msignal"])) )) +

                            0.100000*np.tanh((-((np.where(data["maxbatch_slices2_msignal"] <= -998, ((((data["maxbatch_slices2_msignal"]) / 2.0)) / 2.0), np.sin((((data["maxbatch_slices2_msignal"]) * (data["minbatch_msignal"])))) ))))) +

                            0.100000*np.tanh(np.where(data["rangebatch_slices2"] <= -998, (-1.0), ((np.sin((data["medianbatch_slices2_msignal"]))) / 2.0) )) +

                            0.100000*np.tanh(((data["signal_shift_+1_msignal"]) * (((((np.cos((((np.where((4.09689283370971680) <= -998, (((10.0)) * 2.0), ((data["medianbatch_slices2_msignal"]) * ((-(((10.0)))))) )) * ((8.0)))))) * (((data["signal_shift_+1_msignal"]) / 2.0)))) - (((data["medianbatch_slices2_msignal"]) * ((-(((((((4.12076473236083984)) * 2.0)) * 2.0))))))))))) +

                            0.100000*np.tanh((-((np.sin((np.where(data["abs_avgbatch_msignal"] > -998, data["abs_maxbatch_msignal"], data["abs_minbatch_slices2"] ))))))) +

                            0.100000*np.tanh(np.where(((np.tanh((((np.where(np.where(data["signal_shift_+1_msignal"] > -998, (((3.0)) + (data["abs_avgbatch_slices2"])), (1.0) ) > -998, data["stdbatch_slices2"], data["abs_minbatch_slices2"] )) / 2.0)))) * (np.where(np.tanh(((0.0))) > -998, data["abs_avgbatch_slices2_msignal"], data["maxtominbatch_msignal"] ))) <= -998, data["maxbatch_slices2"], np.sin((np.cos((((data["maxbatch_slices2"]) * 2.0))))) )) +

                            0.100000*np.tanh(np.cos((np.where(np.tanh((np.cos((((((data["abs_maxbatch_msignal"]) * (data["meanbatch_msignal"]))) / 2.0))))) <= -998, ((data["mean_abs_chgbatch_slices2_msignal"]) - (((((data["abs_minbatch_slices2_msignal"]) * 2.0)) * 2.0))), ((data["signal_shift_+1"]) * ((((-3.0)) / 2.0))) )))) +

                            0.100000*np.tanh(((np.where(((data["signal"]) + (data["rangebatch_msignal"])) <= -998, data["minbatch_slices2"], data["signal_shift_+1_msignal"] )) * (np.where(data["minbatch_msignal"] <= -998, data["minbatch_msignal"], (((((((data["minbatch_msignal"]) + (data["abs_maxbatch_slices2_msignal"]))/2.0)) * (((data["maxbatch_slices2"]) + (data["abs_maxbatch_slices2_msignal"]))))) * (((data["abs_maxbatch_msignal"]) + (data["abs_avgbatch_msignal"])))) )))) +

                            0.100000*np.tanh(((data["rangebatch_slices2_msignal"]) - (((((((data["rangebatch_slices2_msignal"]) * (np.cos((((data["abs_maxbatch_slices2_msignal"]) - (((((((data["abs_maxbatch_msignal"]) * (np.cos((((data["abs_maxbatch_slices2_msignal"]) - (((np.cos((data["maxtominbatch_msignal"]))) / 2.0)))))))) * 2.0)) / 2.0)))))))) * 2.0)) * 2.0)))) +

                            0.100000*np.tanh(np.sin((((np.where(((((data["medianbatch_slices2_msignal"]) - (np.sin((np.sin((((np.where(data["abs_minbatch_slices2_msignal"] <= -998, data["maxbatch_slices2_msignal"], data["signal_shift_-1_msignal"] )) * 2.0)))))))) - (data["maxbatch_slices2_msignal"])) > -998, ((data["signal_shift_-1_msignal"]) - (data["maxbatch_slices2_msignal"])), ((data["signal_shift_-1_msignal"]) * 2.0) )) * 2.0)))) +

                            0.100000*np.tanh(((np.cos(((((data["minbatch_msignal"]) + (np.where(np.where(((np.cos(((-(((-(((((data["minbatch_msignal"]) + (np.where((2.0) > -998, data["abs_maxbatch_msignal"], data["abs_maxbatch_msignal"] )))/2.0)))))))))) * 2.0) > -998, (2.0), data["minbatch_msignal"] ) > -998, data["abs_maxbatch_msignal"], data["abs_maxbatch_slices2_msignal"] )))/2.0)))) * 2.0)) +

                            0.100000*np.tanh(np.cos(((-((((data["minbatch_msignal"]) * 2.0))))))) +

                            0.100000*np.tanh(((np.where(((data["medianbatch_slices2"]) + (np.cos((data["abs_maxbatch"])))) <= -998, data["medianbatch_slices2"], ((data["abs_maxbatch_slices2"]) * (((((data["abs_maxbatch_slices2"]) - (data["mean_abs_chgbatch_slices2_msignal"]))) * (np.sin((((data["mean_abs_chgbatch_slices2"]) * ((((-3.0)) * 2.0))))))))) )) * 2.0)) +

                            0.100000*np.tanh(((((((np.sin((data["stdbatch_msignal"]))) + ((((data["abs_maxbatch_msignal"]) + (np.sin((np.where(data["rangebatch_slices2_msignal"] > -998, np.tanh((np.cos((data["mean_abs_chgbatch_slices2"])))), data["minbatch"] )))))/2.0)))) / 2.0)) * (data["stdbatch_msignal"]))) +

                            0.100000*np.tanh(np.cos((np.where(data["stdbatch_slices2"] > -998, ((data["minbatch_slices2_msignal"]) * (data["maxbatch_slices2_msignal"])), np.cos((np.where(np.tanh(((2.0))) > -998, data["minbatch"], data["maxbatch_slices2_msignal"] ))) )))) +

                            0.100000*np.tanh(data["medianbatch_slices2"]) +

                            0.100000*np.tanh(((np.sin((((data["rangebatch_slices2"]) + (np.where(np.where(data["abs_minbatch_slices2"] > -998, ((data["minbatch_msignal"]) * (((data["rangebatch_slices2_msignal"]) * (data["signal"])))), data["medianbatch_slices2_msignal"] ) > -998, data["rangebatch_slices2_msignal"], np.where((5.0) <= -998, ((data["rangebatch_slices2"]) * (data["rangebatch_slices2_msignal"])), np.sin((data["signal_shift_+1_msignal"])) ) )))))) * 2.0)) +

                            0.100000*np.tanh(((data["signal_shift_-1_msignal"]) * (((data["medianbatch_msignal"]) * (np.where(data["signal_shift_-1_msignal"] <= -998, data["signal_shift_-1_msignal"], (((12.17507171630859375)) - (((data["medianbatch_msignal"]) * ((((data["abs_maxbatch_slices2"]) + ((12.17507171630859375)))/2.0))))) )))))) +

                            0.100000*np.tanh((((1.0)) - (np.cos(((((1.0)) + ((-((((np.cos((np.where(data["rangebatch_slices2_msignal"] <= -998, (1.0), ((data["rangebatch_slices2_msignal"]) + (np.sin((data["medianbatch_msignal"])))) )))) + (((((np.cos((data["maxbatch_slices2"]))) * 2.0)) / 2.0))))))))))))) +

                            0.100000*np.tanh(np.where((-((data["minbatch_slices2_msignal"]))) > -998, ((np.sin((((np.tanh((data["abs_maxbatch_msignal"]))) - (np.where(np.where((0.0) > -998, data["minbatch_slices2_msignal"], data["minbatch_slices2_msignal"] ) <= -998, (7.0), ((data["minbatch_slices2_msignal"]) * (data["maxbatch_slices2_msignal"])) )))))) * 2.0), data["minbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(((np.sin((((np.where(data["maxtominbatch_msignal"] > -998, (((-((data["maxbatch_slices2_msignal"])))) * (data["stdbatch_slices2"])), np.cos((np.cos((((data["maxtominbatch_slices2"]) + (np.tanh(((-((data["maxbatch_slices2_msignal"]))))))))))) )) * 2.0)))) * (data["abs_maxbatch"]))) +

                            0.100000*np.tanh(np.sin((data["abs_maxbatch"]))) +

                            0.100000*np.tanh(np.where(np.cos(((-2.0))) > -998, np.cos((((data["minbatch_slices2_msignal"]) * 2.0))), np.where(data["medianbatch_msignal"] <= -998, data["signal_shift_-1_msignal"], np.where(data["abs_maxbatch_slices2"] > -998, ((data["minbatch_slices2_msignal"]) * 2.0), data["minbatch_slices2_msignal"] ) ) )) +

                            0.100000*np.tanh((3.0)) +

                            0.100000*np.tanh(np.sin((((((((data["stdbatch_slices2"]) + (((data["maxbatch_slices2_msignal"]) / 2.0)))) / 2.0)) * (np.where(((np.tanh((((data["abs_maxbatch_msignal"]) / 2.0)))) * (np.where(data["meanbatch_slices2"] > -998, ((((((data["maxbatch_slices2_msignal"]) / 2.0)) / 2.0)) / 2.0), data["maxbatch_slices2_msignal"] ))) > -998, data["maxbatch_slices2_msignal"], data["mean_abs_chgbatch_slices2_msignal"] )))))) +

                            0.100000*np.tanh(np.sin((((data["minbatch_msignal"]) + ((-((((np.tanh((np.tanh((np.where(data["medianbatch_msignal"] <= -998, np.where(np.sin((data["maxtominbatch"])) > -998, np.tanh((data["maxtominbatch"])), np.where(data["abs_minbatch_msignal"] <= -998, ((data["maxtominbatch"]) * 2.0), ((data["medianbatch_slices2_msignal"]) / 2.0) ) ), data["medianbatch_msignal"] )))))) + (data["medianbatch_msignal"])))))))))) +

                            0.100000*np.tanh(np.cos(((((-3.0)) * (np.where((((data["abs_maxbatch_slices2_msignal"]) + ((((7.52021884918212891)) + (np.where(data["abs_maxbatch_slices2"] > -998, np.where((7.57904481887817383) > -998, data["mean_abs_chgbatch_slices2"], (3.0) ), np.sin((((data["medianbatch_slices2_msignal"]) + (data["medianbatch_msignal"])))) )))))/2.0) > -998, data["maxbatch_slices2_msignal"], (-3.0) )))))) +

                            0.100000*np.tanh(np.where(((((data["abs_maxbatch_slices2"]) + (((data["medianbatch_slices2"]) * (data["minbatch"]))))) * (((((data["medianbatch_msignal"]) + (data["minbatch"]))) * (data["minbatch"])))) <= -998, data["minbatch"], ((data["abs_maxbatch_slices2"]) + (((data["medianbatch_slices2"]) * (data["minbatch"])))) )) +

                            0.100000*np.tanh(((data["minbatch"]) * (((data["maxtominbatch_slices2"]) + (np.where(np.where(data["minbatch_msignal"] > -998, np.where(data["minbatch_msignal"] > -998, data["abs_minbatch_slices2_msignal"], np.where(data["minbatch_msignal"] > -998, data["maxtominbatch_slices2"], np.where((10.0) > -998, data["minbatch_slices2_msignal"], data["stdbatch_msignal"] ) ) ), np.where((10.0) > -998, data["maxtominbatch_slices2"], (3.0) ) ) > -998, data["signal_shift_-1"], (2.07122015953063965) )))))) +

                            0.100000*np.tanh(((data["rangebatch_msignal"]) * (((((data["signal_shift_-1_msignal"]) * (np.where(((data["meanbatch_slices2"]) * 2.0) > -998, np.tanh((((np.where(np.cos((data["rangebatch_msignal"])) > -998, np.tanh((((data["meanbatch_slices2_msignal"]) * 2.0))), np.tanh((data["meanbatch_msignal"])) )) * 2.0))), data["meanbatch_msignal"] )))) * 2.0)))) +

                            0.100000*np.tanh(((((data["maxbatch_slices2_msignal"]) + (((data["signal_shift_+1"]) * (((np.where(np.cos((np.where(data["maxtominbatch_slices2_msignal"] > -998, data["abs_maxbatch"], (((data["abs_maxbatch_slices2_msignal"]) + (data["stdbatch_msignal"]))/2.0) ))) <= -998, ((data["stdbatch_slices2_msignal"]) - (np.cos((data["stdbatch_msignal"])))), data["stdbatch_slices2_msignal"] )) * 2.0)))))) + (data["mean_abs_chgbatch_slices2"]))) +

                            0.100000*np.tanh(np.where((-(((-((data["medianbatch_slices2_msignal"])))))) <= -998, np.tanh((((((data["rangebatch_slices2_msignal"]) / 2.0)) + (data["minbatch_msignal"])))), ((data["minbatch"]) + (((data["mean_abs_chgbatch_slices2"]) + (np.where((1.02791571617126465) <= -998, data["meanbatch_msignal"], np.cos((data["meanbatch_msignal"])) ))))) )) +

                            0.100000*np.tanh(np.where((((((((3.95899510383605957)) / 2.0)) / 2.0)) - (np.tanh((data["meanbatch_msignal"])))) > -998, data["meanbatch_msignal"], np.where(data["mean_abs_chgbatch_slices2"] > -998, data["signal_shift_-1_msignal"], np.where(data["mean_abs_chgbatch_slices2"] > -998, np.where(data["mean_abs_chgbatch_slices2"] > -998, ((data["stdbatch_msignal"]) / 2.0), ((data["mean_abs_chgbatch_slices2"]) / 2.0) ), np.cos((((data["meanbatch_msignal"]) / 2.0))) ) ) )) +

                            0.100000*np.tanh(np.sin((((((data["stdbatch_slices2"]) + (((np.where(((np.sin((((data["maxtominbatch_slices2"]) * 2.0)))) * 2.0) > -998, (-2.0), data["stdbatch_slices2"] )) - (((np.where(data["maxbatch_slices2_msignal"] <= -998, data["minbatch_msignal"], data["minbatch_msignal"] )) / 2.0)))))) * 2.0)))) +

                            0.100000*np.tanh(((data["abs_minbatch_slices2_msignal"]) - (np.where((-((data["signal_shift_+1"]))) <= -998, data["abs_minbatch_slices2_msignal"], ((data["maxtominbatch"]) - (np.where(data["signal_shift_+1"] <= -998, data["abs_minbatch_slices2_msignal"], (-((data["signal_shift_+1"]))) ))) )))) +

                            0.100000*np.tanh(np.where(np.where(np.cos((((data["abs_maxbatch_msignal"]) / 2.0))) <= -998, np.tanh((((data["minbatch_msignal"]) / 2.0))), (0.0) ) <= -998, np.sin(((((data["abs_maxbatch_msignal"]) + (data["abs_minbatch_msignal"]))/2.0))), np.cos((((data["abs_maxbatch_slices2_msignal"]) / 2.0))) )) +

                            0.100000*np.tanh(((np.tanh((((data["stdbatch_msignal"]) * 2.0)))) + ((1.0)))) +

                            0.100000*np.tanh((-((((data["signal_shift_+1_msignal"]) * ((((8.0)) * (np.sin((np.sin((((np.where(data["maxtominbatch"] <= -998, np.cos((np.sin((((np.where(data["rangebatch_msignal"] <= -998, data["abs_minbatch_msignal"], data["maxbatch_slices2_msignal"] )) * 2.0))))), data["maxbatch_slices2_msignal"] )) * 2.0))))))))))))) +

                            0.100000*np.tanh((((((0.0)) / 2.0)) + (np.where(((data["maxtominbatch_slices2_msignal"]) / 2.0) <= -998, ((np.tanh((((data["minbatch_slices2_msignal"]) - (((data["abs_maxbatch_slices2_msignal"]) - (data["medianbatch_slices2_msignal"]))))))) / 2.0), data["meanbatch_slices2"] )))) +

                            0.100000*np.tanh(np.sin((np.where((((data["rangebatch_msignal"]) + (np.sin((np.where((((data["rangebatch_msignal"]) + (data["stdbatch_slices2_msignal"]))/2.0) > -998, data["maxtominbatch_msignal"], data["maxtominbatch_msignal"] )))))/2.0) > -998, data["maxtominbatch_msignal"], data["minbatch_msignal"] )))) +

                            0.100000*np.tanh(np.sin(((((-((np.where(np.sin((data["abs_avgbatch_msignal"])) > -998, data["mean_abs_chgbatch_slices2_msignal"], np.cos(((((data["mean_abs_chgbatch_slices2"]) + ((((data["abs_avgbatch_msignal"]) + (np.tanh((np.tanh((((data["stdbatch_slices2"]) + ((-((np.where(data["mean_abs_chgbatch_msignal"] > -998, data["abs_maxbatch_slices2"], (0.0) ))))))))))))/2.0)))/2.0))) ))))) * 2.0)))) +

                            0.100000*np.tanh(((data["meanbatch_slices2"]) + (((np.tanh((((np.tanh((((np.where(data["minbatch_msignal"] <= -998, data["mean_abs_chgbatch_slices2_msignal"], (3.0) )) + (np.where(data["minbatch_msignal"] > -998, np.cos((data["maxtominbatch_msignal"])), (1.0) )))))) + (((np.tanh((np.tanh((np.sin((np.sin((data["stdbatch_slices2_msignal"]))))))))) * 2.0)))))) - (data["signal_shift_+1"]))))) +

                            0.100000*np.tanh(np.cos((np.where((((np.tanh((data["abs_maxbatch_slices2"]))) + (((np.where(np.tanh((((((data["maxbatch_slices2"]) + (data["abs_maxbatch_slices2"]))) + (data["maxbatch_slices2"])))) > -998, ((data["abs_maxbatch_slices2"]) + (data["maxbatch_slices2"])), (3.0) )) + (data["maxbatch_slices2"]))))/2.0) > -998, ((data["abs_maxbatch_slices2"]) + (data["maxbatch_slices2"])), ((data["abs_maxbatch_slices2"]) + (data["maxbatch_slices2"])) )))) +

                            0.100000*np.tanh(np.cos((((((data["abs_maxbatch_msignal"]) * (data["abs_maxbatch_msignal"]))) * 2.0)))) +

                            0.100000*np.tanh(np.cos((((np.tanh((np.sin((((data["maxtominbatch"]) + (np.tanh((np.sin((np.sin((((np.cos((data["abs_avgbatch_slices2_msignal"]))) * (data["rangebatch_slices2_msignal"]))))))))))))))) - (np.where(data["minbatch"] <= -998, np.cos(((-((((np.tanh(((-((data["abs_avgbatch_msignal"])))))) / 2.0)))))), ((data["minbatch"]) - (data["abs_avgbatch_slices2_msignal"])) )))))) +

                            0.100000*np.tanh(np.where(((data["maxbatch_slices2_msignal"]) + (((data["mean_abs_chgbatch_slices2"]) / 2.0))) > -998, data["mean_abs_chgbatch_slices2"], np.sin((data["signal_shift_+1"])) )) +

                            0.100000*np.tanh(np.where(np.tanh((data["abs_maxbatch_slices2_msignal"])) <= -998, (-((np.cos(((-((np.cos((data["minbatch"])))))))))), np.where(((((data["abs_maxbatch_slices2_msignal"]) / 2.0)) + (np.cos((data["mean_abs_chgbatch_msignal"])))) <= -998, np.tanh((((data["abs_minbatch_msignal"]) * 2.0))), np.cos(((((data["abs_maxbatch_slices2_msignal"]) + ((1.0)))/2.0))) ) )) +

                            0.100000*np.tanh(np.cos(((((((((((((data["signal_shift_-1_msignal"]) / 2.0)) + (((np.cos((((data["minbatch_slices2_msignal"]) / 2.0)))) * 2.0)))) + (data["minbatch"]))/2.0)) + (np.cos((((data["minbatch_slices2_msignal"]) / 2.0)))))) + (data["abs_avgbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh(np.sin((((((-((data["abs_avgbatch_slices2_msignal"])))) + (np.sin((((np.tanh((data["mean_abs_chgbatch_slices2"]))) * (np.sin((data["medianbatch_slices2"]))))))))/2.0)))) +

                            0.100000*np.tanh(((((((np.cos((data["abs_maxbatch"]))) * (np.where(((((np.where(np.sin((data["maxbatch_slices2"])) > -998, (0.73015469312667847), data["maxbatch_slices2"] )) + ((3.0)))) + (data["mean_abs_chgbatch_slices2_msignal"])) <= -998, data["medianbatch_msignal"], np.where((3.0) <= -998, ((data["medianbatch_slices2_msignal"]) + ((2.0))), data["abs_minbatch_slices2_msignal"] ) )))) + ((3.0)))) + (data["mean_abs_chgbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(np.sin((data["meanbatch_msignal"]))) +

                            0.100000*np.tanh(((((((3.0)) + ((5.65421962738037109)))/2.0)) * (np.cos((np.where((((data["meanbatch_slices2_msignal"]) + ((((((5.06128358840942383)) / 2.0)) * 2.0)))/2.0) <= -998, (2.0), ((((5.06128358840942383)) + (data["rangebatch_slices2"]))/2.0) )))))) +

                            0.100000*np.tanh(np.where(((data["abs_minbatch_slices2"]) / 2.0) > -998, np.cos((data["maxtominbatch_slices2_msignal"])), np.cos((np.where(((data["rangebatch_msignal"]) / 2.0) <= -998, ((np.cos((data["maxtominbatch_slices2_msignal"]))) / 2.0), ((((data["meanbatch_slices2_msignal"]) + (data["minbatch_msignal"]))) / 2.0) ))) )) +

                            0.100000*np.tanh(np.where(data["minbatch"] <= -998, np.tanh((np.sin((np.where(data["signal_shift_+1"] <= -998, data["medianbatch_slices2_msignal"], np.sin((data["minbatch"])) ))))), (((data["meanbatch_msignal"]) + (((data["maxtominbatch_slices2_msignal"]) * (np.where(np.tanh((np.tanh((np.sin((np.tanh((data["meanbatch_msignal"])))))))) <= -998, data["medianbatch_slices2_msignal"], np.sin((data["minbatch"])) )))))/2.0) )) +

                            0.100000*np.tanh(np.sin((((data["minbatch_msignal"]) + ((-((np.where(((data["maxbatch_msignal"]) * 2.0) > -998, data["meanbatch_msignal"], data["minbatch_msignal"] ))))))))) +

                            0.100000*np.tanh(np.where(np.where(np.where(data["abs_avgbatch_msignal"] <= -998, data["signal_shift_+1_msignal"], data["signal_shift_+1_msignal"] ) > -998, data["signal"], data["medianbatch_msignal"] ) <= -998, np.cos((data["abs_avgbatch_msignal"])), ((data["medianbatch_slices2_msignal"]) * (((data["signal_shift_+1_msignal"]) - (np.where(np.sin((((data["medianbatch_slices2_msignal"]) - (np.cos((data["maxtominbatch_slices2_msignal"])))))) > -998, data["medianbatch_slices2_msignal"], np.cos((data["abs_avgbatch_msignal"])) ))))) )) +

                            0.100000*np.tanh(data["abs_maxbatch"]) +

                            0.100000*np.tanh(((np.where(np.cos((data["abs_minbatch_slices2_msignal"])) <= -998, data["minbatch_slices2_msignal"], data["minbatch_slices2_msignal"] )) + ((3.18847489356994629)))) +

                            0.100000*np.tanh(((data["rangebatch_slices2_msignal"]) + ((-((np.sin((np.where(((np.cos(((((11.96301269531250000)) / 2.0)))) + (np.tanh((np.cos((data["rangebatch_msignal"])))))) > -998, data["signal_shift_+1"], data["stdbatch_slices2"] ))))))))) +

                            0.100000*np.tanh(np.sin((((np.where(data["abs_avgbatch_slices2_msignal"] <= -998, np.where(data["abs_minbatch_msignal"] > -998, np.cos((np.where(data["minbatch"] <= -998, data["abs_avgbatch_slices2_msignal"], np.where((2.0) > -998, data["rangebatch_slices2_msignal"], ((data["stdbatch_slices2_msignal"]) - (data["rangebatch_slices2_msignal"])) ) ))), data["rangebatch_msignal"] ), data["rangebatch_slices2_msignal"] )) * 2.0)))) +

                            0.100000*np.tanh(((data["abs_maxbatch"]) * (np.cos((np.where((-1.0) <= -998, np.cos((np.where((3.0) <= -998, (((data["stdbatch_slices2_msignal"]) + ((-((((data["stdbatch_slices2_msignal"]) - (data["abs_maxbatch_slices2_msignal"])))))))/2.0), (((3.0)) * (data["maxbatch_slices2_msignal"])) ))), (((3.0)) * (data["maxbatch_slices2_msignal"])) )))))) +

                            0.100000*np.tanh(np.where(data["stdbatch_msignal"] > -998, data["stdbatch_slices2"], ((((np.cos(((2.0)))) * 2.0)) * 2.0) )) +

                            0.100000*np.tanh(((((np.sin((np.cos((((data["maxbatch_slices2_msignal"]) / 2.0)))))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(((data["abs_maxbatch_msignal"]) * (np.sin(((-((np.where((((np.sin(((-((np.where((((np.tanh(((2.0)))) + (((data["abs_maxbatch_msignal"]) / 2.0)))/2.0) <= -998, np.tanh((data["abs_maxbatch_msignal"])), data["abs_maxbatch_msignal"] ))))))) + (data["minbatch_slices2"]))/2.0) <= -998, np.where(data["meanbatch_slices2"] <= -998, data["abs_maxbatch_msignal"], data["abs_maxbatch_msignal"] ), data["abs_maxbatch_msignal"] ))))))))) +

                            0.100000*np.tanh(np.where((-((np.sin((np.where(((np.cos((data["abs_minbatch_slices2"]))) / 2.0) > -998, data["medianbatch_slices2"], data["abs_minbatch_slices2"] )))))) > -998, (-((np.sin((data["medianbatch_slices2"]))))), ((data["stdbatch_slices2"]) / 2.0) )) +

                            0.100000*np.tanh(np.where(np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, (((-(((((5.0)) + (data["minbatch_slices2_msignal"])))))) * 2.0), data["minbatch_slices2_msignal"] ) <= -998, (((5.0)) + (data["minbatch_slices2_msignal"])), np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, data["minbatch_msignal"], ((((data["maxbatch_msignal"]) / 2.0)) * ((((5.0)) + (data["minbatch_msignal"])))) ) )) +

                            0.100000*np.tanh((3.0)) +

                            0.100000*np.tanh(np.cos(((((data["abs_maxbatch_slices2_msignal"]) + ((((((data["abs_minbatch_slices2"]) + (((np.cos((np.tanh((np.where(data["maxtominbatch_msignal"] > -998, (((-3.0)) / 2.0), data["abs_maxbatch_slices2_msignal"] )))))) / 2.0)))/2.0)) * (np.tanh((data["maxtominbatch_msignal"]))))))/2.0)))) +

                            0.100000*np.tanh(np.where(((((((data["maxtominbatch_msignal"]) / 2.0)) * 2.0)) * 2.0) <= -998, np.where(data["stdbatch_slices2"] > -998, np.sin(((((data["meanbatch_slices2_msignal"]) + (np.where(np.cos((np.sin((data["stdbatch_slices2"])))) > -998, data["medianbatch_slices2"], (-((data["meanbatch_msignal"]))) )))/2.0))), data["medianbatch_slices2_msignal"] ), np.sin(((((-((data["meanbatch_msignal"])))) * ((-1.0))))) )) +

                            0.100000*np.tanh(np.where(((np.tanh((np.cos((np.tanh((data["meanbatch_msignal"]))))))) / 2.0) <= -998, data["meanbatch_msignal"], data["stdbatch_slices2"] )) +

                            0.100000*np.tanh(data["mean_abs_chgbatch_slices2_msignal"]) +

                            0.100000*np.tanh(np.cos((np.where(((((data["meanbatch_slices2_msignal"]) / 2.0)) * 2.0) <= -998, np.sin((np.cos(((((-1.0)) + (np.cos((((((np.cos((data["minbatch"]))) - (np.cos((data["abs_avgbatch_msignal"]))))) / 2.0))))))))), data["abs_avgbatch_msignal"] )))) +

                            0.100000*np.tanh(((data["maxbatch_slices2"]) * 2.0)) +

                            0.100000*np.tanh(np.sin((((((((data["minbatch_msignal"]) - (((data["signal_shift_+1_msignal"]) - (np.sin((np.cos((np.cos((np.tanh((np.cos((((((data["minbatch_msignal"]) - (((data["signal_shift_+1_msignal"]) - (np.tanh((((data["minbatch_msignal"]) * 2.0)))))))) * 2.0)))))))))))))))) * 2.0)) * 2.0)))) +

                            0.099609*np.tanh(np.sin((np.where(((((data["maxtominbatch"]) / 2.0)) / 2.0) <= -998, ((np.cos((((data["rangebatch_slices2_msignal"]) + (data["abs_maxbatch_slices2"]))))) / 2.0), np.cos((((data["rangebatch_slices2_msignal"]) + (data["abs_maxbatch_slices2"])))) )))) +

                            0.100000*np.tanh(np.cos(((((data["meanbatch_msignal"]) + (((((((((data["medianbatch_msignal"]) - (((np.sin((np.sin((np.cos(((((np.sin((data["maxtominbatch_msignal"]))) + (((data["medianbatch_slices2_msignal"]) / 2.0)))/2.0)))))))) / 2.0)))) - (data["stdbatch_slices2"]))) / 2.0)) + (((((((data["mean_abs_chgbatch_slices2_msignal"]) + (data["stdbatch_slices2"]))/2.0)) + (data["maxtominbatch_msignal"]))/2.0)))))/2.0)))) +

                            0.100000*np.tanh(np.cos((((data["minbatch_msignal"]) - ((((data["abs_maxbatch_slices2"]) + (((np.where(np.tanh((np.where(data["minbatch_msignal"] <= -998, (((data["minbatch_msignal"]) + (data["minbatch_msignal"]))/2.0), data["abs_maxbatch_slices2"] ))) <= -998, (10.0), (1.0) )) / 2.0)))/2.0)))))) +

                            0.100000*np.tanh((((np.tanh((np.where(np.cos((data["mean_abs_chgbatch_slices2"])) > -998, np.tanh((data["mean_abs_chgbatch_slices2"])), np.tanh((data["abs_maxbatch_slices2_msignal"])) )))) + (data["mean_abs_chgbatch_slices2"]))/2.0)) +

                            0.100000*np.tanh(((data["minbatch"]) + (((((data["abs_avgbatch_msignal"]) + (np.sin((((data["abs_avgbatch_msignal"]) * (np.where(data["rangebatch_slices2_msignal"] <= -998, np.tanh((((data["minbatch"]) + (((np.sin((data["abs_avgbatch_msignal"]))) + (((data["abs_avgbatch_msignal"]) / 2.0))))))), np.tanh((data["abs_avgbatch_msignal"])) )))))))) + (np.sin((data["abs_avgbatch_msignal"]))))))) +

                            0.100000*np.tanh(np.where(data["abs_minbatch_slices2"] <= -998, ((np.cos((data["maxbatch_msignal"]))) + (data["medianbatch_slices2"])), np.cos((np.sin((np.tanh((((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, np.tanh((data["maxbatch_msignal"])), data["stdbatch_msignal"] )) + (np.tanh((data["maxbatch_slices2_msignal"])))))))))) )) +

                            0.100000*np.tanh((1.0)) +

                            0.100000*np.tanh(np.cos((np.where((-2.0) <= -998, np.sin(((14.56566810607910156))), (-((data["minbatch"]))) )))) +

                            0.100000*np.tanh((((((data["minbatch_slices2_msignal"]) + ((((np.sin((((data["mean_abs_chgbatch_slices2"]) / 2.0)))) + (data["stdbatch_slices2_msignal"]))/2.0)))/2.0)) / 2.0)) +

                            0.100000*np.tanh(((data["maxbatch_slices2_msignal"]) * (np.cos((np.cos((np.cos((((data["rangebatch_msignal"]) * (np.where(np.where((10.98459339141845703) <= -998, (-((data["rangebatch_slices2_msignal"]))), data["mean_abs_chgbatch_slices2"] ) > -998, data["mean_abs_chgbatch_slices2"], ((data["meanbatch_slices2_msignal"]) * (np.sin((data["rangebatch_msignal"])))) )))))))))))) +

                            0.100000*np.tanh(np.where((((-((np.sin(((4.0))))))) + ((((data["medianbatch_slices2_msignal"]) + ((((((-(((3.0))))) * 2.0)) * 2.0)))/2.0))) > -998, np.cos((data["medianbatch_slices2_msignal"])), ((((((np.cos((np.tanh((np.cos((data["medianbatch_slices2_msignal"]))))))) * 2.0)) * 2.0)) * 2.0) )) +

                            0.089932*np.tanh(np.sin((((data["abs_minbatch_msignal"]) * (np.sin(((((np.where((((((-((np.sin((data["signal_shift_+1_msignal"])))))) * (np.tanh((np.sin((data["abs_minbatch_msignal"]))))))) / 2.0) > -998, ((((np.sin((((data["abs_maxbatch_slices2_msignal"]) * 2.0)))) / 2.0)) * 2.0), data["maxtominbatch_slices2_msignal"] )) + (np.sin((data["signal_shift_+1_msignal"]))))/2.0)))))))) +

                            0.100000*np.tanh(np.where(((data["stdbatch_slices2"]) / 2.0) > -998, data["mean_abs_chgbatch_slices2"], (((np.where(data["maxbatch_slices2"] <= -998, data["mean_abs_chgbatch_slices2"], data["abs_maxbatch_slices2"] )) + (data["abs_minbatch_slices2"]))/2.0) )) +

                            0.098436*np.tanh(np.cos((np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_slices2_msignal"], ((data["abs_avgbatch_slices2_msignal"]) * (data["abs_avgbatch_slices2_msignal"])) )))) +

                            0.100000*np.tanh(((np.cos((((((np.where(((np.where(np.where(data["minbatch_slices2_msignal"] > -998, data["signal_shift_+1"], (14.02256393432617188) ) > -998, data["medianbatch_slices2"], data["abs_maxbatch_slices2_msignal"] )) - (data["signal_shift_+1"])) > -998, data["medianbatch_slices2"], ((np.where(data["abs_maxbatch_slices2"] > -998, data["meanbatch_slices2"], data["maxtominbatch_slices2"] )) - (data["signal_shift_+1"])) )) - (data["signal_shift_+1"]))) * 2.0)))) * 2.0)) +

                            0.100000*np.tanh(data["mean_abs_chgbatch_slices2"]) +

                            0.100000*np.tanh(np.tanh((np.sin((np.tanh((data["stdbatch_slices2"]))))))) +

                            0.100000*np.tanh(((((((np.where(data["maxtominbatch_slices2_msignal"] > -998, (((data["minbatch"]) + (np.cos((np.sin((data["minbatch"]))))))/2.0), (((data["abs_avgbatch_msignal"]) + ((3.0)))/2.0) )) * (np.sin(((((data["abs_avgbatch_msignal"]) + (((data["maxbatch_slices2_msignal"]) - ((((data["minbatch"]) + (np.cos((data["abs_minbatch_slices2_msignal"]))))/2.0)))))/2.0)))))) * 2.0)) / 2.0)) +

                            0.100000*np.tanh(np.where(data["rangebatch_slices2_msignal"] <= -998, ((data["medianbatch_slices2_msignal"]) - (np.where(((data["abs_avgbatch_slices2"]) - ((-((data["meanbatch_slices2_msignal"]))))) <= -998, data["maxtominbatch"], np.cos((np.tanh((((np.cos(((0.0)))) / 2.0))))) ))), data["rangebatch_slices2"] )) +

                            0.100000*np.tanh(np.cos((np.sin((data["minbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh(np.sin((((((np.where(data["maxbatch_slices2"] <= -998, np.where(data["abs_avgbatch_slices2_msignal"] <= -998, (8.0), (-((data["medianbatch_slices2"]))) ), np.cos((data["abs_avgbatch_slices2_msignal"])) )) - (data["minbatch_msignal"]))) / 2.0)))) +

                            0.093744*np.tanh(((data["maxtominbatch"]) * (np.cos((((np.where(np.sin((((((data["maxtominbatch_msignal"]) / 2.0)) * ((-((((data["abs_maxbatch_slices2_msignal"]) / 2.0)))))))) > -998, data["minbatch_msignal"], ((np.cos((np.cos((((data["minbatch_msignal"]) * 2.0)))))) * (data["maxtominbatch"])) )) * 2.0)))))) +

                            0.100000*np.tanh(np.where(np.sin((((np.cos((data["meanbatch_msignal"]))) / 2.0))) <= -998, data["medianbatch_slices2"], np.cos((((((np.where(data["abs_maxbatch_slices2"] > -998, data["abs_maxbatch"], np.tanh(((((np.tanh((data["signal_shift_+1_msignal"]))) + (data["abs_maxbatch_slices2"]))/2.0))) )) + (((((data["stdbatch_slices2_msignal"]) * 2.0)) / 2.0)))) * 2.0))) )) +

                            0.099707*np.tanh(((data["maxbatch_slices2_msignal"]) + (((data["stdbatch_slices2_msignal"]) + (((np.where(data["abs_minbatch_msignal"] <= -998, data["stdbatch_slices2_msignal"], np.sin((((data["meanbatch_slices2"]) / 2.0))) )) / 2.0)))))) +

                            0.100000*np.tanh(np.cos(((((((((2.0)) - (data["mean_abs_chgbatch_msignal"]))) * 2.0)) + (np.tanh(((((-((((np.sin(((((((np.cos((np.where(data["minbatch_slices2"] <= -998, np.where((0.0) > -998, ((np.cos((data["minbatch"]))) * 2.0), (-1.0) ), data["stdbatch_msignal"] )))) / 2.0)) + (data["minbatch"]))/2.0)))) * 2.0))))) / 2.0)))))))) +

                            0.100000*np.tanh(((np.tanh((np.tanh((data["mean_abs_chgbatch_slices2_msignal"]))))) / 2.0)) +

                            0.100000*np.tanh(np.where(np.tanh((np.where(data["stdbatch_slices2"] <= -998, (8.0), data["maxbatch_slices2"] ))) > -998, data["maxbatch_slices2"], data["maxbatch_slices2"] )))  

    

    def GP_class_3(self,data):

        return self.Output( -2.012666 +

                            0.100000*np.tanh(((np.sin((((data["minbatch_msignal"]) + (np.where(np.where(data["maxtominbatch_slices2_msignal"] <= -998, np.sin((((data["maxbatch_msignal"]) * 2.0))), (-((data["abs_minbatch_slices2_msignal"]))) ) > -998, (0.0), (0.0) )))))) * 2.0)) +

                            0.100000*np.tanh(((np.where(data["minbatch_msignal"] > -998, np.sin((data["minbatch_msignal"])), np.where(((np.sin((np.sin((data["minbatch_msignal"]))))) + (np.sin((data["minbatch_msignal"])))) <= -998, np.tanh(((((-3.0)) * 2.0))), data["minbatch_msignal"] ) )) * 2.0)) +

                            0.100000*np.tanh(((np.where((-3.0) > -998, np.sin((np.where(data["minbatch_msignal"] > -998, data["minbatch_msignal"], data["rangebatch_slices2"] ))), data["medianbatch_slices2_msignal"] )) * 2.0)) +

                            0.100000*np.tanh(np.sin((np.where(np.where(data["minbatch_msignal"] > -998, np.where(data["signal_shift_-1_msignal"] > -998, (-((data["meanbatch_slices2"]))), np.where(((data["minbatch_msignal"]) + (data["abs_maxbatch_slices2_msignal"])) > -998, data["signal_shift_-1_msignal"], data["abs_maxbatch_msignal"] ) ), data["abs_maxbatch_slices2_msignal"] ) > -998, data["minbatch_msignal"], np.sin((np.sin((data["signal_shift_-1_msignal"])))) )))) +

                            0.100000*np.tanh(((((np.where((0.0) > -998, np.sin((data["minbatch_msignal"])), (((0.0)) + (((np.sin((data["signal_shift_-1"]))) * ((-((np.tanh((((data["maxtominbatch_msignal"]) + (np.where(data["rangebatch_slices2_msignal"] <= -998, data["rangebatch_slices2_msignal"], data["signal"] )))))))))))) )) * (data["rangebatch_slices2_msignal"]))) - ((-((((np.cos((data["abs_maxbatch_msignal"]))) * 2.0))))))) +

                            0.100000*np.tanh(((data["rangebatch_slices2_msignal"]) * ((-((np.where(np.where(data["abs_minbatch_slices2_msignal"] > -998, data["abs_minbatch_slices2_msignal"], data["abs_avgbatch_msignal"] ) > -998, data["abs_minbatch_slices2_msignal"], (((data["meanbatch_slices2_msignal"]) + (data["abs_avgbatch_msignal"]))/2.0) ))))))) +

                            0.100000*np.tanh(np.where(np.where(data["mean_abs_chgbatch_slices2"] > -998, data["signal_shift_+1"], data["abs_minbatch_slices2_msignal"] ) > -998, np.sin((np.sin((data["minbatch_slices2_msignal"])))), (((data["abs_minbatch_slices2"]) + (((data["abs_minbatch_slices2"]) * 2.0)))/2.0) )) +

                            0.100000*np.tanh(((np.where(np.sin((data["minbatch_msignal"])) > -998, np.sin((data["minbatch_msignal"])), (((np.sin((data["minbatch_msignal"]))) + (data["minbatch_msignal"]))/2.0) )) * 2.0)) +

                            0.100000*np.tanh(np.where(np.cos((((data["rangebatch_slices2"]) * 2.0))) > -998, ((np.cos((((data["rangebatch_slices2"]) - (data["meanbatch_slices2_msignal"]))))) * 2.0), (((data["rangebatch_msignal"]) + (((data["rangebatch_slices2"]) * (np.sin((np.tanh((data["maxtominbatch_slices2"]))))))))/2.0) )) +

                            0.100000*np.tanh(((np.where(np.sin((np.where(np.sin((np.sin((data["minbatch_slices2_msignal"])))) > -998, np.sin((data["minbatch_slices2_msignal"])), (-((data["minbatch_slices2_msignal"]))) ))) > -998, np.sin((((np.sin((data["minbatch_slices2_msignal"]))) * 2.0))), np.tanh((data["minbatch_slices2_msignal"])) )) * 2.0)) +

                            0.100000*np.tanh((((np.sin((np.where((6.0) <= -998, np.sin((data["abs_maxbatch_msignal"])), np.tanh((data["rangebatch_msignal"])) )))) + (((np.sin((data["minbatch_msignal"]))) * (data["rangebatch_msignal"]))))/2.0)) +

                            0.100000*np.tanh(np.sin((np.where(np.tanh((data["minbatch_msignal"])) <= -998, data["meanbatch_slices2"], np.where(data["minbatch_msignal"] > -998, data["minbatch_msignal"], np.where(((np.sin((np.where(data["stdbatch_msignal"] > -998, data["minbatch_msignal"], data["abs_maxbatch_slices2"] )))) + (data["abs_avgbatch_slices2_msignal"])) <= -998, data["abs_avgbatch_slices2_msignal"], data["minbatch_msignal"] ) ) )))) +

                            0.100000*np.tanh(np.where(((((data["maxbatch_msignal"]) * 2.0)) * 2.0) <= -998, (6.23791122436523438), ((np.where(np.sin((np.sin((((data["maxbatch_msignal"]) * 2.0))))) <= -998, ((data["abs_maxbatch_slices2"]) / 2.0), ((np.sin((np.sin((((data["maxbatch_msignal"]) * 2.0)))))) * 2.0) )) * 2.0) )) +

                            0.100000*np.tanh(((np.sin((np.where(data["abs_maxbatch_msignal"] > -998, data["minbatch_msignal"], np.where(data["minbatch_msignal"] > -998, data["rangebatch_msignal"], np.where(data["signal_shift_+1"] > -998, np.where(np.where(data["minbatch_msignal"] <= -998, data["maxtominbatch_msignal"], np.sin((np.where(data["abs_maxbatch_msignal"] > -998, data["minbatch_msignal"], data["minbatch_msignal"] ))) ) > -998, data["minbatch_msignal"], np.cos(((-((data["signal_shift_+1"]))))) ), data["medianbatch_slices2"] ) ) )))) * 2.0)) +

                            0.100000*np.tanh(((((data["minbatch_msignal"]) + (((((np.sin((data["minbatch_msignal"]))) * 2.0)) * 2.0)))) + (((((np.sin((np.sin((data["minbatch_msignal"]))))) * 2.0)) * 2.0)))) +

                            0.100000*np.tanh(((((np.cos((data["signal"]))) * 2.0)) + (np.cos((data["signal"]))))) +

                            0.100000*np.tanh(np.sin((((((((((((data["minbatch_msignal"]) + (((((np.tanh(((((2.0)) + ((0.0)))))) / 2.0)) + (data["meanbatch_slices2_msignal"]))))) * (data["stdbatch_slices2_msignal"]))) / 2.0)) + (data["minbatch_msignal"]))) - (np.cos((data["minbatch_msignal"]))))))) +

                            0.100000*np.tanh(np.sin((np.where(data["minbatch_msignal"] > -998, data["minbatch_msignal"], np.sin(((((np.tanh((data["signal_shift_+1_msignal"]))) + (data["abs_avgbatch_slices2_msignal"]))/2.0))) )))) +

                            0.100000*np.tanh(np.sin((np.where(data["abs_maxbatch_msignal"] > -998, data["minbatch_msignal"], ((data["signal_shift_+1"]) + (((data["stdbatch_slices2_msignal"]) * 2.0))) )))) +

                            0.100000*np.tanh(((((data["stdbatch_slices2_msignal"]) + (np.where(np.sin((data["minbatch_slices2_msignal"])) <= -998, ((np.tanh((data["signal"]))) - (data["rangebatch_slices2_msignal"])), np.sin((data["abs_minbatch_msignal"])) )))) * (((np.sin(((((data["minbatch_msignal"]) + (np.sin((data["abs_minbatch_msignal"]))))/2.0)))) * 2.0)))) +

                            0.100000*np.tanh((((5.76851034164428711)) * (np.cos((np.where(np.where((3.36750698089599609) > -998, data["mean_abs_chgbatch_slices2"], np.where((5.76851034164428711) <= -998, ((data["mean_abs_chgbatch_slices2"]) * 2.0), (((data["signal"]) + (((((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0)) - (data["abs_maxbatch_msignal"]))))/2.0) ) ) > -998, ((data["mean_abs_chgbatch_slices2"]) * 2.0), data["signal"] )))))) +

                            0.100000*np.tanh(((np.cos((((((((data["stdbatch_msignal"]) + (((((((data["stdbatch_msignal"]) + ((((((data["stdbatch_msignal"]) / 2.0)) + (data["stdbatch_msignal"]))/2.0)))/2.0)) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)))/2.0)) + (np.where(((data["stdbatch_msignal"]) * 2.0) <= -998, ((data["stdbatch_msignal"]) * 2.0), ((data["stdbatch_msignal"]) * 2.0) )))/2.0)))) * 2.0)) +

                            0.100000*np.tanh((-((np.where(np.tanh((data["rangebatch_slices2"])) <= -998, ((np.cos((data["rangebatch_slices2_msignal"]))) / 2.0), ((data["minbatch_slices2_msignal"]) * (np.sin((((((data["rangebatch_slices2"]) + (np.where((3.37196302413940430) <= -998, ((((((data["minbatch_slices2_msignal"]) + (data["signal_shift_+1_msignal"]))) * 2.0)) * 2.0), data["minbatch_slices2_msignal"] )))) * 2.0))))) ))))) +

                            0.100000*np.tanh(((np.cos((np.where(np.cos((data["abs_avgbatch_slices2_msignal"])) > -998, data["abs_avgbatch_slices2_msignal"], ((((np.sin((data["abs_maxbatch_slices2"]))) + (((np.sin(((3.0)))) - ((((5.0)) * 2.0)))))) * 2.0) )))) * (((data["rangebatch_msignal"]) - (np.cos((data["abs_maxbatch_slices2_msignal"]))))))) +

                            0.100000*np.tanh(((data["medianbatch_slices2_msignal"]) * (((np.where(((((data["rangebatch_msignal"]) + (data["stdbatch_slices2"]))) - (data["meanbatch_msignal"])) <= -998, data["medianbatch_slices2_msignal"], ((data["rangebatch_msignal"]) * (((data["rangebatch_msignal"]) * (np.cos((np.where(((data["stdbatch_slices2"]) - (data["medianbatch_slices2_msignal"])) > -998, data["stdbatch_slices2"], data["mean_abs_chgbatch_slices2_msignal"] ))))))) )) * (data["medianbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh(((np.sin((((data["maxbatch_msignal"]) * 2.0)))) + (np.where(data["rangebatch_slices2_msignal"] > -998, np.sin((((data["maxbatch_msignal"]) * 2.0))), np.where(((data["maxbatch_msignal"]) * 2.0) > -998, data["rangebatch_slices2_msignal"], (((-((np.tanh((np.sin((((data["maxbatch_msignal"]) * 2.0))))))))) * 2.0) ) )))) +

                            0.100000*np.tanh(np.sin((((np.where(((np.where(((((np.where(data["mean_abs_chgbatch_slices2"] > -998, data["maxbatch_slices2_msignal"], data["abs_maxbatch_msignal"] )) * 2.0)) * 2.0) > -998, data["maxbatch_slices2_msignal"], data["stdbatch_slices2_msignal"] )) * 2.0) > -998, data["maxbatch_slices2_msignal"], data["maxbatch_slices2_msignal"] )) * 2.0)))) +

                            0.100000*np.tanh(np.where(data["rangebatch_slices2"] <= -998, np.where(data["signal"] <= -998, (((2.0)) * (data["stdbatch_slices2_msignal"])), (((-((data["abs_avgbatch_slices2"])))) / 2.0) ), ((data["maxbatch_slices2"]) * (((np.sin((((np.where(data["medianbatch_msignal"] > -998, data["abs_maxbatch_msignal"], np.sin(((((-2.0)) * 2.0))) )) * 2.0)))) / 2.0))) )) +

                            0.100000*np.tanh(((((((np.sin((np.where(data["rangebatch_msignal"] > -998, data["minbatch_msignal"], np.where(data["abs_maxbatch"] > -998, data["minbatch_msignal"], data["rangebatch_msignal"] ) )))) / 2.0)) / 2.0)) + (((data["abs_maxbatch_msignal"]) * (np.sin((((data["abs_avgbatch_msignal"]) + (data["stdbatch_slices2_msignal"]))))))))) +

                            0.100000*np.tanh(((np.sin((np.where(((np.sin((np.where(((data["maxbatch_slices2_msignal"]) * 2.0) > -998, ((data["maxbatch_slices2_msignal"]) * 2.0), ((data["medianbatch_slices2_msignal"]) * 2.0) )))) * 2.0) > -998, ((data["maxbatch_slices2_msignal"]) * 2.0), np.where(data["rangebatch_msignal"] > -998, data["minbatch_msignal"], np.cos((np.sin((data["minbatch_msignal"])))) ) )))) * 2.0)) +

                            0.100000*np.tanh(((np.where(data["abs_maxbatch_msignal"] <= -998, np.where(data["stdbatch_msignal"] > -998, data["abs_maxbatch_msignal"], ((data["meanbatch_msignal"]) * 2.0) ), ((np.where(((data["stdbatch_slices2"]) * 2.0) > -998, data["abs_maxbatch_msignal"], ((np.cos((data["stdbatch_slices2"]))) + ((2.0))) )) * (np.cos((data["stdbatch_slices2"])))) )) * 2.0)) +

                            0.100000*np.tanh(((np.cos((np.where((-((data["abs_minbatch_slices2"]))) > -998, data["mean_abs_chgbatch_slices2_msignal"], (-((((((np.where(np.cos(((1.0))) > -998, data["mean_abs_chgbatch_slices2_msignal"], data["meanbatch_slices2"] )) * 2.0)) / 2.0)))) )))) * 2.0)) +

                            0.100000*np.tanh(((np.cos((np.where(data["rangebatch_slices2"] <= -998, ((np.sin((np.cos((np.where((((14.13066387176513672)) * (np.tanh((data["abs_avgbatch_msignal"])))) <= -998, data["rangebatch_slices2_msignal"], data["abs_avgbatch_slices2_msignal"] )))))) * 2.0), data["abs_avgbatch_slices2_msignal"] )))) * 2.0)) +

                            0.100000*np.tanh(np.sin((((((data["maxbatch_slices2_msignal"]) + (((np.cos((np.cos((((((np.cos((np.cos((((data["maxbatch_slices2_msignal"]) * (((np.sin(((((2.0)) * 2.0)))) / 2.0)))))))) / 2.0)) * (np.sin((np.sin((((data["maxbatch_slices2_msignal"]) * 2.0)))))))))))) / 2.0)))) * 2.0)))) +

                            0.100000*np.tanh(np.where((-((np.cos(((((data["maxtominbatch_slices2"]) + (np.sin((np.tanh((((np.sin((((np.sin((data["minbatch_msignal"]))) * (np.where(data["minbatch_msignal"] > -998, (2.0), np.cos((data["stdbatch_slices2"])) )))))) / 2.0)))))))/2.0)))))) <= -998, data["meanbatch_msignal"], ((np.sin((data["minbatch_msignal"]))) * ((2.0))) )) +

                            0.100000*np.tanh(((((np.cos((((((data["rangebatch_slices2"]) - (data["meanbatch_msignal"]))) + (np.cos((data["meanbatch_msignal"]))))))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.where(((data["abs_maxbatch"]) - (((np.cos((np.tanh(((-((data["abs_maxbatch_slices2"])))))))) * 2.0))) <= -998, ((data["abs_maxbatch_slices2"]) * (data["signal_shift_-1"])), ((((np.cos((data["abs_avgbatch_slices2_msignal"]))) * (data["stdbatch_slices2"]))) * (((data["abs_maxbatch"]) * (data["abs_maxbatch_slices2"])))) )) +

                            0.100000*np.tanh(np.cos(((((3.0)) * (np.cos((data["minbatch_msignal"]))))))) +

                            0.100000*np.tanh(((np.sin((np.where(np.where(((np.where(data["minbatch_msignal"] > -998, data["abs_minbatch_msignal"], np.sin((data["abs_minbatch_slices2"])) )) / 2.0) <= -998, (-((((data["medianbatch_slices2"]) / 2.0)))), data["rangebatch_msignal"] ) <= -998, data["minbatch_msignal"], ((data["maxbatch_msignal"]) * 2.0) )))) * 2.0)) +

                            0.100000*np.tanh(np.sin((np.sin((((np.where(data["signal_shift_+1"] > -998, data["maxbatch_slices2_msignal"], np.where(data["meanbatch_msignal"] > -998, ((np.tanh(((-1.0)))) * (((np.sin((((data["maxbatch_slices2_msignal"]) * (((data["maxbatch_slices2_msignal"]) * (data["rangebatch_msignal"]))))))) / 2.0))), data["maxbatch_slices2_msignal"] ) )) * 2.0)))))) +

                            0.100000*np.tanh(np.where(np.sin((np.sin((data["minbatch_msignal"])))) <= -998, np.where(data["minbatch_msignal"] > -998, (-(((2.0)))), np.sin((data["abs_maxbatch_slices2"])) ), np.where(np.tanh((data["meanbatch_msignal"])) <= -998, (5.0), np.sin((data["minbatch_msignal"])) ) )) +

                            0.100000*np.tanh(((((np.cos((data["stdbatch_slices2"]))) * (np.where(data["signal"] > -998, data["rangebatch_msignal"], np.cos((data["stdbatch_slices2"])) )))) * (np.where(data["maxbatch_slices2"] <= -998, data["signal_shift_+1"], np.where(data["maxbatch_slices2_msignal"] > -998, data["signal"], (((-((np.cos((np.tanh((data["stdbatch_slices2_msignal"])))))))) / 2.0) ) )))) +

                            0.100000*np.tanh(((np.sin((data["maxbatch_msignal"]))) * ((-((((data["meanbatch_slices2"]) * 2.0))))))) +

                            0.100000*np.tanh(((((np.sin((np.where(np.where(data["minbatch_slices2_msignal"] <= -998, data["signal_shift_+1_msignal"], (-3.0) ) > -998, data["minbatch_slices2_msignal"], np.sin((np.where(np.where(data["minbatch_slices2_msignal"] <= -998, data["abs_minbatch_slices2"], (-3.0) ) > -998, np.sin((data["mean_abs_chgbatch_slices2"])), data["minbatch_slices2_msignal"] ))) )))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.sin((((np.where(((data["stdbatch_msignal"]) + (np.tanh(((((-2.0)) + (data["abs_maxbatch_msignal"])))))) <= -998, np.sin((((data["mean_abs_chgbatch_msignal"]) * ((-(((1.0)))))))), (3.0) )) * ((-(((((3.0)) - ((-((data["meanbatch_slices2_msignal"]))))))))))))) +

                            0.100000*np.tanh(((np.cos((((np.where((((-3.0)) + (np.sin((((data["maxbatch_slices2_msignal"]) * 2.0))))) > -998, ((data["maxbatch_slices2_msignal"]) * ((-3.0))), ((np.where((((0.0)) + (np.where(data["signal_shift_-1"] > -998, ((data["maxbatch_slices2_msignal"]) * ((-3.0))), data["maxbatch_slices2_msignal"] ))) > -998, (-3.0), data["maxbatch_slices2_msignal"] )) / 2.0) )) / 2.0)))) * 2.0)) +

                            0.100000*np.tanh(np.where((1.0) > -998, np.cos((((data["meanbatch_msignal"]) - (data["rangebatch_slices2"])))), ((data["rangebatch_slices2"]) / 2.0) )) +

                            0.100000*np.tanh(((((np.where(np.where(data["minbatch_slices2_msignal"] <= -998, data["minbatch_slices2_msignal"], (-((np.cos((data["stdbatch_slices2"]))))) ) <= -998, ((data["stdbatch_slices2"]) - (data["minbatch_slices2_msignal"])), np.sin((((np.cos((((np.sin((((data["maxbatch_msignal"]) * 2.0)))) * 2.0)))) * 2.0))) )) - (data["minbatch_slices2_msignal"]))) * (((np.sin((((data["maxbatch_msignal"]) * 2.0)))) * 2.0)))) +

                            0.100000*np.tanh(((data["abs_minbatch_msignal"]) - (((data["medianbatch_slices2_msignal"]) * ((((6.32978200912475586)) * (((((3.0)) + ((((data["minbatch_msignal"]) + (np.where(data["abs_maxbatch_msignal"] <= -998, ((((data["minbatch_msignal"]) / 2.0)) * 2.0), np.where(data["minbatch_msignal"] > -998, data["abs_maxbatch_msignal"], (-((np.sin((data["minbatch_slices2_msignal"]))))) ) )))/2.0)))/2.0)))))))) +

                            0.100000*np.tanh(np.sin((np.where((((14.78631019592285156)) + (((np.sin((data["maxbatch_slices2_msignal"]))) - ((-((data["medianbatch_slices2_msignal"]))))))) > -998, data["minbatch_slices2_msignal"], np.sin((data["signal_shift_-1_msignal"])) )))) +

                            0.100000*np.tanh(((np.cos(((-(((-((((np.tanh((data["minbatch_slices2_msignal"]))) + (np.where(np.cos((np.sin((((np.cos((data["minbatch_slices2_msignal"]))) * ((-((data["minbatch_msignal"]))))))))) > -998, ((data["maxbatch_slices2_msignal"]) * 2.0), np.sin(((-(((2.0)))))) )))))))))))) * 2.0)) +

                            0.100000*np.tanh(((np.sin(((((data["rangebatch_msignal"]) + ((((data["signal"]) + (np.where(data["rangebatch_msignal"] > -998, (-2.0), data["rangebatch_slices2"] )))/2.0)))/2.0)))) * 2.0)) +

                            0.100000*np.tanh(np.sin((np.where(((data["abs_maxbatch_slices2_msignal"]) * 2.0) > -998, np.cos((((((((data["stdbatch_slices2"]) * 2.0)) + (data["abs_maxbatch_msignal"]))) * 2.0))), data["maxtominbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh(np.where(((data["abs_maxbatch_msignal"]) * (data["minbatch_slices2_msignal"])) <= -998, (-2.0), np.where((((data["stdbatch_slices2"]) + (data["minbatch"]))/2.0) <= -998, (-2.0), np.cos((((((np.cos((data["stdbatch_slices2"]))) - (data["maxbatch_slices2_msignal"]))) * 2.0))) ) )) +

                            0.100000*np.tanh(np.cos((((np.where(data["abs_maxbatch"] <= -998, (((data["stdbatch_msignal"]) + (((np.where(np.where(np.where(data["abs_maxbatch"] > -998, data["stdbatch_slices2_msignal"], data["stdbatch_msignal"] ) > -998, (-3.0), (-3.0) ) <= -998, (1.0), np.sin(((((((data["mean_abs_chgbatch_slices2_msignal"]) + ((-3.0)))/2.0)) - ((-((data["abs_avgbatch_slices2_msignal"]))))))) )) * 2.0)))/2.0), data["maxbatch_slices2_msignal"] )) * 2.0)))) +

                            0.100000*np.tanh(((np.cos(((((((data["abs_avgbatch_slices2_msignal"]) + (np.where(data["medianbatch_slices2_msignal"] > -998, np.where(data["signal_shift_-1_msignal"] > -998, data["abs_minbatch_msignal"], data["rangebatch_slices2"] ), np.tanh((np.where((2.0) <= -998, (5.04165315628051758), np.cos((((np.cos((data["minbatch_slices2_msignal"]))) - (data["signal_shift_-1_msignal"])))) ))) )))/2.0)) * 2.0)))) * 2.0)) +

                            0.100000*np.tanh(np.where(np.tanh((data["stdbatch_slices2"])) > -998, np.sin((data["minbatch_slices2_msignal"])), data["minbatch_msignal"] )) +

                            0.100000*np.tanh((((-((np.cos(((-((((data["abs_maxbatch_msignal"]) * (np.tanh((np.where((((-((data["maxbatch_msignal"])))) * 2.0) <= -998, np.where(data["maxbatch_msignal"] <= -998, np.where(data["maxbatch_msignal"] <= -998, data["medianbatch_slices2_msignal"], (-((np.cos((data["rangebatch_msignal"]))))) ), data["medianbatch_slices2_msignal"] ), data["medianbatch_msignal"] )))))))))))))) * 2.0)) +

                            0.100000*np.tanh(((data["minbatch_msignal"]) * (np.tanh((np.sin(((((((np.sin((np.where(np.where((2.0) <= -998, data["abs_maxbatch_slices2_msignal"], data["medianbatch_slices2"] ) > -998, data["abs_avgbatch_msignal"], (((-1.0)) * (data["signal_shift_+1_msignal"])) )))) + (data["abs_maxbatch_slices2_msignal"]))) + (data["minbatch_msignal"]))/2.0)))))))) +

                            0.100000*np.tanh((((((((np.where(((data["minbatch"]) * 2.0) > -998, data["stdbatch_msignal"], data["minbatch_slices2_msignal"] )) + (np.cos(((1.0)))))/2.0)) * 2.0)) * 2.0)) +

                            0.100000*np.tanh((-((((np.where(data["minbatch"] > -998, data["abs_avgbatch_msignal"], ((data["mean_abs_chgbatch_slices2"]) - (data["mean_abs_chgbatch_slices2_msignal"])) )) - (np.cos(((-((np.where(((data["meanbatch_slices2_msignal"]) + (data["minbatch"])) > -998, data["maxtominbatch_slices2"], (((data["minbatch"]) + (np.tanh((np.cos((data["abs_avgbatch_msignal"]))))))/2.0) )))))))))))) +

                            0.100000*np.tanh(np.where((((0.98785066604614258)) + (((np.tanh((np.sin((data["mean_abs_chgbatch_slices2"]))))) + ((((-((data["minbatch_msignal"])))) * (data["maxbatch_slices2_msignal"])))))) <= -998, data["stdbatch_msignal"], ((data["minbatch_msignal"]) + (((((((data["minbatch_msignal"]) + (((((((np.sin((np.sin((data["minbatch_msignal"]))))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) * 2.0))) )) +

                            0.100000*np.tanh((((((np.tanh((np.tanh((((((((data["minbatch_msignal"]) - (data["maxtominbatch"]))) * 2.0)) - (((np.sin(((((3.0)) - (data["mean_abs_chgbatch_msignal"]))))) - ((10.0)))))))))) + (data["stdbatch_slices2_msignal"]))/2.0)) * (((data["mean_abs_chgbatch_msignal"]) * (((((data["minbatch_msignal"]) - (data["maxtominbatch"]))) * 2.0)))))) +

                            0.100000*np.tanh(((data["medianbatch_slices2_msignal"]) * (((data["maxtominbatch_slices2_msignal"]) - (np.where((-((data["rangebatch_slices2"]))) <= -998, ((data["maxtominbatch_slices2_msignal"]) - (((data["medianbatch_slices2_msignal"]) * (((data["maxtominbatch_slices2_msignal"]) - (np.tanh((np.tanh((np.where(data["minbatch"] <= -998, (-1.0), np.cos((np.sin((data["abs_minbatch_slices2_msignal"])))) ))))))))))), ((data["minbatch_slices2_msignal"]) * 2.0) )))))) +

                            0.100000*np.tanh(((np.cos(((((data["rangebatch_msignal"]) + (((((data["stdbatch_msignal"]) + (np.where((-3.0) > -998, ((data["abs_avgbatch_msignal"]) * ((-3.0))), (((((-(((((-(((-3.0))))) * 2.0))))) * ((-3.0)))) + (data["abs_avgbatch_msignal"])) )))) * 2.0)))/2.0)))) * 2.0)) +

                            0.100000*np.tanh((((((((data["signal"]) + (data["signal"]))/2.0)) * (data["stdbatch_slices2"]))) * (((np.sin((((((((((data["signal"]) + (data["rangebatch_slices2_msignal"]))/2.0)) * (data["stdbatch_slices2"]))) + (np.sin((((((((((data["rangebatch_slices2_msignal"]) + (data["signal"]))/2.0)) * (data["stdbatch_slices2"]))) + (np.sin((np.sin((data["stdbatch_slices2"]))))))/2.0)))))/2.0)))) * 2.0)))) +

                            0.100000*np.tanh((-((np.cos((np.where(np.cos((np.where(np.cos(((-2.0))) <= -998, np.cos((((data["abs_minbatch_slices2"]) / 2.0))), np.cos(((((-2.0)) * ((-2.0))))) ))) > -998, (((-2.0)) * (data["medianbatch_slices2_msignal"])), np.cos((data["medianbatch_slices2_msignal"])) ))))))) +

                            0.100000*np.tanh(((np.where(data["rangebatch_slices2_msignal"] <= -998, np.sin((data["stdbatch_slices2_msignal"])), (((((((3.0)) - (data["maxtominbatch_slices2"]))) + (((data["mean_abs_chgbatch_msignal"]) * 2.0)))) * (data["signal_shift_-1"])) )) - (data["abs_avgbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(((((np.cos((np.where(np.cos((np.where(data["minbatch"] > -998, np.where(data["abs_maxbatch_slices2_msignal"] <= -998, np.cos((data["abs_avgbatch_msignal"])), data["medianbatch_msignal"] ), ((np.tanh((data["abs_avgbatch_msignal"]))) * 2.0) ))) > -998, ((data["abs_maxbatch_slices2_msignal"]) * 2.0), data["signal"] )))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(((((np.cos(((((data["minbatch_slices2_msignal"]) + (((data["abs_maxbatch_msignal"]) + (((data["stdbatch_slices2"]) + (np.cos((data["abs_maxbatch_msignal"]))))))))/2.0)))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(data["signal_shift_+1"]) +

                            0.100000*np.tanh(np.where(np.sin((data["rangebatch_slices2_msignal"])) <= -998, np.where(data["maxtominbatch_slices2_msignal"] <= -998, (((((0.0)) + (np.where((3.0) > -998, ((data["medianbatch_slices2_msignal"]) - (np.tanh((data["maxbatch_slices2"])))), (3.0) )))) * 2.0), data["medianbatch_msignal"] ), np.cos(((((data["minbatch_msignal"]) + ((3.0)))/2.0))) )) +

                            0.100000*np.tanh(np.where(data["maxtominbatch"] > -998, ((data["meanbatch_msignal"]) * (((data["maxtominbatch"]) + (data["medianbatch_slices2"])))), data["maxtominbatch_msignal"] )) +

                            0.100000*np.tanh(((((data["stdbatch_slices2"]) * 2.0)) * (np.sin((((np.where((((-((((((np.where(((((data["stdbatch_slices2"]) * 2.0)) * 2.0) <= -998, data["maxbatch_slices2_msignal"], data["signal_shift_-1_msignal"] )) * (data["maxbatch_slices2_msignal"]))) * 2.0))))) / 2.0) <= -998, ((((data["stdbatch_slices2"]) * 2.0)) * 2.0), ((data["stdbatch_slices2"]) * 2.0) )) * (data["maxbatch_slices2_msignal"]))))))) +

                            0.100000*np.tanh(((((data["minbatch_msignal"]) - (data["maxtominbatch"]))) + (np.cos((np.cos(((((np.cos((((((data["minbatch_msignal"]) - (np.where(data["maxtominbatch"] <= -998, data["maxtominbatch"], data["maxtominbatch"] )))) + (data["abs_avgbatch_slices2_msignal"]))))) + (data["maxtominbatch"]))/2.0)))))))) +

                            0.100000*np.tanh(np.sin((np.where((-((((data["maxbatch_msignal"]) * (np.where(data["medianbatch_msignal"] <= -998, data["abs_minbatch_slices2"], (((-(((0.0))))) * 2.0) )))))) <= -998, ((data["maxbatch_msignal"]) * 2.0), ((data["maxbatch_msignal"]) * 2.0) )))) +

                            0.100000*np.tanh((((-((np.cos((data["abs_maxbatch_msignal"])))))) * 2.0)) +

                            0.100000*np.tanh(((data["abs_maxbatch_msignal"]) * (np.where(data["meanbatch_msignal"] <= -998, (-2.0), np.where(np.where(data["rangebatch_slices2_msignal"] <= -998, data["abs_maxbatch_slices2_msignal"], (-2.0) ) <= -998, data["abs_maxbatch_msignal"], np.cos((np.where(data["meanbatch_msignal"] <= -998, (-1.0), np.where((11.08011150360107422) <= -998, data["abs_maxbatch_slices2_msignal"], data["meanbatch_msignal"] ) ))) ) )))) +

                            0.100000*np.tanh((-((np.sin((np.where((-((np.tanh((((np.where((((data["abs_maxbatch_slices2_msignal"]) + (data["minbatch_slices2_msignal"]))/2.0) > -998, data["medianbatch_slices2"], data["minbatch_slices2_msignal"] )) / 2.0)))))) > -998, (((data["abs_maxbatch_slices2_msignal"]) + (data["minbatch_slices2_msignal"]))/2.0), data["minbatch_slices2_msignal"] ))))))) +

                            0.100000*np.tanh(np.where(data["abs_maxbatch_slices2_msignal"] <= -998, np.tanh((np.where(data["abs_maxbatch_msignal"] > -998, np.where(data["mean_abs_chgbatch_msignal"] > -998, data["signal_shift_+1_msignal"], data["maxbatch_msignal"] ), np.cos((np.where((-2.0) > -998, data["abs_avgbatch_slices2_msignal"], data["maxbatch_msignal"] ))) ))), np.sin((((data["maxbatch_msignal"]) * 2.0))) )) +

                            0.100000*np.tanh(((np.where(((np.where(data["minbatch_msignal"] <= -998, data["signal_shift_-1"], data["maxbatch_msignal"] )) * 2.0) > -998, np.sin((data["minbatch_msignal"])), data["signal_shift_-1_msignal"] )) - ((-((data["signal_shift_-1_msignal"])))))) +

                            0.100000*np.tanh(np.sin((((data["minbatch_msignal"]) + ((-((np.where(((np.where(((data["meanbatch_msignal"]) / 2.0) <= -998, data["meanbatch_msignal"], ((((data["signal_shift_-1"]) + (data["mean_abs_chgbatch_msignal"]))) + (data["mean_abs_chgbatch_msignal"])) )) + (data["meanbatch_msignal"])) <= -998, ((data["minbatch_msignal"]) + (data["meanbatch_msignal"])), ((data["meanbatch_msignal"]) / 2.0) ))))))))) +

                            0.100000*np.tanh(((((data["minbatch_slices2_msignal"]) + (np.where(((data["medianbatch_msignal"]) * (data["mean_abs_chgbatch_slices2_msignal"])) > -998, np.where(np.where(data["medianbatch_msignal"] > -998, data["stdbatch_slices2"], data["medianbatch_msignal"] ) <= -998, np.tanh((np.tanh((data["abs_minbatch_slices2_msignal"])))), data["medianbatch_msignal"] ), np.tanh(((((data["minbatch_slices2_msignal"]) + (data["abs_minbatch_slices2_msignal"]))/2.0))) )))) - (np.tanh((np.cos(((0.0)))))))) +

                            0.100000*np.tanh((((10.0)) * (np.cos((((data["meanbatch_slices2_msignal"]) - (np.where(data["rangebatch_slices2"] > -998, data["rangebatch_slices2"], ((data["meanbatch_slices2_msignal"]) * (np.cos((data["maxbatch_slices2_msignal"])))) )))))))) +

                            0.100000*np.tanh(np.where(data["maxbatch_slices2_msignal"] > -998, data["signal_shift_+1_msignal"], np.where(((np.where(np.sin(((((1.0)) / 2.0))) > -998, data["abs_avgbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] )) * 2.0) > -998, data["signal_shift_+1_msignal"], data["abs_maxbatch_slices2"] ) )) +

                            0.100000*np.tanh(np.sin((((data["abs_minbatch_slices2_msignal"]) - (((np.where((((((data["abs_maxbatch_slices2_msignal"]) - (np.tanh(((3.0)))))) + (np.where((((np.cos((data["abs_avgbatch_msignal"]))) + (data["rangebatch_slices2"]))/2.0) > -998, data["rangebatch_msignal"], (3.0) )))/2.0) <= -998, (2.0), ((data["rangebatch_slices2"]) + (data["abs_avgbatch_msignal"])) )) * 2.0)))))) +

                            0.100000*np.tanh(np.where(data["abs_maxbatch_slices2_msignal"] <= -998, ((data["signal_shift_+1"]) * 2.0), np.cos((((data["abs_maxbatch_slices2_msignal"]) * 2.0))) )) +

                            0.100000*np.tanh((((8.66331195831298828)) + (((((((data["signal"]) * (data["minbatch"]))) - (((data["signal_shift_+1"]) / 2.0)))) - (np.sin((data["meanbatch_slices2"]))))))) +

                            0.100000*np.tanh(((np.cos((((((2.0)) + (((data["minbatch_msignal"]) * (np.where(((data["minbatch_msignal"]) * (np.cos((data["minbatch_msignal"])))) <= -998, ((data["meanbatch_msignal"]) - (data["minbatch_msignal"])), data["mean_abs_chgbatch_slices2"] )))))/2.0)))) * 2.0)) +

                            0.100000*np.tanh(((np.cos((((np.where((((((np.tanh((data["maxtominbatch"]))) / 2.0)) + (((((np.tanh((data["minbatch"]))) / 2.0)) + (np.cos((np.tanh((data["mean_abs_chgbatch_slices2_msignal"]))))))))/2.0) <= -998, (14.91737747192382812), ((((np.tanh((data["mean_abs_chgbatch_slices2_msignal"]))) / 2.0)) + (data["abs_maxbatch_slices2_msignal"])) )) * 2.0)))) * 2.0)) +

                            0.100000*np.tanh((((3.0)) * (np.cos((((np.tanh(((-2.0)))) - (((data["minbatch_slices2_msignal"]) - (((data["maxbatch_slices2_msignal"]) * (np.where((1.0) <= -998, np.where(np.where(data["meanbatch_slices2"] <= -998, ((data["minbatch_slices2_msignal"]) - (data["meanbatch_slices2_msignal"])), (3.0) ) <= -998, np.tanh(((-2.0))), (3.0) ), (3.0) )))))))))))) +

                            0.100000*np.tanh(((data["minbatch"]) + (np.tanh(((1.0)))))) +

                            0.100000*np.tanh(np.sin((((((data["maxbatch_slices2_msignal"]) + (np.where(((((data["minbatch_msignal"]) + ((1.03063368797302246)))) * 2.0) <= -998, (((((1.03063368797302246)) * (((data["maxbatch_slices2_msignal"]) + ((1.03063368797302246)))))) * (((data["meanbatch_slices2"]) * (data["abs_minbatch_msignal"])))), np.cos(((-1.0))) )))) * 2.0)))) +

                            0.100000*np.tanh(np.where(((data["abs_avgbatch_msignal"]) * 2.0) > -998, (1.0), np.where(data["rangebatch_slices2_msignal"] > -998, data["maxbatch_slices2"], data["medianbatch_slices2_msignal"] ) )) +

                            0.100000*np.tanh((((-((np.cos((np.where(data["abs_maxbatch_msignal"] > -998, data["abs_maxbatch_msignal"], (((((((((-2.0)) / 2.0)) / 2.0)) * ((((4.10017585754394531)) + (np.where(data["abs_maxbatch_msignal"] > -998, data["maxtominbatch"], (-((data["mean_abs_chgbatch_slices2_msignal"]))) )))))) * 2.0) ))))))) * ((((data["abs_maxbatch_msignal"]) + ((-2.0)))/2.0)))) +

                            0.100000*np.tanh(((np.sin((((np.where(np.where(data["maxbatch_slices2_msignal"] > -998, np.sin((data["abs_maxbatch_msignal"])), data["maxbatch_slices2_msignal"] ) > -998, np.sin((np.cos(((-((((data["maxbatch_slices2_msignal"]) - (((data["stdbatch_slices2"]) * 2.0)))))))))), data["stdbatch_msignal"] )) * 2.0)))) * 2.0)) +

                            0.100000*np.tanh(np.sin((((data["minbatch_msignal"]) - (((np.where(np.where(((data["minbatch_msignal"]) + (data["abs_avgbatch_slices2_msignal"])) > -998, data["minbatch_msignal"], ((data["minbatch_msignal"]) - (((np.where(data["minbatch_slices2_msignal"] <= -998, np.sin((data["abs_maxbatch_msignal"])), data["medianbatch_msignal"] )) / 2.0))) ) <= -998, np.tanh((data["abs_maxbatch_slices2_msignal"])), data["medianbatch_msignal"] )) / 2.0)))))) +

                            0.100000*np.tanh(((np.tanh((np.sin((((data["meanbatch_slices2_msignal"]) - (((data["rangebatch_slices2"]) + (np.where(np.sin((data["stdbatch_slices2"])) > -998, (-1.0), ((data["stdbatch_msignal"]) - ((-1.0))) )))))))))) * 2.0)) +

                            0.100000*np.tanh(((np.where(np.tanh((((data["medianbatch_msignal"]) * (np.where(data["mean_abs_chgbatch_msignal"] <= -998, ((data["abs_avgbatch_slices2_msignal"]) - (data["mean_abs_chgbatch_msignal"])), data["abs_minbatch_slices2_msignal"] ))))) > -998, data["abs_minbatch_slices2_msignal"], ((data["abs_avgbatch_slices2_msignal"]) * (data["abs_maxbatch"])) )) + ((-((data["maxtominbatch"])))))) +

                            0.100000*np.tanh((((1.16699361801147461)) + ((((((((((((data["maxtominbatch_slices2"]) * (((data["maxtominbatch_msignal"]) - (data["stdbatch_msignal"]))))) + (data["rangebatch_slices2"]))) * 2.0)) + (data["maxtominbatch_msignal"]))/2.0)) / 2.0)))) +

                            0.100000*np.tanh(((np.where(data["medianbatch_slices2"] > -998, data["meanbatch_msignal"], (-((((((np.where(data["medianbatch_slices2"] > -998, data["meanbatch_msignal"], (-((((data["medianbatch_slices2"]) - (np.where(data["signal_shift_-1_msignal"] <= -998, (-((data["minbatch_msignal"]))), data["stdbatch_slices2"] )))))) )) + (data["minbatch_msignal"]))) - (np.where(data["signal_shift_-1_msignal"] <= -998, (-1.0), (0.0) )))))) )) + (data["minbatch_msignal"]))) +

                            0.100000*np.tanh(np.cos(((-(((6.25715494155883789))))))) +

                            0.100000*np.tanh(np.cos((data["signal_shift_-1"]))) +

                            0.100000*np.tanh((((7.0)) + ((((-((np.cos(((-((np.cos(((3.0)))))))))))) + (data["minbatch_msignal"]))))) +

                            0.100000*np.tanh(((((data["abs_avgbatch_slices2_msignal"]) + (np.where(np.sin((((((-3.0)) + (np.where(((data["abs_minbatch_slices2_msignal"]) / 2.0) > -998, data["abs_avgbatch_slices2_msignal"], (-3.0) )))/2.0))) <= -998, ((data["mean_abs_chgbatch_msignal"]) / 2.0), data["minbatch_msignal"] )))) * (data["medianbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(np.tanh((((data["meanbatch_msignal"]) * (((data["maxtominbatch_slices2"]) + (((np.where(data["maxtominbatch_slices2"] > -998, data["rangebatch_msignal"], np.tanh((np.where(data["rangebatch_slices2"] <= -998, data["minbatch_msignal"], ((data["meanbatch_slices2_msignal"]) * 2.0) ))) )) * 2.0)))))))) +

                            0.100000*np.tanh(((np.sin(((-((((data["abs_maxbatch_slices2_msignal"]) + (np.where(data["minbatch"] <= -998, data["signal_shift_+1_msignal"], np.tanh((np.where(((np.tanh((np.where(np.where((-((data["minbatch_msignal"]))) <= -998, data["meanbatch_msignal"], data["minbatch_msignal"] ) <= -998, data["stdbatch_msignal"], data["meanbatch_msignal"] )))) / 2.0) <= -998, data["meanbatch_slices2_msignal"], data["meanbatch_msignal"] ))) ))))))))) * 2.0)) +

                            0.100000*np.tanh(np.sin((((np.where(np.sin(((((-((((data["abs_avgbatch_msignal"]) * 2.0))))) - (data["signal_shift_-1_msignal"])))) > -998, data["abs_minbatch_msignal"], np.where(data["minbatch"] <= -998, data["abs_avgbatch_slices2_msignal"], data["maxtominbatch_msignal"] ) )) - ((((data["signal_shift_-1_msignal"]) + (data["rangebatch_slices2_msignal"]))/2.0)))))) +

                            0.100000*np.tanh(((data["medianbatch_slices2_msignal"]) + (np.where(data["meanbatch_slices2_msignal"] > -998, data["minbatch_msignal"], np.where(data["meanbatch_slices2"] > -998, data["abs_minbatch_slices2_msignal"], np.where(data["minbatch_msignal"] <= -998, data["mean_abs_chgbatch_slices2"], data["medianbatch_slices2_msignal"] ) ) )))) +

                            0.100000*np.tanh(((data["signal_shift_+1"]) + (((((data["maxbatch_slices2"]) + (data["signal_shift_+1"]))) + ((((((1.0)) + (data["signal_shift_-1"]))) * (np.tanh((((data["signal_shift_+1"]) + (((np.cos(((1.0)))) / 2.0)))))))))))) +

                            0.100000*np.tanh(np.where(data["minbatch"] > -998, ((((-1.0)) + ((((data["minbatch"]) + (data["rangebatch_slices2_msignal"]))/2.0)))/2.0), ((data["medianbatch_slices2"]) * 2.0) )) +

                            0.100000*np.tanh(np.cos((np.where(np.tanh((np.where(np.cos((data["mean_abs_chgbatch_msignal"])) <= -998, np.cos((data["maxbatch_slices2"])), ((data["maxbatch_slices2"]) / 2.0) ))) > -998, ((data["abs_maxbatch_slices2"]) * 2.0), np.cos((((data["abs_maxbatch_slices2"]) * 2.0))) )))) +

                            0.100000*np.tanh(((np.where(data["abs_minbatch_slices2"] <= -998, np.where(data["abs_maxbatch_slices2_msignal"] <= -998, data["medianbatch_slices2_msignal"], data["maxbatch_slices2"] ), data["medianbatch_slices2_msignal"] )) + (((((np.cos((((np.where(((data["rangebatch_msignal"]) * (data["maxbatch_slices2_msignal"])) <= -998, data["medianbatch_slices2_msignal"], data["maxbatch_slices2"] )) + (data["rangebatch_msignal"]))))) * 2.0)) * 2.0)))) +

                            0.100000*np.tanh(np.sin((np.sin((np.where(data["medianbatch_msignal"] > -998, np.cos((data["rangebatch_slices2"])), ((np.sin(((((data["rangebatch_msignal"]) + (np.sin((data["minbatch_slices2"]))))/2.0)))) / 2.0) )))))) +

                            0.100000*np.tanh(((np.cos((((np.where(data["signal_shift_+1_msignal"] <= -998, np.where((((2.0)) * (data["medianbatch_slices2_msignal"])) > -998, ((data["medianbatch_slices2_msignal"]) - (data["minbatch_msignal"])), data["signal_shift_-1"] ), (2.0) )) + (((data["medianbatch_slices2_msignal"]) - (data["minbatch_msignal"]))))))) * (data["signal_shift_+1"]))) +

                            0.100000*np.tanh(((((np.where(data["abs_maxbatch_msignal"] > -998, ((np.sin((((((((np.cos((data["abs_avgbatch_slices2_msignal"]))) + (data["maxbatch_msignal"]))/2.0)) + (data["abs_maxbatch_msignal"]))/2.0)))) * 2.0), np.tanh((np.tanh((np.cos((data["maxbatch_msignal"])))))) )) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.where(((((((data["signal_shift_+1_msignal"]) + (((data["signal_shift_-1_msignal"]) - (data["maxtominbatch_msignal"]))))) * (data["abs_maxbatch_slices2"]))) * 2.0) <= -998, ((data["abs_avgbatch_msignal"]) * (((data["signal_shift_-1_msignal"]) * (data["maxtominbatch_msignal"])))), ((np.tanh((np.cos((data["abs_maxbatch_slices2"]))))) * (data["abs_avgbatch_msignal"])) )) +

                            0.100000*np.tanh(((data["maxbatch_slices2"]) / 2.0)) +

                            0.100000*np.tanh(((((((-3.0)) + (np.where((((((-3.0)) + (((np.cos((data["rangebatch_msignal"]))) + (data["rangebatch_msignal"]))))) + (((data["maxbatch_slices2"]) + ((-3.0))))) > -998, ((data["abs_minbatch_msignal"]) + (data["maxbatch_slices2"])), data["maxbatch_slices2"] )))/2.0)) * (np.cos((((data["rangebatch_msignal"]) + (((data["maxbatch_slices2"]) + ((-3.0)))))))))) +

                            0.100000*np.tanh(np.sin((np.where(data["rangebatch_slices2_msignal"] > -998, data["rangebatch_slices2_msignal"], np.where(data["signal_shift_-1_msignal"] > -998, data["rangebatch_slices2_msignal"], data["rangebatch_slices2_msignal"] ) )))) +

                            0.100000*np.tanh(((((((data["abs_avgbatch_msignal"]) * 2.0)) * 2.0)) * ((((data["mean_abs_chgbatch_msignal"]) + (data["rangebatch_slices2_msignal"]))/2.0)))) +

                            0.100000*np.tanh(((((np.sin(((((((np.cos(((-((((data["maxbatch_slices2_msignal"]) * 2.0))))))) + (data["abs_avgbatch_slices2_msignal"]))/2.0)) / 2.0)))) / 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.sin((((np.cos((np.where((-1.0) > -998, ((np.where(((data["signal_shift_+1_msignal"]) / 2.0) <= -998, np.sin((np.cos((np.tanh((np.tanh((data["signal_shift_+1"])))))))), data["maxbatch_slices2_msignal"] )) * 2.0), ((data["abs_maxbatch"]) * ((((-((np.tanh((((data["rangebatch_slices2_msignal"]) * 2.0))))))) * 2.0))) )))) * 2.0)))) +

                            0.100000*np.tanh((((((3.0)) * 2.0)) + (((np.where(data["signal_shift_+1"] <= -998, ((data["signal_shift_+1_msignal"]) + (((np.sin(((((3.0)) + (((data["abs_maxbatch_slices2"]) / 2.0)))))) + (np.tanh((data["maxtominbatch"])))))), data["minbatch_msignal"] )) + (np.sin(((-((data["abs_maxbatch_slices2_msignal"])))))))))) +

                            0.100000*np.tanh(((((data["stdbatch_slices2_msignal"]) * (np.sin((np.where(((np.tanh(((0.0)))) * 2.0) > -998, data["signal_shift_-1"], np.where(((data["stdbatch_slices2_msignal"]) * (np.where(np.tanh((data["minbatch_slices2"])) > -998, data["signal_shift_+1"], data["signal_shift_-1"] ))) > -998, data["signal_shift_-1"], data["signal_shift_-1"] ) )))))) * 2.0)) +

                            0.100000*np.tanh(((np.cos((((np.cos((np.cos((((((((np.where(data["abs_minbatch_msignal"] <= -998, data["signal_shift_-1"], np.cos((data["signal_shift_+1"])) )) + (data["abs_avgbatch_msignal"]))) * (data["abs_minbatch_msignal"]))) + ((3.0)))))))) + (((data["signal_shift_+1"]) - (data["minbatch"]))))))) * ((((3.0)) + (data["abs_maxbatch"]))))) +

                            0.100000*np.tanh((-((((data["signal"]) * ((-((np.tanh((((np.tanh((np.sin(((((data["abs_avgbatch_slices2_msignal"]) + ((-((data["minbatch_msignal"])))))/2.0)))))) * (np.where(data["maxtominbatch_slices2_msignal"] > -998, data["maxbatch_slices2"], (((np.tanh(((-((data["maxbatch_msignal"])))))) + (np.sin((np.where(data["signal_shift_-1"] > -998, data["maxtominbatch_slices2"], data["maxtominbatch_slices2"] )))))/2.0) )))))))))))))) +

                            0.100000*np.tanh(((((((((data["signal_shift_-1"]) * (data["rangebatch_msignal"]))) * (np.cos((data["signal_shift_-1"]))))) - (np.cos((np.cos(((((data["rangebatch_msignal"]) + (np.where(data["abs_avgbatch_msignal"] <= -998, ((data["abs_avgbatch_slices2"]) * (data["signal_shift_-1"])), data["mean_abs_chgbatch_slices2_msignal"] )))/2.0)))))))) / 2.0)) +

                            0.100000*np.tanh(np.cos((np.where((((data["meanbatch_slices2_msignal"]) + (data["meanbatch_msignal"]))/2.0) <= -998, (-((np.where(data["meanbatch_msignal"] <= -998, data["meanbatch_slices2_msignal"], data["maxtominbatch_msignal"] )))), ((data["rangebatch_slices2"]) - ((((data["meanbatch_slices2_msignal"]) + (data["meanbatch_msignal"]))/2.0))) )))) +

                            0.100000*np.tanh((-((np.cos((data["abs_minbatch_msignal"])))))) +

                            0.100000*np.tanh(np.sin((np.where(((np.where(data["medianbatch_slices2_msignal"] > -998, np.sin((np.tanh((data["signal"])))), ((data["stdbatch_msignal"]) / 2.0) )) * 2.0) <= -998, data["medianbatch_slices2_msignal"], ((data["minbatch_msignal"]) + (data["stdbatch_msignal"])) )))) +

                            0.100000*np.tanh(np.cos(((0.0)))) +

                            0.100000*np.tanh((-((((data["maxtominbatch_slices2"]) - (np.where((((-((data["minbatch_slices2"])))) / 2.0) > -998, data["minbatch"], ((data["minbatch_slices2"]) / 2.0) ))))))) +

                            0.100000*np.tanh(np.sin((((data["rangebatch_slices2_msignal"]) + (np.sin((np.where(((np.where(((data["rangebatch_slices2_msignal"]) + (data["rangebatch_msignal"])) > -998, ((data["signal_shift_-1_msignal"]) / 2.0), np.where(data["signal"] > -998, np.sin((data["stdbatch_slices2"])), np.tanh(((3.0))) ) )) * 2.0) <= -998, data["rangebatch_msignal"], ((data["stdbatch_slices2"]) - (data["rangebatch_msignal"])) )))))))) +

                            0.100000*np.tanh(np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, np.cos(((((((((data["abs_avgbatch_msignal"]) * 2.0)) + (np.cos((((np.where(np.cos((data["mean_abs_chgbatch_slices2_msignal"])) > -998, np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2"], data["mean_abs_chgbatch_slices2_msignal"] ), np.sin((data["stdbatch_msignal"])) )) * 2.0)))))/2.0)) - (data["signal_shift_-1"])))), data["mean_abs_chgbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(((((np.sin((data["minbatch_slices2_msignal"]))) / 2.0)) * ((((12.05493068695068359)) * (np.where(data["minbatch_slices2_msignal"] <= -998, np.tanh((data["minbatch_slices2_msignal"])), (-(((((data["meanbatch_slices2_msignal"]) + (np.cos((data["abs_maxbatch_slices2"]))))/2.0)))) )))))) +

                            0.100000*np.tanh(np.where(np.where(data["mean_abs_chgbatch_msignal"] <= -998, data["meanbatch_slices2"], (11.41673564910888672) ) <= -998, data["meanbatch_msignal"], np.cos(((((-2.0)) - (((((np.where((1.0) > -998, data["abs_maxbatch_slices2_msignal"], ((data["maxbatch_slices2"]) * 2.0) )) * 2.0)) * 2.0))))) )) +

                            0.100000*np.tanh((((np.where(data["abs_maxbatch_slices2"] > -998, np.tanh((np.cos(((((0.0)) + (data["stdbatch_slices2"])))))), np.cos((data["medianbatch_slices2"])) )) + (np.where(((data["rangebatch_slices2"]) * 2.0) <= -998, data["abs_minbatch_msignal"], ((np.sin((data["meanbatch_slices2"]))) * (((data["minbatch_msignal"]) + (data["meanbatch_slices2_msignal"])))) )))/2.0)) +

                            0.100000*np.tanh(((np.sin((np.sin((data["signal"]))))) * (np.where(np.where(data["medianbatch_slices2_msignal"] > -998, data["signal"], np.tanh((np.sin((data["rangebatch_slices2_msignal"])))) ) <= -998, np.sin((data["rangebatch_slices2_msignal"])), ((data["rangebatch_slices2_msignal"]) - (data["maxtominbatch_slices2_msignal"])) )))) +

                            0.100000*np.tanh(((data["medianbatch_slices2"]) - (((((((((data["abs_avgbatch_slices2"]) + (np.tanh((((((-((data["signal"])))) + (data["mean_abs_chgbatch_slices2"]))/2.0)))))/2.0)) - (data["medianbatch_slices2"]))) + ((((((((data["medianbatch_slices2"]) + ((((data["maxbatch_msignal"]) + (data["maxbatch_msignal"]))/2.0)))) * (np.tanh((data["abs_maxbatch_slices2"]))))) + (data["medianbatch_msignal"]))/2.0)))/2.0)))) +

                            0.100000*np.tanh(((np.cos((data["rangebatch_slices2_msignal"]))) + (np.cos((data["signal_shift_+1"]))))) +

                            0.100000*np.tanh(((((np.where(data["abs_maxbatch_msignal"] <= -998, data["abs_avgbatch_slices2_msignal"], data["abs_avgbatch_slices2_msignal"] )) * (data["mean_abs_chgbatch_slices2"]))) * (np.where(np.tanh((np.cos((np.tanh((data["signal"])))))) <= -998, data["abs_maxbatch_slices2"], data["signal_shift_-1_msignal"] )))) +

                            0.100000*np.tanh(((data["meanbatch_slices2"]) * (np.sin((((data["abs_maxbatch_slices2_msignal"]) * (((np.where(np.sin(((1.0))) <= -998, (((1.0)) - (np.where((1.0) <= -998, (1.0), data["abs_avgbatch_msignal"] ))), data["stdbatch_slices2"] )) - ((-1.0)))))))))) +

                            0.100000*np.tanh(((np.cos((((data["abs_maxbatch"]) - (np.tanh((((data["abs_maxbatch_slices2_msignal"]) * ((((4.0)) + (((data["abs_maxbatch_slices2_msignal"]) * (((data["abs_maxbatch_msignal"]) + (np.where(data["stdbatch_slices2"] > -998, (((4.0)) / 2.0), ((data["medianbatch_slices2_msignal"]) + (np.cos((data["mean_abs_chgbatch_slices2"])))) )))))))))))))))) * (data["meanbatch_msignal"]))) +

                            0.100000*np.tanh(((((data["abs_avgbatch_slices2_msignal"]) * ((((data["rangebatch_slices2_msignal"]) + (((data["abs_avgbatch_slices2_msignal"]) + (((np.sin((data["abs_avgbatch_slices2_msignal"]))) / 2.0)))))/2.0)))) * 2.0)) +

                            0.100000*np.tanh(np.sin((((np.where(data["minbatch"] > -998, data["abs_avgbatch_slices2_msignal"], (((2.0)) + (((np.where(data["rangebatch_slices2"] > -998, data["rangebatch_slices2"], data["rangebatch_slices2"] )) / 2.0))) )) + (data["minbatch_msignal"]))))) +

                            0.100000*np.tanh(data["minbatch"]) +

                            0.100000*np.tanh(np.where((-(((((-(((((-3.0)) * 2.0))))) * ((((data["maxtominbatch"]) + (data["medianbatch_msignal"]))/2.0)))))) <= -998, data["abs_maxbatch_slices2"], data["rangebatch_slices2_msignal"] )) +

                            0.100000*np.tanh((((1.0)) * (np.sin((((np.cos(((7.38433790206909180)))) * (((data["meanbatch_slices2_msignal"]) * (((data["abs_maxbatch_slices2_msignal"]) - ((((data["maxbatch_msignal"]) + (((data["meanbatch_slices2_msignal"]) / 2.0)))/2.0)))))))))))) +

                            0.100000*np.tanh(np.cos((data["medianbatch_msignal"]))) +

                            0.100000*np.tanh(np.where((((((np.where(data["signal_shift_-1_msignal"] > -998, np.where(data["signal_shift_-1_msignal"] <= -998, data["signal_shift_+1"], np.sin((data["signal_shift_-1_msignal"])) ), (-3.0) )) / 2.0)) + (data["mean_abs_chgbatch_slices2"]))/2.0) <= -998, data["medianbatch_slices2"], data["signal_shift_-1_msignal"] )) +

                            0.100000*np.tanh(np.where(((np.sin(((1.0)))) / 2.0) > -998, data["stdbatch_slices2"], np.sin((data["abs_maxbatch_msignal"])) )) +

                            0.100000*np.tanh(np.sin((((np.where((-((data["rangebatch_msignal"]))) > -998, data["meanbatch_slices2_msignal"], ((data["rangebatch_slices2_msignal"]) * (data["mean_abs_chgbatch_slices2_msignal"])) )) * 2.0)))) +

                            0.100000*np.tanh(np.where(data["maxtominbatch_msignal"] <= -998, np.sin(((-((data["signal"]))))), (3.0) )) +

                            0.100000*np.tanh(np.sin((((np.where(np.sin(((0.0))) <= -998, (0.0), data["minbatch_msignal"] )) - (np.where(((((data["signal_shift_+1_msignal"]) + (data["signal_shift_+1_msignal"]))) - (data["abs_minbatch_slices2"])) <= -998, data["abs_minbatch_slices2"], (((((((1.0)) + (data["signal_shift_+1_msignal"]))) - (data["maxbatch_msignal"]))) / 2.0) )))))) +

                            0.100000*np.tanh(np.cos((((np.where(data["abs_maxbatch"] <= -998, (((((data["abs_maxbatch"]) + (((data["medianbatch_msignal"]) * 2.0)))/2.0)) / 2.0), data["meanbatch_slices2_msignal"] )) - (data["rangebatch_msignal"]))))) +

                            0.100000*np.tanh(np.where(data["rangebatch_slices2_msignal"] <= -998, np.sin((((data["abs_maxbatch_slices2"]) / 2.0))), np.where(np.sin((((((2.00612473487854004)) + ((2.00612473487854004)))/2.0))) <= -998, np.where((2.00612473487854004) <= -998, ((((2.00612473487854004)) + ((2.00612473487854004)))/2.0), np.sin((((data["rangebatch_slices2_msignal"]) + (((data["medianbatch_msignal"]) / 2.0))))) ), np.sin((data["rangebatch_slices2_msignal"])) ) )) +

                            0.100000*np.tanh(np.where(((np.cos((np.cos((np.cos((data["abs_avgbatch_slices2"]))))))) - (((np.sin(((1.0)))) / 2.0))) > -998, np.cos((data["abs_minbatch_slices2"])), ((data["signal_shift_+1"]) / 2.0) )) +

                            0.100000*np.tanh(np.cos((np.sin((np.cos((data["abs_minbatch_slices2"]))))))) +

                            0.100000*np.tanh(np.cos((np.where((((((np.cos((((((data["abs_maxbatch_msignal"]) / 2.0)) * ((-2.0)))))) / 2.0)) + (np.cos((data["minbatch_msignal"]))))/2.0) > -998, data["maxtominbatch_msignal"], ((np.sin((data["minbatch"]))) / 2.0) )))) +

                            0.100000*np.tanh(np.tanh((np.where(data["maxbatch_slices2"] > -998, data["meanbatch_slices2"], np.where(np.tanh((data["maxbatch_slices2"])) <= -998, (((((data["medianbatch_slices2_msignal"]) + (((data["maxbatch_slices2"]) / 2.0)))/2.0)) * 2.0), ((np.sin(((-2.0)))) + (((data["meanbatch_slices2"]) / 2.0))) ) )))) +

                            0.100000*np.tanh(np.sin((((np.tanh((((data["signal_shift_-1_msignal"]) - (np.sin(((((np.sin((data["signal_shift_-1_msignal"]))) + (((np.where(data["signal_shift_-1_msignal"] <= -998, (((0.0)) - (np.cos((data["maxbatch_slices2_msignal"])))), data["medianbatch_slices2_msignal"] )) * ((0.0)))))/2.0)))))))) - (np.cos((data["maxbatch_msignal"]))))))) +

                            0.100000*np.tanh(np.sin((np.where(np.cos((data["maxbatch_slices2"])) > -998, data["signal"], (-((np.where(data["maxbatch_slices2"] > -998, np.sin((((data["medianbatch_slices2_msignal"]) * 2.0))), data["maxbatch_slices2_msignal"] )))) )))) +

                            0.100000*np.tanh((((((((((2.0)) + (((((data["meanbatch_slices2"]) + (data["maxtominbatch"]))) * (((data["meanbatch_slices2"]) - ((-((data["maxtominbatch"])))))))))/2.0)) + (((((data["meanbatch_slices2"]) + (((data["meanbatch_slices2"]) + (data["maxtominbatch"]))))) * (((data["maxtominbatch"]) - (data["maxtominbatch"]))))))/2.0)) + (data["stdbatch_msignal"]))) +

                            0.100000*np.tanh(np.where(np.sin((np.sin((data["minbatch_slices2_msignal"])))) > -998, ((np.tanh((data["abs_maxbatch_slices2"]))) / 2.0), (((2.0)) / 2.0) )) +

                            0.100000*np.tanh(((np.tanh((np.tanh((data["mean_abs_chgbatch_slices2"]))))) / 2.0)) +

                            0.100000*np.tanh(np.sin((((np.cos((((data["abs_avgbatch_slices2_msignal"]) - (np.cos((data["abs_maxbatch_slices2"]))))))) - (((data["abs_maxbatch_slices2"]) - (((np.where(data["medianbatch_slices2_msignal"] > -998, data["minbatch_slices2"], np.sin((np.cos((np.tanh((((data["abs_maxbatch_slices2"]) - (np.cos(((-((data["medianbatch_slices2_msignal"]))))))))))))) )) * 2.0)))))))) +

                            0.100000*np.tanh(np.where(data["meanbatch_msignal"] <= -998, np.cos((data["abs_maxbatch"])), ((((data["meanbatch_msignal"]) * (((np.cos((np.where((-2.0) <= -998, (-1.0), np.where((0.0) > -998, (1.0), data["abs_maxbatch"] ) )))) - ((-((np.cos((data["abs_maxbatch"])))))))))) * 2.0) )) +

                            0.100000*np.tanh(np.sin((((data["minbatch_slices2_msignal"]) - (np.where(data["rangebatch_slices2_msignal"] > -998, data["signal_shift_-1_msignal"], np.sin((((data["minbatch_slices2_msignal"]) - (np.where(data["maxtominbatch"] > -998, data["signal_shift_-1_msignal"], np.where(np.where(data["abs_avgbatch_msignal"] <= -998, (((data["maxtominbatch_msignal"]) + (data["abs_avgbatch_msignal"]))/2.0), (2.0) ) > -998, np.sin((data["signal_shift_+1"])), np.tanh((data["maxtominbatch"])) ) ))))) )))))) +

                            0.100000*np.tanh(np.where(data["rangebatch_msignal"] <= -998, data["abs_maxbatch_slices2_msignal"], np.cos(((((data["rangebatch_msignal"]) + (data["abs_maxbatch"]))/2.0))) )) +

                            0.100000*np.tanh(np.cos((((((data["abs_maxbatch_slices2_msignal"]) - (np.where(np.where((-((data["abs_maxbatch_slices2_msignal"]))) > -998, data["minbatch"], (0.0) ) <= -998, data["stdbatch_slices2_msignal"], np.where(np.where(np.cos((data["signal"])) <= -998, data["signal"], np.where(data["signal_shift_+1_msignal"] <= -998, (-2.0), data["signal"] ) ) <= -998, data["stdbatch_slices2_msignal"], np.tanh((data["signal_shift_+1_msignal"])) ) )))) * 2.0)))) +

                            0.100000*np.tanh(np.where(np.tanh(((((np.cos(((3.0)))) + (((data["abs_maxbatch_slices2"]) * 2.0)))/2.0))) <= -998, data["maxtominbatch_slices2"], np.where(np.cos(((-((data["maxtominbatch_slices2"]))))) > -998, (1.0), ((np.cos((np.cos((data["signal_shift_+1"]))))) + (data["abs_maxbatch"])) ) )) +

                            0.100000*np.tanh(((((np.cos((((np.where(data["abs_maxbatch_slices2_msignal"] > -998, (-(((-((data["maxbatch_msignal"])))))), np.where(data["maxbatch_msignal"] > -998, data["signal_shift_+1"], ((data["maxtominbatch_slices2"]) + (data["signal_shift_+1_msignal"])) ) )) / 2.0)))) * 2.0)) * 2.0)) +

                            0.099609*np.tanh(((np.tanh((np.cos((data["abs_minbatch_slices2"]))))) * 2.0)) +

                            0.100000*np.tanh(np.where(data["stdbatch_msignal"] > -998, data["signal_shift_+1_msignal"], data["meanbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(np.where(np.tanh((np.tanh(((2.0))))) <= -998, np.where(data["maxbatch_slices2_msignal"] > -998, data["medianbatch_slices2_msignal"], data["rangebatch_slices2"] ), ((np.sin((((data["maxtominbatch_msignal"]) * 2.0)))) * 2.0) )) +

                            0.100000*np.tanh(((((((7.0)) + ((-3.0)))) + ((2.0)))/2.0)) +

                            0.100000*np.tanh(((np.tanh((np.where(((data["rangebatch_msignal"]) + (data["maxbatch_slices2"])) <= -998, np.where(data["maxbatch_slices2"] > -998, data["medianbatch_slices2_msignal"], data["medianbatch_slices2"] ), np.cos((((data["rangebatch_msignal"]) + (data["maxbatch_slices2"])))) )))) * 2.0)) +

                            0.100000*np.tanh((-((np.where(data["medianbatch_slices2"] <= -998, data["stdbatch_slices2_msignal"], ((np.where(np.where(((((np.sin((np.cos((data["abs_maxbatch_slices2_msignal"]))))) * 2.0)) * 2.0) > -998, np.cos((data["abs_maxbatch_slices2_msignal"])), np.sin((data["meanbatch_slices2_msignal"])) ) > -998, data["meanbatch_slices2_msignal"], (-((data["meanbatch_slices2_msignal"]))) )) * (((np.sin((np.cos((data["abs_maxbatch_slices2_msignal"]))))) * 2.0))) ))))) +

                            0.100000*np.tanh(np.where((((-1.0)) * (((((np.cos((np.cos(((((((data["abs_avgbatch_slices2_msignal"]) + ((1.0)))) + (np.cos((((data["abs_maxbatch"]) * 2.0)))))/2.0)))))) * 2.0)) / 2.0))) > -998, np.cos(((((data["maxtominbatch"]) + (data["maxbatch_slices2"]))/2.0))), data["maxtominbatch_msignal"] )) +

                            0.100000*np.tanh((-((np.cos((np.where(np.cos((np.where(((data["minbatch_slices2_msignal"]) - (np.cos((np.cos((np.cos((data["medianbatch_msignal"])))))))) > -998, data["maxbatch_slices2"], (((data["abs_avgbatch_msignal"]) + (np.tanh((((data["maxtominbatch_slices2_msignal"]) - (data["mean_abs_chgbatch_slices2_msignal"]))))))/2.0) ))) <= -998, (2.0), data["abs_maxbatch_slices2"] ))))))) +

                            0.100000*np.tanh(np.cos((np.where(((((data["rangebatch_slices2_msignal"]) + ((((2.0)) * (np.cos((data["stdbatch_slices2"]))))))) + (np.where(data["medianbatch_slices2_msignal"] <= -998, (1.0), np.where(np.cos((np.sin((data["minbatch"])))) > -998, data["mean_abs_chgbatch_msignal"], data["signal_shift_+1"] ) ))) <= -998, (0.0), data["minbatch"] )))) +

                            0.100000*np.tanh((((np.where(np.sin((data["abs_avgbatch_slices2_msignal"])) > -998, data["abs_avgbatch_slices2_msignal"], np.sin((data["rangebatch_slices2_msignal"])) )) + ((-((np.tanh((data["meanbatch_slices2_msignal"])))))))/2.0)) +

                            0.100000*np.tanh(np.tanh((((((np.cos((np.sin((np.tanh((((np.tanh((np.sin((((np.sin((((data["mean_abs_chgbatch_msignal"]) + (np.tanh((data["rangebatch_slices2_msignal"]))))))) * 2.0)))))) / 2.0)))))))) * 2.0)) - (np.tanh(((-((data["rangebatch_msignal"])))))))))) +

                            0.089932*np.tanh(np.sin(((((data["meanbatch_slices2"]) + ((((-((data["medianbatch_slices2_msignal"])))) / 2.0)))/2.0)))) +

                            0.100000*np.tanh(np.sin((np.tanh((np.where(data["mean_abs_chgbatch_msignal"] > -998, np.where(((np.tanh((np.where(data["abs_avgbatch_msignal"] <= -998, (2.0), data["meanbatch_slices2_msignal"] )))) * (data["abs_avgbatch_slices2"])) <= -998, data["abs_maxbatch_msignal"], data["abs_avgbatch_msignal"] ), ((np.cos((data["meanbatch_slices2_msignal"]))) / 2.0) )))))) +

                            0.098436*np.tanh(((np.sin((((((((np.sin((np.where(data["rangebatch_msignal"] <= -998, data["meanbatch_slices2_msignal"], np.cos((data["maxtominbatch_slices2"])) )))) / 2.0)) + (np.where(data["maxtominbatch"] <= -998, data["rangebatch_msignal"], data["medianbatch_slices2_msignal"] )))) * 2.0)))) * 2.0)) +

                            0.100000*np.tanh(np.where((((-3.0)) / 2.0) <= -998, (2.0), ((((((np.sin((((data["abs_maxbatch_msignal"]) * 2.0)))) * 2.0)) * 2.0)) * (((((data["abs_maxbatch_msignal"]) * 2.0)) * ((-3.0))))) )) +

                            0.100000*np.tanh(((((-((np.sin((data["abs_maxbatch_slices2_msignal"])))))) + ((2.0)))/2.0)) +

                            0.100000*np.tanh(np.where(data["abs_avgbatch_slices2"] > -998, np.cos((((data["abs_maxbatch_slices2"]) * 2.0))), np.where(((data["maxtominbatch_msignal"]) / 2.0) > -998, (1.0), data["abs_maxbatch_msignal"] ) )) +

                            0.100000*np.tanh(np.sin((((data["minbatch_msignal"]) - (np.where(np.sin((((data["minbatch_msignal"]) - (np.where(np.sin((((data["minbatch_msignal"]) - (data["meanbatch_msignal"])))) > -998, data["meanbatch_msignal"], np.where(data["meanbatch_msignal"] <= -998, (((3.0)) / 2.0), data["meanbatch_msignal"] ) ))))) > -998, data["meanbatch_msignal"], np.tanh((np.sin((np.cos((data["minbatch_msignal"])))))) )))))) +

                            0.100000*np.tanh((((((data["signal_shift_-1"]) + (((data["meanbatch_slices2"]) / 2.0)))) + (((((2.0)) + (np.where(data["medianbatch_slices2"] > -998, data["medianbatch_slices2"], (((data["abs_minbatch_slices2"]) + (((np.cos((((np.cos((data["medianbatch_slices2"]))) - (data["meanbatch_slices2_msignal"]))))) + ((9.0)))))/2.0) )))/2.0)))/2.0)) +

                            0.100000*np.tanh(((data["minbatch_msignal"]) * (np.sin(((((data["minbatch_msignal"]) + (np.sin((np.tanh(((-1.0)))))))/2.0)))))) +

                            0.100000*np.tanh((((((np.where(np.sin(((((-3.0)) * 2.0))) > -998, np.cos((((np.cos((data["minbatch"]))) / 2.0))), data["medianbatch_slices2_msignal"] )) + (np.tanh(((((0.0)) - ((((0.0)) - ((((data["minbatch_slices2"]) + (np.sin((data["minbatch"]))))/2.0)))))))))/2.0)) / 2.0)) +

                            0.093744*np.tanh(np.where(np.where(data["rangebatch_msignal"] > -998, (2.0), np.where(data["abs_maxbatch_slices2_msignal"] > -998, np.cos((data["signal_shift_+1"])), np.cos((np.sin((np.where((((data["abs_maxbatch_slices2_msignal"]) + ((-1.0)))/2.0) > -998, data["minbatch_msignal"], data["signal_shift_-1_msignal"] ))))) ) ) <= -998, np.sin((data["maxbatch_msignal"])), np.cos((np.sin((data["signal_shift_-1_msignal"])))) )) +

                            0.100000*np.tanh(np.cos((np.where(data["abs_maxbatch_msignal"] <= -998, np.cos((((data["abs_maxbatch_msignal"]) - (np.cos(((0.0))))))), ((((((data["maxtominbatch_msignal"]) * 2.0)) * 2.0)) + ((-((data["abs_maxbatch_msignal"]))))) )))) +

                            0.099707*np.tanh(np.cos((((data["abs_maxbatch_msignal"]) * (((data["medianbatch_slices2"]) + (((((np.tanh((np.tanh((((np.where(np.tanh((data["medianbatch_slices2"])) > -998, np.where((((data["meanbatch_slices2_msignal"]) + ((8.0)))/2.0) <= -998, (1.0), np.tanh((np.tanh((data["medianbatch_slices2"])))) ), data["rangebatch_msignal"] )) * 2.0)))))) / 2.0)) * 2.0)))))))) +

                            0.100000*np.tanh((((data["abs_maxbatch_slices2"]) + (((((3.0)) + (((np.where(data["maxbatch_slices2"] > -998, np.where(data["abs_avgbatch_slices2"] > -998, (-((np.cos((data["abs_maxbatch_slices2"]))))), np.where(data["abs_avgbatch_slices2"] <= -998, (((-((data["abs_minbatch_slices2"])))) * 2.0), data["maxbatch_slices2"] ) ), (-1.0) )) / 2.0)))/2.0)))/2.0)) +

                            0.100000*np.tanh(np.cos((np.where(data["signal_shift_-1"] > -998, ((data["minbatch_msignal"]) + (data["signal_shift_+1_msignal"])), np.tanh(((((data["minbatch_msignal"]) + (((data["minbatch_msignal"]) + (data["signal_shift_+1_msignal"]))))/2.0))) )))) +

                            0.100000*np.tanh(np.where(np.sin(((0.0))) > -998, ((data["abs_minbatch_msignal"]) + (data["abs_avgbatch_slices2"])), (((data["medianbatch_slices2_msignal"]) + (np.sin((data["meanbatch_msignal"]))))/2.0) )))    

      

    def GP_class_4(self,data):

        return self.Output( -2.516509 +

                            0.100000*np.tanh(np.where(np.where(np.sin(((0.0))) <= -998, data["signal_shift_-1_msignal"], np.sin((data["abs_maxbatch"])) ) > -998, data["meanbatch_slices2"], np.where(np.tanh(((-((data["abs_maxbatch"]))))) <= -998, np.sin((data["mean_abs_chgbatch_slices2"])), ((np.where(data["medianbatch_slices2"] > -998, data["mean_abs_chgbatch_msignal"], data["meanbatch_slices2"] )) * 2.0) ) )) +

                            0.100000*np.tanh(np.where(((np.cos((data["maxbatch_slices2"]))) + (((data["stdbatch_slices2_msignal"]) - (((data["signal_shift_-1"]) - (data["maxtominbatch"])))))) <= -998, ((data["medianbatch_slices2"]) * 2.0), ((data["medianbatch_slices2"]) - (np.where(data["signal_shift_-1"] <= -998, (((-2.0)) * (data["signal_shift_+1"])), data["medianbatch_slices2_msignal"] ))) )) +

                            0.100000*np.tanh(np.where((-1.0) <= -998, ((((np.where((0.0) <= -998, ((data["signal_shift_-1"]) - (((data["stdbatch_msignal"]) * 2.0))), np.where(((data["signal_shift_-1"]) / 2.0) <= -998, np.tanh((data["signal_shift_-1"])), data["signal_shift_+1"] ) )) * 2.0)) * (data["stdbatch_msignal"])), data["signal_shift_+1"] )) +

                            0.100000*np.tanh(np.where(np.sin((data["mean_abs_chgbatch_slices2"])) <= -998, data["abs_avgbatch_msignal"], ((((data["meanbatch_slices2"]) + (((data["abs_maxbatch_slices2_msignal"]) * (((((data["stdbatch_slices2"]) - (data["abs_minbatch_msignal"]))) / 2.0)))))) - (np.where(data["meanbatch_slices2"] > -998, data["mean_abs_chgbatch_slices2"], data["rangebatch_slices2_msignal"] ))) )) +

                            0.100000*np.tanh(np.where(np.cos(((((((data["abs_maxbatch_slices2_msignal"]) - (np.where((-((data["abs_maxbatch_msignal"]))) > -998, np.tanh(((-((data["minbatch_slices2_msignal"]))))), ((data["abs_maxbatch_msignal"]) - (((data["abs_maxbatch_slices2"]) + (data["abs_maxbatch_slices2_msignal"])))) )))) + (np.sin((data["minbatch_slices2_msignal"]))))/2.0))) <= -998, np.sin((data["minbatch_slices2_msignal"])), ((np.sin((data["minbatch_slices2_msignal"]))) * 2.0) )) +

                            0.100000*np.tanh(np.where((-((data["rangebatch_slices2"]))) > -998, np.cos((data["rangebatch_slices2"])), np.tanh((np.cos((data["rangebatch_slices2"])))) )) +

                            0.100000*np.tanh(((data["abs_maxbatch"]) * (np.sin((np.where(np.where(((np.cos((np.tanh(((1.21274137496948242)))))) * (data["mean_abs_chgbatch_slices2_msignal"])) > -998, data["minbatch_slices2_msignal"], data["signal_shift_+1"] ) > -998, np.where(data["abs_maxbatch"] > -998, data["minbatch_msignal"], np.sin((np.tanh((data["abs_maxbatch"])))) ), data["mean_abs_chgbatch_msignal"] )))))) +

                            0.100000*np.tanh(((((data["rangebatch_msignal"]) * (np.tanh((np.sin((((np.where(((data["rangebatch_msignal"]) - (((np.tanh((data["maxbatch_slices2_msignal"]))) - (data["abs_maxbatch_msignal"])))) > -998, (-((data["maxbatch_slices2_msignal"]))), data["maxbatch_slices2_msignal"] )) * 2.0)))))))) * 2.0)) +

                            0.100000*np.tanh(((data["medianbatch_slices2"]) * (np.sin((np.where(((data["rangebatch_slices2_msignal"]) * (data["mean_abs_chgbatch_slices2"])) > -998, data["abs_maxbatch_msignal"], np.where(((data["meanbatch_slices2"]) / 2.0) > -998, data["medianbatch_slices2"], (-((np.sin((data["abs_maxbatch_msignal"]))))) ) )))))) +

                            0.100000*np.tanh(((np.where(((data["abs_maxbatch_slices2_msignal"]) * 2.0) > -998, np.tanh((np.sin((data["minbatch_slices2_msignal"])))), data["signal_shift_+1_msignal"] )) * 2.0)) +

                            0.100000*np.tanh(((np.cos((data["minbatch_msignal"]))) * 2.0)) +

                            0.100000*np.tanh(((((data["medianbatch_slices2"]) * (np.sin((data["maxbatch_slices2_msignal"]))))) + (np.where(data["maxbatch_slices2_msignal"] > -998, np.sin((np.where(data["medianbatch_slices2_msignal"] > -998, np.sin((((np.sin((data["signal"]))) * 2.0))), data["meanbatch_slices2"] ))), ((np.sin((data["signal_shift_-1"]))) * 2.0) )))) +

                            0.100000*np.tanh(np.cos((((data["minbatch_msignal"]) + (np.where(((data["minbatch_msignal"]) * (data["minbatch_msignal"])) > -998, data["meanbatch_msignal"], (-((data["meanbatch_msignal"]))) )))))) +

                            0.100000*np.tanh(((((data["stdbatch_msignal"]) * (np.where(((data["abs_maxbatch_msignal"]) * (((data["abs_maxbatch_msignal"]) * (data["abs_avgbatch_slices2"])))) > -998, (((data["minbatch"]) + (data["minbatch_msignal"]))/2.0), (-((data["maxtominbatch"]))) )))) - ((((2.0)) + (((data["meanbatch_slices2"]) * ((((6.65203332901000977)) * (data["meanbatch_slices2_msignal"]))))))))) +

                            0.100000*np.tanh(((((np.sin((np.where(np.sin((np.cos((np.where(((np.where(((data["rangebatch_slices2"]) * 2.0) <= -998, data["maxtominbatch_slices2"], data["minbatch_msignal"] )) * 2.0) > -998, data["rangebatch_slices2"], data["minbatch_slices2"] ))))) <= -998, data["rangebatch_slices2"], data["minbatch_msignal"] )))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh((((-((data["maxbatch_msignal"])))) * (((np.sin((data["maxtominbatch"]))) + (((np.sin((data["abs_maxbatch"]))) + (((data["abs_avgbatch_msignal"]) * 2.0)))))))) +

                            0.100000*np.tanh(((((np.sin((np.where(data["abs_avgbatch_slices2"] > -998, data["rangebatch_slices2"], ((np.where(np.where(((data["meanbatch_slices2_msignal"]) * 2.0) > -998, data["rangebatch_slices2"], ((data["abs_avgbatch_msignal"]) - (np.sin(((13.59499073028564453))))) ) > -998, data["abs_avgbatch_slices2"], ((data["rangebatch_slices2"]) * 2.0) )) * 2.0) )))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.sin((data["rangebatch_slices2"]))) +

                            0.100000*np.tanh(((((np.cos((((data["minbatch_msignal"]) - (np.where(np.sin((np.where(data["signal"] > -998, data["minbatch_msignal"], np.where(np.cos((np.cos((data["minbatch_msignal"])))) <= -998, (0.0), data["minbatch_msignal"] ) ))) <= -998, np.where(data["abs_avgbatch_slices2"] > -998, (10.50418758392333984), data["maxbatch_slices2_msignal"] ), np.cos((data["minbatch_slices2_msignal"])) )))))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh((-((np.where((1.0) <= -998, (-1.0), ((np.where(np.sin((data["maxtominbatch_slices2"])) <= -998, (-1.0), (1.0) )) + (((np.sin((((np.sin(((-1.0)))) * (data["minbatch_msignal"]))))) * 2.0))) ))))) +

                            0.100000*np.tanh(((((np.where(data["signal_shift_-1"] > -998, data["rangebatch_slices2"], (((-2.0)) * (np.cos((data["maxbatch_msignal"])))) )) * 2.0)) * (((data["minbatch_slices2_msignal"]) * (np.cos((data["maxbatch_msignal"]))))))) +

                            0.100000*np.tanh(((((((data["abs_maxbatch_slices2_msignal"]) + (np.sin((np.where(data["rangebatch_slices2"] > -998, np.where(data["medianbatch_slices2_msignal"] > -998, data["rangebatch_slices2"], data["stdbatch_slices2_msignal"] ), ((((data["abs_maxbatch_slices2_msignal"]) * 2.0)) - (data["rangebatch_slices2"])) )))))) * (np.sin((data["abs_maxbatch_slices2_msignal"]))))) * 2.0)) +

                            0.100000*np.tanh(np.where((10.14795780181884766) <= -998, np.sin((data["signal_shift_-1"])), ((np.where(((data["minbatch_msignal"]) * 2.0) <= -998, (((2.0)) * (((((data["minbatch_msignal"]) * 2.0)) * 2.0))), np.sin((np.sin((((np.sin((((data["minbatch_msignal"]) * 2.0)))) * 2.0))))) )) * 2.0) )) +

                            0.100000*np.tanh(((((data["signal_shift_+1"]) * 2.0)) * (np.sin((data["rangebatch_slices2"]))))) +

                            0.100000*np.tanh(((np.where(np.cos((data["abs_maxbatch_msignal"])) <= -998, (((((3.0)) * (data["minbatch_msignal"]))) * (data["medianbatch_slices2"])), np.where(data["minbatch_msignal"] <= -998, (0.0), np.where(data["abs_maxbatch_msignal"] <= -998, (-3.0), data["meanbatch_msignal"] ) ) )) + (((data["abs_maxbatch_msignal"]) * (np.cos((data["minbatch_msignal"]))))))) +

                            0.100000*np.tanh(((np.sin((np.where(np.sin(((-((np.where(np.sin((np.where(data["minbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], np.cos((data["meanbatch_slices2_msignal"])) ))) > -998, data["minbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] )))))) > -998, data["minbatch_slices2_msignal"], data["minbatch_slices2_msignal"] )))) * 2.0)) +

                            0.100000*np.tanh(((data["abs_minbatch_slices2_msignal"]) * (((np.sin((np.sin((((((data["minbatch"]) + (np.where(((np.sin((((((data["minbatch"]) + (np.where(data["minbatch"] <= -998, data["rangebatch_msignal"], data["medianbatch_slices2_msignal"] )))) * 2.0)))) + (data["minbatch"])) > -998, data["rangebatch_msignal"], ((data["abs_minbatch_slices2_msignal"]) * (data["minbatch"])) )))) * 2.0)))))) * 2.0)))) +

                            0.100000*np.tanh(((data["abs_maxbatch_msignal"]) * (np.sin((np.where(np.sin(((3.0))) > -998, data["rangebatch_slices2"], ((np.where(np.sin((data["abs_maxbatch_msignal"])) > -998, data["abs_maxbatch_msignal"], data["minbatch_slices2_msignal"] )) * 2.0) )))))) +

                            0.100000*np.tanh(((((((np.where(np.sin((np.where(data["maxbatch_msignal"] <= -998, data["signal_shift_-1"], data["rangebatch_slices2"] ))) <= -998, np.sin((data["maxbatch_slices2"])), data["rangebatch_slices2"] )) - (data["abs_avgbatch_msignal"]))) * (np.where(data["abs_minbatch_slices2"] <= -998, data["maxbatch_slices2_msignal"], np.sin((data["abs_maxbatch_msignal"])) )))) / 2.0)) +

                            0.100000*np.tanh((((10.0)) * (np.sin((((data["maxbatch_msignal"]) + (np.where(((((((data["medianbatch_slices2"]) + (data["maxtominbatch_msignal"]))) * (data["maxbatch_msignal"]))) * 2.0) <= -998, (((data["abs_avgbatch_slices2"]) + ((10.46251010894775391)))/2.0), (-2.0) )))))))) +

                            0.100000*np.tanh(((np.cos((((data["rangebatch_slices2"]) - (np.where(np.cos((((data["rangebatch_slices2"]) - (np.where(np.cos((data["rangebatch_slices2"])) <= -998, data["rangebatch_slices2"], np.sin((((data["abs_maxbatch_msignal"]) / 2.0))) ))))) <= -998, data["rangebatch_slices2"], np.cos((data["rangebatch_slices2"])) )))))) * 2.0)) +

                            0.100000*np.tanh(((np.sin((np.where(np.tanh((data["abs_maxbatch_slices2_msignal"])) <= -998, np.sin(((((np.where(data["meanbatch_slices2_msignal"] <= -998, (-((data["mean_abs_chgbatch_msignal"]))), data["abs_maxbatch_slices2_msignal"] )) + (((data["signal_shift_+1"]) - (np.sin((((((data["minbatch_slices2_msignal"]) / 2.0)) * 2.0)))))))/2.0))), data["abs_maxbatch_slices2_msignal"] )))) * 2.0)) +

                            0.100000*np.tanh((((-2.0)) * (np.where(data["stdbatch_slices2_msignal"] <= -998, (-((data["rangebatch_msignal"]))), np.cos((np.where((((((-2.0)) * (((((-((data["abs_minbatch_slices2"])))) + (data["mean_abs_chgbatch_msignal"]))/2.0)))) / 2.0) > -998, data["maxbatch_slices2_msignal"], data["maxbatch_slices2_msignal"] ))) )))) +

                            0.100000*np.tanh(((((data["rangebatch_slices2_msignal"]) - ((-1.0)))) * (np.where(((data["minbatch_msignal"]) * (((np.sin((data["abs_maxbatch_msignal"]))) * 2.0))) <= -998, data["rangebatch_slices2"], np.where(np.sin((np.cos(((-((data["minbatch_msignal"]))))))) > -998, np.cos((data["minbatch_msignal"])), np.cos((((data["maxtominbatch"]) * (np.tanh((data["maxtominbatch_slices2"])))))) ) )))) +

                            0.100000*np.tanh(np.cos((((data["minbatch_msignal"]) + (np.where((((((data["signal_shift_+1"]) + ((((data["stdbatch_slices2_msignal"]) + (data["medianbatch_msignal"]))/2.0)))/2.0)) / 2.0) > -998, np.where(data["minbatch_msignal"] > -998, data["meanbatch_msignal"], data["minbatch_msignal"] ), np.tanh((data["abs_maxbatch_msignal"])) )))))) +

                            0.100000*np.tanh(((np.sin((np.sin((np.sin((data["abs_maxbatch_msignal"]))))))) + (((((((data["medianbatch_msignal"]) + (np.tanh((np.sin((np.sin((data["minbatch_msignal"]))))))))) + (np.cos((data["minbatch_msignal"]))))) + (np.sin((data["medianbatch_msignal"]))))))) +

                            0.100000*np.tanh(np.where(data["minbatch"] <= -998, data["meanbatch_slices2"], ((np.where(data["rangebatch_slices2"] <= -998, ((((data["abs_minbatch_msignal"]) * (data["medianbatch_slices2"]))) / 2.0), data["meanbatch_msignal"] )) + (((np.where(((np.sin((data["maxbatch_slices2_msignal"]))) + (data["meanbatch_slices2"])) > -998, np.sin((data["maxbatch_slices2_msignal"])), data["meanbatch_slices2_msignal"] )) * (((np.sin((data["rangebatch_slices2"]))) + (data["abs_maxbatch_msignal"])))))) )) +

                            0.100000*np.tanh(((data["stdbatch_slices2_msignal"]) - (((data["abs_maxbatch_slices2"]) * (np.where(np.where((2.0) <= -998, np.sin((data["abs_maxbatch_msignal"])), np.sin((((data["stdbatch_slices2_msignal"]) - (np.sin((np.sin((data["abs_maxbatch_msignal"])))))))) ) <= -998, (-((np.sin((data["medianbatch_slices2"]))))), np.sin((((data["meanbatch_msignal"]) - (np.sin((np.sin((data["abs_maxbatch_msignal"])))))))) )))))) +

                            0.100000*np.tanh(((np.sin((np.where(data["minbatch_slices2"] > -998, data["minbatch_msignal"], np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, data["minbatch_slices2"], (3.0) ) )))) - (np.where(data["minbatch_slices2"] > -998, np.cos(((1.0))), ((np.tanh((((data["stdbatch_slices2"]) / 2.0)))) * ((-(((((np.sin((np.sin(((-1.0)))))) + (np.cos((data["minbatch_slices2"]))))/2.0)))))) )))) +

                            0.100000*np.tanh(((np.cos((data["maxbatch_slices2_msignal"]))) * (np.where(((data["meanbatch_slices2_msignal"]) * 2.0) <= -998, ((data["medianbatch_slices2"]) + (np.cos((((data["mean_abs_chgbatch_msignal"]) + (np.sin(((0.0))))))))), (((data["minbatch"]) + (data["minbatch_slices2_msignal"]))/2.0) )))) +

                            0.100000*np.tanh(np.cos((((data["rangebatch_slices2"]) - (np.cos((np.cos(((((-2.0)) - (np.cos((((data["maxbatch_slices2_msignal"]) * (((np.sin(((-(((((3.0)) + (np.cos((((data["rangebatch_slices2"]) - (np.cos((np.tanh((np.cos((data["minbatch_msignal"])))))))))))))))))) * (data["abs_minbatch_slices2_msignal"]))))))))))))))))) +

                            0.100000*np.tanh((-(((((7.61211109161376953)) - (np.where(data["medianbatch_slices2"] <= -998, ((data["maxbatch_slices2_msignal"]) + (np.tanh((((data["maxbatch_slices2_msignal"]) / 2.0))))), ((((-3.0)) + (data["medianbatch_msignal"]))/2.0) ))))))) +

                            0.100000*np.tanh(np.sin((((((data["medianbatch_msignal"]) + (np.tanh((((data["minbatch_msignal"]) / 2.0)))))) + ((((np.cos(((-((((((np.where(data["minbatch_msignal"] > -998, np.sin((data["signal_shift_+1"])), data["minbatch_slices2"] )) * (data["signal_shift_+1"]))) * 2.0))))))) + (data["minbatch_msignal"]))/2.0)))))) +

                            0.100000*np.tanh(np.where(data["rangebatch_slices2_msignal"] <= -998, data["mean_abs_chgbatch_msignal"], np.sin((((data["rangebatch_slices2"]) + (np.sin((np.tanh((np.cos((np.where((-((data["maxbatch_slices2_msignal"]))) <= -998, (1.0), (-1.0) ))))))))))) )) +

                            0.100000*np.tanh(data["meanbatch_slices2"]) +

                            0.100000*np.tanh((((((((data["abs_avgbatch_msignal"]) - (np.cos((((data["stdbatch_slices2_msignal"]) * 2.0)))))) * (np.sin((data["abs_maxbatch_slices2_msignal"]))))) + (np.tanh((np.cos((((data["abs_minbatch_slices2"]) + (np.where(np.tanh((data["mean_abs_chgbatch_msignal"])) > -998, ((data["abs_maxbatch_slices2_msignal"]) * ((-3.0))), data["minbatch_slices2"] )))))))))/2.0)) +

                            0.100000*np.tanh(np.where(data["abs_minbatch_slices2_msignal"] > -998, ((data["meanbatch_slices2_msignal"]) + (data["minbatch_slices2_msignal"])), np.tanh((data["abs_maxbatch_slices2_msignal"])) )) +

                            0.100000*np.tanh(np.where(data["rangebatch_msignal"] > -998, np.where(data["medianbatch_slices2_msignal"] <= -998, ((np.cos((np.sin((data["maxbatch_msignal"]))))) - (np.sin(((((((3.0)) - (((data["abs_avgbatch_slices2"]) * (np.sin((data["abs_maxbatch_msignal"]))))))) / 2.0))))), np.sin((data["maxbatch_msignal"])) ), np.cos((np.sin((data["maxbatch_msignal"])))) )) +

                            0.100000*np.tanh((((((-((((data["medianbatch_slices2_msignal"]) * 2.0))))) * (np.sin((data["medianbatch_slices2_msignal"]))))) + (np.sin((data["medianbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh(((((((data["medianbatch_msignal"]) + (np.sin((((np.sin((data["meanbatch_msignal"]))) * ((-((((data["rangebatch_slices2_msignal"]) + (data["signal_shift_-1"])))))))))))/2.0)) + (((data["maxbatch_slices2_msignal"]) * (((np.sin(((((((data["meanbatch_msignal"]) * 2.0)) + (np.sin((data["minbatch_msignal"]))))/2.0)))) * ((-(((((data["abs_maxbatch"]) + (data["rangebatch_slices2_msignal"]))/2.0))))))))))/2.0)) +

                            0.100000*np.tanh(((data["maxbatch_slices2_msignal"]) * (np.sin((((((data["maxbatch_msignal"]) * 2.0)) * (np.tanh(((-2.0)))))))))) +

                            0.100000*np.tanh((-((((np.where(np.sin((((data["signal_shift_+1_msignal"]) + ((-((np.sin((np.where(data["rangebatch_slices2"] <= -998, data["signal_shift_+1_msignal"], (((data["minbatch_slices2"]) + (((data["maxbatch_slices2_msignal"]) * (((data["rangebatch_slices2"]) * (data["minbatch_msignal"]))))))/2.0) )))))))))) > -998, data["stdbatch_slices2_msignal"], data["rangebatch_slices2"] )) * (np.sin((np.sin((data["rangebatch_slices2"])))))))))) +

                            0.100000*np.tanh(np.sin((np.where(np.sin((data["rangebatch_slices2"])) > -998, data["abs_maxbatch_slices2_msignal"], np.sin((np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], np.where((((10.0)) * 2.0) > -998, data["abs_maxbatch_slices2_msignal"], ((data["abs_avgbatch_slices2"]) - (((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, (-3.0), data["abs_minbatch_msignal"] )) + (data["abs_maxbatch_slices2_msignal"])))) ) ))) )))) +

                            0.100000*np.tanh((((((((((data["abs_minbatch_slices2"]) + (data["medianbatch_slices2_msignal"]))/2.0)) - (((np.sin((((np.sin(((-((data["abs_maxbatch_msignal"])))))) / 2.0)))) - ((((data["maxbatch_slices2_msignal"]) + (data["medianbatch_slices2_msignal"]))/2.0)))))) - (np.sin(((0.0)))))) - (((np.sin((data["medianbatch_slices2_msignal"]))) - (data["medianbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh((((-2.0)) - (np.where(np.tanh((((data["abs_avgbatch_slices2_msignal"]) + (np.sin(((((data["abs_avgbatch_slices2_msignal"]) + (data["minbatch_slices2_msignal"]))/2.0))))))) <= -998, ((np.sin((np.sin(((3.0)))))) - ((((-2.0)) * 2.0))), (-((data["meanbatch_msignal"]))) )))) +

                            0.100000*np.tanh(((data["maxbatch_msignal"]) * (np.sin((((np.where(data["medianbatch_slices2"] <= -998, data["meanbatch_slices2"], (-2.0) )) - (np.where(data["maxtominbatch_slices2_msignal"] > -998, data["abs_maxbatch_msignal"], np.where(np.where(data["stdbatch_slices2"] > -998, (2.0), data["maxtominbatch_slices2_msignal"] ) <= -998, (((3.0)) * (data["stdbatch_slices2"])), ((data["signal_shift_-1_msignal"]) + (data["stdbatch_slices2_msignal"])) ) )))))))) +

                            0.100000*np.tanh(((np.sin((((data["abs_maxbatch_msignal"]) * 2.0)))) * ((((data["minbatch_msignal"]) + ((((data["minbatch_msignal"]) + (((np.sin((((data["abs_maxbatch_msignal"]) * 2.0)))) * (((((((data["minbatch_msignal"]) + (np.sin((data["minbatch_msignal"]))))/2.0)) + (np.tanh(((-((np.tanh((((((data["abs_avgbatch_slices2"]) - ((1.0)))) / 2.0))))))))))/2.0)))))/2.0)))/2.0)))) +

                            0.100000*np.tanh(np.where(((np.sin(((((-((data["meanbatch_slices2_msignal"])))) * 2.0)))) * 2.0) <= -998, (-((np.where(data["abs_avgbatch_msignal"] <= -998, data["maxbatch_slices2_msignal"], ((data["medianbatch_slices2_msignal"]) + (np.sin(((((-((data["meanbatch_slices2_msignal"])))) * 2.0))))) )))), ((data["medianbatch_slices2_msignal"]) + (np.sin(((((-((data["meanbatch_slices2_msignal"])))) * 2.0))))) )) +

                            0.100000*np.tanh((((((((data["abs_avgbatch_msignal"]) * (data["medianbatch_slices2_msignal"]))) - (np.where(data["maxtominbatch_slices2"] > -998, data["rangebatch_slices2"], data["meanbatch_slices2"] )))) + (data["medianbatch_slices2_msignal"]))/2.0)) +

                            0.100000*np.tanh(((((((data["meanbatch_msignal"]) - (np.where(data["abs_maxbatch_msignal"] > -998, (((((3.82403707504272461)) + ((2.0)))) * (((np.where(data["stdbatch_slices2"] > -998, np.cos((((data["abs_maxbatch_msignal"]) + (data["minbatch_msignal"])))), ((data["abs_maxbatch_msignal"]) + (data["minbatch_msignal"])) )) / 2.0))), ((data["abs_maxbatch_msignal"]) + (data["minbatch_msignal"])) )))) / 2.0)) * 2.0)) +

                            0.100000*np.tanh(((data["abs_maxbatch"]) * ((((np.where(np.where(data["medianbatch_msignal"] <= -998, data["abs_maxbatch"], data["medianbatch_slices2_msignal"] ) > -998, data["mean_abs_chgbatch_slices2_msignal"], (-1.0) )) + ((((data["rangebatch_slices2_msignal"]) + (np.where(((((1.0)) + (data["maxtominbatch_msignal"]))/2.0) > -998, (((-((data["mean_abs_chgbatch_slices2"])))) + (data["medianbatch_slices2_msignal"])), data["signal_shift_+1"] )))/2.0)))/2.0)))) +

                            0.100000*np.tanh(np.where((((((data["maxtominbatch_slices2_msignal"]) / 2.0)) + (data["abs_minbatch_slices2_msignal"]))/2.0) <= -998, (((np.sin((data["abs_maxbatch_msignal"]))) + (data["meanbatch_slices2_msignal"]))/2.0), (((-3.0)) + (((np.sin((data["abs_maxbatch_slices2_msignal"]))) * (data["rangebatch_slices2"])))) )) +

                            0.100000*np.tanh((((data["meanbatch_slices2_msignal"]) + (np.where((((data["stdbatch_msignal"]) + (np.sin((data["maxtominbatch_slices2_msignal"]))))/2.0) <= -998, (((data["stdbatch_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))/2.0), np.cos((data["minbatch_msignal"])) )))/2.0)) +

                            0.100000*np.tanh(((data["maxbatch_slices2"]) * ((((data["medianbatch_slices2_msignal"]) + (np.sin((np.where((-3.0) <= -998, ((data["minbatch_msignal"]) + (((((data["maxbatch_slices2"]) * (((data["medianbatch_slices2_msignal"]) + (np.cos((np.sin((data["abs_minbatch_slices2"]))))))))) / 2.0))), ((data["minbatch_msignal"]) + (np.cos((data["meanbatch_msignal"])))) )))))/2.0)))) +

                            0.100000*np.tanh(np.sin((np.where(data["signal_shift_+1_msignal"] > -998, data["maxbatch_slices2_msignal"], np.tanh((data["signal"])) )))) +

                            0.100000*np.tanh((((data["signal_shift_-1_msignal"]) + ((((data["signal_shift_-1_msignal"]) + ((((((((np.where(data["abs_minbatch_slices2"] > -998, (3.0), np.cos((np.where(np.cos(((((data["abs_avgbatch_slices2"]) + (data["abs_maxbatch_msignal"]))/2.0))) > -998, data["meanbatch_slices2_msignal"], (((data["maxbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2_msignal"]))/2.0) ))) )) / 2.0)) - (np.tanh((data["signal_shift_-1"]))))) + (data["signal_shift_-1_msignal"]))/2.0)))/2.0)))/2.0)) +

                            0.100000*np.tanh(((((data["abs_maxbatch_slices2"]) * (np.where(data["medianbatch_slices2_msignal"] > -998, (-((np.sin((data["meanbatch_msignal"]))))), data["stdbatch_slices2"] )))) * 2.0)) +

                            0.100000*np.tanh((((data["medianbatch_msignal"]) + ((((((data["medianbatch_msignal"]) * 2.0)) + ((3.0)))/2.0)))/2.0)) +

                            0.100000*np.tanh(((data["medianbatch_msignal"]) + (np.cos((np.where(data["medianbatch_msignal"] > -998, data["minbatch_msignal"], np.sin((data["abs_minbatch_slices2"])) )))))) +

                            0.100000*np.tanh(((((np.sin((np.where(data["minbatch_slices2_msignal"] <= -998, np.where(data["abs_maxbatch_slices2"] <= -998, np.cos((data["minbatch_msignal"])), ((np.sin((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, (2.0), data["minbatch_slices2_msignal"] )))) - (data["rangebatch_slices2"])) ), data["abs_maxbatch_slices2"] )))) - (data["rangebatch_slices2"]))) * (np.cos((data["abs_maxbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh((((-(((((data["meanbatch_slices2_msignal"]) + (((data["meanbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))))/2.0))))) * (np.sin(((((((data["rangebatch_slices2"]) + ((((0.0)) * (data["minbatch_slices2_msignal"]))))/2.0)) * 2.0)))))) +

                            0.100000*np.tanh(((data["medianbatch_msignal"]) - (np.where((((8.0)) / 2.0) > -998, (3.0), (((-((((data["medianbatch_msignal"]) - (data["maxbatch_slices2"])))))) * 2.0) )))) +

                            0.100000*np.tanh((-((((((((np.sin((np.sin((np.cos((data["maxbatch_msignal"]))))))) * (data["abs_maxbatch"]))) + (data["stdbatch_msignal"]))) + (((((np.sin((data["meanbatch_slices2_msignal"]))) * (((data["meanbatch_slices2_msignal"]) * (data["abs_maxbatch"]))))) + (data["stdbatch_msignal"])))))))) +

                            0.100000*np.tanh(((data["meanbatch_slices2_msignal"]) + ((-(((-((((np.sin((data["meanbatch_slices2_msignal"]))) * (((np.where(data["minbatch_msignal"] <= -998, ((np.tanh((data["meanbatch_slices2_msignal"]))) - (data["maxtominbatch_slices2_msignal"])), (-3.0) )) * (((((data["stdbatch_slices2"]) + (np.where((-(((-3.0)))) > -998, data["meanbatch_slices2_msignal"], data["stdbatch_slices2"] )))) * 2.0)))))))))))))) +

                            0.100000*np.tanh(np.sin((np.where(data["stdbatch_slices2"] > -998, np.cos((np.where(np.cos((np.where((1.0) <= -998, np.sin((data["signal_shift_+1"])), ((np.sin((data["signal_shift_-1_msignal"]))) - (data["mean_abs_chgbatch_slices2_msignal"])) ))) <= -998, np.sin((data["signal_shift_+1"])), data["minbatch_msignal"] ))), np.cos((data["medianbatch_slices2_msignal"])) )))) +

                            0.100000*np.tanh(((data["rangebatch_slices2"]) * (((((np.sin((data["rangebatch_slices2"]))) - (((((data["rangebatch_slices2"]) + (((np.where(((((data["minbatch"]) * (((data["abs_maxbatch"]) * 2.0)))) / 2.0) > -998, np.sin((data["minbatch_msignal"])), data["abs_maxbatch"] )) / 2.0)))) * (np.cos((data["maxbatch_slices2_msignal"]))))))) * 2.0)))) +

                            0.100000*np.tanh(np.sin((np.sin((((((data["signal"]) * 2.0)) / 2.0)))))) +

                            0.100000*np.tanh(((np.where(((((np.where(data["maxtominbatch_slices2"] <= -998, data["stdbatch_slices2_msignal"], np.sin((data["maxtominbatch_slices2_msignal"])) )) - (np.where(data["signal_shift_-1_msignal"] <= -998, data["abs_minbatch_slices2"], np.tanh((data["abs_minbatch_slices2"])) )))) * 2.0) > -998, ((data["medianbatch_msignal"]) + (data["signal_shift_-1"])), (-2.0) )) / 2.0)) +

                            0.100000*np.tanh(((data["minbatch_msignal"]) * (np.where(((((np.cos((data["minbatch_msignal"]))) * (data["minbatch_msignal"]))) * (np.where(data["maxtominbatch_slices2_msignal"] > -998, data["meanbatch_msignal"], ((np.cos((data["minbatch_msignal"]))) * (data["meanbatch_msignal"])) ))) > -998, ((np.cos((data["minbatch_msignal"]))) * (data["meanbatch_msignal"])), data["minbatch_msignal"] )))) +

                            0.100000*np.tanh(np.where(((((data["meanbatch_msignal"]) * ((-((data["abs_maxbatch_slices2_msignal"])))))) - (np.cos((np.cos((data["minbatch"])))))) > -998, ((((data["abs_avgbatch_msignal"]) * ((-((np.cos((data["minbatch_msignal"])))))))) - (np.cos((data["minbatch"])))), data["maxtominbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(np.where(np.sin((data["abs_minbatch_slices2_msignal"])) <= -998, np.where(((data["maxbatch_slices2_msignal"]) * 2.0) <= -998, data["minbatch_msignal"], ((data["medianbatch_slices2"]) / 2.0) ), ((np.sin((np.cos((((((data["minbatch_msignal"]) + (((data["abs_minbatch_slices2_msignal"]) - (data["abs_maxbatch_slices2_msignal"]))))) / 2.0)))))) * 2.0) )) +

                            0.100000*np.tanh((((((((data["abs_avgbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_msignal"]))/2.0)) / 2.0)) + (np.sin((data["signal_shift_+1_msignal"]))))) +

                            0.100000*np.tanh(((data["medianbatch_msignal"]) + (np.where((((5.0)) + (data["signal_shift_-1_msignal"])) <= -998, data["signal_shift_-1_msignal"], ((np.sin((np.sin((data["maxbatch_msignal"]))))) * 2.0) )))) +

                            0.100000*np.tanh(((np.sin(((-1.0)))) - (np.where((((1.0)) * ((((-3.0)) * (((data["maxtominbatch_msignal"]) + (((data["maxtominbatch"]) * 2.0))))))) <= -998, np.sin((data["minbatch_msignal"])), np.cos(((((((-1.0)) - ((-((data["rangebatch_slices2"])))))) - (((data["rangebatch_slices2"]) / 2.0))))) )))) +

                            0.100000*np.tanh(((np.where(((((data["abs_maxbatch_slices2_msignal"]) * (data["maxtominbatch_slices2"]))) * 2.0) <= -998, np.sin((data["maxbatch_slices2_msignal"])), ((data["abs_maxbatch_slices2_msignal"]) * (data["maxtominbatch_slices2"])) )) * (((np.cos((np.where(data["stdbatch_msignal"] <= -998, data["signal_shift_-1_msignal"], (((data["meanbatch_msignal"]) + (np.sin((np.sin((data["maxbatch_slices2_msignal"]))))))/2.0) )))) * (np.cos((data["abs_minbatch_msignal"]))))))) +

                            0.100000*np.tanh(((np.cos((data["minbatch_msignal"]))) * 2.0)) +

                            0.100000*np.tanh(np.sin((np.where(np.where(data["maxtominbatch_slices2"] > -998, data["maxbatch_slices2_msignal"], data["abs_maxbatch_msignal"] ) > -998, data["maxbatch_slices2_msignal"], np.sin((np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["maxbatch_slices2_msignal"], data["abs_maxbatch_msignal"] ))) )))) +

                            0.100000*np.tanh(np.where(np.where(((data["minbatch_slices2"]) - (data["medianbatch_msignal"])) > -998, data["rangebatch_slices2"], data["mean_abs_chgbatch_msignal"] ) > -998, data["medianbatch_slices2_msignal"], np.where(data["meanbatch_msignal"] <= -998, ((data["medianbatch_msignal"]) * (((data["rangebatch_slices2"]) * 2.0))), np.tanh(((((((3.0)) + (data["medianbatch_msignal"]))) * 2.0))) ) )) +

                            0.100000*np.tanh(np.where(np.where(np.sin((np.sin((data["meanbatch_slices2"])))) > -998, (((1.0)) * 2.0), data["meanbatch_msignal"] ) > -998, np.sin((data["abs_maxbatch_slices2_msignal"])), np.sin(((-2.0))) )) +

                            0.100000*np.tanh(((np.sin((np.sin(((((-((((data["minbatch_msignal"]) + ((((-2.0)) - (np.where(np.sin((np.sin(((((-((data["rangebatch_slices2"])))) - (data["mean_abs_chgbatch_msignal"])))))) > -998, data["rangebatch_slices2"], data["rangebatch_slices2"] ))))))))) / 2.0)))))) * 2.0)) +

                            0.100000*np.tanh(np.where((-((((data["maxtominbatch_slices2"]) * 2.0)))) <= -998, np.where(data["meanbatch_slices2"] > -998, data["maxtominbatch_slices2"], ((data["abs_avgbatch_slices2_msignal"]) * 2.0) ), np.where(data["minbatch_slices2"] > -998, data["minbatch_slices2"], (4.0) ) )) +

                            0.100000*np.tanh(np.cos((np.where(data["medianbatch_slices2"] > -998, data["abs_maxbatch_slices2"], ((data["meanbatch_slices2"]) * (np.where(np.where(((np.where(np.cos((data["abs_maxbatch_slices2"])) > -998, np.sin((data["abs_maxbatch_slices2"])), ((data["rangebatch_slices2_msignal"]) * 2.0) )) / 2.0) > -998, data["abs_maxbatch_slices2"], np.cos((data["maxbatch_slices2"])) ) > -998, data["rangebatch_slices2_msignal"], data["signal_shift_+1"] ))) )))) +

                            0.100000*np.tanh((((np.sin((((data["minbatch_msignal"]) + (data["minbatch_msignal"]))))) + (((((((((data["minbatch_msignal"]) * (np.cos((np.where(data["minbatch_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], np.where((0.0) > -998, (0.0), (0.0) ) )))))) / 2.0)) / 2.0)) * 2.0)))/2.0)) +

                            0.100000*np.tanh(((data["signal"]) * (np.sin((np.where(np.where(((data["rangebatch_slices2"]) + (data["signal"])) > -998, np.cos(((-1.0))), np.sin((data["signal"])) ) > -998, data["rangebatch_slices2"], data["stdbatch_msignal"] )))))) +

                            0.100000*np.tanh(((data["medianbatch_msignal"]) + ((((((data["medianbatch_slices2_msignal"]) / 2.0)) + (data["medianbatch_msignal"]))/2.0)))) +

                            0.100000*np.tanh(np.where((-(((-((((data["abs_maxbatch_msignal"]) + (data["maxbatch_slices2_msignal"])))))))) <= -998, np.sin((data["stdbatch_msignal"])), np.sin((np.sin((np.where(((np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, data["stdbatch_msignal"], data["meanbatch_msignal"] )) * (data["rangebatch_slices2"])) <= -998, (-2.0), np.where(data["medianbatch_slices2"] <= -998, data["rangebatch_slices2"], data["abs_maxbatch_msignal"] ) ))))) )) +

                            0.100000*np.tanh(np.where((9.0) <= -998, data["abs_maxbatch_slices2_msignal"], np.where((9.0) <= -998, (-((data["maxtominbatch_slices2_msignal"]))), ((((np.sin((data["abs_maxbatch_slices2_msignal"]))) * (data["abs_maxbatch_slices2_msignal"]))) - (np.sin(((((data["rangebatch_slices2"]) + ((((data["maxbatch_slices2"]) + (((data["abs_avgbatch_msignal"]) * (data["minbatch_msignal"]))))/2.0)))/2.0))))) ) )) +

                            0.100000*np.tanh(((np.where(data["medianbatch_msignal"] <= -998, data["meanbatch_slices2"], np.cos((((np.where(np.sin(((-((data["abs_maxbatch_slices2_msignal"]))))) <= -998, (4.45979928970336914), data["maxbatch_msignal"] )) * 2.0))) )) * 2.0)) +

                            0.100000*np.tanh((-(((((((data["abs_minbatch_slices2"]) + (np.where((((data["abs_maxbatch"]) + (np.sin((np.cos(((1.0)))))))/2.0) <= -998, np.sin((data["meanbatch_slices2_msignal"])), data["abs_avgbatch_slices2"] )))/2.0)) * (np.sin((np.where((1.0) > -998, data["medianbatch_msignal"], np.cos(((((data["meanbatch_slices2"]) + (np.sin((np.cos(((1.0)))))))/2.0))) ))))))))) +

                            0.100000*np.tanh(((((np.cos(((((np.tanh((np.tanh((data["abs_maxbatch"]))))) + (data["maxtominbatch_msignal"]))/2.0)))) * 2.0)) * (np.where(((((((3.93995141983032227)) + (data["rangebatch_slices2"]))/2.0)) - (data["stdbatch_slices2"])) > -998, np.where(data["medianbatch_slices2_msignal"] > -998, data["rangebatch_msignal"], data["abs_maxbatch_msignal"] ), (6.0) )))) +

                            0.100000*np.tanh(np.sin((((data["maxbatch_msignal"]) - (np.where(((np.cos(((((((0.0)) * 2.0)) + (((data["maxbatch_msignal"]) + ((1.0)))))))) * 2.0) > -998, np.where(data["maxtominbatch"] > -998, data["minbatch_msignal"], ((data["maxbatch_msignal"]) * 2.0) ), ((np.sin((np.cos((data["maxbatch_msignal"]))))) / 2.0) )))))) +

                            0.100000*np.tanh(((data["medianbatch_msignal"]) * (np.where(data["signal_shift_+1"] <= -998, np.where(data["abs_maxbatch_msignal"] > -998, np.where(data["signal_shift_-1_msignal"] <= -998, data["minbatch_msignal"], (-((np.tanh((((((data["abs_maxbatch_msignal"]) / 2.0)) * (data["minbatch_msignal"]))))))) ), (0.0) ), np.sin((data["minbatch_msignal"])) )))) +

                            0.100000*np.tanh((((-((((np.sin((data["meanbatch_msignal"]))) * 2.0))))) * 2.0)) +

                            0.100000*np.tanh(np.where(((((((np.sin((((((((data["minbatch_slices2_msignal"]) * 2.0)) * 2.0)) / 2.0)))) * 2.0)) * 2.0)) / 2.0) > -998, np.sin((data["maxtominbatch_slices2_msignal"])), ((data["signal_shift_-1_msignal"]) * ((-((np.tanh((np.sin(((-((np.tanh((((((data["medianbatch_msignal"]) / 2.0)) * 2.0))))))))))))))) )) +

                            0.100000*np.tanh(np.where(np.sin((data["abs_maxbatch_msignal"])) <= -998, data["abs_minbatch_slices2"], np.sin((((data["abs_maxbatch_msignal"]) - (np.where(data["maxbatch_msignal"] > -998, data["signal_shift_-1_msignal"], np.where((-1.0) <= -998, data["abs_maxbatch_msignal"], data["signal_shift_+1"] ) ))))) )) +

                            0.100000*np.tanh(np.sin((((data["abs_maxbatch"]) + (np.cos((np.cos((np.cos((((np.where(data["abs_maxbatch_msignal"] > -998, (2.0), data["meanbatch_slices2_msignal"] )) - (((data["signal_shift_+1"]) * 2.0)))))))))))))) +

                            0.100000*np.tanh(data["medianbatch_slices2_msignal"]) +

                            0.100000*np.tanh(np.sin((((data["signal_shift_-1_msignal"]) - (((np.where(((data["abs_avgbatch_msignal"]) * (np.tanh((data["minbatch"])))) > -998, data["meanbatch_msignal"], (10.0) )) * 2.0)))))) +

                            0.100000*np.tanh(np.tanh((np.sin((np.tanh((data["maxtominbatch"]))))))) +

                            0.100000*np.tanh(((np.where(((((data["minbatch_msignal"]) * 2.0)) / 2.0) > -998, ((((np.where(data["mean_abs_chgbatch_msignal"] > -998, np.cos((data["minbatch_msignal"])), data["signal_shift_-1"] )) * 2.0)) * 2.0), np.where(((np.cos((data["minbatch_msignal"]))) + (data["minbatch_msignal"])) <= -998, (0.0), ((data["signal_shift_-1"]) * 2.0) ) )) * (np.cos((data["meanbatch_msignal"]))))) +

                            0.100000*np.tanh(np.cos((((data["minbatch"]) + (((data["signal_shift_-1"]) - (((data["mean_abs_chgbatch_msignal"]) * (np.cos((np.cos((((np.cos((((data["minbatch"]) + (((np.tanh((data["maxtominbatch"]))) - (((data["mean_abs_chgbatch_msignal"]) * (np.cos((np.cos((np.cos((np.sin((data["mean_abs_chgbatch_msignal"]))))))))))))))))) + (data["minbatch"]))))))))))))))) +

                            0.100000*np.tanh(np.where(data["maxtominbatch_slices2_msignal"] > -998, np.sin((data["maxtominbatch_slices2"])), np.sin((data["stdbatch_msignal"])) )) +

                            0.100000*np.tanh(np.sin((np.where(np.sin((data["maxtominbatch_msignal"])) <= -998, np.where(np.where((-(((0.0)))) <= -998, data["abs_avgbatch_slices2"], data["abs_maxbatch_slices2"] ) > -998, (-1.0), np.sin((np.sin((data["abs_maxbatch_slices2_msignal"])))) ), data["abs_maxbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh(np.cos((((((data["medianbatch_slices2_msignal"]) - ((((((np.cos((np.tanh(((((data["abs_minbatch_slices2"]) + ((((np.cos((data["meanbatch_msignal"]))) + (data["abs_avgbatch_msignal"]))/2.0)))/2.0)))))) + (data["abs_minbatch_slices2"]))/2.0)) * 2.0)))) / 2.0)))) +

                            0.100000*np.tanh(np.cos((np.where(((((data["mean_abs_chgbatch_slices2"]) * (np.sin((np.where(np.cos((((data["meanbatch_msignal"]) * 2.0))) > -998, data["signal"], (0.0) )))))) + (((np.cos((np.cos((data["maxtominbatch_msignal"]))))) * 2.0))) > -998, data["rangebatch_msignal"], np.tanh((np.sin((((np.cos(((1.0)))) + (np.sin((data["stdbatch_msignal"])))))))) )))) +

                            0.100000*np.tanh(np.where(data["minbatch_msignal"] > -998, np.sin((data["abs_maxbatch_msignal"])), np.where(data["signal_shift_+1"] > -998, ((np.sin((np.sin((np.sin(((3.0)))))))) * 2.0), (((data["meanbatch_slices2_msignal"]) + (np.sin((data["abs_maxbatch_slices2_msignal"]))))/2.0) ) )) +

                            0.100000*np.tanh((((data["rangebatch_slices2"]) + (np.where(data["stdbatch_slices2_msignal"] > -998, data["signal_shift_+1_msignal"], (((data["abs_minbatch_slices2"]) + (((np.where(data["signal_shift_+1_msignal"] > -998, np.cos((data["minbatch_msignal"])), np.where(data["abs_minbatch_slices2"] <= -998, data["maxbatch_msignal"], (((data["stdbatch_slices2_msignal"]) + (data["stdbatch_slices2_msignal"]))/2.0) ) )) * (np.sin((np.cos((np.sin((data["maxtominbatch"]))))))))))/2.0) )))/2.0)) +

                            0.100000*np.tanh(((data["signal_shift_-1_msignal"]) * ((((((1.0)) + (data["meanbatch_slices2"]))) * ((((1.0)) + (((np.where(data["rangebatch_slices2"] > -998, data["meanbatch_msignal"], (((((((((((1.0)) + (((np.where(data["rangebatch_slices2"] > -998, data["rangebatch_slices2"], (-2.0) )) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) * 2.0) )) * 2.0)))))))) +

                            0.100000*np.tanh(np.sin((((data["mean_abs_chgbatch_msignal"]) * (((data["meanbatch_slices2"]) + (np.where((-((data["rangebatch_msignal"]))) <= -998, ((data["meanbatch_slices2"]) + (np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, ((data["mean_abs_chgbatch_msignal"]) * (((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0))), data["maxtominbatch_msignal"] ))), data["abs_maxbatch_slices2_msignal"] )))))))) +

                            0.100000*np.tanh(np.where((-((((data["minbatch_slices2_msignal"]) / 2.0)))) > -998, np.sin((np.sin((data["maxbatch_slices2_msignal"])))), np.where(np.sin(((3.0))) > -998, np.sin((data["abs_minbatch_slices2_msignal"])), np.cos((np.where((4.69076633453369141) <= -998, data["maxtominbatch_slices2_msignal"], np.where(((data["minbatch_msignal"]) / 2.0) <= -998, data["minbatch_slices2_msignal"], data["minbatch_msignal"] ) ))) ) )) +

                            0.100000*np.tanh(((data["signal_shift_+1_msignal"]) - (((data["medianbatch_msignal"]) * (np.where(np.where(data["maxtominbatch_msignal"] > -998, ((data["medianbatch_msignal"]) - (data["signal_shift_-1"])), ((np.where((0.0) <= -998, data["maxtominbatch"], np.sin((data["abs_avgbatch_msignal"])) )) * (((data["medianbatch_msignal"]) / 2.0))) ) > -998, data["medianbatch_msignal"], np.sin((data["medianbatch_msignal"])) )))))) +

                            0.100000*np.tanh(np.where(np.cos((data["minbatch_msignal"])) > -998, ((((data["signal_shift_+1_msignal"]) - ((-((np.sin((data["maxbatch_slices2_msignal"])))))))) * (data["maxbatch_slices2_msignal"])), np.sin((((((((6.22805833816528320)) + (np.where(data["meanbatch_slices2"] <= -998, np.sin((data["maxbatch_slices2_msignal"])), ((data["abs_minbatch_msignal"]) * (((data["maxtominbatch_slices2_msignal"]) * (np.sin((data["maxbatch_slices2_msignal"])))))) )))/2.0)) * 2.0))) )) +

                            0.100000*np.tanh(np.where(data["maxtominbatch_msignal"] <= -998, data["maxbatch_msignal"], ((data["maxbatch_msignal"]) * (np.where(data["abs_maxbatch"] <= -998, data["abs_minbatch_slices2"], (((((1.18234658241271973)) * 2.0)) * (((data["signal_shift_+1_msignal"]) * (((np.where(data["signal_shift_+1_msignal"] > -998, data["medianbatch_msignal"], data["abs_minbatch_slices2"] )) * 2.0))))) ))) )) +

                            0.100000*np.tanh((-((np.sin(((((data["minbatch_msignal"]) + (np.where(data["signal_shift_+1"] <= -998, np.sin(((3.0))), (3.0) )))/2.0))))))) +

                            0.100000*np.tanh(np.sin((((np.where(data["meanbatch_msignal"] > -998, data["signal_shift_+1"], np.where(np.tanh((np.tanh((data["meanbatch_msignal"])))) > -998, data["abs_maxbatch_msignal"], data["abs_avgbatch_msignal"] ) )) - (data["abs_avgbatch_msignal"]))))) +

                            0.100000*np.tanh(np.sin((np.where(data["maxtominbatch_slices2"] > -998, data["abs_minbatch_msignal"], ((np.sin((np.where(np.sin((np.tanh((np.sin((data["abs_minbatch_msignal"])))))) > -998, data["abs_minbatch_msignal"], (0.0) )))) * 2.0) )))) +

                            0.100000*np.tanh(np.sin((np.sin((np.where(data["medianbatch_slices2"] > -998, data["maxbatch_slices2_msignal"], ((data["abs_maxbatch_slices2"]) * (data["signal_shift_-1"])) )))))) +

                            0.100000*np.tanh(np.sin((((data["rangebatch_slices2"]) + (((np.cos((np.sin((((data["meanbatch_msignal"]) + (np.cos((data["stdbatch_slices2_msignal"]))))))))) / 2.0)))))) +

                            0.100000*np.tanh(((data["meanbatch_msignal"]) + (np.where(data["maxbatch_slices2"] > -998, np.where(data["meanbatch_msignal"] <= -998, data["abs_maxbatch_slices2_msignal"], (((data["medianbatch_msignal"]) + (np.where((-((data["rangebatch_slices2_msignal"]))) > -998, np.where(np.cos((np.sin((data["abs_avgbatch_msignal"])))) > -998, data["medianbatch_msignal"], (((data["medianbatch_msignal"]) + (data["medianbatch_msignal"]))/2.0) ), data["medianbatch_msignal"] )))/2.0) ), data["meanbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh(np.cos((((np.tanh(((((((1.0)) - (np.cos(((((data["abs_minbatch_slices2"]) + (np.cos((data["minbatch_msignal"]))))/2.0)))))) - (np.where(data["rangebatch_slices2"] > -998, data["meanbatch_msignal"], np.sin((np.tanh((data["medianbatch_slices2"])))) )))))) - (data["rangebatch_slices2"]))))) +

                            0.100000*np.tanh(np.where(np.where(np.cos((data["stdbatch_slices2"])) > -998, np.sin((data["stdbatch_slices2"])), ((np.where(data["meanbatch_msignal"] > -998, np.sin((((data["signal_shift_+1_msignal"]) * (data["medianbatch_msignal"])))), (-((np.sin((((data["medianbatch_slices2"]) * (data["medianbatch_msignal"]))))))) )) / 2.0) ) > -998, np.sin((((data["signal_shift_+1_msignal"]) * (data["medianbatch_msignal"])))), (-((data["medianbatch_msignal"]))) )) +

                            0.100000*np.tanh(np.where(data["rangebatch_slices2_msignal"] > -998, np.where(data["abs_maxbatch_slices2_msignal"] <= -998, np.cos((((data["minbatch_slices2"]) + (((data["rangebatch_slices2"]) + (data["rangebatch_slices2"])))))), np.cos((((data["minbatch_slices2"]) + (((np.cos((np.tanh((np.cos((data["signal_shift_+1_msignal"]))))))) - (np.where((-2.0) > -998, data["maxbatch_slices2"], data["medianbatch_msignal"] ))))))) ), data["mean_abs_chgbatch_slices2"] )) +

                            0.100000*np.tanh(((np.cos((np.where(((np.where(((data["signal_shift_-1_msignal"]) * (((((data["abs_maxbatch_slices2_msignal"]) * 2.0)) * (data["signal_shift_-1_msignal"])))) <= -998, data["signal_shift_-1"], ((data["abs_maxbatch_slices2_msignal"]) * (data["signal_shift_-1_msignal"])) )) * 2.0) <= -998, data["abs_minbatch_slices2"], ((data["abs_maxbatch_slices2_msignal"]) * 2.0) )))) * (((((data["abs_maxbatch_slices2_msignal"]) * 2.0)) * (data["signal_shift_-1_msignal"]))))) +

                            0.100000*np.tanh(((((data["maxbatch_slices2"]) / 2.0)) * (np.where(np.tanh((data["maxbatch_slices2"])) <= -998, ((np.where(data["signal_shift_-1_msignal"] > -998, data["medianbatch_slices2_msignal"], ((np.tanh((((data["medianbatch_slices2_msignal"]) + (data["maxbatch_slices2"]))))) - (np.tanh((data["maxbatch_slices2"])))) )) / 2.0), ((data["medianbatch_slices2_msignal"]) + (data["signal_shift_-1_msignal"])) )))) +

                            0.100000*np.tanh(np.sin((np.where(np.where(((np.cos((np.cos((data["mean_abs_chgbatch_slices2"]))))) / 2.0) > -998, np.sin((data["minbatch_msignal"])), data["medianbatch_slices2_msignal"] ) > -998, data["abs_maxbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh(np.where(np.cos((np.tanh(((-((((np.sin((data["abs_minbatch_slices2_msignal"]))) * (np.where(data["abs_minbatch_slices2"] > -998, data["abs_minbatch_slices2"], (((-((data["abs_minbatch_slices2"])))) + (data["minbatch_slices2"])) )))))))))) > -998, np.sin((data["abs_minbatch_slices2_msignal"])), ((np.cos(((-((data["meanbatch_slices2_msignal"])))))) * (data["abs_minbatch_msignal"])) )) +

                            0.100000*np.tanh(np.cos((np.where(np.sin(((((data["maxbatch_slices2_msignal"]) + (data["medianbatch_msignal"]))/2.0))) > -998, (((-(((-((data["meanbatch_slices2"]))))))) + (np.where(data["rangebatch_msignal"] <= -998, np.sin((np.sin((data["abs_minbatch_slices2_msignal"])))), data["rangebatch_slices2_msignal"] ))), np.cos((np.sin((data["rangebatch_msignal"])))) )))) +

                            0.100000*np.tanh(np.where(data["signal"] <= -998, ((data["meanbatch_slices2_msignal"]) * (data["maxbatch_slices2"])), np.where((7.0) <= -998, data["maxbatch_slices2"], (((3.0)) * ((((0.0)) + (((data["signal_shift_-1_msignal"]) - (((data["signal_shift_-1_msignal"]) * (((((data["medianbatch_msignal"]) * ((3.0)))) * (data["meanbatch_slices2_msignal"])))))))))) ) )) +

                            0.100000*np.tanh(((np.sin(((((((((-1.0)) * (data["minbatch_msignal"]))) + (data["abs_maxbatch_msignal"]))) - (np.where(data["rangebatch_msignal"] > -998, np.where(np.tanh((np.sin((np.where((((-1.0)) * (data["abs_maxbatch_slices2_msignal"])) <= -998, data["rangebatch_msignal"], (1.0) ))))) <= -998, data["rangebatch_msignal"], np.sin((data["abs_avgbatch_slices2_msignal"])) ), data["minbatch_slices2"] )))))) * 2.0)) +

                            0.100000*np.tanh(np.sin((((np.cos((((np.tanh((((data["medianbatch_slices2"]) - (((data["minbatch_slices2_msignal"]) * (data["meanbatch_slices2_msignal"]))))))) * (data["medianbatch_msignal"]))))) - (np.where(np.cos((data["meanbatch_slices2_msignal"])) <= -998, data["minbatch_msignal"], ((data["maxtominbatch_msignal"]) * (data["abs_avgbatch_slices2"])) )))))) +

                            0.100000*np.tanh(((((np.sin((((((np.where(np.sin(((-(((0.0)))))) > -998, data["maxbatch_slices2_msignal"], ((np.sin((data["maxbatch_slices2_msignal"]))) * 2.0) )) + ((0.0)))) + ((((-((data["signal_shift_-1_msignal"])))) / 2.0)))))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(data["medianbatch_msignal"]) +

                            0.100000*np.tanh((((((((-((np.where(data["medianbatch_slices2"] > -998, np.sin((np.cos(((3.0))))), (-((((data["medianbatch_msignal"]) - (((((data["maxtominbatch_slices2"]) - (((data["maxtominbatch_slices2"]) / 2.0)))) + ((-((data["minbatch_slices2"])))))))))) ))))) + (((data["maxtominbatch_slices2"]) + (((data["maxtominbatch_slices2"]) / 2.0)))))) / 2.0)) / 2.0)) +

                            0.100000*np.tanh(np.cos(((((-((np.where(data["medianbatch_slices2_msignal"] <= -998, np.where(data["minbatch_msignal"] > -998, ((np.cos((np.sin((np.cos((data["rangebatch_slices2_msignal"]))))))) * 2.0), data["minbatch"] ), ((((data["maxbatch_slices2_msignal"]) * (data["mean_abs_chgbatch_slices2"]))) * 2.0) ))))) * 2.0)))) +

                            0.100000*np.tanh(np.cos((np.where(np.cos((np.where(np.where(np.cos((data["abs_maxbatch"])) > -998, data["abs_maxbatch"], np.where(data["signal_shift_+1"] <= -998, data["signal_shift_-1_msignal"], data["abs_maxbatch"] ) ) > -998, data["abs_maxbatch"], data["rangebatch_slices2"] ))) > -998, data["abs_maxbatch"], data["abs_minbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh(np.sin((((np.sin((data["minbatch_msignal"]))) + (np.cos((data["minbatch_msignal"]))))))) +

                            0.100000*np.tanh(np.where(data["signal_shift_+1"] > -998, data["medianbatch_msignal"], (-1.0) )) +

                            0.100000*np.tanh(np.cos((((data["rangebatch_slices2_msignal"]) - (np.where(data["rangebatch_slices2_msignal"] <= -998, ((np.where(np.cos((((data["abs_minbatch_slices2_msignal"]) * ((((((data["signal_shift_+1"]) - (((((0.0)) + (data["rangebatch_slices2_msignal"]))/2.0)))) + (data["abs_minbatch_slices2_msignal"]))/2.0))))) <= -998, (-((data["rangebatch_slices2_msignal"]))), ((data["signal_shift_+1"]) * 2.0) )) / 2.0), (-((data["signal"]))) )))))) +

                            0.100000*np.tanh(((np.cos((((data["abs_minbatch_msignal"]) - (np.where(np.sin((data["abs_minbatch_msignal"])) > -998, ((data["rangebatch_msignal"]) / 2.0), ((data["medianbatch_msignal"]) - (np.where(data["abs_minbatch_msignal"] > -998, ((data["rangebatch_msignal"]) / 2.0), data["abs_minbatch_msignal"] ))) )))))) + (((((data["meanbatch_msignal"]) / 2.0)) / 2.0)))) +

                            0.100000*np.tanh(((np.where(data["meanbatch_slices2"] <= -998, np.cos((np.cos((((data["abs_maxbatch_msignal"]) * 2.0))))), ((((data["signal_shift_-1_msignal"]) * (data["abs_maxbatch_msignal"]))) * (data["maxbatch_slices2"])) )) * (np.where((((np.cos((data["rangebatch_msignal"]))) + (data["signal_shift_-1_msignal"]))/2.0) <= -998, (((3.0)) - (data["mean_abs_chgbatch_slices2_msignal"])), np.cos((((data["abs_maxbatch_msignal"]) * 2.0))) )))) +

                            0.100000*np.tanh(np.where(((np.sin((((data["abs_maxbatch_slices2_msignal"]) / 2.0)))) + (np.sin((data["abs_minbatch_slices2"])))) <= -998, ((np.tanh((np.sin((data["abs_minbatch_msignal"]))))) * (data["abs_minbatch_slices2_msignal"])), np.sin((data["meanbatch_slices2"])) )) +

                            0.100000*np.tanh(data["medianbatch_msignal"]) +

                            0.100000*np.tanh((-((np.sin((np.where(np.where((3.0) <= -998, data["minbatch_msignal"], ((data["mean_abs_chgbatch_msignal"]) - (np.cos((data["signal_shift_-1_msignal"])))) ) <= -998, np.sin((np.sin((((np.cos((data["minbatch"]))) - (np.sin((data["signal_shift_-1_msignal"])))))))), ((data["signal_shift_-1_msignal"]) - (((data["maxbatch_slices2_msignal"]) + (np.sin((data["medianbatch_slices2_msignal"])))))) ))))))) +

                            0.100000*np.tanh(np.where((((13.93729686737060547)) * 2.0) <= -998, ((np.cos((np.cos((data["signal_shift_+1"]))))) * 2.0), (((13.93729686737060547)) * (((np.sin((data["minbatch_msignal"]))) * (data["medianbatch_slices2_msignal"])))) )) +

                            0.100000*np.tanh(((np.where((((data["medianbatch_msignal"]) + (((data["maxtominbatch"]) - (data["medianbatch_slices2_msignal"]))))/2.0) <= -998, (-((np.sin((np.where((2.0) <= -998, np.cos((np.sin((((data["maxbatch_slices2_msignal"]) - (np.sin((((data["maxtominbatch_slices2"]) / 2.0))))))))), np.sin((data["signal_shift_-1_msignal"])) )))))), (6.0) )) / 2.0)) +

                            0.100000*np.tanh(np.cos((data["meanbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(np.where((-1.0) <= -998, np.sin(((-1.0))), (((((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) * (data["maxtominbatch"]))) + (data["mean_abs_chgbatch_msignal"]))/2.0) )) +

                            0.100000*np.tanh(np.sin(((((-((((data["abs_maxbatch_slices2"]) * (((data["maxtominbatch_msignal"]) + (((np.sin((np.sin(((((((-((np.sin((((np.sin(((3.0)))) + (data["maxtominbatch_msignal"])))))))) - (((data["abs_maxbatch_msignal"]) / 2.0)))) / 2.0)))))) / 2.0))))))))) / 2.0)))) +

                            0.100000*np.tanh(data["medianbatch_slices2"]) +

                            0.100000*np.tanh(((data["abs_minbatch_slices2"]) / 2.0)) +

                            0.100000*np.tanh(((data["maxtominbatch"]) * (np.where(data["maxtominbatch"] > -998, np.tanh(((-((((((data["medianbatch_msignal"]) * 2.0)) * 2.0)))))), np.tanh((data["meanbatch_msignal"])) )))) +

                            0.100000*np.tanh(((data["maxtominbatch_slices2_msignal"]) + (np.where((((1.0)) + (((((0.0)) + (data["signal_shift_+1_msignal"]))/2.0))) > -998, (7.0), np.tanh((((data["abs_maxbatch_slices2_msignal"]) * (data["medianbatch_slices2_msignal"])))) )))) +

                            0.100000*np.tanh(np.sin((np.cos((np.tanh((np.cos((((np.where(data["abs_minbatch_slices2_msignal"] > -998, (-((np.cos((np.where((((((data["maxtominbatch_slices2_msignal"]) / 2.0)) + (data["abs_maxbatch_slices2"]))/2.0) > -998, (((np.tanh((np.cos((data["abs_avgbatch_slices2_msignal"]))))) + ((-1.0)))/2.0), (-(((0.0)))) )))))), data["abs_avgbatch_msignal"] )) / 2.0)))))))))) +

                            0.100000*np.tanh((((((-2.0)) + (data["abs_maxbatch_msignal"]))) * (((data["signal_shift_+1_msignal"]) * (np.where(data["stdbatch_msignal"] > -998, (((((-2.0)) + ((((-2.0)) + (data["abs_maxbatch_msignal"]))))) * ((-2.0))), np.where(np.where(data["abs_maxbatch"] > -998, data["abs_maxbatch"], data["abs_maxbatch_slices2"] ) > -998, data["abs_maxbatch"], data["signal_shift_+1_msignal"] ) )))))) +

                            0.100000*np.tanh((((data["meanbatch_slices2"]) + (((((data["maxtominbatch_msignal"]) * (np.tanh((np.cos((((np.where(data["rangebatch_slices2_msignal"] > -998, data["rangebatch_slices2_msignal"], ((data["maxtominbatch_msignal"]) * (np.tanh((((data["abs_minbatch_slices2"]) + (((data["meanbatch_slices2"]) / 2.0))))))) )) * 2.0)))))))) / 2.0)))/2.0)) +

                            0.100000*np.tanh(np.where((-((data["stdbatch_slices2"]))) <= -998, (((np.sin(((-((data["abs_maxbatch_slices2_msignal"])))))) + (np.sin((data["abs_avgbatch_slices2_msignal"]))))/2.0), np.sin(((-((data["abs_avgbatch_slices2_msignal"]))))) )) +

                            0.100000*np.tanh(np.tanh((((np.where(np.cos((data["meanbatch_slices2_msignal"])) > -998, data["signal_shift_-1_msignal"], np.cos((data["abs_maxbatch_slices2_msignal"])) )) * (np.cos((((np.where((-((data["abs_maxbatch_slices2_msignal"]))) <= -998, np.where(data["meanbatch_slices2_msignal"] > -998, data["signal_shift_-1_msignal"], data["minbatch_slices2"] ), data["maxbatch_slices2_msignal"] )) - ((-((data["abs_maxbatch_slices2_msignal"])))))))))))) +

                            0.100000*np.tanh(np.sin((np.where(data["meanbatch_msignal"] > -998, np.cos((data["abs_maxbatch"])), np.where(np.where((-((((data["medianbatch_slices2_msignal"]) / 2.0)))) > -998, np.cos((np.cos((data["abs_maxbatch"])))), ((np.cos((data["medianbatch_slices2_msignal"]))) - (np.cos((np.sin(((0.0))))))) ) <= -998, (-((data["maxtominbatch_slices2"]))), ((data["medianbatch_slices2_msignal"]) + (data["rangebatch_slices2"])) ) )))) +

                            0.100000*np.tanh(((data["minbatch_slices2"]) - (np.where(((((((np.cos((((data["abs_maxbatch_slices2_msignal"]) - ((((data["abs_minbatch_msignal"]) + (data["signal_shift_+1"]))/2.0)))))) + (((data["minbatch_slices2_msignal"]) + (((data["maxbatch_slices2_msignal"]) * 2.0)))))/2.0)) + (data["signal_shift_+1"]))/2.0) > -998, np.where(data["minbatch_slices2_msignal"] <= -998, ((data["minbatch_slices2"]) / 2.0), data["minbatch"] ), (((data["minbatch_slices2_msignal"]) + (data["signal_shift_-1"]))/2.0) )))) +

                            0.100000*np.tanh(np.cos((np.where(data["rangebatch_slices2_msignal"] <= -998, ((data["maxtominbatch_msignal"]) * 2.0), ((data["abs_maxbatch_slices2_msignal"]) - (data["mean_abs_chgbatch_slices2"])) )))) +

                            0.100000*np.tanh(np.cos(((((((((((data["meanbatch_msignal"]) / 2.0)) / 2.0)) + (data["abs_maxbatch_msignal"]))/2.0)) - (data["meanbatch_slices2"]))))) +

                            0.100000*np.tanh(((((data["abs_maxbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2_msignal"]))) * (((data["abs_maxbatch_slices2_msignal"]) * (((data["abs_maxbatch_slices2_msignal"]) / 2.0)))))) +

                            0.100000*np.tanh(((np.sin((data["abs_minbatch_slices2"]))) * 2.0)) +

                            0.099609*np.tanh(np.where(np.cos((data["rangebatch_slices2_msignal"])) <= -998, np.where(np.sin((data["meanbatch_slices2"])) <= -998, data["mean_abs_chgbatch_slices2"], np.cos((((data["meanbatch_slices2"]) / 2.0))) ), np.sin((((np.where(np.sin((np.tanh((data["maxbatch_slices2"])))) > -998, data["maxbatch_slices2"], data["abs_minbatch_slices2_msignal"] )) * 2.0))) )) +

                            0.100000*np.tanh(np.sin(((((data["medianbatch_msignal"]) + (np.cos((np.tanh(((7.54751968383789062)))))))/2.0)))) +

                            0.100000*np.tanh(((((data["mean_abs_chgbatch_slices2"]) + (np.cos((np.cos((data["maxbatch_msignal"]))))))) * (np.sin(((-(((((data["rangebatch_slices2"]) + (data["maxbatch_msignal"]))/2.0))))))))) +

                            0.100000*np.tanh(np.where((((data["abs_maxbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2"]))/2.0) <= -998, np.cos((np.tanh((data["abs_maxbatch"])))), np.where((((((np.sin((np.cos((data["signal_shift_+1_msignal"]))))) + (data["abs_avgbatch_slices2"]))/2.0)) / 2.0) <= -998, data["abs_avgbatch_slices2_msignal"], data["abs_minbatch_slices2"] ) )) +

                            0.100000*np.tanh((-((np.cos((np.where(((((data["abs_maxbatch_msignal"]) - (data["maxtominbatch_msignal"]))) * (data["medianbatch_msignal"])) > -998, data["maxtominbatch_slices2_msignal"], np.where((((-((np.cos((((data["medianbatch_slices2"]) / 2.0))))))) + (data["maxtominbatch_slices2_msignal"])) > -998, (9.43180370330810547), np.sin((data["abs_avgbatch_msignal"])) ) ))))))) +

                            0.100000*np.tanh(np.where((10.73174285888671875) <= -998, data["abs_maxbatch_slices2_msignal"], np.tanh(((((((((data["signal_shift_-1_msignal"]) + (((np.sin((((np.tanh((np.sin((data["maxtominbatch_slices2"]))))) / 2.0)))) * 2.0)))/2.0)) + (((np.tanh(((0.0)))) * 2.0)))) / 2.0))) )) +

                            0.100000*np.tanh(np.cos((((np.tanh((np.sin((np.sin((np.tanh((((data["signal"]) + (data["abs_maxbatch_slices2"]))))))))))) * 2.0)))) +

                            0.100000*np.tanh(np.sin((np.where(np.sin((data["maxbatch_slices2_msignal"])) > -998, np.sin((np.where((-((np.sin((np.cos(((-((data["abs_avgbatch_msignal"])))))))))) > -998, ((data["abs_maxbatch_slices2"]) * 2.0), data["medianbatch_slices2_msignal"] ))), ((((((data["maxbatch_slices2_msignal"]) - (((data["medianbatch_slices2_msignal"]) * (data["maxbatch_slices2_msignal"]))))) * (data["mean_abs_chgbatch_msignal"]))) * 2.0) )))) +

                            0.100000*np.tanh(((((np.where(data["medianbatch_msignal"] <= -998, data["signal_shift_+1_msignal"], np.sin((data["maxbatch_slices2_msignal"])) )) - (np.tanh((data["meanbatch_slices2"]))))) + (((data["signal_shift_+1_msignal"]) + (((np.sin((data["maxbatch_slices2_msignal"]))) * 2.0)))))) +

                            0.100000*np.tanh(np.sin((np.where(data["minbatch_msignal"] > -998, ((data["maxbatch_msignal"]) * 2.0), np.where(((np.sin(((-3.0)))) / 2.0) <= -998, (0.0), np.sin((((((data["mean_abs_chgbatch_msignal"]) * 2.0)) * (data["stdbatch_slices2_msignal"])))) ) )))) +

                            0.100000*np.tanh(((((np.where(data["medianbatch_msignal"] > -998, (-((((((((data["mean_abs_chgbatch_slices2"]) + (data["abs_maxbatch"]))/2.0)) + (data["mean_abs_chgbatch_slices2"]))/2.0)))), np.cos((np.sin((data["abs_avgbatch_slices2_msignal"])))) )) + (data["abs_maxbatch_msignal"]))) * 2.0)) +

                            0.089932*np.tanh(np.tanh((np.sin((np.tanh((np.where(data["mean_abs_chgbatch_msignal"] > -998, (0.0), np.where(data["rangebatch_msignal"] > -998, np.cos((data["abs_maxbatch_slices2_msignal"])), np.tanh((data["abs_maxbatch_slices2_msignal"])) ) )))))))) +

                            0.100000*np.tanh(np.where(data["signal_shift_+1_msignal"] <= -998, np.where(data["minbatch"] <= -998, ((data["medianbatch_slices2_msignal"]) * (np.tanh((data["maxtominbatch"])))), (((data["abs_minbatch_slices2"]) + (data["minbatch"]))/2.0) ), np.tanh((np.where(np.where(data["mean_abs_chgbatch_msignal"] > -998, data["signal_shift_-1"], (1.0) ) <= -998, data["maxbatch_slices2_msignal"], data["mean_abs_chgbatch_slices2"] ))) )) +

                            0.098436*np.tanh((1.0)) +

                            0.100000*np.tanh(np.cos((np.where(np.where(data["medianbatch_msignal"] > -998, data["abs_maxbatch_slices2"], np.cos(((-((data["meanbatch_slices2"]))))) ) > -998, data["abs_maxbatch_slices2"], data["abs_maxbatch_slices2"] )))) +

                            0.100000*np.tanh(np.where(data["maxbatch_slices2_msignal"] > -998, np.sin((np.sin((((data["maxbatch_msignal"]) - ((1.0))))))), ((np.sin((data["maxbatch_slices2_msignal"]))) * 2.0) )) +

                            0.100000*np.tanh(np.cos((((((np.tanh((np.cos((data["abs_maxbatch_msignal"]))))) + (((np.where((1.0) > -998, data["minbatch_msignal"], np.tanh(((2.0))) )) * (np.where((((((-3.0)) - (((np.sin((data["abs_maxbatch_slices2"]))) * 2.0)))) / 2.0) > -998, data["maxbatch_slices2_msignal"], data["abs_maxbatch_slices2"] )))))) / 2.0)))) +

                            0.100000*np.tanh(((data["mean_abs_chgbatch_slices2"]) * (data["abs_minbatch_slices2"]))) +

                            0.100000*np.tanh(np.sin((np.cos((np.sin((data["stdbatch_slices2"]))))))) +

                            0.100000*np.tanh(((data["signal_shift_+1_msignal"]) + (((np.sin((np.where(((data["signal_shift_-1_msignal"]) * (data["maxbatch_slices2_msignal"])) <= -998, np.cos((((((((data["maxbatch_slices2_msignal"]) * 2.0)) * (data["maxbatch_slices2_msignal"]))) * (data["maxbatch_slices2_msignal"])))), data["maxbatch_slices2_msignal"] )))) * 2.0)))) +

                            0.100000*np.tanh(np.cos((np.where(data["meanbatch_msignal"] > -998, ((data["minbatch_msignal"]) - (((data["abs_avgbatch_slices2_msignal"]) / 2.0))), np.where(((data["rangebatch_slices2"]) - (np.cos(((0.0))))) > -998, (0.0), (14.92436885833740234) ) )))) +

                            0.093744*np.tanh((((-((np.cos((((data["minbatch_msignal"]) / 2.0))))))) + (np.sin((np.where(data["rangebatch_msignal"] > -998, data["meanbatch_msignal"], np.tanh((data["abs_maxbatch_slices2"])) )))))) +

                            0.100000*np.tanh(((np.where(np.tanh((data["signal_shift_-1"])) <= -998, np.tanh(((((3.0)) * 2.0))), np.sin((np.where(data["abs_avgbatch_slices2_msignal"] > -998, np.where(data["abs_avgbatch_slices2_msignal"] > -998, (((((3.0)) * 2.0)) * (data["abs_maxbatch_msignal"])), (0.0) ), data["meanbatch_slices2_msignal"] ))) )) * 2.0)) +

                            0.099707*np.tanh((2.0)) +

                            0.100000*np.tanh(np.cos((np.cos(((((np.sin((data["mean_abs_chgbatch_msignal"]))) + (data["abs_minbatch_slices2"]))/2.0)))))) +

                            0.100000*np.tanh(np.sin((((data["abs_maxbatch"]) + ((((data["medianbatch_msignal"]) + (np.sin(((-((np.sin((((((data["abs_avgbatch_slices2_msignal"]) - (np.where(data["abs_maxbatch_msignal"] > -998, data["mean_abs_chgbatch_slices2_msignal"], np.sin((np.cos((data["abs_maxbatch"])))) )))) + (((np.where((0.0) <= -998, data["abs_minbatch_msignal"], data["signal_shift_+1"] )) / 2.0))))))))))))/2.0)))))) +

                            0.100000*np.tanh(np.sin((np.where(np.where(((data["medianbatch_msignal"]) * 2.0) > -998, data["abs_maxbatch_slices2"], (((np.cos((data["signal_shift_+1_msignal"]))) + (np.where(((((data["abs_maxbatch_slices2"]) / 2.0)) / 2.0) <= -998, np.where((6.54726648330688477) > -998, data["abs_minbatch_slices2"], data["mean_abs_chgbatch_slices2"] ), data["signal_shift_+1"] )))/2.0) ) <= -998, (0.0), data["signal_shift_+1"] )))))   

   

    def GP_class_5(self,data):

        return self.Output( -2.889212 +

                            0.100000*np.tanh(np.where(((np.tanh((np.where(data["rangebatch_slices2"] <= -998, data["meanbatch_slices2"], data["meanbatch_slices2"] )))) - (np.where(np.where(np.where(data["abs_maxbatch"] > -998, data["meanbatch_slices2"], data["medianbatch_slices2"] ) <= -998, data["maxbatch_msignal"], data["medianbatch_slices2"] ) > -998, np.tanh((((data["rangebatch_msignal"]) * 2.0))), (1.0) ))) > -998, data["signal_shift_-1"], data["meanbatch_slices2"] )) +

                            0.100000*np.tanh(((((((np.sin((data["maxbatch_slices2_msignal"]))) * ((-(((((-3.0)) - (np.cos((data["rangebatch_slices2"])))))))))) / 2.0)) * (((((((data["signal"]) / 2.0)) / 2.0)) + (np.where(((data["stdbatch_slices2"]) / 2.0) > -998, np.where(np.tanh((data["signal_shift_+1"])) > -998, data["signal"], np.tanh((data["signal_shift_-1"])) ), data["meanbatch_slices2"] )))))) +

                            0.100000*np.tanh(np.where(data["maxbatch_msignal"] <= -998, (3.0), np.where(data["signal_shift_+1"] > -998, data["signal_shift_+1"], data["rangebatch_slices2"] ) )) +

                            0.100000*np.tanh(np.where(data["abs_maxbatch_msignal"] <= -998, np.where(data["maxbatch_msignal"] > -998, data["maxtominbatch_slices2"], ((((((((((np.where((0.0) <= -998, (((0.0)) / 2.0), data["medianbatch_msignal"] )) + ((((((data["signal"]) + (data["signal"]))) + (((data["stdbatch_slices2_msignal"]) * 2.0)))/2.0)))/2.0)) * (data["abs_avgbatch_slices2_msignal"]))) + (((data["abs_minbatch_slices2"]) / 2.0)))/2.0)) * 2.0) ), data["signal"] )) +

                            0.100000*np.tanh((((-((np.where(((((np.cos((data["medianbatch_slices2"]))) - (data["maxbatch_msignal"]))) * 2.0) <= -998, data["medianbatch_msignal"], (-((data["medianbatch_slices2"]))) ))))) + ((((((3.0)) / 2.0)) * 2.0)))) +

                            0.100000*np.tanh((((-2.0)) + (((data["meanbatch_msignal"]) - (np.where((((-2.0)) + (data["stdbatch_msignal"])) > -998, data["maxtominbatch_slices2_msignal"], np.sin((np.sin(((((np.tanh((np.sin((data["abs_maxbatch"]))))) + ((((-1.0)) / 2.0)))/2.0))))) )))))) +

                            0.100000*np.tanh(((((((np.cos(((3.0)))) - (np.where(data["stdbatch_slices2"] <= -998, (-((data["maxtominbatch_slices2"]))), np.where(data["maxbatch_slices2_msignal"] <= -998, np.cos(((((data["maxbatch_slices2_msignal"]) + ((-((np.cos((data["maxbatch_slices2_msignal"])))))))/2.0))), np.where(data["minbatch"] > -998, data["meanbatch_msignal"], data["maxbatch_slices2_msignal"] ) ) )))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.where(data["medianbatch_slices2_msignal"] <= -998, (-((data["minbatch_msignal"]))), (-((np.where(data["meanbatch_slices2_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], data["minbatch_slices2"] )))) )) +

                            0.100000*np.tanh(((np.sin((data["maxbatch_slices2_msignal"]))) * (np.where(np.tanh((((data["maxbatch_slices2"]) / 2.0))) > -998, data["medianbatch_slices2"], (((4.27818489074707031)) * (data["maxbatch_slices2_msignal"])) )))) +

                            0.100000*np.tanh(np.where(data["minbatch"] <= -998, data["mean_abs_chgbatch_slices2_msignal"], ((data["meanbatch_msignal"]) * (data["abs_avgbatch_slices2_msignal"])) )) +

                            0.100000*np.tanh(((np.where((1.0) > -998, data["maxbatch_slices2_msignal"], ((np.cos((np.where(data["maxbatch_slices2_msignal"] <= -998, data["minbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] )))) / 2.0) )) * ((-((np.cos((data["meanbatch_slices2_msignal"])))))))) +

                            0.100000*np.tanh((((-3.0)) * (np.where((((((-3.0)) * ((-3.0)))) * ((2.0))) > -998, np.cos((data["meanbatch_slices2_msignal"])), ((data["meanbatch_slices2_msignal"]) * (((np.cos((data["maxtominbatch_msignal"]))) / 2.0))) )))) +

                            0.100000*np.tanh(np.where(data["medianbatch_slices2"] > -998, data["signal_shift_+1"], (((((((((np.sin((data["minbatch_slices2"]))) * 2.0)) + (((data["signal_shift_+1_msignal"]) * ((-3.0)))))) + (data["maxtominbatch_slices2"]))/2.0)) - ((-3.0))) )) +

                            0.100000*np.tanh(np.sin((((np.where(data["abs_avgbatch_msignal"] > -998, data["abs_maxbatch_msignal"], ((np.tanh((data["abs_maxbatch"]))) - (((np.where(np.tanh(((0.0))) > -998, data["minbatch_slices2"], ((np.where(data["maxbatch_slices2_msignal"] > -998, ((((np.tanh((data["maxbatch_msignal"]))) + (data["signal_shift_+1"]))) / 2.0), data["abs_maxbatch_msignal"] )) * 2.0) )) * 2.0))) )) * 2.0)))) +

                            0.100000*np.tanh(((((np.cos((np.where(np.cos((((data["minbatch_slices2"]) * 2.0))) > -998, data["maxbatch_msignal"], ((data["medianbatch_slices2_msignal"]) + (np.where(data["stdbatch_slices2"] > -998, data["maxbatch_msignal"], np.cos((((np.cos((data["maxbatch_msignal"]))) * 2.0))) ))) )))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(((np.cos((data["minbatch_msignal"]))) * (np.where(data["minbatch_msignal"] > -998, data["signal_shift_+1"], np.sin(((((2.0)) * (data["minbatch_msignal"])))) )))) +

                            0.100000*np.tanh(np.where(np.tanh((data["maxtominbatch_msignal"])) > -998, ((np.where((-1.0) > -998, np.cos((np.where(((np.cos((data["maxbatch_slices2_msignal"]))) / 2.0) > -998, data["maxbatch_slices2_msignal"], np.cos((np.tanh((data["maxbatch_slices2_msignal"])))) ))), data["maxbatch_slices2_msignal"] )) * 2.0), ((data["rangebatch_slices2"]) + ((-1.0))) )) +

                            0.100000*np.tanh(((data["abs_maxbatch_slices2_msignal"]) * (np.sin((((np.tanh((np.cos((np.sin((np.tanh((((((data["abs_maxbatch_slices2_msignal"]) + (data["signal_shift_-1"]))) + ((8.92028522491455078)))))))))))) + (np.where(((data["rangebatch_slices2"]) + (data["mean_abs_chgbatch_slices2"])) > -998, data["abs_maxbatch_slices2_msignal"], (-((np.sin((data["abs_maxbatch_slices2_msignal"]))))) )))))))) +

                            0.100000*np.tanh(((np.sin((((data["abs_maxbatch_msignal"]) - (np.cos((np.where(data["signal_shift_+1"] <= -998, np.where(data["abs_maxbatch_msignal"] <= -998, np.tanh((np.sin((data["abs_maxbatch_msignal"])))), data["meanbatch_msignal"] ), data["abs_maxbatch_msignal"] )))))))) - (np.cos((np.where(data["signal_shift_+1"] <= -998, data["abs_maxbatch"], data["meanbatch_msignal"] )))))) +

                            0.100000*np.tanh(((np.cos((np.where((-3.0) <= -998, data["medianbatch_slices2"], data["maxbatch_msignal"] )))) + (np.cos((data["minbatch_msignal"]))))) +

                            0.100000*np.tanh(((data["meanbatch_msignal"]) * (((np.where(data["maxbatch_msignal"] > -998, data["meanbatch_msignal"], data["abs_minbatch_msignal"] )) * (np.sin((np.where(data["meanbatch_msignal"] > -998, data["maxbatch_msignal"], np.where(np.where(((data["signal_shift_-1"]) + (np.sin((data["maxbatch_msignal"])))) > -998, data["meanbatch_msignal"], data["abs_minbatch_msignal"] ) > -998, data["maxtominbatch"], data["meanbatch_msignal"] ) )))))))) +

                            0.100000*np.tanh(np.where(np.where(np.sin((np.sin((data["abs_maxbatch_slices2_msignal"])))) > -998, (((3.0)) + (data["maxbatch_msignal"])), data["medianbatch_slices2"] ) <= -998, data["signal"], ((data["signal_shift_+1"]) * (np.sin((np.cos((data["maxbatch_slices2_msignal"])))))) )) +

                            0.100000*np.tanh(((np.cos((np.where(np.sin((((np.where(np.cos((data["abs_maxbatch_slices2_msignal"])) > -998, data["abs_maxbatch_slices2_msignal"], data["abs_maxbatch_slices2_msignal"] )) + (data["abs_maxbatch_slices2_msignal"])))) > -998, ((data["abs_maxbatch_slices2_msignal"]) - (np.where(data["abs_maxbatch_slices2_msignal"] > -998, np.cos((data["abs_maxbatch_slices2_msignal"])), np.tanh((data["minbatch_slices2_msignal"])) ))), ((data["signal_shift_+1"]) * 2.0) )))) * 2.0)) +

                            0.100000*np.tanh(np.where(np.cos((((((np.cos((data["abs_maxbatch_slices2_msignal"]))) * 2.0)) * 2.0))) > -998, ((np.cos((data["abs_maxbatch_slices2_msignal"]))) * 2.0), ((np.where(np.cos((np.tanh((data["abs_maxbatch_slices2_msignal"])))) > -998, ((np.cos((data["abs_maxbatch_slices2_msignal"]))) * 2.0), data["medianbatch_slices2"] )) - (data["abs_maxbatch_slices2_msignal"])) )) +

                            0.100000*np.tanh(((data["medianbatch_msignal"]) * (np.cos(((((data["rangebatch_slices2"]) + (np.tanh((np.cos(((((data["rangebatch_slices2"]) + (np.tanh((data["medianbatch_msignal"]))))/2.0)))))))/2.0)))))) +

                            0.100000*np.tanh((((6.92381620407104492)) * (np.sin((((np.where((6.92381620407104492) > -998, data["meanbatch_slices2_msignal"], np.sin((np.cos((np.where((6.92381620407104492) > -998, data["medianbatch_msignal"], (6.92381620407104492) ))))) )) - ((2.0)))))))) +

                            0.100000*np.tanh(((data["signal_shift_-1"]) * (np.cos((np.where(((data["abs_maxbatch_slices2"]) * (np.cos((((data["stdbatch_slices2"]) * 2.0))))) <= -998, np.tanh(((((data["signal_shift_-1"]) + (((data["signal_shift_-1"]) * 2.0)))/2.0))), data["minbatch_msignal"] )))))) +

                            0.100000*np.tanh(((((np.sin((np.where(np.sin((((np.sin((((data["stdbatch_slices2"]) - ((-((np.sin(((((data["mean_abs_chgbatch_slices2"]) + ((1.0)))/2.0))))))))))) * 2.0))) > -998, data["abs_maxbatch_msignal"], (6.79426574707031250) )))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.where(np.cos(((-((data["minbatch_msignal"]))))) <= -998, data["mean_abs_chgbatch_slices2_msignal"], ((np.cos((((data["medianbatch_msignal"]) * 2.0)))) * (np.where((-((data["minbatch_msignal"]))) <= -998, np.sin((np.where(data["meanbatch_slices2_msignal"] > -998, data["abs_maxbatch_msignal"], data["signal"] ))), data["meanbatch_slices2_msignal"] ))) )) +

                            0.100000*np.tanh(((np.where((((2.0)) * 2.0) > -998, np.sin((data["maxbatch_msignal"])), (-((np.cos(((((((((1.0)) * (np.cos((data["medianbatch_slices2"]))))) * 2.0)) * 2.0)))))) )) * 2.0)) +

                            0.100000*np.tanh(np.cos((np.where(data["stdbatch_slices2_msignal"] > -998, data["rangebatch_slices2"], np.where(data["maxtominbatch_msignal"] > -998, data["rangebatch_slices2"], np.cos((((data["medianbatch_slices2"]) * ((3.0))))) ) )))) +

                            0.100000*np.tanh(((data["minbatch_msignal"]) * (np.where((-3.0) > -998, np.cos((data["meanbatch_msignal"])), np.cos(((-(((-3.0)))))) )))) +

                            0.100000*np.tanh(((((((np.where(np.where(((np.sin((((((data["stdbatch_slices2"]) * 2.0)) * 2.0)))) * 2.0) > -998, np.cos((data["maxbatch_slices2_msignal"])), (4.41485977172851562) ) > -998, np.cos((data["abs_maxbatch_slices2_msignal"])), (((data["meanbatch_msignal"]) + ((5.0)))/2.0) )) * 2.0)) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.sin((((np.where(np.where(data["meanbatch_slices2"] <= -998, data["signal_shift_-1_msignal"], ((data["stdbatch_slices2_msignal"]) + ((-3.0))) ) > -998, np.sin((np.where(data["abs_maxbatch_msignal"] <= -998, (-((data["abs_avgbatch_msignal"]))), (((-1.0)) - (data["medianbatch_msignal"])) ))), np.sin(((((3.0)) + (data["meanbatch_slices2"])))) )) * 2.0)))) +

                            0.100000*np.tanh(np.cos((np.where(data["mean_abs_chgbatch_msignal"] > -998, data["minbatch_msignal"], np.where((0.15266779065132141) <= -998, ((data["maxbatch_slices2"]) / 2.0), ((np.where(data["maxbatch_slices2"] > -998, data["meanbatch_msignal"], data["minbatch_msignal"] )) * (np.where(data["stdbatch_slices2"] > -998, data["meanbatch_msignal"], data["maxtominbatch_msignal"] ))) ) )))) +

                            0.100000*np.tanh(np.sin((np.where((-(((0.0)))) > -998, data["abs_maxbatch_msignal"], np.where(np.cos((data["medianbatch_msignal"])) <= -998, np.sin((data["meanbatch_slices2"])), (((3.0)) - (np.sin((np.where(np.cos((data["minbatch"])) > -998, data["meanbatch_slices2"], ((np.cos((data["abs_maxbatch_msignal"]))) * 2.0) ))))) ) )))) +

                            0.100000*np.tanh(np.where(np.sin((data["maxbatch_slices2_msignal"])) > -998, ((((np.cos((data["abs_maxbatch_slices2_msignal"]))) * 2.0)) * 2.0), np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["maxtominbatch_slices2"], data["maxbatch_slices2_msignal"] ) )) +

                            0.100000*np.tanh(((np.where(((data["rangebatch_slices2"]) - (np.where(data["maxtominbatch_slices2"] > -998, data["stdbatch_slices2"], (3.67932653427124023) ))) <= -998, (((5.0)) * (np.sin((data["abs_maxbatch_slices2_msignal"])))), (-1.0) )) + (data["meanbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(np.where(((data["rangebatch_slices2"]) / 2.0) <= -998, (-((data["abs_minbatch_slices2_msignal"]))), ((np.cos((data["meanbatch_msignal"]))) * ((-((((data["rangebatch_slices2"]) * 2.0)))))) )) +

                            0.100000*np.tanh(((np.where(((np.tanh((((((data["rangebatch_slices2"]) - (((np.tanh(((12.08567523956298828)))) - (np.cos((np.tanh((data["maxbatch_slices2"]))))))))) / 2.0)))) / 2.0) > -998, data["medianbatch_slices2"], (-(((((data["maxbatch_msignal"]) + (((np.sin((data["abs_maxbatch_slices2"]))) / 2.0)))/2.0)))) )) * (np.sin((data["abs_maxbatch_msignal"]))))) +

                            0.100000*np.tanh(((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, (((-(((((-1.0)) + (((data["meanbatch_slices2_msignal"]) * (data["abs_maxbatch_msignal"])))))))) / 2.0), ((((((np.cos((data["maxbatch_slices2_msignal"]))) * 2.0)) + (((((np.sin((data["rangebatch_msignal"]))) - (np.sin((((data["abs_maxbatch_slices2_msignal"]) / 2.0)))))) / 2.0)))) * 2.0) )) * 2.0)) +

                            0.100000*np.tanh(np.where(data["medianbatch_msignal"] > -998, ((data["abs_avgbatch_slices2"]) - ((2.0))), np.sin((np.where(data["medianbatch_msignal"] <= -998, ((data["maxbatch_msignal"]) + (((np.where(data["rangebatch_slices2"] > -998, data["medianbatch_slices2"], np.sin((((data["medianbatch_slices2_msignal"]) - (data["abs_minbatch_msignal"])))) )) * (data["medianbatch_slices2_msignal"])))), data["abs_maxbatch_msignal"] ))) )) +

                            0.100000*np.tanh(np.where((1.0) <= -998, ((((-((np.cos(((1.0))))))) + ((-((data["abs_avgbatch_msignal"])))))/2.0), ((data["abs_avgbatch_msignal"]) * (((data["minbatch_msignal"]) * (np.cos((data["minbatch_slices2_msignal"])))))) )) +

                            0.100000*np.tanh(np.where((0.0) > -998, np.tanh((np.where(data["signal"] > -998, ((data["maxbatch_slices2_msignal"]) - ((6.0))), np.tanh((np.where(data["maxtominbatch_msignal"] > -998, (-((data["abs_avgbatch_slices2_msignal"]))), data["maxbatch_slices2_msignal"] ))) ))), np.where(data["maxtominbatch_msignal"] > -998, data["maxbatch_slices2_msignal"], (0.0) ) )) +

                            0.100000*np.tanh(np.where((((data["maxtominbatch_slices2"]) + (np.where(((np.cos((data["minbatch_slices2_msignal"]))) - (data["medianbatch_slices2_msignal"])) > -998, (-((data["mean_abs_chgbatch_slices2"]))), (-((data["minbatch_slices2_msignal"]))) )))/2.0) > -998, (-((((data["abs_avgbatch_slices2_msignal"]) * (np.cos((data["minbatch_slices2_msignal"]))))))), data["maxtominbatch_slices2"] )) +

                            0.100000*np.tanh(np.where(data["abs_minbatch_slices2_msignal"] > -998, ((np.sin(((((data["abs_minbatch_slices2_msignal"]) + (((data["medianbatch_slices2_msignal"]) * 2.0)))/2.0)))) * 2.0), np.tanh((((((((((data["medianbatch_slices2_msignal"]) * 2.0)) * 2.0)) / 2.0)) / 2.0))) )) +

                            0.100000*np.tanh(((((np.where(np.tanh((data["rangebatch_msignal"])) <= -998, data["maxbatch_msignal"], np.sin(((((((data["mean_abs_chgbatch_msignal"]) - (((data["rangebatch_msignal"]) + (data["abs_avgbatch_slices2"]))))) + (data["maxbatch_msignal"]))/2.0))) )) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.where(((np.cos((np.cos((data["rangebatch_slices2"]))))) * (data["maxbatch_msignal"])) <= -998, (1.0), ((data["rangebatch_slices2"]) * ((-((((((data["signal"]) - (data["maxbatch_msignal"]))) * (np.sin((np.sin((data["minbatch_msignal"]))))))))))) )) +

                            0.100000*np.tanh((((-((np.cos((np.where(np.sin((data["maxbatch_slices2_msignal"])) <= -998, (0.74201720952987671), (((data["minbatch_msignal"]) + (((np.cos(((((np.where(np.cos((data["minbatch_msignal"])) > -998, np.where(data["maxbatch_slices2"] <= -998, np.cos(((14.60533332824707031))), data["abs_avgbatch_slices2"] ), data["medianbatch_slices2_msignal"] )) + (((data["minbatch_msignal"]) / 2.0)))/2.0)))) / 2.0)))/2.0) ))))))) * 2.0)) +

                            0.100000*np.tanh(np.cos((((data["minbatch"]) - (np.where(((((-2.0)) + ((-((data["medianbatch_slices2_msignal"])))))/2.0) > -998, data["signal_shift_-1"], ((np.cos((data["signal_shift_+1_msignal"]))) * ((0.0))) )))))) +

                            0.100000*np.tanh(np.sin((np.where(np.sin((np.where(((data["abs_maxbatch_msignal"]) * 2.0) > -998, data["abs_maxbatch_msignal"], data["abs_maxbatch_msignal"] ))) > -998, data["abs_maxbatch_msignal"], np.where(data["rangebatch_msignal"] <= -998, data["rangebatch_slices2"], data["abs_maxbatch"] ) )))) +

                            0.100000*np.tanh((((((data["medianbatch_slices2"]) + (((data["medianbatch_slices2"]) * (data["medianbatch_slices2"]))))/2.0)) * (np.sin((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, data["minbatch_slices2"], np.sin((((data["abs_maxbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2_msignal"])))) )))))) +

                            0.100000*np.tanh(np.sin((np.where(np.tanh(((-((np.where(data["signal_shift_-1_msignal"] <= -998, np.sin((((((data["stdbatch_msignal"]) * 2.0)) + ((-((data["abs_avgbatch_msignal"]))))))), data["abs_maxbatch_msignal"] )))))) > -998, data["abs_maxbatch_msignal"], np.where(data["medianbatch_msignal"] > -998, data["medianbatch_msignal"], ((data["abs_maxbatch"]) + (data["abs_maxbatch_slices2_msignal"])) ) )))) +

                            0.100000*np.tanh(((((np.sin((np.cos((np.where(data["abs_avgbatch_msignal"] > -998, data["minbatch_slices2_msignal"], (((np.sin((data["maxbatch_msignal"]))) + (np.tanh((((((data["maxtominbatch_msignal"]) - (np.sin((data["maxbatch_msignal"]))))) * (data["rangebatch_slices2"]))))))/2.0) )))))) * 2.0)) - (np.cos((data["meanbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh(((np.sin(((((data["abs_maxbatch_slices2_msignal"]) + (((np.where(np.cos((np.where(((np.where(data["rangebatch_slices2"] > -998, ((np.where(data["abs_maxbatch_slices2"] > -998, data["rangebatch_slices2"], np.sin((data["abs_maxbatch_slices2"])) )) * 2.0), data["meanbatch_msignal"] )) * 2.0) > -998, data["meanbatch_slices2"], data["abs_maxbatch_slices2_msignal"] ))) > -998, data["rangebatch_slices2"], (3.0) )) * 2.0)))/2.0)))) * 2.0)) +

                            0.100000*np.tanh(((np.sin((((data["stdbatch_slices2_msignal"]) + (data["stdbatch_msignal"]))))) + (((data["medianbatch_slices2_msignal"]) - (((np.cos((np.where(data["stdbatch_slices2_msignal"] > -998, (0.50687325000762939), np.cos((np.where(((data["abs_avgbatch_slices2_msignal"]) * 2.0) > -998, data["signal_shift_+1_msignal"], data["rangebatch_slices2"] ))) )))) * 2.0)))))) +

                            0.100000*np.tanh(np.sin((np.sin((np.where(np.sin((data["maxbatch_slices2_msignal"])) <= -998, np.where(data["maxbatch_slices2_msignal"] > -998, data["stdbatch_msignal"], ((data["maxbatch_slices2_msignal"]) + ((-((data["abs_maxbatch_slices2_msignal"]))))) ), ((data["maxbatch_slices2_msignal"]) * 2.0) )))))) +

                            0.100000*np.tanh(np.where((14.41534996032714844) <= -998, np.sin((np.tanh((((((np.sin((data["maxtominbatch_slices2"]))) / 2.0)) * 2.0))))), np.sin((data["maxbatch_slices2_msignal"])) )) +

                            0.100000*np.tanh(((data["meanbatch_msignal"]) - (np.sin((((np.where(data["minbatch_msignal"] <= -998, np.tanh((((np.tanh((((((data["meanbatch_msignal"]) + (np.tanh((((np.cos((data["minbatch_slices2"]))) - ((9.08853149414062500)))))))) + (((data["medianbatch_slices2_msignal"]) + (data["signal_shift_+1_msignal"]))))))) * 2.0))), data["stdbatch_slices2_msignal"] )) - (data["mean_abs_chgbatch_msignal"]))))))) +

                            0.100000*np.tanh(np.where(((((data["stdbatch_slices2_msignal"]) / 2.0)) * 2.0) > -998, data["stdbatch_slices2_msignal"], np.where(data["maxbatch_slices2_msignal"] > -998, ((((data["stdbatch_slices2_msignal"]) / 2.0)) - (((data["maxbatch_slices2_msignal"]) + (np.sin((np.where(data["abs_avgbatch_msignal"] > -998, np.sin(((((data["signal"]) + (((data["abs_avgbatch_msignal"]) * 2.0)))/2.0))), data["meanbatch_slices2_msignal"] ))))))), (((-1.0)) + (data["stdbatch_slices2_msignal"])) ) )) +

                            0.100000*np.tanh(((data["abs_maxbatch_slices2"]) * (np.where(data["minbatch_msignal"] <= -998, (0.0), np.sin((((data["medianbatch_slices2_msignal"]) * (((((np.cos((data["maxbatch_slices2_msignal"]))) / 2.0)) - (np.where(data["maxtominbatch_slices2"] <= -998, np.where(data["stdbatch_slices2"] > -998, np.sin((data["medianbatch_slices2"])), data["mean_abs_chgbatch_slices2_msignal"] ), np.where(data["maxbatch_slices2_msignal"] > -998, data["stdbatch_slices2"], ((data["abs_avgbatch_msignal"]) * 2.0) ) ))))))) )))) +

                            0.100000*np.tanh(((np.cos((np.where(np.sin(((((np.cos((data["abs_maxbatch_slices2_msignal"]))) + (np.cos(((0.0)))))/2.0))) <= -998, np.cos((data["rangebatch_slices2"])), ((data["rangebatch_slices2"]) - (np.cos((((data["signal_shift_+1_msignal"]) * (data["maxbatch_msignal"])))))) )))) * 2.0)) +

                            0.100000*np.tanh(((((data["medianbatch_slices2_msignal"]) * (np.sin((np.where((2.84215521812438965) <= -998, (((1.0)) + (((data["minbatch_msignal"]) * 2.0))), ((data["minbatch_msignal"]) + (((((((np.cos(((9.16893291473388672)))) / 2.0)) / 2.0)) / 2.0))) )))))) * 2.0)) +

                            0.100000*np.tanh(((data["meanbatch_msignal"]) * (np.sin((np.where(((data["medianbatch_slices2_msignal"]) * 2.0) <= -998, data["abs_minbatch_slices2"], np.where(data["minbatch_msignal"] <= -998, data["meanbatch_msignal"], data["minbatch_msignal"] ) )))))) +

                            0.100000*np.tanh(((data["rangebatch_slices2"]) * (np.sin((np.where(data["mean_abs_chgbatch_slices2"] <= -998, data["minbatch"], (-(((((((data["rangebatch_slices2"]) - (data["abs_avgbatch_msignal"]))) + (np.sin((np.tanh((np.where(data["meanbatch_slices2_msignal"] <= -998, data["abs_avgbatch_msignal"], data["rangebatch_slices2"] )))))))/2.0)))) )))))) +

                            0.100000*np.tanh(data["stdbatch_slices2"]) +

                            0.100000*np.tanh(np.where(np.sin((data["stdbatch_msignal"])) <= -998, (-((data["medianbatch_msignal"]))), ((data["stdbatch_msignal"]) * 2.0) )) +

                            0.100000*np.tanh(np.cos((np.where(((data["abs_maxbatch_msignal"]) - (data["signal_shift_-1"])) <= -998, ((np.cos((((data["mean_abs_chgbatch_slices2"]) + (((data["meanbatch_slices2"]) - ((0.0)))))))) * (data["abs_avgbatch_msignal"])), data["minbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh(((((np.where(data["meanbatch_msignal"] > -998, data["meanbatch_msignal"], data["abs_minbatch_slices2_msignal"] )) + (data["maxtominbatch_slices2_msignal"]))) + (np.where(data["abs_minbatch_slices2_msignal"] > -998, data["meanbatch_msignal"], np.where(data["maxtominbatch_slices2"] > -998, ((data["stdbatch_msignal"]) - ((-((data["medianbatch_slices2_msignal"]))))), data["maxtominbatch_slices2_msignal"] ) )))) +

                            0.100000*np.tanh(((data["meanbatch_slices2_msignal"]) * (((np.sin((np.where(((data["medianbatch_msignal"]) * (((np.sin((np.where(data["signal_shift_+1_msignal"] > -998, data["minbatch_msignal"], data["rangebatch_slices2"] )))) * 2.0))) > -998, data["minbatch_msignal"], np.where((1.0) > -998, data["signal_shift_+1_msignal"], data["minbatch_msignal"] ) )))) * 2.0)))) +

                            0.100000*np.tanh((((((data["rangebatch_slices2"]) + (((data["signal_shift_+1"]) * 2.0)))/2.0)) * (((np.sin((((np.sin((np.tanh(((((data["medianbatch_slices2"]) + (np.where(np.sin((data["maxbatch_slices2_msignal"])) <= -998, np.sin((data["abs_minbatch_slices2"])), (3.0) )))/2.0)))))) + (np.where(data["medianbatch_slices2_msignal"] > -998, data["rangebatch_slices2"], (1.0) )))))) * 2.0)))) +

                            0.100000*np.tanh(((np.cos((data["minbatch_msignal"]))) * (np.where((((((data["maxbatch_slices2"]) - (np.cos((np.where(data["minbatch"] > -998, data["minbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] )))))) + (data["maxtominbatch"]))/2.0) > -998, data["abs_maxbatch"], ((data["rangebatch_slices2_msignal"]) * 2.0) )))) +

                            0.100000*np.tanh(((data["meanbatch_msignal"]) - (np.where(np.where(data["abs_maxbatch_msignal"] <= -998, np.where(data["abs_minbatch_msignal"] > -998, data["medianbatch_slices2_msignal"], np.tanh(((10.89451313018798828))) ), ((data["medianbatch_slices2_msignal"]) + (((data["medianbatch_slices2"]) * 2.0))) ) <= -998, data["meanbatch_msignal"], np.tanh(((10.89451313018798828))) )))) +

                            0.100000*np.tanh(((np.sin(((((-((((((data["abs_maxbatch_slices2"]) - ((-2.0)))) + (np.where(np.sin((data["rangebatch_slices2_msignal"])) > -998, data["rangebatch_slices2_msignal"], np.where(((data["abs_maxbatch_msignal"]) + ((((-((((data["rangebatch_slices2_msignal"]) * 2.0))))) / 2.0))) > -998, data["mean_abs_chgbatch_slices2_msignal"], data["minbatch_slices2"] ) ))))))) / 2.0)))) * 2.0)) +

                            0.100000*np.tanh(np.where(((((data["abs_minbatch_msignal"]) - (data["abs_minbatch_slices2"]))) - (data["minbatch_slices2_msignal"])) > -998, np.sin((data["medianbatch_slices2"])), np.tanh((((data["minbatch_slices2"]) * (np.where(data["abs_maxbatch"] <= -998, (-(((0.0)))), ((data["stdbatch_slices2_msignal"]) / 2.0) ))))) )) +

                            0.100000*np.tanh(np.cos((((data["medianbatch_msignal"]) + (np.where(data["minbatch_msignal"] <= -998, np.cos((data["medianbatch_msignal"])), ((((data["minbatch_msignal"]) + (np.cos((((data["medianbatch_slices2_msignal"]) - (data["abs_maxbatch_slices2_msignal"]))))))) / 2.0) )))))) +

                            0.100000*np.tanh((((-3.0)) * (np.where(data["stdbatch_slices2_msignal"] <= -998, (3.0), np.where(np.sin(((((-3.0)) * ((((-((data["meanbatch_slices2_msignal"])))) * 2.0))))) <= -998, ((np.cos((data["abs_minbatch_msignal"]))) - (np.sin((np.tanh((((np.sin((data["meanbatch_slices2_msignal"]))) + (data["maxbatch_slices2_msignal"])))))))), np.cos((data["meanbatch_slices2_msignal"])) ) )))) +

                            0.100000*np.tanh((((-2.0)) * (np.tanh((np.sin((np.cos(((((np.where(data["minbatch_msignal"] > -998, (-((data["minbatch_msignal"]))), data["minbatch_slices2_msignal"] )) + (np.tanh((((((np.cos((np.where(np.cos(((-2.0))) > -998, data["minbatch_msignal"], data["maxtominbatch_slices2"] )))) * 2.0)) * 2.0)))))/2.0)))))))))) +

                            0.100000*np.tanh(((data["signal_shift_+1"]) * (np.sin((np.where(data["signal_shift_+1"] > -998, data["rangebatch_slices2"], np.sin((np.where((((-3.0)) + (data["mean_abs_chgbatch_slices2"])) > -998, ((data["abs_minbatch_slices2_msignal"]) + (data["maxtominbatch_slices2"])), ((data["maxtominbatch_slices2_msignal"]) - (data["signal_shift_+1"])) ))) )))))) +

                            0.100000*np.tanh(((((data["medianbatch_msignal"]) * (np.sin((np.where((((((-2.0)) + (np.cos(((-2.0)))))) + (np.sin((np.sin((((((data["medianbatch_msignal"]) + (np.cos((data["mean_abs_chgbatch_slices2"]))))) + ((-2.0))))))))) <= -998, data["maxtominbatch_msignal"], np.where(data["maxbatch_slices2"] > -998, data["minbatch_msignal"], data["abs_maxbatch"] ) )))))) * 2.0)) +

                            0.100000*np.tanh(((np.sin((data["minbatch_msignal"]))) * (((((data["meanbatch_slices2_msignal"]) + (data["meanbatch_slices2_msignal"]))) + ((((data["minbatch_msignal"]) + ((-((((((data["minbatch_msignal"]) / 2.0)) / 2.0))))))/2.0)))))) +

                            0.100000*np.tanh(((data["medianbatch_slices2_msignal"]) * (np.sin((np.where((-((((np.sin((np.cos((data["medianbatch_slices2_msignal"]))))) + (data["medianbatch_slices2_msignal"]))))) <= -998, data["medianbatch_slices2_msignal"], data["minbatch_msignal"] )))))) +

                            0.100000*np.tanh(np.where(data["abs_minbatch_msignal"] <= -998, (-((((data["signal_shift_+1_msignal"]) + ((-((data["abs_minbatch_slices2_msignal"])))))))), ((np.sin((data["rangebatch_slices2"]))) * 2.0) )) +

                            0.100000*np.tanh(np.where(((data["rangebatch_slices2_msignal"]) - (((data["stdbatch_slices2"]) - (((data["abs_maxbatch_slices2_msignal"]) + ((((-((data["signal_shift_-1"])))) * (data["mean_abs_chgbatch_slices2"])))))))) > -998, (((3.0)) * (((((np.cos(((((-((data["abs_maxbatch_slices2_msignal"])))) * (np.tanh((data["abs_maxbatch_slices2_msignal"]))))))) / 2.0)) * 2.0))), ((data["abs_maxbatch_slices2_msignal"]) * 2.0) )) +

                            0.100000*np.tanh(np.tanh((np.sin((np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["maxbatch_msignal"], np.cos(((-((data["rangebatch_slices2"]))))) )))))) +

                            0.100000*np.tanh(((data["signal_shift_+1_msignal"]) * (np.where(((data["stdbatch_slices2_msignal"]) - (((data["minbatch_slices2_msignal"]) + (np.cos((data["rangebatch_slices2"])))))) > -998, data["rangebatch_slices2"], data["stdbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh(np.sin((np.where(data["stdbatch_slices2_msignal"] <= -998, ((data["maxtominbatch"]) - (np.sin(((3.0))))), np.where(data["maxbatch_slices2_msignal"] > -998, data["maxbatch_msignal"], data["maxbatch_slices2_msignal"] ) )))) +

                            0.100000*np.tanh(np.where(np.where(np.where(data["signal_shift_+1"] > -998, data["abs_maxbatch_slices2"], ((data["abs_minbatch_msignal"]) * 2.0) ) <= -998, np.where(np.where(np.tanh((np.tanh((data["medianbatch_slices2_msignal"])))) <= -998, data["meanbatch_slices2"], np.sin((np.sin((data["maxbatch_msignal"])))) ) > -998, data["minbatch"], (1.0) ), data["mean_abs_chgbatch_slices2_msignal"] ) <= -998, data["minbatch_msignal"], np.cos((data["minbatch_msignal"])) )) +

                            0.100000*np.tanh(np.where(((data["rangebatch_slices2"]) * (((data["abs_avgbatch_msignal"]) * (((data["signal_shift_-1_msignal"]) / 2.0))))) <= -998, (((((((np.tanh(((((data["mean_abs_chgbatch_slices2"]) + ((((-((data["signal_shift_-1_msignal"])))) * 2.0)))/2.0)))) / 2.0)) + (((data["meanbatch_msignal"]) * 2.0)))) + (data["rangebatch_slices2"]))/2.0), ((data["abs_avgbatch_msignal"]) * ((-((data["signal_shift_-1_msignal"]))))) )) +

                            0.100000*np.tanh(np.sin((((data["meanbatch_slices2_msignal"]) - ((((data["signal_shift_-1"]) + (np.where(data["maxtominbatch_slices2_msignal"] <= -998, ((((((data["medianbatch_slices2_msignal"]) - (((data["signal_shift_-1"]) / 2.0)))) - (np.where(data["meanbatch_slices2_msignal"] <= -998, data["maxtominbatch_slices2_msignal"], np.cos((data["maxtominbatch"])) )))) - ((1.0))), np.cos((data["stdbatch_msignal"])) )))/2.0)))))) +

                            0.100000*np.tanh(np.sin((np.where(data["rangebatch_slices2_msignal"] > -998, data["maxbatch_msignal"], np.where(data["maxbatch_slices2_msignal"] > -998, np.where((-((data["mean_abs_chgbatch_slices2"]))) <= -998, data["maxbatch_msignal"], (((2.0)) + (data["maxbatch_msignal"])) ), ((data["maxbatch_msignal"]) / 2.0) ) )))) +

                            0.100000*np.tanh(np.where(np.tanh((np.where(np.sin((data["meanbatch_slices2_msignal"])) > -998, data["meanbatch_msignal"], ((data["meanbatch_msignal"]) * (np.cos((data["stdbatch_msignal"])))) ))) > -998, ((data["abs_avgbatch_slices2_msignal"]) / 2.0), ((((data["signal"]) * (data["signal_shift_-1_msignal"]))) * (((data["minbatch_msignal"]) - (data["meanbatch_slices2_msignal"])))) )) +

                            0.100000*np.tanh(((data["signal_shift_-1"]) * (np.sin((np.sin((data["rangebatch_slices2"]))))))) +

                            0.100000*np.tanh(np.sin((np.sin((((data["maxbatch_slices2_msignal"]) - (np.sin((((np.where(data["maxbatch_slices2_msignal"] > -998, np.sin((np.sin((((data["maxbatch_slices2_msignal"]) - (data["abs_avgbatch_slices2_msignal"])))))), ((data["rangebatch_slices2"]) * 2.0) )) - ((0.0)))))))))))) +

                            0.100000*np.tanh(((np.sin((np.sin(((((data["meanbatch_slices2_msignal"]) + (np.where(((data["minbatch_msignal"]) * 2.0) > -998, data["rangebatch_slices2"], (-((((np.where(data["medianbatch_msignal"] <= -998, data["minbatch_msignal"], data["maxtominbatch_slices2_msignal"] )) * (((np.sin((data["maxtominbatch_msignal"]))) + (((data["abs_minbatch_slices2"]) / 2.0)))))))) )))/2.0)))))) * 2.0)) +

                            0.100000*np.tanh(np.cos((((data["maxbatch_slices2_msignal"]) + ((-((data["signal_shift_-1"])))))))) +

                            0.100000*np.tanh(np.where(np.sin((np.tanh((data["medianbatch_slices2_msignal"])))) > -998, data["stdbatch_slices2_msignal"], data["signal_shift_-1_msignal"] )) +

                            0.100000*np.tanh(((np.where(((np.sin((data["abs_maxbatch"]))) - (data["signal"])) > -998, ((data["abs_maxbatch_slices2_msignal"]) * 2.0), np.tanh((np.tanh((data["meanbatch_slices2"])))) )) + (np.tanh((data["abs_avgbatch_msignal"]))))) +

                            0.100000*np.tanh(np.cos((data["abs_maxbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(np.where(data["maxtominbatch_slices2"] <= -998, (1.0), np.sin((np.where(data["medianbatch_slices2"] <= -998, np.where(np.sin((np.where(np.cos((data["mean_abs_chgbatch_slices2"])) > -998, data["meanbatch_slices2"], data["maxtominbatch_slices2"] ))) > -998, np.cos((data["abs_maxbatch"])), ((np.sin((np.tanh((data["rangebatch_slices2"]))))) * (data["rangebatch_slices2"])) ), np.where(data["abs_maxbatch"] > -998, data["medianbatch_slices2"], data["medianbatch_slices2"] ) ))) )) +

                            0.100000*np.tanh(((np.cos((data["stdbatch_msignal"]))) * (((((data["abs_avgbatch_msignal"]) * 2.0)) - (np.where(((data["abs_avgbatch_slices2"]) - (data["meanbatch_msignal"])) <= -998, np.where(data["abs_avgbatch_msignal"] <= -998, data["meanbatch_slices2_msignal"], ((data["meanbatch_slices2"]) * (((data["meanbatch_slices2_msignal"]) * (np.tanh((np.cos((data["maxbatch_msignal"])))))))) ), ((data["signal_shift_+1"]) * 2.0) )))))) +

                            0.100000*np.tanh(((((np.sin((((data["abs_maxbatch_slices2_msignal"]) - (data["minbatch_msignal"]))))) * 2.0)) - (np.where(((data["minbatch_msignal"]) + (np.sin((((data["abs_maxbatch_slices2_msignal"]) - (data["minbatch_msignal"])))))) > -998, np.cos((data["meanbatch_msignal"])), ((((np.sin((((data["abs_maxbatch_slices2_msignal"]) - (data["minbatch_msignal"]))))) * 2.0)) * 2.0) )))) +

                            0.100000*np.tanh(np.sin((np.where(np.where(data["minbatch_slices2_msignal"] <= -998, data["abs_avgbatch_slices2_msignal"], data["minbatch_slices2"] ) > -998, data["abs_maxbatch_slices2_msignal"], (((np.tanh((data["rangebatch_slices2"]))) + (((data["abs_avgbatch_msignal"]) * 2.0)))/2.0) )))) +

                            0.100000*np.tanh(((data["meanbatch_msignal"]) + (data["abs_minbatch_slices2"]))) +

                            0.100000*np.tanh(((((data["abs_avgbatch_slices2_msignal"]) * (np.where((((1.80057454109191895)) * (np.sin((np.tanh((data["signal_shift_+1_msignal"])))))) > -998, data["minbatch_msignal"], np.sin((np.where(data["minbatch_slices2"] > -998, data["minbatch_msignal"], data["minbatch_slices2"] ))) )))) * (data["signal_shift_+1_msignal"]))) +

                            0.100000*np.tanh(np.where(((((data["signal_shift_-1"]) / 2.0)) / 2.0) > -998, np.cos((((data["minbatch_msignal"]) - (data["signal_shift_-1_msignal"])))), np.where(data["signal_shift_+1"] > -998, data["stdbatch_slices2"], ((np.cos((np.where((-3.0) <= -998, (((1.0)) * 2.0), data["minbatch_slices2_msignal"] )))) / 2.0) ) )) +

                            0.100000*np.tanh(((np.where(data["signal_shift_-1_msignal"] > -998, (-(((((0.0)) + (data["signal_shift_-1_msignal"]))))), ((data["rangebatch_msignal"]) * (np.tanh((np.where(data["maxbatch_msignal"] > -998, (-((((data["abs_maxbatch_slices2_msignal"]) * 2.0)))), (0.0) ))))) )) * (((np.cos((((data["abs_maxbatch_slices2_msignal"]) * 2.0)))) * 2.0)))) +

                            0.100000*np.tanh(np.sin((data["rangebatch_slices2"]))) +

                            0.100000*np.tanh(((((np.cos(((((3.0)) + ((((data["minbatch_msignal"]) + (np.tanh((np.cos(((((3.0)) + ((((np.sin((data["abs_avgbatch_slices2"]))) + (np.cos((np.cos((np.where(np.tanh((((data["signal_shift_-1_msignal"]) * 2.0))) <= -998, data["mean_abs_chgbatch_slices2"], data["minbatch_msignal"] )))))))/2.0)))))))))/2.0)))))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(((data["mean_abs_chgbatch_msignal"]) * (((np.where(data["signal_shift_-1"] > -998, np.sin((data["signal_shift_-1"])), (((np.where(np.cos((data["rangebatch_slices2"])) <= -998, np.cos((data["minbatch_slices2"])), data["signal_shift_-1_msignal"] )) + (data["mean_abs_chgbatch_msignal"]))/2.0) )) * 2.0)))) +

                            0.100000*np.tanh(np.where(np.sin(((-2.0))) > -998, data["signal_shift_-1_msignal"], np.cos(((((((((data["signal_shift_-1_msignal"]) - (np.sin(((5.0)))))) - (data["signal_shift_-1_msignal"]))) + ((7.22872352600097656)))/2.0))) )) +

                            0.100000*np.tanh((-((((data["meanbatch_slices2_msignal"]) * (np.where((((((3.0)) - (data["meanbatch_slices2_msignal"]))) / 2.0) <= -998, ((data["meanbatch_slices2_msignal"]) * (np.tanh((data["signal_shift_-1_msignal"])))), ((data["signal_shift_-1_msignal"]) * ((((3.0)) - (data["meanbatch_slices2_msignal"])))) ))))))) +

                            0.100000*np.tanh((((-((np.cos((((data["minbatch_msignal"]) * (((np.where(data["minbatch_msignal"] > -998, data["mean_abs_chgbatch_slices2"], np.where(np.tanh((data["mean_abs_chgbatch_slices2"])) <= -998, ((data["mean_abs_chgbatch_slices2"]) * 2.0), np.cos((((np.tanh((np.tanh((np.where(np.tanh((data["mean_abs_chgbatch_slices2"])) > -998, data["abs_maxbatch_msignal"], data["mean_abs_chgbatch_slices2"] )))))) / 2.0))) ) )) / 2.0))))))))) * 2.0)) +

                            0.100000*np.tanh(((np.sin((((data["stdbatch_slices2"]) * ((((data["maxbatch_msignal"]) + ((((data["stdbatch_slices2"]) + (((data["medianbatch_msignal"]) / 2.0)))/2.0)))/2.0)))))) * 2.0)) +

                            0.100000*np.tanh(np.cos((np.where(((data["abs_maxbatch"]) + (data["signal_shift_-1_msignal"])) > -998, data["abs_maxbatch"], np.where(np.cos((np.where(np.where((1.07999706268310547) > -998, data["mean_abs_chgbatch_msignal"], (3.0) ) > -998, data["maxbatch_slices2"], np.where(data["abs_maxbatch_slices2"] > -998, data["rangebatch_msignal"], data["maxbatch_slices2"] ) ))) > -998, (-1.0), data["signal_shift_+1_msignal"] ) )))) +

                            0.100000*np.tanh(np.sin((((np.cos((data["meanbatch_msignal"]))) + (np.where(data["abs_avgbatch_slices2_msignal"] > -998, ((data["abs_maxbatch"]) - (data["signal_shift_+1"])), np.sin((data["abs_maxbatch"])) )))))) +

                            0.100000*np.tanh(np.sin((np.where(((np.tanh(((((-((data["stdbatch_msignal"])))) / 2.0)))) / 2.0) > -998, np.where(data["abs_maxbatch_slices2"] <= -998, np.sin((data["abs_avgbatch_msignal"])), data["maxbatch_msignal"] ), ((data["maxtominbatch_msignal"]) * (np.where(np.sin((data["mean_abs_chgbatch_msignal"])) > -998, data["abs_maxbatch"], np.sin((np.cos(((0.0))))) ))) )))) +

                            0.100000*np.tanh(((((np.cos((((np.where(((data["rangebatch_slices2_msignal"]) - (((np.where(data["medianbatch_msignal"] <= -998, ((data["maxtominbatch_msignal"]) + ((((((data["signal_shift_+1_msignal"]) + (((data["medianbatch_slices2"]) * (data["medianbatch_slices2"]))))) + (data["maxtominbatch_msignal"]))/2.0))), data["rangebatch_msignal"] )) / 2.0))) <= -998, (1.0), data["rangebatch_msignal"] )) / 2.0)))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.sin((np.cos((data["abs_maxbatch_slices2"]))))) +

                            0.100000*np.tanh((((np.cos((((((((data["signal_shift_+1_msignal"]) - (data["maxbatch_msignal"]))) * 2.0)) + (((data["maxbatch_slices2_msignal"]) / 2.0)))))) + (data["mean_abs_chgbatch_slices2"]))/2.0)) +

                            0.100000*np.tanh(((data["maxtominbatch_slices2_msignal"]) + ((((7.0)) - (np.cos(((((7.0)) + (((((data["maxtominbatch_slices2_msignal"]) + (((data["maxtominbatch_slices2_msignal"]) + ((((5.39329433441162109)) - ((7.0)))))))) + (((((data["maxtominbatch_slices2_msignal"]) + (((data["minbatch"]) * 2.0)))) * 2.0)))))))))))) +

                            0.100000*np.tanh(((data["signal_shift_-1_msignal"]) * (((((np.where(np.sin(((((5.0)) / 2.0))) > -998, data["abs_avgbatch_slices2_msignal"], ((data["signal_shift_-1_msignal"]) * (((np.cos((data["abs_maxbatch_msignal"]))) * (data["minbatch_msignal"])))) )) * 2.0)) * (((((data["minbatch_msignal"]) * 2.0)) - ((5.0)))))))) +

                            0.100000*np.tanh(np.cos(((((-((((data["medianbatch_slices2"]) + (np.where(data["maxtominbatch_msignal"] > -998, (-((data["maxtominbatch_msignal"]))), ((data["maxbatch_slices2_msignal"]) + (((data["medianbatch_slices2"]) * 2.0))) ))))))) * 2.0)))) +

                            0.100000*np.tanh(np.sin((np.where(((data["minbatch_slices2"]) - (np.where(data["meanbatch_msignal"] > -998, data["maxbatch_slices2_msignal"], data["meanbatch_msignal"] ))) > -998, data["maxbatch_slices2_msignal"], np.sin((np.sin((data["maxbatch_slices2_msignal"])))) )))) +

                            0.100000*np.tanh(((data["abs_maxbatch_msignal"]) * (np.cos((np.where(data["minbatch_msignal"] > -998, data["minbatch_msignal"], np.sin(((-(((-((np.cos((np.where((((2.0)) - (data["signal_shift_+1"])) > -998, data["minbatch_msignal"], np.sin((((((-2.0)) + (data["rangebatch_msignal"]))/2.0))) ))))))))))) )))))) +

                            0.100000*np.tanh(np.where(((((data["abs_minbatch_slices2"]) * 2.0)) * ((((((data["abs_minbatch_slices2_msignal"]) + ((1.0)))/2.0)) * (((((4.0)) + (((np.cos((((data["rangebatch_slices2_msignal"]) - (np.sin((data["meanbatch_msignal"]))))))) / 2.0)))/2.0))))) > -998, data["medianbatch_slices2_msignal"], ((data["rangebatch_slices2_msignal"]) * (data["rangebatch_slices2_msignal"])) )) +

                            0.100000*np.tanh(np.where(np.cos(((3.0))) <= -998, np.cos((data["medianbatch_slices2"])), np.cos((np.where(np.sin((((np.cos((np.cos((np.sin((data["signal_shift_-1"]))))))) / 2.0))) <= -998, np.cos((np.where(data["meanbatch_slices2"] <= -998, np.cos((np.where(data["abs_avgbatch_slices2"] <= -998, (2.75408935546875000), data["rangebatch_slices2"] ))), data["maxbatch_slices2_msignal"] ))), data["rangebatch_slices2"] ))) )) +

                            0.100000*np.tanh((-((np.cos((np.where((-(((-1.0)))) > -998, data["abs_avgbatch_slices2_msignal"], np.cos((np.where((9.34756851196289062) > -998, data["abs_avgbatch_slices2_msignal"], ((((-2.0)) + ((-1.0)))/2.0) ))) ))))))) +

                            0.100000*np.tanh(np.where(np.cos((((((-1.0)) + (np.where(data["mean_abs_chgbatch_slices2"] <= -998, data["maxtominbatch"], np.where(data["signal_shift_-1"] > -998, data["meanbatch_slices2"], (0.0) ) )))/2.0))) > -998, np.sin((((data["abs_maxbatch"]) / 2.0))), np.tanh((np.cos((data["maxtominbatch_slices2_msignal"])))) )) +

                            0.100000*np.tanh(((np.where(((np.cos((data["abs_avgbatch_slices2"]))) + ((2.0))) > -998, data["signal_shift_+1"], (-((np.where(data["maxtominbatch"] > -998, data["meanbatch_slices2"], np.where(((np.where(data["signal_shift_-1"] > -998, data["medianbatch_slices2"], data["rangebatch_msignal"] )) * 2.0) <= -998, data["signal_shift_-1"], np.sin((((np.sin((np.sin(((2.0)))))) * 2.0))) ) )))) )) * 2.0)) +

                            0.100000*np.tanh((((np.tanh((data["signal_shift_-1_msignal"]))) + (np.sin(((((3.0)) * (((data["abs_minbatch_slices2_msignal"]) * (data["meanbatch_slices2"]))))))))/2.0)) +

                            0.100000*np.tanh(np.cos((((data["signal_shift_-1"]) - (np.where(((np.cos((data["stdbatch_slices2"]))) * ((((((-((np.tanh(((1.0))))))) * (((data["maxbatch_slices2"]) * 2.0)))) * (data["minbatch_slices2_msignal"])))) <= -998, data["maxbatch_msignal"], data["maxbatch_slices2"] )))))) +

                            0.100000*np.tanh((0.0)) +

                            0.100000*np.tanh((1.0)) +

                            0.100000*np.tanh(np.where(np.sin((np.cos((data["stdbatch_slices2_msignal"])))) <= -998, data["abs_minbatch_msignal"], np.cos((data["medianbatch_slices2_msignal"])) )) +

                            0.100000*np.tanh((-((np.sin((np.where(np.tanh((np.where((((data["meanbatch_msignal"]) + (((((((data["abs_maxbatch_slices2"]) - (np.tanh((((data["rangebatch_slices2"]) * 2.0)))))) * (data["abs_avgbatch_msignal"]))) / 2.0)))/2.0) <= -998, (2.0), (2.0) ))) <= -998, (2.0), (((data["abs_avgbatch_slices2_msignal"]) + (((data["abs_maxbatch_msignal"]) - (np.cos((data["abs_minbatch_msignal"]))))))/2.0) ))))))) +

                            0.100000*np.tanh(np.sin((np.where(np.where((((((data["rangebatch_slices2"]) + (data["medianbatch_slices2_msignal"]))/2.0)) * 2.0) > -998, data["abs_minbatch_slices2"], data["meanbatch_slices2_msignal"] ) > -998, data["abs_maxbatch_slices2_msignal"], ((data["abs_maxbatch_slices2"]) / 2.0) )))) +

                            0.100000*np.tanh(np.cos((np.where(((np.cos((data["maxtominbatch_slices2_msignal"]))) / 2.0) <= -998, np.cos(((((((data["abs_maxbatch_slices2"]) * (((data["meanbatch_slices2"]) - ((((data["abs_minbatch_msignal"]) + ((-((((np.sin((data["abs_maxbatch_msignal"]))) * 2.0))))))/2.0)))))) + (np.cos((data["minbatch_msignal"]))))/2.0))), data["minbatch_msignal"] )))) +

                            0.100000*np.tanh((((((((1.0)) * ((((((data["abs_maxbatch_msignal"]) * ((-1.0)))) + (np.tanh((((np.sin((((data["signal_shift_-1"]) * (np.cos((((data["medianbatch_slices2"]) * 2.0)))))))) * 2.0)))))/2.0)))) / 2.0)) / 2.0)) +

                            0.100000*np.tanh((((((np.where(np.sin(((-((np.tanh((np.sin((data["meanbatch_slices2"]))))))))) <= -998, data["medianbatch_slices2_msignal"], np.sin((np.where(np.cos((data["signal_shift_-1"])) > -998, ((data["medianbatch_slices2"]) * (((data["abs_maxbatch_slices2"]) - (np.cos((data["mean_abs_chgbatch_slices2"])))))), (0.0) ))) )) * 2.0)) + (data["meanbatch_msignal"]))/2.0)) +

                            0.100000*np.tanh(np.sin((np.tanh((data["rangebatch_msignal"]))))) +

                            0.100000*np.tanh(np.where(np.sin(((((-((data["maxbatch_msignal"])))) * ((((0.0)) + (np.tanh((np.cos((np.where(data["abs_avgbatch_slices2"] > -998, data["abs_maxbatch_slices2_msignal"], (-((np.cos((np.sin((data["signal_shift_+1"]))))))) ))))))))))) > -998, np.cos((np.cos((data["medianbatch_slices2_msignal"])))), np.sin((data["abs_maxbatch"])) )) +

                            0.100000*np.tanh(np.cos((((np.tanh(((-1.0)))) * (data["minbatch_slices2"]))))) +

                            0.100000*np.tanh(np.sin((((np.sin((((((data["rangebatch_slices2"]) * 2.0)) + (data["maxbatch_slices2_msignal"]))))) * 2.0)))) +

                            0.100000*np.tanh(np.sin((np.where(np.sin((np.sin(((-3.0))))) > -998, (((-3.0)) + (((data["signal_shift_+1"]) - (data["abs_avgbatch_slices2_msignal"])))), ((((data["meanbatch_msignal"]) + ((((((-3.0)) + (((data["signal_shift_+1"]) - (data["abs_avgbatch_slices2_msignal"]))))) - (((data["signal_shift_+1"]) / 2.0)))))) / 2.0) )))) +

                            0.100000*np.tanh(np.where((((np.where(data["mean_abs_chgbatch_slices2"] > -998, np.sin((data["abs_maxbatch"])), np.tanh(((-1.0))) )) + ((-1.0)))/2.0) > -998, np.tanh((data["meanbatch_msignal"])), data["medianbatch_msignal"] )) +

                            0.100000*np.tanh(np.cos((np.where((-((np.cos((np.sin((((np.cos((data["maxtominbatch_slices2_msignal"]))) / 2.0)))))))) <= -998, np.cos((np.sin((data["mean_abs_chgbatch_msignal"])))), ((data["mean_abs_chgbatch_msignal"]) - ((((np.where(data["signal_shift_-1"] > -998, (3.0), data["medianbatch_slices2"] )) + ((-((data["stdbatch_msignal"])))))/2.0))) )))) +

                            0.100000*np.tanh(np.cos((((data["maxtominbatch_slices2_msignal"]) - (((np.cos((np.sin((np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) * (np.cos(((((np.cos((((data["maxbatch_slices2_msignal"]) * (((((((10.0)) / 2.0)) + ((-((data["maxbatch_slices2"])))))/2.0)))))) + (data["maxtominbatch_slices2"]))/2.0)))))))))))) * ((((np.sin((data["abs_minbatch_slices2"]))) + (data["medianbatch_slices2"]))/2.0)))))))) +

                            0.100000*np.tanh(((((data["mean_abs_chgbatch_slices2"]) / 2.0)) + (((((np.sin((((np.tanh((((data["signal_shift_+1_msignal"]) / 2.0)))) * ((-((np.sin((((data["signal_shift_+1_msignal"]) + (np.cos((data["signal_shift_+1_msignal"])))))))))))))) / 2.0)) * (np.cos((data["maxtominbatch"]))))))) +

                            0.100000*np.tanh(((((data["maxtominbatch_slices2"]) / 2.0)) * (((np.tanh((data["abs_minbatch_slices2"]))) * 2.0)))) +

                            0.100000*np.tanh(np.cos((np.where(data["abs_minbatch_slices2_msignal"] <= -998, np.where(np.sin((data["abs_minbatch_slices2"])) <= -998, data["abs_minbatch_slices2"], data["rangebatch_slices2"] ), (-((data["rangebatch_slices2"]))) )))) +

                            0.100000*np.tanh(((np.cos((np.sin((np.where(data["signal_shift_+1_msignal"] > -998, ((np.tanh((np.where(data["signal_shift_+1_msignal"] > -998, data["maxbatch_slices2_msignal"], ((((data["maxbatch_msignal"]) * ((2.0)))) / 2.0) )))) / 2.0), data["maxbatch_slices2_msignal"] )))))) / 2.0)) +

                            0.100000*np.tanh((((data["maxtominbatch_msignal"]) + (((data["medianbatch_slices2"]) + ((((data["medianbatch_slices2"]) + (((data["medianbatch_slices2"]) + (((data["medianbatch_slices2"]) + ((((data["maxtominbatch_msignal"]) + ((((data["medianbatch_slices2"]) + (np.cos((((data["signal_shift_-1_msignal"]) + (((((data["signal_shift_+1_msignal"]) * (data["medianbatch_slices2_msignal"]))) / 2.0)))))))/2.0)))/2.0)))))))/2.0)))))/2.0)) +

                            0.100000*np.tanh(np.where((4.0) <= -998, data["abs_minbatch_slices2"], np.cos((data["stdbatch_slices2"])) )) +

                            0.100000*np.tanh(((data["abs_avgbatch_slices2_msignal"]) * (((np.where(data["medianbatch_msignal"] <= -998, data["signal_shift_-1"], data["meanbatch_msignal"] )) / 2.0)))) +

                            0.100000*np.tanh(np.sin((((data["meanbatch_msignal"]) + (data["meanbatch_msignal"]))))) +

                            0.100000*np.tanh(np.cos((np.where((((data["meanbatch_msignal"]) + ((0.0)))/2.0) > -998, np.where(((data["signal"]) - (data["medianbatch_slices2"])) > -998, data["abs_maxbatch_slices2_msignal"], (0.0) ), data["abs_avgbatch_slices2"] )))) +

                            0.100000*np.tanh(np.where((((np.cos((data["rangebatch_slices2"]))) + (np.tanh((data["rangebatch_slices2"]))))/2.0) > -998, np.cos((data["rangebatch_slices2"])), ((data["minbatch"]) * 2.0) )) +

                            0.100000*np.tanh(((np.where(np.where((1.0) > -998, data["medianbatch_msignal"], ((((data["rangebatch_msignal"]) * (np.cos((((data["rangebatch_slices2"]) / 2.0)))))) / 2.0) ) > -998, data["medianbatch_msignal"], ((data["maxtominbatch_slices2_msignal"]) / 2.0) )) * (np.cos((((data["rangebatch_slices2"]) / 2.0)))))) +

                            0.100000*np.tanh(np.where(np.where(((data["medianbatch_msignal"]) + (data["medianbatch_msignal"])) > -998, np.where(((np.tanh((((data["medianbatch_slices2_msignal"]) / 2.0)))) + (np.sin((((((((data["signal_shift_+1_msignal"]) / 2.0)) + (data["abs_minbatch_msignal"]))) / 2.0))))) > -998, data["signal_shift_+1_msignal"], (-((data["abs_maxbatch_slices2"]))) ), np.tanh((np.tanh((data["medianbatch_msignal"])))) ) > -998, data["signal_shift_+1_msignal"], data["signal_shift_+1_msignal"] )) +

                            0.100000*np.tanh(np.where((-1.0) > -998, np.cos(((((10.16103267669677734)) * (data["stdbatch_slices2"])))), np.cos((((data["maxbatch_slices2"]) * (np.cos(((((10.16103267669677734)) * (data["stdbatch_slices2"])))))))) )) +

                            0.100000*np.tanh((((data["meanbatch_msignal"]) + ((2.0)))/2.0)) +

                            0.100000*np.tanh(((np.cos((np.cos((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))))))) + (np.cos((data["abs_maxbatch_slices2"]))))) +

                            0.100000*np.tanh(np.sin(((0.0)))) +

                            0.100000*np.tanh(np.cos((((((((11.63140583038330078)) + (data["stdbatch_slices2"]))/2.0)) - (np.where(np.sin((np.where((3.0) > -998, data["abs_maxbatch_msignal"], data["maxtominbatch"] ))) <= -998, data["abs_maxbatch"], data["abs_maxbatch_msignal"] )))))) +

                            0.100000*np.tanh(np.cos((np.where(data["abs_maxbatch"] > -998, data["minbatch_msignal"], (((((data["medianbatch_slices2"]) / 2.0)) + (np.where((0.0) <= -998, data["abs_minbatch_msignal"], np.cos((((data["medianbatch_slices2_msignal"]) / 2.0))) )))/2.0) )))) +

                            0.100000*np.tanh(np.tanh(((((((data["abs_minbatch_slices2"]) - (np.where(((data["medianbatch_msignal"]) * (np.sin((np.where(data["stdbatch_slices2"] > -998, data["abs_minbatch_slices2"], (-((data["medianbatch_slices2"]))) ))))) > -998, np.sin((((data["medianbatch_slices2"]) * 2.0))), np.sin((np.tanh((data["meanbatch_msignal"])))) )))) + (np.sin((np.tanh((data["signal"]))))))/2.0)))) +

                            0.100000*np.tanh(((((np.where((((data["mean_abs_chgbatch_slices2_msignal"]) + (np.sin((data["stdbatch_slices2_msignal"]))))/2.0) <= -998, data["mean_abs_chgbatch_msignal"], np.sin((data["abs_maxbatch"])) )) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.where((((3.0)) / 2.0) <= -998, np.where(((np.tanh((data["mean_abs_chgbatch_slices2_msignal"]))) - (np.sin(((-((np.tanh((data["signal_shift_-1"]))))))))) <= -998, data["abs_maxbatch_msignal"], data["meanbatch_slices2"] ), np.sin((np.where(data["signal_shift_-1"] > -998, data["abs_maxbatch_msignal"], np.tanh((data["meanbatch_slices2"])) ))) )) +

                            0.100000*np.tanh((-((np.cos((np.where(((data["mean_abs_chgbatch_msignal"]) - (data["stdbatch_slices2"])) > -998, ((data["maxtominbatch"]) * (data["minbatch_msignal"])), np.tanh((data["minbatch_msignal"])) ))))))) +

                            0.100000*np.tanh(np.where((-((((data["medianbatch_slices2_msignal"]) / 2.0)))) > -998, np.sin((((((data["minbatch"]) - (np.sin((np.where(data["signal_shift_-1_msignal"] > -998, data["stdbatch_slices2_msignal"], data["medianbatch_slices2"] )))))) * 2.0))), (-3.0) )) +

                            0.100000*np.tanh(np.cos((((np.sin(((-((np.where(np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, np.cos((data["meanbatch_msignal"])), data["abs_maxbatch_slices2_msignal"] ) <= -998, np.where(np.cos((data["abs_minbatch_msignal"])) <= -998, ((np.tanh((data["mean_abs_chgbatch_slices2_msignal"]))) / 2.0), (-3.0) ), data["abs_minbatch_msignal"] ))))))) * 2.0)))) +

                            0.100000*np.tanh(np.where(data["stdbatch_slices2"] <= -998, ((data["meanbatch_slices2_msignal"]) - (np.tanh((np.sin((np.tanh(((((1.0)) / 2.0))))))))), ((data["mean_abs_chgbatch_slices2_msignal"]) * (np.sin((((data["meanbatch_msignal"]) + (data["minbatch_slices2_msignal"])))))) )) +

                            0.099609*np.tanh(np.where(data["minbatch_msignal"] <= -998, data["abs_minbatch_slices2_msignal"], np.sin((np.where(np.where(data["abs_minbatch_msignal"] > -998, np.tanh((data["medianbatch_slices2_msignal"])), np.where(np.sin((np.where((2.0) > -998, data["signal_shift_+1_msignal"], (3.0) ))) > -998, data["abs_maxbatch_msignal"], (-((((data["abs_maxbatch_msignal"]) * (data["rangebatch_slices2"]))))) ) ) > -998, data["abs_maxbatch_msignal"], data["signal_shift_+1_msignal"] ))) )) +

                            0.100000*np.tanh(((data["medianbatch_msignal"]) / 2.0)) +

                            0.100000*np.tanh(np.where(((data["rangebatch_slices2_msignal"]) / 2.0) > -998, data["abs_minbatch_slices2"], np.sin((((((data["maxtominbatch"]) / 2.0)) * (data["abs_minbatch_slices2"])))) )) +

                            0.100000*np.tanh(np.where(data["stdbatch_msignal"] > -998, np.cos((data["mean_abs_chgbatch_msignal"])), np.where((0.0) > -998, np.where(data["mean_abs_chgbatch_msignal"] <= -998, np.cos((data["abs_maxbatch_slices2"])), data["signal"] ), np.where(data["abs_minbatch_msignal"] <= -998, data["mean_abs_chgbatch_msignal"], (-((np.where(data["medianbatch_slices2_msignal"] > -998, (-1.0), (((((((np.tanh((data["maxtominbatch_slices2"]))) * 2.0)) / 2.0)) + (data["medianbatch_msignal"]))/2.0) )))) ) ) )) +

                            0.100000*np.tanh(np.tanh((np.sin((data["medianbatch_slices2"]))))) +

                            0.100000*np.tanh(np.where((9.32021331787109375) > -998, data["signal_shift_-1"], data["signal_shift_-1"] )) +

                            0.100000*np.tanh((-((np.where((-((((np.sin((np.tanh((data["rangebatch_msignal"]))))) * (np.where((0.0) > -998, data["signal_shift_-1_msignal"], data["abs_maxbatch_slices2_msignal"] )))))) <= -998, ((np.cos(((-(((-((np.tanh((data["meanbatch_msignal"]))))))))))) - ((-3.0))), np.tanh(((0.0))) ))))) +

                            0.100000*np.tanh(np.tanh((np.cos(((((data["medianbatch_slices2"]) + (((np.cos((data["medianbatch_slices2"]))) / 2.0)))/2.0)))))) +

                            0.100000*np.tanh((3.0)) +

                            0.100000*np.tanh(((np.sin((np.tanh((np.where(((((data["maxtominbatch_slices2_msignal"]) * (np.tanh((data["signal_shift_+1_msignal"]))))) * 2.0) > -998, data["signal_shift_+1_msignal"], (-((data["minbatch_slices2_msignal"]))) )))))) * 2.0)) +

                            0.100000*np.tanh(np.cos((data["abs_maxbatch"]))) +

                            0.089932*np.tanh(np.sin((np.where(data["maxtominbatch"] <= -998, np.where(data["medianbatch_slices2_msignal"] > -998, (((np.tanh((data["meanbatch_msignal"]))) + (data["signal_shift_+1_msignal"]))/2.0), data["mean_abs_chgbatch_msignal"] ), np.tanh((np.sin((np.where(data["medianbatch_slices2_msignal"] <= -998, data["stdbatch_slices2_msignal"], (((data["medianbatch_msignal"]) + ((-1.0)))/2.0) ))))) )))) +

                            0.100000*np.tanh(np.where(np.tanh((((((-((data["abs_avgbatch_slices2"])))) + (data["meanbatch_msignal"]))/2.0))) <= -998, np.cos((((data["abs_avgbatch_msignal"]) + (data["stdbatch_msignal"])))), data["abs_avgbatch_slices2"] )) +

                            0.098436*np.tanh(((((((((data["abs_minbatch_slices2"]) + (data["abs_avgbatch_slices2_msignal"]))) / 2.0)) + (((np.cos((data["abs_maxbatch_msignal"]))) / 2.0)))) / 2.0)) +

                            0.100000*np.tanh(np.sin((np.tanh((np.sin((data["maxtominbatch_slices2_msignal"]))))))) +

                            0.100000*np.tanh(np.sin(((((data["minbatch_msignal"]) + ((((((np.tanh((((((data["minbatch_msignal"]) * 2.0)) + (data["abs_minbatch_slices2_msignal"]))))) * 2.0)) + ((((((np.tanh((((data["minbatch_msignal"]) + ((((-1.0)) * (data["meanbatch_slices2_msignal"]))))))) * 2.0)) + (data["mean_abs_chgbatch_slices2"]))/2.0)))/2.0)))/2.0)))) +

                            0.100000*np.tanh(np.sin((data["abs_avgbatch_slices2_msignal"]))) +

                            0.100000*np.tanh((((data["abs_minbatch_slices2"]) + (data["rangebatch_slices2"]))/2.0)) +

                            0.100000*np.tanh(np.sin((((np.tanh((np.tanh((np.tanh((np.tanh((np.tanh((((np.cos((((np.where(data["rangebatch_msignal"] > -998, np.sin((np.sin(((2.0))))), np.tanh((np.cos(((((2.0)) / 2.0))))) )) * ((0.0)))))) / 2.0)))))))))))) / 2.0)))) +

                            0.100000*np.tanh(np.cos((np.tanh((((((np.tanh((np.cos((np.sin((((data["medianbatch_slices2_msignal"]) * 2.0)))))))) / 2.0)) * 2.0)))))) +

                            0.100000*np.tanh((((data["medianbatch_msignal"]) + (np.cos((np.sin((np.tanh((np.sin((((((data["abs_minbatch_slices2"]) + ((((data["abs_minbatch_msignal"]) + (data["medianbatch_msignal"]))/2.0)))) / 2.0)))))))))))/2.0)) +

                            0.093744*np.tanh((((((np.tanh((data["signal"]))) - ((0.0)))) + (((np.cos((np.cos((data["abs_maxbatch_slices2_msignal"]))))) / 2.0)))/2.0)) +

                            0.100000*np.tanh(np.cos((np.where(np.where(data["mean_abs_chgbatch_msignal"] > -998, data["abs_maxbatch_slices2"], np.cos((np.sin((data["medianbatch_msignal"])))) ) <= -998, np.cos((data["mean_abs_chgbatch_msignal"])), np.where(data["abs_maxbatch_slices2"] <= -998, (((np.cos((data["maxtominbatch_slices2"]))) + (data["mean_abs_chgbatch_slices2"]))/2.0), data["abs_minbatch_slices2_msignal"] ) )))) +

                            0.099707*np.tanh(np.cos((((((data["abs_maxbatch_msignal"]) - (np.sin((np.where(data["maxbatch_slices2"] > -998, np.sin((data["abs_minbatch_slices2"])), ((np.tanh(((3.0)))) - ((3.0))) )))))) / 2.0)))) +

                            0.100000*np.tanh((((data["abs_maxbatch"]) + (data["abs_maxbatch"]))/2.0)) +

                            0.100000*np.tanh(np.sin((((np.where(((np.cos((np.sin((((data["maxtominbatch_msignal"]) - ((2.0)))))))) / 2.0) > -998, data["stdbatch_msignal"], ((data["medianbatch_slices2"]) * 2.0) )) * (np.where(data["mean_abs_chgbatch_slices2"] > -998, data["medianbatch_slices2"], np.sin((np.tanh((np.cos((((np.cos((np.tanh(((1.0)))))) / 2.0))))))) )))))) +

                            0.100000*np.tanh(((np.where(((data["minbatch_slices2_msignal"]) + (np.cos((np.cos((data["minbatch_msignal"])))))) <= -998, data["meanbatch_msignal"], ((np.cos((data["abs_maxbatch_slices2_msignal"]))) * 2.0) )) + ((((data["medianbatch_slices2_msignal"]) + (data["medianbatch_msignal"]))/2.0)))))  

    

    def GP_class_6(self,data):

        return self.Output( -3.281070 +

                            0.100000*np.tanh(np.where((-((np.where(data["maxbatch_msignal"] <= -998, np.tanh((np.where(np.cos((((data["meanbatch_msignal"]) - (data["medianbatch_msignal"])))) <= -998, np.cos((data["meanbatch_msignal"])), np.cos((data["medianbatch_slices2"])) ))), ((np.sin((data["meanbatch_msignal"]))) + (data["medianbatch_msignal"])) )))) <= -998, (-1.0), data["signal_shift_+1"] )) +

                            0.100000*np.tanh(np.where((-1.0) > -998, data["meanbatch_msignal"], ((data["medianbatch_msignal"]) / 2.0) )) +

                            0.100000*np.tanh(np.where((((data["maxtominbatch"]) + (data["maxtominbatch"]))/2.0) <= -998, ((data["abs_avgbatch_msignal"]) / 2.0), data["signal"] )) +

                            0.100000*np.tanh(((np.sin((data["meanbatch_slices2_msignal"]))) * 2.0)) +

                            0.100000*np.tanh(((data["medianbatch_slices2"]) * (np.where(np.where(data["abs_avgbatch_slices2_msignal"] <= -998, np.tanh((data["maxbatch_msignal"])), np.tanh(((1.0))) ) <= -998, (((data["mean_abs_chgbatch_slices2"]) + ((2.0)))/2.0), np.where(((data["abs_avgbatch_slices2_msignal"]) * ((2.0))) > -998, data["abs_avgbatch_slices2_msignal"], (((data["mean_abs_chgbatch_slices2"]) + ((2.0)))/2.0) ) )))) +

                            0.100000*np.tanh(((data["medianbatch_msignal"]) + (((((((data["medianbatch_msignal"]) * 2.0)) * 2.0)) * 2.0)))) +

                            0.100000*np.tanh(np.where(np.cos(((((np.tanh((((data["abs_maxbatch"]) / 2.0)))) + (data["abs_maxbatch"]))/2.0))) <= -998, data["meanbatch_msignal"], ((data["abs_maxbatch"]) - ((6.31022024154663086))) )) +

                            0.100000*np.tanh(np.cos((np.where(data["abs_maxbatch_slices2"] > -998, data["maxbatch_slices2_msignal"], (((((data["medianbatch_msignal"]) + (np.where((((10.0)) * (data["maxbatch_slices2_msignal"])) > -998, (1.0), data["maxbatch_slices2_msignal"] )))) + (np.where((2.0) > -998, (((0.10073664039373398)) * ((2.0))), data["signal"] )))/2.0) )))) +

                            0.100000*np.tanh(data["medianbatch_slices2"]) +

                            0.100000*np.tanh(np.where(data["meanbatch_slices2"] > -998, data["signal_shift_-1"], ((((np.where(data["abs_maxbatch"] > -998, (((((data["signal_shift_-1"]) / 2.0)) + (np.tanh((np.tanh((data["abs_maxbatch_slices2_msignal"]))))))/2.0), ((((4.89288330078125000)) + (np.where(data["signal_shift_-1"] > -998, data["maxbatch_slices2"], (((((data["abs_maxbatch_msignal"]) + (np.tanh((data["maxbatch_slices2"]))))/2.0)) / 2.0) )))/2.0) )) / 2.0)) * (data["signal_shift_-1"])) )) +

                            0.100000*np.tanh((-2.0)) +

                            0.100000*np.tanh(((data["meanbatch_slices2_msignal"]) * 2.0)) +

                            0.100000*np.tanh(np.where(((data["abs_minbatch_slices2"]) * (data["medianbatch_msignal"])) <= -998, data["maxbatch_slices2_msignal"], np.where(data["meanbatch_slices2"] <= -998, ((data["maxbatch_slices2_msignal"]) - (((data["signal"]) + (data["maxtominbatch_msignal"])))), np.where(data["maxbatch_slices2_msignal"] > -998, ((((np.tanh((data["medianbatch_msignal"]))) * (data["medianbatch_slices2"]))) * 2.0), data["meanbatch_slices2"] ) ) )) +

                            0.100000*np.tanh(np.where(((data["meanbatch_msignal"]) * 2.0) > -998, data["medianbatch_msignal"], ((np.where(((data["meanbatch_msignal"]) / 2.0) > -998, np.cos((np.where(data["meanbatch_msignal"] > -998, data["medianbatch_msignal"], data["maxbatch_msignal"] ))), data["stdbatch_msignal"] )) * 2.0) )) +

                            0.100000*np.tanh(np.where(((data["stdbatch_slices2"]) / 2.0) <= -998, data["signal_shift_+1_msignal"], data["meanbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(((((data["signal"]) * 2.0)) * (data["medianbatch_msignal"]))) +

                            0.100000*np.tanh(np.where(np.tanh((data["maxbatch_slices2"])) <= -998, ((((data["signal_shift_-1"]) * (data["abs_avgbatch_msignal"]))) + (data["meanbatch_slices2_msignal"])), ((data["abs_avgbatch_msignal"]) * (np.sin((data["medianbatch_msignal"])))) )) +

                            0.100000*np.tanh(np.where((4.81611108779907227) > -998, ((np.where(np.cos((((data["medianbatch_slices2_msignal"]) - (np.sin((data["meanbatch_slices2_msignal"])))))) > -998, data["medianbatch_msignal"], data["meanbatch_slices2"] )) * 2.0), ((np.tanh((((data["rangebatch_msignal"]) + (data["medianbatch_msignal"]))))) / 2.0) )) +

                            0.100000*np.tanh(np.where((-3.0) > -998, np.where(np.sin((data["rangebatch_slices2_msignal"])) > -998, data["meanbatch_msignal"], (3.0) ), ((data["abs_maxbatch"]) * 2.0) )) +

                            0.100000*np.tanh(np.where(np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, np.sin((data["maxbatch_msignal"])), data["meanbatch_slices2_msignal"] ) > -998, ((np.where(data["rangebatch_msignal"] > -998, data["meanbatch_slices2_msignal"], np.cos(((0.0))) )) * (((((data["maxbatch_msignal"]) * 2.0)) * 2.0))), (3.0) )) +

                            0.100000*np.tanh(np.cos((np.where(data["signal"] > -998, data["maxbatch_msignal"], data["maxbatch_msignal"] )))) +

                            0.100000*np.tanh(np.where((((-2.0)) + (((data["medianbatch_slices2_msignal"]) * (((data["medianbatch_slices2_msignal"]) * (data["signal_shift_+1"])))))) <= -998, ((((-2.0)) + (((((((data["meanbatch_slices2_msignal"]) * 2.0)) * 2.0)) * (data["meanbatch_msignal"]))))/2.0), (((-2.0)) + (((data["medianbatch_slices2_msignal"]) * (((data["maxbatch_slices2_msignal"]) + (np.cos((data["maxbatch_slices2_msignal"])))))))) )) +

                            0.100000*np.tanh(np.where(((((data["maxbatch_slices2"]) - ((((4.0)) * 2.0)))) - (((data["mean_abs_chgbatch_slices2"]) + (np.where((-(((6.57437086105346680)))) > -998, np.sin((data["meanbatch_slices2_msignal"])), (((-3.0)) * 2.0) ))))) > -998, data["meanbatch_slices2_msignal"], data["mean_abs_chgbatch_slices2"] )) +

                            0.100000*np.tanh(((np.cos((np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["maxbatch_slices2_msignal"], np.where(data["maxbatch_slices2_msignal"] > -998, data["minbatch_slices2_msignal"], ((np.sin((np.where(data["maxbatch_slices2_msignal"] > -998, np.cos((data["meanbatch_slices2"])), ((data["medianbatch_slices2"]) / 2.0) )))) * 2.0) ) )))) * 2.0)) +

                            0.100000*np.tanh(np.where((2.0) <= -998, ((data["mean_abs_chgbatch_slices2"]) * ((((data["abs_maxbatch_slices2_msignal"]) + (data["stdbatch_slices2"]))/2.0))), ((np.where((-((data["abs_avgbatch_msignal"]))) <= -998, np.tanh((data["mean_abs_chgbatch_slices2_msignal"])), data["mean_abs_chgbatch_slices2_msignal"] )) / 2.0) )) +

                            0.100000*np.tanh(np.sin((np.where(data["signal_shift_+1_msignal"] > -998, np.cos((data["abs_maxbatch_slices2_msignal"])), np.cos((np.where(((np.sin((data["medianbatch_slices2_msignal"]))) * (data["medianbatch_slices2_msignal"])) > -998, np.tanh((np.where((3.0) > -998, data["abs_maxbatch_slices2_msignal"], np.cos((np.where(data["abs_minbatch_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], ((data["maxbatch_msignal"]) / 2.0) ))) ))), data["signal_shift_+1"] ))) )))) +

                            0.100000*np.tanh(((np.cos((np.where(np.cos(((-((np.where((-((data["minbatch_msignal"]))) > -998, data["minbatch_msignal"], np.cos((((data["meanbatch_slices2_msignal"]) - (data["signal"])))) )))))) > -998, data["minbatch_msignal"], np.cos((data["abs_minbatch_slices2"])) )))) * 2.0)) +

                            0.100000*np.tanh(((data["signal_shift_-1"]) * (((np.sin((data["medianbatch_slices2_msignal"]))) + (((data["abs_maxbatch_msignal"]) + ((((-3.0)) * 2.0)))))))) +

                            0.100000*np.tanh(np.where(data["signal"] <= -998, (((data["signal"]) + (data["signal"]))/2.0), data["meanbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(((data["mean_abs_chgbatch_slices2"]) * (((data["maxbatch_msignal"]) * (np.sin(((-(((((data["maxbatch_msignal"]) + (((np.tanh((data["maxbatch_msignal"]))) / 2.0)))/2.0))))))))))) +

                            0.100000*np.tanh(((np.where(data["abs_maxbatch_msignal"] <= -998, data["stdbatch_slices2_msignal"], data["abs_maxbatch_msignal"] )) * (np.cos((data["maxbatch_msignal"]))))) +

                            0.100000*np.tanh(((np.cos(((-((np.where(((np.sin((((data["minbatch_msignal"]) / 2.0)))) * 2.0) <= -998, np.tanh((np.cos((np.where((-((data["abs_maxbatch_msignal"]))) <= -998, data["maxtominbatch_msignal"], (-((np.where(((np.cos((data["abs_maxbatch_msignal"]))) * 2.0) <= -998, data["maxtominbatch"], data["maxbatch_msignal"] )))) ))))), data["minbatch_msignal"] ))))))) * 2.0)) +

                            0.100000*np.tanh(((np.cos((np.where((0.0) <= -998, np.sin((data["minbatch"])), np.where(np.where(data["maxbatch_msignal"] <= -998, (((13.52190876007080078)) * 2.0), data["abs_minbatch_slices2"] ) > -998, np.where(data["maxbatch_msignal"] > -998, data["minbatch_msignal"], ((data["meanbatch_msignal"]) - ((4.0))) ), np.tanh(((((-1.0)) - (data["meanbatch_msignal"])))) ) )))) * 2.0)) +

                            0.100000*np.tanh(data["meanbatch_slices2_msignal"]) +

                            0.100000*np.tanh(((np.cos((((np.where((-(((-2.0)))) > -998, (-((data["abs_maxbatch_msignal"]))), ((data["meanbatch_msignal"]) + (data["minbatch_msignal"])) )) * 2.0)))) * 2.0)) +

                            0.100000*np.tanh(np.where(data["rangebatch_slices2"] <= -998, data["abs_avgbatch_msignal"], ((np.cos(((((3.0)) * (np.where(np.tanh((np.cos((np.where((3.0) > -998, data["abs_avgbatch_slices2_msignal"], data["minbatch_msignal"] ))))) > -998, data["stdbatch_slices2"], ((((data["meanbatch_msignal"]) * (data["stdbatch_slices2"]))) - (data["abs_maxbatch_msignal"])) )))))) * 2.0) )) +

                            0.100000*np.tanh(np.where(np.where((-1.0) <= -998, (0.0), data["maxbatch_msignal"] ) > -998, ((np.cos((data["maxbatch_msignal"]))) * 2.0), (((np.cos((data["maxbatch_msignal"]))) + (np.where(np.cos(((3.0))) > -998, ((np.cos((data["maxbatch_msignal"]))) * 2.0), data["maxbatch_msignal"] )))/2.0) )) +

                            0.100000*np.tanh(np.cos((np.where((((data["signal"]) + (np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], ((((-3.0)) + (data["maxtominbatch_slices2"]))/2.0) )))/2.0) > -998, data["abs_maxbatch_slices2_msignal"], np.tanh((np.cos((data["maxbatch_slices2"])))) )))) +

                            0.100000*np.tanh(((np.cos((np.where(data["meanbatch_slices2_msignal"] > -998, data["maxbatch_slices2_msignal"], np.where(((data["meanbatch_msignal"]) * (data["stdbatch_slices2"])) <= -998, np.where(data["medianbatch_slices2_msignal"] <= -998, data["abs_minbatch_slices2"], data["medianbatch_slices2_msignal"] ), (-((np.cos((data["meanbatch_msignal"]))))) ) )))) * 2.0)) +

                            0.100000*np.tanh(((np.cos((data["minbatch_msignal"]))) + (np.cos((np.where(((data["mean_abs_chgbatch_slices2_msignal"]) * (np.sin((data["abs_maxbatch"])))) <= -998, data["medianbatch_slices2"], data["minbatch_msignal"] )))))) +

                            0.100000*np.tanh(((np.cos((((data["maxbatch_msignal"]) - (np.cos((np.where((((np.where(data["signal"] > -998, data["meanbatch_slices2_msignal"], (2.0) )) + ((14.88569545745849609)))/2.0) <= -998, (((data["medianbatch_msignal"]) + (data["maxtominbatch_slices2"]))/2.0), data["meanbatch_slices2_msignal"] )))))))) * 2.0)) +

                            0.100000*np.tanh(((np.where((1.0) > -998, data["maxbatch_slices2"], data["abs_minbatch_msignal"] )) * (np.where(((data["medianbatch_msignal"]) / 2.0) <= -998, data["maxbatch_slices2_msignal"], np.cos((np.where(data["medianbatch_slices2"] > -998, data["minbatch_msignal"], data["maxbatch_slices2"] ))) )))) +

                            0.100000*np.tanh(np.cos((np.where(np.cos((np.cos((np.cos((((np.where((0.0) > -998, np.cos((np.cos((data["minbatch_slices2_msignal"])))), ((data["medianbatch_slices2_msignal"]) + ((0.0))) )) * 2.0))))))) > -998, data["minbatch_msignal"], (((-(((-1.0))))) + ((0.0))) )))) +

                            0.100000*np.tanh(np.where(np.where(np.cos((np.tanh((data["minbatch_msignal"])))) > -998, (3.0), np.cos((np.tanh(((((((data["signal_shift_-1"]) * (data["medianbatch_slices2_msignal"]))) + (np.tanh((np.cos((np.sin((data["minbatch_msignal"]))))))))/2.0))))) ) <= -998, (2.0), data["medianbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(np.where(data["abs_maxbatch_slices2_msignal"] <= -998, np.where(np.where(data["abs_maxbatch_slices2_msignal"] <= -998, np.sin((((np.sin((data["abs_maxbatch_slices2_msignal"]))) + (data["abs_maxbatch_slices2_msignal"])))), np.cos((data["maxbatch_slices2_msignal"])) ) > -998, data["minbatch_slices2_msignal"], (-(((((data["abs_maxbatch_slices2_msignal"]) + ((-1.0)))/2.0)))) ), np.cos((data["abs_maxbatch_slices2_msignal"])) )) +

                            0.100000*np.tanh(np.sin((np.where(data["maxtominbatch_msignal"] > -998, data["minbatch_slices2_msignal"], np.cos(((1.0))) )))) +

                            0.100000*np.tanh(((((data["maxbatch_msignal"]) * (data["stdbatch_msignal"]))) * ((((-3.0)) * 2.0)))) +

                            0.100000*np.tanh(np.where(data["abs_maxbatch_slices2_msignal"] > -998, np.cos((data["abs_maxbatch_slices2_msignal"])), np.where(((np.sin((data["abs_maxbatch_slices2_msignal"]))) / 2.0) > -998, np.where(((np.tanh((data["abs_maxbatch_slices2_msignal"]))) - (data["meanbatch_slices2_msignal"])) > -998, np.cos((data["abs_maxbatch_slices2_msignal"])), np.where(data["mean_abs_chgbatch_slices2"] > -998, np.cos((data["abs_maxbatch_slices2_msignal"])), data["abs_maxbatch_slices2_msignal"] ) ), data["meanbatch_slices2_msignal"] ) )) +

                            0.100000*np.tanh(np.cos((np.where(np.where(((np.where((2.0) <= -998, (((data["signal_shift_+1"]) + ((-((np.where(data["maxtominbatch_msignal"] <= -998, data["signal_shift_+1"], np.sin(((-((data["abs_maxbatch_msignal"]))))) ))))))/2.0), data["abs_maxbatch_msignal"] )) + ((-2.0))) > -998, data["abs_maxbatch_slices2_msignal"], data["abs_maxbatch_msignal"] ) <= -998, data["abs_avgbatch_slices2"], data["abs_maxbatch_msignal"] )))) +

                            0.100000*np.tanh(np.sin((((data["minbatch_msignal"]) + (np.where((-2.0) > -998, data["meanbatch_msignal"], ((data["maxbatch_slices2_msignal"]) * (data["minbatch_slices2_msignal"])) )))))) +

                            0.100000*np.tanh((-((np.cos((data["maxtominbatch_slices2"])))))) +

                            0.100000*np.tanh(np.cos((np.where(data["minbatch_msignal"] > -998, data["minbatch_msignal"], np.sin((data["minbatch_msignal"])) )))) +

                            0.100000*np.tanh((((((-3.0)) + (((data["stdbatch_slices2"]) * (((data["mean_abs_chgbatch_slices2"]) + (np.where(data["abs_maxbatch_msignal"] > -998, np.where(data["abs_maxbatch_msignal"] > -998, np.sin((data["medianbatch_msignal"])), ((np.where(np.sin((np.cos((data["abs_maxbatch_msignal"])))) > -998, data["mean_abs_chgbatch_slices2"], data["stdbatch_slices2"] )) * 2.0) ), ((data["maxtominbatch_slices2"]) * 2.0) )))))))) * 2.0)) +

                            0.100000*np.tanh((((-((((((np.tanh((np.cos((np.tanh(((-((np.cos((data["maxbatch_slices2_msignal"])))))))))))) * 2.0)) * 2.0))))) - (((data["minbatch_slices2_msignal"]) * (np.where((-((np.tanh((data["rangebatch_msignal"]))))) <= -998, np.tanh((np.where(((data["maxbatch_slices2_msignal"]) / 2.0) > -998, data["minbatch_slices2_msignal"], (-2.0) ))), np.cos((data["maxbatch_slices2_msignal"])) )))))) +

                            0.100000*np.tanh(np.where(((((np.where(data["stdbatch_msignal"] > -998, data["meanbatch_msignal"], ((((np.where(data["abs_avgbatch_slices2_msignal"] <= -998, data["abs_minbatch_slices2"], data["abs_maxbatch"] )) / 2.0)) * (np.cos(((-3.0))))) )) / 2.0)) - (((((3.0)) + ((-3.0)))/2.0))) > -998, data["medianbatch_msignal"], data["mean_abs_chgbatch_slices2"] )) +

                            0.100000*np.tanh(np.cos((np.where((((-3.0)) * 2.0) > -998, data["minbatch_msignal"], np.where(data["minbatch_msignal"] <= -998, (-3.0), np.cos((data["stdbatch_slices2"])) ) )))) +

                            0.100000*np.tanh(((np.where(np.cos((data["mean_abs_chgbatch_msignal"])) > -998, np.cos((((data["signal"]) - (np.cos((np.sin((data["minbatch_slices2_msignal"])))))))), (((((np.cos((data["meanbatch_slices2_msignal"]))) - (np.sin((data["meanbatch_slices2_msignal"]))))) + (np.sin((((data["signal"]) - (data["abs_maxbatch_slices2_msignal"]))))))/2.0) )) * 2.0)) +

                            0.100000*np.tanh(((np.cos(((((((3.0)) * (np.where(data["minbatch_slices2"] > -998, data["stdbatch_msignal"], np.where((3.0) <= -998, np.where(((data["abs_minbatch_slices2_msignal"]) * 2.0) > -998, data["stdbatch_msignal"], ((data["abs_maxbatch_msignal"]) * (data["stdbatch_msignal"])) ), (((3.0)) / 2.0) ) )))) / 2.0)))) * 2.0)) +

                            0.100000*np.tanh((-3.0)) +

                            0.100000*np.tanh(np.sin((np.where(np.where((((-((np.sin((np.cos((data["abs_minbatch_msignal"])))))))) / 2.0) > -998, data["minbatch_msignal"], ((data["abs_maxbatch_msignal"]) - (((data["maxbatch_msignal"]) - (((np.where(data["abs_maxbatch_msignal"] > -998, data["meanbatch_msignal"], data["mean_abs_chgbatch_slices2"] )) / 2.0))))) ) > -998, data["mean_abs_chgbatch_slices2"], ((np.sin((((data["maxbatch_slices2_msignal"]) / 2.0)))) * 2.0) )))) +

                            0.100000*np.tanh(np.cos((np.where(data["maxbatch_slices2_msignal"] <= -998, np.sin((data["signal_shift_-1"])), (((-3.0)) * (data["stdbatch_slices2"])) )))) +

                            0.100000*np.tanh(((data["stdbatch_msignal"]) * ((-((np.cos(((((np.cos((np.tanh((data["meanbatch_slices2_msignal"]))))) + (data["abs_avgbatch_slices2"]))/2.0))))))))) +

                            0.100000*np.tanh(np.cos(((((-2.0)) + (((np.where(np.tanh((np.tanh((np.cos((((np.cos((np.where(data["abs_maxbatch"] > -998, data["maxtominbatch"], data["mean_abs_chgbatch_slices2"] )))) * 2.0))))))) > -998, data["minbatch_msignal"], (((data["abs_maxbatch"]) + (data["minbatch_msignal"]))/2.0) )) / 2.0)))))) +

                            0.100000*np.tanh(((np.where(data["maxtominbatch"] > -998, (3.0), np.where(data["rangebatch_slices2"] > -998, data["meanbatch_slices2"], data["abs_maxbatch_slices2_msignal"] ) )) * (((data["abs_maxbatch_slices2_msignal"]) * 2.0)))) +

                            0.100000*np.tanh(np.cos((((np.tanh((np.where(data["rangebatch_msignal"] > -998, data["maxtominbatch_slices2_msignal"], data["rangebatch_slices2"] )))) * (data["maxbatch_msignal"]))))) +

                            0.100000*np.tanh(((data["abs_avgbatch_msignal"]) * (np.sin((np.cos((np.where(((data["mean_abs_chgbatch_slices2_msignal"]) + (np.cos((data["meanbatch_slices2_msignal"])))) > -998, data["minbatch_msignal"], data["abs_avgbatch_msignal"] )))))))) +

                            0.100000*np.tanh(np.sin((((data["minbatch_slices2"]) - (np.cos((((np.where(data["signal"] > -998, data["medianbatch_msignal"], data["abs_minbatch_slices2"] )) + (((((data["medianbatch_msignal"]) - (data["medianbatch_msignal"]))) * 2.0)))))))))) +

                            0.100000*np.tanh(np.where(data["mean_abs_chgbatch_msignal"] > -998, np.cos((data["maxbatch_msignal"])), ((data["signal_shift_-1"]) * (np.cos(((2.0))))) )) +

                            0.100000*np.tanh(np.cos((np.where(np.sin(((((-1.0)) - (np.where(data["abs_avgbatch_slices2_msignal"] <= -998, data["rangebatch_slices2"], data["signal_shift_+1_msignal"] ))))) > -998, np.where(data["rangebatch_msignal"] > -998, data["minbatch_msignal"], (((-((((((0.0)) + (data["medianbatch_msignal"]))/2.0))))) / 2.0) ), (1.0) )))) +

                            0.100000*np.tanh(((np.cos((np.where((-1.0) > -998, data["abs_maxbatch_slices2"], np.where((-3.0) > -998, np.where(data["minbatch_slices2_msignal"] > -998, (-(((((((np.where(data["minbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2"], (14.31780338287353516) )) + (data["stdbatch_slices2_msignal"]))/2.0)) / 2.0)))), data["maxtominbatch_msignal"] ), data["meanbatch_slices2_msignal"] ) )))) * 2.0)) +

                            0.100000*np.tanh(np.where(np.sin(((((data["stdbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2"]))/2.0))) <= -998, np.sin(((3.0))), ((np.cos((((data["abs_maxbatch_slices2_msignal"]) * (np.cos((np.sin((data["maxbatch_msignal"]))))))))) * 2.0) )) +

                            0.100000*np.tanh(np.where(np.cos(((-((data["maxtominbatch_msignal"]))))) <= -998, np.where(np.tanh((((np.tanh(((-2.0)))) / 2.0))) <= -998, data["maxbatch_msignal"], data["abs_avgbatch_slices2"] ), ((np.where(data["maxbatch_msignal"] <= -998, data["maxbatch_msignal"], ((np.cos(((((-((data["maxbatch_msignal"])))) * 2.0)))) * 2.0) )) * 2.0) )) +

                            0.100000*np.tanh(((((np.where(data["meanbatch_msignal"] > -998, np.cos((data["abs_maxbatch_msignal"])), ((np.sin((np.cos((data["signal_shift_-1_msignal"]))))) * (data["meanbatch_msignal"])) )) * (data["signal_shift_+1"]))) + (np.where(data["stdbatch_msignal"] <= -998, np.where(data["signal_shift_+1"] > -998, data["abs_avgbatch_slices2"], data["meanbatch_msignal"] ), np.sin((((data["medianbatch_msignal"]) * 2.0))) )))) +

                            0.100000*np.tanh(np.sin(((-(((0.0))))))) +

                            0.100000*np.tanh(((np.sin((((((data["minbatch_msignal"]) - (np.where(data["abs_avgbatch_slices2"] <= -998, np.cos((data["maxtominbatch"])), np.sin((((((data["medianbatch_slices2_msignal"]) / 2.0)) + (data["abs_avgbatch_msignal"])))) )))) / 2.0)))) * 2.0)) +

                            0.100000*np.tanh(((((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) * 2.0)) * (np.sin((np.where((3.0) > -998, data["maxtominbatch_slices2_msignal"], np.sin((data["stdbatch_slices2"])) )))))) +

                            0.100000*np.tanh((((-((np.cos((((np.where((((((np.sin((data["maxbatch_msignal"]))) / 2.0)) + (np.where(np.cos((((np.where(data["meanbatch_msignal"] > -998, data["meanbatch_msignal"], np.sin((np.sin((((data["meanbatch_slices2_msignal"]) * 2.0))))) )) + (data["medianbatch_slices2"])))) <= -998, data["maxtominbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] )))/2.0) > -998, data["meanbatch_msignal"], data["meanbatch_msignal"] )) * 2.0))))))) * 2.0)) +

                            0.100000*np.tanh(np.where(((data["meanbatch_msignal"]) * 2.0) > -998, data["maxbatch_slices2_msignal"], ((((data["signal_shift_-1_msignal"]) / 2.0)) * (data["signal_shift_-1"])) )) +

                            0.100000*np.tanh(np.where(np.sin((((data["minbatch_msignal"]) * 2.0))) > -998, (-3.0), ((np.where(np.cos(((((-1.0)) * 2.0))) <= -998, np.sin((np.sin((np.cos((((np.tanh(((1.0)))) * 2.0))))))), data["signal_shift_+1"] )) / 2.0) )) +

                            0.100000*np.tanh(np.where((((-((np.where(data["meanbatch_msignal"] <= -998, data["meanbatch_msignal"], data["medianbatch_msignal"] ))))) / 2.0) > -998, data["medianbatch_msignal"], (-(((-((((data["signal_shift_-1"]) / 2.0))))))) )) +

                            0.100000*np.tanh(data["meanbatch_msignal"]) +

                            0.100000*np.tanh(np.sin((((((data["abs_maxbatch"]) * 2.0)) * ((((-1.0)) * 2.0)))))) +

                            0.100000*np.tanh(((data["rangebatch_slices2"]) * (np.cos(((((np.where(data["rangebatch_msignal"] > -998, data["rangebatch_msignal"], data["rangebatch_msignal"] )) + ((-((np.sin((np.cos(((((data["stdbatch_slices2"]) + ((-((data["signal"])))))/2.0))))))))))/2.0)))))) +

                            0.100000*np.tanh(((data["meanbatch_slices2"]) + (((((data["maxtominbatch"]) - (np.sin((np.where(data["meanbatch_slices2"] <= -998, np.sin((np.tanh((data["abs_avgbatch_msignal"])))), np.where(np.cos((((data["maxtominbatch"]) - (data["meanbatch_slices2"])))) > -998, data["minbatch_slices2_msignal"], np.tanh((np.sin(((((-((np.sin((data["maxtominbatch"])))))) / 2.0))))) ) )))))) / 2.0)))) +

                            0.100000*np.tanh(np.sin((data["meanbatch_msignal"]))) +

                            0.100000*np.tanh(np.where((((np.sin((((np.tanh((np.sin((np.sin((np.sin((np.where(data["maxtominbatch_msignal"] > -998, data["stdbatch_slices2_msignal"], ((np.sin(((3.32428050041198730)))) - (data["meanbatch_msignal"])) )))))))))) / 2.0)))) + (data["rangebatch_msignal"]))/2.0) > -998, np.sin((data["medianbatch_msignal"])), np.tanh((np.tanh((data["rangebatch_slices2"])))) )) +

                            0.100000*np.tanh(np.sin((((data["minbatch"]) - (np.where(np.cos((np.sin((np.cos((data["stdbatch_msignal"])))))) > -998, ((data["stdbatch_msignal"]) * (data["stdbatch_slices2"])), np.tanh(((((data["signal_shift_-1_msignal"]) + (((data["stdbatch_msignal"]) * 2.0)))/2.0))) )))))) +

                            0.100000*np.tanh(((data["meanbatch_msignal"]) * (np.cos(((((data["maxbatch_slices2"]) + ((-((np.where(data["minbatch"] <= -998, (0.0), np.tanh(((-((np.sin((((((np.where(((data["rangebatch_msignal"]) * (data["medianbatch_msignal"])) <= -998, data["signal_shift_+1"], ((data["abs_minbatch_msignal"]) - ((-1.0))) )) / 2.0)) * 2.0)))))))) ))))))/2.0)))))) +

                            0.100000*np.tanh(((np.cos((np.where(((data["signal"]) * 2.0) > -998, data["minbatch_msignal"], ((data["mean_abs_chgbatch_slices2"]) + (np.cos((np.where(((data["stdbatch_slices2_msignal"]) - (((data["medianbatch_msignal"]) * (data["stdbatch_slices2_msignal"])))) > -998, (-3.0), ((data["abs_avgbatch_slices2_msignal"]) + (np.where(((np.cos(((-3.0)))) * 2.0) <= -998, data["minbatch_msignal"], data["mean_abs_chgbatch_slices2"] ))) ))))) )))) * 2.0)) +

                            0.100000*np.tanh(np.sin((np.where((((1.0)) / 2.0) <= -998, data["signal_shift_-1"], (-3.0) )))) +

                            0.100000*np.tanh(np.where(data["signal_shift_+1_msignal"] > -998, np.sin((((np.sin((data["medianbatch_slices2_msignal"]))) + (np.cos((data["maxbatch_slices2"])))))), (-3.0) )) +

                            0.100000*np.tanh(((((((((((-2.0)) - (data["maxbatch_slices2_msignal"]))) * ((-((np.cos((data["stdbatch_msignal"])))))))) + (((data["minbatch_slices2"]) - (data["medianbatch_msignal"]))))/2.0)) * 2.0)) +

                            0.100000*np.tanh(np.sin((data["meanbatch_msignal"]))) +

                            0.100000*np.tanh(np.sin((((data["meanbatch_slices2"]) + ((((-3.0)) + (np.where((((((-2.0)) + (((((-2.0)) + (data["meanbatch_slices2"]))/2.0)))) / 2.0) <= -998, (0.0), data["signal"] )))))))) +

                            0.100000*np.tanh(np.where(np.where(data["maxbatch_msignal"] > -998, np.cos((data["maxbatch_msignal"])), data["stdbatch_msignal"] ) > -998, np.where(np.cos((((np.where(np.sin((data["meanbatch_msignal"])) > -998, data["stdbatch_slices2_msignal"], (((data["maxbatch_slices2"]) + (data["signal_shift_-1_msignal"]))/2.0) )) * 2.0))) > -998, np.cos((data["maxbatch_msignal"])), (0.0) ), data["minbatch_msignal"] )) +

                            0.100000*np.tanh(np.cos((np.where((-((data["rangebatch_slices2"]))) > -998, np.where(data["abs_minbatch_slices2_msignal"] > -998, (-((data["rangebatch_slices2"]))), ((data["medianbatch_slices2"]) - (np.where(np.where(data["meanbatch_slices2_msignal"] <= -998, (0.0), (((-((data["abs_maxbatch_slices2_msignal"])))) / 2.0) ) > -998, (2.0), ((((data["meanbatch_slices2_msignal"]) * (data["signal_shift_-1_msignal"]))) - (data["mean_abs_chgbatch_slices2"])) ))) ), data["medianbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh(np.sin((((data["stdbatch_msignal"]) - (data["abs_maxbatch_slices2"]))))) +

                            0.100000*np.tanh(np.where(data["maxtominbatch_slices2_msignal"] > -998, (((((((data["signal_shift_+1"]) * 2.0)) + (data["stdbatch_msignal"]))/2.0)) * 2.0), ((np.where(data["signal_shift_+1"] > -998, ((data["signal_shift_+1"]) / 2.0), data["minbatch_msignal"] )) / 2.0) )) +

                            0.100000*np.tanh((((-3.0)) - ((((data["minbatch_msignal"]) + (np.cos((((np.sin((((data["minbatch_slices2_msignal"]) * 2.0)))) * (((np.cos((((np.sin(((((0.0)) * 2.0)))) * ((0.0)))))) - (data["minbatch_msignal"]))))))))/2.0)))) +

                            0.100000*np.tanh(np.cos((np.where((-((((data["rangebatch_slices2"]) * (data["rangebatch_slices2"]))))) <= -998, np.cos((((np.tanh((data["rangebatch_slices2"]))) + (np.tanh((np.tanh((data["abs_maxbatch_slices2_msignal"])))))))), ((data["rangebatch_slices2"]) / 2.0) )))) +

                            0.100000*np.tanh(((np.sin((data["minbatch_msignal"]))) * (((data["signal_shift_-1_msignal"]) + ((((data["signal_shift_-1_msignal"]) + ((((((data["minbatch_msignal"]) / 2.0)) + (data["minbatch_msignal"]))/2.0)))/2.0)))))) +

                            0.100000*np.tanh(((np.sin((((data["minbatch_msignal"]) * (np.tanh(((((np.where(data["abs_maxbatch_msignal"] > -998, data["meanbatch_msignal"], ((np.sin((((data["abs_avgbatch_msignal"]) * (np.where((0.0) > -998, (((3.0)) / 2.0), data["mean_abs_chgbatch_msignal"] )))))) - (data["abs_minbatch_slices2"])) )) + (np.sin((data["minbatch_msignal"]))))/2.0)))))))) * 2.0)) +

                            0.100000*np.tanh(np.cos((np.where(data["medianbatch_msignal"] > -998, data["abs_maxbatch_msignal"], data["signal_shift_-1"] )))) +

                            0.100000*np.tanh(np.sin((((data["minbatch"]) - (np.tanh((np.tanh((((((data["medianbatch_msignal"]) * (np.where(np.cos((data["mean_abs_chgbatch_slices2_msignal"])) > -998, data["medianbatch_msignal"], np.sin((np.sin((data["minbatch"])))) )))) / 2.0)))))))))) +

                            0.100000*np.tanh(np.sin((np.where(np.where(data["stdbatch_slices2_msignal"] <= -998, data["signal_shift_-1"], data["stdbatch_slices2_msignal"] ) <= -998, ((data["minbatch_msignal"]) + (data["signal_shift_-1"])), ((((data["minbatch_msignal"]) + (data["minbatch_slices2"]))) / 2.0) )))) +

                            0.100000*np.tanh(np.sin(((((data["meanbatch_msignal"]) + (data["abs_minbatch_slices2_msignal"]))/2.0)))) +

                            0.100000*np.tanh(((((np.where(data["abs_maxbatch_msignal"] <= -998, np.sin(((-((np.sin((data["maxtominbatch_slices2_msignal"]))))))), ((data["meanbatch_msignal"]) * ((-((data["mean_abs_chgbatch_msignal"]))))) )) / 2.0)) * ((3.19136095046997070)))) +

                            0.100000*np.tanh(data["minbatch_slices2"]) +

                            0.100000*np.tanh(((np.tanh((np.sin((np.where(data["abs_avgbatch_msignal"] <= -998, np.where(data["minbatch"] <= -998, (((-1.0)) - (np.tanh((((data["minbatch"]) / 2.0))))), data["stdbatch_msignal"] ), ((data["abs_avgbatch_msignal"]) - (np.sin((data["abs_maxbatch_slices2_msignal"])))) )))))) * 2.0)) +

                            0.100000*np.tanh((((np.sin(((0.0)))) + ((-((np.sin((np.sin((((np.where(((np.where(data["abs_minbatch_slices2"] > -998, (-2.0), (-((((data["abs_minbatch_slices2"]) * 2.0)))) )) * 2.0) <= -998, ((data["abs_minbatch_slices2_msignal"]) / 2.0), (0.0) )) / 2.0))))))))))/2.0)) +

                            0.100000*np.tanh(np.tanh((((np.sin((((((np.tanh((data["signal"]))) * (np.cos(((-((data["maxtominbatch_slices2_msignal"])))))))) / 2.0)))) - (np.where(np.tanh((data["maxtominbatch_slices2_msignal"])) <= -998, (-1.0), np.tanh((((((np.sin(((1.0)))) + (np.tanh(((-1.0)))))) / 2.0))) )))))) +

                            0.100000*np.tanh(((data["maxtominbatch_slices2_msignal"]) * ((0.0)))) +

                            0.100000*np.tanh(np.where(((((data["rangebatch_slices2_msignal"]) / 2.0)) / 2.0) > -998, np.sin((((data["abs_maxbatch_slices2"]) + ((-1.0))))), data["abs_avgbatch_msignal"] )) +

                            0.100000*np.tanh(np.tanh((data["medianbatch_msignal"]))) +

                            0.100000*np.tanh(np.sin((data["medianbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(np.cos((np.where((-3.0) <= -998, data["rangebatch_msignal"], np.where((((((-3.0)) - (data["mean_abs_chgbatch_msignal"]))) * 2.0) > -998, data["abs_maxbatch_slices2"], np.where((((((-3.0)) - (data["mean_abs_chgbatch_msignal"]))) * 2.0) <= -998, data["maxtominbatch_slices2_msignal"], data["abs_minbatch_slices2_msignal"] ) ) )))) +

                            0.100000*np.tanh(np.cos((np.where(((data["minbatch_msignal"]) + (((np.where(data["abs_avgbatch_slices2"] <= -998, (-((((data["stdbatch_msignal"]) + (data["abs_minbatch_slices2"]))))), np.where(data["maxtominbatch_slices2_msignal"] > -998, np.sin((data["meanbatch_slices2"])), data["signal_shift_+1_msignal"] ) )) + (np.tanh((data["rangebatch_slices2"])))))) <= -998, ((data["stdbatch_slices2_msignal"]) / 2.0), ((data["stdbatch_msignal"]) + (data["mean_abs_chgbatch_msignal"])) )))) +

                            0.100000*np.tanh(np.sin((np.sin((((np.tanh((data["meanbatch_msignal"]))) / 2.0)))))) +

                            0.100000*np.tanh((((((((((data["maxtominbatch_slices2_msignal"]) + (np.where(data["maxbatch_msignal"] > -998, data["rangebatch_slices2_msignal"], np.tanh((np.sin((np.tanh((data["rangebatch_slices2_msignal"])))))) )))) + ((-3.0)))/2.0)) * 2.0)) * 2.0)) +

                            0.100000*np.tanh((((-2.0)) - ((((data["minbatch_msignal"]) + (np.tanh((data["maxbatch_slices2_msignal"]))))/2.0)))) +

                            0.100000*np.tanh(((((data["abs_maxbatch_slices2"]) * ((-((np.cos((data["rangebatch_msignal"])))))))) * 2.0)) +

                            0.100000*np.tanh((((np.tanh(((-1.0)))) + ((((((data["mean_abs_chgbatch_slices2"]) + (np.sin((data["medianbatch_msignal"]))))/2.0)) - (np.cos((np.where(((data["abs_avgbatch_slices2_msignal"]) * (np.tanh(((2.0))))) <= -998, data["meanbatch_msignal"], ((data["abs_minbatch_msignal"]) / 2.0) )))))))/2.0)) +

                            0.100000*np.tanh(np.cos((data["maxbatch_msignal"]))) +

                            0.100000*np.tanh(np.sin((np.sin((np.where(data["maxbatch_slices2_msignal"] <= -998, np.sin((np.where(data["rangebatch_slices2_msignal"] <= -998, ((((data["stdbatch_slices2_msignal"]) * 2.0)) * 2.0), data["stdbatch_slices2_msignal"] ))), data["stdbatch_slices2_msignal"] )))))) +

                            0.100000*np.tanh(np.tanh((np.where(np.cos((np.tanh((((np.tanh(((0.0)))) + ((((data["abs_avgbatch_msignal"]) + (np.sin((np.tanh((data["signal_shift_+1"]))))))/2.0))))))) > -998, data["medianbatch_msignal"], (((data["signal"]) + ((((np.where(data["maxbatch_msignal"] > -998, data["medianbatch_msignal"], data["abs_avgbatch_msignal"] )) + (data["medianbatch_msignal"]))/2.0)))/2.0) )))) +

                            0.100000*np.tanh(((np.sin((((data["mean_abs_chgbatch_slices2"]) * (((data["rangebatch_msignal"]) + (data["maxbatch_slices2"]))))))) / 2.0)) +

                            0.100000*np.tanh(np.sin((((((np.sin((((np.sin((((data["signal_shift_-1"]) + (((((np.cos((((((data["maxtominbatch_msignal"]) * 2.0)) * (data["meanbatch_slices2"]))))) * 2.0)) / 2.0)))))) * (data["maxtominbatch_msignal"]))))) / 2.0)) - (np.where(data["meanbatch_slices2_msignal"] > -998, ((data["maxtominbatch_msignal"]) * 2.0), data["maxtominbatch_msignal"] )))))) +

                            0.100000*np.tanh(np.where(np.where(((data["meanbatch_msignal"]) * (((np.tanh((np.where(data["signal"] > -998, ((data["signal_shift_+1"]) - (data["maxtominbatch"])), data["signal_shift_-1"] )))) / 2.0))) <= -998, (((data["maxbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))/2.0), data["abs_avgbatch_slices2_msignal"] ) > -998, data["signal_shift_-1"], data["medianbatch_slices2_msignal"] )) +

                            0.100000*np.tanh((((np.sin(((((np.where(np.cos(((2.0))) > -998, data["abs_minbatch_slices2"], (-((data["minbatch_slices2"]))) )) + (((data["rangebatch_msignal"]) * (((((2.0)) + (np.where(np.sin((data["maxtominbatch_slices2_msignal"])) <= -998, data["abs_maxbatch"], np.cos((((data["stdbatch_msignal"]) / 2.0))) )))/2.0)))))/2.0)))) + (np.sin((np.sin((data["signal_shift_-1"]))))))/2.0)) +

                            0.100000*np.tanh(np.sin(((((data["minbatch_msignal"]) + (np.tanh((np.where((0.0) <= -998, ((data["minbatch_slices2_msignal"]) / 2.0), np.where(data["signal_shift_+1"] > -998, (-((data["rangebatch_slices2"]))), (-((((np.sin((((data["stdbatch_slices2"]) * 2.0)))) * ((0.0)))))) ) )))))/2.0)))) +

                            0.100000*np.tanh(np.sin((np.tanh((np.tanh((np.tanh((data["medianbatch_msignal"]))))))))) +

                            0.100000*np.tanh(((((data["maxbatch_msignal"]) * (data["signal_shift_+1_msignal"]))) * (np.sin((np.where(np.where(data["abs_maxbatch_msignal"] > -998, data["abs_maxbatch_msignal"], ((data["signal_shift_+1_msignal"]) * (data["signal_shift_+1_msignal"])) ) > -998, data["abs_maxbatch_msignal"], data["abs_maxbatch_msignal"] )))))) +

                            0.100000*np.tanh(((np.cos(((2.0)))) + (np.sin((np.where(data["abs_maxbatch_slices2_msignal"] > -998, np.where((((2.0)) * ((((-((np.sin(((-2.0))))))) / 2.0))) <= -998, (0.0), np.tanh((data["abs_maxbatch_slices2_msignal"])) ), data["signal_shift_-1"] )))))) +

                            0.100000*np.tanh(np.cos((np.where(data["rangebatch_slices2_msignal"] > -998, ((np.cos((np.where((3.0) <= -998, (-((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))))), data["abs_maxbatch"] )))) * 2.0), np.tanh(((((0.0)) * 2.0))) )))) +

                            0.100000*np.tanh(np.cos(((((2.0)) * (np.where(data["meanbatch_slices2_msignal"] <= -998, data["rangebatch_slices2_msignal"], np.where(data["mean_abs_chgbatch_slices2"] > -998, ((((0.0)) + (((data["minbatch"]) * 2.0)))/2.0), (-3.0) ) )))))) +

                            0.100000*np.tanh((((((2.0)) / 2.0)) / 2.0)) +

                            0.100000*np.tanh(np.sin((((((data["medianbatch_slices2_msignal"]) - ((((7.46158885955810547)) - ((((0.0)) * (np.cos((np.sin((np.cos(((((-((np.sin((data["abs_minbatch_msignal"])))))) + (np.cos((np.cos(((-2.0)))))))))))))))))))) / 2.0)))) +

                            0.100000*np.tanh(np.tanh((np.sin((((np.where(np.sin((data["meanbatch_slices2"])) <= -998, data["abs_avgbatch_msignal"], data["abs_maxbatch_msignal"] )) * 2.0)))))) +

                            0.100000*np.tanh(np.where(np.tanh((np.tanh((np.where(np.cos((data["abs_maxbatch"])) > -998, np.tanh((data["meanbatch_slices2"])), data["signal_shift_-1"] ))))) <= -998, np.cos((np.where(np.tanh((data["abs_avgbatch_msignal"])) <= -998, data["abs_maxbatch"], data["abs_avgbatch_msignal"] ))), ((np.cos((np.where((0.0) <= -998, data["abs_avgbatch_msignal"], data["abs_avgbatch_msignal"] )))) * 2.0) )) +

                            0.100000*np.tanh(np.where((((data["abs_maxbatch_slices2"]) + (data["maxtominbatch_msignal"]))/2.0) <= -998, data["abs_maxbatch_slices2"], (((data["abs_maxbatch_slices2"]) + (np.where(data["medianbatch_slices2_msignal"] > -998, data["maxtominbatch_msignal"], np.where((((data["abs_maxbatch_slices2"]) + (data["maxtominbatch_msignal"]))/2.0) > -998, (10.46357536315917969), data["mean_abs_chgbatch_slices2"] ) )))/2.0) )) +

                            0.100000*np.tanh(((np.sin((np.where((2.0) > -998, np.sin((data["meanbatch_msignal"])), data["abs_maxbatch_msignal"] )))) / 2.0)) +

                            0.100000*np.tanh(np.sin((np.cos((((((((data["stdbatch_slices2"]) + (data["rangebatch_slices2"]))/2.0)) + ((((np.cos((((((((data["stdbatch_slices2"]) + (np.tanh(((2.0)))))/2.0)) + (((((((data["meanbatch_slices2_msignal"]) + (((np.sin((((((data["stdbatch_slices2"]) + (data["medianbatch_msignal"]))) / 2.0)))) / 2.0)))/2.0)) + (data["rangebatch_msignal"]))/2.0)))/2.0)))) + (data["rangebatch_msignal"]))/2.0)))/2.0)))))) +

                            0.100000*np.tanh(np.tanh((np.tanh((((data["medianbatch_slices2_msignal"]) * ((1.0)))))))) +

                            0.100000*np.tanh(((np.where(((np.sin((data["medianbatch_slices2_msignal"]))) / 2.0) <= -998, np.where(data["signal_shift_+1_msignal"] <= -998, np.sin((np.tanh((data["abs_minbatch_slices2"])))), (-3.0) ), (0.0) )) / 2.0)) +

                            0.100000*np.tanh(np.where(((data["signal_shift_+1"]) + (np.tanh((np.tanh((np.tanh((np.cos((np.tanh((np.tanh((data["minbatch"])))))))))))))) > -998, np.cos(((((data["abs_maxbatch_slices2_msignal"]) + (data["medianbatch_slices2"]))/2.0))), data["maxtominbatch_slices2_msignal"] )) +

                            0.100000*np.tanh((-((np.sin((np.where(((np.where(np.sin((((np.cos((data["maxtominbatch_slices2"]))) - (((data["abs_avgbatch_slices2"]) / 2.0))))) <= -998, data["abs_maxbatch"], data["maxtominbatch_slices2"] )) * 2.0) <= -998, ((np.cos((data["signal_shift_-1"]))) / 2.0), np.cos(((-1.0))) ))))))) +

                            0.100000*np.tanh(np.cos((np.where((2.0) > -998, data["abs_maxbatch_slices2_msignal"], np.where(data["abs_maxbatch_msignal"] > -998, data["signal_shift_-1"], np.cos((np.where(data["meanbatch_msignal"] > -998, (((data["signal_shift_-1"]) + (data["maxbatch_msignal"]))/2.0), (((data["signal_shift_-1"]) + (data["medianbatch_msignal"]))/2.0) ))) ) )))) +

                            0.100000*np.tanh(((np.cos((np.cos(((((np.cos(((-1.0)))) + (((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0)))/2.0)))))) / 2.0)) +

                            0.100000*np.tanh((((-3.0)) + ((((data["medianbatch_slices2"]) + (np.sin((((((data["meanbatch_slices2_msignal"]) - ((-(((((-3.0)) / 2.0))))))) + (((np.tanh((data["minbatch"]))) + ((13.18363571166992188)))))))))/2.0)))) +

                            0.100000*np.tanh(data["signal_shift_-1"]) +

                            0.100000*np.tanh(((np.sin((((np.sin(((-((np.cos((np.where((3.0) > -998, np.where(np.tanh(((((2.0)) * 2.0))) > -998, data["meanbatch_msignal"], ((((((-((data["signal_shift_-1_msignal"])))) + ((((3.0)) / 2.0)))/2.0)) * 2.0) ), data["signal_shift_+1"] ))))))))) * 2.0)))) * (data["signal_shift_-1_msignal"]))) +

                            0.100000*np.tanh((-(((((((0.0)) * (np.where(np.tanh(((0.0))) <= -998, np.sin((np.sin((data["stdbatch_slices2_msignal"])))), np.cos((((data["abs_avgbatch_slices2_msignal"]) + (((data["mean_abs_chgbatch_slices2"]) - ((0.0))))))) )))) * 2.0))))) +

                            0.100000*np.tanh(np.cos(((((data["maxtominbatch"]) + (np.where((((((data["minbatch_msignal"]) + (np.cos((data["medianbatch_slices2_msignal"]))))) + (np.where((-2.0) <= -998, np.tanh((np.cos(((((0.0)) + ((((-(((((-3.0)) / 2.0))))) * 2.0))))))), data["abs_avgbatch_slices2_msignal"] )))/2.0) > -998, data["meanbatch_msignal"], data["signal_shift_+1"] )))/2.0)))) +

                            0.100000*np.tanh(data["abs_avgbatch_msignal"]) +

                            0.100000*np.tanh(np.sin((np.where(((data["rangebatch_slices2"]) * (np.sin((((((np.sin((np.where(((np.where(np.sin(((6.0))) > -998, ((data["meanbatch_slices2_msignal"]) / 2.0), data["signal_shift_+1"] )) / 2.0) > -998, data["abs_avgbatch_msignal"], data["rangebatch_slices2_msignal"] )))) / 2.0)) / 2.0))))) <= -998, np.sin((np.cos((data["abs_avgbatch_slices2_msignal"])))), np.sin((np.cos((data["abs_avgbatch_slices2_msignal"])))) )))) +

                            0.100000*np.tanh(((((0.0)) + ((1.0)))/2.0)) +

                            0.100000*np.tanh(np.cos((((np.cos((data["maxtominbatch"]))) + (np.where(data["meanbatch_slices2"] > -998, data["minbatch_slices2_msignal"], np.tanh((np.sin((np.cos((data["stdbatch_msignal"])))))) )))))) +

                            0.100000*np.tanh((((2.0)) - ((((-(((((np.tanh((((data["stdbatch_slices2"]) / 2.0)))) + (np.sin((data["meanbatch_slices2_msignal"]))))/2.0))))) / 2.0)))) +

                            0.100000*np.tanh(np.sin((np.where(((np.tanh(((7.0)))) / 2.0) > -998, data["signal_shift_+1"], np.tanh((data["stdbatch_msignal"])) )))) +

                            0.100000*np.tanh(np.where(np.where(np.tanh(((11.27962875366210938))) > -998, data["maxbatch_slices2_msignal"], data["signal_shift_+1_msignal"] ) > -998, ((data["signal_shift_+1_msignal"]) * (np.sin((data["minbatch_msignal"])))), data["stdbatch_msignal"] )) +

                            0.100000*np.tanh(np.where(((data["signal_shift_+1"]) / 2.0) > -998, np.where(((data["rangebatch_msignal"]) * 2.0) <= -998, np.cos((np.sin(((0.0))))), data["signal_shift_+1"] ), data["signal"] )) +

                            0.100000*np.tanh(np.cos((data["mean_abs_chgbatch_msignal"]))) +

                            0.100000*np.tanh(((data["maxbatch_slices2_msignal"]) / 2.0)) +

                            0.100000*np.tanh(np.sin((np.where(np.tanh((data["signal_shift_+1_msignal"])) <= -998, ((np.sin((np.tanh((np.where((-((((data["abs_maxbatch_slices2"]) * 2.0)))) > -998, data["minbatch"], (-((data["medianbatch_slices2"]))) )))))) / 2.0), np.where(((((data["signal_shift_+1_msignal"]) * 2.0)) * 2.0) <= -998, data["stdbatch_slices2"], data["signal_shift_+1_msignal"] ) )))) +

                            0.100000*np.tanh(np.cos((((data["rangebatch_slices2_msignal"]) + (((data["minbatch_msignal"]) + (np.sin((np.sin((np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["abs_maxbatch"], np.cos((np.sin((data["meanbatch_slices2"])))) )))))))))))) +

                            0.100000*np.tanh(((data["signal_shift_-1"]) * (((data["rangebatch_slices2"]) * (((data["rangebatch_slices2"]) * (np.where(data["minbatch"] <= -998, data["abs_maxbatch_slices2_msignal"], np.cos((data["abs_maxbatch_msignal"])) )))))))) +

                            0.100000*np.tanh((((0.0)) / 2.0)) +

                            0.100000*np.tanh(np.sin((np.where((6.0) > -998, data["stdbatch_slices2_msignal"], ((((data["mean_abs_chgbatch_msignal"]) - ((-((((np.cos(((-((np.cos(((0.0))))))))) / 2.0))))))) * 2.0) )))) +

                            0.100000*np.tanh(((data["signal_shift_+1"]) + (np.where((((((data["meanbatch_msignal"]) / 2.0)) + (data["abs_avgbatch_slices2_msignal"]))/2.0) > -998, ((data["signal"]) + (np.where((2.0) > -998, data["maxtominbatch"], (-(((2.0)))) ))), (-2.0) )))) +

                            0.100000*np.tanh(np.sin((((data["stdbatch_msignal"]) + ((((0.0)) + (np.sin((data["stdbatch_msignal"]))))))))) +

                            0.100000*np.tanh((0.0)) +

                            0.100000*np.tanh(np.sin(((((np.cos((data["maxbatch_slices2_msignal"]))) + (np.tanh(((-((np.cos((np.cos((np.tanh((((np.tanh((np.where(data["abs_maxbatch_slices2"] <= -998, ((np.cos((np.tanh((data["abs_avgbatch_slices2"]))))) + (((np.sin((data["abs_maxbatch_slices2_msignal"]))) / 2.0))), np.cos(((3.0))) )))) / 2.0))))))))))))))/2.0)))) +

                            0.100000*np.tanh(np.tanh(((((5.14618062973022461)) + (((data["abs_avgbatch_slices2_msignal"]) + (((((0.26170855760574341)) + (data["signal_shift_-1_msignal"]))/2.0)))))))) +

                            0.099609*np.tanh(((((np.sin(((((((((0.0)) + (((np.tanh(((0.0)))) / 2.0)))/2.0)) + (data["stdbatch_slices2_msignal"]))/2.0)))) / 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.where(data["mean_abs_chgbatch_slices2"] <= -998, ((data["mean_abs_chgbatch_slices2"]) / 2.0), np.where((1.0) <= -998, (1.0), np.cos(((((0.0)) / 2.0))) ) )) +

                            0.100000*np.tanh(((data["signal_shift_-1_msignal"]) * ((((data["medianbatch_slices2"]) + ((((2.36748266220092773)) - (data["medianbatch_slices2"]))))/2.0)))) +

                            0.100000*np.tanh((((5.0)) * ((((((6.0)) * (np.where(((((((data["rangebatch_msignal"]) + (((data["abs_avgbatch_msignal"]) + ((((6.0)) / 2.0)))))) * (((np.cos((data["abs_avgbatch_msignal"]))) * 2.0)))) / 2.0) <= -998, data["signal_shift_-1_msignal"], np.cos((data["abs_avgbatch_msignal"])) )))) * 2.0)))) +

                            0.100000*np.tanh(np.where(np.tanh(((-1.0))) > -998, np.where(data["maxtominbatch_msignal"] <= -998, data["medianbatch_msignal"], np.where(np.cos((np.tanh((data["rangebatch_slices2_msignal"])))) > -998, data["medianbatch_msignal"], data["medianbatch_msignal"] ) ), (((data["meanbatch_msignal"]) + (data["abs_maxbatch_slices2_msignal"]))/2.0) )) +

                            0.100000*np.tanh(np.sin(((0.0)))) +

                            0.100000*np.tanh(np.where(np.tanh((np.cos((data["abs_maxbatch_slices2_msignal"])))) > -998, np.cos((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, np.where(((np.tanh((data["stdbatch_msignal"]))) / 2.0) > -998, data["signal_shift_+1"], np.where(data["abs_avgbatch_slices2_msignal"] > -998, np.sin(((0.0))), ((np.tanh((data["signal"]))) * 2.0) ) ), data["abs_maxbatch_slices2_msignal"] ))), data["medianbatch_msignal"] )) +

                            0.100000*np.tanh(((np.sin((((np.tanh(((((0.0)) * (np.sin((((((np.where((((-((np.sin((data["meanbatch_msignal"])))))) / 2.0) > -998, data["minbatch"], ((np.cos(((0.0)))) / 2.0) )) * 2.0)) * ((((0.0)) / 2.0)))))))))) / 2.0)))) / 2.0)) +

                            0.100000*np.tanh(np.where(np.sin((np.sin((data["stdbatch_msignal"])))) <= -998, (1.0), np.sin((((((-1.0)) + (np.where((3.0) > -998, data["minbatch_msignal"], ((np.cos((((np.sin((data["medianbatch_slices2_msignal"]))) + (((data["medianbatch_slices2_msignal"]) * ((1.0)))))))) / 2.0) )))/2.0))) )) +

                            0.100000*np.tanh(((np.where(np.where(data["minbatch_msignal"] > -998, (((1.0)) / 2.0), data["stdbatch_msignal"] ) > -998, np.sin((data["stdbatch_slices2_msignal"])), data["minbatch_msignal"] )) * (((((-3.0)) + (((data["rangebatch_msignal"]) / 2.0)))/2.0)))) +

                            0.100000*np.tanh(data["mean_abs_chgbatch_slices2"]) +

                            0.089932*np.tanh(np.cos((((data["rangebatch_msignal"]) + (np.sin((data["abs_maxbatch"]))))))) +

                            0.100000*np.tanh((((data["medianbatch_slices2_msignal"]) + (data["medianbatch_slices2_msignal"]))/2.0)) +

                            0.098436*np.tanh((((((data["meanbatch_msignal"]) + (np.tanh((np.where(data["mean_abs_chgbatch_slices2"] <= -998, np.where((0.0) <= -998, (((-3.0)) / 2.0), data["abs_minbatch_slices2"] ), np.sin((((data["minbatch_slices2_msignal"]) - (data["abs_avgbatch_slices2"])))) )))))/2.0)) / 2.0)) +

                            0.100000*np.tanh(np.cos((((np.where((2.16586637496948242) > -998, data["medianbatch_msignal"], np.sin((np.sin(((-(((((((np.cos((data["meanbatch_msignal"]))) + ((7.0)))/2.0)) * 2.0)))))))) )) / 2.0)))) +

                            0.100000*np.tanh(np.where(data["abs_avgbatch_msignal"] > -998, data["abs_minbatch_slices2"], np.sin(((0.0))) )) +

                            0.100000*np.tanh(np.where(((data["abs_avgbatch_slices2"]) + (np.sin((data["abs_avgbatch_slices2_msignal"])))) <= -998, np.cos((((data["abs_maxbatch_slices2"]) + (data["stdbatch_slices2_msignal"])))), data["abs_avgbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(((np.sin((((np.sin(((((0.0)) / 2.0)))) * (np.sin((data["medianbatch_slices2_msignal"]))))))) / 2.0)) +

                            0.100000*np.tanh(((((np.where(np.where(data["signal_shift_-1"] > -998, data["maxbatch_slices2_msignal"], data["maxbatch_slices2_msignal"] ) > -998, (3.0), data["maxbatch_slices2_msignal"] )) * (np.where(np.sin((data["abs_avgbatch_slices2"])) > -998, ((data["abs_avgbatch_slices2"]) - (((data["maxbatch_slices2_msignal"]) + (data["minbatch"])))), ((((data["abs_avgbatch_slices2"]) - (data["abs_avgbatch_slices2"]))) + (data["rangebatch_slices2"])) )))) * 2.0)) +

                            0.100000*np.tanh(np.where((1.0) > -998, np.tanh((np.tanh(((((np.tanh(((-((((data["abs_maxbatch_slices2"]) / 2.0))))))) + (data["rangebatch_slices2"]))/2.0))))), (0.0) )) +

                            0.100000*np.tanh(np.sin((((data["maxbatch_msignal"]) + (((data["mean_abs_chgbatch_msignal"]) * (data["minbatch"]))))))) +

                            0.093744*np.tanh((((((((data["meanbatch_msignal"]) + (data["abs_avgbatch_slices2"]))) + (data["medianbatch_msignal"]))/2.0)) * ((((((((np.cos((data["meanbatch_msignal"]))) * 2.0)) / 2.0)) + (np.tanh((data["signal_shift_-1"]))))/2.0)))) +

                            0.100000*np.tanh((0.0)) +

                            0.099707*np.tanh(np.tanh((np.where((-2.0) > -998, np.where(data["medianbatch_msignal"] > -998, np.where((-2.0) > -998, ((np.sin((data["medianbatch_msignal"]))) * 2.0), data["meanbatch_slices2_msignal"] ), (-((np.cos((np.sin((data["medianbatch_msignal"]))))))) ), data["signal_shift_-1"] )))) +

                            0.100000*np.tanh(((((np.tanh((np.tanh((((((data["stdbatch_msignal"]) / 2.0)) / 2.0)))))) * (np.sin((np.tanh((np.cos(((((((np.tanh((data["minbatch_slices2"]))) * 2.0)) + (data["medianbatch_slices2_msignal"]))/2.0)))))))))) / 2.0)) +

                            0.100000*np.tanh(np.tanh(((((((np.cos((np.sin((np.cos((np.cos((((np.sin((data["rangebatch_msignal"]))) * 2.0)))))))))) / 2.0)) + (((np.tanh((((data["abs_maxbatch_slices2"]) * 2.0)))) / 2.0)))/2.0)))) +

                            0.100000*np.tanh(data["abs_maxbatch"]))     

    

    def GP_class_7(self,data):

        return self.Output( -2.939938 +

                            0.100000*np.tanh(np.where(np.tanh((((data["abs_maxbatch_slices2_msignal"]) + ((8.0))))) > -998, data["meanbatch_slices2"], (((2.0)) * (data["maxbatch_slices2"])) )) +

                            0.100000*np.tanh((((((data["meanbatch_msignal"]) - (data["meanbatch_msignal"]))) + (((np.tanh((((((data["meanbatch_msignal"]) * 2.0)) * (data["signal_shift_-1"]))))) - (((data["stdbatch_slices2_msignal"]) + (np.sin((data["abs_avgbatch_slices2_msignal"]))))))))/2.0)) +

                            0.100000*np.tanh((-((np.where(data["signal_shift_+1"] > -998, np.sin((data["abs_maxbatch_slices2_msignal"])), np.cos((np.tanh((((np.where(data["signal_shift_-1"] > -998, data["signal_shift_+1"], (-(((((((-((data["minbatch"])))) * 2.0)) / 2.0)))) )) - (((data["maxtominbatch_slices2_msignal"]) * (data["abs_maxbatch_slices2"])))))))) ))))) +

                            0.100000*np.tanh(((data["rangebatch_slices2_msignal"]) * ((-((np.where(np.where(data["stdbatch_slices2_msignal"] <= -998, (1.50265610218048096), (((data["signal"]) + (np.cos(((((10.0)) - (data["medianbatch_slices2"]))))))/2.0) ) <= -998, data["medianbatch_slices2"], np.tanh((np.where(data["abs_maxbatch"] > -998, (((10.0)) - (data["rangebatch_slices2"])), data["meanbatch_slices2"] ))) ))))))) +

                            0.100000*np.tanh(((np.sin((data["abs_maxbatch_slices2_msignal"]))) * (np.where(np.cos((np.cos(((((data["meanbatch_msignal"]) + (np.where(data["signal_shift_-1"] > -998, data["stdbatch_slices2"], (-((data["signal"]))) )))/2.0))))) > -998, data["mean_abs_chgbatch_msignal"], data["maxtominbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh(((data["medianbatch_msignal"]) + (data["medianbatch_slices2"]))) +

                            0.100000*np.tanh(np.where(data["signal_shift_-1"] > -998, data["signal_shift_-1"], (((((np.tanh((np.where(np.where((1.0) > -998, data["signal_shift_-1"], (((data["maxbatch_slices2"]) + (data["signal_shift_-1"]))/2.0) ) > -998, (((1.0)) * (data["maxbatch_slices2_msignal"])), ((data["signal_shift_-1"]) + ((((data["signal_shift_-1"]) + (data["abs_avgbatch_msignal"]))/2.0))) )))) * 2.0)) + (data["minbatch_slices2_msignal"]))/2.0) )) +

                            0.100000*np.tanh((-((((((data["rangebatch_msignal"]) * (np.where(np.where(np.sin((data["minbatch_slices2"])) <= -998, data["maxbatch_slices2_msignal"], ((data["minbatch"]) * ((((2.0)) / 2.0))) ) <= -998, data["maxbatch_slices2_msignal"], np.sin((data["maxbatch_slices2_msignal"])) )))) / 2.0))))) +

                            0.100000*np.tanh(((data["abs_maxbatch_slices2"]) * ((-((np.where((((data["mean_abs_chgbatch_slices2"]) + (data["medianbatch_slices2"]))/2.0) <= -998, (((((data["signal_shift_+1"]) / 2.0)) + (np.cos((((data["mean_abs_chgbatch_msignal"]) * 2.0)))))/2.0), np.where(((np.cos((data["minbatch_slices2"]))) - (((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0))) > -998, np.cos((data["stdbatch_slices2"])), ((data["medianbatch_slices2"]) - (data["meanbatch_msignal"])) ) ))))))) +

                            0.100000*np.tanh(np.where((-((data["stdbatch_slices2"]))) > -998, data["medianbatch_slices2"], (((np.tanh((data["medianbatch_slices2"]))) + (np.tanh((np.where(data["signal_shift_-1"] <= -998, data["mean_abs_chgbatch_slices2"], ((data["maxtominbatch_slices2"]) + (np.sin((data["maxbatch_slices2"])))) )))))/2.0) )) +

                            0.100000*np.tanh(((np.sin((((data["minbatch_msignal"]) / 2.0)))) * (data["abs_maxbatch_msignal"]))) +

                            0.100000*np.tanh(((np.where(np.where(np.tanh((((((((13.26908016204833984)) + (data["signal_shift_+1"]))/2.0)) / 2.0))) <= -998, data["mean_abs_chgbatch_slices2_msignal"], (-((data["signal_shift_-1"]))) ) <= -998, data["signal_shift_+1_msignal"], np.sin((((data["minbatch_msignal"]) / 2.0))) )) * 2.0)) +

                            0.100000*np.tanh(((data["maxbatch_slices2_msignal"]) + (np.where(data["maxbatch_slices2_msignal"] > -998, (-3.0), ((np.cos(((-3.0)))) + (data["meanbatch_slices2"])) )))) +

                            0.100000*np.tanh((((-3.0)) + (((np.where(np.sin((data["meanbatch_msignal"])) <= -998, data["abs_minbatch_msignal"], data["maxbatch_msignal"] )) + ((((-3.0)) + (((np.where(data["maxbatch_slices2_msignal"] > -998, np.sin((data["meanbatch_slices2"])), (9.0) )) - ((((-((data["medianbatch_msignal"])))) - ((((-((data["meanbatch_msignal"])))) * ((-3.0)))))))))))))) +

                            0.100000*np.tanh(((data["minbatch_msignal"]) * (np.where((((data["signal_shift_+1"]) + ((((np.sin((((data["rangebatch_slices2"]) * (((data["minbatch_msignal"]) + (((data["maxbatch_msignal"]) / 2.0)))))))) + (np.sin((data["maxbatch_msignal"]))))/2.0)))/2.0) > -998, np.sin((data["minbatch_msignal"])), data["minbatch_msignal"] )))) +

                            0.100000*np.tanh(((data["signal_shift_+1"]) * (((((np.where(data["signal"] <= -998, (1.0), data["abs_avgbatch_msignal"] )) + (data["rangebatch_msignal"]))) - (np.sin((np.where(((data["abs_avgbatch_slices2_msignal"]) / 2.0) <= -998, data["signal"], np.tanh((data["minbatch_slices2"])) )))))))) +

                            0.100000*np.tanh(((data["minbatch_msignal"]) - (((data["abs_minbatch_slices2_msignal"]) * (np.where(((np.where(((data["abs_minbatch_slices2_msignal"]) - (data["maxbatch_slices2_msignal"])) > -998, (-1.0), np.where(data["minbatch_msignal"] <= -998, (((0.0)) * (data["maxbatch_slices2_msignal"])), data["maxbatch_slices2_msignal"] ) )) + (data["abs_minbatch_slices2_msignal"])) <= -998, data["signal_shift_+1_msignal"], ((((data["meanbatch_slices2_msignal"]) * 2.0)) + (data["maxbatch_slices2_msignal"])) )))))) +

                            0.100000*np.tanh(((data["abs_maxbatch"]) - (((((13.83147907257080078)) + (np.where(data["abs_maxbatch"] > -998, data["minbatch_slices2"], np.sin((np.sin((np.where(np.where(data["abs_maxbatch"] > -998, data["signal_shift_-1"], np.tanh((data["mean_abs_chgbatch_msignal"])) ) > -998, ((((data["signal_shift_+1_msignal"]) * 2.0)) * ((-(((13.83147907257080078)))))), ((((13.83147907257080078)) + ((((13.83147907257080078)) * 2.0)))/2.0) ))))) )))/2.0)))) +

                            0.100000*np.tanh(np.where(((data["mean_abs_chgbatch_msignal"]) - (np.where(np.where(np.cos((data["mean_abs_chgbatch_msignal"])) <= -998, data["maxtominbatch_slices2_msignal"], data["medianbatch_slices2_msignal"] ) > -998, (3.0), data["signal_shift_-1_msignal"] ))) > -998, ((np.tanh((np.sin(((-2.0)))))) - (np.sin((data["maxbatch_msignal"])))), data["abs_avgbatch_msignal"] )) +

                            0.100000*np.tanh(((data["maxbatch_msignal"]) * (np.cos((data["minbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh(np.sin((np.where(np.where(data["signal"] > -998, data["mean_abs_chgbatch_slices2_msignal"], np.sin((np.where(np.sin(((-(((((((2.24183011054992676)) * (data["abs_maxbatch_msignal"]))) / 2.0)))))) > -998, (-((data["abs_maxbatch_msignal"]))), ((((-((data["signal_shift_-1_msignal"])))) + (data["abs_minbatch_msignal"]))/2.0) ))) ) > -998, (-((data["abs_maxbatch_msignal"]))), data["maxbatch_slices2"] )))) +

                            0.100000*np.tanh(((((data["signal_shift_-1"]) * 2.0)) * ((((data["meanbatch_msignal"]) + (np.cos(((((np.sin((np.cos((data["meanbatch_msignal"]))))) + (np.cos((np.sin((((data["signal_shift_-1"]) * 2.0)))))))/2.0)))))/2.0)))) +

                            0.100000*np.tanh(((np.sin(((-((np.where(np.sin((np.sin((np.sin(((-((np.where(data["abs_minbatch_slices2"] > -998, data["maxbatch_slices2_msignal"], (((7.03530025482177734)) * ((0.0))) )))))))))) > -998, data["maxbatch_slices2_msignal"], np.sin(((-((np.sin((data["signal_shift_-1_msignal"]))))))) ))))))) * 2.0)) +

                            0.100000*np.tanh(np.where((((data["minbatch_slices2"]) + (np.where(data["signal_shift_+1_msignal"] <= -998, (3.0), ((np.cos(((3.0)))) * 2.0) )))/2.0) <= -998, data["stdbatch_slices2"], np.sin(((((((4.0)) - (((np.cos((data["maxbatch_slices2_msignal"]))) * 2.0)))) * 2.0))) )) +

                            0.100000*np.tanh(np.where(data["medianbatch_slices2_msignal"] > -998, np.cos((np.where(data["medianbatch_slices2_msignal"] <= -998, np.tanh((data["mean_abs_chgbatch_msignal"])), data["medianbatch_slices2_msignal"] ))), (-((data["meanbatch_slices2_msignal"]))) )) +

                            0.100000*np.tanh((((((data["maxbatch_msignal"]) * (np.sin(((-((((data["minbatch_msignal"]) - ((((((data["abs_minbatch_msignal"]) + (data["maxbatch_msignal"]))/2.0)) / 2.0))))))))))) + (data["signal_shift_-1_msignal"]))/2.0)) +

                            0.100000*np.tanh(((np.where(data["abs_avgbatch_slices2_msignal"] > -998, np.cos((((data["medianbatch_msignal"]) * 2.0))), (((((data["medianbatch_msignal"]) + (((data["meanbatch_slices2_msignal"]) / 2.0)))) + (np.sin(((((((data["medianbatch_msignal"]) + (data["medianbatch_msignal"]))/2.0)) * 2.0)))))/2.0) )) * 2.0)) +

                            0.100000*np.tanh(((np.cos((data["medianbatch_msignal"]))) + (((np.cos((((np.where(((data["meanbatch_msignal"]) + (np.where(data["signal_shift_+1"] > -998, (((((3.0)) * 2.0)) * 2.0), ((data["mean_abs_chgbatch_slices2"]) * (((data["meanbatch_slices2"]) * 2.0))) ))) > -998, data["meanbatch_msignal"], (-((data["signal"]))) )) * 2.0)))) * 2.0)))) +

                            0.100000*np.tanh(((((((data["maxbatch_msignal"]) - ((((((9.0)) + (np.where((-1.0) <= -998, ((data["maxtominbatch_slices2_msignal"]) - (np.sin((np.tanh((data["meanbatch_slices2_msignal"])))))), ((np.cos((data["maxbatch_slices2_msignal"]))) - (np.sin(((((-2.0)) - ((((-(((1.0))))) / 2.0))))))) )))) / 2.0)))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(((((data["stdbatch_msignal"]) - (((data["maxtominbatch_slices2_msignal"]) - (((data["medianbatch_slices2_msignal"]) + (((np.cos((data["minbatch_slices2_msignal"]))) * (((data["maxbatch_slices2"]) - (data["minbatch_slices2_msignal"]))))))))))) * (np.tanh((data["medianbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh(((data["abs_maxbatch_slices2"]) * (np.where((3.0) <= -998, data["stdbatch_slices2"], np.cos((np.where((((((0.0)) + (((((np.cos((data["signal_shift_+1"]))) * 2.0)) * 2.0)))) * 2.0) > -998, data["maxbatch_msignal"], (-1.0) ))) )))) +

                            0.100000*np.tanh(np.where(data["mean_abs_chgbatch_msignal"] > -998, np.sin((((np.sin(((((data["minbatch_msignal"]) + (np.tanh((data["abs_maxbatch_slices2_msignal"]))))/2.0)))) * 2.0))), data["minbatch_msignal"] )) +

                            0.100000*np.tanh(((np.sin(((-((np.where(data["minbatch"] > -998, data["abs_maxbatch_slices2_msignal"], ((data["abs_maxbatch_slices2_msignal"]) * 2.0) ))))))) * 2.0)) +

                            0.100000*np.tanh(np.where(np.cos((data["abs_minbatch_slices2_msignal"])) <= -998, ((data["maxtominbatch"]) * (((((data["signal_shift_-1"]) * (((np.sin((data["rangebatch_slices2_msignal"]))) * (data["mean_abs_chgbatch_slices2"]))))) * 2.0))), ((np.cos((((data["medianbatch_slices2_msignal"]) * 2.0)))) * 2.0) )) +

                            0.100000*np.tanh(((((np.sin(((-((np.where(data["rangebatch_msignal"] > -998, data["abs_maxbatch_msignal"], np.where(data["abs_minbatch_slices2_msignal"] <= -998, ((data["abs_minbatch_msignal"]) + (((np.cos((np.tanh((data["medianbatch_msignal"]))))) * 2.0))), np.sin((np.cos((np.tanh((data["abs_avgbatch_msignal"])))))) ) ))))))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(np.where(data["meanbatch_slices2_msignal"] > -998, np.cos((data["abs_maxbatch_msignal"])), np.where((((((((6.66490459442138672)) * ((-(((-3.0))))))) / 2.0)) - (((data["rangebatch_msignal"]) + (((np.cos((data["maxbatch_slices2_msignal"]))) / 2.0))))) > -998, data["abs_avgbatch_slices2_msignal"], data["stdbatch_msignal"] ) )) +

                            0.100000*np.tanh(((data["maxbatch_slices2_msignal"]) * ((-((np.where(((data["maxbatch_slices2_msignal"]) * (data["maxbatch_msignal"])) > -998, np.cos((((data["maxbatch_slices2_msignal"]) + (np.where(data["abs_maxbatch"] > -998, data["abs_maxbatch_msignal"], (-((((((data["stdbatch_slices2_msignal"]) * 2.0)) * (data["maxbatch_slices2_msignal"]))))) ))))), data["maxbatch_slices2_msignal"] ))))))) +

                            0.100000*np.tanh(np.cos((np.where(data["medianbatch_slices2_msignal"] > -998, data["medianbatch_msignal"], np.sin(((((((data["maxtominbatch_slices2"]) + (data["maxbatch_slices2"]))/2.0)) - (data["maxtominbatch"])))) )))) +

                            0.100000*np.tanh(np.where(data["abs_avgbatch_slices2"] <= -998, data["abs_maxbatch_msignal"], (((-2.0)) * (np.where(np.where(((((np.cos((((data["stdbatch_slices2_msignal"]) * 2.0)))) * 2.0)) / 2.0) <= -998, data["medianbatch_slices2_msignal"], np.sin((((data["abs_maxbatch_msignal"]) * 2.0))) ) <= -998, ((np.sin((np.sin(((((-2.0)) * ((2.0)))))))) * 2.0), np.sin((data["minbatch_msignal"])) ))) )) +

                            0.100000*np.tanh(((((((data["medianbatch_msignal"]) + (data["medianbatch_slices2_msignal"]))) * (np.where((((-1.0)) - (np.tanh((data["abs_maxbatch"])))) > -998, data["abs_maxbatch_msignal"], np.cos((data["meanbatch_slices2_msignal"])) )))) - (data["abs_avgbatch_msignal"]))) +

                            0.100000*np.tanh(np.where(data["signal_shift_-1"] <= -998, data["signal"], ((data["meanbatch_msignal"]) * ((-((((data["medianbatch_msignal"]) + (data["medianbatch_msignal"]))))))) )) +

                            0.100000*np.tanh(np.where(data["maxtominbatch_slices2_msignal"] <= -998, np.sin((((((data["minbatch_slices2"]) / 2.0)) / 2.0))), (-(((-((np.cos((data["medianbatch_msignal"])))))))) )) +

                            0.100000*np.tanh(np.cos((data["minbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(((((data["mean_abs_chgbatch_msignal"]) * (np.where(((((((data["signal_shift_-1"]) * (data["mean_abs_chgbatch_msignal"]))) + (data["mean_abs_chgbatch_msignal"]))) * 2.0) > -998, ((data["minbatch"]) * 2.0), data["maxtominbatch"] )))) - (((((data["signal"]) * (np.sin((data["abs_maxbatch_slices2_msignal"]))))) + (data["rangebatch_slices2_msignal"]))))) +

                            0.100000*np.tanh(np.cos((data["minbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(((np.where((((-2.0)) + (((((np.tanh((data["signal_shift_-1_msignal"]))) + (data["signal_shift_-1"]))) * (((data["maxtominbatch_slices2_msignal"]) * (np.where(((np.cos(((0.0)))) * (data["maxtominbatch_slices2_msignal"])) <= -998, data["meanbatch_slices2_msignal"], data["rangebatch_slices2"] ))))))) <= -998, data["minbatch_slices2_msignal"], (8.41061878204345703) )) * (np.cos((data["maxbatch_msignal"]))))) +

                            0.100000*np.tanh((-((np.where(((data["signal_shift_+1_msignal"]) / 2.0) <= -998, ((((((np.where((-1.0) > -998, (10.0), ((data["meanbatch_msignal"]) * 2.0) )) - (np.cos((data["signal_shift_+1_msignal"]))))) - (np.where(data["mean_abs_chgbatch_msignal"] <= -998, data["rangebatch_msignal"], np.sin((data["minbatch"])) )))) / 2.0), np.sin((data["abs_maxbatch_msignal"])) ))))) +

                            0.100000*np.tanh(np.sin((((np.where(np.where(data["minbatch_msignal"] <= -998, ((data["medianbatch_slices2"]) + (data["maxbatch_slices2"])), data["minbatch_msignal"] ) <= -998, np.where(np.sin(((((3.0)) / 2.0))) > -998, data["minbatch_msignal"], np.sin((data["minbatch"])) ), np.where(data["rangebatch_slices2_msignal"] > -998, data["minbatch_msignal"], data["abs_avgbatch_slices2_msignal"] ) )) / 2.0)))) +

                            0.100000*np.tanh(np.where(np.sin(((((((((data["maxbatch_slices2_msignal"]) + (np.sin((data["maxbatch_slices2_msignal"]))))/2.0)) * 2.0)) * (np.sin((np.where((1.0) <= -998, (-2.0), ((np.sin((data["minbatch_msignal"]))) * 2.0) ))))))) <= -998, (((0.0)) / 2.0), ((np.sin((((data["minbatch_msignal"]) / 2.0)))) * 2.0) )) +

                            0.100000*np.tanh(np.sin((((data["maxbatch_slices2_msignal"]) - (np.where((-((data["maxtominbatch_slices2_msignal"]))) > -998, (3.0), data["abs_avgbatch_slices2_msignal"] )))))) +

                            0.100000*np.tanh(data["maxtominbatch_slices2"]) +

                            0.100000*np.tanh(((np.sin((((np.cos((data["stdbatch_slices2_msignal"]))) - (data["maxbatch_msignal"]))))) * 2.0)) +

                            0.100000*np.tanh(np.cos((((data["medianbatch_msignal"]) * ((((((np.cos((data["abs_maxbatch_slices2"]))) + (data["maxbatch_slices2"]))/2.0)) / 2.0)))))) +

                            0.100000*np.tanh(np.where(data["medianbatch_msignal"] <= -998, np.where(np.sin((((data["stdbatch_slices2"]) * 2.0))) > -998, (((((-2.0)) / 2.0)) * 2.0), data["rangebatch_slices2_msignal"] ), ((data["minbatch_slices2_msignal"]) / 2.0) )) +

                            0.100000*np.tanh(np.where(np.cos((np.where(np.cos((((data["maxbatch_slices2_msignal"]) + (np.cos((data["abs_avgbatch_slices2_msignal"])))))) <= -998, (2.0), data["meanbatch_slices2"] ))) > -998, data["meanbatch_slices2"], (-2.0) )) +

                            0.100000*np.tanh(np.cos((np.where(data["maxbatch_msignal"] > -998, data["abs_maxbatch_msignal"], data["medianbatch_slices2"] )))) +

                            0.100000*np.tanh(np.where(np.tanh((np.sin((np.tanh((data["medianbatch_slices2_msignal"])))))) <= -998, (-((np.cos((data["stdbatch_slices2"]))))), (-((np.cos((data["stdbatch_slices2"]))))) )) +

                            0.100000*np.tanh(np.where((-3.0) <= -998, np.sin((np.cos((np.where((-3.0) > -998, data["medianbatch_msignal"], np.where((2.0) > -998, data["stdbatch_slices2"], data["maxbatch_slices2_msignal"] ) ))))), ((np.cos((data["meanbatch_msignal"]))) + (((data["medianbatch_msignal"]) * 2.0))) )) +

                            0.100000*np.tanh(((((np.where(np.tanh((data["maxtominbatch_slices2_msignal"])) > -998, ((((np.sin(((((data["minbatch_msignal"]) + (np.sin(((((data["minbatch_msignal"]) + (((((np.tanh((data["abs_minbatch_slices2_msignal"]))) / 2.0)) / 2.0)))/2.0)))))/2.0)))) * 2.0)) * 2.0), data["abs_minbatch_slices2_msignal"] )) * 2.0)) * 2.0)) +

                            0.100000*np.tanh(((data["maxtominbatch"]) + (np.where(np.where((((data["maxtominbatch"]) + (((data["signal_shift_-1"]) + (data["maxtominbatch"]))))/2.0) <= -998, data["maxbatch_slices2_msignal"], data["stdbatch_slices2_msignal"] ) > -998, data["medianbatch_slices2"], (((-((data["medianbatch_slices2"])))) / 2.0) )))) +

                            0.100000*np.tanh(((data["meanbatch_slices2_msignal"]) - (data["abs_avgbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(np.where(((((((((np.sin((data["minbatch_slices2_msignal"]))) / 2.0)) - (data["minbatch_slices2"]))) + (np.tanh((((((data["minbatch_msignal"]) * 2.0)) + (data["abs_avgbatch_slices2"]))))))) * 2.0) > -998, ((np.sin((data["medianbatch_slices2_msignal"]))) * 2.0), ((np.sin((((((np.sin((data["medianbatch_slices2_msignal"]))) * 2.0)) / 2.0)))) * 2.0) )) +

                            0.100000*np.tanh(np.where((-((np.cos((((((data["abs_maxbatch"]) * 2.0)) * 2.0)))))) > -998, data["minbatch_msignal"], data["mean_abs_chgbatch_slices2"] )) +

                            0.100000*np.tanh(((np.where(data["signal_shift_-1_msignal"] <= -998, np.sin((data["abs_maxbatch_slices2"])), np.sin((np.where(data["minbatch_slices2"] > -998, data["abs_maxbatch"], np.where(np.where(data["meanbatch_slices2_msignal"] > -998, data["abs_maxbatch"], data["signal_shift_+1"] ) <= -998, np.sin((data["minbatch"])), data["medianbatch_slices2_msignal"] ) ))) )) * 2.0)) +

                            0.100000*np.tanh(np.cos((data["medianbatch_msignal"]))) +

                            0.100000*np.tanh((-((np.where(data["meanbatch_msignal"] > -998, ((data["meanbatch_msignal"]) * (data["meanbatch_msignal"])), data["meanbatch_slices2_msignal"] ))))) +

                            0.100000*np.tanh(((np.cos((data["medianbatch_msignal"]))) + (np.where(np.where(data["medianbatch_slices2"] > -998, np.where(np.where(data["maxtominbatch_slices2_msignal"] <= -998, (((data["mean_abs_chgbatch_slices2"]) + (data["medianbatch_msignal"]))/2.0), (1.0) ) > -998, data["medianbatch_msignal"], np.where(data["mean_abs_chgbatch_slices2"] <= -998, ((data["signal"]) - (data["medianbatch_msignal"])), ((data["maxbatch_slices2_msignal"]) / 2.0) ) ), data["mean_abs_chgbatch_slices2"] ) <= -998, data["abs_avgbatch_msignal"], data["medianbatch_msignal"] )))) +

                            0.100000*np.tanh(np.sin((np.sin((data["maxtominbatch_msignal"]))))) +

                            0.100000*np.tanh((((-((np.where(np.sin((data["stdbatch_msignal"])) <= -998, (((np.sin((data["maxbatch_msignal"]))) + (data["meanbatch_slices2"]))/2.0), np.sin((((data["minbatch_msignal"]) + (np.cos((data["maxtominbatch"])))))) ))))) * 2.0)) +

                            0.100000*np.tanh((((((data["meanbatch_slices2_msignal"]) + ((((((data["abs_maxbatch_slices2_msignal"]) + (np.sin((np.cos((np.sin((data["signal_shift_+1_msignal"]))))))))/2.0)) / 2.0)))/2.0)) * 2.0)) +

                            0.100000*np.tanh((-((np.where((-1.0) <= -998, ((np.cos((np.sin((data["minbatch_slices2_msignal"]))))) * (((((np.sin(((((((((((((np.sin((np.where(data["stdbatch_slices2_msignal"] <= -998, ((data["signal_shift_-1_msignal"]) / 2.0), data["rangebatch_msignal"] )))) / 2.0)) + (data["medianbatch_slices2_msignal"]))/2.0)) * 2.0)) / 2.0)) / 2.0)))) / 2.0)) * 2.0))), np.tanh((data["mean_abs_chgbatch_msignal"])) ))))) +

                            0.100000*np.tanh(np.where(((np.where(data["minbatch"] > -998, data["abs_minbatch_slices2_msignal"], (-(((1.0)))) )) * 2.0) > -998, np.cos(((2.0))), (3.0) )) +

                            0.100000*np.tanh(((((((((((((((-((data["abs_minbatch_msignal"])))) + ((((-1.0)) / 2.0)))/2.0)) + (np.cos((((data["medianbatch_msignal"]) / 2.0)))))/2.0)) + (data["maxtominbatch"]))/2.0)) * (np.cos((np.where(data["meanbatch_slices2_msignal"] > -998, data["abs_minbatch_msignal"], (0.0) )))))) / 2.0)) +

                            0.100000*np.tanh(np.where(data["abs_maxbatch_slices2_msignal"] <= -998, data["meanbatch_slices2"], ((data["stdbatch_msignal"]) + (data["meanbatch_slices2"])) )) +

                            0.100000*np.tanh(((data["rangebatch_msignal"]) - (np.where(data["meanbatch_slices2"] <= -998, np.cos((np.where(data["abs_maxbatch_msignal"] > -998, data["maxtominbatch_slices2_msignal"], np.where((0.0) <= -998, (7.0), np.where(np.tanh((data["abs_minbatch_slices2_msignal"])) <= -998, ((np.where(data["maxtominbatch"] <= -998, data["abs_minbatch_msignal"], (7.0) )) - ((7.0))), ((data["rangebatch_msignal"]) / 2.0) ) ) ))), data["abs_maxbatch"] )))) +

                            0.100000*np.tanh(((data["signal_shift_-1_msignal"]) * (np.where((((((data["signal_shift_-1_msignal"]) * 2.0)) + ((3.0)))/2.0) > -998, data["meanbatch_slices2_msignal"], np.where(data["meanbatch_slices2_msignal"] > -998, data["maxtominbatch_slices2_msignal"], ((data["meanbatch_slices2_msignal"]) * (data["medianbatch_slices2_msignal"])) ) )))) +

                            0.100000*np.tanh(np.cos((np.where(((data["minbatch_slices2"]) * (data["maxtominbatch_slices2_msignal"])) <= -998, np.where(data["abs_minbatch_slices2"] <= -998, np.cos((np.tanh((data["maxtominbatch_slices2"])))), data["stdbatch_msignal"] ), np.where((-((np.tanh((np.cos((data["maxtominbatch_msignal"]))))))) > -998, data["maxtominbatch_slices2_msignal"], (-((np.cos((data["maxtominbatch_msignal"]))))) ) )))) +

                            0.100000*np.tanh(np.sin(((((data["minbatch_msignal"]) + (data["abs_minbatch_slices2"]))/2.0)))) +

                            0.100000*np.tanh(np.where(data["signal"] > -998, ((data["abs_maxbatch_slices2"]) * (((data["signal_shift_-1_msignal"]) * (data["medianbatch_msignal"])))), np.sin((((data["meanbatch_slices2_msignal"]) * (((((np.tanh(((0.0)))) - (((data["minbatch_slices2"]) * ((((data["medianbatch_slices2"]) + (((((data["abs_avgbatch_slices2"]) - ((1.0)))) * 2.0)))/2.0)))))) / 2.0))))) )) +

                            0.100000*np.tanh((((((np.cos((data["abs_avgbatch_msignal"]))) + (np.where(((((((data["meanbatch_msignal"]) * 2.0)) * 2.0)) * (data["mean_abs_chgbatch_slices2"])) > -998, data["abs_avgbatch_slices2"], np.where(data["meanbatch_msignal"] <= -998, data["meanbatch_msignal"], data["abs_maxbatch_slices2_msignal"] ) )))/2.0)) * (np.cos((((data["meanbatch_msignal"]) * 2.0)))))) +

                            0.100000*np.tanh(np.where(data["signal_shift_-1_msignal"] <= -998, data["minbatch_msignal"], np.where((((-((data["signal_shift_-1_msignal"])))) * (np.where(np.tanh((np.where(data["stdbatch_slices2"] > -998, np.cos((data["minbatch_msignal"])), data["maxbatch_msignal"] ))) <= -998, data["minbatch_msignal"], data["signal_shift_+1"] ))) > -998, np.sin((((data["minbatch_msignal"]) / 2.0))), data["meanbatch_msignal"] ) )) +

                            0.100000*np.tanh(np.where(((np.where((2.0) > -998, np.cos(((-(((-((((data["minbatch_slices2_msignal"]) / 2.0))))))))), (-(((-1.0)))) )) / 2.0) > -998, ((np.cos((data["rangebatch_msignal"]))) * (data["maxtominbatch_slices2"])), (1.0) )) +

                            0.100000*np.tanh((-((np.sin((np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], ((((-((data["rangebatch_slices2_msignal"])))) + (((((data["mean_abs_chgbatch_msignal"]) / 2.0)) / 2.0)))/2.0) ))))))) +

                            0.100000*np.tanh(np.where(data["abs_avgbatch_slices2"] <= -998, (1.0), np.sin((np.where(data["abs_minbatch_slices2_msignal"] <= -998, data["maxbatch_slices2_msignal"], np.where((-2.0) <= -998, ((data["minbatch_msignal"]) + (data["minbatch_slices2"])), data["maxtominbatch_msignal"] ) ))) )) +

                            0.100000*np.tanh(((data["maxtominbatch_slices2"]) / 2.0)) +

                            0.100000*np.tanh(((data["rangebatch_slices2"]) * ((((((data["meanbatch_msignal"]) + (data["mean_abs_chgbatch_slices2"]))/2.0)) * (((data["signal_shift_+1_msignal"]) * (np.where(data["meanbatch_slices2"] > -998, np.where(data["maxbatch_slices2_msignal"] > -998, data["meanbatch_slices2_msignal"], ((np.where(data["rangebatch_slices2"] > -998, data["meanbatch_msignal"], data["signal_shift_+1_msignal"] )) * (np.where(data["signal_shift_+1_msignal"] > -998, data["meanbatch_msignal"], data["signal_shift_+1_msignal"] ))) ), (10.39286136627197266) )))))))) +

                            0.100000*np.tanh(np.cos((data["meanbatch_slices2"]))) +

                            0.100000*np.tanh(np.where(np.sin(((((-1.0)) * 2.0))) <= -998, data["meanbatch_msignal"], np.cos((data["maxtominbatch_slices2_msignal"])) )) +

                            0.100000*np.tanh(np.where(np.sin((np.sin((data["maxbatch_slices2"])))) <= -998, data["meanbatch_msignal"], np.sin((data["meanbatch_msignal"])) )) +

                            0.100000*np.tanh(np.sin((np.where(np.where(data["abs_maxbatch_msignal"] <= -998, np.cos((data["abs_maxbatch"])), np.sin((data["medianbatch_msignal"])) ) > -998, np.where(data["medianbatch_msignal"] > -998, data["meanbatch_slices2_msignal"], np.cos((np.sin((np.sin((np.cos((data["medianbatch_msignal"])))))))) ), data["medianbatch_slices2"] )))) +

                            0.100000*np.tanh(((data["mean_abs_chgbatch_msignal"]) * (np.where((((np.where(np.sin((data["stdbatch_slices2_msignal"])) > -998, (((((-3.0)) / 2.0)) * 2.0), data["mean_abs_chgbatch_slices2_msignal"] )) + (data["minbatch_slices2"]))/2.0) <= -998, np.sin((data["meanbatch_slices2_msignal"])), np.sin((data["maxbatch_msignal"])) )))) +

                            0.100000*np.tanh(((np.cos((data["minbatch_msignal"]))) * (((data["meanbatch_slices2_msignal"]) * (np.where((((((data["minbatch_msignal"]) + (data["stdbatch_msignal"]))/2.0)) * ((((np.cos((np.where(np.cos((data["meanbatch_slices2_msignal"])) > -998, data["minbatch_msignal"], data["meanbatch_msignal"] )))) + (data["minbatch_msignal"]))/2.0))) > -998, data["signal"], np.cos((np.sin((np.cos((data["minbatch_msignal"])))))) )))))) +

                            0.100000*np.tanh(np.tanh((((data["medianbatch_slices2_msignal"]) * (np.tanh(((-3.0)))))))) +

                            0.100000*np.tanh(((np.where(np.tanh(((((2.0)) + ((0.0))))) <= -998, np.tanh((((np.sin((np.sin((np.cos((data["meanbatch_msignal"]))))))) - (np.sin((data["signal_shift_-1_msignal"])))))), np.cos((data["meanbatch_msignal"])) )) * (np.cos((data["meanbatch_msignal"]))))) +

                            0.100000*np.tanh(np.where(np.sin((data["abs_avgbatch_slices2_msignal"])) <= -998, np.sin((data["rangebatch_slices2_msignal"])), np.cos(((((data["rangebatch_slices2"]) + (np.cos((np.where(data["abs_avgbatch_slices2_msignal"] > -998, np.cos(((2.0))), (-3.0) )))))/2.0))) )) +

                            0.100000*np.tanh(((((np.where(((data["signal_shift_-1_msignal"]) / 2.0) <= -998, np.tanh((data["rangebatch_slices2"])), ((data["signal_shift_-1_msignal"]) * (data["medianbatch_msignal"])) )) * 2.0)) - ((-((((data["signal_shift_-1_msignal"]) * (data["medianbatch_msignal"])))))))) +

                            0.100000*np.tanh(np.sin((((data["minbatch"]) - (((((-((np.sin((data["minbatch"])))))) + (np.where(data["signal_shift_-1_msignal"] > -998, data["abs_avgbatch_msignal"], (0.0) )))/2.0)))))) +

                            0.100000*np.tanh((((-((((data["meanbatch_msignal"]) * (((data["medianbatch_msignal"]) + ((-2.0))))))))) - (data["abs_avgbatch_msignal"]))) +

                            0.100000*np.tanh(np.sin((np.cos((np.where(np.tanh((np.sin((data["medianbatch_msignal"])))) > -998, data["abs_avgbatch_slices2_msignal"], ((np.sin((np.sin((((data["abs_avgbatch_msignal"]) * (data["abs_avgbatch_slices2_msignal"]))))))) * (data["abs_avgbatch_msignal"])) )))))) +

                            0.100000*np.tanh(((np.where(np.sin((np.sin((data["rangebatch_slices2_msignal"])))) <= -998, np.tanh(((-((data["maxtominbatch_slices2_msignal"]))))), data["stdbatch_msignal"] )) * (((((data["minbatch_slices2"]) * 2.0)) * 2.0)))) +

                            0.100000*np.tanh(((((((data["meanbatch_msignal"]) * (np.cos((data["meanbatch_msignal"]))))) * 2.0)) * (np.where(((((np.cos((((data["signal_shift_-1_msignal"]) * (np.cos(((((data["maxtominbatch"]) + (data["rangebatch_slices2"]))/2.0)))))))) * (data["mean_abs_chgbatch_msignal"]))) * (data["signal_shift_-1_msignal"])) <= -998, np.where((-3.0) <= -998, ((data["minbatch_msignal"]) / 2.0), data["signal_shift_-1_msignal"] ), data["signal_shift_-1_msignal"] )))) +

                            0.100000*np.tanh(np.cos((((data["abs_maxbatch_msignal"]) * (np.where(np.where(data["rangebatch_msignal"] > -998, data["mean_abs_chgbatch_msignal"], ((((data["maxtominbatch"]) - (((data["maxbatch_msignal"]) - (np.where((1.0) <= -998, (2.0), (3.0) )))))) + ((((data["meanbatch_slices2"]) + (np.cos((data["abs_maxbatch_msignal"]))))/2.0))) ) <= -998, data["minbatch_msignal"], data["mean_abs_chgbatch_slices2"] )))))) +

                            0.100000*np.tanh(np.sin(((-((np.where(((data["meanbatch_msignal"]) * 2.0) > -998, ((np.cos((np.sin((np.sin((np.tanh((np.tanh((data["stdbatch_slices2_msignal"]))))))))))) - (data["signal"])), np.cos((data["rangebatch_msignal"])) ))))))) +

                            0.100000*np.tanh(((data["signal_shift_-1_msignal"]) * (np.cos((np.where(np.where(((data["rangebatch_msignal"]) / 2.0) <= -998, (-3.0), data["medianbatch_msignal"] ) > -998, data["abs_maxbatch_slices2_msignal"], np.where(data["signal_shift_-1_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], ((((((data["rangebatch_msignal"]) * (np.cos(((-(((((12.71890544891357422)) * 2.0))))))))) + ((1.0)))) * 2.0) ) )))))) +

                            0.100000*np.tanh(np.where(data["signal_shift_-1_msignal"] > -998, np.where((-((data["minbatch"]))) > -998, ((((np.sin((((data["signal_shift_+1"]) + ((-((data["minbatch"])))))))) * 2.0)) * 2.0), ((np.sin((((((((-(((-((((data["signal_shift_+1"]) + ((-((data["minbatch"])))))))))))) + ((0.0)))/2.0)) / 2.0)))) * 2.0) ), data["signal_shift_+1"] )) +

                            0.100000*np.tanh(np.cos((np.where(((data["signal_shift_+1"]) * (np.where(((data["mean_abs_chgbatch_slices2"]) * (((data["rangebatch_slices2"]) + (data["meanbatch_msignal"])))) <= -998, (0.0), ((((np.sin((data["meanbatch_msignal"]))) / 2.0)) / 2.0) ))) > -998, ((((np.cos((np.tanh((np.cos((data["abs_minbatch_slices2"]))))))) / 2.0)) - ((11.25226688385009766))), data["meanbatch_msignal"] )))) +

                            0.100000*np.tanh(np.sin((np.cos((np.cos(((-((np.where((2.0) <= -998, np.sin((data["abs_minbatch_slices2"])), (((((((4.0)) - (np.where((2.0) <= -998, (-1.0), (((-((((data["maxtominbatch_slices2"]) * 2.0))))) - ((-((np.where(data["minbatch_slices2"] > -998, data["maxtominbatch"], data["abs_maxbatch_slices2"] )))))) )))) * 2.0)) * 2.0) ))))))))))) +

                            0.100000*np.tanh(np.cos((((np.where((((((data["minbatch"]) + (data["rangebatch_slices2"]))/2.0)) / 2.0) <= -998, np.tanh(((-(((-((data["abs_avgbatch_slices2"])))))))), ((data["abs_minbatch_msignal"]) * (((np.where(data["maxbatch_slices2"] <= -998, np.where(data["meanbatch_msignal"] > -998, data["rangebatch_slices2_msignal"], data["minbatch_slices2"] ), data["maxtominbatch"] )) / 2.0))) )) * 2.0)))) +

                            0.100000*np.tanh(np.cos((data["abs_maxbatch_slices2"]))) +

                            0.100000*np.tanh(((np.tanh((data["meanbatch_msignal"]))) * (((((data["signal_shift_+1_msignal"]) - (((data["meanbatch_msignal"]) - ((((data["signal_shift_+1_msignal"]) + ((((0.0)) * (((np.sin((data["signal_shift_-1_msignal"]))) - (((data["medianbatch_slices2_msignal"]) - (np.sin((data["maxtominbatch_slices2"]))))))))))/2.0)))))) - (data["meanbatch_msignal"]))))) +

                            0.100000*np.tanh(np.cos((np.where(data["stdbatch_slices2_msignal"] > -998, data["maxbatch_msignal"], np.where(((data["abs_avgbatch_msignal"]) - (((((data["signal_shift_+1"]) - ((0.0)))) - (data["signal_shift_-1_msignal"])))) <= -998, data["abs_maxbatch_slices2"], ((((np.cos((((np.cos((np.tanh(((2.0)))))) * 2.0)))) / 2.0)) - (data["maxbatch_msignal"])) ) )))) +

                            0.100000*np.tanh(np.where(((data["abs_maxbatch_slices2_msignal"]) / 2.0) <= -998, np.tanh((((data["abs_maxbatch_slices2_msignal"]) * 2.0))), ((np.cos((((((data["abs_maxbatch_slices2_msignal"]) * (data["mean_abs_chgbatch_slices2"]))) + (data["mean_abs_chgbatch_slices2"]))))) * 2.0) )) +

                            0.100000*np.tanh(np.where(data["rangebatch_slices2"] > -998, (-1.0), np.where(((data["meanbatch_slices2_msignal"]) / 2.0) <= -998, data["abs_avgbatch_msignal"], np.where(data["stdbatch_msignal"] <= -998, np.cos(((((-1.0)) * 2.0))), (-1.0) ) ) )) +

                            0.100000*np.tanh(np.where((-((data["minbatch_msignal"]))) > -998, ((((data["medianbatch_msignal"]) * ((-((data["minbatch_msignal"])))))) * (np.cos((np.where((11.26893234252929688) > -998, ((data["minbatch_msignal"]) * 2.0), data["minbatch_msignal"] ))))), ((data["maxbatch_slices2_msignal"]) * 2.0) )) +

                            0.100000*np.tanh(np.sin((((np.where(((data["abs_avgbatch_slices2_msignal"]) / 2.0) <= -998, ((data["signal_shift_-1"]) + ((-((data["rangebatch_slices2"]))))), (-((((np.where(data["abs_avgbatch_msignal"] > -998, data["medianbatch_slices2_msignal"], np.where(data["medianbatch_slices2_msignal"] > -998, data["signal_shift_-1"], data["maxbatch_slices2_msignal"] ) )) * 2.0)))) )) * 2.0)))) +

                            0.100000*np.tanh(((np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["abs_minbatch_slices2"], np.tanh((np.tanh((np.where((((-((data["abs_maxbatch_slices2_msignal"])))) + ((-2.0))) > -998, ((((-3.0)) + (np.where(data["signal_shift_+1_msignal"] <= -998, data["abs_maxbatch_msignal"], np.sin((data["signal_shift_+1_msignal"])) )))/2.0), data["mean_abs_chgbatch_slices2_msignal"] ))))) )) + (data["signal_shift_+1"]))) +

                            0.100000*np.tanh(((np.sin((np.where(((data["meanbatch_slices2"]) + ((4.0))) <= -998, ((np.sin((((((data["stdbatch_msignal"]) - (data["meanbatch_slices2"]))) * 2.0)))) + ((4.0))), ((((data["stdbatch_msignal"]) - (((data["meanbatch_slices2"]) + (((((-2.0)) + ((-3.0)))/2.0)))))) * 2.0) )))) * 2.0)) +

                            0.100000*np.tanh(((np.cos((np.where(((data["meanbatch_msignal"]) / 2.0) <= -998, data["mean_abs_chgbatch_slices2"], np.where(np.where(((np.tanh((data["abs_minbatch_slices2"]))) * 2.0) > -998, data["signal_shift_+1"], data["abs_minbatch_slices2"] ) > -998, data["mean_abs_chgbatch_msignal"], data["mean_abs_chgbatch_slices2"] ) )))) * 2.0)) +

                            0.100000*np.tanh(np.where(data["stdbatch_slices2_msignal"] > -998, (((data["rangebatch_msignal"]) + (data["maxtominbatch_msignal"]))/2.0), data["maxtominbatch_msignal"] )) +

                            0.100000*np.tanh((-((((((data["stdbatch_msignal"]) - (data["signal_shift_+1_msignal"]))) + (np.sin((np.where(np.cos((((data["abs_maxbatch"]) * (((data["maxbatch_msignal"]) - (data["signal_shift_+1_msignal"])))))) <= -998, ((data["abs_minbatch_slices2"]) / 2.0), (((0.0)) * 2.0) ))))))))) +

                            0.100000*np.tanh(((np.where(data["medianbatch_slices2"] > -998, np.where(data["meanbatch_slices2_msignal"] > -998, np.where((0.0) > -998, data["meanbatch_slices2_msignal"], (((-2.0)) / 2.0) ), data["meanbatch_slices2_msignal"] ), data["mean_abs_chgbatch_slices2_msignal"] )) * (np.where(data["rangebatch_msignal"] > -998, np.cos(((-((data["stdbatch_slices2_msignal"]))))), np.sin((data["stdbatch_slices2_msignal"])) )))) +

                            0.100000*np.tanh(np.sin((np.where(data["maxbatch_slices2_msignal"] <= -998, np.where(data["abs_maxbatch_slices2"] > -998, data["maxtominbatch"], (((-((np.sin((data["minbatch_msignal"])))))) * (np.sin(((3.0))))) ), data["mean_abs_chgbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh(np.sin((((np.cos((data["medianbatch_msignal"]))) * 2.0)))) +

                            0.100000*np.tanh((-((np.where(np.cos((data["abs_avgbatch_msignal"])) <= -998, (-((np.where(((((data["minbatch_slices2_msignal"]) / 2.0)) + (data["medianbatch_msignal"])) <= -998, np.cos((data["maxbatch_slices2_msignal"])), data["abs_maxbatch_msignal"] )))), np.cos((data["rangebatch_msignal"])) ))))) +

                            0.100000*np.tanh((((np.tanh((np.where(data["stdbatch_slices2"] > -998, np.where(np.where(data["abs_minbatch_slices2"] > -998, data["abs_minbatch_slices2"], ((((((-2.0)) / 2.0)) + (data["signal_shift_-1"]))/2.0) ) <= -998, data["medianbatch_slices2_msignal"], data["signal_shift_-1"] ), data["abs_avgbatch_msignal"] )))) + ((((data["stdbatch_slices2"]) + (np.sin(((0.0)))))/2.0)))/2.0)) +

                            0.100000*np.tanh(((np.where(np.sin(((-3.0))) <= -998, (-3.0), np.where(np.tanh((np.cos((data["minbatch_slices2_msignal"])))) <= -998, data["maxtominbatch_slices2_msignal"], ((np.sin((data["abs_avgbatch_slices2"]))) - (np.sin((np.cos((np.sin((np.sin((np.where((-2.0) > -998, np.sin((data["mean_abs_chgbatch_slices2_msignal"])), np.tanh((np.cos((data["medianbatch_msignal"])))) ))))))))))) ) )) / 2.0)) +

                            0.100000*np.tanh(np.where(np.cos((data["abs_maxbatch_msignal"])) > -998, ((data["signal_shift_+1"]) * (np.where(((((data["abs_maxbatch_msignal"]) * 2.0)) / 2.0) > -998, np.cos((data["abs_maxbatch_msignal"])), np.cos(((((2.0)) + (data["maxbatch_slices2_msignal"])))) ))), data["abs_maxbatch_msignal"] )) +

                            0.100000*np.tanh((((((-((np.cos((((np.where(np.sin((data["rangebatch_msignal"])) > -998, np.sin(((((12.10184764862060547)) / 2.0))), data["abs_minbatch_slices2"] )) * 2.0))))))) / 2.0)) * (np.sin(((((12.10184764862060547)) * 2.0)))))) +

                            0.100000*np.tanh(np.cos((np.cos((data["meanbatch_msignal"]))))) +

                            0.100000*np.tanh(np.tanh((((np.where(np.cos((((data["abs_avgbatch_msignal"]) * (np.cos((np.cos((np.where(data["abs_maxbatch_slices2_msignal"] > -998, np.sin((data["meanbatch_msignal"])), (0.0) ))))))))) <= -998, data["maxtominbatch"], np.sin((data["stdbatch_msignal"])) )) - (data["mean_abs_chgbatch_msignal"]))))) +

                            0.100000*np.tanh(((data["maxbatch_slices2_msignal"]) * (np.sin((((data["maxbatch_slices2_msignal"]) * ((((data["maxtominbatch_msignal"]) + (np.where(((np.tanh((((data["abs_maxbatch_slices2_msignal"]) - ((0.94611191749572754)))))) - (np.cos((data["abs_maxbatch_slices2_msignal"])))) > -998, np.sin((data["medianbatch_msignal"])), ((((data["maxtominbatch_slices2_msignal"]) / 2.0)) + (np.tanh((((data["medianbatch_msignal"]) * (data["maxtominbatch_slices2"])))))) )))/2.0)))))))) +

                            0.100000*np.tanh(np.sin((np.sin((data["mean_abs_chgbatch_msignal"]))))) +

                            0.100000*np.tanh(np.sin((np.sin((np.where(data["meanbatch_msignal"] > -998, data["minbatch"], np.cos(((-((((data["signal"]) - (((data["meanbatch_msignal"]) - (data["signal"]))))))))) )))))) +

                            0.100000*np.tanh(np.sin((np.sin((((np.sin((data["maxtominbatch_slices2_msignal"]))) - (np.tanh(((-((((data["abs_maxbatch"]) - (np.where((0.14157417416572571) <= -998, (2.0), (4.0) ))))))))))))))) +

                            0.100000*np.tanh(((np.where(((((data["abs_minbatch_slices2_msignal"]) * (((((data["meanbatch_slices2_msignal"]) * (np.cos((((data["medianbatch_msignal"]) * 2.0)))))) * 2.0)))) + (((data["maxbatch_slices2"]) * 2.0))) <= -998, data["minbatch"], ((((data["minbatch"]) * 2.0)) * (((((((data["medianbatch_msignal"]) * (data["maxbatch_slices2"]))) * (np.cos((data["maxbatch_slices2"]))))) * 2.0))) )) * 2.0)) +

                            0.100000*np.tanh(np.where(np.sin((((np.sin(((((((-((data["abs_maxbatch_msignal"])))) * 2.0)) * 2.0)))) * 2.0))) <= -998, (-(((((((-(((((((-(((((-((data["abs_maxbatch_msignal"])))) * 2.0))))) * 2.0)) * 2.0))))) * 2.0)) * 2.0)))), np.sin((np.sin(((((((-((data["abs_maxbatch_msignal"])))) * 2.0)) * 2.0))))) )) +

                            0.100000*np.tanh(np.cos((((data["abs_maxbatch"]) - (np.where((((((((((-((((((data["rangebatch_slices2_msignal"]) + ((-(((-((((data["abs_avgbatch_slices2"]) * (data["mean_abs_chgbatch_slices2_msignal"]))))))))))) * 2.0))))) * 2.0)) / 2.0)) * 2.0)) * 2.0) <= -998, data["minbatch"], ((data["maxtominbatch_msignal"]) * ((-3.0))) )))))) +

                            0.100000*np.tanh(((np.sin((np.where(data["signal_shift_+1"] <= -998, np.tanh((data["minbatch_msignal"])), np.tanh((data["signal_shift_-1_msignal"])) )))) / 2.0)) +

                            0.100000*np.tanh(np.sin(((((((-3.0)) * 2.0)) * (np.sin((np.cos((np.where(np.cos((((np.sin((data["signal_shift_+1_msignal"]))) * 2.0))) > -998, data["abs_maxbatch_msignal"], np.sin((data["stdbatch_msignal"])) )))))))))) +

                            0.100000*np.tanh(np.cos(((((3.0)) * (np.where(((((3.0)) + (data["minbatch_msignal"]))/2.0) > -998, (-((np.sin((((data["mean_abs_chgbatch_slices2"]) * (np.where((((3.0)) * 2.0) > -998, data["minbatch_msignal"], data["meanbatch_msignal"] )))))))), (((3.0)) * (data["minbatch_msignal"])) )))))) +

                            0.100000*np.tanh(((np.sin((np.where(np.cos((np.sin((np.cos((data["medianbatch_msignal"])))))) <= -998, (((np.tanh((((data["medianbatch_msignal"]) - (data["medianbatch_msignal"]))))) + ((0.0)))/2.0), np.cos((data["medianbatch_msignal"])) )))) * 2.0)) +

                            0.100000*np.tanh(np.where(np.where(np.cos((data["abs_maxbatch_slices2_msignal"])) > -998, np.sin((data["maxtominbatch"])), data["medianbatch_slices2"] ) > -998, np.where(data["medianbatch_msignal"] <= -998, data["medianbatch_slices2"], np.tanh((np.cos((data["abs_maxbatch_slices2_msignal"])))) ), data["meanbatch_slices2"] )) +

                            0.100000*np.tanh(np.sin(((((9.28414630889892578)) * (np.sin((np.sin((data["meanbatch_slices2_msignal"]))))))))) +

                            0.100000*np.tanh(np.cos((((np.cos((np.where(data["minbatch_slices2_msignal"] > -998, data["medianbatch_slices2"], ((np.sin(((-(((((-((((np.sin(((8.56348323822021484)))) - (np.cos((data["maxbatch_slices2_msignal"])))))))) - (np.cos((data["abs_avgbatch_slices2_msignal"])))))))))) * 2.0) )))) - (data["medianbatch_slices2"]))))) +

                            0.100000*np.tanh(np.sin((((np.where(np.cos((np.sin((data["rangebatch_slices2"])))) > -998, data["minbatch_msignal"], np.sin((np.sin((((np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["minbatch_msignal"], np.where((-2.0) > -998, ((data["maxbatch_slices2_msignal"]) / 2.0), np.where((-((data["minbatch_slices2_msignal"]))) > -998, data["minbatch_msignal"], data["minbatch_slices2_msignal"] ) ) )) / 2.0))))) )) / 2.0)))) +

                            0.100000*np.tanh(((((data["signal_shift_-1_msignal"]) * (np.where(np.tanh((np.where(np.sin((data["medianbatch_msignal"])) <= -998, (10.0), data["mean_abs_chgbatch_slices2"] ))) > -998, data["medianbatch_msignal"], ((((np.tanh((data["mean_abs_chgbatch_slices2"]))) + (data["rangebatch_msignal"]))) / 2.0) )))) / 2.0)) +

                            0.100000*np.tanh(((data["signal"]) * (((np.sin((np.cos((data["mean_abs_chgbatch_msignal"]))))) * (np.where(data["meanbatch_slices2_msignal"] > -998, data["maxbatch_msignal"], np.where((-((data["abs_avgbatch_msignal"]))) <= -998, np.cos((np.cos((np.tanh(((-3.0))))))), (0.0) ) )))))) +

                            0.100000*np.tanh((-(((((((-3.0)) - (data["abs_minbatch_msignal"]))) * (((data["medianbatch_slices2"]) * (np.where(np.sin((data["minbatch_msignal"])) <= -998, (-((np.cos((data["medianbatch_msignal"]))))), ((np.sin((np.sin((data["minbatch_msignal"]))))) * 2.0) ))))))))) +

                            0.100000*np.tanh(np.where((-1.0) > -998, (((np.sin(((0.0)))) + (np.tanh((np.sin(((((np.tanh(((((np.cos((data["maxtominbatch_slices2_msignal"]))) + (np.sin((data["mean_abs_chgbatch_slices2_msignal"]))))/2.0)))) + (np.cos(((-3.0)))))/2.0)))))))/2.0), np.sin((np.cos((np.sin((data["abs_minbatch_msignal"])))))) )) +

                            0.100000*np.tanh(((((((data["maxtominbatch"]) + (((((data["signal_shift_-1"]) / 2.0)) - (data["stdbatch_msignal"]))))) - (data["medianbatch_msignal"]))) + (np.cos((np.where((((((data["medianbatch_slices2_msignal"]) / 2.0)) + (((data["meanbatch_msignal"]) - (((data["minbatch_slices2_msignal"]) * 2.0)))))/2.0) > -998, data["minbatch_msignal"], data["minbatch_msignal"] )))))) +

                            0.100000*np.tanh((-((np.tanh((((data["signal_shift_-1_msignal"]) / 2.0))))))) +

                            0.100000*np.tanh(((np.where(((np.cos((np.cos((data["abs_maxbatch_msignal"]))))) * (((((data["abs_avgbatch_msignal"]) / 2.0)) / 2.0))) > -998, data["signal_shift_-1"], np.where(data["mean_abs_chgbatch_slices2"] <= -998, np.sin(((3.0))), data["signal_shift_-1"] ) )) + (((np.sin(((3.0)))) - (((data["abs_avgbatch_slices2_msignal"]) * (((data["abs_avgbatch_slices2_msignal"]) / 2.0)))))))) +

                            0.100000*np.tanh(np.cos((((((data["abs_minbatch_slices2_msignal"]) * (((data["medianbatch_slices2"]) / 2.0)))) - (np.where(np.sin((np.cos((data["medianbatch_slices2"])))) <= -998, np.cos((data["medianbatch_slices2"])), ((((data["medianbatch_slices2"]) + (data["rangebatch_slices2"]))) - (np.where(data["rangebatch_slices2"] <= -998, data["medianbatch_slices2_msignal"], np.cos((np.cos((data["rangebatch_slices2"])))) ))) )))))) +

                            0.100000*np.tanh(np.where(np.where(data["rangebatch_slices2_msignal"] <= -998, (3.0), data["medianbatch_msignal"] ) <= -998, data["meanbatch_msignal"], np.tanh((np.sin((data["medianbatch_msignal"])))) )) +

                            0.100000*np.tanh(np.where(data["meanbatch_msignal"] <= -998, ((data["meanbatch_msignal"]) * 2.0), np.cos(((((np.where(data["signal_shift_+1_msignal"] <= -998, np.tanh((np.where(((data["maxtominbatch_msignal"]) - (data["minbatch_slices2"])) <= -998, data["maxtominbatch_msignal"], (((data["minbatch_slices2"]) + (np.where(data["abs_maxbatch"] > -998, (-3.0), data["meanbatch_slices2"] )))/2.0) ))), data["meanbatch_msignal"] )) + (np.cos((data["maxtominbatch"]))))/2.0))) )) +

                            0.100000*np.tanh(np.where(np.tanh((((np.tanh((data["meanbatch_msignal"]))) / 2.0))) <= -998, data["medianbatch_msignal"], np.sin((data["rangebatch_slices2_msignal"])) )) +

                            0.100000*np.tanh(np.cos((((np.where(np.sin((data["maxtominbatch_slices2_msignal"])) > -998, data["medianbatch_slices2"], np.sin(((-3.0))) )) + (((np.where(data["abs_maxbatch"] <= -998, data["meanbatch_slices2"], data["meanbatch_slices2_msignal"] )) * 2.0)))))) +

                            0.100000*np.tanh(np.where(np.cos(((-((data["signal"]))))) <= -998, ((np.tanh((data["abs_maxbatch_slices2_msignal"]))) / 2.0), (((1.0)) - (((((-((data["signal_shift_-1"])))) + (np.where((-((np.cos((data["abs_maxbatch"]))))) <= -998, (2.0), data["signal"] )))/2.0))) )) +

                            0.100000*np.tanh(np.where(data["abs_minbatch_slices2"] <= -998, data["signal"], np.cos((((data["meanbatch_slices2_msignal"]) / 2.0))) )) +

                            0.100000*np.tanh((((0.0)) * (np.where(((data["abs_avgbatch_slices2_msignal"]) * ((((0.22312408685684204)) * (np.sin((((((np.tanh((data["abs_avgbatch_slices2_msignal"]))) / 2.0)) / 2.0))))))) <= -998, data["abs_minbatch_slices2"], np.sin((np.sin((np.tanh((((data["abs_minbatch_slices2_msignal"]) / 2.0))))))) )))) +

                            0.100000*np.tanh(((data["abs_minbatch_slices2_msignal"]) * (np.sin((np.where(np.sin((np.where(np.sin((np.where((((((-((data["abs_minbatch_slices2_msignal"])))) * 2.0)) - (data["abs_minbatch_slices2_msignal"])) > -998, ((((data["abs_minbatch_slices2_msignal"]) * (data["meanbatch_msignal"]))) / 2.0), data["abs_minbatch_slices2_msignal"] ))) > -998, data["maxbatch_slices2_msignal"], data["maxbatch_slices2_msignal"] ))) > -998, data["maxbatch_slices2_msignal"], data["maxbatch_slices2_msignal"] )))))) +

                            0.100000*np.tanh(((np.where(np.where(data["maxbatch_slices2_msignal"] <= -998, data["medianbatch_slices2_msignal"], np.sin((data["signal_shift_+1"])) ) <= -998, data["abs_minbatch_slices2_msignal"], np.sin((data["signal_shift_+1"])) )) / 2.0)) +

                            0.100000*np.tanh(np.cos((np.cos(((((-3.0)) + (data["abs_maxbatch_slices2"]))))))) +

                            0.100000*np.tanh(np.cos((np.cos((np.tanh(((-((np.sin((data["abs_minbatch_slices2"])))))))))))) +

                            0.100000*np.tanh(((np.tanh(((((((((3.0)) * (data["signal_shift_+1_msignal"]))) / 2.0)) / 2.0)))) / 2.0)) +

                            0.100000*np.tanh(((np.where(np.tanh((data["rangebatch_msignal"])) <= -998, data["abs_minbatch_msignal"], np.tanh((np.where((((((((-1.0)) * 2.0)) / 2.0)) / 2.0) <= -998, np.sin((np.sin((((data["medianbatch_msignal"]) * 2.0))))), np.sin(((((-((((data["medianbatch_msignal"]) * 2.0))))) * 2.0))) ))) )) * 2.0)) +

                            0.100000*np.tanh(np.sin((np.tanh((np.where(data["abs_minbatch_slices2"] > -998, ((np.tanh((data["maxtominbatch_slices2"]))) / 2.0), (((np.sin(((0.0)))) + (np.sin((((data["medianbatch_slices2_msignal"]) * (np.where(data["abs_minbatch_slices2_msignal"] > -998, np.sin((((data["maxbatch_slices2_msignal"]) * 2.0))), (0.0) )))))))/2.0) )))))) +

                            0.100000*np.tanh(np.tanh((((data["stdbatch_slices2"]) - (data["meanbatch_msignal"]))))) +

                            0.100000*np.tanh(np.cos((np.where(np.sin((np.cos((((((np.cos((((np.tanh((data["meanbatch_msignal"]))) - (((data["abs_avgbatch_slices2_msignal"]) * 2.0)))))) / 2.0)) - (np.where(np.cos(((1.0))) <= -998, data["abs_minbatch_msignal"], data["abs_avgbatch_slices2_msignal"] ))))))) <= -998, data["abs_minbatch_msignal"], ((data["abs_avgbatch_slices2_msignal"]) * 2.0) )))) +

                            0.100000*np.tanh(np.where((2.0) <= -998, np.tanh((np.cos((data["minbatch_msignal"])))), np.tanh((np.sin(((((np.cos((np.cos((((np.cos((((np.cos((data["meanbatch_slices2_msignal"]))) * 2.0)))) / 2.0)))))) + (np.cos((np.cos((data["mean_abs_chgbatch_msignal"]))))))/2.0))))) )) +

                            0.100000*np.tanh(((data["medianbatch_slices2_msignal"]) * (np.sin((((data["maxbatch_msignal"]) + (np.sin((((np.where(((np.tanh(((1.0)))) + (np.sin(((0.32830962538719177))))) > -998, (0.32830962538719177), data["medianbatch_slices2_msignal"] )) + (np.tanh(((((1.0)) * ((-3.0)))))))))))))))) +

                            0.100000*np.tanh((0.0)) +

                            0.100000*np.tanh(np.sin((np.where(((((np.where(data["abs_maxbatch_slices2"] <= -998, data["rangebatch_msignal"], data["stdbatch_slices2"] )) / 2.0)) * (((data["signal_shift_+1"]) * 2.0))) <= -998, ((data["maxtominbatch"]) / 2.0), ((data["meanbatch_slices2_msignal"]) * (data["medianbatch_msignal"])) )))) +

                            0.099609*np.tanh(np.sin((np.where(np.cos((np.where((-((np.tanh((data["signal_shift_+1"]))))) <= -998, ((data["meanbatch_slices2"]) / 2.0), np.where(data["meanbatch_slices2_msignal"] > -998, ((data["minbatch_msignal"]) * (data["maxbatch_msignal"])), np.tanh((data["abs_minbatch_slices2"])) ) ))) > -998, ((data["minbatch_msignal"]) * (data["maxbatch_msignal"])), np.tanh((data["maxtominbatch"])) )))) +

                            0.100000*np.tanh(np.tanh((((np.sin((((data["medianbatch_slices2_msignal"]) * ((((data["mean_abs_chgbatch_slices2"]) + (data["medianbatch_msignal"]))/2.0)))))) * (data["medianbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh((((data["abs_maxbatch_slices2_msignal"]) + (np.where(data["maxbatch_slices2_msignal"] <= -998, data["meanbatch_slices2_msignal"], (-3.0) )))/2.0)) +

                            0.100000*np.tanh(np.tanh((((data["medianbatch_slices2_msignal"]) * (np.sin((np.sin((np.sin((np.sin((data["signal_shift_+1"]))))))))))))) +

                            0.100000*np.tanh(np.where((((-((np.tanh((((np.tanh((np.cos(((0.0)))))) * 2.0))))))) + (np.sin((np.cos((np.tanh(((((((1.0)) * (data["maxtominbatch_msignal"]))) - (np.sin(((1.0))))))))))))) <= -998, np.cos((np.cos((data["medianbatch_slices2_msignal"])))), (((data["stdbatch_slices2"]) + ((-1.0)))/2.0) )) +

                            0.100000*np.tanh(np.tanh(((1.0)))) +

                            0.100000*np.tanh(np.where((-((np.cos((((data["medianbatch_msignal"]) + (np.sin(((-3.0)))))))))) > -998, data["abs_minbatch_slices2"], np.cos((((data["medianbatch_msignal"]) + (data["abs_minbatch_slices2"])))) )) +

                            0.100000*np.tanh(np.where(data["meanbatch_slices2_msignal"] > -998, np.sin((data["meanbatch_slices2_msignal"])), data["meanbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(((((((((np.sin((data["abs_minbatch_slices2"]))) / 2.0)) * ((0.0)))) * 2.0)) * ((0.0)))) +

                            0.100000*np.tanh(np.where(((np.where((-((np.tanh((data["abs_maxbatch_slices2"]))))) > -998, (0.0), ((data["minbatch_slices2"]) * 2.0) )) + (data["maxtominbatch_slices2_msignal"])) <= -998, np.where(data["abs_maxbatch_slices2"] <= -998, data["abs_maxbatch"], (0.0) ), (0.0) )) +

                            0.100000*np.tanh(((np.cos((((data["abs_avgbatch_slices2_msignal"]) / 2.0)))) / 2.0)) +

                            0.089932*np.tanh(((np.sin((np.cos((data["minbatch_msignal"]))))) / 2.0)) +

                            0.100000*np.tanh(((((np.where(((data["mean_abs_chgbatch_slices2"]) / 2.0) > -998, np.where(((np.sin(((-((((data["rangebatch_slices2"]) / 2.0))))))) + (data["meanbatch_msignal"])) <= -998, np.where((-((np.cos((data["maxbatch_slices2"]))))) > -998, np.tanh((data["maxbatch_msignal"])), data["meanbatch_msignal"] ), ((data["stdbatch_slices2"]) / 2.0) ), data["abs_maxbatch_msignal"] )) * ((1.0)))) + (data["abs_avgbatch_msignal"]))) +

                            0.098436*np.tanh(((data["abs_minbatch_slices2_msignal"]) * (np.where(data["abs_minbatch_slices2_msignal"] > -998, np.where(((data["abs_minbatch_slices2_msignal"]) * (data["abs_minbatch_slices2_msignal"])) > -998, np.where(data["mean_abs_chgbatch_slices2"] > -998, np.cos((data["maxtominbatch_slices2_msignal"])), (1.30311763286590576) ), ((((((data["abs_avgbatch_slices2"]) / 2.0)) - (data["abs_minbatch_slices2"]))) / 2.0) ), ((np.cos((data["maxtominbatch_slices2_msignal"]))) / 2.0) )))) +

                            0.100000*np.tanh(((data["abs_maxbatch_slices2_msignal"]) + (((((data["signal_shift_-1"]) / 2.0)) - (np.where(((data["maxbatch_slices2_msignal"]) + (data["stdbatch_slices2"])) <= -998, data["abs_avgbatch_msignal"], np.sin((data["abs_maxbatch_msignal"])) )))))) +

                            0.100000*np.tanh(np.cos((np.where(((np.where(np.sin((np.cos(((3.0))))) <= -998, data["maxbatch_slices2"], ((data["mean_abs_chgbatch_msignal"]) - (data["abs_avgbatch_slices2"])) )) + (((data["abs_maxbatch"]) * (data["maxbatch_slices2"])))) <= -998, ((np.where((-2.0) <= -998, data["mean_abs_chgbatch_slices2"], data["maxbatch_slices2"] )) / 2.0), (((data["maxbatch_slices2"]) + (data["abs_maxbatch"]))/2.0) )))) +

                            0.100000*np.tanh(np.where(np.where(((data["abs_maxbatch_slices2"]) + (data["mean_abs_chgbatch_slices2"])) <= -998, ((np.sin((((((2.0)) + (((np.sin((((np.tanh((np.tanh((((((np.tanh((data["maxbatch_msignal"]))) + (data["meanbatch_msignal"]))) + (data["rangebatch_msignal"]))))))) / 2.0)))) / 2.0)))/2.0)))) * 2.0), data["maxbatch_msignal"] ) > -998, np.cos((data["meanbatch_msignal"])), data["signal_shift_+1"] )) +

                            0.100000*np.tanh(((data["medianbatch_slices2_msignal"]) / 2.0)) +

                            0.100000*np.tanh(np.where(data["signal_shift_-1"] <= -998, (2.0), np.where(np.cos((np.tanh(((((((((data["stdbatch_msignal"]) + ((((np.sin((data["signal_shift_-1"]))) + (data["signal_shift_+1"]))/2.0)))/2.0)) * (((data["signal_shift_+1"]) + (np.where(data["abs_maxbatch_slices2"] <= -998, (2.0), data["signal_shift_-1"] )))))) + (data["signal_shift_+1"])))))) > -998, data["signal_shift_-1"], (2.0) ) )) +

                            0.100000*np.tanh(np.where((((((np.cos((data["maxbatch_msignal"]))) * ((0.0)))) + (np.cos((data["stdbatch_msignal"]))))/2.0) > -998, (((((np.cos((data["maxbatch_msignal"]))) * (data["signal_shift_-1_msignal"]))) + (np.tanh(((((data["maxbatch_msignal"]) + ((0.0)))/2.0)))))/2.0), data["maxbatch_msignal"] )) +

                            0.100000*np.tanh(np.where(((data["signal_shift_-1"]) * 2.0) > -998, (((0.0)) * ((0.0))), ((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0) )) +

                            0.093744*np.tanh(np.where(data["maxtominbatch"] > -998, np.sin(((((data["meanbatch_msignal"]) + (np.tanh((data["meanbatch_msignal"]))))/2.0))), np.sin((np.sin((np.where(data["meanbatch_msignal"] > -998, (-((data["abs_maxbatch_slices2_msignal"]))), np.where(data["stdbatch_slices2"] > -998, (((data["medianbatch_msignal"]) + (data["meanbatch_msignal"]))/2.0), np.tanh((np.tanh((data["signal_shift_-1"])))) ) ))))) )) +

                            0.100000*np.tanh(np.cos((((data["medianbatch_slices2_msignal"]) - ((2.0)))))) +

                            0.099707*np.tanh(((data["signal"]) * (np.where(np.where((1.0) <= -998, np.tanh((((data["abs_maxbatch_msignal"]) * 2.0))), data["minbatch_slices2_msignal"] ) <= -998, np.where(data["medianbatch_slices2_msignal"] <= -998, np.cos((data["signal"])), data["medianbatch_slices2_msignal"] ), np.cos((np.where(data["medianbatch_msignal"] > -998, data["medianbatch_msignal"], ((data["medianbatch_msignal"]) * 2.0) ))) )))) +

                            0.100000*np.tanh(np.where(((np.tanh((data["mean_abs_chgbatch_msignal"]))) * (np.cos((np.where((0.0) <= -998, (0.0), (0.0) ))))) > -998, ((((np.where((-2.0) <= -998, np.sin((data["signal_shift_+1_msignal"])), np.tanh(((-((data["medianbatch_slices2_msignal"]))))) )) * 2.0)) / 2.0), (0.0) )) +

                            0.100000*np.tanh(data["abs_maxbatch_msignal"]) +

                            0.100000*np.tanh(np.tanh((np.cos((np.tanh(((-(((((data["mean_abs_chgbatch_slices2_msignal"]) + (np.cos((np.sin(((-((data["rangebatch_slices2_msignal"])))))))))/2.0))))))))))))   

        

    def GP_class_8(self,data):

        return self.Output( -3.014039 +

                            0.100000*np.tanh(((data["medianbatch_slices2"]) + (np.where(np.sin(((((np.tanh(((((((-((np.sin((data["stdbatch_msignal"])))))) * (np.cos((data["abs_maxbatch"]))))) - (data["abs_maxbatch_slices2"]))))) + (np.tanh(((((1.0)) - (np.tanh((data["mean_abs_chgbatch_slices2"]))))))))/2.0))) > -998, data["signal_shift_+1"], data["signal_shift_-1"] )))) +

                            0.100000*np.tanh(np.where(data["maxtominbatch_msignal"] <= -998, (((data["mean_abs_chgbatch_msignal"]) + (data["medianbatch_slices2"]))/2.0), data["medianbatch_slices2"] )) +

                            0.100000*np.tanh(np.where(np.where(((data["signal"]) / 2.0) > -998, (((data["medianbatch_slices2_msignal"]) + (((np.cos((data["medianbatch_slices2_msignal"]))) * (np.where(np.cos((data["stdbatch_slices2"])) > -998, data["signal"], np.cos((((data["stdbatch_slices2"]) / 2.0))) )))))/2.0), np.cos(((8.0))) ) > -998, data["signal_shift_+1"], np.sin((data["signal_shift_-1"])) )) +

                            0.100000*np.tanh((-((((data["abs_avgbatch_msignal"]) * 2.0))))) +

                            0.100000*np.tanh((-(((((-((((data["signal"]) + (((((data["signal"]) / 2.0)) - ((-1.0))))))))) + (((((11.33666324615478516)) + (data["meanbatch_slices2"]))/2.0))))))) +

                            0.100000*np.tanh(((np.where(data["abs_maxbatch"] <= -998, ((np.where(data["signal_shift_+1"] <= -998, (2.0), (-(((-1.0)))) )) + (data["signal_shift_-1"])), (3.0) )) * ((((data["medianbatch_slices2"]) + ((-3.0)))/2.0)))) +

                            0.100000*np.tanh(np.where(data["stdbatch_msignal"] <= -998, data["abs_maxbatch_slices2_msignal"], (-((np.where(np.where(np.sin((np.where((-2.0) > -998, np.where(data["minbatch_slices2"] > -998, np.cos((data["abs_maxbatch"])), data["maxtominbatch_msignal"] ), data["signal_shift_-1"] ))) <= -998, data["abs_avgbatch_msignal"], data["signal_shift_-1"] ) <= -998, ((data["rangebatch_slices2_msignal"]) - (data["abs_avgbatch_msignal"])), data["abs_avgbatch_msignal"] )))) )) +

                            0.100000*np.tanh(((((((((np.tanh((data["mean_abs_chgbatch_slices2_msignal"]))) + (data["mean_abs_chgbatch_msignal"]))) - (data["medianbatch_slices2_msignal"]))) + (data["mean_abs_chgbatch_msignal"]))) * (np.sin((np.where(np.sin(((-2.0))) > -998, data["abs_maxbatch_msignal"], np.sin((((((data["minbatch_msignal"]) / 2.0)) * 2.0))) )))))) +

                            0.100000*np.tanh((((((-((np.where(data["maxtominbatch_slices2"] > -998, data["meanbatch_msignal"], data["rangebatch_slices2"] ))))) / 2.0)) * (np.where(data["meanbatch_slices2"] > -998, data["signal_shift_+1"], np.where(data["maxtominbatch_slices2"] > -998, data["meanbatch_msignal"], (-((((np.where(data["maxtominbatch_slices2"] > -998, (-2.0), data["meanbatch_msignal"] )) - ((((2.0)) * (np.sin((data["maxbatch_slices2_msignal"]))))))))) ) )))) +

                            0.100000*np.tanh(((data["minbatch"]) * (np.where(data["maxtominbatch"] > -998, np.sin((data["abs_avgbatch_slices2"])), np.sin((data["abs_avgbatch_slices2"])) )))) +

                            0.100000*np.tanh((((data["signal"]) + (data["minbatch_slices2"]))/2.0)) +

                            0.100000*np.tanh(((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, (((-3.0)) * (data["abs_maxbatch_slices2_msignal"])), (-2.0) )) - (((np.cos(((2.0)))) * (data["medianbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh(((((((np.sin(((((data["minbatch_msignal"]) + (((np.sin((np.sin(((3.0)))))) * 2.0)))/2.0)))) * 2.0)) * 2.0)) - (data["abs_minbatch_slices2"]))) +

                            0.100000*np.tanh(np.where(np.sin((data["minbatch_slices2"])) > -998, (-((data["meanbatch_slices2_msignal"]))), np.where(np.where((-(((-((data["meanbatch_slices2_msignal"])))))) > -998, data["mean_abs_chgbatch_slices2"], data["meanbatch_slices2"] ) > -998, ((np.where((-1.0) > -998, data["maxbatch_msignal"], data["rangebatch_slices2_msignal"] )) * 2.0), data["abs_maxbatch"] ) )) +

                            0.100000*np.tanh((((np.where(data["maxbatch_slices2"] > -998, data["signal"], np.where(data["maxbatch_slices2"] <= -998, (7.0), (((((data["signal_shift_+1"]) - (np.tanh((data["signal_shift_+1"]))))) + (((data["medianbatch_slices2"]) + (np.sin((((data["maxbatch_slices2"]) / 2.0)))))))/2.0) ) )) + (data["medianbatch_slices2"]))/2.0)) +

                            0.100000*np.tanh(((((data["medianbatch_msignal"]) * (np.cos((np.where((-2.0) > -998, np.tanh((data["abs_maxbatch_slices2"])), (((((data["medianbatch_msignal"]) + (np.cos(((-1.0)))))/2.0)) * (((data["abs_maxbatch"]) * (data["abs_minbatch_msignal"])))) )))))) * (data["abs_minbatch_msignal"]))) +

                            0.100000*np.tanh(((((data["medianbatch_msignal"]) * (((data["meanbatch_msignal"]) + (np.where(data["mean_abs_chgbatch_msignal"] <= -998, ((data["medianbatch_slices2"]) / 2.0), (((5.0)) * ((-((((data["stdbatch_slices2"]) * 2.0)))))) )))))) - (((data["medianbatch_msignal"]) + (data["abs_minbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh((((11.49825382232666016)) * (np.sin(((-((np.cos((np.where(np.cos((np.where(((np.tanh(((((-2.0)) * 2.0)))) / 2.0) > -998, data["signal_shift_-1"], (11.49825382232666016) ))) > -998, data["stdbatch_slices2"], data["stdbatch_slices2_msignal"] ))))))))))) +

                            0.100000*np.tanh(((data["abs_maxbatch_msignal"]) * (np.cos(((((np.where(np.cos((np.sin(((((-1.0)) + (data["meanbatch_msignal"])))))) <= -998, ((np.cos((np.cos((data["signal_shift_-1"]))))) * 2.0), data["minbatch_msignal"] )) + ((-3.0)))/2.0)))))) +

                            0.100000*np.tanh((-((np.sin((np.where((0.49771082401275635) > -998, data["maxbatch_msignal"], ((((-1.0)) + (np.where(((((data["stdbatch_slices2"]) * 2.0)) + ((2.0))) > -998, data["maxbatch_msignal"], np.cos((((np.where(np.tanh(((0.49771082401275635))) > -998, (((0.49771082401275635)) * 2.0), (((((0.49771082401275635)) + (data["signal"]))) * 2.0) )) / 2.0))) )))/2.0) ))))))) +

                            0.100000*np.tanh((((((-3.0)) + (((data["meanbatch_msignal"]) * ((-3.0)))))) * (np.sin((data["meanbatch_msignal"]))))) +

                            0.100000*np.tanh(np.where(((data["maxbatch_slices2_msignal"]) * 2.0) > -998, (2.0), data["rangebatch_slices2"] )) +

                            0.100000*np.tanh(((np.sin(((((-((data["abs_minbatch_slices2_msignal"])))) + (data["abs_maxbatch_slices2"]))))) - (np.where(data["abs_avgbatch_slices2_msignal"] <= -998, (0.0), (-((np.where((-1.0) > -998, np.sin((data["abs_minbatch_slices2_msignal"])), np.tanh(((-((np.tanh(((-((((data["signal"]) + (np.tanh(((-3.0))))))))))))))) )))) )))) +

                            0.100000*np.tanh(((data["abs_minbatch_slices2_msignal"]) * (np.cos((np.where((((0.0)) * (((((((data["stdbatch_slices2_msignal"]) * (np.cos((np.where(np.cos((data["abs_maxbatch_slices2_msignal"])) > -998, data["abs_maxbatch_slices2_msignal"], data["rangebatch_slices2"] )))))) * 2.0)) / 2.0))) > -998, data["stdbatch_slices2"], ((np.where(data["abs_maxbatch_slices2"] > -998, ((data["abs_maxbatch_slices2_msignal"]) * (data["abs_minbatch_slices2_msignal"])), data["stdbatch_slices2"] )) / 2.0) )))))) +

                            0.100000*np.tanh(((data["maxtominbatch_slices2_msignal"]) * (np.cos((np.where(((data["minbatch_slices2_msignal"]) * (np.where(data["abs_minbatch_slices2"] <= -998, data["rangebatch_msignal"], (10.0) ))) > -998, data["stdbatch_msignal"], data["minbatch_msignal"] )))))) +

                            0.100000*np.tanh(((data["medianbatch_msignal"]) * (((((data["abs_avgbatch_slices2"]) * 2.0)) * (np.cos(((((data["abs_maxbatch_slices2_msignal"]) + (np.cos((np.tanh((((np.cos((np.where(((data["abs_maxbatch_slices2_msignal"]) * (data["maxtominbatch_slices2_msignal"])) <= -998, ((data["abs_minbatch_slices2_msignal"]) * (data["abs_avgbatch_slices2"])), data["abs_maxbatch_slices2_msignal"] )))) * (np.cos((data["stdbatch_slices2_msignal"]))))))))))/2.0)))))))) +

                            0.100000*np.tanh(((data["medianbatch_msignal"]) * ((-3.0)))) +

                            0.100000*np.tanh(((np.cos((data["abs_maxbatch_slices2_msignal"]))) * ((((-2.0)) - ((((((-1.0)) + (data["rangebatch_slices2_msignal"]))) - (np.tanh(((-((data["abs_maxbatch_slices2_msignal"])))))))))))) +

                            0.100000*np.tanh(np.tanh(((-((np.tanh((np.where(((data["minbatch"]) * ((2.0))) > -998, (((data["medianbatch_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))/2.0), (-((np.sin(((-((np.where(data["rangebatch_slices2"] > -998, data["abs_minbatch_slices2"], data["signal_shift_+1_msignal"] ))))))))) ))))))))) +

                            0.100000*np.tanh(np.where((-((np.where((((-3.0)) + (data["minbatch_msignal"])) > -998, (0.0), np.where((-3.0) > -998, (((-((data["minbatch_msignal"])))) - ((7.0))), ((((data["abs_minbatch_slices2_msignal"]) * (data["maxtominbatch"]))) - ((0.0))) ) )))) > -998, (((-((data["minbatch_msignal"])))) - ((7.0))), (0.0) )) +

                            0.100000*np.tanh(((data["maxbatch_msignal"]) - (np.where((((data["abs_maxbatch_slices2_msignal"]) + (np.where((((np.sin((np.cos((((data["maxbatch_msignal"]) - (data["maxbatch_msignal"]))))))) + (((data["maxbatch_msignal"]) + (np.cos((data["maxtominbatch_msignal"]))))))/2.0) <= -998, data["maxbatch_msignal"], np.cos(((1.50217688083648682))) )))/2.0) > -998, (3.0), data["maxbatch_msignal"] )))) +

                            0.100000*np.tanh(np.cos((((data["maxbatch_msignal"]) + (np.where((((((-3.0)) / 2.0)) / 2.0) > -998, (((3.0)) / 2.0), data["maxbatch_msignal"] )))))) +

                            0.100000*np.tanh(((((np.sin(((-3.0)))) - (data["mean_abs_chgbatch_slices2_msignal"]))) - (((data["stdbatch_slices2_msignal"]) * 2.0)))) +

                            0.100000*np.tanh((-(((4.0))))) +

                            0.100000*np.tanh(((data["minbatch_slices2_msignal"]) * (np.sin((((data["abs_maxbatch_msignal"]) * (np.where(np.where(data["rangebatch_slices2"] > -998, (1.0), (((np.sin((data["minbatch_slices2_msignal"]))) + ((3.49070620536804199)))/2.0) ) > -998, (1.0), (((-1.0)) * ((-1.0))) )))))))) +

                            0.100000*np.tanh((((-((np.cos((np.where(((data["signal_shift_-1_msignal"]) * (((((-2.0)) + (np.cos((np.where((((2.0)) * (data["signal_shift_-1"])) <= -998, (3.0), data["maxbatch_slices2_msignal"] )))))/2.0))) <= -998, (-((data["medianbatch_msignal"]))), data["minbatch_msignal"] ))))))) * 2.0)) +

                            0.100000*np.tanh(((np.tanh((((np.tanh((np.cos((data["abs_avgbatch_slices2"]))))) - (np.where(((np.sin((np.where(data["mean_abs_chgbatch_slices2"] <= -998, (2.0), data["abs_maxbatch_msignal"] )))) * 2.0) > -998, np.sin((data["maxbatch_msignal"])), data["maxtominbatch_slices2"] )))))) * 2.0)) +

                            0.100000*np.tanh((((((((2.0)) * (data["meanbatch_slices2_msignal"]))) * (np.sin((data["abs_maxbatch_msignal"]))))) * 2.0)) +

                            0.100000*np.tanh((((((-((data["maxbatch_slices2_msignal"])))) / 2.0)) + ((-((((((((data["meanbatch_slices2_msignal"]) * (((((data["abs_maxbatch_slices2_msignal"]) * 2.0)) + (np.cos((np.where(data["minbatch_slices2"] <= -998, ((data["abs_maxbatch_msignal"]) * 2.0), (-(((((-(((-(((-1.0)))))))) / 2.0)))) )))))))) / 2.0)) * 2.0))))))) +

                            0.100000*np.tanh(np.sin(((((data["minbatch_msignal"]) + (data["medianbatch_msignal"]))/2.0)))) +

                            0.100000*np.tanh(((data["abs_maxbatch_msignal"]) * (np.cos((np.where(np.cos((((np.tanh((((data["abs_maxbatch_msignal"]) / 2.0)))) * 2.0))) > -998, ((data["abs_maxbatch_slices2_msignal"]) * 2.0), np.sin((((((data["abs_maxbatch_msignal"]) * 2.0)) * 2.0))) )))))) +

                            0.100000*np.tanh(((np.where(np.tanh((data["maxbatch_msignal"])) <= -998, data["minbatch_msignal"], np.tanh((data["stdbatch_msignal"])) )) * (np.where(((data["meanbatch_msignal"]) * (data["minbatch_slices2"])) <= -998, ((data["rangebatch_msignal"]) * (np.where((8.0) <= -998, np.tanh((data["rangebatch_msignal"])), data["rangebatch_msignal"] ))), np.sin((data["minbatch_msignal"])) )))) +

                            0.100000*np.tanh(((np.sin(((((data["minbatch_msignal"]) + (data["signal_shift_-1_msignal"]))/2.0)))) - ((((data["minbatch_slices2"]) + (np.tanh(((((-((((np.cos((data["abs_minbatch_msignal"]))) + (data["abs_maxbatch_slices2"])))))) * (data["maxtominbatch_slices2"]))))))/2.0)))) +

                            0.100000*np.tanh(((((data["medianbatch_msignal"]) - (((np.where(((((((data["rangebatch_slices2"]) / 2.0)) * 2.0)) * 2.0) <= -998, (((0.0)) / 2.0), np.sin((np.where(data["maxtominbatch"] > -998, ((data["minbatch_msignal"]) - (np.tanh(((-1.0))))), np.sin((data["rangebatch_slices2"])) ))) )) * (data["rangebatch_slices2"]))))) / 2.0)) +

                            0.100000*np.tanh(((np.where(((data["minbatch_slices2_msignal"]) * 2.0) > -998, data["minbatch_msignal"], data["meanbatch_slices2_msignal"] )) * (np.cos((data["minbatch_msignal"]))))) +

                            0.100000*np.tanh(((np.tanh((((data["minbatch_msignal"]) * (np.cos((data["minbatch_msignal"]))))))) - (np.where(((data["stdbatch_slices2"]) + (data["medianbatch_slices2"])) > -998, data["abs_minbatch_slices2"], ((data["stdbatch_slices2"]) - (np.where(data["minbatch_msignal"] > -998, np.cos((np.sin((data["rangebatch_slices2"])))), ((data["stdbatch_slices2"]) - (np.sin(((-((((data["medianbatch_msignal"]) * 2.0)))))))) ))) )))) +

                            0.100000*np.tanh(np.sin(((((0.0)) - (np.where((12.20356369018554688) > -998, data["maxbatch_msignal"], ((np.where(np.sin((data["minbatch_slices2_msignal"])) <= -998, (0.0), data["abs_maxbatch_slices2_msignal"] )) - ((((-2.0)) * 2.0))) )))))) +

                            0.100000*np.tanh(np.sin((np.sin((((np.sin((((np.where(np.cos((data["meanbatch_slices2_msignal"])) > -998, data["abs_maxbatch_msignal"], data["abs_maxbatch_msignal"] )) * 2.0)))) * 2.0)))))) +

                            0.100000*np.tanh(np.sin(((-((((((data["abs_maxbatch_slices2_msignal"]) + (np.where(data["maxbatch_slices2_msignal"] > -998, (-1.0), np.sin((data["rangebatch_slices2_msignal"])) )))) * 2.0))))))) +

                            0.100000*np.tanh(np.where(data["rangebatch_slices2_msignal"] > -998, (-((((data["rangebatch_slices2_msignal"]) * (((((np.sin((data["stdbatch_slices2_msignal"]))) / 2.0)) * 2.0)))))), ((((data["maxtominbatch_msignal"]) + ((6.0)))) * 2.0) )) +

                            0.100000*np.tanh(np.where(np.where(np.cos((((data["maxbatch_slices2_msignal"]) * 2.0))) > -998, np.cos(((-(((((10.0)) * (((data["rangebatch_slices2_msignal"]) / 2.0)))))))), data["maxbatch_slices2_msignal"] ) > -998, ((data["maxbatch_slices2_msignal"]) * (np.cos((((data["maxbatch_slices2_msignal"]) * 2.0))))), (-2.0) )) +

                            0.100000*np.tanh(data["signal"]) +

                            0.100000*np.tanh(np.where(np.sin((np.where(((data["stdbatch_slices2"]) * (np.where(np.sin(((2.0))) <= -998, data["meanbatch_slices2"], data["abs_avgbatch_slices2_msignal"] ))) > -998, data["stdbatch_slices2_msignal"], ((data["abs_minbatch_slices2"]) + (np.tanh((data["abs_avgbatch_slices2"])))) ))) <= -998, (-3.0), np.sin((data["abs_maxbatch"])) )) +

                            0.100000*np.tanh(data["stdbatch_slices2_msignal"]) +

                            0.100000*np.tanh(((((data["abs_maxbatch"]) * (np.sin(((((np.tanh((np.tanh((np.sin((np.cos((data["meanbatch_msignal"]))))))))) + (np.sin((((data["meanbatch_msignal"]) + (data["signal"]))))))/2.0)))))) * 2.0)) +

                            0.100000*np.tanh((((((-3.0)) + (data["abs_maxbatch_msignal"]))) * 2.0)) +

                            0.100000*np.tanh(np.where(((((data["abs_maxbatch_slices2_msignal"]) * 2.0)) / 2.0) > -998, np.cos((((data["abs_maxbatch_slices2_msignal"]) * 2.0))), ((((np.sin((np.cos((((data["abs_maxbatch_slices2_msignal"]) * 2.0)))))) / 2.0)) / 2.0) )) +

                            0.100000*np.tanh(np.where(data["minbatch_slices2"] <= -998, ((data["rangebatch_slices2_msignal"]) + (data["abs_maxbatch_slices2_msignal"])), (((((-((data["meanbatch_msignal"])))) * 2.0)) * (((((((-2.0)) - (np.sin(((-((data["meanbatch_msignal"])))))))) + (data["maxbatch_slices2_msignal"]))/2.0))) )) +

                            0.100000*np.tanh(np.sin((np.where(data["minbatch_slices2"] <= -998, np.cos((data["stdbatch_slices2"])), ((data["abs_maxbatch_msignal"]) * 2.0) )))) +

                            0.100000*np.tanh((((((-((data["signal"])))) / 2.0)) * (np.where((1.0) > -998, np.sin((data["stdbatch_slices2_msignal"])), (((((data["signal_shift_+1"]) + (np.cos((np.where(((data["meanbatch_msignal"]) + (data["stdbatch_slices2_msignal"])) <= -998, data["abs_avgbatch_slices2_msignal"], data["minbatch_slices2_msignal"] )))))/2.0)) + ((-(((-3.0)))))) )))) +

                            0.100000*np.tanh((((np.cos(((((data["abs_avgbatch_msignal"]) + (data["minbatch_slices2_msignal"]))/2.0)))) + ((-((((data["abs_avgbatch_msignal"]) - (((np.where(data["abs_maxbatch_slices2"] > -998, (-2.0), data["maxtominbatch_slices2"] )) - (data["abs_maxbatch_slices2"])))))))))/2.0)) +

                            0.100000*np.tanh(((data["maxtominbatch"]) + (np.where(np.where(data["maxtominbatch"] <= -998, np.cos((data["maxtominbatch"])), data["signal_shift_-1"] ) <= -998, data["maxtominbatch"], data["signal_shift_-1"] )))) +

                            0.100000*np.tanh((-((np.where((0.0) <= -998, np.where(np.where((((data["signal"]) + (np.sin((data["abs_avgbatch_slices2_msignal"]))))/2.0) <= -998, data["maxtominbatch_slices2"], (-((np.tanh((data["medianbatch_slices2_msignal"]))))) ) > -998, (-((data["abs_avgbatch_msignal"]))), ((data["abs_avgbatch_msignal"]) * 2.0) ), data["rangebatch_slices2"] ))))) +

                            0.100000*np.tanh(((np.sin(((((data["minbatch_slices2_msignal"]) + (np.sin((np.where((-2.0) <= -998, data["signal_shift_-1_msignal"], data["minbatch_slices2_msignal"] )))))/2.0)))) * 2.0)) +

                            0.100000*np.tanh(((np.sin((((((((((1.0)) / 2.0)) * 2.0)) + ((-(((((((((((-3.0)) / 2.0)) * ((-((data["minbatch_slices2_msignal"])))))) * (data["abs_avgbatch_msignal"]))) * (np.sin((data["maxtominbatch_msignal"])))))))))/2.0)))) / 2.0)) +

                            0.100000*np.tanh(np.where(np.sin((((((((np.sin((np.tanh((data["maxbatch_slices2"]))))) / 2.0)) / 2.0)) / 2.0))) <= -998, np.where((-((((data["abs_avgbatch_slices2_msignal"]) / 2.0)))) <= -998, data["abs_maxbatch_slices2_msignal"], data["meanbatch_msignal"] ), (-((np.cos((data["minbatch_msignal"]))))) )) +

                            0.100000*np.tanh((((-((np.where(data["rangebatch_slices2_msignal"] > -998, data["minbatch"], data["maxtominbatch_slices2_msignal"] ))))) + ((-((np.where(((np.where(data["stdbatch_slices2_msignal"] <= -998, data["stdbatch_slices2_msignal"], ((data["signal_shift_-1"]) - ((7.0))) )) / 2.0) <= -998, np.where(data["meanbatch_slices2_msignal"] <= -998, data["maxtominbatch_slices2_msignal"], data["abs_maxbatch_msignal"] ), (-((((data["signal_shift_-1"]) - ((7.0)))))) ))))))) +

                            0.100000*np.tanh((((-2.0)) + (np.where(data["rangebatch_slices2_msignal"] > -998, data["maxbatch_slices2_msignal"], ((np.sin(((((data["meanbatch_slices2"]) + (data["maxtominbatch_msignal"]))/2.0)))) * ((((-2.0)) + (np.where(data["maxtominbatch_msignal"] > -998, data["maxbatch_slices2_msignal"], (((-2.0)) * 2.0) ))))) )))) +

                            0.100000*np.tanh(np.sin((np.where(np.tanh((((data["abs_avgbatch_slices2_msignal"]) * (data["maxtominbatch"])))) > -998, (-1.0), np.cos((np.where(data["rangebatch_slices2_msignal"] <= -998, np.sin((data["abs_avgbatch_msignal"])), data["abs_minbatch_slices2"] ))) )))) +

                            0.100000*np.tanh(((np.where(np.tanh(((-((np.where((-(((-((data["meanbatch_msignal"])))))) > -998, data["minbatch_msignal"], np.where(data["meanbatch_msignal"] > -998, data["medianbatch_msignal"], np.where(data["abs_minbatch_slices2_msignal"] > -998, data["maxbatch_msignal"], data["meanbatch_msignal"] ) ) )))))) > -998, (-((np.tanh((np.tanh((((data["meanbatch_msignal"]) / 2.0)))))))), data["minbatch_slices2"] )) * 2.0)) +

                            0.100000*np.tanh(np.sin((data["rangebatch_slices2_msignal"]))) +

                            0.100000*np.tanh(((data["mean_abs_chgbatch_slices2_msignal"]) * (np.sin((np.where((0.77013391256332397) > -998, np.where(data["minbatch_msignal"] <= -998, data["signal"], data["medianbatch_msignal"] ), ((data["abs_minbatch_slices2_msignal"]) + (np.where(((data["meanbatch_slices2_msignal"]) * 2.0) <= -998, data["maxbatch_msignal"], np.cos((np.where(data["medianbatch_msignal"] <= -998, data["abs_maxbatch_slices2_msignal"], data["rangebatch_msignal"] ))) ))) )))))) +

                            0.100000*np.tanh(((np.cos((((data["abs_maxbatch_msignal"]) / 2.0)))) * (((((((np.cos((np.tanh(((-2.0)))))) / 2.0)) + (((data["stdbatch_slices2_msignal"]) + (np.where(((((data["stdbatch_slices2_msignal"]) + (np.cos((((data["abs_maxbatch_msignal"]) / 2.0)))))) * 2.0) > -998, data["stdbatch_slices2_msignal"], np.sin(((0.0))) )))))) * 2.0)))) +

                            0.100000*np.tanh(np.where(data["minbatch"] <= -998, np.tanh((data["maxtominbatch_slices2_msignal"])), np.sin((np.cos((data["maxtominbatch_slices2_msignal"])))) )) +

                            0.100000*np.tanh(np.where((-2.0) <= -998, (((((((data["abs_avgbatch_slices2_msignal"]) + (np.where(data["abs_minbatch_slices2"] > -998, np.sin(((((data["minbatch_msignal"]) + (data["abs_maxbatch_slices2"]))/2.0))), data["minbatch_msignal"] )))) + (data["abs_avgbatch_slices2_msignal"]))/2.0)) * 2.0), (-1.0) )) +

                            0.100000*np.tanh((((-((((data["abs_maxbatch_slices2_msignal"]) * 2.0))))) - (np.where(np.where(data["minbatch"] <= -998, data["abs_avgbatch_slices2_msignal"], data["meanbatch_slices2"] ) <= -998, np.where(((data["maxbatch_msignal"]) / 2.0) <= -998, data["abs_avgbatch_slices2_msignal"], ((data["maxtominbatch_msignal"]) + (((np.sin((np.cos((data["signal_shift_-1"]))))) * 2.0))) ), data["minbatch_msignal"] )))) +

                            0.100000*np.tanh(np.cos((((np.where(data["maxtominbatch"] > -998, np.where(np.cos((data["minbatch_slices2"])) > -998, data["maxtominbatch_msignal"], (-((data["maxtominbatch_slices2"]))) ), np.tanh((data["maxbatch_slices2_msignal"])) )) * 2.0)))) +

                            0.100000*np.tanh(((data["meanbatch_msignal"]) * ((-((np.where(data["meanbatch_msignal"] > -998, (3.0), np.cos((np.sin(((-((np.where((6.0) > -998, np.where(data["abs_maxbatch_slices2_msignal"] <= -998, (0.0), data["medianbatch_msignal"] ), data["signal_shift_+1_msignal"] )))))))) ))))))) +

                            0.100000*np.tanh(np.sin((((np.sin((data["signal_shift_-1"]))) + (data["signal_shift_-1_msignal"]))))) +

                            0.100000*np.tanh(np.where(np.where(np.where((((-3.0)) * (data["meanbatch_msignal"])) > -998, ((data["signal_shift_-1_msignal"]) - (((np.cos(((-2.0)))) * 2.0))), data["maxbatch_slices2_msignal"] ) > -998, (-2.0), (-((((np.sin((data["maxbatch_msignal"]))) * 2.0)))) ) > -998, (-((data["medianbatch_msignal"]))), ((data["abs_maxbatch_msignal"]) * 2.0) )) +

                            0.100000*np.tanh(((np.cos(((((-(((((-1.0)) * (np.where((((-(((9.35854721069335938))))) - (data["abs_minbatch_slices2_msignal"])) > -998, data["abs_maxbatch"], data["maxbatch_msignal"] ))))))) - (data["abs_minbatch_slices2_msignal"]))))) * 2.0)) +

                            0.100000*np.tanh(((((((0.0)) + (data["abs_avgbatch_slices2"]))/2.0)) / 2.0)) +

                            0.100000*np.tanh(np.sin(((-((np.where(data["medianbatch_slices2_msignal"] > -998, np.cos((np.cos((((data["maxtominbatch_slices2"]) - (data["signal_shift_+1_msignal"])))))), np.where(data["maxtominbatch_slices2"] > -998, data["abs_minbatch_slices2"], data["signal_shift_+1_msignal"] ) ))))))) +

                            0.100000*np.tanh(np.tanh((((data["signal_shift_-1"]) * (np.where((-((data["mean_abs_chgbatch_msignal"]))) > -998, ((np.cos((((data["maxbatch_slices2_msignal"]) * (np.where(data["mean_abs_chgbatch_slices2"] <= -998, ((data["signal_shift_+1_msignal"]) / 2.0), data["stdbatch_slices2"] )))))) / 2.0), np.where((-((data["meanbatch_slices2"]))) > -998, data["maxtominbatch_msignal"], ((data["signal_shift_-1"]) * (data["signal_shift_-1"])) ) )))))) +

                            0.100000*np.tanh(np.sin((((np.where(((data["rangebatch_msignal"]) + ((-((np.sin((data["rangebatch_slices2_msignal"]))))))) <= -998, data["stdbatch_msignal"], (6.0) )) - (data["maxbatch_msignal"]))))) +

                            0.100000*np.tanh(np.where(np.sin((((data["maxtominbatch_msignal"]) + (data["maxbatch_slices2_msignal"])))) > -998, data["rangebatch_msignal"], (((data["signal_shift_-1_msignal"]) + (np.cos((((data["abs_maxbatch_msignal"]) * 2.0)))))/2.0) )) +

                            0.100000*np.tanh(np.where(data["signal_shift_+1_msignal"] > -998, data["minbatch_msignal"], np.tanh((((((np.cos(((-3.0)))) * 2.0)) * (np.where(data["minbatch_slices2_msignal"] > -998, (-2.0), ((((data["minbatch_slices2_msignal"]) * 2.0)) * 2.0) ))))) )) +

                            0.100000*np.tanh(((np.where(data["meanbatch_slices2_msignal"] <= -998, ((np.cos((np.tanh((np.tanh((data["minbatch"]))))))) * (data["rangebatch_msignal"])), data["rangebatch_msignal"] )) * (np.cos(((((((np.cos((data["abs_maxbatch_msignal"]))) / 2.0)) + (data["minbatch_msignal"]))/2.0)))))) +

                            0.100000*np.tanh(((data["abs_avgbatch_slices2_msignal"]) * (((np.cos((data["meanbatch_slices2_msignal"]))) - (((((((np.cos((data["minbatch_msignal"]))) / 2.0)) * ((-((((data["maxbatch_slices2"]) * (data["rangebatch_msignal"])))))))) + (((np.where((4.0) > -998, data["medianbatch_slices2_msignal"], np.cos((np.cos((data["abs_avgbatch_slices2_msignal"])))) )) * ((4.0)))))))))) +

                            0.100000*np.tanh(((((data["medianbatch_msignal"]) * 2.0)) / 2.0)) +

                            0.100000*np.tanh((((((1.0)) / 2.0)) + (np.where((-(((-2.0)))) <= -998, (-2.0), data["maxbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh(np.sin((np.where((9.0) <= -998, np.cos((data["meanbatch_slices2"])), data["rangebatch_msignal"] )))) +

                            0.100000*np.tanh((((np.where((1.0) > -998, data["mean_abs_chgbatch_slices2"], np.sin((np.sin((((data["maxtominbatch_slices2"]) + (data["mean_abs_chgbatch_slices2_msignal"])))))) )) + (data["abs_maxbatch_slices2_msignal"]))/2.0)) +

                            0.100000*np.tanh(((((data["signal_shift_-1_msignal"]) * (((((((data["abs_maxbatch_slices2_msignal"]) / 2.0)) + (np.where(data["maxbatch_slices2_msignal"] > -998, data["meanbatch_msignal"], np.where(((((data["signal_shift_-1_msignal"]) + (data["abs_maxbatch_slices2_msignal"]))) * 2.0) > -998, data["medianbatch_slices2_msignal"], data["medianbatch_slices2_msignal"] ) )))) + (np.where(data["meanbatch_msignal"] > -998, data["meanbatch_msignal"], (-((np.cos((data["medianbatch_slices2"]))))) )))))) * (data["abs_maxbatch_slices2_msignal"]))) +

                            0.100000*np.tanh(np.where(((np.where(data["rangebatch_slices2_msignal"] <= -998, ((data["mean_abs_chgbatch_slices2"]) * (data["signal"])), ((((((-2.0)) + (data["abs_maxbatch_slices2_msignal"]))/2.0)) * (data["stdbatch_slices2_msignal"])) )) * 2.0) <= -998, np.cos((data["mean_abs_chgbatch_slices2"])), ((((np.cos((((data["mean_abs_chgbatch_slices2"]) * ((5.10179901123046875)))))) * 2.0)) * 2.0) )) +

                            0.100000*np.tanh(np.where((((np.sin(((-2.0)))) + (((((np.where((0.0) > -998, data["signal_shift_-1_msignal"], np.cos((data["maxtominbatch_slices2"])) )) / 2.0)) / 2.0)))/2.0) <= -998, data["signal_shift_+1_msignal"], ((data["medianbatch_slices2_msignal"]) / 2.0) )) +

                            0.100000*np.tanh(((((((np.sin((((np.where(((((np.where(data["medianbatch_slices2_msignal"] <= -998, np.where(data["abs_minbatch_msignal"] <= -998, np.cos((data["signal_shift_+1_msignal"])), ((np.sin((data["signal_shift_+1_msignal"]))) * 2.0) ), data["medianbatch_msignal"] )) * 2.0)) * 2.0) <= -998, data["minbatch_slices2"], data["signal_shift_+1_msignal"] )) - (((data["medianbatch_msignal"]) * 2.0)))))) * 2.0)) * 2.0)) * 2.0)) +

                            0.100000*np.tanh((((-((((np.cos((((np.where(data["maxtominbatch_slices2"] <= -998, np.where((0.0) <= -998, data["maxtominbatch_slices2"], data["signal_shift_+1_msignal"] ), ((data["abs_maxbatch_msignal"]) * ((4.0))) )) / 2.0)))) * (np.where(data["abs_maxbatch_msignal"] <= -998, data["medianbatch_slices2_msignal"], data["signal_shift_+1_msignal"] ))))))) * 2.0)) +

                            0.100000*np.tanh(np.where(((data["abs_avgbatch_slices2"]) * (((data["meanbatch_slices2"]) * 2.0))) > -998, data["abs_minbatch_slices2"], np.sin((data["rangebatch_slices2_msignal"])) )) +

                            0.100000*np.tanh(np.where(np.where(np.where(((data["minbatch_slices2_msignal"]) * 2.0) <= -998, data["abs_maxbatch_slices2_msignal"], np.where(data["mean_abs_chgbatch_slices2"] > -998, ((data["minbatch_slices2"]) * 2.0), data["rangebatch_msignal"] ) ) <= -998, ((np.sin((np.where(data["abs_avgbatch_msignal"] > -998, data["abs_maxbatch"], np.sin((((data["signal"]) + (data["rangebatch_slices2_msignal"])))) )))) / 2.0), data["abs_minbatch_slices2"] ) <= -998, data["minbatch_slices2"], np.cos((data["maxtominbatch_msignal"])) )) +

                            0.100000*np.tanh(np.cos((np.where((((data["maxbatch_slices2_msignal"]) + (np.cos((data["maxbatch_slices2_msignal"]))))/2.0) > -998, np.sin((np.tanh((data["mean_abs_chgbatch_msignal"])))), (-((data["mean_abs_chgbatch_msignal"]))) )))) +

                            0.100000*np.tanh(((np.where(((((data["mean_abs_chgbatch_slices2"]) - (data["meanbatch_msignal"]))) + ((((data["rangebatch_slices2_msignal"]) + (((np.cos(((2.0)))) * (data["abs_minbatch_slices2"]))))/2.0))) <= -998, ((data["mean_abs_chgbatch_slices2"]) * (data["maxtominbatch_msignal"])), np.where(data["abs_maxbatch"] <= -998, (1.77194643020629883), ((data["meanbatch_msignal"]) * ((((-2.0)) + ((-((data["medianbatch_slices2_msignal"]))))))) ) )) / 2.0)) +

                            0.100000*np.tanh(((((np.tanh(((-3.0)))) / 2.0)) + ((((14.01316928863525391)) * (np.where(((((-1.0)) + ((-((data["abs_minbatch_slices2"])))))/2.0) <= -998, data["mean_abs_chgbatch_msignal"], ((data["signal_shift_+1_msignal"]) + (data["meanbatch_slices2_msignal"])) )))))) +

                            0.100000*np.tanh((-((np.tanh((np.sin((((np.cos((data["minbatch_msignal"]))) / 2.0))))))))) +

                            0.100000*np.tanh(np.tanh((((np.cos(((((data["rangebatch_msignal"]) + (data["meanbatch_slices2"]))/2.0)))) - (np.cos((data["minbatch_msignal"]))))))) +

                            0.100000*np.tanh(np.sin((np.where(np.tanh((data["signal_shift_-1_msignal"])) <= -998, np.where(data["abs_maxbatch_msignal"] > -998, ((np.sin((data["rangebatch_msignal"]))) - (data["abs_maxbatch_msignal"])), np.sin((((((6.04850435256958008)) + (np.where(data["maxtominbatch"] <= -998, data["maxtominbatch"], np.tanh((data["abs_maxbatch_slices2"])) )))/2.0))) ), ((data["mean_abs_chgbatch_msignal"]) + (data["maxtominbatch"])) )))) +

                            0.100000*np.tanh(((((((((data["maxtominbatch_msignal"]) + (data["maxtominbatch_msignal"]))/2.0)) + (data["maxtominbatch"]))/2.0)) / 2.0)) +

                            0.100000*np.tanh(np.cos((np.where(np.where(data["signal_shift_-1_msignal"] > -998, np.tanh((data["mean_abs_chgbatch_msignal"])), np.cos((data["meanbatch_slices2_msignal"])) ) <= -998, ((data["signal"]) * 2.0), np.cos((((data["mean_abs_chgbatch_msignal"]) + (data["mean_abs_chgbatch_msignal"])))) )))) +

                            0.100000*np.tanh(((np.cos(((-((((((((np.sin((data["stdbatch_msignal"]))) + (data["abs_maxbatch"]))) - ((((-((np.where(np.cos((np.cos((((data["stdbatch_msignal"]) / 2.0))))) > -998, data["meanbatch_slices2"], (2.0) ))))) / 2.0)))) * 2.0))))))) * 2.0)) +

                            0.100000*np.tanh((-((((np.where((2.0) > -998, np.cos((((data["medianbatch_slices2_msignal"]) + (np.where(data["abs_maxbatch_msignal"] > -998, (-2.0), ((np.cos(((0.0)))) * 2.0) ))))), (-2.0) )) - ((-2.0))))))) +

                            0.100000*np.tanh(np.sin(((((data["minbatch_msignal"]) + (np.tanh((np.cos(((((((data["signal_shift_-1_msignal"]) + ((((((data["minbatch_msignal"]) + (np.cos((np.tanh((data["minbatch_msignal"]))))))/2.0)) * 2.0)))/2.0)) * 2.0)))))))/2.0)))) +

                            0.100000*np.tanh(np.sin(((-(((((np.tanh((((data["maxtominbatch_slices2"]) - (np.where(data["abs_minbatch_slices2_msignal"] <= -998, data["meanbatch_slices2_msignal"], ((data["maxtominbatch_slices2_msignal"]) - (np.tanh((data["medianbatch_slices2"])))) )))))) + (data["abs_avgbatch_slices2_msignal"]))/2.0))))))) +

                            0.100000*np.tanh(((((((data["medianbatch_slices2_msignal"]) * ((((data["medianbatch_slices2"]) + (np.where(data["minbatch_msignal"] <= -998, data["stdbatch_msignal"], np.tanh((data["abs_avgbatch_slices2"])) )))/2.0)))) * 2.0)) * ((((2.0)) - (data["abs_avgbatch_msignal"]))))) +

                            0.100000*np.tanh(np.where(((((((np.cos((data["minbatch_slices2_msignal"]))) + (data["meanbatch_slices2"]))/2.0)) + ((-((data["minbatch_slices2_msignal"])))))/2.0) > -998, np.cos((data["minbatch_slices2_msignal"])), data["rangebatch_msignal"] )) +

                            0.100000*np.tanh(np.cos(((((data["rangebatch_slices2"]) + (((data["maxbatch_msignal"]) - (np.tanh((data["signal_shift_+1"]))))))/2.0)))) +

                            0.100000*np.tanh(np.tanh(((((np.cos((np.tanh((np.sin((((data["meanbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2"]))))))))) + (((data["meanbatch_slices2_msignal"]) / 2.0)))/2.0)))) +

                            0.100000*np.tanh(np.cos((np.sin((np.sin((((np.where(data["maxbatch_msignal"] > -998, (3.0), (-2.0) )) * 2.0)))))))) +

                            0.100000*np.tanh(np.sin((((((((-((data["maxbatch_msignal"])))) + ((-((data["maxbatch_msignal"])))))) + (np.where(np.sin((np.where(data["abs_avgbatch_slices2_msignal"] <= -998, (-((data["maxbatch_msignal"]))), data["maxbatch_msignal"] ))) > -998, data["signal_shift_+1_msignal"], (-((np.where(np.sin((np.where(data["maxtominbatch_msignal"] > -998, data["maxbatch_msignal"], data["signal_shift_+1_msignal"] ))) > -998, data["meanbatch_slices2"], data["maxbatch_msignal"] )))) )))/2.0)))) +

                            0.100000*np.tanh(np.where(np.tanh((np.tanh((np.cos((data["abs_maxbatch_msignal"])))))) <= -998, (10.0), (-1.0) )) +

                            0.100000*np.tanh((((((data["rangebatch_msignal"]) / 2.0)) + ((((((data["abs_maxbatch"]) * (data["maxbatch_msignal"]))) + (data["abs_maxbatch_msignal"]))/2.0)))/2.0)) +

                            0.100000*np.tanh(np.cos((((np.sin((np.where(((np.sin((np.cos((((np.sin((data["abs_minbatch_msignal"]))) - (((data["abs_minbatch_msignal"]) - (np.sin((np.sin((data["medianbatch_slices2"]))))))))))))) + (((data["abs_minbatch_msignal"]) - (data["maxbatch_slices2"])))) > -998, data["medianbatch_slices2"], (3.88278818130493164) )))) + (((data["abs_minbatch_msignal"]) - (data["maxbatch_slices2"]))))))) +

                            0.100000*np.tanh(np.cos((((((data["rangebatch_msignal"]) + (np.where(np.tanh((data["medianbatch_slices2"])) > -998, data["medianbatch_slices2"], ((data["meanbatch_slices2"]) * 2.0) )))) / 2.0)))) +

                            0.100000*np.tanh(np.tanh(((((data["maxtominbatch"]) + (np.where((-1.0) <= -998, np.tanh(((((((data["abs_maxbatch_msignal"]) * (data["medianbatch_slices2"]))) + (data["abs_avgbatch_msignal"]))/2.0))), np.where(((np.tanh((data["mean_abs_chgbatch_msignal"]))) - (np.tanh((data["abs_avgbatch_slices2"])))) <= -998, (-(((0.0)))), data["signal_shift_+1"] ) )))/2.0)))) +

                            0.100000*np.tanh(np.sin((((np.where(np.tanh((((data["meanbatch_slices2_msignal"]) - (np.where((((-1.0)) * ((-1.0))) <= -998, data["signal_shift_-1"], (-((((data["meanbatch_msignal"]) + (np.cos((((data["signal_shift_+1_msignal"]) + (np.tanh((((data["mean_abs_chgbatch_msignal"]) * 2.0)))))))))))) ))))) <= -998, ((data["meanbatch_slices2_msignal"]) * 2.0), data["medianbatch_slices2_msignal"] )) * 2.0)))) +

                            0.100000*np.tanh(np.where(((np.sin((np.sin((np.where(data["maxtominbatch_slices2"] <= -998, np.sin((data["medianbatch_slices2_msignal"])), data["medianbatch_slices2_msignal"] )))))) / 2.0) <= -998, data["signal_shift_+1_msignal"], np.cos((np.cos((np.cos((data["mean_abs_chgbatch_msignal"])))))) )) +

                            0.100000*np.tanh(((np.sin((((np.where(data["abs_minbatch_slices2_msignal"] <= -998, np.where(((((np.cos((((np.sin((((data["abs_maxbatch_msignal"]) + (data["minbatch_slices2_msignal"]))))) * 2.0)))) + (data["abs_minbatch_slices2_msignal"]))) * 2.0) > -998, data["abs_maxbatch_msignal"], np.sin((data["abs_maxbatch_msignal"])) ), (((data["abs_minbatch_slices2_msignal"]) + (((data["abs_maxbatch_msignal"]) + (data["minbatch_slices2_msignal"]))))/2.0) )) * 2.0)))) * 2.0)) +

                            0.100000*np.tanh(((((data["signal_shift_-1_msignal"]) * (((data["maxbatch_slices2"]) * (((data["maxbatch_slices2"]) * (np.cos((((data["medianbatch_msignal"]) * 2.0)))))))))) - (((np.cos(((-((data["abs_maxbatch"])))))) + (((((data["medianbatch_msignal"]) * (data["signal_shift_-1_msignal"]))) - (((np.where(data["maxbatch_slices2"] > -998, data["maxtominbatch_slices2"], (0.0) )) * 2.0)))))))) +

                            0.100000*np.tanh(((np.sin(((-3.0)))) / 2.0)) +

                            0.100000*np.tanh(np.where(np.sin((data["meanbatch_slices2_msignal"])) <= -998, np.cos(((((((((np.sin((data["rangebatch_slices2_msignal"]))) * 2.0)) + (data["rangebatch_slices2_msignal"]))/2.0)) * 2.0))), np.sin((((data["minbatch_msignal"]) * ((3.0))))) )) +

                            0.100000*np.tanh(np.sin((((((0.0)) + (np.sin((np.where(np.tanh((data["signal_shift_-1_msignal"])) <= -998, np.sin(((-3.0))), np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["maxtominbatch_slices2_msignal"], np.where(data["rangebatch_slices2"] > -998, (2.0), np.tanh(((-((data["medianbatch_msignal"]))))) ) ) )))))/2.0)))) +

                            0.100000*np.tanh(np.sin(((-((((((data["minbatch_msignal"]) * (((np.where(data["meanbatch_msignal"] > -998, data["maxbatch_slices2_msignal"], np.tanh((((data["maxbatch_slices2_msignal"]) + (np.where(data["minbatch_msignal"] > -998, data["mean_abs_chgbatch_slices2"], (-((((data["signal"]) / 2.0)))) ))))) )) / 2.0)))) / 2.0))))))) +

                            0.100000*np.tanh(np.cos((np.where(np.where(np.sin((np.tanh((data["minbatch_slices2"])))) <= -998, ((data["signal_shift_-1_msignal"]) / 2.0), ((data["mean_abs_chgbatch_msignal"]) / 2.0) ) <= -998, data["abs_maxbatch_slices2"], (-((data["abs_maxbatch_slices2"]))) )))) +

                            0.100000*np.tanh(((((np.sin((((data["stdbatch_slices2_msignal"]) - (np.where(data["maxbatch_slices2_msignal"] > -998, data["meanbatch_slices2"], np.tanh((np.cos((((((data["abs_minbatch_msignal"]) * 2.0)) * 2.0))))) )))))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh((-((((data["signal_shift_-1_msignal"]) * (np.cos((np.where(data["medianbatch_slices2_msignal"] > -998, ((data["abs_maxbatch_msignal"]) * 2.0), np.cos((np.where((-((data["abs_minbatch_slices2"]))) > -998, (((-((((data["signal_shift_-1_msignal"]) * (data["meanbatch_msignal"])))))) * 2.0), (((np.cos((data["abs_maxbatch_msignal"]))) + (((((data["abs_maxbatch_msignal"]) * 2.0)) / 2.0)))/2.0) ))) ))))))))) +

                            0.100000*np.tanh(np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) * (np.cos((np.where(((data["rangebatch_msignal"]) * 2.0) <= -998, data["abs_avgbatch_slices2"], np.where(((np.cos((data["abs_maxbatch_slices2"]))) / 2.0) > -998, data["meanbatch_msignal"], (((data["stdbatch_slices2"]) + ((((((data["medianbatch_msignal"]) / 2.0)) + (data["mean_abs_chgbatch_slices2"]))/2.0)))/2.0) ) )))))))) +

                            0.100000*np.tanh((((0.0)) * (data["minbatch_slices2"]))) +

                            0.100000*np.tanh((((0.0)) * 2.0)) +

                            0.100000*np.tanh(np.where((((data["mean_abs_chgbatch_slices2_msignal"]) + (np.sin((np.tanh((data["rangebatch_msignal"]))))))/2.0) > -998, np.cos((np.sin(((((data["abs_avgbatch_msignal"]) + (np.sin(((-((np.tanh((np.where(np.cos((data["mean_abs_chgbatch_slices2"])) <= -998, data["abs_minbatch_msignal"], np.cos((data["meanbatch_slices2_msignal"])) ))))))))))/2.0))))), ((data["medianbatch_msignal"]) - (data["maxbatch_msignal"])) )) +

                            0.100000*np.tanh((((-((np.tanh((((data["medianbatch_slices2_msignal"]) - (np.sin(((0.0))))))))))) * (np.sin((data["minbatch_msignal"]))))) +

                            0.100000*np.tanh(np.cos((np.where(np.sin((((np.sin((np.sin((((np.cos((data["mean_abs_chgbatch_slices2"]))) * 2.0)))))) / 2.0))) <= -998, ((np.sin((np.sin((((((data["signal_shift_+1"]) / 2.0)) - (data["stdbatch_slices2"]))))))) - (data["abs_minbatch_slices2_msignal"])), data["medianbatch_msignal"] )))) +

                            0.100000*np.tanh(np.cos((np.where(data["meanbatch_slices2"] > -998, ((np.sin((data["maxtominbatch"]))) * ((((0.0)) + ((-((np.where(data["maxtominbatch_msignal"] <= -998, (-((data["maxtominbatch_msignal"]))), data["signal"] )))))))), np.sin(((((((((data["medianbatch_slices2"]) + (data["signal_shift_-1_msignal"]))/2.0)) * 2.0)) * 2.0))) )))) +

                            0.100000*np.tanh(np.sin(((((data["maxbatch_slices2"]) + (np.where((0.0) > -998, data["rangebatch_msignal"], np.where((0.0) > -998, data["rangebatch_msignal"], (2.0) ) )))/2.0)))) +

                            0.100000*np.tanh(np.where(np.sin((data["signal_shift_+1"])) > -998, ((((np.cos((((np.sin(((-1.0)))) * 2.0)))) * 2.0)) * 2.0), np.sin((np.sin((((np.tanh((np.where((0.0) <= -998, data["mean_abs_chgbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] )))) - ((-((data["rangebatch_msignal"]))))))))) )) +

                            0.100000*np.tanh(np.cos((np.cos((np.where(data["minbatch_slices2"] <= -998, np.cos(((-((data["abs_minbatch_slices2"]))))), np.sin((((data["meanbatch_msignal"]) + (data["meanbatch_msignal"])))) )))))) +

                            0.100000*np.tanh(np.sin(((-((((np.sin((np.where(np.tanh((((np.where(np.tanh((np.tanh(((0.0))))) > -998, ((np.tanh((np.tanh(((((((data["meanbatch_msignal"]) + ((0.0)))/2.0)) / 2.0)))))) / 2.0), data["meanbatch_slices2_msignal"] )) / 2.0))) <= -998, (((data["mean_abs_chgbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2"]))/2.0), (0.0) )))) / 2.0))))))) +

                            0.100000*np.tanh((-((np.where(np.sin(((0.0))) > -998, ((data["abs_avgbatch_slices2_msignal"]) * (((np.cos((np.cos((np.sin(((-((((((np.sin(((0.0)))) - (((data["signal_shift_+1_msignal"]) * (np.tanh((data["abs_minbatch_slices2_msignal"]))))))) * ((0.0))))))))))))) / 2.0))), np.tanh((data["meanbatch_slices2_msignal"])) ))))) +

                            0.100000*np.tanh((((-((np.cos((((np.cos((data["abs_minbatch_msignal"]))) * ((-((((data["meanbatch_slices2_msignal"]) * (np.where(data["minbatch_slices2"] > -998, ((data["signal_shift_+1_msignal"]) * 2.0), (((((((5.0)) - (data["maxtominbatch_slices2_msignal"]))) / 2.0)) * 2.0) )))))))))))))) * ((((0.0)) * ((0.0)))))) +

                            0.100000*np.tanh(np.sin(((-((np.where(((((data["maxbatch_slices2_msignal"]) * (((data["mean_abs_chgbatch_slices2"]) * 2.0)))) * 2.0) <= -998, np.tanh((data["maxbatch_slices2_msignal"])), ((((data["mean_abs_chgbatch_slices2"]) * (((data["maxbatch_slices2_msignal"]) * ((-((data["mean_abs_chgbatch_slices2"])))))))) * 2.0) ))))))) +

                            0.100000*np.tanh(np.cos((np.where((-((data["abs_minbatch_slices2_msignal"]))) > -998, data["meanbatch_msignal"], (((-((data["abs_maxbatch_slices2_msignal"])))) - (np.sin((np.cos((np.tanh(((0.0))))))))) )))) +

                            0.100000*np.tanh(np.where(data["medianbatch_slices2_msignal"] > -998, np.sin((((((np.where((3.0) > -998, ((data["meanbatch_slices2_msignal"]) / 2.0), data["maxtominbatch_msignal"] )) / 2.0)) / 2.0))), np.cos((np.tanh((data["maxbatch_slices2"])))) )) +

                            0.100000*np.tanh((((data["abs_avgbatch_slices2_msignal"]) + (np.where(data["mean_abs_chgbatch_slices2"] > -998, data["maxbatch_msignal"], np.where(((np.sin((data["abs_maxbatch_msignal"]))) - ((1.0))) <= -998, np.tanh((np.where(data["abs_avgbatch_slices2"] > -998, data["maxbatch_msignal"], data["maxbatch_msignal"] ))), ((data["meanbatch_slices2"]) * (data["abs_maxbatch_slices2_msignal"])) ) )))/2.0)) +

                            0.100000*np.tanh((((((data["mean_abs_chgbatch_slices2"]) + ((0.26138311624526978)))/2.0)) * ((-((np.cos((np.cos((((np.where(data["medianbatch_slices2_msignal"] > -998, (((((0.57094943523406982)) * 2.0)) * (data["signal_shift_+1_msignal"])), data["mean_abs_chgbatch_slices2"] )) / 2.0))))))))))) +

                            0.100000*np.tanh(((np.where(np.sin((np.where(data["abs_maxbatch_msignal"] > -998, (0.0), ((data["maxbatch_slices2_msignal"]) - (((data["abs_maxbatch_slices2"]) - (np.tanh((((data["meanbatch_slices2_msignal"]) * ((9.0))))))))) ))) <= -998, (1.0), ((data["abs_maxbatch_slices2_msignal"]) / 2.0) )) - ((((9.0)) + (data["minbatch_msignal"]))))) +

                            0.100000*np.tanh(((((data["rangebatch_slices2_msignal"]) + ((((((data["abs_avgbatch_msignal"]) + (data["maxtominbatch_slices2"]))) + (((data["maxbatch_msignal"]) + (((data["medianbatch_msignal"]) / 2.0)))))/2.0)))) / 2.0)) +

                            0.100000*np.tanh(np.where(data["minbatch_slices2"] > -998, np.cos(((-2.0))), np.tanh((data["minbatch_slices2_msignal"])) )) +

                            0.100000*np.tanh(np.cos((((((data["maxtominbatch_msignal"]) * (np.where(np.sin((data["abs_avgbatch_slices2"])) > -998, data["abs_avgbatch_slices2"], np.where(data["abs_avgbatch_slices2"] <= -998, data["stdbatch_msignal"], np.sin((np.cos((data["maxtominbatch_msignal"])))) ) )))) / 2.0)))) +

                            0.100000*np.tanh(np.where((((((np.tanh((data["abs_maxbatch_slices2_msignal"]))) / 2.0)) + (np.sin(((-((((((data["abs_maxbatch_slices2"]) + ((3.0)))) * 2.0))))))))/2.0) <= -998, (1.0), np.sin((np.cos(((-(((1.0)))))))) )) +

                            0.100000*np.tanh(np.where(((np.tanh((data["medianbatch_slices2"]))) / 2.0) > -998, data["abs_minbatch_slices2"], (((np.where(data["meanbatch_msignal"] <= -998, (-1.0), data["meanbatch_msignal"] )) + (np.where((1.0) > -998, data["abs_minbatch_slices2"], data["abs_maxbatch_msignal"] )))/2.0) )) +

                            0.100000*np.tanh(np.cos((((((((data["rangebatch_msignal"]) + (((((data["minbatch"]) * 2.0)) / 2.0)))) * 2.0)) * 2.0)))) +

                            0.100000*np.tanh(np.where(data["stdbatch_msignal"] > -998, (((np.cos((np.cos((np.sin((np.cos((np.sin((data["medianbatch_slices2"]))))))))))) + (np.sin((np.cos((data["abs_maxbatch_slices2"]))))))/2.0), data["abs_maxbatch_slices2_msignal"] )) +

                            0.100000*np.tanh(((data["maxbatch_msignal"]) - (np.where(data["maxbatch_msignal"] > -998, (3.0), (3.0) )))) +

                            0.100000*np.tanh(np.cos((((np.cos((((np.cos((np.tanh((np.where(((np.sin((data["signal_shift_+1_msignal"]))) * 2.0) > -998, (3.0), data["abs_minbatch_slices2"] )))))) * 2.0)))) / 2.0)))) +

                            0.100000*np.tanh(np.where(data["abs_avgbatch_slices2_msignal"] <= -998, ((((data["abs_avgbatch_msignal"]) * ((0.07882835716009140)))) / 2.0), (0.0) )) +

                            0.100000*np.tanh(data["rangebatch_msignal"]) +

                            0.100000*np.tanh(np.where(np.where(np.cos(((7.95549821853637695))) > -998, np.tanh(((((-1.0)) - ((1.0))))), np.tanh((np.sin((np.cos((((data["minbatch_msignal"]) * (data["meanbatch_msignal"])))))))) ) <= -998, ((((0.0)) + (data["medianbatch_slices2_msignal"]))/2.0), ((np.tanh((data["maxbatch_slices2_msignal"]))) / 2.0) )) +

                            0.100000*np.tanh(np.where(data["abs_avgbatch_msignal"] <= -998, np.where(data["minbatch_msignal"] <= -998, (((-((data["abs_maxbatch_slices2"])))) + (np.cos(((2.0))))), (-2.0) ), (-(((1.0)))) )) +

                            0.100000*np.tanh((-((((np.cos((data["maxbatch_slices2_msignal"]))) * ((1.0))))))) +

                            0.100000*np.tanh(((np.tanh((np.cos((np.where((2.0) > -998, np.cos((data["maxtominbatch"])), np.tanh((np.tanh(((((np.cos((np.tanh((((data["stdbatch_slices2_msignal"]) * 2.0)))))) + (data["stdbatch_slices2_msignal"]))/2.0))))) )))))) / 2.0)) +

                            0.100000*np.tanh(np.cos((((data["abs_maxbatch_msignal"]) * (np.where(data["stdbatch_slices2"] <= -998, np.where(data["mean_abs_chgbatch_msignal"] > -998, data["mean_abs_chgbatch_msignal"], ((np.where(data["mean_abs_chgbatch_msignal"] > -998, data["mean_abs_chgbatch_msignal"], data["mean_abs_chgbatch_msignal"] )) - (np.where(data["abs_maxbatch_slices2"] > -998, data["maxbatch_slices2"], data["mean_abs_chgbatch_msignal"] ))) ), np.where(data["maxbatch_slices2"] > -998, data["mean_abs_chgbatch_msignal"], ((np.tanh((data["signal_shift_+1_msignal"]))) - (data["abs_maxbatch_slices2"])) ) )))))) +

                            0.100000*np.tanh(np.where(np.where(((np.sin((np.cos((data["mean_abs_chgbatch_slices2"]))))) + (data["maxtominbatch_slices2"])) <= -998, data["maxbatch_msignal"], ((np.cos((data["maxbatch_slices2"]))) * 2.0) ) <= -998, np.tanh((np.sin((data["abs_minbatch_slices2"])))), data["abs_minbatch_slices2"] )) +

                            0.100000*np.tanh(np.tanh((((((data["signal"]) + ((((np.sin((data["abs_avgbatch_slices2_msignal"]))) + (data["signal"]))/2.0)))) / 2.0)))) +

                            0.100000*np.tanh(np.where(((data["stdbatch_msignal"]) * (data["abs_maxbatch_msignal"])) <= -998, np.sin((np.where(data["mean_abs_chgbatch_slices2"] > -998, data["abs_maxbatch_msignal"], data["abs_maxbatch_msignal"] ))), (-((np.cos((((data["stdbatch_msignal"]) * (data["abs_maxbatch_msignal"]))))))) )) +

                            0.100000*np.tanh((((0.0)) / 2.0)) +

                            0.099609*np.tanh(np.sin((np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["medianbatch_msignal"], np.where(data["maxtominbatch_slices2_msignal"] > -998, ((np.where((-2.0) <= -998, ((np.sin(((0.0)))) + (np.tanh((((data["maxtominbatch"]) * 2.0))))), (3.0) )) * 2.0), data["medianbatch_slices2_msignal"] ) )))) +

                            0.100000*np.tanh(np.sin((np.where((-((np.cos((np.tanh(((-(((-((np.tanh((np.sin(((((data["maxtominbatch_msignal"]) + (data["rangebatch_slices2"]))/2.0)))))))))))))))))) <= -998, np.sin((((data["maxbatch_slices2"]) / 2.0))), data["maxtominbatch_msignal"] )))) +

                            0.100000*np.tanh(np.cos((np.cos((np.tanh((np.cos((((((np.where(data["mean_abs_chgbatch_slices2"] > -998, data["maxtominbatch_slices2_msignal"], np.tanh(((((((np.cos((((np.sin(((2.0)))) * (data["mean_abs_chgbatch_slices2"]))))) * ((3.0)))) + (data["meanbatch_slices2_msignal"]))/2.0))) )) * 2.0)) / 2.0)))))))))) +

                            0.100000*np.tanh(np.cos((np.where(data["abs_minbatch_slices2"] > -998, ((((data["minbatch_msignal"]) * (data["mean_abs_chgbatch_slices2"]))) * (data["stdbatch_slices2"])), np.cos((np.where(np.tanh(((((data["abs_maxbatch_slices2_msignal"]) + ((-((data["maxbatch_msignal"])))))/2.0))) > -998, ((data["minbatch_msignal"]) * (data["stdbatch_slices2"])), (((data["maxbatch_msignal"]) + (data["stdbatch_slices2"]))/2.0) ))) )))) +

                            0.100000*np.tanh((((np.tanh((data["abs_avgbatch_slices2"]))) + ((((np.tanh((np.where(data["abs_minbatch_slices2"] > -998, np.cos(((-(((((data["abs_avgbatch_slices2"]) + (np.tanh(((((data["rangebatch_slices2"]) + (data["medianbatch_slices2"]))/2.0)))))/2.0)))))), np.sin((((data["abs_minbatch_slices2"]) * 2.0))) )))) + (data["abs_avgbatch_slices2"]))/2.0)))/2.0)) +

                            0.100000*np.tanh(np.tanh((np.cos(((-((((np.tanh(((0.0)))) + ((((data["rangebatch_slices2_msignal"]) + (data["signal_shift_+1"]))/2.0))))))))))) +

                            0.100000*np.tanh(np.cos((((((np.cos((np.where((-((np.sin((data["maxtominbatch"]))))) <= -998, ((((-((data["abs_maxbatch_slices2"])))) + (((data["medianbatch_slices2"]) / 2.0)))/2.0), (-((((data["abs_minbatch_slices2_msignal"]) - (data["medianbatch_slices2"]))))) )))) * 2.0)) * 2.0)))) +

                            0.100000*np.tanh(((((np.tanh((np.cos((((np.cos((np.tanh((((np.sin(((0.0)))) / 2.0)))))) / 2.0)))))) / 2.0)) / 2.0)) +

                            0.100000*np.tanh((((((np.tanh((((data["signal_shift_-1_msignal"]) * 2.0)))) + (data["maxtominbatch"]))/2.0)) / 2.0)) +

                            0.100000*np.tanh(np.where(np.cos((np.cos((((np.tanh(((1.0)))) / 2.0))))) > -998, np.tanh(((0.52521955966949463))), (((2.0)) - (data["maxbatch_slices2"])) )) +

                            0.100000*np.tanh(((np.cos((data["abs_maxbatch_msignal"]))) + ((-((data["meanbatch_msignal"])))))) +

                            0.089932*np.tanh(((np.where(np.where((2.0) > -998, np.cos((((np.cos((((data["rangebatch_slices2_msignal"]) - (data["maxbatch_slices2_msignal"]))))) / 2.0))), data["meanbatch_slices2_msignal"] ) > -998, np.cos((data["abs_avgbatch_slices2_msignal"])), data["medianbatch_slices2_msignal"] )) * 2.0)) +

                            0.100000*np.tanh((((-((np.where(data["signal"] <= -998, ((np.sin((((data["abs_maxbatch_msignal"]) * 2.0)))) * 2.0), np.sin((((np.sin((((data["maxbatch_msignal"]) * 2.0)))) * 2.0))) ))))) * 2.0)) +

                            0.098436*np.tanh(np.tanh(((-(((0.0))))))) +

                            0.100000*np.tanh(np.tanh(((((((2.0)) / 2.0)) / 2.0)))) +

                            0.100000*np.tanh(np.where(data["maxbatch_msignal"] > -998, np.cos((((data["rangebatch_msignal"]) + (data["minbatch_slices2"])))), ((np.cos((np.where(data["rangebatch_msignal"] > -998, np.cos((((data["rangebatch_msignal"]) * 2.0))), data["rangebatch_msignal"] )))) * 2.0) )) +

                            0.100000*np.tanh(((np.tanh((np.tanh((((np.cos((data["maxbatch_msignal"]))) + (((np.cos((data["maxbatch_slices2_msignal"]))) * (np.where(np.where(data["mean_abs_chgbatch_slices2"] > -998, data["abs_maxbatch_slices2_msignal"], data["mean_abs_chgbatch_slices2"] ) > -998, data["abs_avgbatch_slices2_msignal"], np.cos((data["abs_maxbatch_slices2_msignal"])) )))))))))) * 2.0)) +

                            0.100000*np.tanh(np.tanh((np.where(np.cos(((-(((((data["medianbatch_slices2_msignal"]) + (((((np.sin((data["medianbatch_slices2_msignal"]))) * 2.0)) / 2.0)))/2.0)))))) <= -998, (((data["abs_maxbatch_msignal"]) + (data["maxbatch_slices2_msignal"]))/2.0), (((data["medianbatch_slices2_msignal"]) + (data["maxbatch_slices2_msignal"]))/2.0) )))) +

                            0.100000*np.tanh(np.sin((((data["abs_avgbatch_slices2_msignal"]) * (((np.where((((0.0)) * 2.0) > -998, np.where(data["abs_maxbatch_msignal"] <= -998, ((data["maxbatch_slices2_msignal"]) / 2.0), data["signal_shift_+1_msignal"] ), np.where(np.sin((data["abs_maxbatch_slices2"])) <= -998, np.tanh((data["abs_maxbatch_slices2_msignal"])), (0.0) ) )) / 2.0)))))) +

                            0.100000*np.tanh(np.cos((((data["signal_shift_-1_msignal"]) / 2.0)))) +

                            0.100000*np.tanh(np.sin((np.sin((((((((data["meanbatch_msignal"]) - (np.where(np.sin((((np.where((1.68522870540618896) <= -998, (0.0), np.sin((((data["abs_maxbatch_msignal"]) / 2.0))) )) / 2.0))) > -998, data["signal_shift_+1_msignal"], data["meanbatch_msignal"] )))) * 2.0)) * 2.0)))))) +

                            0.093744*np.tanh(np.where(data["signal_shift_+1_msignal"] <= -998, (0.0), ((np.tanh(((((((data["abs_minbatch_slices2"]) / 2.0)) + (np.tanh(((((((data["signal_shift_+1_msignal"]) / 2.0)) + (np.sin((np.tanh((data["stdbatch_msignal"]))))))/2.0)))))/2.0)))) / 2.0) )) +

                            0.100000*np.tanh(((np.tanh((((((data["abs_avgbatch_msignal"]) - (np.cos((data["maxbatch_msignal"]))))) - (np.sin((np.sin((np.where((-3.0) > -998, data["stdbatch_msignal"], data["minbatch_msignal"] )))))))))) / 2.0)) +

                            0.099707*np.tanh(np.sin((np.sin((((np.where(data["abs_minbatch_msignal"] <= -998, data["maxtominbatch_msignal"], (((-((data["medianbatch_msignal"])))) * 2.0) )) * 2.0)))))) +

                            0.100000*np.tanh(((np.where(np.tanh(((((((data["abs_maxbatch_slices2"]) + (np.where((0.0) <= -998, data["abs_minbatch_msignal"], np.cos((data["maxtominbatch_slices2"])) )))/2.0)) / 2.0))) <= -998, np.cos((data["abs_maxbatch_slices2"])), np.tanh((np.cos((np.where(np.tanh((((data["medianbatch_slices2_msignal"]) * 2.0))) > -998, data["abs_maxbatch_slices2"], (-((data["abs_minbatch_slices2"]))) ))))) )) / 2.0)) +

                            0.100000*np.tanh(np.where(np.sin((data["abs_minbatch_slices2_msignal"])) <= -998, np.cos((np.cos((data["abs_maxbatch_slices2_msignal"])))), np.where(data["signal_shift_-1"] > -998, ((np.cos((((data["medianbatch_msignal"]) + (np.tanh((((np.cos((np.cos((np.tanh((np.cos((((data["minbatch"]) / 2.0)))))))))) / 2.0)))))))) - ((0.0))), (((data["meanbatch_msignal"]) + ((0.0)))/2.0) ) )) +

                            0.100000*np.tanh(np.tanh((((data["abs_maxbatch_slices2_msignal"]) - (np.cos((np.cos((((((data["signal"]) * (np.tanh(((((-((data["mean_abs_chgbatch_slices2"])))) - (np.cos((data["abs_avgbatch_slices2"]))))))))) * (np.where(data["minbatch_slices2_msignal"] > -998, data["maxbatch_slices2_msignal"], data["minbatch"] )))))))))))))  

    

    def GP_class_9(self,data):

        return self.Output( -3.603840 +

                            0.100000*np.tanh((-((((np.where((-(((((data["maxtominbatch_msignal"]) + ((((-(((-((data["minbatch_msignal"]))))))) * 2.0)))/2.0)))) <= -998, ((data["signal_shift_-1"]) * 2.0), data["abs_avgbatch_msignal"] )) * 2.0))))) +

                            0.100000*np.tanh(np.where((((-(((-(((-(((-((data["meanbatch_slices2"]))))))))))))) + ((-((data["maxtominbatch"]))))) > -998, data["medianbatch_slices2"], np.sin(((0.0))) )) +

                            0.100000*np.tanh(np.where(((((np.where(np.tanh((np.sin((data["signal_shift_+1"])))) <= -998, data["abs_avgbatch_msignal"], data["signal_shift_-1"] )) / 2.0)) * 2.0) <= -998, (-((data["abs_avgbatch_msignal"]))), (-((data["abs_avgbatch_msignal"]))) )) +

                            0.100000*np.tanh(((np.where(data["rangebatch_msignal"] > -998, (((((-3.0)) - (data["abs_minbatch_slices2_msignal"]))) - (np.cos((((data["abs_maxbatch_msignal"]) * ((-(((2.0)))))))))), (3.0) )) + (np.sin((((data["medianbatch_slices2"]) * 2.0)))))) +

                            0.100000*np.tanh(np.cos((np.tanh((data["maxbatch_slices2"]))))) +

                            0.100000*np.tanh(np.where(data["rangebatch_msignal"] > -998, np.where(np.sin(((((3.92485237121582031)) + (((data["signal_shift_-1"]) * 2.0))))) <= -998, data["mean_abs_chgbatch_msignal"], (-((data["medianbatch_msignal"]))) ), np.sin((np.tanh(((9.90578651428222656))))) )) +

                            0.100000*np.tanh(np.where(np.where(data["signal_shift_+1"] <= -998, data["abs_avgbatch_msignal"], (-((data["abs_avgbatch_msignal"]))) ) <= -998, (((((((data["signal_shift_+1"]) - (np.tanh((data["signal_shift_-1"]))))) + (((data["abs_avgbatch_slices2"]) - (data["abs_avgbatch_msignal"]))))/2.0)) * (((data["abs_avgbatch_msignal"]) * (data["abs_avgbatch_msignal"])))), np.where((-((data["abs_avgbatch_msignal"]))) <= -998, data["abs_avgbatch_msignal"], (-((data["abs_avgbatch_msignal"]))) ) )) +

                            0.100000*np.tanh(np.where(data["meanbatch_slices2_msignal"] > -998, data["meanbatch_slices2"], np.tanh((np.sin((data["minbatch_slices2"])))) )) +

                            0.100000*np.tanh(data["signal_shift_+1"]) +

                            0.100000*np.tanh(((((-(((((data["stdbatch_slices2_msignal"]) + (np.cos((data["maxtominbatch_slices2_msignal"]))))/2.0))))) + ((-((data["abs_minbatch_slices2_msignal"])))))/2.0)) +

                            0.100000*np.tanh(((((data["abs_maxbatch"]) / 2.0)) + (((np.where((((((((-3.0)) + ((((-3.0)) + ((((0.0)) + (data["abs_maxbatch"]))))))/2.0)) + (data["stdbatch_msignal"]))/2.0) > -998, (-2.0), data["abs_maxbatch"] )) + (np.where(data["abs_avgbatch_slices2"] <= -998, data["abs_maxbatch"], np.tanh(((-((data["abs_avgbatch_slices2"]))))) )))))) +

                            0.100000*np.tanh(((((data["signal_shift_-1"]) - (data["medianbatch_slices2"]))) + (np.where((-((((((data["signal_shift_-1"]) - (((data["stdbatch_slices2"]) * 2.0)))) + ((-(((-((data["mean_abs_chgbatch_slices2"]))))))))))) > -998, (-1.0), data["stdbatch_slices2"] )))) +

                            0.100000*np.tanh((-((((data["meanbatch_slices2_msignal"]) + (np.where((0.0) <= -998, data["abs_maxbatch_slices2"], (((data["meanbatch_msignal"]) + (np.cos(((((-((((((data["medianbatch_slices2_msignal"]) * ((1.0)))) / 2.0))))) * 2.0)))))/2.0) ))))))) +

                            0.100000*np.tanh((((((((((-1.0)) - (((np.tanh((((np.where(data["meanbatch_msignal"] > -998, data["medianbatch_slices2_msignal"], ((data["medianbatch_msignal"]) * 2.0) )) / 2.0)))) / 2.0)))) - (np.tanh((data["medianbatch_msignal"]))))) * 2.0)) * 2.0)) +

                            0.100000*np.tanh((((((7.50564193725585938)) * (((((data["minbatch_slices2"]) / 2.0)) * 2.0)))) * ((-((data["signal_shift_-1"])))))) +

                            0.100000*np.tanh((-((np.where(((data["maxtominbatch_slices2_msignal"]) * (((data["signal_shift_+1"]) * ((((8.0)) - ((-(((((-((data["maxtominbatch_msignal"])))) / 2.0)))))))))) > -998, data["abs_avgbatch_msignal"], np.where(data["abs_avgbatch_msignal"] > -998, data["abs_avgbatch_msignal"], np.where(data["maxtominbatch"] > -998, data["medianbatch_msignal"], data["rangebatch_msignal"] ) ) ))))) +

                            0.100000*np.tanh(((((data["maxtominbatch_msignal"]) * (np.cos((data["abs_minbatch_slices2_msignal"]))))) - (data["rangebatch_slices2"]))) +

                            0.100000*np.tanh((-((((data["medianbatch_slices2_msignal"]) - (np.cos((np.where(np.tanh((data["signal"])) > -998, (-3.0), ((np.cos((np.where((((np.sin((np.cos(((13.76372241973876953)))))) + ((2.0)))/2.0) > -998, data["stdbatch_slices2_msignal"], ((((13.76372241973876953)) + ((((-3.0)) * (data["signal_shift_+1_msignal"]))))/2.0) )))) / 2.0) ))))))))) +

                            0.100000*np.tanh((((-((np.cos((data["abs_avgbatch_slices2"])))))) * 2.0)) +

                            0.100000*np.tanh((-((np.where(np.where(((((np.tanh((((((((data["stdbatch_msignal"]) * (((data["minbatch_msignal"]) * 2.0)))) * (np.sin((data["abs_minbatch_msignal"]))))) * 2.0)))) * 2.0)) * 2.0) <= -998, ((data["maxbatch_slices2_msignal"]) * (data["abs_minbatch_slices2"])), data["maxbatch_slices2_msignal"] ) > -998, data["abs_avgbatch_msignal"], ((data["signal_shift_+1"]) * ((1.0))) ))))) +

                            0.100000*np.tanh(((((data["minbatch_slices2_msignal"]) * (np.cos((np.where((-3.0) <= -998, ((((3.0)) + (((np.where(data["abs_minbatch_msignal"] <= -998, np.cos((data["meanbatch_msignal"])), data["minbatch_slices2_msignal"] )) / 2.0)))/2.0), data["meanbatch_msignal"] )))))) * 2.0)) +

                            0.100000*np.tanh((((((data["maxbatch_msignal"]) + ((((3.0)) * (np.where(data["medianbatch_slices2_msignal"] > -998, data["meanbatch_msignal"], ((((data["stdbatch_slices2_msignal"]) / 2.0)) + ((((((data["maxbatch_msignal"]) + (data["maxbatch_msignal"]))/2.0)) * (data["signal_shift_-1"])))) )))))/2.0)) * (data["stdbatch_msignal"]))) +

                            0.100000*np.tanh((((np.where(data["abs_avgbatch_msignal"] > -998, data["mean_abs_chgbatch_slices2_msignal"], ((np.cos((data["mean_abs_chgbatch_msignal"]))) - (data["abs_avgbatch_slices2"])) )) + (((data["rangebatch_msignal"]) * (data["rangebatch_slices2"]))))/2.0)) +

                            0.100000*np.tanh((-((data["medianbatch_slices2_msignal"])))) +

                            0.100000*np.tanh(((data["minbatch_msignal"]) * (np.where((-((np.cos((data["signal_shift_-1_msignal"]))))) <= -998, data["abs_maxbatch_slices2"], np.tanh((np.cos(((-((np.where(np.cos((np.sin((np.cos((np.sin((data["abs_avgbatch_slices2_msignal"])))))))) <= -998, np.sin(((-(((-1.0)))))), data["medianbatch_slices2_msignal"] )))))))) )))) +

                            0.100000*np.tanh(np.sin(((((-(((-((np.sin((((((data["stdbatch_slices2"]) - ((((-(((-((((data["minbatch_slices2_msignal"]) / 2.0)))))))) * 2.0)))) * 2.0)))))))))) - (((data["signal_shift_+1_msignal"]) * 2.0)))))) +

                            0.100000*np.tanh(((((((data["abs_minbatch_slices2_msignal"]) * (np.cos(((((-(((0.0))))) * 2.0)))))) / 2.0)) * (np.cos((((((np.where(((((((data["rangebatch_slices2_msignal"]) * 2.0)) / 2.0)) * 2.0) <= -998, (-((data["abs_minbatch_slices2_msignal"]))), data["medianbatch_slices2_msignal"] )) / 2.0)) * 2.0)))))) +

                            0.100000*np.tanh(np.where(data["maxbatch_msignal"] > -998, (-((data["meanbatch_msignal"]))), np.where(np.tanh((data["signal_shift_-1_msignal"])) <= -998, np.where(data["maxbatch_msignal"] <= -998, data["stdbatch_slices2_msignal"], np.cos((data["abs_avgbatch_slices2_msignal"])) ), np.where(data["signal_shift_-1_msignal"] <= -998, np.where(((data["medianbatch_slices2_msignal"]) - (data["rangebatch_slices2_msignal"])) <= -998, data["meanbatch_slices2_msignal"], np.sin((data["medianbatch_msignal"])) ), data["abs_minbatch_slices2_msignal"] ) ) )) +

                            0.100000*np.tanh(np.sin((((np.where(np.cos((data["maxbatch_slices2_msignal"])) > -998, data["medianbatch_msignal"], ((data["abs_avgbatch_msignal"]) - (((((data["rangebatch_slices2"]) - (data["abs_avgbatch_msignal"]))) / 2.0))) )) - (data["minbatch_msignal"]))))) +

                            0.100000*np.tanh((((np.sin((data["abs_minbatch_slices2"]))) + (np.where(((((data["maxbatch_msignal"]) + (data["meanbatch_msignal"]))) - (((((np.sin((np.where((-((data["rangebatch_msignal"]))) > -998, (6.0), np.sin((data["minbatch_msignal"])) )))) / 2.0)) / 2.0))) <= -998, data["maxbatch_slices2_msignal"], data["rangebatch_slices2"] )))/2.0)) +

                            0.100000*np.tanh(((((np.where(np.where(data["signal_shift_-1_msignal"] <= -998, data["stdbatch_msignal"], np.cos((((data["minbatch_msignal"]) * (np.cos((((((1.0)) + (((data["minbatch_msignal"]) + ((2.0)))))/2.0))))))) ) > -998, (-3.0), data["abs_minbatch_slices2_msignal"] )) + ((-((data["stdbatch_msignal"])))))) - (data["meanbatch_msignal"]))) +

                            0.100000*np.tanh(((((((-3.0)) + ((-1.0)))/2.0)) - (((((((data["meanbatch_slices2_msignal"]) * (np.tanh((np.sin((data["abs_maxbatch_slices2_msignal"]))))))) * (np.where((-((data["meanbatch_slices2_msignal"]))) <= -998, data["abs_maxbatch_slices2_msignal"], (-((data["abs_minbatch_slices2_msignal"]))) )))) + (((data["abs_maxbatch_slices2_msignal"]) * 2.0)))))) +

                            0.100000*np.tanh(((np.sin((data["minbatch_msignal"]))) * (np.where(np.cos(((-((data["maxbatch_slices2_msignal"]))))) > -998, (((data["abs_avgbatch_slices2"]) + (data["abs_maxbatch_slices2_msignal"]))/2.0), np.where(data["abs_avgbatch_msignal"] <= -998, np.cos((data["stdbatch_msignal"])), data["abs_maxbatch_msignal"] ) )))) +

                            0.100000*np.tanh(((data["medianbatch_slices2_msignal"]) * (np.where(((data["abs_avgbatch_slices2_msignal"]) / 2.0) <= -998, np.sin((data["rangebatch_slices2"])), np.sin((data["rangebatch_slices2"])) )))) +

                            0.100000*np.tanh(((data["rangebatch_slices2"]) - (np.where(np.cos((data["abs_minbatch_slices2"])) > -998, (9.59055137634277344), np.where(data["abs_minbatch_slices2"] > -998, (9.59055137634277344), ((np.where(data["rangebatch_slices2"] <= -998, np.where((-1.0) <= -998, np.where(data["maxtominbatch_slices2"] > -998, data["mean_abs_chgbatch_msignal"], data["maxtominbatch_slices2_msignal"] ), ((data["rangebatch_slices2"]) - ((9.59055137634277344))) ), data["signal_shift_+1"] )) * 2.0) ) )))) +

                            0.100000*np.tanh(((((2.0)) + (np.where((2.0) > -998, np.where((2.0) > -998, (3.0), (((((-((((data["abs_maxbatch"]) / 2.0))))) * (data["abs_maxbatch_slices2"]))) - (((data["stdbatch_slices2"]) + (data["minbatch_slices2_msignal"])))) ), data["stdbatch_msignal"] )))/2.0)) +

                            0.100000*np.tanh((-((((((np.cos((((np.where(data["abs_maxbatch"] > -998, (-((((data["meanbatch_msignal"]) * 2.0)))), data["meanbatch_msignal"] )) / 2.0)))) + (data["meanbatch_msignal"]))) * 2.0))))) +

                            0.100000*np.tanh(((np.sin((np.where(data["signal"] <= -998, ((((((data["minbatch_msignal"]) * (np.cos(((0.0)))))) / 2.0)) - (data["abs_maxbatch_slices2_msignal"])), data["stdbatch_slices2_msignal"] )))) - (((np.where((-((((data["maxbatch_slices2"]) / 2.0)))) <= -998, data["maxtominbatch_slices2"], data["medianbatch_slices2"] )) * (np.cos((data["meanbatch_slices2_msignal"]))))))) +

                            0.100000*np.tanh((((((3.0)) / 2.0)) * ((((((-(((-((data["stdbatch_msignal"]))))))) / 2.0)) + (((data["stdbatch_slices2"]) - (((data["meanbatch_msignal"]) * 2.0)))))))) +

                            0.100000*np.tanh((((-3.0)) * ((((7.0)) - (((data["minbatch_slices2_msignal"]) * (np.where(((((data["minbatch_slices2_msignal"]) + (np.tanh(((((data["minbatch_msignal"]) + ((-3.0)))/2.0)))))) + (((data["minbatch_slices2_msignal"]) * (np.tanh(((-3.0))))))) <= -998, (-1.0), np.tanh(((-3.0))) )))))))) +

                            0.100000*np.tanh((-((((np.cos((np.where(np.cos((data["maxbatch_msignal"])) > -998, np.where(data["medianbatch_slices2"] > -998, data["minbatch_msignal"], (-((((((np.where(((((data["maxbatch_msignal"]) * 2.0)) / 2.0) > -998, data["signal_shift_-1"], (-1.0) )) * 2.0)) * 2.0)))) ), np.tanh((data["abs_maxbatch_msignal"])) )))) * 2.0))))) +

                            0.100000*np.tanh(((((data["mean_abs_chgbatch_slices2"]) * (((data["meanbatch_slices2"]) * ((-(((-((data["maxbatch_slices2_msignal"]))))))))))) * (((np.sin(((((-((data["abs_maxbatch_slices2_msignal"])))) * 2.0)))) * (data["rangebatch_slices2"]))))) +

                            0.100000*np.tanh((-(((((9.52286720275878906)) * (np.cos((np.where(np.tanh((data["abs_minbatch_slices2_msignal"])) > -998, data["medianbatch_msignal"], np.where(data["medianbatch_msignal"] <= -998, data["medianbatch_msignal"], np.where(data["mean_abs_chgbatch_msignal"] > -998, data["medianbatch_msignal"], np.tanh((data["abs_maxbatch_slices2_msignal"])) ) ) ))))))))) +

                            0.100000*np.tanh(np.where(np.tanh((np.where(np.tanh(((3.0))) > -998, ((data["minbatch_slices2_msignal"]) / 2.0), ((((((((data["minbatch_slices2_msignal"]) / 2.0)) / 2.0)) + (((((data["minbatch_msignal"]) + (data["minbatch"]))) / 2.0)))) + (data["signal_shift_-1"])) ))) > -998, np.sin((((data["minbatch_slices2_msignal"]) / 2.0))), ((data["maxtominbatch"]) + (data["signal_shift_-1"])) )) +

                            0.100000*np.tanh((((-3.0)) * (np.cos((np.where((((-3.0)) * (np.sin((np.where(data["signal_shift_+1"] > -998, data["medianbatch_msignal"], ((np.tanh((data["abs_avgbatch_slices2_msignal"]))) - (((data["mean_abs_chgbatch_slices2"]) * (data["medianbatch_msignal"])))) ))))) > -998, data["medianbatch_msignal"], ((data["meanbatch_slices2_msignal"]) - (np.where((-3.0) > -998, data["medianbatch_msignal"], data["meanbatch_slices2_msignal"] ))) )))))) +

                            0.100000*np.tanh((-((np.cos((np.where(data["abs_minbatch_slices2_msignal"] <= -998, np.where(data["signal"] > -998, data["signal"], ((np.where(((data["maxtominbatch_msignal"]) / 2.0) <= -998, data["abs_maxbatch"], (1.0) )) * ((-((np.tanh(((1.0)))))))) ), (-((data["medianbatch_msignal"]))) ))))))) +

                            0.100000*np.tanh(((data["mean_abs_chgbatch_slices2_msignal"]) * (((np.sin(((((3.0)) * (np.where(((data["abs_maxbatch_msignal"]) * (np.sin((np.cos(((((3.0)) * (data["abs_maxbatch_msignal"])))))))) <= -998, data["rangebatch_msignal"], (((data["abs_maxbatch_msignal"]) + (np.cos((np.tanh((data["mean_abs_chgbatch_slices2_msignal"]))))))/2.0) )))))) * 2.0)))) +

                            0.100000*np.tanh(np.where(np.sin(((((-((data["abs_maxbatch_slices2"])))) / 2.0))) > -998, np.sin((((((data["abs_maxbatch_slices2"]) + (np.where(data["mean_abs_chgbatch_slices2"] > -998, data["stdbatch_slices2_msignal"], ((np.tanh((data["minbatch_slices2"]))) / 2.0) )))) * 2.0))), data["mean_abs_chgbatch_slices2"] )) +

                            0.100000*np.tanh(np.sin((data["minbatch_msignal"]))) +

                            0.100000*np.tanh(((((np.sin((np.where(((((data["medianbatch_msignal"]) * 2.0)) * (data["abs_maxbatch"])) <= -998, np.where(((data["signal"]) - (data["maxbatch_slices2"])) <= -998, data["mean_abs_chgbatch_slices2_msignal"], data["abs_maxbatch"] ), ((data["medianbatch_msignal"]) * 2.0) )))) * 2.0)) * (data["abs_maxbatch"]))) +

                            0.100000*np.tanh(((np.where(np.where(np.cos(((-1.0))) <= -998, data["minbatch_msignal"], ((((data["maxbatch_msignal"]) / 2.0)) * 2.0) ) <= -998, np.cos((data["maxbatch_msignal"])), np.cos((data["maxbatch_msignal"])) )) * (np.where((-1.0) > -998, data["mean_abs_chgbatch_slices2_msignal"], data["abs_maxbatch_msignal"] )))) +

                            0.100000*np.tanh(((np.sin(((((data["minbatch_msignal"]) + (np.tanh(((-(((((((data["maxtominbatch"]) + (((((data["abs_minbatch_slices2_msignal"]) * 2.0)) - (np.sin((np.tanh((data["signal_shift_-1_msignal"]))))))))/2.0)) / 2.0))))))))/2.0)))) * 2.0)) +

                            0.100000*np.tanh(np.where(data["medianbatch_slices2_msignal"] > -998, np.where(((((data["meanbatch_slices2"]) / 2.0)) - (data["abs_maxbatch_slices2_msignal"])) <= -998, data["medianbatch_msignal"], ((((((((data["medianbatch_slices2_msignal"]) * (((data["medianbatch_slices2_msignal"]) * (data["abs_minbatch_msignal"]))))) * 2.0)) * (np.cos((data["abs_maxbatch_slices2_msignal"]))))) * 2.0) ), ((data["abs_maxbatch"]) + (((data["abs_maxbatch_slices2_msignal"]) / 2.0))) )) +

                            0.100000*np.tanh((((data["mean_abs_chgbatch_msignal"]) + ((-2.0)))/2.0)) +

                            0.100000*np.tanh(np.sin((np.where(np.where((-2.0) > -998, data["maxbatch_slices2_msignal"], np.where((((-(((0.0))))) * 2.0) <= -998, np.where((-((data["mean_abs_chgbatch_slices2_msignal"]))) > -998, (-2.0), data["signal"] ), data["maxtominbatch_msignal"] ) ) <= -998, (((-2.0)) * ((((data["minbatch"]) + (data["rangebatch_slices2"]))/2.0))), (((-2.0)) * (data["maxbatch_slices2_msignal"])) )))) +

                            0.100000*np.tanh(((((data["mean_abs_chgbatch_slices2_msignal"]) * (np.cos((data["maxbatch_msignal"]))))) + ((((((((np.sin((((data["signal_shift_+1"]) / 2.0)))) + (np.tanh(((-2.0)))))) / 2.0)) + (np.tanh((data["rangebatch_slices2"]))))/2.0)))) +

                            0.100000*np.tanh((((-((data["abs_maxbatch_slices2_msignal"])))) / 2.0)) +

                            0.100000*np.tanh((((2.0)) * (np.sin((((np.where(((np.where(data["abs_avgbatch_slices2_msignal"] <= -998, data["maxbatch_msignal"], data["meanbatch_msignal"] )) * 2.0) <= -998, data["rangebatch_slices2_msignal"], data["meanbatch_msignal"] )) * 2.0)))))) +

                            0.100000*np.tanh(((data["abs_avgbatch_msignal"]) * (((np.cos((np.where(data["abs_avgbatch_slices2"] > -998, data["meanbatch_msignal"], ((data["abs_maxbatch_msignal"]) + (np.where(data["maxtominbatch"] > -998, data["maxtominbatch"], ((data["abs_avgbatch_msignal"]) * 2.0) ))) )))) * 2.0)))) +

                            0.100000*np.tanh(((data["abs_minbatch_slices2_msignal"]) - (np.where(((data["maxbatch_slices2_msignal"]) / 2.0) > -998, np.sin(((((((((np.sin((np.where(np.tanh((data["minbatch_msignal"])) > -998, (((6.96787595748901367)) * (data["minbatch_msignal"])), data["minbatch_slices2_msignal"] )))) - ((8.0)))) / 2.0)) + (data["abs_avgbatch_slices2_msignal"]))/2.0))), data["stdbatch_slices2_msignal"] )))) +

                            0.100000*np.tanh(np.where(((data["signal_shift_-1"]) * (np.sin(((((np.sin(((-(((-((data["abs_maxbatch_slices2"]))))))))) + (data["minbatch_slices2_msignal"]))/2.0))))) <= -998, data["maxbatch_slices2_msignal"], data["signal_shift_-1_msignal"] )) +

                            0.100000*np.tanh((0.0)) +

                            0.100000*np.tanh((-((((np.where(((np.where(((np.cos((data["maxbatch_slices2_msignal"]))) - (data["maxbatch_slices2_msignal"])) > -998, (0.0), (3.0) )) / 2.0) <= -998, (-(((8.0)))), np.cos((data["maxbatch_slices2_msignal"])) )) * 2.0))))) +

                            0.100000*np.tanh((((np.where((-((np.sin((np.where(data["signal_shift_+1_msignal"] <= -998, data["minbatch"], data["signal_shift_-1_msignal"] )))))) <= -998, data["meanbatch_slices2"], data["minbatch_slices2"] )) + (np.sin((np.sin(((-((np.where(np.where(data["minbatch_slices2"] <= -998, (0.0), (1.0) ) > -998, (((data["rangebatch_slices2_msignal"]) + (data["medianbatch_slices2_msignal"]))/2.0), (-2.0) ))))))))))/2.0)) +

                            0.100000*np.tanh(np.where(np.sin((np.tanh(((0.0))))) <= -998, data["abs_maxbatch_slices2"], ((np.cos((np.where(np.tanh((data["rangebatch_slices2_msignal"])) <= -998, data["meanbatch_slices2"], data["minbatch_msignal"] )))) * (data["stdbatch_slices2_msignal"])) )) +

                            0.100000*np.tanh((((((data["maxtominbatch_msignal"]) + (data["abs_maxbatch_slices2"]))/2.0)) / 2.0)) +

                            0.100000*np.tanh(np.sin((data["mean_abs_chgbatch_msignal"]))) +

                            0.100000*np.tanh(np.where(np.where(((((data["abs_maxbatch_msignal"]) + (data["mean_abs_chgbatch_slices2"]))) - (data["abs_avgbatch_slices2"])) <= -998, (((-2.0)) * (data["minbatch_slices2_msignal"])), (-((np.sin((data["mean_abs_chgbatch_slices2"]))))) ) <= -998, data["abs_maxbatch_msignal"], (-((np.cos((data["maxbatch_msignal"]))))) )) +

                            0.100000*np.tanh(np.sin((np.where(np.where(data["maxtominbatch_slices2"] > -998, (-((np.where(data["abs_maxbatch_slices2"] > -998, (-(((-3.0)))), data["medianbatch_msignal"] )))), np.where(data["abs_avgbatch_msignal"] <= -998, data["rangebatch_msignal"], ((data["abs_maxbatch_slices2"]) * 2.0) ) ) <= -998, ((data["rangebatch_slices2_msignal"]) * 2.0), (-((np.cos((data["signal_shift_+1_msignal"]))))) )))) +

                            0.100000*np.tanh(((((data["abs_avgbatch_slices2_msignal"]) * (np.cos((np.where(data["minbatch_msignal"] > -998, data["minbatch_msignal"], data["minbatch_msignal"] )))))) * 2.0)) +

                            0.100000*np.tanh(((((-3.0)) + (np.where(((data["abs_minbatch_slices2"]) * ((-((data["stdbatch_slices2"]))))) > -998, np.sin((((np.cos(((((data["rangebatch_slices2"]) + ((((((((3.0)) / 2.0)) + (((data["medianbatch_slices2"]) / 2.0)))) / 2.0)))/2.0)))) * (((data["abs_maxbatch_slices2_msignal"]) + (np.sin(((-2.0))))))))), np.cos((data["stdbatch_msignal"])) )))/2.0)) +

                            0.100000*np.tanh(((np.cos((((np.where((-2.0) > -998, data["maxbatch_msignal"], ((np.where(data["abs_avgbatch_msignal"] <= -998, data["rangebatch_slices2_msignal"], (-((np.sin(((((((2.0)) * 2.0)) / 2.0)))))) )) * 2.0) )) * 2.0)))) * 2.0)) +

                            0.100000*np.tanh(((np.tanh((data["abs_maxbatch_msignal"]))) * 2.0)) +

                            0.100000*np.tanh(np.where(np.sin(((-2.0))) <= -998, np.cos((((data["maxtominbatch_slices2"]) * 2.0))), (((data["maxtominbatch_slices2"]) + (((((np.tanh((data["signal_shift_-1_msignal"]))) - (np.tanh((data["maxbatch_slices2_msignal"]))))) / 2.0)))/2.0) )) +

                            0.100000*np.tanh(np.sin((np.where(((((-((data["abs_maxbatch_msignal"])))) + (np.tanh((np.where(data["minbatch_msignal"] > -998, (1.0), (-((np.sin((np.tanh(((-(((((data["maxbatch_slices2"]) + (np.cos((data["signal_shift_+1"]))))/2.0))))))))))) )))))/2.0) > -998, data["signal_shift_+1_msignal"], ((data["signal_shift_+1_msignal"]) / 2.0) )))) +

                            0.100000*np.tanh(((data["signal_shift_-1"]) * (((np.where(data["mean_abs_chgbatch_slices2"] > -998, np.where(np.tanh(((-3.0))) > -998, data["signal_shift_-1"], data["stdbatch_slices2"] ), data["abs_avgbatch_slices2_msignal"] )) * 2.0)))) +

                            0.100000*np.tanh(((np.tanh((data["signal_shift_+1_msignal"]))) + ((((((data["signal_shift_+1_msignal"]) + (np.tanh(((1.0)))))/2.0)) * (np.where(data["maxtominbatch_slices2_msignal"] <= -998, (0.0), ((((data["abs_maxbatch_msignal"]) + ((((((data["mean_abs_chgbatch_slices2"]) * 2.0)) + (np.tanh((data["signal_shift_+1_msignal"]))))/2.0)))) * 2.0) )))))) +

                            0.100000*np.tanh((-((np.where(data["medianbatch_msignal"] > -998, np.where(data["abs_minbatch_slices2"] > -998, np.sin((np.sin((data["meanbatch_msignal"])))), np.sin((np.tanh((((data["rangebatch_msignal"]) * ((((-2.0)) * 2.0))))))) ), (((3.0)) - (data["signal_shift_+1"])) ))))) +

                            0.100000*np.tanh((-((np.sin((np.where(((((((data["mean_abs_chgbatch_slices2_msignal"]) * (np.tanh(((-((np.cos((data["rangebatch_slices2"])))))))))) / 2.0)) * 2.0) <= -998, (0.0), ((data["abs_maxbatch_slices2_msignal"]) * 2.0) ))))))) +

                            0.100000*np.tanh(np.sin((np.where(data["medianbatch_slices2_msignal"] <= -998, data["abs_avgbatch_slices2"], np.where(np.where(data["abs_avgbatch_slices2"] <= -998, data["maxbatch_msignal"], ((np.where(data["medianbatch_slices2"] > -998, data["meanbatch_msignal"], data["signal_shift_-1_msignal"] )) * 2.0) ) <= -998, (0.0), ((np.where((-3.0) > -998, np.sin((np.where((-3.0) <= -998, (-3.0), ((data["meanbatch_msignal"]) * 2.0) ))), (-2.0) )) * 2.0) ) )))) +

                            0.100000*np.tanh(((np.where((-3.0) > -998, (-2.0), (-(((-(((-(((6.0)))))))))) )) * (np.cos((np.where(((data["mean_abs_chgbatch_slices2"]) - ((-((data["abs_avgbatch_slices2_msignal"]))))) > -998, data["maxbatch_slices2_msignal"], (((((((data["maxbatch_slices2_msignal"]) * 2.0)) + ((((data["minbatch_msignal"]) + (data["meanbatch_slices2_msignal"]))/2.0)))/2.0)) * 2.0) )))))) +

                            0.100000*np.tanh(np.cos(((-((np.where(((data["minbatch"]) / 2.0) <= -998, np.sin((data["maxtominbatch_slices2_msignal"])), (-3.0) ))))))) +

                            0.100000*np.tanh(((data["maxtominbatch"]) - (np.where(((np.tanh(((-(((-(((-((data["mean_abs_chgbatch_slices2_msignal"])))))))))))) / 2.0) > -998, data["mean_abs_chgbatch_slices2_msignal"], np.where((-((((data["medianbatch_slices2"]) - (data["abs_maxbatch_slices2_msignal"]))))) <= -998, data["maxtominbatch"], data["abs_maxbatch_msignal"] ) )))) +

                            0.100000*np.tanh((((-((np.sin((((np.where(data["meanbatch_msignal"] > -998, data["maxbatch_slices2_msignal"], (-((np.cos((data["maxtominbatch"]))))) )) * 2.0))))))) * 2.0)) +

                            0.100000*np.tanh(((data["maxbatch_msignal"]) - (data["rangebatch_slices2"]))) +

                            0.100000*np.tanh((((data["signal_shift_+1"]) + (data["minbatch_slices2"]))/2.0)) +

                            0.100000*np.tanh((((((2.25989747047424316)) * (np.where(data["stdbatch_msignal"] <= -998, np.cos((np.where(data["maxtominbatch"] <= -998, data["rangebatch_slices2"], (-(((((-((((((0.0)) + (data["minbatch_slices2"]))/2.0))))) * 2.0)))) ))), (((((((3.0)) - (data["maxbatch_slices2"]))) * (np.sin((data["meanbatch_msignal"]))))) - (data["abs_maxbatch_slices2_msignal"])) )))) / 2.0)) +

                            0.100000*np.tanh((-(((((data["abs_avgbatch_slices2_msignal"]) + (np.cos((np.sin((np.where(((np.cos((data["medianbatch_slices2_msignal"]))) + ((-((data["mean_abs_chgbatch_slices2_msignal"]))))) > -998, (-2.0), np.sin(((((data["abs_avgbatch_msignal"]) + (np.cos((data["minbatch"]))))/2.0))) )))))))/2.0))))) +

                            0.100000*np.tanh(np.where(((np.where(data["minbatch_msignal"] <= -998, data["maxtominbatch_slices2_msignal"], data["mean_abs_chgbatch_slices2"] )) * 2.0) > -998, data["signal_shift_-1_msignal"], np.where(((((data["abs_maxbatch_slices2"]) * (((((data["maxbatch_msignal"]) / 2.0)) * 2.0)))) - ((-1.0))) > -998, data["medianbatch_slices2_msignal"], (-(((1.0)))) ) )) +

                            0.100000*np.tanh(np.cos((data["rangebatch_slices2_msignal"]))) +

                            0.100000*np.tanh(np.tanh((np.tanh((np.sin((np.where(data["maxbatch_msignal"] <= -998, np.tanh((((((3.0)) + ((-((np.where(data["signal"] <= -998, (3.0), (4.0) ))))))/2.0))), (((3.0)) * (data["maxtominbatch_slices2"])) )))))))) +

                            0.100000*np.tanh(np.tanh((((data["abs_minbatch_slices2"]) - (np.sin((((data["signal_shift_-1"]) - (data["mean_abs_chgbatch_slices2_msignal"]))))))))) +

                            0.100000*np.tanh((-((np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["meanbatch_msignal"], ((((data["signal_shift_-1"]) / 2.0)) * 2.0) ))))) +

                            0.100000*np.tanh(((((((np.where((2.0) > -998, ((data["stdbatch_slices2_msignal"]) - (data["abs_maxbatch_slices2"])), ((((data["rangebatch_slices2"]) - ((6.62983560562133789)))) * (np.sin((np.where(data["abs_avgbatch_msignal"] <= -998, data["signal_shift_+1_msignal"], data["rangebatch_msignal"] ))))) )) - (np.cos((data["abs_minbatch_slices2_msignal"]))))) / 2.0)) * (np.cos((np.where((2.0) <= -998, data["maxbatch_slices2"], data["meanbatch_msignal"] )))))) +

                            0.100000*np.tanh(((np.where((((data["mean_abs_chgbatch_slices2"]) + ((3.0)))/2.0) <= -998, (-1.0), np.cos((((np.where((((((np.where(data["maxbatch_msignal"] > -998, data["minbatch_msignal"], (-3.0) )) * (data["medianbatch_msignal"]))) + (np.where(data["meanbatch_slices2_msignal"] > -998, data["medianbatch_msignal"], np.cos((data["abs_maxbatch_slices2_msignal"])) )))/2.0) > -998, data["minbatch_msignal"], (-1.0) )) / 2.0))) )) * 2.0)) +

                            0.100000*np.tanh(((((np.cos((data["abs_minbatch_slices2"]))) * 2.0)) * (np.cos((np.where(((np.where(data["abs_minbatch_msignal"] > -998, np.sin(((2.0))), data["maxbatch_slices2_msignal"] )) / 2.0) <= -998, np.tanh((data["mean_abs_chgbatch_slices2_msignal"])), np.tanh(((-((np.cos((data["abs_maxbatch_slices2_msignal"]))))))) )))))) +

                            0.100000*np.tanh(np.sin((np.where(np.sin((data["minbatch_msignal"])) <= -998, np.sin((np.tanh((((np.sin((np.sin((data["abs_maxbatch_slices2_msignal"]))))) + (((data["abs_minbatch_slices2"]) - ((((data["minbatch_msignal"]) + (np.sin((np.cos((data["minbatch_slices2_msignal"]))))))/2.0))))))))), ((((((-2.0)) / 2.0)) + (data["minbatch_msignal"]))/2.0) )))) +

                            0.100000*np.tanh(np.where(np.where(data["signal_shift_+1_msignal"] > -998, np.sin((data["abs_maxbatch_slices2_msignal"])), ((np.sin((data["signal_shift_-1_msignal"]))) * (data["abs_minbatch_msignal"])) ) > -998, ((data["signal_shift_-1_msignal"]) * (((np.sin((data["abs_maxbatch_msignal"]))) * (data["minbatch_slices2_msignal"])))), np.cos((data["rangebatch_slices2_msignal"])) )) +

                            0.100000*np.tanh(((np.sin((np.where(((data["abs_minbatch_msignal"]) * (np.tanh((np.sin((np.sin((np.sin((data["rangebatch_msignal"])))))))))) <= -998, np.where((-3.0) <= -998, data["abs_maxbatch_slices2_msignal"], data["mean_abs_chgbatch_msignal"] ), data["rangebatch_msignal"] )))) * (data["rangebatch_msignal"]))) +

                            0.100000*np.tanh(np.cos((data["maxtominbatch_slices2"]))) +

                            0.100000*np.tanh(np.sin((np.sin((data["stdbatch_slices2"]))))) +

                            0.100000*np.tanh((((((np.sin((data["abs_maxbatch"]))) * 2.0)) + (((((((data["maxtominbatch"]) / 2.0)) * 2.0)) / 2.0)))/2.0)) +

                            0.100000*np.tanh(np.cos((((((np.sin((np.sin(((-((data["mean_abs_chgbatch_slices2_msignal"])))))))) / 2.0)) * 2.0)))) +

                            0.100000*np.tanh(((((1.0)) + (np.where((((np.cos(((-3.0)))) + (data["abs_minbatch_slices2"]))/2.0) > -998, data["minbatch_msignal"], data["minbatch_msignal"] )))/2.0)) +

                            0.100000*np.tanh(np.where(((data["medianbatch_slices2_msignal"]) + ((((np.sin(((3.0)))) + (data["abs_maxbatch_msignal"]))/2.0))) <= -998, ((np.cos((data["minbatch_slices2"]))) * (np.where((-(((-(((-2.0))))))) <= -998, data["maxtominbatch_slices2_msignal"], data["medianbatch_msignal"] ))), ((data["signal_shift_-1_msignal"]) - (data["medianbatch_msignal"])) )) +

                            0.100000*np.tanh(((data["maxtominbatch_slices2"]) + (((data["maxtominbatch_msignal"]) / 2.0)))) +

                            0.100000*np.tanh(((data["minbatch_slices2"]) + (np.sin((np.cos((data["meanbatch_slices2"]))))))) +

                            0.100000*np.tanh(np.sin((((data["medianbatch_msignal"]) * (np.where(((data["rangebatch_slices2_msignal"]) + ((-2.0))) > -998, data["abs_maxbatch_slices2_msignal"], np.sin((np.cos(((1.0))))) )))))) +

                            0.100000*np.tanh(((data["maxbatch_slices2_msignal"]) * ((-((((np.where(data["minbatch"] > -998, data["maxbatch_slices2_msignal"], ((np.where(data["maxbatch_slices2_msignal"] <= -998, (((data["minbatch"]) + (data["meanbatch_slices2"]))/2.0), ((((data["abs_maxbatch_slices2"]) / 2.0)) / 2.0) )) - (data["maxbatch_slices2_msignal"])) )) + (np.where(((data["minbatch"]) * 2.0) > -998, data["minbatch"], (5.0) ))))))))) +

                            0.100000*np.tanh(np.where(((data["meanbatch_slices2"]) - (((data["mean_abs_chgbatch_slices2_msignal"]) - (((data["abs_minbatch_slices2"]) - ((((data["signal_shift_-1_msignal"]) + ((((-1.0)) / 2.0)))/2.0))))))) > -998, data["signal_shift_-1_msignal"], (((((np.sin((data["mean_abs_chgbatch_msignal"]))) + ((0.0)))/2.0)) - ((((data["abs_maxbatch_msignal"]) + (data["meanbatch_slices2"]))/2.0))) )) +

                            0.100000*np.tanh(((((np.cos(((0.0)))) * (np.sin(((-((np.where(data["maxbatch_slices2"] > -998, np.cos((data["signal"])), np.where(data["signal_shift_-1_msignal"] > -998, np.sin((((data["meanbatch_slices2_msignal"]) + ((4.72090721130371094))))), ((data["rangebatch_msignal"]) * 2.0) ) ))))))))) * (data["signal_shift_+1"]))) +

                            0.100000*np.tanh(np.where(data["stdbatch_slices2_msignal"] <= -998, ((np.sin((((((((2.0)) / 2.0)) + (data["maxtominbatch_slices2"]))/2.0)))) - (data["stdbatch_slices2"])), ((np.sin((np.sin(((-1.0)))))) * 2.0) )) +

                            0.100000*np.tanh(((((((data["mean_abs_chgbatch_msignal"]) + (((np.where(data["maxbatch_slices2_msignal"] > -998, data["rangebatch_slices2_msignal"], data["maxtominbatch_msignal"] )) / 2.0)))/2.0)) + (((data["signal_shift_-1"]) + (((data["signal_shift_+1"]) - ((8.0)))))))/2.0)) +

                            0.100000*np.tanh(((data["medianbatch_msignal"]) * (np.cos((np.where(((((((((2.60863852500915527)) * 2.0)) + (np.cos(((((2.60863852500915527)) * 2.0)))))/2.0)) / 2.0) > -998, data["maxbatch_msignal"], ((((-((((((data["abs_maxbatch"]) * 2.0)) / 2.0))))) + (np.sin((data["rangebatch_msignal"]))))/2.0) )))))) +

                            0.100000*np.tanh(np.cos((np.sin(((2.0)))))) +

                            0.100000*np.tanh(((np.sin((np.cos((np.where(np.tanh((data["mean_abs_chgbatch_msignal"])) > -998, data["maxtominbatch_msignal"], np.cos((np.cos((((((((data["abs_minbatch_msignal"]) * 2.0)) * (data["minbatch_msignal"]))) * 2.0))))) )))))) / 2.0)) +

                            0.100000*np.tanh((-((np.where(np.tanh((((data["minbatch_msignal"]) * ((2.0))))) > -998, data["abs_minbatch_slices2"], np.sin((((data["abs_maxbatch_slices2_msignal"]) * (np.where(data["abs_minbatch_slices2"] > -998, data["abs_minbatch_slices2"], ((data["minbatch_slices2"]) - (np.tanh(((((0.0)) * 2.0))))) ))))) ))))) +

                            0.100000*np.tanh(np.cos((np.where(data["minbatch"] > -998, data["rangebatch_slices2_msignal"], np.where(data["abs_maxbatch_msignal"] > -998, data["mean_abs_chgbatch_msignal"], np.where(np.where((1.0) > -998, data["rangebatch_msignal"], np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_msignal"], data["rangebatch_slices2_msignal"] ) ) > -998, np.where(data["meanbatch_slices2"] > -998, data["maxtominbatch_msignal"], data["maxtominbatch_msignal"] ), np.cos((data["rangebatch_slices2"])) ) ) )))) +

                            0.100000*np.tanh(np.tanh((np.cos((((data["maxbatch_msignal"]) * 2.0)))))) +

                            0.100000*np.tanh(np.sin((np.tanh(((-3.0)))))) +

                            0.100000*np.tanh(np.where(data["signal_shift_-1"] > -998, (((-((data["maxbatch_msignal"])))) * (np.sin((np.cos((((data["signal_shift_-1"]) - (np.where(data["signal_shift_+1"] <= -998, np.cos((((data["signal_shift_-1"]) - (data["maxbatch_slices2"])))), data["maxbatch_slices2"] ))))))))), data["maxbatch_slices2"] )) +

                            0.100000*np.tanh((((((((((np.where(data["medianbatch_slices2_msignal"] > -998, ((((data["medianbatch_slices2"]) * (data["medianbatch_slices2"]))) / 2.0), data["meanbatch_msignal"] )) + (np.cos((np.sin((data["medianbatch_msignal"]))))))/2.0)) * 2.0)) * 2.0)) * (((data["maxtominbatch_slices2"]) + (((((data["signal_shift_+1"]) * 2.0)) * 2.0)))))) +

                            0.100000*np.tanh(np.where(np.cos((((((data["abs_avgbatch_slices2"]) / 2.0)) * (data["maxbatch_msignal"])))) <= -998, data["signal_shift_+1_msignal"], data["signal_shift_-1_msignal"] )) +

                            0.100000*np.tanh(data["rangebatch_slices2_msignal"]) +

                            0.100000*np.tanh(np.sin((np.where(data["minbatch_msignal"] > -998, data["minbatch_msignal"], data["minbatch_msignal"] )))) +

                            0.100000*np.tanh(np.cos((((((data["signal_shift_-1"]) - ((-((data["meanbatch_msignal"])))))) - (np.cos((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, ((data["maxtominbatch"]) * (data["rangebatch_msignal"])), ((data["mean_abs_chgbatch_slices2_msignal"]) - (np.sin((np.where(data["rangebatch_msignal"] <= -998, data["rangebatch_slices2_msignal"], data["rangebatch_msignal"] ))))) )))))))) +

                            0.100000*np.tanh(np.where(np.cos((np.tanh((data["rangebatch_slices2"])))) <= -998, np.sin((((data["signal_shift_-1"]) / 2.0))), np.where(data["signal_shift_-1"] <= -998, (1.73182046413421631), (((data["maxtominbatch_slices2_msignal"]) + (((data["signal_shift_-1"]) + (data["rangebatch_slices2"]))))/2.0) ) )) +

                            0.100000*np.tanh(np.tanh((((data["signal_shift_+1_msignal"]) - (data["signal_shift_+1_msignal"]))))) +

                            0.100000*np.tanh(((data["minbatch_slices2"]) / 2.0)) +

                            0.100000*np.tanh(np.sin((((((data["minbatch_msignal"]) + (np.where(np.tanh(((3.0))) <= -998, np.where((-((np.sin((np.sin((np.cos((data["abs_maxbatch_slices2_msignal"]))))))))) > -998, data["abs_maxbatch_slices2_msignal"], ((data["abs_minbatch_slices2"]) * 2.0) ), data["stdbatch_slices2"] )))) / 2.0)))) +

                            0.100000*np.tanh(np.tanh(((((data["minbatch_msignal"]) + (((data["signal_shift_-1_msignal"]) - (((data["abs_avgbatch_slices2"]) * (np.where(data["rangebatch_slices2_msignal"] <= -998, ((data["minbatch"]) - (((data["meanbatch_msignal"]) * (np.where((0.0) <= -998, np.tanh((data["meanbatch_slices2_msignal"])), (-((data["minbatch_msignal"]))) ))))), data["minbatch"] )))))))/2.0)))) +

                            0.100000*np.tanh(np.cos((np.where(((np.where(data["abs_maxbatch_slices2"] > -998, data["minbatch_msignal"], data["abs_maxbatch"] )) / 2.0) <= -998, np.tanh((data["signal_shift_-1_msignal"])), ((np.where(((np.where(data["minbatch_slices2_msignal"] > -998, data["minbatch_msignal"], np.tanh((np.where(data["signal_shift_-1_msignal"] <= -998, (-2.0), data["abs_maxbatch_msignal"] ))) )) * 2.0) > -998, data["minbatch_msignal"], np.sin((data["minbatch_msignal"])) )) / 2.0) )))) +

                            0.100000*np.tanh(np.tanh((np.sin((np.where(data["stdbatch_slices2"] <= -998, data["stdbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] )))))) +

                            0.100000*np.tanh(np.where(data["rangebatch_msignal"] > -998, data["mean_abs_chgbatch_slices2"], np.cos((((np.where(data["abs_minbatch_slices2_msignal"] > -998, (1.0), np.cos((np.sin(((((data["maxtominbatch_slices2"]) + (np.tanh(((1.0)))))/2.0))))) )) - ((-((data["minbatch"]))))))) )) +

                            0.100000*np.tanh(np.sin((np.sin((((np.where(data["signal_shift_-1_msignal"] > -998, np.sin(((0.0))), ((np.sin((np.sin(((-(((-((data["signal_shift_-1_msignal"]))))))))))) * 2.0) )) * 2.0)))))) +

                            0.100000*np.tanh(np.sin((((np.cos((((np.cos((np.sin((((data["abs_maxbatch_slices2_msignal"]) / 2.0)))))) - (np.sin(((((8.85962772369384766)) + (((np.where(data["medianbatch_msignal"] <= -998, (1.0), (-(((-((((data["medianbatch_msignal"]) / 2.0))))))) )) / 2.0)))))))))) / 2.0)))) +

                            0.100000*np.tanh((-((np.tanh((data["rangebatch_slices2_msignal"])))))) +

                            0.100000*np.tanh(((np.where(np.cos((data["meanbatch_slices2_msignal"])) <= -998, ((data["maxbatch_slices2"]) - (data["mean_abs_chgbatch_slices2"])), np.tanh((np.cos(((((np.where(data["meanbatch_slices2_msignal"] > -998, (12.17550086975097656), data["abs_maxbatch_slices2"] )) + (data["minbatch_msignal"]))/2.0))))) )) * 2.0)) +

                            0.100000*np.tanh(np.cos((np.cos((np.tanh(((((-((data["signal_shift_+1_msignal"])))) / 2.0)))))))) +

                            0.100000*np.tanh(np.sin(((-1.0)))) +

                            0.100000*np.tanh(np.sin((((data["meanbatch_msignal"]) - (np.where(data["abs_maxbatch"] > -998, data["minbatch_msignal"], ((((data["minbatch_msignal"]) + (data["rangebatch_msignal"]))) - (np.where(data["maxbatch_slices2_msignal"] > -998, np.sin((((data["maxbatch_msignal"]) - (((((((data["maxbatch_slices2_msignal"]) / 2.0)) - (np.where(data["stdbatch_slices2"] > -998, data["minbatch_msignal"], data["minbatch_msignal"] )))) / 2.0))))), (2.0) ))) )))))) +

                            0.100000*np.tanh(np.where(((((((np.tanh((data["mean_abs_chgbatch_msignal"]))) / 2.0)) * (data["stdbatch_msignal"]))) + (data["mean_abs_chgbatch_slices2"])) <= -998, ((np.tanh((data["mean_abs_chgbatch_slices2"]))) * 2.0), (-((np.cos((((np.cos((np.cos((data["maxbatch_msignal"]))))) * (data["maxbatch_msignal"]))))))) )) +

                            0.100000*np.tanh(((data["meanbatch_slices2_msignal"]) * (((np.where(((np.cos((((np.sin((np.tanh((data["minbatch_slices2"]))))) / 2.0)))) / 2.0) > -998, data["medianbatch_msignal"], ((data["abs_avgbatch_msignal"]) / 2.0) )) * (np.cos((np.sin((np.sin(((-((np.sin((np.sin((np.sin((np.sin(((((4.0)) + ((-2.0))))))))))))))))))))))))) +

                            0.100000*np.tanh((((((-3.0)) * (((np.sin((np.where(((np.where((((data["abs_minbatch_slices2"]) + (data["signal_shift_+1_msignal"]))/2.0) <= -998, data["abs_maxbatch_msignal"], data["signal_shift_+1_msignal"] )) * (data["maxbatch_slices2_msignal"])) > -998, data["abs_maxbatch_msignal"], (-((data["abs_maxbatch_msignal"]))) )))) * 2.0)))) * 2.0)) +

                            0.100000*np.tanh(np.where(data["abs_minbatch_slices2_msignal"] <= -998, np.where(data["minbatch"] <= -998, ((np.where(data["minbatch"] <= -998, np.where(data["stdbatch_msignal"] <= -998, np.sin((data["stdbatch_msignal"])), data["minbatch_slices2_msignal"] ), data["stdbatch_slices2"] )) * 2.0), np.sin(((((np.sin((data["abs_minbatch_slices2_msignal"]))) + (data["stdbatch_msignal"]))/2.0))) ), (-3.0) )) +

                            0.100000*np.tanh(((((((((data["signal_shift_+1"]) - (((data["abs_minbatch_slices2"]) - (data["medianbatch_msignal"]))))) - (np.sin((np.where(data["abs_minbatch_slices2"] <= -998, data["abs_maxbatch_slices2_msignal"], data["abs_minbatch_slices2"] )))))) - (data["maxbatch_slices2_msignal"]))) - ((-((data["abs_maxbatch_msignal"])))))) +

                            0.100000*np.tanh(np.cos((((data["mean_abs_chgbatch_msignal"]) * (np.where(((np.tanh((data["signal_shift_+1"]))) * (data["signal_shift_-1"])) > -998, data["abs_maxbatch_slices2_msignal"], data["maxbatch_msignal"] )))))) +

                            0.100000*np.tanh(np.where(((np.sin((np.where((((data["mean_abs_chgbatch_msignal"]) + ((0.0)))/2.0) <= -998, data["minbatch_slices2"], ((np.sin((data["abs_minbatch_msignal"]))) / 2.0) )))) / 2.0) > -998, np.sin((((data["stdbatch_msignal"]) - (data["maxbatch_slices2_msignal"])))), (((data["stdbatch_msignal"]) + (data["minbatch_slices2_msignal"]))/2.0) )) +

                            0.100000*np.tanh(data["signal_shift_+1_msignal"]) +

                            0.100000*np.tanh(np.cos((np.where(((np.where(np.sin((data["abs_maxbatch_msignal"])) > -998, data["maxbatch_slices2"], np.sin(((((1.0)) * 2.0))) )) / 2.0) <= -998, ((np.where(data["rangebatch_msignal"] > -998, np.sin((((((data["maxtominbatch_slices2"]) + (data["maxbatch_slices2"]))) + (data["rangebatch_msignal"])))), data["minbatch_slices2_msignal"] )) / 2.0), data["rangebatch_msignal"] )))) +

                            0.100000*np.tanh((((((data["stdbatch_slices2"]) * 2.0)) + (np.cos(((1.0)))))/2.0)) +

                            0.100000*np.tanh(((((((data["rangebatch_slices2"]) + (data["stdbatch_msignal"]))) * (np.sin((((((((((((-((data["stdbatch_msignal"])))) / 2.0)) * ((-((np.tanh((data["rangebatch_slices2"])))))))) + (data["rangebatch_slices2"]))/2.0)) * (np.sin((np.cos((np.where((-(((-3.0)))) <= -998, (8.60173892974853516), data["meanbatch_slices2_msignal"] )))))))))))) * 2.0)) +

                            0.100000*np.tanh(data["abs_maxbatch_slices2_msignal"]) +

                            0.100000*np.tanh(((np.cos(((((data["minbatch_msignal"]) + (np.sin((((((data["minbatch_msignal"]) + (np.where((0.0) > -998, np.cos(((-(((((((np.cos(((0.0)))) + (np.cos((np.tanh(((-((data["minbatch_slices2_msignal"])))))))))/2.0)) + (((data["minbatch_msignal"]) * 2.0)))))))), data["minbatch_msignal"] )))) * (data["maxbatch_slices2"]))))))/2.0)))) * 2.0)) +

                            0.100000*np.tanh(np.cos((((data["maxtominbatch_msignal"]) - (np.tanh((np.where(np.tanh(((((data["minbatch_slices2_msignal"]) + (np.sin((((np.tanh(((-3.0)))) / 2.0)))))/2.0))) <= -998, ((np.tanh((np.tanh(((-3.0)))))) / 2.0), (((np.sin(((1.0)))) + (data["stdbatch_msignal"]))/2.0) )))))))) +

                            0.100000*np.tanh((((10.0)) * (np.cos((((np.where(((((-3.0)) + (((data["abs_minbatch_msignal"]) * 2.0)))/2.0) > -998, (10.0), (((data["maxbatch_slices2"]) + (data["abs_minbatch_msignal"]))/2.0) )) * (((((((data["maxbatch_slices2"]) + (data["abs_minbatch_msignal"]))/2.0)) + (data["abs_maxbatch"]))/2.0)))))))) +

                            0.100000*np.tanh(np.cos((np.where(data["abs_maxbatch_slices2"] > -998, np.where(np.where(data["minbatch"] > -998, data["abs_maxbatch_slices2"], data["meanbatch_slices2_msignal"] ) > -998, data["abs_maxbatch_slices2"], (10.88412094116210938) ), data["abs_maxbatch_slices2"] )))) +

                            0.100000*np.tanh(np.cos((((np.sin((((np.where(np.cos((np.sin((data["abs_maxbatch_msignal"])))) <= -998, (-3.0), data["rangebatch_slices2"] )) * 2.0)))) + (np.tanh((np.tanh((np.cos((np.sin((np.cos((data["stdbatch_msignal"]))))))))))))))) +

                            0.100000*np.tanh((((data["maxtominbatch"]) + (np.tanh((data["minbatch"]))))/2.0)) +

                            0.100000*np.tanh(((data["rangebatch_msignal"]) + (np.tanh((((((data["abs_maxbatch_msignal"]) + ((((np.cos((((((((data["rangebatch_slices2_msignal"]) * 2.0)) * (data["rangebatch_slices2"]))) + (data["rangebatch_msignal"]))))) + (data["rangebatch_msignal"]))/2.0)))) / 2.0)))))) +

                            0.100000*np.tanh((0.0)) +

                            0.100000*np.tanh(((((-3.0)) + (np.where(np.sin(((((-3.0)) - (data["abs_minbatch_slices2"])))) > -998, data["signal_shift_+1"], np.cos((((np.where(data["abs_avgbatch_msignal"] > -998, np.tanh(((3.0))), ((data["mean_abs_chgbatch_msignal"]) / 2.0) )) + (np.sin(((-(((0.69674152135848999)))))))))) )))/2.0)) +

                            0.100000*np.tanh(((np.sin(((-3.0)))) - (np.sin((data["medianbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh((3.0)) +

                            0.100000*np.tanh(((data["stdbatch_slices2"]) * (np.where(((data["maxbatch_slices2_msignal"]) + (((data["minbatch_slices2_msignal"]) * (np.sin((data["abs_maxbatch_slices2"])))))) <= -998, np.cos((((np.where(data["minbatch_slices2_msignal"] <= -998, data["medianbatch_slices2_msignal"], np.tanh(((-1.0))) )) / 2.0))), np.sin((data["minbatch_slices2_msignal"])) )))) +

                            0.100000*np.tanh(np.cos((np.where(np.where(np.cos((np.where((-(((-((data["minbatch"])))))) > -998, data["medianbatch_slices2_msignal"], ((data["abs_maxbatch_msignal"]) - (data["abs_maxbatch_slices2_msignal"])) ))) > -998, data["medianbatch_slices2_msignal"], ((data["mean_abs_chgbatch_msignal"]) * 2.0) ) > -998, data["medianbatch_slices2_msignal"], ((data["abs_maxbatch_slices2"]) * 2.0) )))) +

                            0.100000*np.tanh(np.cos((np.where(((data["abs_minbatch_slices2_msignal"]) * 2.0) > -998, data["maxtominbatch_msignal"], ((((np.cos((np.sin((np.cos((np.where(data["abs_maxbatch"] > -998, data["maxtominbatch_msignal"], data["maxtominbatch_msignal"] )))))))) * ((-1.0)))) * 2.0) )))) +

                            0.100000*np.tanh(np.where(data["signal_shift_+1"] <= -998, data["stdbatch_slices2"], (((np.where(data["maxbatch_msignal"] <= -998, data["rangebatch_slices2"], data["rangebatch_msignal"] )) + (np.tanh((((((data["signal_shift_+1"]) + (np.where(data["signal_shift_+1"] <= -998, data["stdbatch_slices2"], (((data["maxbatch_msignal"]) + (((((data["signal"]) / 2.0)) - (data["maxtominbatch_slices2"]))))/2.0) )))) * 2.0)))))/2.0) )) +

                            0.100000*np.tanh(np.sin(((((data["maxbatch_msignal"]) + (np.where(np.cos(((-1.0))) <= -998, np.sin((data["stdbatch_slices2"])), data["minbatch_msignal"] )))/2.0)))) +

                            0.100000*np.tanh(np.where(np.sin((data["maxtominbatch"])) <= -998, data["maxbatch_msignal"], ((((data["abs_minbatch_slices2_msignal"]) * 2.0)) * (((((np.sin((((np.where(data["abs_minbatch_slices2_msignal"] <= -998, data["maxbatch_msignal"], ((data["abs_minbatch_slices2_msignal"]) - (data["abs_maxbatch_slices2"])) )) * 2.0)))) * 2.0)) * 2.0))) )) +

                            0.100000*np.tanh(np.cos((np.cos((data["medianbatch_slices2_msignal"]))))) +

                            0.100000*np.tanh(((data["maxtominbatch_slices2"]) / 2.0)) +

                            0.100000*np.tanh(np.sin((((((1.0)) + ((-(((((np.where((0.45042169094085693) > -998, ((np.sin((data["mean_abs_chgbatch_slices2_msignal"]))) / 2.0), data["signal_shift_+1_msignal"] )) + (np.sin(((0.0)))))/2.0))))))/2.0)))) +

                            0.099609*np.tanh(((((data["abs_avgbatch_msignal"]) - ((3.0)))) / 2.0)) +

                            0.100000*np.tanh(((data["signal"]) * ((0.0)))) +

                            0.100000*np.tanh(np.where(data["signal_shift_-1_msignal"] > -998, np.tanh((((data["rangebatch_slices2_msignal"]) / 2.0))), np.cos((data["mean_abs_chgbatch_slices2"])) )) +

                            0.100000*np.tanh(((np.sin((np.where(np.cos((np.cos(((0.0))))) > -998, np.where(data["abs_maxbatch_slices2"] <= -998, ((np.sin((((((data["maxbatch_msignal"]) / 2.0)) / 2.0)))) / 2.0), (1.0) ), (((0.0)) * 2.0) )))) / 2.0)) +

                            0.100000*np.tanh(np.cos((data["mean_abs_chgbatch_slices2"]))) +

                            0.100000*np.tanh(np.where((((-((((data["abs_minbatch_msignal"]) + ((((-((((data["signal_shift_-1_msignal"]) / 2.0))))) + (np.sin((data["stdbatch_slices2"])))))))))) + (data["maxtominbatch"])) <= -998, ((np.cos((data["maxtominbatch"]))) / 2.0), (0.43098816275596619) )) +

                            0.100000*np.tanh(np.cos((((np.where(np.sin((((((np.cos((np.cos((np.tanh((data["meanbatch_msignal"]))))))) / 2.0)) / 2.0))) > -998, data["minbatch_msignal"], np.where(((data["medianbatch_slices2_msignal"]) / 2.0) > -998, data["minbatch_msignal"], ((np.where(np.where(data["medianbatch_slices2_msignal"] > -998, np.cos((data["stdbatch_slices2"])), (0.0) ) <= -998, data["stdbatch_slices2"], data["abs_maxbatch_slices2"] )) / 2.0) ) )) / 2.0)))) +

                            0.100000*np.tanh(np.tanh((np.sin((((((data["signal_shift_+1"]) + (np.where(((data["maxbatch_slices2_msignal"]) / 2.0) > -998, data["mean_abs_chgbatch_msignal"], np.sin((np.cos(((((data["stdbatch_slices2"]) + (((data["stdbatch_msignal"]) + (((((np.cos(((((3.0)) * 2.0)))) / 2.0)) * 2.0)))))/2.0))))) )))) / 2.0)))))) +

                            0.100000*np.tanh(np.sin((((data["abs_minbatch_msignal"]) + (np.where(np.sin(((0.0))) > -998, data["medianbatch_msignal"], ((data["minbatch"]) - ((-(((3.0)))))) )))))) +

                            0.100000*np.tanh(np.where(data["maxtominbatch_slices2_msignal"] <= -998, ((data["mean_abs_chgbatch_slices2"]) - (data["maxbatch_msignal"])), (((-((data["maxbatch_slices2_msignal"])))) * (np.cos((((data["abs_maxbatch_slices2_msignal"]) * ((-((data["meanbatch_slices2_msignal"]))))))))) )) +

                            0.100000*np.tanh(np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) * (np.where(np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) * (data["mean_abs_chgbatch_slices2_msignal"])))) > -998, data["rangebatch_msignal"], np.sin((((np.where(data["abs_maxbatch"] > -998, np.tanh((data["abs_maxbatch_msignal"])), ((data["mean_abs_chgbatch_slices2_msignal"]) * (data["mean_abs_chgbatch_slices2_msignal"])) )) * 2.0))) )))))) +

                            0.089932*np.tanh(((data["rangebatch_msignal"]) - ((-((np.where(((data["rangebatch_msignal"]) - ((-((np.where(((data["rangebatch_msignal"]) * 2.0) > -998, data["maxtominbatch"], np.sin((data["rangebatch_msignal"])) )))))) > -998, data["maxtominbatch"], data["mean_abs_chgbatch_slices2"] ))))))) +

                            0.100000*np.tanh(((np.sin(((-((data["minbatch_msignal"])))))) * (np.where(data["maxtominbatch_slices2"] <= -998, data["maxtominbatch_slices2"], np.cos((data["meanbatch_slices2_msignal"])) )))) +

                            0.098436*np.tanh(((np.cos((np.where((((-2.0)) - (data["signal_shift_-1_msignal"])) <= -998, data["stdbatch_slices2_msignal"], np.where((2.0) > -998, data["rangebatch_msignal"], np.where(data["signal_shift_-1_msignal"] <= -998, ((np.cos((data["rangebatch_msignal"]))) * 2.0), ((data["medianbatch_msignal"]) / 2.0) ) ) )))) / 2.0)) +

                            0.100000*np.tanh(((((np.sin((np.sin((data["rangebatch_slices2"]))))) * (data["abs_avgbatch_slices2_msignal"]))) * (((data["abs_maxbatch_msignal"]) - (data["abs_maxbatch_msignal"]))))) +

                            0.100000*np.tanh(np.tanh((((((np.tanh((((np.tanh((data["abs_maxbatch_slices2"]))) * ((((data["mean_abs_chgbatch_slices2"]) + (data["medianbatch_msignal"]))/2.0)))))) * (np.tanh(((-1.0)))))) / 2.0)))) +

                            0.100000*np.tanh(np.where(((data["maxtominbatch_slices2_msignal"]) * 2.0) > -998, np.sin((data["maxbatch_slices2_msignal"])), (((-3.0)) + (((data["medianbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"])))) )) +

                            0.100000*np.tanh(np.cos(((-((np.tanh((np.where((((2.0)) * 2.0) > -998, np.cos((((np.tanh((data["signal_shift_+1"]))) / 2.0))), data["maxbatch_msignal"] ))))))))) +

                            0.100000*np.tanh(np.sin((np.where(((((((data["abs_maxbatch_msignal"]) + ((-(((0.0))))))/2.0)) + (np.cos((data["mean_abs_chgbatch_msignal"]))))/2.0) > -998, np.sin((np.cos((data["maxbatch_slices2"])))), data["meanbatch_slices2"] )))) +

                            0.100000*np.tanh(((np.sin((np.cos((np.tanh(((((-((((np.tanh(((0.0)))) * 2.0))))) / 2.0)))))))) / 2.0)) +

                            0.100000*np.tanh(np.where(data["signal_shift_-1_msignal"] <= -998, np.tanh(((2.0))), (((0.0)) * ((2.0))) )) +

                            0.093744*np.tanh(np.where((0.0) <= -998, np.where(np.cos((data["maxtominbatch_msignal"])) > -998, (-(((((((np.tanh((data["stdbatch_slices2_msignal"]))) + (data["maxbatch_msignal"]))/2.0)) * 2.0)))), (((((data["stdbatch_slices2_msignal"]) + (((data["medianbatch_msignal"]) + (data["medianbatch_msignal"]))))/2.0)) + (data["maxbatch_slices2"])) ), data["rangebatch_msignal"] )) +

                            0.100000*np.tanh(np.where(np.tanh((data["signal_shift_+1_msignal"])) > -998, (-1.0), ((np.where((((-3.0)) + (np.where(data["signal_shift_-1_msignal"] > -998, (2.0), (((((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0)) + ((0.0)))/2.0) ))) <= -998, (-3.0), ((((-3.0)) + ((-3.0)))/2.0) )) * 2.0) )) +

                            0.099707*np.tanh(((np.cos(((-(((((0.0)) - (np.tanh((data["meanbatch_msignal"])))))))))) / 2.0)) +

                            0.100000*np.tanh((((0.0)) * (np.sin((np.tanh((data["stdbatch_msignal"]))))))) +

                            0.100000*np.tanh(np.cos((np.where(((np.sin((data["medianbatch_msignal"]))) / 2.0) <= -998, (((((data["minbatch_msignal"]) + (data["medianbatch_slices2"]))/2.0)) * (data["abs_maxbatch"])), (((data["minbatch_msignal"]) + (((np.sin((data["medianbatch_msignal"]))) * 2.0)))/2.0) )))) +

                            0.100000*np.tanh((((-((np.sin((np.tanh(((((0.0)) * (((((((((0.0)) + (data["stdbatch_slices2_msignal"]))/2.0)) / 2.0)) - (np.sin(((-1.0))))))))))))))) / 2.0)))    

    

    def GP_class_10(self,data):

        return self.Output( -4.940363 +

                        0.100000*np.tanh((((((3.0)) - ((7.0)))) * 2.0)) +

                        0.100000*np.tanh((-((np.where(data["medianbatch_slices2_msignal"] <= -998, data["abs_maxbatch_slices2_msignal"], data["abs_minbatch_slices2_msignal"] ))))) +

                        0.100000*np.tanh((-(((((np.tanh((((np.where((-3.0) > -998, (-3.0), data["signal"] )) * 2.0)))) + (data["medianbatch_slices2_msignal"]))/2.0))))) +

                        0.100000*np.tanh(np.sin(((((-1.0)) * 2.0)))) +

                        0.100000*np.tanh(np.cos(((((data["minbatch_slices2"]) + (np.sin(((-(((-(((-((data["mean_abs_chgbatch_msignal"])))))))))))))/2.0)))) +

                        0.100000*np.tanh((((((data["meanbatch_slices2"]) + ((-(((((3.0)) - ((-((np.tanh((data["meanbatch_slices2"]))))))))))))/2.0)) + ((-((np.tanh((np.cos((((data["meanbatch_slices2_msignal"]) + (((np.where(data["minbatch_slices2_msignal"] > -998, (((3.0)) + (np.tanh((np.cos((data["maxtominbatch"])))))), data["signal_shift_-1_msignal"] )) / 2.0))))))))))))) +

                        0.100000*np.tanh(np.where((-((data["stdbatch_msignal"]))) <= -998, np.tanh(((6.0))), data["maxtominbatch_slices2_msignal"] )) +

                        0.100000*np.tanh(np.tanh((data["maxtominbatch_slices2"]))) +

                        0.100000*np.tanh(((np.cos((np.where(data["medianbatch_msignal"] > -998, data["mean_abs_chgbatch_slices2_msignal"], (-1.0) )))) * (data["abs_maxbatch_msignal"]))) +

                        0.100000*np.tanh(np.sin((((np.tanh((((data["signal_shift_-1"]) * (np.cos((data["rangebatch_msignal"]))))))) / 2.0)))) +

                        0.100000*np.tanh(np.where(data["signal_shift_-1"] <= -998, (((0.0)) / 2.0), ((data["abs_avgbatch_slices2"]) - ((((-1.0)) - ((((data["meanbatch_slices2_msignal"]) + (data["meanbatch_msignal"]))/2.0))))) )) +

                        0.100000*np.tanh(np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, np.tanh(((-(((0.0)))))), data["signal"] )) +

                        0.100000*np.tanh(((np.cos((((data["abs_minbatch_msignal"]) * (data["signal_shift_+1_msignal"]))))) * (((np.sin((np.tanh(((((((3.0)) * (data["medianbatch_msignal"]))) - (data["stdbatch_msignal"]))))))) / 2.0)))) +

                        0.100000*np.tanh((-((np.cos(((-((data["medianbatch_slices2"]))))))))) +

                        0.100000*np.tanh(((np.sin(((-(((-(((-((np.sin((((data["minbatch_slices2_msignal"]) / 2.0))))))))))))))) + ((((-2.0)) + (np.cos((data["abs_minbatch_slices2_msignal"]))))))) +

                        0.100000*np.tanh(((data["abs_maxbatch_slices2"]) / 2.0)) +

                        0.100000*np.tanh(np.where(data["medianbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_slices2"], ((data["rangebatch_slices2"]) - ((((data["minbatch"]) + ((-(((1.0))))))/2.0))) )) +

                        0.100000*np.tanh(data["signal_shift_-1"]) +

                        0.100000*np.tanh(np.tanh((np.sin((data["abs_maxbatch_slices2"]))))) +

                        0.100000*np.tanh((2.30253386497497559)) +

                        0.100000*np.tanh(np.where(np.cos((((np.sin((data["abs_maxbatch_slices2_msignal"]))) * ((((0.0)) + (((np.sin((data["rangebatch_msignal"]))) * 2.0))))))) <= -998, data["abs_avgbatch_slices2_msignal"], data["abs_maxbatch_slices2_msignal"] )) +

                        0.100000*np.tanh((-((np.tanh((((np.where(np.cos((np.cos((np.tanh((np.sin((data["maxtominbatch"])))))))) <= -998, np.tanh((data["abs_maxbatch_slices2"])), data["maxtominbatch"] )) * (np.tanh(((-((data["mean_abs_chgbatch_msignal"]))))))))))))) +

                        0.100000*np.tanh(((np.cos((np.sin((data["rangebatch_slices2"]))))) + (data["stdbatch_msignal"]))) +

                        0.100000*np.tanh(np.where((-3.0) <= -998, np.where(((((((6.0)) + ((-1.0)))/2.0)) - (np.sin(((((2.0)) / 2.0))))) <= -998, np.sin(((((-1.0)) / 2.0))), (((-((np.tanh(((2.0))))))) * 2.0) ), np.where(data["signal_shift_+1"] > -998, (((0.36186227202415466)) - (data["abs_avgbatch_slices2_msignal"])), (1.0) ) )) +

                        0.100000*np.tanh(data["minbatch_msignal"]) +

                        0.100000*np.tanh(((np.tanh(((((-((np.sin((np.cos((data["signal_shift_+1"])))))))) * 2.0)))) / 2.0)) +

                        0.100000*np.tanh((((7.0)) - (((((((data["signal"]) - (data["stdbatch_slices2"]))) - (data["maxbatch_slices2_msignal"]))) - (data["minbatch_msignal"]))))) +

                        0.100000*np.tanh((1.28159558773040771)) +

                        0.100000*np.tanh(np.tanh((np.cos((np.where(data["medianbatch_slices2"] > -998, ((data["abs_avgbatch_msignal"]) * 2.0), (-((data["maxtominbatch_slices2"]))) )))))) +

                        0.100000*np.tanh(np.sin(((((2.0)) / 2.0)))) +

                        0.100000*np.tanh(((data["rangebatch_slices2"]) * ((((-3.0)) * (((((np.where((0.0) > -998, data["rangebatch_slices2"], (-(((-3.0)))) )) * (data["medianbatch_msignal"]))) - (data["medianbatch_slices2"]))))))) +

                        0.100000*np.tanh(((np.where(data["rangebatch_msignal"] > -998, data["medianbatch_slices2"], (-(((((((data["abs_minbatch_slices2"]) + (np.cos((data["abs_avgbatch_msignal"]))))/2.0)) + (((np.tanh((data["meanbatch_slices2"]))) - (np.where(data["signal_shift_+1"] > -998, data["medianbatch_slices2"], ((data["rangebatch_slices2_msignal"]) / 2.0) )))))))) )) * 2.0)) +

                        0.100000*np.tanh(((data["meanbatch_msignal"]) - (data["maxtominbatch_msignal"]))) +

                        0.100000*np.tanh((((((((data["signal_shift_+1"]) + (np.cos((((data["abs_maxbatch"]) * 2.0)))))/2.0)) - ((3.0)))) + (np.where(data["signal_shift_-1"] > -998, data["signal_shift_+1"], (-2.0) )))) +

                        0.100000*np.tanh((((-2.0)) - (np.where(np.where((((2.0)) * (np.sin((data["meanbatch_msignal"])))) <= -998, data["rangebatch_slices2"], data["meanbatch_msignal"] ) <= -998, (-2.0), np.where(data["stdbatch_slices2_msignal"] <= -998, data["maxbatch_slices2"], np.where(data["rangebatch_slices2"] <= -998, data["rangebatch_slices2"], data["meanbatch_msignal"] ) ) )))) +

                        0.100000*np.tanh(((((((np.cos((np.sin((np.where(data["meanbatch_msignal"] <= -998, np.cos(((3.0))), np.tanh((np.sin((np.tanh((np.cos((data["rangebatch_slices2_msignal"])))))))) )))))) - (data["mean_abs_chgbatch_msignal"]))) - (data["abs_maxbatch_slices2_msignal"]))) * 2.0)) +

                        0.100000*np.tanh(((np.where(data["abs_maxbatch_msignal"] > -998, data["maxbatch_slices2_msignal"], (4.0) )) - (((np.sin((data["medianbatch_slices2_msignal"]))) / 2.0)))) +

                        0.100000*np.tanh(data["abs_maxbatch_msignal"]) +

                        0.100000*np.tanh((((((((((data["abs_maxbatch_msignal"]) / 2.0)) - (data["meanbatch_msignal"]))) + ((-((data["maxtominbatch_slices2"])))))/2.0)) - ((((data["meanbatch_slices2_msignal"]) + (np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["maxbatch_slices2_msignal"], ((data["abs_avgbatch_slices2_msignal"]) * 2.0) )))/2.0)))) +

                        0.100000*np.tanh(data["signal_shift_-1"]) +

                        0.100000*np.tanh((((((((-2.0)) * ((((-((data["meanbatch_slices2"])))) - (np.where((-((data["meanbatch_slices2_msignal"]))) <= -998, data["meanbatch_slices2_msignal"], data["minbatch_msignal"] )))))) * (((data["medianbatch_msignal"]) - ((-1.0)))))) - ((((((data["minbatch_slices2_msignal"]) + (data["meanbatch_slices2_msignal"]))/2.0)) - (np.sin((data["meanbatch_slices2_msignal"]))))))) +

                        0.100000*np.tanh(np.where(((data["abs_minbatch_msignal"]) * (np.tanh((((np.sin((data["minbatch"]))) * (np.cos((data["maxtominbatch_slices2_msignal"])))))))) > -998, (-2.0), data["abs_avgbatch_msignal"] )) +

                        0.100000*np.tanh(((np.tanh(((2.0)))) - (data["abs_avgbatch_msignal"]))) +

                        0.100000*np.tanh(((np.tanh((np.sin((np.where(np.tanh(((((data["maxtominbatch_slices2"]) + ((-((((np.sin(((3.0)))) + ((((3.0)) * 2.0))))))))/2.0))) > -998, data["rangebatch_slices2"], data["maxbatch_slices2_msignal"] )))))) - (data["medianbatch_msignal"]))) +

                        0.100000*np.tanh(((data["abs_avgbatch_slices2_msignal"]) - ((((((((data["maxbatch_slices2_msignal"]) + ((((11.60081100463867188)) * (np.where(((((data["meanbatch_slices2"]) / 2.0)) * 2.0) <= -998, np.where(data["minbatch_msignal"] > -998, data["rangebatch_msignal"], ((np.tanh(((0.0)))) * 2.0) ), ((data["meanbatch_slices2_msignal"]) * ((1.0))) )))))) * 2.0)) + (((data["meanbatch_slices2_msignal"]) - (data["medianbatch_slices2"]))))/2.0)))) +

                        0.100000*np.tanh((-((((np.cos((data["maxtominbatch_slices2"]))) / 2.0))))) +

                        0.100000*np.tanh((((-((np.where((((((((data["abs_maxbatch_slices2_msignal"]) * 2.0)) + ((-((data["stdbatch_msignal"])))))/2.0)) / 2.0) > -998, (2.0), np.where(data["stdbatch_slices2"] <= -998, ((((-1.0)) + (data["maxtominbatch_slices2_msignal"]))/2.0), data["mean_abs_chgbatch_slices2"] ) ))))) + ((-((data["abs_avgbatch_slices2_msignal"])))))) +

                        0.100000*np.tanh((-((((((np.where(np.tanh((data["rangebatch_msignal"])) > -998, data["meanbatch_msignal"], ((data["signal"]) * (np.tanh((data["signal_shift_-1"])))) )) * 2.0)) + ((3.0))))))) +

                        0.100000*np.tanh((((((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, data["signal"], data["mean_abs_chgbatch_slices2_msignal"] )) + (((data["mean_abs_chgbatch_slices2_msignal"]) * (data["medianbatch_slices2_msignal"]))))/2.0)) * (np.where(data["mean_abs_chgbatch_msignal"] > -998, data["mean_abs_chgbatch_slices2"], np.where(np.where(data["medianbatch_msignal"] <= -998, data["medianbatch_slices2_msignal"], data["signal"] ) > -998, (((data["signal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0), (((data["stdbatch_slices2"]) + (data["signal"]))/2.0) ) )))) +

                        0.100000*np.tanh(np.where(data["rangebatch_slices2"] > -998, data["maxbatch_slices2_msignal"], np.where(data["meanbatch_msignal"] <= -998, data["abs_minbatch_msignal"], (((-(((((data["meanbatch_msignal"]) + (data["maxbatch_slices2_msignal"]))/2.0))))) * 2.0) ) )) +

                        0.100000*np.tanh((((-(((((data["abs_maxbatch_slices2_msignal"]) + (np.where(((data["maxbatch_msignal"]) * (data["abs_maxbatch_slices2_msignal"])) > -998, data["meanbatch_slices2_msignal"], np.cos(((((-((((data["meanbatch_slices2_msignal"]) + (data["stdbatch_msignal"])))))) * 2.0))) )))/2.0))))) * 2.0)) +

                        0.100000*np.tanh(np.where(np.where(data["signal_shift_+1_msignal"] <= -998, ((np.cos((((data["abs_maxbatch_slices2_msignal"]) / 2.0)))) * 2.0), data["meanbatch_slices2"] ) <= -998, np.where((2.70422172546386719) <= -998, ((np.tanh((np.sin(((1.0)))))) / 2.0), ((data["signal_shift_-1"]) / 2.0) ), data["stdbatch_slices2"] )) +

                        0.100000*np.tanh((-((((((data["medianbatch_msignal"]) - (np.where(data["mean_abs_chgbatch_slices2"] > -998, (((-3.0)) + ((-((data["abs_avgbatch_slices2_msignal"]))))), data["abs_avgbatch_msignal"] )))) + (((((-(((-3.0))))) + (((((np.tanh((data["stdbatch_slices2"]))) - (data["medianbatch_slices2_msignal"]))) + (data["abs_avgbatch_slices2_msignal"]))))/2.0))))))) +

                        0.100000*np.tanh(np.where(data["medianbatch_msignal"] > -998, ((((np.where((-(((0.0)))) > -998, data["medianbatch_msignal"], ((np.tanh((data["abs_maxbatch"]))) / 2.0) )) * (data["medianbatch_msignal"]))) - (np.where(np.tanh((data["rangebatch_slices2"])) <= -998, data["stdbatch_slices2"], data["maxbatch_msignal"] ))), np.sin((((data["mean_abs_chgbatch_msignal"]) * ((-1.0))))) )) +

                        0.100000*np.tanh((-(((((data["abs_avgbatch_slices2_msignal"]) + ((((((((((data["signal_shift_+1_msignal"]) + (data["stdbatch_slices2"]))/2.0)) * 2.0)) - ((((((data["medianbatch_msignal"]) + (np.sin((((((data["medianbatch_msignal"]) * 2.0)) + (data["meanbatch_msignal"]))))))/2.0)) - (data["abs_maxbatch_msignal"]))))) / 2.0)))/2.0))))) +

                        0.100000*np.tanh((((((0.0)) / 2.0)) + ((((((((((-3.0)) + (data["meanbatch_slices2"]))/2.0)) + (data["rangebatch_slices2"]))/2.0)) * (((data["medianbatch_slices2"]) * (data["stdbatch_slices2"]))))))) +

                        0.100000*np.tanh((-((np.where((((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)) + ((((((1.0)) * 2.0)) + ((-((((data["mean_abs_chgbatch_slices2"]) * 2.0))))))))/2.0) > -998, data["abs_avgbatch_msignal"], ((np.where((2.0) > -998, ((((data["mean_abs_chgbatch_slices2"]) * (np.tanh((data["stdbatch_slices2"]))))) + (data["abs_minbatch_msignal"])), data["stdbatch_slices2"] )) + (data["signal"])) ))))) +

                        0.100000*np.tanh(np.where((-(((0.0)))) <= -998, np.where(((data["medianbatch_slices2_msignal"]) / 2.0) > -998, (((data["abs_avgbatch_slices2"]) + (np.where(data["medianbatch_msignal"] > -998, (2.0), (1.0) )))/2.0), data["stdbatch_slices2"] ), data["mean_abs_chgbatch_slices2"] )) +

                        0.100000*np.tanh((-((((data["abs_maxbatch_msignal"]) + (((data["meanbatch_slices2_msignal"]) - (np.cos((((data["meanbatch_slices2_msignal"]) - (((np.cos((((data["abs_maxbatch_msignal"]) + (data["meanbatch_slices2_msignal"]))))) * 2.0))))))))))))) +

                        0.100000*np.tanh((-((((np.where((2.0) <= -998, data["medianbatch_msignal"], data["stdbatch_slices2_msignal"] )) - (np.where(data["medianbatch_msignal"] > -998, (-3.0), np.where(data["stdbatch_msignal"] <= -998, data["medianbatch_msignal"], (-((data["stdbatch_slices2"]))) ) ))))))) +

                        0.100000*np.tanh(((np.tanh((((((-2.0)) + (((np.cos((data["maxbatch_slices2_msignal"]))) - (np.sin((np.sin((np.tanh(((((3.0)) - (np.sin((data["maxbatch_slices2_msignal"]))))))))))))))/2.0)))) - (((data["signal_shift_-1"]) * (((((data["abs_avgbatch_slices2_msignal"]) + (((data["abs_maxbatch_msignal"]) + (data["meanbatch_slices2_msignal"]))))) / 2.0)))))) +

                        0.100000*np.tanh((((-((np.sin((np.where(np.where(((data["abs_maxbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2_msignal"])) <= -998, (-1.0), ((((-3.0)) + (np.cos((data["rangebatch_slices2"]))))/2.0) ) <= -998, data["stdbatch_slices2"], (((data["medianbatch_msignal"]) + (data["abs_maxbatch_slices2_msignal"]))/2.0) ))))))) * 2.0)) +

                        0.100000*np.tanh((-(((((data["maxbatch_slices2_msignal"]) + ((((data["medianbatch_slices2_msignal"]) + (np.where((-((((data["maxbatch_msignal"]) * 2.0)))) > -998, ((data["abs_avgbatch_slices2_msignal"]) + (((np.tanh(((-2.0)))) * 2.0))), data["maxbatch_slices2"] )))/2.0)))/2.0))))) +

                        0.100000*np.tanh((((((-((data["meanbatch_msignal"])))) * 2.0)) - ((((((data["abs_maxbatch_slices2"]) + (data["abs_maxbatch_slices2"]))/2.0)) - (((np.where(data["meanbatch_slices2_msignal"] <= -998, (-((data["meanbatch_msignal"]))), data["abs_maxbatch_slices2"] )) / 2.0)))))) +

                        0.100000*np.tanh(np.where(np.where(data["abs_maxbatch"] <= -998, data["maxbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] ) <= -998, ((data["stdbatch_msignal"]) - ((2.0))), ((data["signal"]) + ((-(((1.0)))))) )) +

                        0.100000*np.tanh((((6.0)) - (((((np.where((10.0) > -998, data["abs_minbatch_msignal"], ((data["meanbatch_slices2"]) + (np.where(np.sin((data["medianbatch_slices2_msignal"])) > -998, data["abs_avgbatch_slices2_msignal"], data["abs_avgbatch_slices2_msignal"] ))) )) + (np.where((2.0) <= -998, (-1.0), ((data["meanbatch_slices2"]) + ((10.0))) )))) + (data["medianbatch_slices2_msignal"]))))) +

                        0.100000*np.tanh(((((((data["abs_maxbatch_slices2"]) + (((np.sin((data["meanbatch_msignal"]))) * ((((data["rangebatch_slices2"]) + (data["abs_maxbatch_slices2"]))/2.0)))))/2.0)) + (np.sin((data["meanbatch_slices2_msignal"]))))/2.0)) +

                        0.100000*np.tanh(((np.tanh((np.tanh((((data["maxbatch_slices2"]) - (((((data["abs_minbatch_slices2"]) + (data["abs_minbatch_slices2"]))) * (data["abs_maxbatch_slices2_msignal"]))))))))) / 2.0)) +

                        0.100000*np.tanh((-((((((data["medianbatch_slices2_msignal"]) + ((((1.0)) + (((np.where(data["medianbatch_slices2_msignal"] > -998, np.cos((data["medianbatch_msignal"])), (((((-((np.sin(((2.0))))))) * 2.0)) * 2.0) )) / 2.0)))))) * 2.0))))) +

                        0.100000*np.tanh(np.where(((data["abs_avgbatch_slices2"]) - ((((2.0)) * 2.0))) <= -998, data["minbatch_slices2"], ((data["rangebatch_slices2"]) * (((data["rangebatch_slices2"]) * (np.cos(((((data["rangebatch_msignal"]) + (np.where(data["rangebatch_msignal"] > -998, data["signal"], np.sin((((((data["maxbatch_slices2_msignal"]) + ((-((data["rangebatch_msignal"])))))) * 2.0))) )))/2.0))))))) )) +

                        0.100000*np.tanh(np.where(((data["abs_minbatch_msignal"]) * (data["abs_maxbatch_slices2_msignal"])) <= -998, ((data["abs_minbatch_msignal"]) * (((data["signal"]) + (data["medianbatch_slices2"])))), ((((data["rangebatch_slices2"]) * (((data["stdbatch_slices2_msignal"]) - (((((data["medianbatch_slices2_msignal"]) * (data["medianbatch_slices2"]))) / 2.0)))))) * 2.0) )) +

                        0.100000*np.tanh((-((np.where(((data["meanbatch_slices2_msignal"]) + (data["maxbatch_msignal"])) <= -998, (((((3.0)) - (((data["meanbatch_slices2_msignal"]) + (np.tanh((data["signal_shift_+1"]))))))) / 2.0), ((((data["meanbatch_slices2_msignal"]) + (data["maxbatch_msignal"]))) * 2.0) ))))) +

                        0.100000*np.tanh(((((((np.where(data["maxbatch_msignal"] > -998, (3.0), data["abs_minbatch_msignal"] )) + (data["meanbatch_msignal"]))) * (((data["minbatch"]) - (data["medianbatch_slices2"]))))) * 2.0)) +

                        0.100000*np.tanh((-((((((data["abs_avgbatch_msignal"]) + (np.cos(((-((((data["abs_maxbatch_slices2"]) * (data["maxbatch_slices2"])))))))))) - (np.tanh((np.where(data["medianbatch_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] ))))))))) +

                        0.100000*np.tanh(np.where(((np.sin((data["signal_shift_+1_msignal"]))) + (((((-1.0)) + (np.sin((np.where(((data["maxtominbatch_slices2"]) * 2.0) <= -998, ((data["minbatch_slices2_msignal"]) / 2.0), (((-(((((3.0)) * 2.0))))) / 2.0) )))))/2.0))) > -998, data["signal_shift_+1"], np.tanh(((-(((-1.0)))))) )) +

                        0.100000*np.tanh(((((np.sin((np.where(np.cos((data["abs_maxbatch_msignal"])) > -998, np.where(data["maxtominbatch_msignal"] <= -998, data["signal"], ((np.cos((data["abs_maxbatch_slices2_msignal"]))) * 2.0) ), ((((-((data["medianbatch_msignal"])))) + (np.sin((data["signal"]))))/2.0) )))) * 2.0)) * 2.0)) +

                        0.100000*np.tanh(np.where(data["abs_maxbatch_slices2"] <= -998, (13.08781337738037109), (-((((data["medianbatch_slices2"]) + (np.where(((data["maxbatch_slices2"]) + (data["signal"])) > -998, data["stdbatch_msignal"], ((((data["signal"]) + (data["signal"]))) + (((data["signal"]) + ((((np.tanh((np.cos((np.cos((((data["medianbatch_slices2_msignal"]) * 2.0)))))))) + (data["maxbatch_msignal"]))/2.0))))) )))))) )) +

                        0.100000*np.tanh(((((data["abs_maxbatch_slices2_msignal"]) + (data["medianbatch_msignal"]))) * ((-((data["abs_maxbatch_slices2_msignal"])))))) +

                        0.100000*np.tanh(((np.cos(((((data["minbatch_msignal"]) + (np.where(((((data["minbatch_slices2_msignal"]) * (data["medianbatch_msignal"]))) * 2.0) > -998, np.sin(((((((((data["minbatch_msignal"]) - (data["stdbatch_slices2"]))) - (data["signal_shift_-1_msignal"]))) + (data["minbatch_msignal"]))/2.0))), (2.0) )))/2.0)))) * 2.0)) +

                        0.100000*np.tanh(np.where(data["abs_avgbatch_slices2"] <= -998, (-((np.where(data["abs_maxbatch_msignal"] > -998, (((-3.0)) / 2.0), ((((data["minbatch_msignal"]) / 2.0)) - (data["abs_avgbatch_slices2"])) )))), np.where((((10.07003498077392578)) + (data["abs_maxbatch"])) <= -998, ((data["signal_shift_-1_msignal"]) * 2.0), ((np.sin((data["minbatch_msignal"]))) * 2.0) ) )) +

                        0.100000*np.tanh(((((data["medianbatch_msignal"]) - ((7.0)))) * (((np.tanh(((-(((-((np.cos((((np.where((3.0) > -998, data["mean_abs_chgbatch_slices2"], data["medianbatch_slices2"] )) * 2.0)))))))))))) + (((data["maxbatch_slices2_msignal"]) + (np.tanh((data["medianbatch_slices2_msignal"]))))))))) +

                        0.100000*np.tanh(((data["signal_shift_-1"]) - (np.where(np.where((-(((2.0)))) <= -998, (3.0), ((((np.sin((data["meanbatch_msignal"]))) * 2.0)) * (data["abs_maxbatch"])) ) > -998, ((data["maxbatch_slices2_msignal"]) * 2.0), np.where((1.0) <= -998, data["signal_shift_-1"], (8.0) ) )))) +

                        0.100000*np.tanh(((((np.sin((data["medianbatch_msignal"]))) / 2.0)) * (((np.where(np.where((-3.0) <= -998, data["rangebatch_msignal"], data["medianbatch_slices2"] ) <= -998, data["maxbatch_msignal"], data["abs_maxbatch_slices2"] )) + (np.sin(((((data["maxbatch_msignal"]) + (data["meanbatch_slices2"]))/2.0)))))))) +

                        0.100000*np.tanh(((data["mean_abs_chgbatch_slices2"]) + (((((data["abs_avgbatch_slices2"]) / 2.0)) - ((-((np.where(np.where(data["abs_maxbatch"] <= -998, ((data["signal_shift_+1_msignal"]) / 2.0), np.where(np.tanh(((1.0))) <= -998, (10.59714984893798828), data["abs_maxbatch"] ) ) > -998, data["signal_shift_+1_msignal"], np.sin((data["abs_maxbatch"])) ))))))))) +

                        0.100000*np.tanh((((((2.0)) - (np.cos((np.where(data["maxtominbatch"] > -998, data["minbatch_msignal"], ((data["abs_maxbatch_slices2_msignal"]) - (np.tanh(((3.0))))) )))))) - (((data["meanbatch_msignal"]) + ((((-1.0)) + ((((np.tanh((data["abs_minbatch_msignal"]))) + (((((data["abs_avgbatch_slices2"]) / 2.0)) / 2.0)))/2.0)))))))) +

                        0.100000*np.tanh(np.where(data["maxbatch_slices2"] <= -998, np.sin((data["meanbatch_msignal"])), ((np.where(np.sin((np.where(data["signal_shift_+1"] <= -998, data["meanbatch_msignal"], np.sin((np.sin((data["meanbatch_msignal"])))) ))) <= -998, data["meanbatch_slices2"], ((((np.sin((data["meanbatch_msignal"]))) * 2.0)) * 2.0) )) * 2.0) )) +

                        0.100000*np.tanh(((((data["abs_avgbatch_slices2"]) - (((((np.sin((((((data["maxbatch_slices2_msignal"]) / 2.0)) / 2.0)))) + (data["maxbatch_slices2_msignal"]))) + (((((data["abs_maxbatch_msignal"]) * 2.0)) / 2.0)))))) * 2.0)) +

                        0.100000*np.tanh(np.sin(((((((data["minbatch_slices2_msignal"]) + (np.sin((((data["meanbatch_slices2"]) + (np.where((((data["abs_maxbatch_slices2"]) + (data["signal_shift_+1"]))/2.0) > -998, data["abs_minbatch_slices2_msignal"], np.sin(((((data["minbatch_slices2_msignal"]) + (np.sin(((1.0)))))/2.0))) )))))))/2.0)) + (np.tanh((data["signal_shift_-1"]))))))) +

                        0.100000*np.tanh(((np.where(((np.sin((data["minbatch_msignal"]))) - (data["maxbatch_msignal"])) > -998, np.tanh((np.sin((data["meanbatch_msignal"])))), data["maxtominbatch"] )) * 2.0)) +

                        0.100000*np.tanh(np.where(((np.sin(((-3.0)))) * (((((np.where(data["minbatch_msignal"] <= -998, (-3.0), (-2.0) )) * 2.0)) / 2.0))) <= -998, data["signal_shift_+1_msignal"], (((((((-2.0)) - (data["medianbatch_slices2_msignal"]))) * (data["abs_maxbatch_msignal"]))) * (data["abs_maxbatch_msignal"])) )) +

                        0.100000*np.tanh((((data["signal_shift_-1_msignal"]) + ((-((np.where(np.cos((np.sin((((data["signal_shift_-1_msignal"]) * 2.0))))) > -998, data["medianbatch_msignal"], ((((((data["rangebatch_msignal"]) * 2.0)) + (((data["signal_shift_-1_msignal"]) * 2.0)))) * ((((2.0)) - ((((-((data["mean_abs_chgbatch_slices2"])))) * 2.0))))) ))))))/2.0)) +

                        0.100000*np.tanh(((np.cos(((((data["signal_shift_+1"]) + (data["mean_abs_chgbatch_msignal"]))/2.0)))) * 2.0)) +

                        0.100000*np.tanh((((data["maxbatch_slices2"]) + (data["mean_abs_chgbatch_slices2"]))/2.0)) +

                        0.100000*np.tanh(np.where(data["meanbatch_slices2"] > -998, data["signal_shift_+1_msignal"], ((data["meanbatch_msignal"]) * ((-((np.tanh((data["stdbatch_slices2_msignal"]))))))) )) +

                        0.100000*np.tanh(np.sin((data["minbatch"]))) +

                        0.100000*np.tanh(((data["maxtominbatch"]) - (data["stdbatch_slices2_msignal"]))) +

                        0.100000*np.tanh(np.where(np.where(data["signal_shift_-1_msignal"] > -998, np.where((((4.0)) * 2.0) > -998, (4.0), (-1.0) ), data["maxtominbatch_slices2"] ) > -998, data["signal_shift_-1_msignal"], np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["maxtominbatch"], (((((data["rangebatch_msignal"]) + (data["maxtominbatch_slices2"]))/2.0)) * 2.0) ) )) +

                        0.100000*np.tanh(((data["meanbatch_msignal"]) * (((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)))) +

                        0.100000*np.tanh(((((data["medianbatch_slices2"]) / 2.0)) * (((((np.where(data["maxbatch_slices2_msignal"] <= -998, data["minbatch"], ((np.where(data["medianbatch_slices2_msignal"] > -998, np.sin((data["medianbatch_msignal"])), data["medianbatch_slices2_msignal"] )) * 2.0) )) * 2.0)) + ((((((data["stdbatch_slices2"]) + (np.cos((data["rangebatch_slices2_msignal"]))))/2.0)) * 2.0)))))) +

                        0.100000*np.tanh(np.where(data["maxtominbatch_msignal"] <= -998, np.sin((np.sin((data["maxtominbatch_msignal"])))), data["maxtominbatch_msignal"] )) +

                        0.100000*np.tanh(np.sin(((((((0.0)) + (data["mean_abs_chgbatch_msignal"]))) / 2.0)))) +

                        0.100000*np.tanh(np.sin((np.sin(((((-3.0)) * (np.sin((np.cos((((data["minbatch_msignal"]) + ((-3.0)))))))))))))) +

                        0.100000*np.tanh(((np.sin((np.where(data["signal_shift_-1_msignal"] > -998, data["meanbatch_msignal"], np.where(data["abs_maxbatch_msignal"] > -998, np.tanh((data["abs_maxbatch"])), np.sin((data["signal_shift_+1_msignal"])) ) )))) * 2.0)) +

                        0.100000*np.tanh(np.where(np.where((1.0) > -998, data["abs_avgbatch_msignal"], np.where((-((data["signal_shift_-1_msignal"]))) <= -998, ((data["abs_minbatch_slices2"]) / 2.0), data["rangebatch_slices2_msignal"] ) ) <= -998, (3.0), (-((((((((data["abs_maxbatch_msignal"]) / 2.0)) - (np.cos(((((data["stdbatch_slices2"]) + (data["abs_minbatch_slices2"]))/2.0)))))) + (((data["abs_avgbatch_msignal"]) - (data["signal_shift_-1_msignal"]))))))) )) +

                        0.100000*np.tanh(data["maxtominbatch"]) +

                        0.100000*np.tanh((((data["signal_shift_-1"]) + (np.where(((data["minbatch_msignal"]) * (data["abs_avgbatch_slices2"])) <= -998, data["signal_shift_-1"], (((-((data["abs_avgbatch_slices2"])))) - (np.cos((((data["meanbatch_slices2_msignal"]) / 2.0))))) )))/2.0)) +

                        0.100000*np.tanh(np.where(np.tanh((np.sin((((np.sin((data["minbatch_msignal"]))) * 2.0))))) <= -998, np.where(((data["maxtominbatch_slices2_msignal"]) * 2.0) > -998, data["mean_abs_chgbatch_msignal"], ((data["rangebatch_slices2"]) + (np.tanh((np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) + (np.sin((data["mean_abs_chgbatch_slices2_msignal"])))))))))) ), np.sin((((np.sin((data["mean_abs_chgbatch_msignal"]))) * 2.0))) )) +

                        0.100000*np.tanh((-((data["medianbatch_msignal"])))) +

                        0.100000*np.tanh(np.where((-2.0) <= -998, np.tanh((((data["signal_shift_+1_msignal"]) - ((((np.tanh((data["meanbatch_slices2"]))) + (np.where(np.sin((data["maxbatch_slices2_msignal"])) > -998, data["minbatch"], data["maxtominbatch"] )))/2.0))))), data["signal_shift_-1_msignal"] )) +

                        0.100000*np.tanh(((data["signal_shift_+1_msignal"]) + (data["meanbatch_msignal"]))) +

                        0.100000*np.tanh(((((data["signal_shift_-1"]) + (np.where((-1.0) > -998, ((data["minbatch_slices2"]) + (data["maxtominbatch_slices2"])), np.where((-((((data["meanbatch_slices2"]) * 2.0)))) > -998, np.where(((data["signal_shift_-1"]) + (data["abs_avgbatch_msignal"])) > -998, (-3.0), data["mean_abs_chgbatch_msignal"] ), (-((data["abs_avgbatch_msignal"]))) ) )))) * 2.0)) +

                        0.100000*np.tanh(np.cos((((((np.where(data["signal_shift_+1_msignal"] > -998, data["medianbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] )) - (np.tanh(((((data["signal_shift_+1_msignal"]) + (np.tanh(((-2.0)))))/2.0)))))) * 2.0)))) +

                        0.100000*np.tanh(np.sin((data["abs_maxbatch_msignal"]))) +

                        0.100000*np.tanh(data["mean_abs_chgbatch_slices2"]) +

                        0.100000*np.tanh(np.where((2.72697401046752930) > -998, data["signal_shift_+1_msignal"], np.where(data["mean_abs_chgbatch_msignal"] > -998, data["signal_shift_+1_msignal"], data["signal_shift_-1_msignal"] ) )) +

                        0.100000*np.tanh(np.cos(((((np.sin(((((1.0)) * ((-((data["signal_shift_+1"])))))))) + (data["minbatch_slices2_msignal"]))/2.0)))) +

                        0.100000*np.tanh((((((data["signal_shift_-1_msignal"]) * 2.0)) + (np.where((((((np.cos((data["maxbatch_msignal"]))) * 2.0)) + (((data["signal_shift_-1_msignal"]) + (data["meanbatch_slices2"]))))/2.0) <= -998, np.sin((np.where(np.where(data["signal_shift_-1_msignal"] > -998, data["signal_shift_+1_msignal"], data["signal_shift_-1_msignal"] ) > -998, data["signal_shift_+1_msignal"], data["signal_shift_-1_msignal"] ))), data["signal_shift_-1_msignal"] )))/2.0)) +

                        0.100000*np.tanh(np.where((-(((((((((data["medianbatch_slices2"]) * (((data["stdbatch_msignal"]) - (data["stdbatch_msignal"]))))) / 2.0)) + ((0.0)))/2.0)))) <= -998, ((((((data["medianbatch_slices2"]) / 2.0)) * 2.0)) * (data["signal_shift_+1_msignal"])), (((data["minbatch_slices2"]) + ((((np.tanh((data["signal_shift_-1_msignal"]))) + (data["signal_shift_+1_msignal"]))/2.0)))/2.0) )) +

                        0.100000*np.tanh(np.where(np.cos((data["mean_abs_chgbatch_slices2"])) > -998, data["rangebatch_slices2_msignal"], ((np.tanh((data["signal_shift_+1_msignal"]))) / 2.0) )) +

                        0.100000*np.tanh(((((((data["signal_shift_-1"]) + (((data["meanbatch_msignal"]) + ((((((0.0)) * 2.0)) / 2.0)))))/2.0)) + (((((((0.0)) + (data["stdbatch_slices2"]))) + (np.where(data["signal_shift_+1_msignal"] <= -998, np.tanh(((-3.0))), ((((data["stdbatch_msignal"]) / 2.0)) + (data["signal_shift_+1_msignal"])) )))/2.0)))/2.0)) +

                        0.100000*np.tanh(((data["signal_shift_+1"]) + (((data["abs_maxbatch_msignal"]) * (np.where(data["medianbatch_msignal"] <= -998, data["abs_maxbatch_msignal"], ((np.where(np.tanh(((-((((data["mean_abs_chgbatch_msignal"]) * ((-((((data["abs_maxbatch_msignal"]) * (data["meanbatch_slices2"])))))))))))) > -998, np.where(data["mean_abs_chgbatch_msignal"] > -998, data["mean_abs_chgbatch_msignal"], np.cos((data["signal_shift_+1"])) ), data["medianbatch_slices2"] )) / 2.0) )))))) +

                        0.100000*np.tanh(((data["signal_shift_+1"]) + (np.where(((data["stdbatch_slices2_msignal"]) * 2.0) <= -998, data["abs_minbatch_msignal"], (-((np.where(np.cos((data["stdbatch_slices2"])) <= -998, np.where(data["medianbatch_slices2_msignal"] > -998, data["signal_shift_-1_msignal"], data["medianbatch_msignal"] ), ((data["maxbatch_msignal"]) + (data["meanbatch_slices2"])) )))) )))) +

                        0.100000*np.tanh(((data["stdbatch_slices2"]) * (((np.where(np.where(((((data["stdbatch_slices2"]) / 2.0)) * (data["mean_abs_chgbatch_msignal"])) <= -998, data["medianbatch_slices2"], data["minbatch_msignal"] ) > -998, np.sin((data["mean_abs_chgbatch_msignal"])), (2.0) )) * 2.0)))) +

                        0.100000*np.tanh(((data["signal_shift_+1_msignal"]) + (((((data["maxtominbatch"]) + (data["signal_shift_+1_msignal"]))) + (((((4.0)) + (data["rangebatch_slices2_msignal"]))/2.0)))))) +

                        0.100000*np.tanh(((data["signal_shift_-1"]) + (((data["signal_shift_-1_msignal"]) - (np.where(data["meanbatch_slices2"] > -998, np.where(data["signal"] > -998, (5.43797636032104492), data["abs_maxbatch_msignal"] ), data["signal_shift_-1"] )))))) +

                        0.100000*np.tanh((((data["signal_shift_+1"]) + (np.tanh((np.where(np.cos((data["minbatch"])) > -998, np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["medianbatch_msignal"], data["maxtominbatch_slices2_msignal"] ), np.where(((data["maxtominbatch"]) - (((np.where(data["stdbatch_msignal"] <= -998, data["medianbatch_msignal"], data["maxtominbatch"] )) + (data["minbatch"])))) > -998, data["minbatch"], (-(((((3.0)) * 2.0)))) ) )))))/2.0)) +

                        0.100000*np.tanh(np.where(data["abs_maxbatch_slices2_msignal"] <= -998, (-2.0), np.where(np.tanh(((1.0))) > -998, data["meanbatch_msignal"], data["meanbatch_msignal"] ) )) +

                        0.100000*np.tanh(np.sin((np.where((((-((np.where(((data["signal_shift_+1_msignal"]) * 2.0) <= -998, ((np.tanh((np.sin((np.where(((data["abs_avgbatch_slices2"]) / 2.0) > -998, data["mean_abs_chgbatch_msignal"], data["abs_avgbatch_slices2"] )))))) / 2.0), data["stdbatch_msignal"] ))))) / 2.0) > -998, ((data["stdbatch_slices2_msignal"]) * 2.0), np.sin((np.sin((data["maxbatch_slices2"])))) )))) +

                        0.100000*np.tanh(np.tanh((((((data["signal_shift_-1"]) - (((np.where(data["signal_shift_+1"] <= -998, data["abs_maxbatch_slices2_msignal"], data["stdbatch_msignal"] )) + (np.where(np.cos((data["signal_shift_-1_msignal"])) <= -998, data["abs_avgbatch_slices2"], np.sin((((((((3.0)) + (data["signal_shift_-1"]))) + (data["signal_shift_-1"]))/2.0))) )))))) * (np.cos((((data["maxbatch_slices2_msignal"]) - (data["minbatch_msignal"]))))))))) +

                        0.100000*np.tanh(np.sin((np.where(data["mean_abs_chgbatch_slices2"] <= -998, (((3.0)) + (data["signal_shift_+1"])), np.where((((-1.0)) * (np.sin(((1.0))))) <= -998, data["abs_minbatch_msignal"], ((np.cos((np.sin((np.sin(((3.0)))))))) * 2.0) ) )))) +

                        0.100000*np.tanh(np.where(data["maxtominbatch"] <= -998, data["maxtominbatch"], np.where(data["mean_abs_chgbatch_slices2"] > -998, (3.0), data["minbatch_slices2_msignal"] ) )) +

                        0.100000*np.tanh(np.where(data["signal_shift_-1_msignal"] <= -998, data["meanbatch_msignal"], ((((np.sin((data["meanbatch_msignal"]))) * 2.0)) / 2.0) )) +

                        0.100000*np.tanh(((np.sin((((data["stdbatch_slices2_msignal"]) * 2.0)))) + (((((data["signal_shift_-1_msignal"]) - (((data["signal_shift_-1_msignal"]) * (np.cos((((data["abs_minbatch_slices2_msignal"]) / 2.0)))))))) - (data["signal_shift_-1_msignal"]))))) +

                        0.100000*np.tanh(np.sin((np.where(((np.tanh((data["signal"]))) - ((-((np.sin((data["abs_minbatch_slices2"]))))))) <= -998, np.tanh((data["minbatch_msignal"])), ((data["minbatch_msignal"]) + (np.cos((((np.sin((np.where(np.sin((np.tanh((data["abs_maxbatch_msignal"])))) <= -998, ((data["mean_abs_chgbatch_msignal"]) * 2.0), data["maxtominbatch"] )))) / 2.0))))) )))) +

                        0.100000*np.tanh(((np.sin((data["minbatch_msignal"]))) * 2.0)) +

                        0.100000*np.tanh(((np.tanh((data["abs_avgbatch_slices2_msignal"]))) - (data["minbatch_slices2"]))) +

                        0.100000*np.tanh(np.sin((np.sin(((((((((data["meanbatch_msignal"]) * 2.0)) - (data["signal_shift_+1_msignal"]))) + (np.sin((((data["maxtominbatch_msignal"]) * 2.0)))))/2.0)))))) +

                        0.100000*np.tanh(np.where(data["signal_shift_-1_msignal"] <= -998, np.tanh((((((np.sin((data["stdbatch_slices2_msignal"]))) / 2.0)) + ((-((data["rangebatch_slices2"]))))))), ((data["signal_shift_-1_msignal"]) / 2.0) )) +

                        0.100000*np.tanh(np.cos((((np.where(((np.cos(((3.0)))) + (((data["signal"]) * (np.where((((np.cos((data["signal_shift_-1"]))) + (data["maxbatch_slices2_msignal"]))/2.0) > -998, data["maxtominbatch_slices2_msignal"], ((data["minbatch_msignal"]) / 2.0) ))))) > -998, data["minbatch_msignal"], data["meanbatch_slices2"] )) / 2.0)))) +

                        0.100000*np.tanh(((np.cos((data["minbatch_msignal"]))) * 2.0)) +

                        0.100000*np.tanh(np.where((-(((((data["abs_minbatch_slices2"]) + ((-3.0)))/2.0)))) > -998, np.sin((np.where(((((((1.0)) + (data["medianbatch_slices2"]))/2.0)) / 2.0) <= -998, (((1.0)) + (data["medianbatch_msignal"])), data["medianbatch_msignal"] ))), (-((np.tanh((data["maxtominbatch_msignal"]))))) )) +

                        0.100000*np.tanh(np.where(data["medianbatch_msignal"] > -998, np.where(np.sin((((data["signal_shift_+1"]) - (((np.where(data["maxtominbatch_slices2"] <= -998, ((((-3.0)) + (data["signal_shift_+1"]))/2.0), (-3.0) )) / 2.0))))) > -998, np.where(((data["abs_maxbatch_msignal"]) / 2.0) <= -998, data["abs_maxbatch_msignal"], (((-2.0)) - (data["medianbatch_msignal"])) ), (((7.0)) - (data["abs_maxbatch"])) ), (-1.0) )) +

                        0.100000*np.tanh(((data["maxtominbatch"]) + (((np.where(np.where(np.where(np.sin(((((((-((np.cos(((-3.0))))))) * 2.0)) * ((-((data["abs_maxbatch_slices2"]))))))) <= -998, (4.0), data["rangebatch_slices2_msignal"] ) <= -998, (4.0), data["meanbatch_slices2"] ) <= -998, data["maxtominbatch_slices2"], data["stdbatch_slices2"] )) * 2.0)))) +

                        0.100000*np.tanh(((np.where(((data["rangebatch_msignal"]) + ((9.0))) > -998, ((data["signal_shift_-1_msignal"]) * 2.0), np.where(data["minbatch_msignal"] > -998, ((data["signal_shift_-1"]) * 2.0), np.tanh((data["maxtominbatch_slices2_msignal"])) ) )) - (np.tanh((((data["abs_avgbatch_slices2_msignal"]) / 2.0)))))) +

                        0.100000*np.tanh(((data["abs_maxbatch_slices2"]) * (np.sin((np.where((1.0) <= -998, data["minbatch_msignal"], np.where(((data["abs_maxbatch_slices2"]) + (data["abs_minbatch_msignal"])) <= -998, data["mean_abs_chgbatch_slices2_msignal"], (((((-((data["mean_abs_chgbatch_msignal"])))) * ((((((((data["maxbatch_msignal"]) + (data["abs_maxbatch_slices2"]))/2.0)) / 2.0)) / 2.0)))) * 2.0) ) )))))) +

                        0.100000*np.tanh((((np.where(((data["maxtominbatch_slices2"]) / 2.0) > -998, data["signal_shift_+1_msignal"], np.sin((((np.sin((data["mean_abs_chgbatch_msignal"]))) * (((data["abs_maxbatch_slices2_msignal"]) * (data["signal_shift_+1_msignal"])))))) )) + (data["signal_shift_+1_msignal"]))/2.0)) +

                        0.100000*np.tanh((((np.where((-3.0) > -998, data["maxtominbatch_slices2"], (((((np.cos(((((0.0)) * 2.0)))) + (np.where(np.tanh((np.sin((data["abs_avgbatch_slices2"])))) <= -998, np.sin((((data["medianbatch_slices2"]) * 2.0))), data["abs_avgbatch_slices2_msignal"] )))) + (np.where(data["medianbatch_slices2"] <= -998, data["mean_abs_chgbatch_slices2_msignal"], np.cos((data["medianbatch_slices2"])) )))/2.0) )) + ((0.0)))/2.0)) +

                        0.100000*np.tanh(np.where((2.0) <= -998, np.cos((data["abs_maxbatch_slices2"])), np.sin((data["mean_abs_chgbatch_msignal"])) )) +

                        0.100000*np.tanh(((((np.tanh((data["mean_abs_chgbatch_slices2"]))) / 2.0)) + (np.cos((data["abs_minbatch_slices2_msignal"]))))) +

                        0.100000*np.tanh((-((np.cos((np.where(data["minbatch_msignal"] > -998, data["meanbatch_slices2_msignal"], (((data["signal_shift_-1_msignal"]) + (np.where(data["signal_shift_+1_msignal"] > -998, data["signal_shift_+1_msignal"], np.where(data["minbatch_slices2"] > -998, (((data["minbatch_msignal"]) + (data["signal_shift_+1_msignal"]))/2.0), data["maxtominbatch_slices2"] ) )))/2.0) ))))))) +

                        0.100000*np.tanh(((np.sin((np.where((-((((data["signal_shift_+1_msignal"]) + ((((data["abs_avgbatch_slices2_msignal"]) + ((4.20596218109130859)))/2.0)))))) > -998, np.tanh((data["abs_minbatch_slices2"])), np.tanh((data["signal_shift_+1_msignal"])) )))) / 2.0)) +

                        0.100000*np.tanh(np.where(((np.cos((((data["stdbatch_slices2"]) + ((((data["meanbatch_msignal"]) + (data["signal_shift_+1_msignal"]))/2.0)))))) * 2.0) > -998, ((((data["maxbatch_slices2_msignal"]) + (np.sin(((-((data["medianbatch_msignal"])))))))) * 2.0), data["meanbatch_slices2_msignal"] )) +

                        0.100000*np.tanh(np.tanh((np.where(np.where(data["medianbatch_slices2_msignal"] > -998, ((((((data["meanbatch_msignal"]) + (data["maxtominbatch_msignal"]))) / 2.0)) + (((data["meanbatch_msignal"]) + (np.sin((data["stdbatch_slices2_msignal"])))))), data["meanbatch_msignal"] ) > -998, np.sin((data["meanbatch_msignal"])), ((np.tanh((data["abs_avgbatch_slices2_msignal"]))) * (np.sin((data["signal_shift_+1_msignal"])))) )))) +

                        0.100000*np.tanh(((np.where(np.where(data["signal_shift_+1_msignal"] > -998, np.cos(((((0.0)) / 2.0))), np.cos((data["signal_shift_+1_msignal"])) ) > -998, data["signal_shift_-1"], np.where(data["maxtominbatch_msignal"] > -998, data["signal_shift_+1_msignal"], (((-(((9.84513950347900391))))) * 2.0) ) )) + ((-3.0)))) +

                        0.100000*np.tanh(np.where(np.tanh((data["rangebatch_msignal"])) > -998, data["signal_shift_+1_msignal"], ((np.where(data["maxbatch_slices2"] > -998, data["signal_shift_+1_msignal"], data["signal_shift_+1_msignal"] )) - (data["maxbatch_slices2"])) )) +

                        0.100000*np.tanh(np.sin((data["mean_abs_chgbatch_msignal"]))) +

                        0.100000*np.tanh((((((np.sin((np.where(data["minbatch_msignal"] <= -998, data["abs_minbatch_msignal"], data["minbatch_msignal"] )))) + (np.cos(((2.0)))))/2.0)) * 2.0)) +

                        0.100000*np.tanh(((np.sin((np.where(data["stdbatch_slices2"] <= -998, data["stdbatch_slices2_msignal"], np.where(((((-((((np.sin((data["meanbatch_msignal"]))) + (np.cos((data["meanbatch_msignal"])))))))) + (data["abs_avgbatch_slices2_msignal"]))/2.0) <= -998, (1.0), data["abs_avgbatch_slices2_msignal"] ) )))) * 2.0)) +

                        0.100000*np.tanh(((np.where((((0.0)) + ((-1.0))) <= -998, (1.0), np.cos((((data["abs_avgbatch_slices2_msignal"]) / 2.0))) )) / 2.0)) +

                        0.100000*np.tanh(np.sin((np.where(np.where(np.sin((np.where(data["minbatch"] > -998, data["minbatch_msignal"], (-1.0) ))) > -998, data["minbatch_msignal"], (-3.0) ) > -998, data["minbatch_msignal"], ((((np.where(data["minbatch_msignal"] > -998, data["minbatch_msignal"], ((data["abs_maxbatch_msignal"]) / 2.0) )) / 2.0)) + (data["maxbatch_slices2"])) )))) +

                        0.100000*np.tanh(np.where(((np.tanh((((np.sin((data["maxtominbatch"]))) / 2.0)))) / 2.0) <= -998, (((((np.cos((np.sin((data["signal_shift_+1"]))))) * 2.0)) + ((-((np.cos((data["mean_abs_chgbatch_msignal"])))))))/2.0), (((data["signal_shift_+1"]) + (data["maxtominbatch"]))/2.0) )) +

                        0.100000*np.tanh(np.sin((data["abs_avgbatch_slices2_msignal"]))) +

                        0.100000*np.tanh(np.where(data["maxtominbatch_slices2"] <= -998, np.where(data["rangebatch_slices2_msignal"] > -998, np.cos((data["abs_maxbatch_slices2_msignal"])), (1.0) ), ((np.where(data["maxtominbatch_slices2"] > -998, data["abs_maxbatch_slices2"], data["stdbatch_msignal"] )) - (data["signal_shift_+1"])) )) +

                        0.100000*np.tanh(np.where(np.sin(((((3.0)) * (data["signal_shift_+1_msignal"])))) > -998, data["signal_shift_+1_msignal"], data["maxtominbatch"] )) +

                        0.100000*np.tanh(np.where((((np.where((((data["abs_maxbatch_slices2"]) + (data["maxtominbatch_slices2_msignal"]))/2.0) > -998, data["signal_shift_+1_msignal"], (-2.0) )) + (data["maxbatch_slices2_msignal"]))/2.0) <= -998, ((data["abs_maxbatch"]) * ((-((((data["mean_abs_chgbatch_slices2"]) * (data["abs_maxbatch_msignal"]))))))), np.tanh((data["maxbatch_msignal"])) )) +

                        0.100000*np.tanh(np.where(data["signal_shift_-1_msignal"] > -998, data["maxtominbatch"], (-3.0) )) +

                        0.100000*np.tanh(((np.cos((data["signal_shift_-1"]))) + (data["stdbatch_slices2"]))) +

                        0.100000*np.tanh(np.tanh((np.cos((((np.cos((data["stdbatch_slices2_msignal"]))) * (np.sin((np.tanh((np.sin((data["meanbatch_msignal"]))))))))))))) +

                        0.100000*np.tanh(((np.sin((data["meanbatch_msignal"]))) + (np.sin((((np.where(np.where(((data["minbatch"]) - (data["signal_shift_+1_msignal"])) <= -998, data["medianbatch_slices2"], np.sin((((((data["minbatch_msignal"]) / 2.0)) * 2.0))) ) > -998, ((data["maxbatch_msignal"]) + ((6.0))), data["signal_shift_-1_msignal"] )) * 2.0)))))) +

                        0.100000*np.tanh(np.sin((data["medianbatch_msignal"]))) +

                        0.100000*np.tanh(np.where(data["signal_shift_-1_msignal"] > -998, ((np.where(data["signal_shift_-1_msignal"] > -998, data["signal_shift_-1_msignal"], np.where(data["meanbatch_msignal"] <= -998, ((data["minbatch_slices2_msignal"]) + (((data["signal_shift_+1"]) / 2.0))), data["minbatch_slices2_msignal"] ) )) - (data["abs_avgbatch_slices2_msignal"])), ((data["signal_shift_-1_msignal"]) * 2.0) )) +

                        0.100000*np.tanh(np.where(np.cos((np.where((-1.0) <= -998, np.sin((data["meanbatch_msignal"])), data["maxtominbatch_slices2_msignal"] ))) > -998, ((((data["minbatch_slices2"]) * ((2.0)))) - (data["maxbatch_slices2_msignal"])), data["rangebatch_slices2_msignal"] )) +

                        0.100000*np.tanh(data["signal_shift_-1_msignal"]) +

                        0.099609*np.tanh(np.where(((data["stdbatch_msignal"]) * 2.0) > -998, (-((np.where(data["maxtominbatch"] <= -998, np.where(((((data["meanbatch_slices2"]) / 2.0)) / 2.0) > -998, ((data["meanbatch_slices2"]) / 2.0), (((((0.0)) - (data["minbatch_msignal"]))) + (data["maxtominbatch_slices2"])) ), data["meanbatch_slices2_msignal"] )))), data["maxbatch_slices2"] )) +

                        0.100000*np.tanh(np.where(((data["mean_abs_chgbatch_msignal"]) + (np.tanh((data["medianbatch_msignal"])))) <= -998, np.cos((np.tanh((((np.tanh(((2.0)))) + (data["abs_avgbatch_slices2_msignal"])))))), (-((((data["signal_shift_+1_msignal"]) * (np.where(data["signal_shift_+1_msignal"] > -998, data["medianbatch_msignal"], ((data["minbatch_msignal"]) + (data["meanbatch_slices2_msignal"])) )))))) )) +

                        0.100000*np.tanh(data["rangebatch_msignal"]) +

                        0.100000*np.tanh(np.cos(((((data["minbatch_msignal"]) + (np.where(np.cos(((-((data["maxtominbatch_slices2"]))))) <= -998, (6.0), data["minbatch_msignal"] )))/2.0)))) +

                        0.100000*np.tanh(((np.where((-2.0) > -998, np.cos(((((data["minbatch_msignal"]) + (np.cos((data["stdbatch_slices2_msignal"]))))/2.0))), np.sin((data["minbatch_msignal"])) )) * 2.0)) +

                        0.100000*np.tanh(np.sin(((((-((np.where(((np.sin((data["maxtominbatch_slices2"]))) / 2.0) <= -998, data["signal_shift_+1"], np.where(np.cos((((np.tanh((((np.cos(((7.0)))) * 2.0)))) / 2.0))) <= -998, np.cos((data["maxtominbatch_slices2"])), np.cos((data["minbatch"])) ) ))))) * 2.0)))) +

                        0.100000*np.tanh(np.sin((((data["abs_maxbatch_slices2"]) + (np.where(np.cos((np.where(((data["minbatch_msignal"]) * 2.0) <= -998, (-((data["minbatch_msignal"]))), ((np.where(np.cos((np.where(data["medianbatch_slices2"] <= -998, data["stdbatch_slices2"], data["minbatch_slices2_msignal"] ))) > -998, (-((data["rangebatch_msignal"]))), data["abs_maxbatch_slices2"] )) * 2.0) ))) > -998, (-((data["rangebatch_msignal"]))), data["abs_maxbatch_slices2"] )))))) +

                        0.100000*np.tanh(np.where((((((0.0)) * 2.0)) / 2.0) > -998, np.sin((np.where(data["signal_shift_+1_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], np.tanh((np.sin((((data["medianbatch_msignal"]) * ((((data["abs_minbatch_slices2_msignal"]) + (((((np.sin((np.tanh((((data["maxbatch_slices2_msignal"]) / 2.0)))))) / 2.0)) * 2.0)))/2.0))))))) ))), (3.0) )) +

                        0.100000*np.tanh(((data["signal_shift_-1_msignal"]) * 2.0)) +

                        0.100000*np.tanh(np.where(data["medianbatch_slices2"] > -998, (3.95284748077392578), ((np.sin((np.where(data["medianbatch_slices2"] > -998, (7.0), data["minbatch_msignal"] )))) - (((data["signal"]) / 2.0))) )) +

                        0.100000*np.tanh(np.sin((((np.where((((data["stdbatch_slices2"]) + (np.cos((data["maxtominbatch_slices2"]))))/2.0) > -998, ((np.where(np.sin((data["signal_shift_+1_msignal"])) > -998, data["maxbatch_slices2_msignal"], np.where(((data["meanbatch_msignal"]) / 2.0) > -998, ((((data["signal_shift_+1_msignal"]) + (np.sin((((data["meanbatch_slices2"]) / 2.0)))))) * 2.0), (((-3.0)) / 2.0) ) )) * 2.0), data["abs_minbatch_msignal"] )) * 2.0)))) +

                        0.089932*np.tanh(((np.where(data["abs_maxbatch_slices2"] <= -998, np.sin(((((-2.0)) * 2.0))), ((np.tanh((np.cos((data["abs_maxbatch"]))))) * 2.0) )) * (np.tanh((((data["stdbatch_slices2_msignal"]) + (data["meanbatch_slices2"]))))))) +

                        0.100000*np.tanh(np.where(data["mean_abs_chgbatch_msignal"] > -998, ((data["maxtominbatch"]) + (data["abs_avgbatch_slices2"])), data["medianbatch_msignal"] )) +

                        0.098436*np.tanh(((((data["signal_shift_-1_msignal"]) / 2.0)) + (np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["abs_maxbatch_slices2_msignal"], (((data["medianbatch_msignal"]) + (((data["maxbatch_slices2"]) - (((np.where(data["minbatch_msignal"] > -998, data["abs_maxbatch_msignal"], ((data["meanbatch_slices2"]) + (((data["abs_avgbatch_slices2_msignal"]) / 2.0))) )) * 2.0)))))/2.0) )))) +

                        0.100000*np.tanh(np.where(data["signal_shift_+1_msignal"] > -998, ((((data["maxbatch_slices2_msignal"]) / 2.0)) * (((np.tanh((data["signal_shift_+1_msignal"]))) / 2.0))), data["maxtominbatch_msignal"] )) +

                        0.100000*np.tanh(((((np.cos((np.cos((((np.cos((np.where(data["medianbatch_slices2"] <= -998, data["abs_maxbatch"], np.tanh((data["abs_avgbatch_slices2"])) )))) / 2.0)))))) * 2.0)) / 2.0)) +

                        0.100000*np.tanh(np.cos(((((np.tanh((data["minbatch_msignal"]))) + (data["minbatch_msignal"]))/2.0)))) +

                        0.100000*np.tanh(np.sin((np.sin((np.where(np.sin(((-2.0))) <= -998, data["rangebatch_msignal"], np.sin((data["minbatch_msignal"])) )))))) +

                        0.100000*np.tanh(np.where(np.sin((np.sin((data["maxtominbatch_slices2"])))) > -998, np.cos((data["signal_shift_+1"])), data["maxbatch_msignal"] )) +

                        0.100000*np.tanh(np.cos((((data["abs_maxbatch_slices2"]) / 2.0)))) +

                        0.100000*np.tanh(((np.where((((-((((data["minbatch"]) * 2.0))))) / 2.0) > -998, data["minbatch"], (((0.0)) * 2.0) )) / 2.0)) +

                        0.093744*np.tanh(np.where(data["medianbatch_slices2"] <= -998, (1.0), (2.0) )) +

                        0.100000*np.tanh(np.cos(((-((np.where(data["signal_shift_-1_msignal"] <= -998, data["signal_shift_-1_msignal"], data["signal"] ))))))) +

                        0.099707*np.tanh(((np.cos((np.where(((data["minbatch_msignal"]) * (np.cos((np.cos((np.cos(((11.96339130401611328))))))))) > -998, np.sin((data["abs_maxbatch_msignal"])), np.cos(((((-((data["maxtominbatch_msignal"])))) * 2.0))) )))) - (np.tanh((((data["maxbatch_slices2_msignal"]) / 2.0)))))) +

                        0.100000*np.tanh(np.where(((((((np.cos((((((((np.cos((((data["abs_maxbatch_slices2"]) - ((((0.0)) * 2.0)))))) / 2.0)) / 2.0)) / 2.0)))) / 2.0)) / 2.0)) / 2.0) > -998, np.cos((data["abs_maxbatch"])), ((data["maxtominbatch_slices2"]) * 2.0) )) +

                        0.100000*np.tanh(np.cos((np.sin((data["rangebatch_msignal"]))))) +

                        0.100000*np.tanh(np.sin((np.where((0.0) <= -998, (((((-((data["abs_avgbatch_slices2"])))) * 2.0)) * 2.0), ((((data["medianbatch_msignal"]) * 2.0)) * 2.0) ))))) 
def maddest(d, axis=None):

    return np.mean(np.absolute(d - np.mean(d, axis)), axis)



def high_pass_filter(x, low_cutoff=1000, sample_rate=10000):



    nyquist = 0.5 * sample_rate

    norm_low_cutoff = low_cutoff / nyquist

    print(norm_low_cutoff)

    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')

    filtered_sig = signal.sosfilt(sos, x)



    return filtered_sig



def denoise_signal( x, wavelet='db4', level=1):

    

    coeff = pywt.wavedec( x, wavelet, mode="per" )

    sigma = (1/0.6745) * maddest( coeff[-level] )

    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )

    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )

    return pywt.waverec( coeff, wavelet, mode='per' )
def add_rooling_data(df : pd.DataFrame) -> pd.DataFrame:

    window_sizes = [10, 50, 100, 1000]

    for window in window_sizes:

        df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()

        df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()

    return df
base = '../input/liverpool-ion-switching'

train = pd.read_csv(os.path.join(base + '/train.csv'))

test  = pd.read_csv(os.path.join(base + '/test.csv'))
def features(df):

    df = df.sort_values(by=['time']).reset_index(drop=True)

    

    df.index = ((df.time * 10_000) - 1).values

    df['batch'] = df.index // 25_000

    df['batch_index'] = df.index  - (df.batch * 25_000)

    df['batch_slices'] = df['batch_index']  // 2500

    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)

    

    for c in ['batch','batch_slices2']:

        d = {}

        d['mean'+c] = df.groupby([c])['signal'].mean()

        d['median'+c] = df.groupby([c])['signal'].median()

        d['max'+c] = df.groupby([c])['signal'].max()

        d['min'+c] = df.groupby([c])['signal'].min()

        d['std'+c] = df.groupby([c])['signal'].std()

        d['mean_abs_chg'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))

        d['abs_max'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))

        d['abs_min'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))

        d['range'+c] = d['max'+c] - d['min'+c]

        d['maxtomin'+c] = d['max'+c] / d['min'+c]

        d['abs_avg'+c] = (d['abs_min'+c] + d['abs_max'+c]) / 2

        for v in d:

            df[v] = df[c].map(d[v].to_dict())



    

    #add shifts

    df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])

    df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+1'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-1'][i] = np.nan



    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels', 'batch',

                                                    'batch_index', 'batch_slices', 'batch_slices2'

                                                    

                                                   ]]:

        df[c+'_msignal'] = df[c] - df['signal']

        

    return df



train = features(train)

test = features(test)



col = [c for c in train.columns if c not in ['time', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2',

                                             'mean_abs_chgbatch', 'meanbatch', 'rangebatch', 'stdbatch',

                                             'maxbatch', 'medianbatch', 'abs_minbatch', 'abs_avgbatch']]

target = train['open_channels']

train = train[col]
train.replace(np.inf,np.nan,inplace=True)

train.replace(-np.inf,np.nan,inplace=True)

train.fillna(-999,inplace=True)

test.replace(np.inf,np.nan,inplace=True)

test.replace(-np.inf,np.nan,inplace=True)

test.fillna(-999,inplace=True)
gp = GP()

train_preds = gp.GrabPredictions(train)

test_preds = gp.GrabPredictions(test)
f1_score(target.values,np.argmax(train_preds.values,axis=1),average='macro')

test['open_channels'] = np.argmax(test_preds.values,axis=1)

test[['time','open_channels']].to_csv('gpsubmission.csv', index=False, float_format='%.4f')
from functools import partial

import scipy as sp

class OptimizedRounder(object):



    def __init__(self):

        self.coef_ = 0



    def loss(self, coef, X, y):

        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        return -metrics.f1_score(y, X_p, average = 'macro')



    def fit(self, X, y):

        loss_partial = partial(self.loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        return (pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).astype(np.int8)



    def coefficients(self):

        return self.coef_['x']





def optimize_predictions(preds, coeffs):

    

    preds[preds <= coeffs[0]] = 0

    preds[np.where(np.logical_and(preds > coeffs[0], preds <= coeffs[1]))] = 1

    preds[np.where(np.logical_and(preds > coeffs[1], preds <= coeffs[2]))] = 2

    preds[np.where(np.logical_and(preds > coeffs[2], preds <= coeffs[3]))] = 3

    preds[np.where(np.logical_and(preds > coeffs[3], preds <= coeffs[4]))] = 4

    preds[np.where(np.logical_and(preds > coeffs[4], preds <= coeffs[5]))] = 5

    preds[np.where(np.logical_and(preds > coeffs[5], preds <= coeffs[6]))] = 6

    preds[np.where(np.logical_and(preds > coeffs[6], preds <= coeffs[7]))] = 7

    preds[np.where(np.logical_and(preds > coeffs[7], preds <= coeffs[8]))] = 8

    preds[np.where(np.logical_and(preds > coeffs[8], preds <= coeffs[9]))] = 9

    preds[preds > coeffs[9]] = 10

    preds = preds.astype(np.int8)

    return preds



def MacroF1Metric(preds, dtrain):

    labels = dtrain.get_label()

    preds = np.round(np.clip(preds, 0, 10)).astype(int)

    score = metrics.f1_score(labels, preds, average = 'macro')

    return ('MacroF1Metric', score, True)
idx = np.zeros(train.shape[0]).astype(bool)

idx[::5] = True

idx[2::5] = True

train.drop(train.index[idx],inplace=True,axis=0)

gc.collect()

idx = np.logical_not(idx)

train_preds = train_preds[idx]

target = target[idx]

for c in range(11):

    train['gp_'+str(int(c))] = train_preds.values[:,c]

    test['gp_'+str(int(c))] = test_preds.values[:,c]

x1, x2, y1, y2 = model_selection.train_test_split(train, target, test_size=0.2, random_state=7)

del train

gc.collect()

col = x1.columns

import lightgbm as lgb

import lightgbm as lgb

params = {'learning_rate': 0.1, 'max_depth': -1, 'num_leaves':2**7+1, 'metric': 'rmse', 'random_state': 7, 'n_jobs':-1, 'sample_fraction':0.33} 

model = lgb.train(params, lgb.Dataset(x1, y1), 22222,  lgb.Dataset(x2, y2), verbose_eval=0, early_stopping_rounds=250, feval=MacroF1Metric)

preds_lgb = (model.predict(test[col], num_iteration=model.best_iteration)).astype(np.float16)

oof_lgb = (model.predict(x2, num_iteration=model.best_iteration)).astype(np.float16)



optR = OptimizedRounder()

optR.fit(oof_lgb,y2)

coeffs = optR.coefficients()

preds_lgb = optimize_predictions(preds_lgb,coeffs)

oof_lgb = optimize_predictions(oof_lgb,coeffs)
print('f1_score',f1_score(y2,oof_lgb,average='macro'))

test['open_channels'] = preds_lgb

test[['time','open_channels']].to_csv('submission.csv', index=False, float_format='%.4f')