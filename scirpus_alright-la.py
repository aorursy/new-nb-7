import os

import gc

import numpy as np

import pandas as pd

from sklearn.metrics import f1_score

from sklearn import model_selection
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

                            0.050000*np.tanh(((np.minimum((((((np.where(data["abs_maxbatch"] > -998, ((data["minbatch_slices2_msignal"]) - (np.minimum(((data["abs_maxbatch"])), ((np.where(np.maximum(((data["abs_maxbatch"])), ((data["abs_maxbatch_msignal"]))) > -998, data["maxtominbatch_slices2_msignal"], data["minbatch_slices2_msignal"] )))))), data["maxtominbatch_slices2_msignal"] )) > ((((data["minbatch_slices2_msignal"]) <= (np.where(data["abs_maxbatch"] > -998, data["maxtominbatch_slices2"], data["stdbatch_msignal"] )))*1.)))*1.))), ((data["minbatch_slices2_msignal"])))) + (data["maxtominbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(np.minimum(((data["maxtominbatch_slices2_msignal"])), ((((np.where(((data["signal_shift_-1"]) * 2.0) > -998, ((data["maxtominbatch_slices2_msignal"]) + (data["minbatch_msignal"])), data["minbatch_msignal"] )) * 2.0))))) +

                            0.050000*np.tanh(np.where(np.where(data["minbatch_msignal"] <= -998, np.cos((data["meanbatch_msignal"])), np.cos((data["abs_minbatch_msignal"])) ) > -998, ((((data["minbatch_msignal"]) - (data["signal_shift_-1"]))) * 2.0), data["signal"] )) +

                            0.050000*np.tanh(((np.where(np.maximum(((data["maxtominbatch_msignal"])), ((np.minimum(((data["meanbatch_slices2"])), ((np.minimum(((((data["minbatch_slices2_msignal"]) / 2.0))), ((np.minimum(((data["abs_maxbatch_slices2"])), ((np.sin((((data["maxtominbatch_msignal"]) + (data["maxtominbatch_msignal"])))))))))))))))) > -998, data["minbatch_slices2_msignal"], ((data["signal_shift_+1_msignal"]) - (data["mean_abs_chgbatch_msignal"])) )) + (data["maxtominbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((data["abs_minbatch_msignal"]) + (np.where(np.where(data["meanbatch_msignal"] <= -998, data["maxtominbatch_msignal"], ((((data["mean_abs_chgbatch_msignal"]) + (data["minbatch_msignal"]))) - (np.minimum(((((data["maxtominbatch_msignal"]) * 2.0))), ((data["maxtominbatch_msignal"]))))) ) > -998, data["minbatch_msignal"], data["abs_minbatch_msignal"] )))) +

                            0.050000*np.tanh(((data["abs_avgbatch_msignal"]) - (np.where(((((data["abs_avgbatch_msignal"]) - (np.where(np.minimum(((data["maxtominbatch_slices2"])), (((6.20043182373046875)))) > -998, (((6.20043182373046875)) + (data["signal"])), data["signal"] )))) * ((((data["abs_minbatch_slices2_msignal"]) <= ((6.20043182373046875)))*1.))) > -998, (((6.20043182373046875)) + (data["signal"])), (((6.20042848587036133)) + (data["signal"])) )))) +

                            0.050000*np.tanh(np.where(data["minbatch_msignal"] <= -998, np.where((7.01191520690917969) > -998, data["maxtominbatch_slices2_msignal"], data["maxtominbatch_slices2_msignal"] ), ((data["abs_avgbatch_slices2"]) * (((np.maximum(((data["minbatch_msignal"])), ((data["maxtominbatch_msignal"])))) + (np.where(((data["maxtominbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2"])) > -998, data["minbatch_msignal"], np.where(data["maxtominbatch_msignal"] > -998, data["rangebatch_msignal"], data["maxtominbatch_slices2_msignal"] ) ))))) )) +

                            0.050000*np.tanh(np.where(((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0) > -998, ((data["maxtominbatch_msignal"]) + (data["minbatch_slices2_msignal"])), (((np.where(np.where(((data["abs_maxbatch_msignal"]) + (((data["maxtominbatch_slices2_msignal"]) + (((data["minbatch_slices2_msignal"]) * 2.0))))) > -998, ((data["maxtominbatch_msignal"]) + (data["minbatch_slices2_msignal"])), ((data["abs_minbatch_slices2_msignal"]) + (data["minbatch_slices2_msignal"])) ) > -998, data["mean_abs_chgbatch_slices2_msignal"], data["minbatch_slices2_msignal"] )) <= (data["minbatch_slices2_msignal"]))*1.) )) +

                            0.050000*np.tanh(((((data["minbatch_slices2_msignal"]) - (np.tanh((((((data["abs_maxbatch"]) - (data["abs_minbatch_msignal"]))) / 2.0)))))) - (((np.where(data["maxtominbatch_msignal"] <= -998, ((data["abs_maxbatch"]) - (((((data["abs_maxbatch"]) - (data["abs_minbatch_msignal"]))) * 2.0))), ((data["abs_maxbatch"]) - (((data["abs_minbatch_msignal"]) * 2.0))) )) * 2.0)))) +

                            0.050000*np.tanh(((((data["minbatch_slices2_msignal"]) + ((((((data["minbatch_slices2_msignal"]) + ((((((((data["maxbatch_msignal"]) * (((data["maxtominbatch_msignal"]) + (np.cos((data["rangebatch_slices2"]))))))) + (data["minbatch_slices2_msignal"]))) + (data["maxtominbatch_msignal"]))/2.0)))) + (((((data["maxbatch_msignal"]) * (data["maxtominbatch_slices2_msignal"]))) + (data["minbatch_slices2_msignal"]))))/2.0)))) * 2.0)) +

                            0.050000*np.tanh(((np.where(np.minimum(((np.minimum(((data["medianbatch_slices2"])), ((data["maxtominbatch_slices2_msignal"]))))), ((data["maxbatch_slices2"]))) <= -998, (((data["maxtominbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"]))/2.0), data["minbatch_slices2_msignal"] )) + (data["maxtominbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((((np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["minbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] )) + (data["maxtominbatch_slices2_msignal"]))) + (np.where(np.sin((data["minbatch_slices2"])) > -998, data["minbatch_slices2_msignal"], data["maxtominbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(np.where(np.where(data["medianbatch_slices2_msignal"] > -998, ((data["minbatch_msignal"]) + (data["maxtominbatch_msignal"])), ((data["maxtominbatch_msignal"]) - (data["signal_shift_+1"])) ) <= -998, ((data["maxbatch_slices2_msignal"]) + ((((-((data["rangebatch_slices2"])))) + (np.where(data["minbatch"] > -998, data["maxtominbatch_slices2_msignal"], data["minbatch_msignal"] ))))), ((((data["minbatch_slices2_msignal"]) - (data["signal_shift_+1"]))) * 2.0) )) +

                            0.050000*np.tanh(((np.where(((((data["abs_maxbatch"]) * (np.where(data["abs_maxbatch"] > -998, ((data["abs_avgbatch_slices2_msignal"]) - (data["abs_maxbatch"])), data["abs_avgbatch_slices2_msignal"] )))) * 2.0) > -998, ((data["abs_avgbatch_slices2_msignal"]) - (data["abs_maxbatch"])), np.tanh((((data["abs_maxbatch"]) * 2.0))) )) * 2.0)) +

                            0.050000*np.tanh(((data["maxtominbatch_slices2_msignal"]) + (np.where(((((data["stdbatch_slices2"]) - (data["stdbatch_slices2_msignal"]))) + (((data["minbatch_msignal"]) * 2.0))) > -998, data["minbatch_msignal"], data["minbatch_msignal"] )))) +

                            0.050000*np.tanh((((((((data["maxtominbatch_msignal"]) + (np.minimum(((((data["minbatch_msignal"]) + (((data["stdbatch_msignal"]) + (data["minbatch_msignal"])))))), ((((data["maxtominbatch"]) * (np.maximum(((((data["maxtominbatch_msignal"]) + (data["minbatch_msignal"])))), ((data["abs_minbatch_slices2_msignal"]))))))))))/2.0)) + (data["minbatch_msignal"]))) + (((data["minbatch_msignal"]) + (data["maxtominbatch_msignal"]))))) +

                            0.050000*np.tanh(((((data["minbatch_slices2_msignal"]) + (((data["minbatch_slices2_msignal"]) + ((((data["abs_minbatch_msignal"]) + (np.where(np.where(data["signal"] <= -998, data["minbatch_msignal"], ((((((data["maxtominbatch_slices2_msignal"]) * 2.0)) * 2.0)) * 2.0) ) > -998, data["maxtominbatch_slices2_msignal"], data["maxbatch_slices2_msignal"] )))/2.0)))))) * 2.0)) +

                            0.050000*np.tanh((((((((data["maxtominbatch"]) + (((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)) + (data["minbatch_slices2_msignal"]))))/2.0)) * 2.0)) + ((((((data["minbatch_slices2_msignal"]) + (((data["abs_minbatch_msignal"]) + (data["maxtominbatch_slices2"]))))/2.0)) + (((data["minbatch_slices2_msignal"]) * 2.0)))))) +

                            0.050000*np.tanh(((((data["minbatch_slices2_msignal"]) - ((((-((np.maximum(((data["minbatch_slices2_msignal"])), ((np.maximum((((((-((((data["maxtominbatch_slices2_msignal"]) / 2.0))))) / 2.0))), ((np.maximum(((data["abs_minbatch_msignal"])), ((((np.maximum(((data["abs_minbatch_slices2_msignal"])), ((data["abs_minbatch_slices2_msignal"])))) / 2.0)))))))))))))) / 2.0)))) * 2.0)) +

                            0.050000*np.tanh(np.where(((data["abs_minbatch_msignal"]) * (data["abs_minbatch_msignal"])) <= -998, data["meanbatch_msignal"], ((data["abs_minbatch_slices2_msignal"]) - ((-((((((data["stdbatch_slices2_msignal"]) * (data["abs_avgbatch_slices2_msignal"]))) - (np.maximum((((12.22234630584716797))), ((data["abs_minbatch_slices2_msignal"])))))))))) )) +

                            0.050000*np.tanh(np.where(np.where(data["minbatch_slices2_msignal"] > -998, ((((data["minbatch_slices2_msignal"]) + (data["abs_maxbatch"]))) - (data["minbatch_slices2_msignal"])), (((data["maxtominbatch_slices2_msignal"]) <= (data["minbatch_slices2_msignal"]))*1.) ) <= -998, data["mean_abs_chgbatch_msignal"], ((data["minbatch_slices2_msignal"]) + (data["abs_minbatch_slices2_msignal"])) )) +

                            0.050000*np.tanh(((((data["abs_minbatch_msignal"]) + (((((data["minbatch_msignal"]) + ((((data["abs_minbatch_msignal"]) + (((((((data["abs_minbatch_msignal"]) + (data["maxtominbatch_slices2_msignal"]))/2.0)) + (((data["minbatch_msignal"]) / 2.0)))/2.0)))/2.0)))) - (data["rangebatch_slices2"]))))) * 2.0)) +

                            0.050000*np.tanh(((((data["maxtominbatch_slices2_msignal"]) + (data["minbatch_msignal"]))) * 2.0)) +

                            0.050000*np.tanh(((data["minbatch_slices2_msignal"]) - (np.minimum(((np.minimum(((data["signal_shift_+1"])), ((data["abs_minbatch_slices2_msignal"]))))), ((np.maximum((((((data["signal_shift_+1"]) <= (data["minbatch_slices2_msignal"]))*1.))), ((data["signal_shift_+1"]))))))))) +

                            0.050000*np.tanh(((data["abs_minbatch_slices2_msignal"]) + ((-((np.where(((np.where(((data["maxtominbatch_slices2_msignal"]) + (data["abs_maxbatch_msignal"])) > -998, data["abs_minbatch_slices2_msignal"], (((data["signal"]) + (np.tanh((np.where(data["maxtominbatch_msignal"] > -998, data["medianbatch_slices2_msignal"], data["minbatch_msignal"] )))))/2.0) )) + (data["medianbatch_slices2_msignal"])) <= -998, (((data["abs_minbatch_slices2_msignal"]) <= ((-((data["minbatch_msignal"])))))*1.), data["rangebatch_slices2"] ))))))) +

                            0.050000*np.tanh(((data["abs_minbatch_msignal"]) + (((((data["minbatch_slices2"]) + (np.where(data["minbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_msignal"], ((data["maxtominbatch_msignal"]) + (np.cos((((data["minbatch_slices2_msignal"]) + (np.minimum(((data["signal_shift_+1_msignal"])), (((((np.cos((((data["minbatch_slices2_msignal"]) / 2.0)))) + ((-((data["minbatch_slices2"])))))/2.0)))))))))) )))) * 2.0)))) +

                            0.050000*np.tanh(((data["minbatch_slices2_msignal"]) + (np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, data["minbatch_slices2_msignal"], data["maxtominbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((np.minimum(((np.where(data["mean_abs_chgbatch_slices2"] > -998, (((data["maxtominbatch_msignal"]) + (data["maxtominbatch_slices2_msignal"]))/2.0), ((data["minbatch"]) * 2.0) ))), ((data["minbatch_msignal"]))))))) + (((data["maxtominbatch_slices2_msignal"]) - (np.sin((np.tanh((np.sin((np.cos((((((((-((data["minbatch_slices2_msignal"])))) <= (data["stdbatch_msignal"]))*1.)) / 2.0)))))))))))))) +

                            0.050000*np.tanh(np.where(((data["stdbatch_msignal"]) - (((data["maxtominbatch"]) * 2.0))) > -998, ((data["minbatch_msignal"]) + (data["maxtominbatch_msignal"])), np.where(data["medianbatch_slices2"] > -998, data["maxtominbatch_msignal"], np.tanh((np.where(data["maxtominbatch_msignal"] > -998, np.where(data["abs_minbatch_slices2"] > -998, data["maxtominbatch_msignal"], data["maxbatch_slices2_msignal"] ), np.tanh((np.cos((np.sin((np.tanh((data["stdbatch_slices2_msignal"])))))))) ))) ) )) +

                            0.050000*np.tanh(np.where(np.where(data["signal"] > -998, data["minbatch_slices2_msignal"], ((((data["maxtominbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"]))) / 2.0) ) <= -998, np.where(((data["maxtominbatch_slices2_msignal"]) * (data["abs_avgbatch_slices2"])) <= -998, (((((data["maxtominbatch_msignal"]) + (data["minbatch_slices2_msignal"]))/2.0)) * 2.0), data["signal_shift_-1"] ), ((data["maxtominbatch_slices2_msignal"]) + (data["minbatch_slices2_msignal"])) )) +

                            0.050000*np.tanh(((((data["abs_minbatch_msignal"]) - (data["rangebatch_slices2"]))) * (((data["rangebatch_slices2"]) + ((-((np.sin(((((data["rangebatch_slices2"]) <= (((((data["abs_maxbatch_slices2_msignal"]) + ((-((data["abs_maxbatch_slices2_msignal"])))))) * (np.cos(((((data["maxtominbatch_msignal"]) <= (((((data["abs_minbatch_msignal"]) - (data["rangebatch_slices2"]))) / 2.0)))*1.)))))))*1.))))))))))) +

                            0.050000*np.tanh(((data["maxbatch_msignal"]) - (((data["rangebatch_slices2"]) - (((data["minbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"]))))))) +

                            0.050000*np.tanh(((np.where(data["maxtominbatch_msignal"] <= -998, ((((np.where(data["signal"] <= -998, (14.39103507995605469), data["abs_minbatch_slices2"] )) + (((((data["maxtominbatch_msignal"]) * (((((data["signal_shift_+1"]) - (data["minbatch_msignal"]))) * 2.0)))) * 2.0)))) / 2.0), ((((data["minbatch_msignal"]) - (data["signal_shift_+1"]))) * 2.0) )) * 2.0)) +

                            0.050000*np.tanh(((data["abs_minbatch_slices2_msignal"]) + (data["minbatch_msignal"]))) +

                            0.050000*np.tanh(np.where(np.where(data["minbatch_msignal"] <= -998, ((((((data["maxtominbatch_msignal"]) > (((((((data["minbatch_msignal"]) - (data["medianbatch_slices2"]))) * 2.0)) - (((data["minbatch_msignal"]) - (data["medianbatch_slices2"]))))))*1.)) > (data["minbatch_slices2_msignal"]))*1.), data["maxtominbatch_slices2_msignal"] ) <= -998, data["minbatch_slices2_msignal"], ((((((data["minbatch_msignal"]) - (data["medianbatch_slices2"]))) * 2.0)) * 2.0) )) +

                            0.050000*np.tanh(np.where(data["minbatch_slices2_msignal"] <= -998, ((((np.cos((data["abs_maxbatch_slices2_msignal"]))) - (data["maxtominbatch_slices2_msignal"]))) * 2.0), ((np.where(data["maxtominbatch_slices2_msignal"] > -998, data["maxtominbatch_msignal"], np.where(data["signal_shift_+1_msignal"] <= -998, ((((data["maxtominbatch_slices2_msignal"]) - (((((data["abs_minbatch_msignal"]) - (data["maxtominbatch_msignal"]))) * 2.0)))) * 2.0), data["mean_abs_chgbatch_slices2_msignal"] ) )) * (data["rangebatch_slices2_msignal"])) )) +

                            0.050000*np.tanh(np.where(np.maximum((((((data["stdbatch_msignal"]) > (data["maxtominbatch_msignal"]))*1.))), ((data["maxtominbatch_msignal"]))) > -998, data["maxtominbatch_slices2_msignal"], ((((np.where(data["minbatch_msignal"] > -998, data["rangebatch_msignal"], data["minbatch"] )) * 2.0)) / 2.0) )) +

                            0.050000*np.tanh(((((np.tanh((((data["meanbatch_msignal"]) / 2.0)))) + (data["maxtominbatch_msignal"]))) + (data["abs_avgbatch_slices2"]))) +

                            0.050000*np.tanh(((data["maxtominbatch_slices2_msignal"]) + (data["maxtominbatch_slices2"]))) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) + (((data["minbatch_msignal"]) + (((data["abs_minbatch_msignal"]) + (((data["minbatch_msignal"]) + (np.maximum(((np.where(data["medianbatch_slices2"] > -998, data["maxtominbatch_slices2_msignal"], (((data["abs_minbatch_msignal"]) + (data["minbatch_msignal"]))/2.0) ))), ((data["minbatch_msignal"])))))))))))) +

                            0.050000*np.tanh(np.where((((np.maximum(((data["stdbatch_slices2_msignal"])), (((((data["meanbatch_slices2_msignal"]) + ((-((((data["signal"]) - (((data["signal"]) - (data["minbatch_msignal"])))))))))/2.0))))) > (data["maxtominbatch_slices2_msignal"]))*1.) <= -998, data["medianbatch_slices2_msignal"], (-((((data["signal"]) - (np.minimum(((data["minbatch_msignal"])), ((data["maxtominbatch_msignal"])))))))) )) +

                            0.050000*np.tanh(np.minimum(((((data["maxtominbatch_slices2_msignal"]) + (((data["maxtominbatch_msignal"]) - (data["abs_minbatch_slices2_msignal"])))))), ((data["maxtominbatch_slices2_msignal"])))) +

                            0.050000*np.tanh(data["maxtominbatch_slices2_msignal"]) +

                            0.050000*np.tanh(((data["maxtominbatch"]) + (((((((np.maximum(((data["abs_minbatch_msignal"])), ((data["abs_minbatch_msignal"])))) + (((data["minbatch_msignal"]) * 2.0)))) - (data["signal_shift_+1"]))) - (((((data["signal_shift_+1"]) + (data["maxtominbatch_slices2_msignal"]))) / 2.0)))))) +

                            0.050000*np.tanh(((data["abs_maxbatch_msignal"]) + (np.minimum(((((((np.minimum(((data["minbatch_slices2_msignal"])), ((data["minbatch_slices2_msignal"])))) * 2.0)) * 2.0))), ((((data["maxtominbatch_slices2_msignal"]) * 2.0))))))) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) - (np.where(data["abs_minbatch_slices2_msignal"] <= -998, np.where(np.tanh((((data["maxtominbatch_slices2"]) * (data["maxtominbatch_slices2"])))) <= -998, np.where(data["mean_abs_chgbatch_msignal"] <= -998, data["abs_maxbatch_slices2_msignal"], data["signal_shift_-1"] ), data["maxbatch_slices2"] ), data["signal_shift_+1"] )))) +

                            0.050000*np.tanh(((data["maxtominbatch_slices2_msignal"]) - ((((data["minbatch_slices2"]) <= (np.where((((data["abs_minbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2_msignal"]))/2.0) > -998, data["maxtominbatch_slices2_msignal"], ((data["maxtominbatch_slices2_msignal"]) / 2.0) )))*1.)))) +

                            0.050000*np.tanh(((((data["maxbatch_msignal"]) + (np.where(np.where((((((data["maxtominbatch_slices2_msignal"]) * 2.0)) <= (np.cos((((np.minimum(((np.sin((data["abs_avgbatch_slices2"])))), ((data["abs_maxbatch"])))) / 2.0)))))*1.) <= -998, data["medianbatch_slices2_msignal"], data["maxtominbatch_slices2_msignal"] ) <= -998, data["medianbatch_slices2_msignal"], ((data["abs_minbatch_slices2_msignal"]) + (data["minbatch_slices2_msignal"])) )))) + (data["minbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((((data["abs_minbatch_slices2_msignal"]) + (((data["minbatch_msignal"]) + (data["minbatch_msignal"]))))) * (np.maximum(((np.tanh((data["stdbatch_slices2_msignal"])))), ((np.where(((data["maxtominbatch_slices2_msignal"]) + (np.sin((data["minbatch_msignal"])))) > -998, ((((10.0)) + (np.cos((np.maximum(((data["maxtominbatch_slices2_msignal"])), ((data["minbatch_msignal"])))))))/2.0), data["minbatch_slices2_msignal"] ))))))) +

                            0.050000*np.tanh(((((np.minimum(((data["minbatch_slices2_msignal"])), ((data["minbatch_msignal"])))) + (((((np.where(((data["maxbatch_slices2_msignal"]) / 2.0) <= -998, data["minbatch_slices2_msignal"], data["abs_minbatch_msignal"] )) + (data["minbatch_msignal"]))) + ((-((((data["abs_minbatch_slices2_msignal"]) - (np.cos(((0.0))))))))))))) + (((data["mean_abs_chgbatch_slices2_msignal"]) + ((((data["abs_maxbatch"]) > (data["signal_shift_+1"]))*1.)))))) +

                            0.050000*np.tanh(((data["rangebatch_msignal"]) * ((((data["maxtominbatch_msignal"]) + (np.where(data["medianbatch_slices2"] > -998, data["minbatch_msignal"], data["maxtominbatch_slices2_msignal"] )))/2.0)))) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) - (np.where(data["signal_shift_+1_msignal"] > -998, np.minimum(((data["medianbatch_slices2"])), ((np.where(np.where(np.tanh((data["signal_shift_-1"])) > -998, data["signal_shift_-1"], data["minbatch_msignal"] ) <= -998, data["maxtominbatch_slices2_msignal"], (-((data["maxtominbatch_slices2_msignal"]))) )))), data["signal_shift_-1"] )))) +

                            0.050000*np.tanh(np.minimum(((((np.minimum(((data["minbatch_msignal"])), ((np.where(data["maxtominbatch_slices2_msignal"] <= -998, ((((data["maxtominbatch_slices2_msignal"]) * 2.0)) / 2.0), data["maxtominbatch_slices2_msignal"] ))))) + (data["maxtominbatch_msignal"])))), ((((((data["maxtominbatch_slices2_msignal"]) + (np.cos((data["signal_shift_+1_msignal"]))))) / 2.0))))) +

                            0.050000*np.tanh(np.where(((((data["rangebatch_slices2"]) * (((data["minbatch_msignal"]) + (((data["minbatch_msignal"]) + (data["maxtominbatch_slices2_msignal"]))))))) + (((data["minbatch_msignal"]) - (data["medianbatch_msignal"])))) <= -998, ((data["maxtominbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"])), ((data["minbatch_msignal"]) + (data["maxtominbatch_slices2_msignal"])) )) +

                            0.050000*np.tanh(np.where(np.tanh((((np.where(data["stdbatch_slices2_msignal"] > -998, data["maxtominbatch_msignal"], np.where(((data["maxtominbatch_msignal"]) - (data["minbatch_slices2"])) > -998, data["maxtominbatch_msignal"], (((data["abs_avgbatch_slices2_msignal"]) <= (np.where(data["signal_shift_-1_msignal"] <= -998, data["maxtominbatch_msignal"], (((data["maxtominbatch_msignal"]) <= (data["mean_abs_chgbatch_msignal"]))*1.) )))*1.) ) )) * (data["signal_shift_+1"])))) > -998, data["maxtominbatch_msignal"], data["abs_maxbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(np.where(data["meanbatch_slices2_msignal"] > -998, data["maxtominbatch_slices2_msignal"], ((data["maxtominbatch_slices2_msignal"]) + (data["minbatch_slices2_msignal"])) )) +

                            0.050000*np.tanh(np.maximum(((data["maxtominbatch_slices2_msignal"])), ((np.where(np.where(np.minimum((((5.04649209976196289))), ((np.tanh((data["maxtominbatch_slices2_msignal"]))))) <= -998, data["stdbatch_msignal"], data["signal_shift_+1"] ) <= -998, data["signal_shift_+1"], np.where(data["medianbatch_slices2"] > -998, np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, data["signal_shift_+1"], data["abs_minbatch_msignal"] ), data["maxtominbatch_slices2_msignal"] ) ))))) +

                            0.050000*np.tanh(data["maxtominbatch_msignal"]) +

                            0.050000*np.tanh(((np.where(((data["abs_avgbatch_slices2_msignal"]) - (np.maximum(((((data["abs_avgbatch_slices2"]) / 2.0))), ((((((np.cos((data["signal_shift_-1"]))) * (data["mean_abs_chgbatch_msignal"]))) - (data["mean_abs_chgbatch_msignal"]))))))) <= -998, np.where(((data["abs_minbatch_msignal"]) * 2.0) > -998, data["minbatch_msignal"], data["stdbatch_msignal"] ), ((data["maxtominbatch_slices2_msignal"]) + ((-((((np.cos((data["meanbatch_msignal"]))) * 2.0)))))) )) * 2.0)) +

                            0.050000*np.tanh(((((data["minbatch_msignal"]) + (data["abs_minbatch_slices2_msignal"]))) + (np.where(((data["minbatch_slices2_msignal"]) * (np.where(((data["maxtominbatch_msignal"]) + (data["stdbatch_slices2_msignal"])) <= -998, data["abs_minbatch_slices2_msignal"], np.cos((data["mean_abs_chgbatch_msignal"])) ))) > -998, data["minbatch_msignal"], (((data["abs_minbatch_slices2_msignal"]) + (np.cos((data["abs_maxbatch_slices2"]))))/2.0) )))) +

                            0.050000*np.tanh((((data["minbatch_slices2_msignal"]) + (((np.where((((data["mean_abs_chgbatch_slices2"]) <= (data["maxbatch_slices2"]))*1.) > -998, data["mean_abs_chgbatch_slices2_msignal"], data["maxtominbatch"] )) - ((((data["meanbatch_msignal"]) <= (((((np.where(data["abs_minbatch_slices2_msignal"] > -998, data["abs_minbatch_msignal"], ((((data["minbatch_slices2_msignal"]) - (data["mean_abs_chgbatch_slices2_msignal"]))) + (data["abs_avgbatch_msignal"])) )) * 2.0)) * 2.0)))*1.)))))/2.0)) +

                            0.050000*np.tanh(np.where(np.sin((((np.sin(((((data["abs_avgbatch_slices2_msignal"]) + (np.where(data["abs_minbatch_slices2"] > -998, data["maxtominbatch_msignal"], np.sin((np.minimum(((data["meanbatch_slices2"])), ((data["signal_shift_+1"]))))) )))/2.0)))) * 2.0))) > -998, data["maxtominbatch_msignal"], data["stdbatch_msignal"] )) +

                            0.050000*np.tanh(np.minimum(((np.where((((((data["maxtominbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))) + (np.cos((data["abs_minbatch_slices2_msignal"]))))/2.0) > -998, np.minimum(((data["abs_minbatch_slices2_msignal"])), ((data["abs_avgbatch_slices2_msignal"]))), data["maxtominbatch_slices2_msignal"] ))), ((data["abs_minbatch_slices2_msignal"])))) +

                            0.050000*np.tanh(((((((((((data["maxtominbatch_slices2_msignal"]) * 2.0)) * 2.0)) / 2.0)) / 2.0)) - (data["abs_avgbatch_msignal"]))) +

                            0.050000*np.tanh(np.where(data["maxtominbatch_msignal"] <= -998, (((((data["maxtominbatch_slices2_msignal"]) > (((((((data["maxtominbatch_slices2_msignal"]) * 2.0)) / 2.0)) * 2.0)))*1.)) / 2.0), ((data["mean_abs_chgbatch_msignal"]) + (np.minimum(((data["minbatch_msignal"])), ((np.maximum((((13.61060047149658203))), ((((data["abs_minbatch_msignal"]) - (np.sin((((data["abs_minbatch_msignal"]) + (data["mean_abs_chgbatch_msignal"])))))))))))))) )) +

                            0.050000*np.tanh(((data["minbatch_slices2_msignal"]) + (((((data["abs_minbatch_msignal"]) / 2.0)) + (np.minimum((((((((4.05654144287109375)) / 2.0)) * 2.0))), (((((((data["abs_minbatch_msignal"]) / 2.0)) > (data["abs_minbatch_slices2_msignal"]))*1.))))))))) +

                            0.050000*np.tanh(((data["abs_minbatch_msignal"]) + (np.where(np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["maxtominbatch_slices2_msignal"], np.where(data["meanbatch_slices2"] > -998, (-((data["minbatch_slices2_msignal"]))), (((data["stdbatch_msignal"]) > (np.where(data["minbatch_slices2_msignal"] > -998, data["minbatch_slices2_msignal"], (((data["mean_abs_chgbatch_msignal"]) > (data["minbatch_slices2_msignal"]))*1.) )))*1.) ) ) > -998, data["maxtominbatch_slices2_msignal"], data["minbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) + (((data["abs_minbatch_msignal"]) + (np.where(np.minimum(((((data["meanbatch_slices2"]) + (data["minbatch_msignal"])))), ((data["minbatch_msignal"]))) <= -998, ((data["minbatch_msignal"]) / 2.0), ((np.sin((data["minbatch_slices2_msignal"]))) + (((data["abs_minbatch_msignal"]) + (np.where(data["minbatch_slices2_msignal"] <= -998, data["stdbatch_slices2"], data["minbatch_msignal"] ))))) )))))) +

                            0.050000*np.tanh((((data["maxtominbatch_msignal"]) + (np.minimum(((data["maxtominbatch_msignal"])), ((np.maximum(((data["rangebatch_slices2"])), ((((data["minbatch_slices2_msignal"]) - (data["signal_shift_+1"]))))))))))/2.0)) +

                            0.050000*np.tanh(np.where(data["maxtominbatch_msignal"] > -998, ((data["minbatch_msignal"]) - (data["signal_shift_+1"])), np.where(data["abs_minbatch_msignal"] > -998, data["abs_avgbatch_msignal"], ((data["abs_minbatch_msignal"]) - (data["signal_shift_+1"])) ) )) +

                            0.050000*np.tanh(np.minimum(((np.where(np.where(data["maxtominbatch_msignal"] > -998, np.sin((((data["maxtominbatch_msignal"]) * 2.0))), ((data["maxtominbatch_msignal"]) * 2.0) ) > -998, data["abs_minbatch_msignal"], data["maxtominbatch_slices2"] ))), ((((np.minimum(((data["rangebatch_slices2"])), ((np.cos((((data["abs_minbatch_msignal"]) * 2.0))))))) * 2.0))))) +

                            0.050000*np.tanh(((data["maxtominbatch_slices2_msignal"]) + (np.where(((data["stdbatch_msignal"]) / 2.0) > -998, (-(((((np.sin((data["abs_avgbatch_slices2_msignal"]))) > (((data["maxtominbatch_slices2_msignal"]) - (np.sin(((((data["minbatch_slices2"]) <= (data["medianbatch_msignal"]))*1.)))))))*1.)))), np.maximum(((np.sin((data["signal_shift_-1"])))), ((data["minbatch_msignal"]))) )))) +

                            0.050000*np.tanh(((((data["maxtominbatch_msignal"]) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((np.where(data["meanbatch_slices2_msignal"] > -998, ((data["abs_minbatch_slices2_msignal"]) * (np.cos((((data["maxtominbatch_slices2_msignal"]) + (data["stdbatch_msignal"])))))), ((((data["maxtominbatch_slices2_msignal"]) * (np.cos((((data["maxtominbatch_slices2_msignal"]) + (data["stdbatch_msignal"]))))))) * (((((((data["abs_minbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"]))/2.0)) <= (data["maxtominbatch_slices2_msignal"]))*1.))) )) * (data["abs_minbatch_msignal"]))) +

                            0.050000*np.tanh(((data["maxtominbatch_slices2_msignal"]) - ((((((data["mean_abs_chgbatch_slices2_msignal"]) - (data["maxtominbatch_slices2_msignal"]))) > (data["maxtominbatch_slices2_msignal"]))*1.)))) +

                            0.050000*np.tanh(np.where(data["stdbatch_slices2_msignal"] > -998, ((data["maxtominbatch_slices2_msignal"]) - ((((np.tanh((((((data["stdbatch_msignal"]) + (data["abs_maxbatch_slices2_msignal"]))) * 2.0)))) <= (data["stdbatch_slices2_msignal"]))*1.))), np.sin((data["maxtominbatch_slices2_msignal"])) )) +

                            0.050000*np.tanh(np.where(((data["maxtominbatch_slices2_msignal"]) + ((((data["minbatch_slices2_msignal"]) + (np.where(data["abs_minbatch_msignal"] > -998, data["maxtominbatch_msignal"], np.tanh((data["abs_minbatch_msignal"])) )))/2.0))) <= -998, data["maxtominbatch_slices2_msignal"], ((data["abs_minbatch_msignal"]) / 2.0) )) +

                            0.050000*np.tanh(((np.maximum(((((data["medianbatch_slices2"]) + (((((np.maximum(((data["abs_maxbatch_msignal"])), ((np.where(data["stdbatch_slices2"] > -998, data["maxtominbatch_msignal"], data["maxtominbatch"] ))))) * 2.0)) - (data["maxtominbatch_msignal"])))))), ((np.where((1.0) > -998, data["maxtominbatch_msignal"], np.where(data["maxtominbatch_slices2_msignal"] <= -998, np.sin((data["rangebatch_slices2_msignal"])), (1.0) ) ))))) * (((data["minbatch_msignal"]) - (data["signal_shift_-1"]))))) +

                            0.050000*np.tanh(np.where(data["maxtominbatch_msignal"] <= -998, np.minimum(((data["mean_abs_chgbatch_slices2"])), ((data["minbatch_msignal"]))), (((np.minimum((((((data["abs_avgbatch_slices2"]) + (data["meanbatch_slices2"]))/2.0))), ((data["minbatch_msignal"])))) + (data["abs_minbatch_slices2_msignal"]))/2.0) )) +

                            0.050000*np.tanh(((data["abs_minbatch_slices2_msignal"]) - ((((np.where(data["maxbatch_msignal"] > -998, data["mean_abs_chgbatch_slices2_msignal"], data["maxtominbatch"] )) + (data["maxtominbatch"]))/2.0)))) +

                            0.050000*np.tanh(np.where(data["minbatch"] > -998, np.where(np.sin(((((((((data["signal_shift_-1"]) + (data["signal_shift_-1"]))/2.0)) + (data["maxtominbatch_msignal"]))) * 2.0))) > -998, (-((((data["signal_shift_-1"]) + (data["mean_abs_chgbatch_slices2"]))))), ((((data["signal_shift_-1"]) + (((data["signal_shift_-1"]) + (data["rangebatch_slices2"]))))) * 2.0) ), np.sin((((data["abs_minbatch_slices2_msignal"]) * 2.0))) )) +

                            0.050000*np.tanh(((np.where(data["rangebatch_slices2"] > -998, (((((((data["minbatch_slices2_msignal"]) + (np.minimum(((data["maxtominbatch_msignal"])), ((np.where(data["abs_avgbatch_slices2_msignal"] > -998, (9.0), data["minbatch_slices2_msignal"] ))))))) + (np.tanh((data["mean_abs_chgbatch_msignal"]))))) + (data["minbatch_slices2_msignal"]))/2.0), data["minbatch_msignal"] )) * 2.0)) +

                            0.050000*np.tanh(((((np.where(data["minbatch_slices2_msignal"] > -998, (((data["abs_minbatch_slices2_msignal"]) + (((((data["minbatch_slices2_msignal"]) * 2.0)) / 2.0)))/2.0), data["rangebatch_slices2"] )) + (((((data["rangebatch_slices2"]) - (data["mean_abs_chgbatch_slices2_msignal"]))) * (((data["minbatch_slices2_msignal"]) * 2.0)))))) * 2.0)) +

                            0.050000*np.tanh((((data["maxtominbatch_slices2_msignal"]) + (((np.tanh((np.cos((((data["signal_shift_-1"]) * 2.0)))))) * (((data["abs_maxbatch_msignal"]) + (np.cos((np.maximum((((-((np.cos((((data["signal_shift_-1"]) * 2.0)))))))), ((((np.tanh((np.minimum(((data["maxbatch_msignal"])), ((data["maxtominbatch_msignal"])))))) / 2.0))))))))))))/2.0)) +

                            0.050000*np.tanh(np.minimum(((data["maxtominbatch_msignal"])), ((np.where(data["abs_avgbatch_slices2_msignal"] <= -998, data["abs_minbatch_slices2"], np.tanh((data["mean_abs_chgbatch_msignal"])) ))))) +

                            0.050000*np.tanh(((((data["abs_minbatch_msignal"]) - (data["signal"]))) / 2.0)) +

                            0.050000*np.tanh(((data["meanbatch_slices2"]) * ((-((data["abs_maxbatch_slices2"])))))) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) + (np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["maxtominbatch_msignal"], ((data["medianbatch_msignal"]) + (np.where(data["abs_minbatch_slices2"] <= -998, (((data["meanbatch_msignal"]) + (((data["abs_minbatch_msignal"]) * (data["abs_minbatch_slices2_msignal"]))))/2.0), (-(((((11.39513206481933594)) + (data["maxtominbatch_msignal"]))))) ))) )))) +

                            0.050000*np.tanh(np.cos((((np.where(data["rangebatch_slices2"] <= -998, data["rangebatch_slices2"], np.where(data["mean_abs_chgbatch_slices2"] > -998, np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["signal_shift_+1"], ((np.cos((((data["signal_shift_+1"]) * 2.0)))) * (((data["rangebatch_slices2_msignal"]) * 2.0))) ), np.where(data["signal_shift_+1"] <= -998, (-((data["signal_shift_-1"]))), data["abs_maxbatch_slices2_msignal"] ) ) )) * 2.0)))) +

                            0.050000*np.tanh(((data["abs_minbatch_slices2_msignal"]) + (((np.sin((data["abs_maxbatch_msignal"]))) * (((data["abs_maxbatch_msignal"]) * (np.cos(((((data["abs_maxbatch_msignal"]) <= ((((data["abs_maxbatch_slices2"]) <= (((((((np.minimum(((data["mean_abs_chgbatch_slices2"])), ((data["abs_maxbatch_msignal"])))) + (((data["signal_shift_-1"]) + (data["maxtominbatch_msignal"]))))/2.0)) <= (data["signal_shift_-1"]))*1.)))*1.)))*1.)))))))))) +

                            0.050000*np.tanh(((data["abs_avgbatch_slices2_msignal"]) * (((((data["abs_maxbatch_msignal"]) * (np.where(data["abs_maxbatch_msignal"] > -998, np.sin((np.sin((data["abs_maxbatch_msignal"])))), np.maximum(((data["abs_avgbatch_slices2_msignal"])), ((np.tanh((data["minbatch_slices2"]))))) )))) - (data["signal_shift_-1"]))))) +

                            0.050000*np.tanh(((data["maxtominbatch_slices2_msignal"]) - (data["medianbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(np.where((((data["abs_minbatch_msignal"]) <= ((((np.where(data["abs_minbatch_slices2_msignal"] <= -998, np.where((-((data["maxtominbatch_slices2_msignal"]))) > -998, data["abs_minbatch_msignal"], np.tanh((data["meanbatch_slices2"])) ), np.maximum(((data["abs_minbatch_slices2_msignal"])), ((data["abs_minbatch_msignal"]))) )) <= (np.where(data["mean_abs_chgbatch_msignal"] > -998, data["meanbatch_slices2_msignal"], np.tanh((np.cos((data["abs_maxbatch"])))) )))*1.)))*1.) > -998, data["mean_abs_chgbatch_slices2_msignal"], ((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0) )) +

                            0.050000*np.tanh(data["maxtominbatch_msignal"]) +

                            0.050000*np.tanh(((np.sin((np.where(data["meanbatch_msignal"] > -998, ((data["signal_shift_+1"]) - ((((np.minimum(((data["meanbatch_msignal"])), ((data["abs_maxbatch_slices2_msignal"])))) > (data["minbatch_slices2_msignal"]))*1.))), np.sin((data["maxtominbatch_msignal"])) )))) * 2.0)) +

                            0.050000*np.tanh(((np.tanh((np.cos((np.minimum(((((data["meanbatch_slices2"]) * 2.0))), ((data["minbatch_msignal"])))))))) + (((np.cos((np.minimum(((((data["meanbatch_slices2"]) * 2.0))), ((((((data["stdbatch_msignal"]) + (((((np.cos((data["meanbatch_slices2"]))) * 2.0)) * 2.0)))) + (data["rangebatch_msignal"])))))))) * 2.0)))) +

                            0.050000*np.tanh(((data["maxtominbatch_slices2_msignal"]) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((np.minimum(((np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_msignal"], ((data["abs_maxbatch_slices2"]) * ((((((data["mean_abs_chgbatch_msignal"]) <= (data["abs_minbatch_msignal"]))*1.)) + (data["maxbatch_slices2"])))) ))), ((((np.minimum(((data["abs_maxbatch_msignal"])), ((np.sin((data["abs_maxbatch_msignal"])))))) * 2.0)))))), ((np.sin((data["abs_maxbatch_msignal"])))))) +

                            0.050000*np.tanh((-((np.cos((np.minimum(((data["signal_shift_-1"])), (((((data["abs_avgbatch_msignal"]) > ((((data["abs_avgbatch_msignal"]) > (data["signal_shift_-1"]))*1.)))*1.)))))))))) +

                            0.050000*np.tanh(np.sin((np.where((-((np.minimum(((data["minbatch_slices2"])), ((data["abs_maxbatch_msignal"])))))) <= -998, np.sin((data["signal_shift_-1"])), np.where(((np.sin((data["abs_maxbatch_msignal"]))) * (data["mean_abs_chgbatch_slices2"])) > -998, data["abs_maxbatch_msignal"], (-((np.where(data["signal_shift_-1_msignal"] <= -998, data["meanbatch_slices2_msignal"], np.where(np.sin((data["abs_maxbatch_msignal"])) <= -998, np.sin((data["meanbatch_slices2_msignal"])), data["abs_maxbatch_msignal"] ) )))) ) )))) +

                            0.050000*np.tanh(((((((np.where(data["signal"] > -998, data["stdbatch_msignal"], (4.18577051162719727) )) / 2.0)) + (((np.where(np.tanh((np.minimum(((np.tanh((data["abs_minbatch_msignal"])))), (((14.12254238128662109)))))) > -998, data["stdbatch_msignal"], np.minimum(((data["minbatch_slices2"])), ((((data["stdbatch_msignal"]) / 2.0)))) )) / 2.0)))) + (np.sin((data["minbatch_slices2"]))))) +

                            0.050000*np.tanh(data["abs_maxbatch_slices2_msignal"]) +

                            0.050000*np.tanh(np.where(data["signal"] > -998, ((np.cos((((data["signal_shift_-1"]) * 2.0)))) * (np.maximum(((data["abs_avgbatch_msignal"])), (((-((data["signal_shift_-1"])))))))), (((data["abs_avgbatch_msignal"]) > (((((data["mean_abs_chgbatch_msignal"]) * (data["stdbatch_slices2_msignal"]))) * 2.0)))*1.) )) +

                            0.050000*np.tanh(((np.where(np.cos((((np.cos((np.cos((((data["signal_shift_+1"]) * 2.0)))))) * 2.0))) <= -998, np.tanh((np.where(data["mean_abs_chgbatch_msignal"] <= -998, data["signal_shift_-1_msignal"], (-((np.sin((data["mean_abs_chgbatch_msignal"]))))) ))), (-((data["signal_shift_+1"]))) )) * 2.0)) +

                            0.050000*np.tanh((((data["abs_minbatch_msignal"]) + (np.where((((data["maxbatch_slices2_msignal"]) + (data["signal_shift_-1_msignal"]))/2.0) <= -998, ((np.tanh((data["abs_minbatch_slices2"]))) * (np.cos((data["maxtominbatch_slices2_msignal"])))), data["minbatch_msignal"] )))/2.0)) +

                            0.050000*np.tanh(data["maxtominbatch_slices2"]) +

                            0.050000*np.tanh(np.where(data["maxbatch_msignal"] <= -998, np.where(data["rangebatch_slices2"] > -998, np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, data["maxtominbatch_slices2_msignal"], ((data["signal_shift_+1_msignal"]) * (data["minbatch"])) ), data["signal_shift_+1_msignal"] ), ((data["signal_shift_+1_msignal"]) * (data["minbatch"])) )) +

                            0.050000*np.tanh(((((data["abs_maxbatch_slices2_msignal"]) * (np.tanh((np.cos((((data["abs_maxbatch_slices2_msignal"]) - (np.cos((np.cos((np.cos((np.cos((((np.cos((np.cos((np.minimum(((((data["stdbatch_msignal"]) * (data["abs_maxbatch_msignal"])))), ((np.cos((data["abs_maxbatch_msignal"])))))))))) * 2.0)))))))))))))))))) * 2.0)) +

                            0.050000*np.tanh(((data["signal_shift_-1_msignal"]) * (((data["minbatch_slices2"]) - (np.where(((((np.sin((data["signal_shift_-1"]))) - (((data["minbatch_slices2"]) / 2.0)))) + (data["signal_shift_-1_msignal"])) > -998, data["abs_maxbatch_slices2"], np.sin((np.sin((np.sin((data["signal_shift_+1"])))))) )))))) +

                            0.050000*np.tanh(((np.maximum(((data["minbatch_msignal"])), ((data["minbatch_slices2"])))) - (np.where(data["minbatch_slices2"] > -998, data["signal_shift_-1"], ((data["mean_abs_chgbatch_slices2"]) - ((((data["minbatch_slices2"]) <= ((-(((((np.maximum(((data["abs_maxbatch"])), (((((data["minbatch_msignal"]) > (data["signal_shift_-1_msignal"]))*1.))))) <= ((4.33556270599365234)))*1.))))))*1.))) )))) +

                            0.050000*np.tanh(np.where(data["abs_minbatch_msignal"] <= -998, (2.0), ((((data["minbatch_slices2_msignal"]) + (data["abs_minbatch_msignal"]))) * (data["mean_abs_chgbatch_msignal"])) )) +

                            0.050000*np.tanh(((((np.where((((np.cos((((((np.sin((data["abs_maxbatch_msignal"]))) * 2.0)) + (data["mean_abs_chgbatch_slices2_msignal"]))))) + (((data["stdbatch_slices2_msignal"]) / 2.0)))/2.0) <= -998, data["abs_maxbatch_msignal"], np.sin((data["signal_shift_-1"])) )) - (np.sin((data["abs_avgbatch_msignal"]))))) * 2.0)) +

                            0.050000*np.tanh(np.cos((((np.where((((data["abs_minbatch_slices2"]) > (np.where(data["signal_shift_-1"] <= -998, ((data["abs_minbatch_msignal"]) * 2.0), ((np.where(((data["medianbatch_msignal"]) / 2.0) <= -998, ((np.where(data["signal_shift_-1"] <= -998, data["maxtominbatch_msignal"], data["maxbatch_msignal"] )) * 2.0), data["signal_shift_-1"] )) * 2.0) )))*1.) <= -998, data["signal_shift_-1"], data["signal_shift_-1"] )) * 2.0)))) +

                            0.050000*np.tanh(((((((np.sin((data["abs_maxbatch_msignal"]))) * 2.0)) + ((-((np.where(((data["stdbatch_msignal"]) - (np.cos(((10.03387546539306641))))) > -998, np.sin((data["abs_avgbatch_slices2_msignal"])), np.minimum(((data["abs_maxbatch_msignal"])), ((data["abs_maxbatch_slices2"]))) ))))))) * 2.0)) +

                            0.050000*np.tanh(np.where(data["maxtominbatch"] <= -998, data["medianbatch_slices2_msignal"], (-((np.where(data["signal_shift_-1"] > -998, ((((data["signal_shift_-1"]) * 2.0)) - (data["minbatch"])), (((data["minbatch"]) > (np.where(data["rangebatch_msignal"] <= -998, ((((data["signal_shift_-1"]) * 2.0)) - (data["minbatch"])), ((data["signal_shift_-1"]) / 2.0) )))*1.) )))) )) +

                            0.050000*np.tanh(((np.where(data["abs_minbatch_slices2_msignal"] > -998, np.cos((((np.where((((data["maxtominbatch_msignal"]) > (np.cos((data["medianbatch_msignal"]))))*1.) > -998, data["abs_minbatch_msignal"], (-((data["maxtominbatch_msignal"]))) )) * 2.0))), data["medianbatch_slices2_msignal"] )) * 2.0)) +

                            0.050000*np.tanh(np.cos((np.maximum(((np.sin((np.cos((data["rangebatch_msignal"])))))), ((np.maximum((((((np.maximum(((np.cos((np.tanh(((-((((data["maxbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2"]))))))))))), ((data["abs_minbatch_slices2"])))) + (data["minbatch"]))/2.0))), ((data["rangebatch_msignal"]))))))))) +

                            0.050000*np.tanh(((((np.sin((np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) + (np.where(np.sin((data["maxtominbatch_msignal"])) > -998, data["abs_maxbatch"], np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) + (np.where(data["maxbatch_slices2"] > -998, data["abs_maxbatch"], ((data["mean_abs_chgbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"])) ))))) )))))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.cos((np.where((0.0) <= -998, np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, ((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0), ((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0) ), ((np.cos((np.where(((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0) <= -998, np.minimum((((((data["abs_maxbatch_slices2"]) + (data["abs_avgbatch_slices2_msignal"]))/2.0))), ((data["signal_shift_+1"]))), ((data["signal_shift_+1"]) * 2.0) )))) * 2.0) )))) +

                            0.050000*np.tanh(((((data["minbatch_msignal"]) - (((((((data["signal_shift_+1"]) * 2.0)) / 2.0)) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(((((data["rangebatch_msignal"]) * ((((data["signal_shift_-1"]) + (((data["abs_maxbatch"]) / 2.0)))/2.0)))) * (np.minimum(((np.where(data["signal_shift_+1"] <= -998, data["abs_maxbatch"], data["signal_shift_-1"] ))), (((((((1.70269882678985596)) * (data["abs_maxbatch"]))) * (data["signal_shift_+1"])))))))) +

                            0.050000*np.tanh(np.cos((((np.where(data["signal_shift_+1"] > -998, data["signal_shift_+1"], np.maximum(((data["mean_abs_chgbatch_slices2"])), ((np.where(np.cos((data["medianbatch_slices2_msignal"])) > -998, (-((np.minimum(((data["stdbatch_slices2_msignal"])), ((((((data["rangebatch_slices2_msignal"]) - (data["maxtominbatch_slices2"]))) * 2.0))))))), np.where(((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0) <= -998, ((data["abs_maxbatch_msignal"]) * 2.0), data["medianbatch_slices2_msignal"] ) )))) )) * 2.0)))) +

                            0.050000*np.tanh(np.where((((((np.minimum(((data["abs_avgbatch_slices2"])), ((np.sin((((data["minbatch_slices2_msignal"]) * (((data["mean_abs_chgbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"])))))))))) / 2.0)) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0) <= -998, ((((-((data["medianbatch_msignal"])))) <= (np.sin((data["signal_shift_+1"]))))*1.), np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"])))) )) +

                            0.050000*np.tanh(((np.sin(((((((((data["abs_maxbatch_slices2_msignal"]) / 2.0)) + (data["abs_maxbatch_slices2_msignal"]))/2.0)) + ((2.03542518615722656)))))) * 2.0)) +

                            0.050000*np.tanh(np.where((((((((np.maximum(((data["abs_minbatch_slices2"])), ((((((data["abs_maxbatch_msignal"]) - (((data["signal"]) * 2.0)))) * 2.0))))) / 2.0)) / 2.0)) <= (((data["signal"]) * 2.0)))*1.) <= -998, ((data["mean_abs_chgbatch_msignal"]) * 2.0), ((np.cos(((((data["abs_minbatch_slices2"]) + (((data["abs_maxbatch_slices2_msignal"]) - (((data["signal"]) * 2.0)))))/2.0)))) * 2.0) )) +

                            0.050000*np.tanh(data["maxtominbatch_msignal"]) +

                            0.050000*np.tanh(np.where(data["rangebatch_slices2"] > -998, ((data["minbatch_slices2_msignal"]) - (data["signal_shift_-1"])), np.where(data["rangebatch_slices2"] <= -998, data["minbatch_slices2_msignal"], np.where(((data["signal_shift_-1"]) - (data["medianbatch_slices2_msignal"])) <= -998, data["abs_maxbatch_slices2"], data["minbatch_slices2_msignal"] ) ) )) +

                            0.050000*np.tanh(((data["stdbatch_slices2_msignal"]) * (np.cos((((np.maximum(((data["medianbatch_msignal"])), ((np.where(data["abs_maxbatch_slices2"] <= -998, data["rangebatch_msignal"], data["minbatch_slices2"] ))))) - (((np.where(data["maxbatch_slices2_msignal"] <= -998, np.maximum(((data["signal_shift_+1_msignal"])), ((data["medianbatch_msignal"]))), data["maxtominbatch_slices2_msignal"] )) * 2.0)))))))) +

                            0.050000*np.tanh(np.minimum(((np.cos((np.maximum(((np.tanh((data["signal_shift_-1_msignal"])))), ((((data["signal_shift_-1_msignal"]) * 2.0)))))))), (((((((data["signal_shift_+1_msignal"]) <= (data["signal_shift_-1_msignal"]))*1.)) * 2.0))))) +

                            0.050000*np.tanh(np.minimum(((np.sin((((((np.minimum(((np.minimum(((data["signal_shift_-1"])), (((3.79476165771484375)))))), ((np.sin(((1.43593347072601318))))))) * (data["maxtominbatch"]))) * 2.0))))), (((-((data["maxtominbatch"]))))))) +

                            0.050000*np.tanh(((np.where(np.minimum(((np.cos((data["rangebatch_msignal"])))), (((((data["abs_avgbatch_msignal"]) + (data["maxbatch_slices2_msignal"]))/2.0)))) > -998, np.cos((data["abs_maxbatch_msignal"])), np.cos(((((data["abs_avgbatch_msignal"]) + (data["signal_shift_-1"]))/2.0))) )) * 2.0)) +

                            0.050000*np.tanh(((((np.sin((((data["abs_maxbatch_slices2_msignal"]) - (np.sin((np.sin((data["minbatch_msignal"]))))))))) * (data["mean_abs_chgbatch_slices2_msignal"]))) * (np.maximum(((((data["maxbatch_slices2"]) - (np.sin((((np.sin((data["maxtominbatch_slices2"]))) - (data["abs_maxbatch_slices2_msignal"])))))))), ((np.sin((np.cos((((data["abs_maxbatch_slices2_msignal"]) - (((data["meanbatch_slices2"]) / 2.0))))))))))))) +

                            0.050000*np.tanh(np.cos(((((((-((np.maximum(((data["signal_shift_+1_msignal"])), (((-((np.where((-((data["abs_maxbatch_msignal"]))) > -998, data["signal_shift_+1_msignal"], np.cos((data["signal_shift_+1_msignal"])) ))))))))))) + (data["signal_shift_+1"]))) * 2.0)))) +

                            0.050000*np.tanh(np.sin((np.cos((((np.where(np.cos((data["minbatch_slices2"])) <= -998, data["stdbatch_msignal"], data["stdbatch_slices2_msignal"] )) * (((data["signal"]) + (np.tanh((data["stdbatch_msignal"]))))))))))) +

                            0.050000*np.tanh(np.tanh((data["mean_abs_chgbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(np.cos((np.cos((((((data["mean_abs_chgbatch_msignal"]) - (((data["maxtominbatch_slices2_msignal"]) * (np.where(data["maxtominbatch_msignal"] <= -998, data["abs_maxbatch_msignal"], np.where(data["signal_shift_+1"] <= -998, np.where(np.cos((data["abs_maxbatch_msignal"])) <= -998, data["abs_maxbatch"], (0.0) ), (0.0) ) )))))) * 2.0)))))) +

                            0.050000*np.tanh(np.cos((((np.where(data["abs_maxbatch_slices2_msignal"] > -998, (((((data["abs_avgbatch_slices2_msignal"]) * 2.0)) + (data["meanbatch_slices2_msignal"]))/2.0), np.cos((data["minbatch_slices2"])) )) - (np.cos((data["signal_shift_-1"]))))))) +

                            0.050000*np.tanh((-((((((np.maximum((((-((data["signal_shift_-1"]))))), ((((data["signal_shift_+1"]) * 2.0))))) * ((-(((((data["abs_minbatch_slices2"]) + (((data["signal_shift_+1"]) / 2.0)))/2.0))))))) * (np.where(data["abs_minbatch_slices2"] > -998, ((np.cos(((-((np.sin((data["abs_minbatch_msignal"])))))))) - (data["abs_maxbatch_msignal"])), (((data["abs_minbatch_slices2"]) + (data["signal_shift_-1_msignal"]))/2.0) ))))))) +

                            0.050000*np.tanh((((((((data["rangebatch_slices2_msignal"]) > (data["rangebatch_slices2_msignal"]))*1.)) - (np.where((((data["minbatch_msignal"]) + (np.where((-((((data["signal_shift_+1_msignal"]) * (data["abs_maxbatch"]))))) <= -998, np.cos((data["maxtominbatch_slices2"])), data["medianbatch_slices2_msignal"] )))/2.0) <= -998, data["signal_shift_+1_msignal"], data["mean_abs_chgbatch_slices2"] )))) - (np.sin((((data["medianbatch_slices2_msignal"]) * 2.0)))))) +

                            0.050000*np.tanh(((np.cos((np.where(((np.cos((np.where(np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) * ((-((np.tanh((data["signal_shift_-1_msignal"]))))))))) > -998, data["abs_maxbatch_msignal"], np.cos((np.cos((np.cos((np.where(data["medianbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], data["abs_maxbatch_msignal"] ))))))) )))) * 2.0) > -998, data["abs_maxbatch_msignal"], data["abs_maxbatch_slices2_msignal"] )))) * 2.0)) +

                            0.050000*np.tanh(((data["minbatch"]) * (((((np.where(data["signal_shift_+1_msignal"] <= -998, data["signal_shift_-1"], data["signal_shift_+1_msignal"] )) * 2.0)) * 2.0)))) +

                            0.050000*np.tanh(((np.maximum(((((data["signal_shift_-1_msignal"]) * (np.minimum(((data["abs_maxbatch_slices2"])), ((np.cos((((data["maxtominbatch_msignal"]) + (data["abs_maxbatch_slices2"]))))))))))), ((data["mean_abs_chgbatch_msignal"])))) * (np.where(data["minbatch"] <= -998, data["stdbatch_msignal"], ((data["signal_shift_-1_msignal"]) * (data["minbatch"])) )))) +

                            0.050000*np.tanh(np.minimum(((((((np.cos((data["abs_maxbatch_slices2_msignal"]))) * 2.0)) * 2.0))), ((data["abs_minbatch_msignal"])))) +

                            0.050000*np.tanh((-((((np.maximum(((data["signal_shift_+1_msignal"])), ((np.minimum(((data["stdbatch_slices2"])), ((data["minbatch"]))))))) + (data["signal_shift_-1_msignal"])))))) +

                            0.050000*np.tanh(np.where(np.sin((data["medianbatch_slices2_msignal"])) <= -998, (((3.0)) + (data["mean_abs_chgbatch_slices2_msignal"])), ((data["rangebatch_msignal"]) * (np.cos((((data["abs_avgbatch_slices2"]) + (np.maximum(((data["stdbatch_msignal"])), ((((np.sin((data["abs_minbatch_slices2_msignal"]))) - (np.sin((data["rangebatch_msignal"]))))))))))))) )) +

                            0.050000*np.tanh((-((((np.maximum((((-((((data["maxbatch_msignal"]) * (np.maximum(((np.cos((data["signal_shift_+1"])))), ((np.where(data["signal_shift_-1"] <= -998, data["abs_avgbatch_slices2"], data["maxbatch_slices2_msignal"] ))))))))))), ((data["abs_avgbatch_slices2"])))) * (np.maximum(((data["signal_shift_-1"])), ((np.maximum(((np.cos((data["signal_shift_-1"])))), ((data["signal_shift_+1"])))))))))))) +

                            0.050000*np.tanh(((((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) * (data["signal"]))) - (np.cos((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))))))) +

                            0.050000*np.tanh(((np.where(((np.where(np.tanh((data["medianbatch_slices2"])) > -998, data["medianbatch_slices2_msignal"], data["maxbatch_msignal"] )) * (data["abs_maxbatch_slices2_msignal"])) <= -998, np.sin(((((data["abs_avgbatch_msignal"]) + ((((-((data["medianbatch_msignal"])))) * (data["minbatch_slices2_msignal"]))))/2.0))), (13.16489601135253906) )) * (np.cos((np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((data["signal_shift_+1"])))))))) +

                            0.050000*np.tanh(((data["signal_shift_+1_msignal"]) * ((((-((np.sin(((-(((-(((-(((-((np.tanh(((-((((data["maxtominbatch"]) - (data["signal_shift_+1_msignal"]))))))))))))))))))))))))) - (((np.maximum(((data["maxtominbatch_slices2_msignal"])), (((-((data["minbatch"]))))))) * 2.0)))))) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) * (np.minimum(((np.sin((data["abs_avgbatch_slices2_msignal"])))), (((((data["signal_shift_-1"]) + ((((np.where(data["maxbatch_msignal"] <= -998, (((np.sin((data["abs_avgbatch_slices2_msignal"]))) + (np.sin((data["abs_avgbatch_slices2_msignal"]))))/2.0), data["maxtominbatch_msignal"] )) + (data["abs_avgbatch_slices2_msignal"]))/2.0)))/2.0))))))) +

                            0.050000*np.tanh(((((data["signal_shift_+1"]) * (np.where(data["abs_minbatch_slices2"] > -998, data["signal"], ((data["rangebatch_slices2_msignal"]) * (data["abs_maxbatch"])) )))) - (np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2"], data["stdbatch_slices2"] )))) +

                            0.050000*np.tanh(np.where((((np.tanh((data["abs_maxbatch_slices2_msignal"]))) <= ((((np.cos((np.sin((np.maximum(((data["maxtominbatch_msignal"])), ((np.sin((((data["stdbatch_slices2"]) * 2.0))))))))))) <= (data["maxtominbatch_msignal"]))*1.)))*1.) > -998, np.cos((np.sin(((-((data["abs_avgbatch_msignal"]))))))), data["abs_maxbatch"] )) +

                            0.050000*np.tanh(np.maximum((((7.75069904327392578))), ((np.where(data["minbatch_msignal"] <= -998, data["maxbatch_slices2_msignal"], np.maximum(((data["stdbatch_msignal"])), ((data["maxbatch_msignal"]))) ))))) +

                            0.050000*np.tanh(((data["maxbatch_msignal"]) - (((data["abs_maxbatch"]) * (np.where(data["minbatch_slices2"] > -998, data["stdbatch_slices2"], data["maxbatch_msignal"] )))))) +

                            0.050000*np.tanh(np.cos((((np.minimum(((((((data["rangebatch_msignal"]) + ((((np.sin((data["abs_avgbatch_msignal"]))) > (data["maxtominbatch"]))*1.)))) * ((((data["maxtominbatch"]) + (data["abs_avgbatch_msignal"]))/2.0))))), ((np.cos((((np.sin((((data["abs_avgbatch_msignal"]) - (data["stdbatch_slices2_msignal"]))))) + (data["mean_abs_chgbatch_slices2_msignal"])))))))) + (data["abs_maxbatch_msignal"]))))) +

                            0.050000*np.tanh(np.cos((np.where(data["abs_maxbatch_msignal"] > -998, np.where(((data["abs_maxbatch_msignal"]) + (data["meanbatch_msignal"])) <= -998, ((((data["abs_maxbatch_msignal"]) * (data["abs_minbatch_slices2_msignal"]))) / 2.0), data["abs_maxbatch_msignal"] ), data["medianbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(np.maximum(((((np.cos((data["abs_maxbatch_slices2_msignal"]))) * 2.0))), ((np.minimum(((data["maxtominbatch_slices2"])), ((np.where(np.minimum(((data["maxtominbatch_slices2"])), ((data["maxtominbatch_slices2"]))) <= -998, data["abs_maxbatch_slices2_msignal"], (((np.where(((np.cos((data["abs_maxbatch_slices2_msignal"]))) * 2.0) <= -998, ((data["signal_shift_+1"]) - (data["signal"])), data["mean_abs_chgbatch_slices2"] )) > (data["stdbatch_msignal"]))*1.) )))))))) +

                            0.050000*np.tanh(np.cos((np.maximum(((data["abs_maxbatch_msignal"])), ((data["abs_maxbatch_msignal"])))))) +

                            0.050000*np.tanh(np.minimum(((data["maxtominbatch_msignal"])), ((((data["stdbatch_slices2"]) - (((np.where(data["maxbatch_slices2"] > -998, data["signal_shift_-1"], ((np.where((((data["signal_shift_-1"]) + ((((data["signal_shift_-1"]) + (((data["signal_shift_+1_msignal"]) / 2.0)))/2.0)))/2.0) > -998, data["signal_shift_-1"], np.where(data["stdbatch_slices2"] <= -998, ((data["maxtominbatch_slices2_msignal"]) - (data["stdbatch_slices2"])), data["signal_shift_+1"] ) )) * 2.0) )) * 2.0))))))) +

                            0.050000*np.tanh((((((data["abs_maxbatch"]) <= ((-((data["maxtominbatch"])))))*1.)) - ((((((data["signal"]) > (np.cos((np.minimum(((data["abs_maxbatch_msignal"])), ((data["stdbatch_msignal"])))))))*1.)) - (np.sin((((data["stdbatch_msignal"]) * 2.0)))))))) +

                            0.050000*np.tanh(np.sin((((data["abs_maxbatch_msignal"]) + (np.maximum((((((data["stdbatch_msignal"]) <= (np.cos((data["abs_avgbatch_slices2_msignal"]))))*1.))), (((((np.sin((((data["maxtominbatch"]) * 2.0)))) <= (data["abs_minbatch_slices2"]))*1.))))))))) +

                            0.050000*np.tanh(((np.where(np.where(data["maxtominbatch"] <= -998, data["abs_maxbatch_msignal"], ((np.where((2.32152748107910156) > -998, ((data["meanbatch_msignal"]) * (np.tanh((data["maxbatch_msignal"])))), np.tanh((data["meanbatch_slices2"])) )) * (data["medianbatch_slices2_msignal"])) ) > -998, ((np.sin((data["meanbatch_msignal"]))) * (data["meanbatch_msignal"])), ((((((data["abs_avgbatch_slices2_msignal"]) <= (data["abs_maxbatch_slices2_msignal"]))*1.)) <= (data["maxtominbatch"]))*1.) )) * 2.0)) +

                            0.050000*np.tanh(np.where(data["abs_avgbatch_msignal"] > -998, data["maxtominbatch_slices2"], ((data["minbatch_msignal"]) - (((data["minbatch"]) * (data["abs_avgbatch_msignal"])))) )) +

                            0.050000*np.tanh(np.where(data["signal"] <= -998, data["stdbatch_msignal"], (-((((data["rangebatch_msignal"]) * (((data["rangebatch_msignal"]) * (data["signal_shift_-1_msignal"]))))))) )) +

                            0.050000*np.tanh(np.where(data["minbatch_slices2_msignal"] <= -998, data["abs_maxbatch_msignal"], ((((np.cos((data["stdbatch_slices2"]))) * 2.0)) * 2.0) )) +

                            0.050000*np.tanh((-((np.where(data["stdbatch_slices2_msignal"] <= -998, np.cos((np.where(data["stdbatch_slices2_msignal"] > -998, data["stdbatch_slices2_msignal"], ((np.where(data["stdbatch_slices2_msignal"] > -998, data["maxbatch_msignal"], data["rangebatch_slices2_msignal"] )) * 2.0) ))), ((np.cos((np.where(data["maxtominbatch_msignal"] > -998, np.where(data["stdbatch_slices2_msignal"] > -998, data["stdbatch_slices2_msignal"], np.sin((((data["maxbatch_slices2_msignal"]) / 2.0))) ), data["maxtominbatch_msignal"] )))) * 2.0) ))))) +

                            0.050000*np.tanh((((((10.57091808319091797)) - (data["signal_shift_+1"]))) * (np.sin(((((((10.57091808319091797)) - (data["signal_shift_+1"]))) * 2.0)))))) +

                            0.050000*np.tanh(((np.sin((((np.where(data["abs_avgbatch_slices2"] > -998, data["stdbatch_slices2_msignal"], ((data["stdbatch_msignal"]) * (((((((np.where(((((np.sin((data["stdbatch_slices2_msignal"]))) * 2.0)) * (data["minbatch_msignal"])) > -998, ((data["medianbatch_slices2"]) / 2.0), ((((data["meanbatch_slices2"]) * (data["stdbatch_slices2"]))) * 2.0) )) * 2.0)) * 2.0)) * 2.0))) )) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(((np.cos((data["abs_maxbatch_msignal"]))) * (np.where((-((((((data["abs_maxbatch_msignal"]) * (np.maximum(((((np.cos((data["abs_avgbatch_msignal"]))) * (data["mean_abs_chgbatch_slices2_msignal"])))), ((data["meanbatch_slices2_msignal"])))))) * (np.maximum(((data["meanbatch_slices2_msignal"])), ((data["medianbatch_slices2_msignal"])))))))) <= -998, data["abs_avgbatch_slices2_msignal"], ((data["rangebatch_slices2_msignal"]) - (np.tanh((data["rangebatch_slices2"])))) )))) +

                            0.050000*np.tanh((((13.89447116851806641)) * (np.cos((((np.minimum(((data["signal_shift_+1"])), ((((((data["stdbatch_slices2"]) * (data["mean_abs_chgbatch_slices2"]))) * (((data["maxtominbatch_slices2_msignal"]) + (np.minimum(((((((data["abs_minbatch_msignal"]) * 2.0)) * 2.0))), (((3.0)))))))))))) * 2.0)))))) +

                            0.050000*np.tanh((-((((np.where(np.where(data["maxtominbatch_slices2_msignal"] <= -998, np.cos((data["signal_shift_-1_msignal"])), data["meanbatch_slices2_msignal"] ) <= -998, (((data["maxtominbatch"]) > ((5.58624172210693359)))*1.), data["signal_shift_-1_msignal"] )) * 2.0))))) +

                            0.050000*np.tanh(((np.where(data["mean_abs_chgbatch_slices2"] <= -998, np.where(data["medianbatch_slices2_msignal"] <= -998, np.where(data["signal"] > -998, data["abs_minbatch_msignal"], data["maxtominbatch_msignal"] ), (-((np.cos((data["stdbatch_slices2_msignal"]))))) ), (-((((data["abs_avgbatch_slices2_msignal"]) * (np.cos((data["stdbatch_slices2_msignal"]))))))) )) * 2.0)) +

                            0.050000*np.tanh(((np.where(((((data["abs_maxbatch_slices2_msignal"]) * (data["signal_shift_-1_msignal"]))) * (data["minbatch"])) > -998, data["minbatch"], ((data["minbatch_slices2_msignal"]) * (np.where(((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0) <= -998, data["minbatch_slices2_msignal"], ((data["minbatch"]) * (data["minbatch"])) ))) )) * (data["signal_shift_-1_msignal"]))) +

                            0.050000*np.tanh(((np.where(((data["medianbatch_msignal"]) * (data["signal"])) > -998, np.where(np.cos((((np.where(data["medianbatch_msignal"] > -998, np.cos((data["abs_maxbatch_slices2_msignal"])), np.cos(((5.81919717788696289))) )) * (data["abs_maxbatch_slices2_msignal"])))) > -998, np.cos((data["abs_maxbatch_slices2_msignal"])), data["stdbatch_msignal"] ), data["maxtominbatch"] )) * 2.0)) +

                            0.050000*np.tanh(np.sin((((data["stdbatch_slices2"]) + (np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((np.sin((((np.sin((((data["medianbatch_slices2"]) + (np.maximum(((data["rangebatch_msignal"])), ((np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((np.maximum(((data["abs_maxbatch_slices2_msignal"])), (((((data["stdbatch_slices2"]) > (data["maxtominbatch_slices2"]))*1.))))))))))))))) + (np.cos((data["signal_shift_+1"])))))))))))))) +

                            0.050000*np.tanh((-((np.where((((data["abs_avgbatch_msignal"]) + (((((data["stdbatch_msignal"]) + ((-((data["abs_minbatch_msignal"])))))) * 2.0)))/2.0) <= -998, ((data["abs_maxbatch_msignal"]) * (data["abs_avgbatch_slices2_msignal"])), data["signal_shift_+1"] ))))) +

                            0.050000*np.tanh(((np.maximum(((data["medianbatch_slices2_msignal"])), ((data["mean_abs_chgbatch_slices2_msignal"])))) * (np.where(np.where(np.where(data["maxbatch_msignal"] > -998, np.sin((data["minbatch_slices2"])), data["medianbatch_slices2_msignal"] ) > -998, data["rangebatch_slices2_msignal"], data["maxbatch_msignal"] ) > -998, np.sin((np.maximum(((data["abs_avgbatch_slices2"])), ((data["medianbatch_slices2_msignal"]))))), np.maximum(((data["abs_avgbatch_slices2"])), ((data["medianbatch_slices2_msignal"]))) )))) +

                            0.050000*np.tanh(np.cos((np.where(data["abs_minbatch_slices2"] > -998, data["abs_maxbatch_msignal"], (((((np.where(((data["medianbatch_slices2"]) + (data["maxbatch_slices2"])) > -998, data["maxbatch_slices2"], data["rangebatch_msignal"] )) + (data["abs_maxbatch_msignal"]))) <= (np.where(data["minbatch_msignal"] > -998, data["medianbatch_slices2"], (-((data["maxbatch_slices2"]))) )))*1.) )))) +

                            0.050000*np.tanh((((data["maxtominbatch_slices2_msignal"]) + (np.where(np.sin((data["abs_maxbatch_slices2_msignal"])) <= -998, ((data["abs_maxbatch_msignal"]) * 2.0), ((data["abs_maxbatch_slices2_msignal"]) * (np.sin((np.maximum(((np.maximum(((data["abs_maxbatch_msignal"])), ((data["abs_maxbatch_slices2"]))))), ((data["stdbatch_msignal"]))))))) )))/2.0)) +

                            0.050000*np.tanh(np.cos((((((data["signal"]) * 2.0)) - (np.cos((np.tanh(((((((((data["rangebatch_slices2_msignal"]) * 2.0)) + (np.sin((data["signal"]))))/2.0)) / 2.0)))))))))) +

                            0.050000*np.tanh(((((np.sin((data["abs_avgbatch_slices2"]))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((((np.maximum(((data["abs_minbatch_msignal"])), ((((data["abs_maxbatch_msignal"]) * ((((data["maxtominbatch_slices2_msignal"]) + (np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0)))))/2.0))))))) * (np.cos((data["abs_maxbatch_msignal"]))))) * 2.0)) +

                            0.050000*np.tanh(((np.cos((((data["mean_abs_chgbatch_msignal"]) * 2.0)))) + (((np.cos((((data["meanbatch_slices2"]) * 2.0)))) + (np.where(np.cos((((np.cos((((data["signal"]) * 2.0)))) * 2.0))) <= -998, np.sin((data["maxbatch_slices2"])), ((((data["meanbatch_msignal"]) * (np.sin((np.cos((((data["mean_abs_chgbatch_msignal"]) * 2.0)))))))) / 2.0) )))))) +

                            0.050000*np.tanh(np.where(data["stdbatch_slices2"] > -998, np.where(((np.cos((data["stdbatch_slices2"]))) / 2.0) <= -998, np.where((((data["maxtominbatch_msignal"]) > (data["maxtominbatch_msignal"]))*1.) > -998, data["maxbatch_slices2_msignal"], data["signal_shift_-1"] ), ((data["mean_abs_chgbatch_msignal"]) * (np.where(data["abs_maxbatch_slices2_msignal"] > -998, np.cos((data["stdbatch_slices2"])), data["maxtominbatch_slices2"] ))) ), data["signal_shift_-1"] )) +

                            0.050000*np.tanh(np.tanh((np.cos((np.where((10.32619285583496094) > -998, data["abs_maxbatch_slices2_msignal"], (((np.cos((np.cos((data["abs_maxbatch_slices2_msignal"]))))) <= (np.tanh((np.sin((data["stdbatch_slices2_msignal"]))))))*1.) )))))) +

                            0.050000*np.tanh((((((8.21495723724365234)) + ((-((data["maxbatch_slices2_msignal"])))))) * (data["maxbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((data["signal_shift_-1_msignal"]) * (((data["minbatch_slices2"]) - (np.cos((((data["signal_shift_-1_msignal"]) * ((((((((np.cos((data["signal_shift_-1_msignal"]))) <= (data["signal_shift_-1_msignal"]))*1.)) * (data["rangebatch_slices2"]))) - ((-((((np.tanh((data["signal_shift_-1_msignal"]))) / 2.0))))))))))))))) +

                            0.050000*np.tanh(((((data["meanbatch_slices2_msignal"]) * (((np.sin((data["meanbatch_slices2_msignal"]))) * (np.maximum(((data["abs_maxbatch_msignal"])), ((data["maxtominbatch_msignal"])))))))) * (np.maximum(((data["abs_maxbatch_msignal"])), ((data["maxtominbatch_msignal"])))))) +

                            0.050000*np.tanh(((np.cos((np.minimum(((((data["signal_shift_-1"]) * 2.0))), ((np.where(((data["signal_shift_-1"]) * 2.0) > -998, ((data["rangebatch_slices2_msignal"]) - (data["abs_maxbatch_msignal"])), np.sin((data["signal_shift_-1"])) ))))))) * 2.0)) +

                            0.050000*np.tanh(np.cos((((np.where(data["maxtominbatch_slices2_msignal"] <= -998, ((((data["medianbatch_msignal"]) / 2.0)) - (data["mean_abs_chgbatch_slices2_msignal"])), np.maximum(((data["signal"])), ((data["mean_abs_chgbatch_msignal"]))) )) * 2.0)))) +

                            0.050000*np.tanh(np.where(np.where(((data["stdbatch_slices2_msignal"]) * 2.0) > -998, data["maxtominbatch_slices2_msignal"], data["maxtominbatch_slices2_msignal"] ) > -998, data["maxtominbatch_slices2_msignal"], (((data["minbatch_msignal"]) + (np.tanh(((((((-((data["rangebatch_slices2"])))) - (((data["abs_minbatch_slices2_msignal"]) * (data["abs_maxbatch_msignal"]))))) * (((data["abs_maxbatch"]) * 2.0)))))))/2.0) )) +

                            0.050000*np.tanh(np.sin(((-((np.maximum(((data["abs_avgbatch_slices2_msignal"])), (((-((((np.maximum(((np.maximum(((data["minbatch_msignal"])), ((np.tanh((data["maxtominbatch_msignal"]))))))), ((data["minbatch_slices2"])))) * (np.maximum(((data["abs_avgbatch_slices2_msignal"])), ((np.sin((data["abs_minbatch_msignal"]))))))))))))))))))) +

                            0.050000*np.tanh((-((np.where(np.maximum(((np.minimum(((data["signal"])), ((data["abs_avgbatch_slices2_msignal"]))))), ((((data["abs_avgbatch_slices2_msignal"]) * 2.0)))) <= -998, np.maximum(((np.maximum(((data["abs_avgbatch_msignal"])), ((np.minimum(((data["signal_shift_-1_msignal"])), ((data["meanbatch_slices2_msignal"])))))))), ((data["minbatch_msignal"]))), ((data["mean_abs_chgbatch_msignal"]) * (np.cos((data["stdbatch_slices2_msignal"])))) ))))) +

                            0.050000*np.tanh(np.where(data["medianbatch_msignal"] > -998, np.cos((data["abs_maxbatch_msignal"])), np.maximum(((data["medianbatch_slices2_msignal"])), ((((data["mean_abs_chgbatch_slices2_msignal"]) - (((data["stdbatch_slices2_msignal"]) * (np.maximum(((data["minbatch_slices2"])), ((data["abs_avgbatch_msignal"])))))))))) )) +

                            0.050000*np.tanh(((np.sin(((-((np.where((((data["meanbatch_slices2_msignal"]) > (data["abs_maxbatch_msignal"]))*1.) <= -998, (((data["signal_shift_-1"]) <= (data["maxbatch_slices2"]))*1.), np.where(((data["signal"]) + (data["stdbatch_slices2_msignal"])) <= -998, ((data["abs_minbatch_slices2_msignal"]) * 2.0), data["signal_shift_-1"] ) ))))))) - (((((data["medianbatch_slices2"]) * 2.0)) * 2.0)))) +

                            0.050000*np.tanh(np.sin((((data["stdbatch_slices2_msignal"]) * 2.0)))) +

                            0.050000*np.tanh(((((((((data["rangebatch_msignal"]) + (data["mean_abs_chgbatch_msignal"]))/2.0)) - (((data["rangebatch_msignal"]) * (data["signal_shift_+1"]))))) + ((-(((13.75664520263671875))))))/2.0)) +

                            0.050000*np.tanh(data["abs_avgbatch_msignal"]) +

                            0.050000*np.tanh(((data["abs_maxbatch_msignal"]) * (np.where(((np.cos((data["stdbatch_slices2"]))) + ((6.0))) > -998, np.cos((data["stdbatch_slices2"])), np.tanh((data["maxtominbatch_msignal"])) )))) +

                            0.050000*np.tanh(((data["rangebatch_msignal"]) * (((((((np.cos((((data["signal_shift_+1"]) + ((-((data["abs_maxbatch_slices2_msignal"])))))))) - (data["rangebatch_msignal"]))) * (np.cos((((data["signal_shift_+1"]) + ((-((data["abs_maxbatch_slices2_msignal"])))))))))) - (np.cos((np.cos((np.cos((data["medianbatch_slices2_msignal"]))))))))))))  

    

    def GP_class_1(self,data):

        return self.Output( -1.623856 +

                            0.050000*np.tanh(((((np.minimum(((data["abs_minbatch_slices2_msignal"])), (((-((np.cos((((data["maxtominbatch_slices2_msignal"]) + (np.where(data["abs_minbatch_msignal"] > -998, data["maxtominbatch_slices2_msignal"], np.where(data["meanbatch_slices2_msignal"] > -998, np.where(data["abs_maxbatch_slices2"] > -998, data["maxtominbatch_slices2_msignal"], ((data["abs_minbatch_slices2"]) + (data["maxtominbatch_msignal"])) ), np.cos(((-((data["maxtominbatch_slices2_msignal"]))))) ) )))))))))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.where(data["abs_minbatch_msignal"] > -998, np.where(((np.where(data["mean_abs_chgbatch_slices2"] > -998, data["maxtominbatch_slices2_msignal"], data["maxtominbatch_slices2_msignal"] )) - (np.tanh((np.where(data["maxbatch_msignal"] <= -998, data["medianbatch_slices2"], (-((data["meanbatch_msignal"]))) ))))) <= -998, data["rangebatch_slices2_msignal"], (((10.0)) * (data["maxtominbatch_slices2_msignal"])) ), data["maxtominbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(((data["maxtominbatch_slices2_msignal"]) + (np.where((((data["maxtominbatch_slices2_msignal"]) > (np.tanh((data["abs_minbatch_msignal"]))))*1.) > -998, data["maxtominbatch_slices2_msignal"], np.sin((((data["signal_shift_-1"]) + (((data["maxtominbatch_slices2_msignal"]) * 2.0))))) )))) +

                            0.050000*np.tanh(((((data["maxtominbatch_msignal"]) - ((-((data["maxtominbatch_msignal"])))))) - (((((data["maxtominbatch_slices2_msignal"]) - (np.tanh((np.sin((data["maxtominbatch_msignal"]))))))) * (data["maxtominbatch_msignal"]))))) +

                            0.050000*np.tanh(((data["maxtominbatch_msignal"]) + (np.where((4.0) > -998, data["maxtominbatch_msignal"], np.where((((data["minbatch_msignal"]) > (data["abs_maxbatch_slices2_msignal"]))*1.) <= -998, (((data["minbatch_slices2_msignal"]) <= (data["maxtominbatch_msignal"]))*1.), data["maxtominbatch_msignal"] ) )))) +

                            0.050000*np.tanh(((((np.where(data["abs_minbatch_slices2_msignal"] > -998, ((((data["abs_minbatch_slices2_msignal"]) * (((data["maxbatch_slices2"]) - (np.sin((data["abs_maxbatch_slices2_msignal"]))))))) * 2.0), np.minimum(((np.tanh((data["abs_minbatch_msignal"])))), (((((((((data["abs_minbatch_slices2_msignal"]) > (data["abs_minbatch_msignal"]))*1.)) * 2.0)) - (data["maxbatch_slices2"]))))) )) - (data["maxbatch_slices2"]))) * 2.0)) +

                            0.050000*np.tanh(((((((np.minimum(((np.sin((np.maximum(((data["maxtominbatch_slices2_msignal"])), ((data["abs_minbatch_slices2_msignal"]))))))), ((((((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((data["maxtominbatch_msignal"])))) + ((((data["maxtominbatch_msignal"]) + ((-((np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, data["mean_abs_chgbatch_slices2_msignal"], data["mean_abs_chgbatch_slices2_msignal"] ))))))/2.0)))) * 2.0))))) * 2.0)) * 2.0)) * 2.0)) +

                            0.050000*np.tanh((((10.30580806732177734)) * (np.minimum(((data["maxtominbatch_slices2_msignal"])), ((np.where(np.where(data["abs_minbatch_msignal"] > -998, np.sin((np.maximum(((data["abs_minbatch_msignal"])), ((np.maximum(((data["abs_minbatch_slices2_msignal"])), ((np.sin((data["mean_abs_chgbatch_slices2_msignal"])))))))))), np.cos((data["signal_shift_+1"])) ) > -998, np.sin((np.maximum(((data["abs_minbatch_msignal"])), ((np.sin((data["abs_minbatch_slices2_msignal"]))))))), data["maxtominbatch_msignal"] ))))))) +

                            0.050000*np.tanh(((((((((data["maxtominbatch_msignal"]) * ((((((data["meanbatch_slices2"]) + (np.sin((data["stdbatch_msignal"]))))/2.0)) + (((((((((((data["minbatch_slices2_msignal"]) + (data["meanbatch_slices2"]))) * 2.0)) + (((np.sin((data["stdbatch_msignal"]))) * 2.0)))/2.0)) <= (np.sin((((data["stdbatch_msignal"]) * 2.0)))))*1.)))))) * 2.0)) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["meanbatch_slices2"], ((((((np.minimum(((np.sin((np.maximum((((-((np.cos((data["meanbatch_slices2_msignal"]))))))), ((data["abs_minbatch_slices2_msignal"]))))))), ((data["meanbatch_slices2_msignal"])))) + (data["abs_minbatch_msignal"]))) + ((((-((np.cos((((data["abs_minbatch_msignal"]) * 2.0))))))) * (((data["abs_avgbatch_msignal"]) + (data["meanbatch_slices2_msignal"]))))))) * 2.0) )) +

                            0.050000*np.tanh((((((14.95742130279541016)) - (((data["rangebatch_msignal"]) * (np.where(data["abs_minbatch_msignal"] > -998, data["maxtominbatch_slices2_msignal"], data["maxtominbatch_slices2_msignal"] )))))) * (np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["maxtominbatch_slices2_msignal"], data["maxtominbatch_msignal"] )))) +

                            0.050000*np.tanh(((((((((data["maxtominbatch_msignal"]) * 2.0)) * (((data["abs_maxbatch_slices2"]) - (((data["maxtominbatch_msignal"]) * 2.0)))))) + (data["maxtominbatch_msignal"]))) + (data["maxtominbatch_msignal"]))) +

                            0.050000*np.tanh(((np.where(data["maxtominbatch_slices2_msignal"] > -998, np.minimum(((((data["maxtominbatch_msignal"]) + (data["maxtominbatch_msignal"])))), ((np.sin((((data["maxtominbatch_slices2_msignal"]) + (np.tanh((((data["maxtominbatch_slices2_msignal"]) + (((np.minimum(((np.sin((data["maxtominbatch_slices2_msignal"])))), ((np.minimum(((data["mean_abs_chgbatch_msignal"])), ((data["maxtominbatch_msignal"]))))))) + (np.sin((data["maxtominbatch_slices2_msignal"]))))))))))))))), data["maxtominbatch_slices2_msignal"] )) * 2.0)) +

                            0.050000*np.tanh(((data["maxtominbatch_slices2_msignal"]) + (np.where(np.where(data["maxtominbatch_slices2_msignal"] > -998, np.cos((data["maxtominbatch_slices2_msignal"])), data["minbatch_msignal"] ) > -998, data["abs_minbatch_msignal"], np.where(data["medianbatch_slices2"] > -998, data["maxtominbatch_slices2_msignal"], data["maxtominbatch_msignal"] ) )))) +

                            0.050000*np.tanh(((data["rangebatch_slices2"]) * (np.where(((np.sin((data["abs_minbatch_slices2_msignal"]))) * 2.0) <= -998, np.sin((np.where(data["abs_maxbatch_msignal"] <= -998, data["stdbatch_slices2_msignal"], np.minimum(((data["stdbatch_msignal"])), ((((np.sin((np.sin((data["abs_minbatch_slices2_msignal"]))))) * 2.0)))) ))), np.minimum(((data["maxtominbatch_slices2_msignal"])), ((((np.sin((data["abs_minbatch_slices2_msignal"]))) * 2.0)))) )))) +

                            0.050000*np.tanh(np.where(np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((data["rangebatch_slices2"]))) > -998, np.where(data["minbatch"] > -998, data["maxtominbatch_msignal"], np.maximum(((np.where(data["abs_minbatch_slices2_msignal"] > -998, data["maxtominbatch_msignal"], data["rangebatch_msignal"] ))), ((data["maxbatch_msignal"]))) ), np.minimum(((data["maxtominbatch_msignal"])), ((np.maximum(((np.where(data["abs_minbatch_slices2_msignal"] > -998, data["minbatch_slices2"], data["rangebatch_msignal"] ))), ((data["maxbatch_msignal"])))))) )) +

                            0.050000*np.tanh(((np.where((((data["signal"]) + (np.tanh((((data["maxtominbatch_slices2_msignal"]) - (np.where(data["abs_minbatch_msignal"] <= -998, np.cos((data["maxtominbatch_slices2_msignal"])), data["maxtominbatch_msignal"] )))))))/2.0) > -998, data["maxtominbatch_slices2_msignal"], ((((data["maxtominbatch_slices2_msignal"]) * 2.0)) * (data["mean_abs_chgbatch_slices2"])) )) * 2.0)) +

                            0.050000*np.tanh(((data["abs_maxbatch_msignal"]) * (((data["abs_maxbatch_msignal"]) * (np.minimum(((((np.sin((np.where((((((np.sin((np.where(data["abs_maxbatch_msignal"] > -998, data["abs_minbatch_slices2_msignal"], data["abs_minbatch_msignal"] )))) + (data["mean_abs_chgbatch_slices2_msignal"]))) <= (data["signal_shift_-1"]))*1.) > -998, data["abs_minbatch_slices2_msignal"], data["abs_minbatch_msignal"] )))) * 2.0))), ((data["abs_maxbatch_msignal"])))))))) +

                            0.050000*np.tanh(((np.sin((data["medianbatch_msignal"]))) + (np.minimum(((data["maxtominbatch_slices2_msignal"])), ((((((np.where(data["maxtominbatch_slices2_msignal"] <= -998, np.maximum(((data["abs_minbatch_slices2_msignal"])), ((data["maxtominbatch_slices2_msignal"]))), ((np.where((((data["maxtominbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"]))/2.0) > -998, data["abs_minbatch_msignal"], data["abs_minbatch_slices2_msignal"] )) + (data["rangebatch_slices2"])) )) * (np.sin((data["abs_minbatch_slices2_msignal"]))))) * 2.0))))))) +

                            0.050000*np.tanh(((np.minimum(((((np.where(data["maxtominbatch_slices2_msignal"] > -998, data["abs_minbatch_slices2_msignal"], data["abs_minbatch_slices2_msignal"] )) + (((((data["abs_minbatch_slices2_msignal"]) + (data["abs_minbatch_slices2_msignal"]))) * (np.cos((data["maxtominbatch_slices2_msignal"])))))))), ((np.tanh((((np.minimum(((data["abs_minbatch_slices2_msignal"])), ((data["abs_avgbatch_msignal"])))) + (((data["abs_minbatch_slices2_msignal"]) + (data["minbatch_slices2_msignal"])))))))))) * 2.0)) +

                            0.050000*np.tanh(((np.sin((data["abs_avgbatch_slices2_msignal"]))) + (((data["minbatch_slices2_msignal"]) + (((np.sin((data["abs_minbatch_msignal"]))) * (np.maximum(((data["maxtominbatch_msignal"])), ((((data["abs_maxbatch_msignal"]) * 2.0))))))))))) +

                            0.050000*np.tanh(((((np.sin((np.where(((data["maxtominbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"])) <= -998, data["signal_shift_+1_msignal"], data["maxtominbatch_slices2_msignal"] )))) * (data["rangebatch_slices2_msignal"]))) - ((((np.sin((data["abs_avgbatch_msignal"]))) + (((((data["maxtominbatch_slices2_msignal"]) * (data["maxtominbatch_slices2_msignal"]))) * 2.0)))/2.0)))) +

                            0.050000*np.tanh(((data["maxtominbatch_msignal"]) + (np.where(data["abs_avgbatch_msignal"] > -998, ((data["maxtominbatch_msignal"]) + (data["maxtominbatch_slices2_msignal"])), ((data["rangebatch_msignal"]) * (((data["meanbatch_slices2_msignal"]) * (data["signal_shift_+1_msignal"])))) )))) +

                            0.050000*np.tanh(np.where(np.where(data["signal_shift_-1_msignal"] > -998, np.sin((np.minimum(((data["maxtominbatch_msignal"])), ((data["abs_minbatch_slices2_msignal"]))))), data["maxtominbatch"] ) <= -998, np.minimum(((data["maxtominbatch_slices2_msignal"])), ((((data["abs_maxbatch"]) - (data["abs_minbatch_slices2_msignal"]))))), np.minimum(((data["maxtominbatch_msignal"])), ((((data["abs_maxbatch"]) - (data["abs_minbatch_slices2_msignal"]))))) )) +

                            0.050000*np.tanh(np.where(((data["maxtominbatch_msignal"]) + (data["maxtominbatch"])) > -998, data["maxtominbatch_msignal"], data["maxtominbatch_msignal"] )) +

                            0.050000*np.tanh(((((((data["stdbatch_msignal"]) - ((((np.maximum(((data["abs_minbatch_slices2"])), ((np.maximum(((data["stdbatch_msignal"])), (((7.0)))))))) <= (data["mean_abs_chgbatch_slices2_msignal"]))*1.)))) * (data["maxtominbatch_msignal"]))) * ((((((-((np.cos((data["minbatch_msignal"])))))) * (np.where(data["minbatch_msignal"] > -998, data["maxtominbatch_msignal"], ((data["rangebatch_slices2"]) * 2.0) )))) * (data["abs_minbatch_msignal"]))))) +

                            0.050000*np.tanh(((data["maxtominbatch_slices2_msignal"]) + (((data["maxtominbatch_msignal"]) * 2.0)))) +

                            0.050000*np.tanh(((((data["maxtominbatch_msignal"]) * (np.cos((((np.cos((np.where(((np.tanh((data["maxtominbatch_slices2_msignal"]))) - ((-((((data["maxtominbatch_slices2_msignal"]) / 2.0)))))) <= -998, (((data["rangebatch_slices2_msignal"]) + (np.minimum(((((data["mean_abs_chgbatch_msignal"]) / 2.0))), ((data["maxtominbatch_slices2_msignal"])))))/2.0), data["mean_abs_chgbatch_msignal"] )))) * (data["mean_abs_chgbatch_msignal"]))))))) * 2.0)) +

                            0.050000*np.tanh(((((np.where(data["maxtominbatch_msignal"] > -998, data["abs_minbatch_slices2_msignal"], ((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0) )) * 2.0)) - ((((data["maxtominbatch"]) <= (np.where(data["maxtominbatch_msignal"] > -998, ((np.where(data["maxtominbatch_msignal"] > -998, data["abs_minbatch_slices2_msignal"], ((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0) )) - (data["maxtominbatch_msignal"])), data["abs_minbatch_slices2_msignal"] )))*1.)))) +

                            0.050000*np.tanh(np.minimum(((((((np.sin((np.where(((np.sin(((-((data["rangebatch_msignal"])))))) * 2.0) > -998, data["abs_minbatch_msignal"], data["mean_abs_chgbatch_msignal"] )))) * 2.0)) * 2.0))), ((data["maxtominbatch_slices2_msignal"])))) +

                            0.050000*np.tanh(np.where(data["abs_maxbatch_slices2_msignal"] <= -998, data["maxtominbatch_msignal"], np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, ((data["minbatch_slices2"]) / 2.0), ((((((data["abs_maxbatch_msignal"]) - (data["mean_abs_chgbatch_msignal"]))) + ((((11.96386051177978516)) + (data["maxbatch_slices2"]))))) * (np.maximum(((data["minbatch_slices2"])), ((np.minimum(((((np.sin((data["maxtominbatch_msignal"]))) + (data["maxbatch_slices2"])))), ((data["maxtominbatch_msignal"])))))))) ) )) +

                            0.050000*np.tanh(((np.where(data["abs_minbatch_msignal"] > -998, np.minimum(((np.minimum(((((data["stdbatch_slices2"]) - (((np.cos((((np.maximum(((data["mean_abs_chgbatch_msignal"])), ((data["maxtominbatch_slices2_msignal"])))) * 2.0)))) * 2.0))))), ((((data["abs_minbatch_msignal"]) * (data["maxtominbatch_slices2_msignal"]))))))), ((data["abs_minbatch_msignal"]))), ((np.cos((((np.maximum(((data["maxtominbatch_slices2_msignal"])), ((data["maxtominbatch_slices2_msignal"])))) * 2.0)))) * 2.0) )) * 2.0)) +

                            0.050000*np.tanh(((((data["maxtominbatch_msignal"]) * (((data["rangebatch_slices2"]) + (np.where(data["maxtominbatch_slices2_msignal"] <= -998, ((data["signal_shift_+1"]) * 2.0), data["signal_shift_+1"] )))))) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((((np.minimum(((data["maxtominbatch_msignal"])), ((np.tanh((np.where(data["rangebatch_slices2_msignal"] <= -998, ((data["maxbatch_slices2"]) - (data["meanbatch_slices2_msignal"])), np.sin((data["abs_minbatch_msignal"])) ))))))) * 2.0))), (((((data["abs_maxbatch_slices2_msignal"]) + (((data["maxbatch_slices2"]) - (data["meanbatch_slices2_msignal"]))))/2.0))))) +

                            0.050000*np.tanh(np.where((((((data["maxtominbatch"]) * 2.0)) <= (((data["rangebatch_msignal"]) * 2.0)))*1.) > -998, ((((data["maxtominbatch_msignal"]) * 2.0)) * 2.0), data["maxtominbatch_msignal"] )) +

                            0.050000*np.tanh(((data["abs_minbatch_msignal"]) + (np.maximum(((data["maxtominbatch_msignal"])), ((data["abs_minbatch_msignal"])))))) +

                            0.050000*np.tanh(((((data["maxtominbatch_slices2_msignal"]) * (np.maximum(((np.where(data["abs_minbatch_slices2"] > -998, data["abs_maxbatch_msignal"], ((data["maxtominbatch"]) + (data["maxtominbatch_slices2_msignal"])) ))), ((np.where(data["maxtominbatch_slices2_msignal"] <= -998, np.tanh((data["abs_maxbatch_slices2_msignal"])), (((data["abs_avgbatch_slices2"]) > (np.where(np.where(data["maxtominbatch_msignal"] <= -998, data["stdbatch_msignal"], data["abs_avgbatch_msignal"] ) <= -998, data["signal_shift_+1"], data["stdbatch_msignal"] )))*1.) ))))))) * 2.0)) +

                            0.050000*np.tanh(((np.where(np.cos((data["signal"])) > -998, data["maxtominbatch_msignal"], ((np.where(np.where(((data["abs_minbatch_slices2_msignal"]) * 2.0) > -998, data["maxtominbatch_msignal"], ((data["stdbatch_slices2"]) * 2.0) ) <= -998, (-((data["maxtominbatch_msignal"]))), ((data["maxtominbatch_msignal"]) * 2.0) )) * 2.0) )) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((data["abs_minbatch_msignal"])), ((((((np.cos((data["abs_avgbatch_msignal"]))) * (np.minimum(((np.minimum(((((np.cos((data["abs_avgbatch_msignal"]))) * (np.minimum(((data["abs_avgbatch_msignal"])), ((data["signal"]))))))), ((((data["abs_avgbatch_slices2"]) * 2.0)))))), ((data["signal"])))))) * 2.0))))) +

                            0.050000*np.tanh(data["maxtominbatch_slices2_msignal"]) +

                            0.050000*np.tanh(np.where(data["maxtominbatch_msignal"] <= -998, data["maxtominbatch_slices2_msignal"], ((((np.maximum(((np.cos((data["maxtominbatch_msignal"])))), ((data["abs_maxbatch"])))) * 2.0)) * (np.sin((data["abs_minbatch_slices2_msignal"])))) )) +

                            0.050000*np.tanh(np.where(((data["stdbatch_msignal"]) * (np.sin((data["abs_minbatch_slices2_msignal"])))) <= -998, data["maxtominbatch_slices2"], ((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)) + (np.minimum(((data["medianbatch_slices2_msignal"])), ((((data["abs_maxbatch_slices2_msignal"]) * (np.where(np.sin((data["minbatch_msignal"])) > -998, ((data["maxtominbatch_slices2_msignal"]) * (((data["abs_maxbatch_slices2_msignal"]) - (((data["maxtominbatch_slices2_msignal"]) * (data["mean_abs_chgbatch_slices2_msignal"])))))), data["abs_minbatch_msignal"] )))))))) )) +

                            0.050000*np.tanh(((np.sin((data["abs_minbatch_msignal"]))) + (np.where(data["maxtominbatch"] > -998, np.where(data["signal_shift_+1_msignal"] > -998, data["maxtominbatch"], np.where(np.where(data["maxtominbatch"] > -998, data["abs_minbatch_msignal"], data["abs_minbatch_msignal"] ) <= -998, ((np.sin((data["meanbatch_slices2_msignal"]))) - (data["abs_minbatch_slices2_msignal"])), data["maxtominbatch"] ) ), np.cos((data["abs_minbatch_msignal"])) )))) +

                            0.050000*np.tanh(np.where(np.tanh((data["meanbatch_slices2_msignal"])) > -998, np.minimum(((data["abs_minbatch_slices2_msignal"])), ((np.where(data["abs_minbatch_msignal"] > -998, data["maxtominbatch_slices2_msignal"], data["abs_minbatch_slices2_msignal"] )))), np.tanh((np.where(data["meanbatch_msignal"] > -998, data["abs_minbatch_slices2_msignal"], data["abs_minbatch_slices2_msignal"] ))) )) +

                            0.050000*np.tanh(((data["abs_maxbatch_slices2"]) * ((-((np.cos((((data["abs_avgbatch_slices2_msignal"]) + ((-((np.sin((data["abs_avgbatch_slices2_msignal"]))))))))))))))) +

                            0.050000*np.tanh(((np.where(np.minimum(((((data["minbatch_msignal"]) * 2.0))), ((data["signal_shift_-1_msignal"]))) <= -998, (((data["mean_abs_chgbatch_msignal"]) > (np.minimum(((data["abs_minbatch_msignal"])), ((np.tanh((data["minbatch_slices2"])))))))*1.), data["abs_minbatch_slices2_msignal"] )) * ((((-((((data["minbatch_msignal"]) * 2.0))))) - (data["abs_minbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(((np.cos((((np.where(data["rangebatch_msignal"] > -998, data["abs_avgbatch_msignal"], np.where(((((7.0)) <= (data["signal"]))*1.) > -998, data["rangebatch_msignal"], data["maxtominbatch_slices2"] ) )) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(((((data["maxtominbatch_msignal"]) * 2.0)) - (np.cos((data["signal_shift_+1_msignal"]))))) +

                            0.050000*np.tanh(np.where(data["signal_shift_+1_msignal"] > -998, data["maxtominbatch_slices2_msignal"], (((data["abs_minbatch_msignal"]) <= ((((np.tanh((((data["signal_shift_-1_msignal"]) * 2.0)))) <= (data["abs_minbatch_slices2_msignal"]))*1.)))*1.) )) +

                            0.050000*np.tanh(np.sin((np.sin((np.tanh((np.minimum(((((data["maxtominbatch_msignal"]) / 2.0))), ((((((np.minimum(((((((np.sin((data["abs_minbatch_slices2_msignal"]))) * 2.0)) / 2.0))), ((((np.tanh((np.minimum(((np.tanh((((((np.sin((data["abs_minbatch_msignal"]))) * 2.0)) / 2.0))))), ((data["maxtominbatch_msignal"])))))) * 2.0))))) * 2.0)) / 2.0))))))))))) +

                            0.050000*np.tanh(np.where(data["signal"] > -998, np.minimum(((data["rangebatch_msignal"])), ((((np.sin((data["abs_minbatch_slices2_msignal"]))) * 2.0)))), np.tanh((np.where(data["maxtominbatch_msignal"] <= -998, data["stdbatch_slices2_msignal"], data["maxtominbatch"] ))) )) +

                            0.050000*np.tanh(np.where(((data["maxtominbatch_slices2_msignal"]) / 2.0) > -998, ((np.minimum(((data["abs_minbatch_msignal"])), ((np.tanh((np.sin((data["abs_minbatch_msignal"])))))))) + (data["abs_avgbatch_slices2_msignal"])), data["abs_minbatch_msignal"] )) +

                            0.050000*np.tanh(np.minimum(((data["maxtominbatch_slices2_msignal"])), ((data["abs_minbatch_slices2_msignal"])))) +

                            0.050000*np.tanh(((np.where(data["meanbatch_slices2"] <= -998, data["mean_abs_chgbatch_msignal"], data["maxtominbatch_msignal"] )) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((data["abs_minbatch_msignal"])), ((((np.sin((np.where(data["abs_maxbatch_slices2_msignal"] > -998, ((np.minimum(((np.sin((data["abs_minbatch_msignal"])))), ((data["maxtominbatch_slices2_msignal"])))) * 2.0), data["maxtominbatch_slices2_msignal"] )))) - (np.cos((((data["abs_minbatch_msignal"]) * 2.0))))))))) +

                            0.050000*np.tanh(((np.minimum(((np.sin((data["stdbatch_slices2_msignal"])))), ((np.sin((((np.sin((data["abs_minbatch_msignal"]))) * ((((data["mean_abs_chgbatch_slices2"]) + (data["maxtominbatch_msignal"]))/2.0))))))))) * 2.0)) +

                            0.050000*np.tanh(((np.sin(((((((((data["stdbatch_slices2_msignal"]) <= ((((np.sin((((np.sin((data["maxtominbatch_msignal"]))) - (((data["stdbatch_slices2_msignal"]) - (data["abs_avgbatch_msignal"]))))))) <= ((((data["stdbatch_slices2_msignal"]) <= (((np.sin((data["abs_avgbatch_msignal"]))) * 2.0)))*1.)))*1.)))*1.)) - (data["maxtominbatch_msignal"]))) - (data["abs_avgbatch_msignal"]))))) * 2.0)) +

                            0.050000*np.tanh(np.where(((data["maxtominbatch"]) * (data["mean_abs_chgbatch_slices2_msignal"])) > -998, ((np.sin((data["abs_minbatch_slices2_msignal"]))) * (data["abs_avgbatch_slices2_msignal"])), ((((np.where(data["minbatch_slices2_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], ((data["maxtominbatch"]) * (data["abs_maxbatch_msignal"])) )) * (np.sin((data["abs_minbatch_slices2_msignal"]))))) * (data["maxtominbatch_msignal"])) )) +

                            0.050000*np.tanh(np.minimum(((np.maximum(((data["maxtominbatch_msignal"])), ((data["maxtominbatch_slices2"]))))), ((((np.where(data["abs_minbatch_msignal"] > -998, np.where(data["stdbatch_msignal"] > -998, np.sin((data["abs_minbatch_msignal"])), data["maxtominbatch_slices2_msignal"] ), np.where(data["abs_minbatch_msignal"] <= -998, data["maxtominbatch_msignal"], data["maxtominbatch_msignal"] ) )) * 2.0))))) +

                            0.050000*np.tanh(((np.sin((((data["signal_shift_+1"]) - (np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((np.maximum(((((data["signal_shift_+1"]) - (((data["signal_shift_+1"]) - (np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((data["abs_avgbatch_slices2_msignal"]))))))))), ((((data["abs_avgbatch_slices2_msignal"]) * 2.0)))))))))))) * 2.0)) +

                            0.050000*np.tanh(((np.minimum((((((((np.minimum(((np.cos((((data["abs_avgbatch_msignal"]) * 2.0))))), ((np.where(data["medianbatch_slices2"] <= -998, data["signal_shift_-1_msignal"], np.where(data["maxtominbatch_msignal"] > -998, np.sin((data["abs_minbatch_msignal"])), (((np.maximum(((data["maxtominbatch_msignal"])), ((data["maxtominbatch_msignal"])))) + (data["minbatch_slices2_msignal"]))/2.0) ) ))))) * (data["abs_avgbatch_msignal"]))) + (data["medianbatch_slices2"]))/2.0))), ((data["abs_minbatch_msignal"])))) * (data["abs_maxbatch_slices2_msignal"]))) +

                            0.050000*np.tanh((((np.minimum(((data["maxtominbatch_msignal"])), ((((data["maxtominbatch_slices2_msignal"]) / 2.0))))) + (data["maxtominbatch_slices2_msignal"]))/2.0)) +

                            0.050000*np.tanh(((np.where(((np.where(data["maxtominbatch"] > -998, np.cos((((((data["maxtominbatch_slices2_msignal"]) + (data["abs_maxbatch_msignal"]))) + (data["abs_maxbatch_msignal"])))), data["maxbatch_slices2_msignal"] )) * 2.0) > -998, np.cos((((data["maxtominbatch_slices2_msignal"]) + (data["abs_maxbatch_msignal"])))), (((((np.cos((((data["maxtominbatch_msignal"]) + (data["maxtominbatch_slices2_msignal"]))))) * 2.0)) <= (data["abs_maxbatch_msignal"]))*1.) )) * 2.0)) +

                            0.050000*np.tanh(((data["maxtominbatch_msignal"]) + (np.cos((np.sin((((np.sin((((data["rangebatch_msignal"]) / 2.0)))) - (data["abs_minbatch_slices2_msignal"]))))))))) +

                            0.050000*np.tanh(((np.minimum(((data["maxtominbatch_msignal"])), ((np.sin((np.maximum(((np.maximum(((np.tanh((data["maxtominbatch_slices2_msignal"])))), ((data["abs_minbatch_msignal"]))))), ((np.maximum(((np.minimum(((data["mean_abs_chgbatch_msignal"])), ((((data["rangebatch_slices2"]) + (np.where(np.sin((data["maxtominbatch_msignal"])) > -998, np.sin((data["abs_minbatch_msignal"])), ((data["maxtominbatch"]) * 2.0) )))))))), ((data["mean_abs_chgbatch_slices2_msignal"])))))))))))) * 2.0)) +

                            0.050000*np.tanh(np.where(np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["abs_maxbatch_msignal"], ((data["abs_minbatch_msignal"]) * 2.0) ) > -998, data["maxtominbatch_msignal"], data["medianbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(((data["abs_minbatch_slices2_msignal"]) * (np.sin((np.maximum((((((np.where(np.where(data["abs_minbatch_slices2"] > -998, data["maxtominbatch_slices2_msignal"], ((data["abs_minbatch_slices2_msignal"]) + (data["abs_minbatch_slices2_msignal"])) ) > -998, (((((data["abs_minbatch_slices2_msignal"]) / 2.0)) + (((data["maxtominbatch_slices2_msignal"]) * (np.sin((np.maximum(((data["maxtominbatch_slices2_msignal"])), ((data["abs_minbatch_slices2"])))))))))/2.0), data["abs_minbatch_slices2_msignal"] )) > (data["medianbatch_msignal"]))*1.))), ((data["abs_minbatch_slices2_msignal"])))))))) +

                            0.050000*np.tanh(((data["maxtominbatch_msignal"]) * 2.0)) +

                            0.050000*np.tanh(((np.sin((np.minimum((((((-((data["stdbatch_msignal"])))) * 2.0))), ((np.sin((data["maxtominbatch"])))))))) * 2.0)) +

                            0.050000*np.tanh((-((np.cos((np.maximum(((data["abs_avgbatch_slices2_msignal"])), ((data["signal_shift_+1_msignal"]))))))))) +

                            0.050000*np.tanh(np.sin((np.where(np.where(np.sin((data["abs_minbatch_msignal"])) > -998, data["mean_abs_chgbatch_msignal"], data["mean_abs_chgbatch_msignal"] ) > -998, data["abs_minbatch_msignal"], data["mean_abs_chgbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(np.minimum(((data["maxtominbatch_slices2_msignal"])), ((data["maxtominbatch_slices2_msignal"])))) +

                            0.050000*np.tanh(np.minimum(((np.sin((data["maxtominbatch_msignal"])))), ((((np.sin((((data["mean_abs_chgbatch_msignal"]) + (np.where(((data["stdbatch_msignal"]) * (np.sin((((data["mean_abs_chgbatch_msignal"]) + (np.where(((data["mean_abs_chgbatch_msignal"]) * 2.0) > -998, data["abs_minbatch_slices2"], np.cos((data["mean_abs_chgbatch_msignal"])) ))))))) > -998, data["abs_minbatch_slices2"], data["medianbatch_msignal"] )))))) * 2.0))))) +

                            0.050000*np.tanh(np.cos((((np.where(np.cos((np.tanh((np.sin(((((np.tanh((((data["stdbatch_msignal"]) * 2.0)))) + (data["abs_minbatch_msignal"]))/2.0))))))) <= -998, data["minbatch_slices2_msignal"], np.cos((data["signal_shift_-1"])) )) * 2.0)))) +

                            0.050000*np.tanh(((np.sin((np.where((((-((data["rangebatch_slices2_msignal"])))) * 2.0) <= -998, np.where(data["stdbatch_msignal"] <= -998, np.sin((data["abs_avgbatch_slices2_msignal"])), ((data["abs_avgbatch_slices2_msignal"]) * 2.0) ), ((data["abs_avgbatch_slices2_msignal"]) * 2.0) )))) * 2.0)) +

                            0.050000*np.tanh(((np.where(np.cos((data["maxtominbatch_slices2_msignal"])) > -998, (-((np.cos((data["abs_avgbatch_msignal"]))))), (-((((data["stdbatch_slices2_msignal"]) - (data["abs_avgbatch_msignal"]))))) )) * 2.0)) +

                            0.050000*np.tanh(((np.minimum(((np.minimum(((((data["maxtominbatch_slices2_msignal"]) / 2.0))), (((0.0)))))), ((data["stdbatch_slices2"])))) + (((np.sin((data["abs_minbatch_msignal"]))) * (((np.sin((np.maximum(((data["abs_minbatch_msignal"])), ((((data["abs_minbatch_msignal"]) / 2.0))))))) + (((data["abs_minbatch_msignal"]) * (np.tanh((data["abs_minbatch_msignal"]))))))))))) +

                            0.050000*np.tanh(np.minimum(((((data["abs_avgbatch_msignal"]) * 2.0))), ((((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), (((((9.0)) * (np.sin((((np.minimum((((9.0))), ((np.where(data["abs_minbatch_slices2"] <= -998, np.minimum(((data["maxtominbatch_slices2_msignal"])), ((data["abs_avgbatch_msignal"]))), data["abs_avgbatch_slices2_msignal"] ))))) - ((((data["abs_avgbatch_slices2"]) > (data["minbatch_msignal"]))*1.))))))))))) * 2.0))))) +

                            0.050000*np.tanh(((np.minimum(((np.where(data["maxbatch_msignal"] <= -998, data["maxtominbatch_msignal"], data["abs_minbatch_msignal"] ))), ((((data["abs_minbatch_msignal"]) - ((((((data["maxtominbatch_msignal"]) <= (np.where(data["maxtominbatch_msignal"] <= -998, np.minimum(((data["abs_minbatch_msignal"])), ((data["signal_shift_-1"]))), data["mean_abs_chgbatch_msignal"] )))*1.)) * (((data["minbatch_slices2_msignal"]) + (((data["signal_shift_-1"]) * (np.minimum(((data["signal"])), ((data["maxtominbatch_msignal"]))))))))))))))) * 2.0)) +

                            0.050000*np.tanh(np.where(((data["signal"]) * 2.0) <= -998, data["abs_minbatch_slices2_msignal"], np.sin((np.where(np.tanh((data["abs_avgbatch_msignal"])) <= -998, data["stdbatch_slices2_msignal"], np.where(data["minbatch_msignal"] <= -998, (((-((((np.sin((((data["abs_minbatch_slices2_msignal"]) * 2.0)))) + (data["mean_abs_chgbatch_slices2_msignal"])))))) * 2.0), data["maxtominbatch_msignal"] ) ))) )) +

                            0.050000*np.tanh(((np.sin((np.where((((np.minimum(((data["maxtominbatch_msignal"])), ((data["abs_minbatch_slices2_msignal"])))) + (np.cos((data["maxtominbatch_slices2_msignal"]))))/2.0) <= -998, data["abs_minbatch_slices2_msignal"], data["abs_minbatch_slices2_msignal"] )))) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((np.minimum(((data["maxtominbatch_slices2_msignal"])), ((((data["maxtominbatch_slices2_msignal"]) + ((((np.cos(((-((data["rangebatch_slices2_msignal"])))))) > (data["abs_minbatch_msignal"]))*1.)))))))), ((data["maxtominbatch_slices2_msignal"])))) +

                            0.050000*np.tanh(np.where(data["medianbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_slices2_msignal"], data["maxtominbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(((np.minimum(((data["maxtominbatch_msignal"])), ((np.cos((((np.where(data["maxtominbatch_slices2_msignal"] <= -998, np.where(data["maxtominbatch"] <= -998, (2.0), data["abs_avgbatch_msignal"] ), data["abs_avgbatch_msignal"] )) * 2.0))))))) * 2.0)) +

                            0.050000*np.tanh(((data["abs_minbatch_slices2_msignal"]) * (np.sin((np.where(((np.sin((data["abs_minbatch_slices2_msignal"]))) * 2.0) <= -998, np.minimum(((np.where(np.minimum(((np.sin((data["stdbatch_slices2"])))), (((((data["abs_maxbatch"]) + (data["signal_shift_-1"]))/2.0)))) <= -998, data["abs_minbatch_slices2_msignal"], data["abs_maxbatch"] ))), ((np.sin((data["abs_minbatch_slices2_msignal"]))))), data["abs_minbatch_slices2_msignal"] )))))) +

                            0.050000*np.tanh(((((np.where(data["medianbatch_slices2_msignal"] <= -998, data["abs_minbatch_slices2_msignal"], np.sin(((((((np.cos((data["abs_minbatch_msignal"]))) <= (data["rangebatch_slices2"]))*1.)) + (((np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], data["maxtominbatch"] )) * 2.0))))) )) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((data["stdbatch_msignal"]) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((data["abs_minbatch_msignal"])), ((((data["abs_minbatch_slices2_msignal"]) * (np.sin((data["abs_minbatch_msignal"])))))))) +

                            0.050000*np.tanh((((data["medianbatch_msignal"]) + (((data["maxtominbatch_slices2_msignal"]) * (np.where((((data["abs_minbatch_msignal"]) > (((data["abs_avgbatch_slices2_msignal"]) / 2.0)))*1.) > -998, data["abs_avgbatch_slices2_msignal"], np.tanh((data["rangebatch_slices2_msignal"])) )))))/2.0)) +

                            0.050000*np.tanh(((np.sin((data["mean_abs_chgbatch_slices2_msignal"]))) * (np.sin((data["abs_avgbatch_slices2"]))))) +

                            0.050000*np.tanh(np.minimum(((np.sin((data["mean_abs_chgbatch_slices2_msignal"])))), ((data["mean_abs_chgbatch_slices2_msignal"])))) +

                            0.050000*np.tanh(((np.cos((np.where(((data["signal_shift_+1"]) / 2.0) > -998, ((data["abs_maxbatch_msignal"]) + (np.where(np.cos((((data["abs_maxbatch_msignal"]) + (((data["maxbatch_slices2"]) * 2.0))))) > -998, data["maxtominbatch_msignal"], ((data["signal_shift_-1"]) + (data["abs_maxbatch_msignal"])) ))), data["abs_maxbatch_msignal"] )))) * 2.0)) +

                            0.050000*np.tanh(np.sin((((data["abs_minbatch_slices2"]) + (np.where(data["maxtominbatch_slices2_msignal"] > -998, ((data["abs_minbatch_slices2"]) + (np.maximum(((data["mean_abs_chgbatch_slices2_msignal"])), ((np.where((((data["rangebatch_slices2"]) <= (((data["abs_minbatch_slices2_msignal"]) * (((data["abs_minbatch_msignal"]) - (data["mean_abs_chgbatch_slices2_msignal"]))))))*1.) > -998, ((data["abs_avgbatch_msignal"]) * 2.0), ((data["minbatch_msignal"]) * (data["medianbatch_slices2"])) )))))), data["medianbatch_msignal"] )))))) +

                            0.050000*np.tanh(np.sin((((np.sin((((np.where((((((((((((data["minbatch_msignal"]) + (data["medianbatch_slices2"]))/2.0)) > (data["stdbatch_slices2_msignal"]))*1.)) + (data["mean_abs_chgbatch_msignal"]))) <= (data["medianbatch_slices2"]))*1.) > -998, data["minbatch_msignal"], (((data["minbatch_msignal"]) <= (data["stdbatch_msignal"]))*1.) )) * 2.0)))) * 2.0)))) +

                            0.050000*np.tanh(np.where(data["rangebatch_slices2"] <= -998, ((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0), ((data["maxtominbatch_msignal"]) - (np.where(np.minimum(((data["maxtominbatch_msignal"])), ((data["minbatch"]))) > -998, data["signal_shift_-1"], np.sin((np.where(data["maxtominbatch_msignal"] <= -998, data["abs_minbatch_msignal"], data["maxtominbatch_msignal"] ))) ))) )) +

                            0.050000*np.tanh(np.sin((np.where(data["stdbatch_msignal"] <= -998, data["stdbatch_msignal"], data["rangebatch_slices2"] )))) +

                            0.050000*np.tanh((((-((np.cos((np.maximum(((np.minimum(((data["maxbatch_msignal"])), ((np.tanh(((((((data["minbatch_slices2_msignal"]) <= (((data["abs_avgbatch_slices2_msignal"]) * 2.0)))*1.)) * 2.0)))))))), ((((((data["abs_avgbatch_slices2_msignal"]) * (np.sin((((((-((np.maximum(((data["mean_abs_chgbatch_msignal"])), ((data["abs_avgbatch_slices2_msignal"]))))))) <= (data["abs_avgbatch_slices2"]))*1.)))))) * (((data["mean_abs_chgbatch_msignal"]) / 2.0)))))))))))) * 2.0)) +

                            0.050000*np.tanh(np.where((((data["maxtominbatch_msignal"]) <= (((data["maxtominbatch_msignal"]) * 2.0)))*1.) <= -998, np.where(np.cos((data["stdbatch_slices2_msignal"])) <= -998, data["maxbatch_slices2_msignal"], data["stdbatch_slices2_msignal"] ), ((np.sin((((data["mean_abs_chgbatch_msignal"]) - (((np.cos(((-((data["signal_shift_-1"])))))) / 2.0)))))) * 2.0) )) +

                            0.050000*np.tanh(((np.sin((np.maximum(((data["stdbatch_slices2_msignal"])), ((np.where(((data["maxbatch_msignal"]) - (data["mean_abs_chgbatch_slices2"])) <= -998, data["maxtominbatch"], np.where(np.cos((data["abs_minbatch_slices2_msignal"])) <= -998, ((((-((np.sin((data["abs_avgbatch_msignal"])))))) > (data["minbatch"]))*1.), data["maxtominbatch"] ) ))))))) * 2.0)) +

                            0.050000*np.tanh(np.where(np.sin((data["stdbatch_msignal"])) <= -998, ((np.where(((np.sin((((((np.sin((data["stdbatch_msignal"]))) * 2.0)) * 2.0)))) * (data["stdbatch_msignal"])) > -998, data["medianbatch_msignal"], np.maximum(((np.sin((data["stdbatch_msignal"])))), ((data["stdbatch_msignal"]))) )) * 2.0), ((((np.sin((data["stdbatch_msignal"]))) * 2.0)) * 2.0) )) +

                            0.050000*np.tanh(np.where(data["mean_abs_chgbatch_slices2"] <= -998, data["abs_avgbatch_msignal"], (((6.0)) * (np.cos((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, ((np.cos((np.cos((np.where(np.where(data["abs_avgbatch_msignal"] <= -998, ((data["mean_abs_chgbatch_slices2_msignal"]) * (np.cos(((6.0))))), data["maxtominbatch"] ) <= -998, data["abs_avgbatch_slices2_msignal"], ((data["abs_avgbatch_slices2_msignal"]) * 2.0) )))))) * 2.0), ((data["abs_avgbatch_slices2_msignal"]) * 2.0) ))))) )) +

                            0.050000*np.tanh(((np.where(np.cos((data["maxtominbatch"])) > -998, data["maxtominbatch"], np.minimum(((np.cos((np.sin((data["abs_avgbatch_msignal"])))))), ((data["medianbatch_slices2_msignal"]))) )) + (((np.sin((((data["signal_shift_-1"]) - (data["abs_avgbatch_msignal"]))))) * 2.0)))) +

                            0.050000*np.tanh((((7.0)) * (np.cos((((np.minimum(((((data["mean_abs_chgbatch_msignal"]) * (np.cos((data["mean_abs_chgbatch_msignal"])))))), ((data["signal_shift_-1"])))) * (np.where(data["abs_maxbatch_slices2"] <= -998, data["maxtominbatch_slices2"], np.where(data["maxbatch_slices2_msignal"] <= -998, data["mean_abs_chgbatch_msignal"], np.cos((data["mean_abs_chgbatch_msignal"])) ) )))))))) +

                            0.050000*np.tanh(((data["abs_minbatch_slices2_msignal"]) - (data["signal_shift_-1"]))) +

                            0.050000*np.tanh(((data["minbatch"]) * (np.cos((((((np.maximum(((((data["maxtominbatch_slices2_msignal"]) / 2.0))), (((((np.where(((((data["signal_shift_+1"]) / 2.0)) * 2.0) > -998, (((((data["maxbatch_msignal"]) + (((data["maxtominbatch_slices2_msignal"]) * 2.0)))/2.0)) * 2.0), data["abs_avgbatch_msignal"] )) + (data["maxtominbatch_slices2_msignal"]))/2.0))))) * 2.0)) / 2.0)))))) +

                            0.050000*np.tanh(((data["maxtominbatch"]) - (np.cos((((np.where(np.cos((data["stdbatch_msignal"])) <= -998, data["minbatch_slices2"], np.where(np.cos((data["maxtominbatch"])) > -998, data["signal_shift_+1"], data["signal_shift_-1"] ) )) * 2.0)))))) +

                            0.050000*np.tanh(((((np.where(np.sin((data["stdbatch_slices2_msignal"])) > -998, data["mean_abs_chgbatch_slices2_msignal"], data["signal"] )) - (np.tanh((((data["abs_avgbatch_msignal"]) * (((data["abs_maxbatch_slices2"]) * (np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) - (np.cos((data["maxbatch_slices2_msignal"]))))))))))))))) * (np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) - (np.minimum(((data["maxbatch_slices2_msignal"])), ((np.cos((data["mean_abs_chgbatch_slices2_msignal"])))))))))))) +

                            0.050000*np.tanh(((((np.where(data["abs_avgbatch_slices2_msignal"] > -998, np.sin((data["minbatch_msignal"])), ((((data["rangebatch_slices2_msignal"]) * (data["medianbatch_msignal"]))) - (data["maxbatch_slices2_msignal"])) )) + (np.where(data["rangebatch_slices2"] > -998, ((data["rangebatch_slices2_msignal"]) * ((-((np.cos(((-((data["abs_avgbatch_slices2_msignal"])))))))))), data["signal_shift_-1"] )))) - (data["maxbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((data["abs_maxbatch_slices2_msignal"]) * (np.sin(((-((np.cos((np.where(data["mean_abs_chgbatch_msignal"] > -998, ((data["signal_shift_-1"]) - (np.where(data["signal_shift_-1"] <= -998, np.where((((data["maxbatch_slices2"]) <= (data["mean_abs_chgbatch_msignal"]))*1.) <= -998, data["mean_abs_chgbatch_msignal"], data["abs_minbatch_slices2_msignal"] ), np.where(data["stdbatch_slices2_msignal"] <= -998, data["abs_minbatch_slices2_msignal"], data["abs_minbatch_slices2_msignal"] ) ))), ((np.sin(((5.0)))) * 2.0) ))))))))))) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) * (np.cos((((np.minimum((((((((np.minimum(((data["signal_shift_+1"])), ((data["meanbatch_slices2"])))) * 2.0)) > (data["signal_shift_-1"]))*1.))), ((np.minimum(((np.minimum(((data["signal_shift_-1"])), ((np.minimum(((data["signal_shift_+1"])), ((data["abs_minbatch_slices2"])))))))), ((data["abs_minbatch_slices2"]))))))) * 2.0)))))) +

                            0.050000*np.tanh((-((np.where(np.minimum(((data["stdbatch_slices2_msignal"])), ((np.minimum(((data["mean_abs_chgbatch_msignal"])), (((((data["maxtominbatch_msignal"]) + (np.sin((data["maxtominbatch_slices2"]))))/2.0))))))) <= -998, data["mean_abs_chgbatch_msignal"], np.where(data["minbatch"] <= -998, ((data["abs_minbatch_slices2"]) * 2.0), ((data["minbatch"]) * (np.minimum(((data["mean_abs_chgbatch_msignal"])), ((np.sin((data["stdbatch_slices2_msignal"]))))))) ) ))))) +

                            0.050000*np.tanh(((data["meanbatch_slices2"]) + (((data["maxtominbatch_slices2_msignal"]) + ((-((((np.cos(((((9.0)) - (data["abs_avgbatch_slices2_msignal"]))))) * (((((((data["maxtominbatch"]) * (((np.minimum(((data["abs_minbatch_slices2_msignal"])), (((-((((np.sin((data["stdbatch_slices2_msignal"]))) * 2.0)))))))) / 2.0)))) * (((data["mean_abs_chgbatch_msignal"]) * 2.0)))) - ((9.0))))))))))))) +

                            0.050000*np.tanh(((np.where(data["abs_avgbatch_msignal"] <= -998, ((data["abs_minbatch_slices2_msignal"]) - (np.where(data["medianbatch_slices2_msignal"] > -998, data["minbatch"], ((data["abs_avgbatch_msignal"]) / 2.0) ))), np.cos((np.maximum(((data["maxtominbatch_msignal"])), ((((data["abs_avgbatch_msignal"]) - (np.where(data["medianbatch_slices2_msignal"] > -998, data["minbatch"], data["maxtominbatch"] )))))))) )) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((data["abs_minbatch_slices2_msignal"])), ((((np.sin((((data["stdbatch_slices2_msignal"]) - ((((((data["signal_shift_+1"]) * 2.0)) > ((((2.0)) + (((np.where((4.0) > -998, ((np.sin(((((data["maxtominbatch_slices2_msignal"]) > (data["stdbatch_msignal"]))*1.)))) * 2.0), np.tanh((data["stdbatch_slices2_msignal"])) )) * (data["stdbatch_slices2_msignal"]))))))*1.)))))) * 2.0))))) +

                            0.050000*np.tanh(np.minimum(((data["maxbatch_msignal"])), ((np.minimum(((data["stdbatch_slices2_msignal"])), ((((np.sin((np.sin((data["stdbatch_slices2_msignal"]))))) * (data["stdbatch_slices2_msignal"]))))))))) +

                            0.050000*np.tanh(np.where(np.sin((((np.maximum(((data["abs_avgbatch_slices2_msignal"])), ((data["abs_avgbatch_slices2_msignal"])))) - (((np.tanh((data["signal_shift_+1"]))) / 2.0))))) <= -998, data["mean_abs_chgbatch_slices2_msignal"], np.where(data["rangebatch_msignal"] <= -998, ((data["abs_minbatch_slices2_msignal"]) - (data["abs_minbatch_msignal"])), np.sin((((data["signal_shift_+1"]) - (data["abs_avgbatch_slices2_msignal"])))) ) )) +

                            0.050000*np.tanh(np.sin((((data["abs_avgbatch_slices2_msignal"]) + (((data["minbatch_slices2_msignal"]) - (((((np.maximum(((data["signal_shift_+1"])), ((np.sin((((np.sin((((data["minbatch_slices2_msignal"]) + (data["minbatch_slices2_msignal"]))))) + (data["minbatch_slices2_msignal"])))))))) * ((((data["medianbatch_slices2_msignal"]) <= (np.where(data["abs_minbatch_slices2"] <= -998, data["medianbatch_slices2_msignal"], ((data["medianbatch_slices2"]) * 2.0) )))*1.)))) / 2.0)))))))) +

                            0.050000*np.tanh(np.cos((((((data["stdbatch_msignal"]) - (data["signal_shift_-1_msignal"]))) + (np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["mean_abs_chgbatch_msignal"], ((((data["mean_abs_chgbatch_msignal"]) * 2.0)) - ((((data["mean_abs_chgbatch_msignal"]) <= (((np.tanh(((((((data["minbatch_msignal"]) + (np.maximum(((data["maxtominbatch_msignal"])), ((np.sin((np.sin((data["signal_shift_-1_msignal"])))))))))/2.0)) / 2.0)))) / 2.0)))*1.))) )))))) +

                            0.050000*np.tanh(np.sin((np.maximum(((data["stdbatch_slices2_msignal"])), ((np.minimum((((1.0))), ((np.where(np.maximum(((data["stdbatch_slices2_msignal"])), ((data["mean_abs_chgbatch_msignal"]))) > -998, data["maxtominbatch_msignal"], data["maxtominbatch"] )))))))))) +

                            0.050000*np.tanh(((np.sin(((((3.79419302940368652)) * (data["mean_abs_chgbatch_msignal"]))))) - ((((np.maximum(((((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0))), ((((data["abs_minbatch_msignal"]) * 2.0))))) <= ((((np.maximum(((data["signal_shift_-1"])), (((((((3.79419302940368652)) * (np.sin((data["mean_abs_chgbatch_msignal"]))))) * 2.0))))) <= (data["stdbatch_slices2"]))*1.)))*1.)))) +

                            0.050000*np.tanh(np.where(((((((data["mean_abs_chgbatch_msignal"]) * (np.maximum(((data["stdbatch_slices2_msignal"])), ((data["stdbatch_slices2_msignal"])))))) * 2.0)) / 2.0) <= -998, data["signal_shift_+1"], ((((np.sin((data["stdbatch_slices2_msignal"]))) * 2.0)) * 2.0) )) +

                            0.050000*np.tanh(((((((data["stdbatch_slices2_msignal"]) * (data["abs_minbatch_slices2_msignal"]))) + (np.where(((np.where(data["stdbatch_slices2_msignal"] > -998, np.sin(((10.0))), np.sin((np.sin((data["stdbatch_msignal"])))) )) + (((data["abs_minbatch_slices2_msignal"]) * (np.sin((data["abs_avgbatch_slices2_msignal"])))))) > -998, data["maxtominbatch_slices2_msignal"], data["abs_maxbatch_slices2"] )))) * (np.sin((data["stdbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(((((np.maximum(((data["abs_avgbatch_slices2_msignal"])), (((((((data["abs_minbatch_msignal"]) * (data["abs_avgbatch_slices2_msignal"]))) + (data["maxtominbatch_slices2"]))/2.0))))) * ((-((data["signal_shift_+1"])))))) + ((((((data["abs_maxbatch_slices2_msignal"]) <= ((((-((((((data["abs_minbatch_slices2"]) - (data["mean_abs_chgbatch_slices2_msignal"]))) / 2.0))))) * 2.0)))*1.)) - (data["abs_avgbatch_slices2"]))))) +

                            0.050000*np.tanh(((((data["maxtominbatch"]) - (np.where(((np.where(data["stdbatch_slices2_msignal"] > -998, data["signal_shift_-1"], data["abs_avgbatch_msignal"] )) * 2.0) > -998, data["signal_shift_-1"], data["stdbatch_slices2_msignal"] )))) * 2.0)) +

                            0.050000*np.tanh(((((np.sin((np.cos((((data["abs_maxbatch_slices2_msignal"]) - (data["signal_shift_-1"]))))))) * (((data["stdbatch_slices2_msignal"]) * (((((np.sin((np.cos((((data["abs_maxbatch_slices2_msignal"]) - (data["signal_shift_-1"]))))))) + (data["stdbatch_msignal"]))) * (data["rangebatch_slices2_msignal"]))))))) + (np.sin((np.cos((np.maximum(((data["medianbatch_slices2"])), ((data["stdbatch_msignal"])))))))))) +

                            0.050000*np.tanh(((((np.sin((np.maximum(((np.sin((np.sin((np.maximum(((data["stdbatch_slices2_msignal"])), ((data["signal_shift_+1"]))))))))), ((np.maximum(((data["stdbatch_slices2_msignal"])), ((np.where(((np.where(np.sin((np.maximum(((data["stdbatch_slices2_msignal"])), ((np.maximum(((data["abs_avgbatch_slices2"])), ((data["abs_avgbatch_slices2"])))))))) <= -998, data["minbatch"], data["abs_minbatch_slices2"] )) * 2.0) <= -998, data["meanbatch_slices2"], data["abs_minbatch_slices2_msignal"] )))))))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh((((((np.where(data["maxbatch_msignal"] <= -998, data["stdbatch_slices2"], (((-((data["mean_abs_chgbatch_slices2"])))) * 2.0) )) + (np.minimum(((data["abs_minbatch_slices2_msignal"])), ((((data["signal_shift_+1"]) * (((np.cos((data["abs_minbatch_slices2_msignal"]))) - (data["mean_abs_chgbatch_slices2"])))))))))/2.0)) * (((data["signal_shift_-1_msignal"]) * (data["rangebatch_slices2"]))))) +

                            0.050000*np.tanh(((np.sin((np.where(np.sin((np.maximum(((np.sin((np.maximum(((data["abs_avgbatch_slices2"])), (((((np.where(data["maxtominbatch"] <= -998, data["minbatch_slices2"], data["abs_maxbatch_slices2_msignal"] )) > (data["abs_maxbatch_slices2_msignal"]))*1.)))))))), ((data["stdbatch_msignal"]))))) > -998, data["stdbatch_slices2_msignal"], np.where(data["stdbatch_slices2_msignal"] > -998, data["stdbatch_slices2_msignal"], np.where(data["mean_abs_chgbatch_slices2"] <= -998, data["maxtominbatch_slices2_msignal"], data["maxtominbatch_slices2_msignal"] ) ) )))) * 2.0)) +

                            0.050000*np.tanh(np.where(data["maxtominbatch_msignal"] > -998, np.sin((np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_slices2_msignal"], data["rangebatch_slices2"] ))), data["signal_shift_+1"] )) +

                            0.050000*np.tanh(((((np.cos((((np.where(((data["abs_maxbatch_slices2_msignal"]) * 2.0) <= -998, np.where((((7.0)) + (np.cos((data["meanbatch_slices2"])))) > -998, data["abs_maxbatch_msignal"], np.where(data["mean_abs_chgbatch_msignal"] <= -998, data["maxtominbatch_msignal"], data["maxtominbatch_msignal"] ) ), data["maxtominbatch_msignal"] )) + (data["abs_maxbatch_slices2_msignal"]))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh((((8.0)) * (np.where(np.tanh(((((data["maxtominbatch_msignal"]) + ((-((data["stdbatch_slices2_msignal"])))))/2.0))) <= -998, ((data["stdbatch_slices2"]) + (((data["abs_maxbatch_msignal"]) + (np.sin((data["abs_avgbatch_slices2"])))))), np.where(data["rangebatch_slices2_msignal"] <= -998, data["minbatch_slices2"], np.tanh((np.sin((data["stdbatch_slices2_msignal"])))) ) )))) +

                            0.050000*np.tanh(((((np.cos((((np.where(np.tanh((data["stdbatch_slices2_msignal"])) > -998, ((data["medianbatch_slices2"]) * 2.0), data["medianbatch_slices2"] )) + (((np.cos((np.maximum(((data["mean_abs_chgbatch_msignal"])), (((((data["maxtominbatch_slices2_msignal"]) > (((data["rangebatch_slices2"]) * 2.0)))*1.))))))) + (data["stdbatch_slices2_msignal"]))))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh((((data["maxtominbatch_msignal"]) + (np.sin((np.where(data["abs_maxbatch_msignal"] <= -998, np.cos(((-((data["rangebatch_slices2"]))))), data["minbatch_slices2"] )))))/2.0)) +

                            0.050000*np.tanh((-((((((data["abs_avgbatch_msignal"]) - ((((data["abs_minbatch_slices2"]) + (np.cos((((data["signal"]) * (data["signal_shift_+1"]))))))/2.0)))) * (np.cos((((data["signal"]) * (data["signal_shift_+1"])))))))))) +

                            0.050000*np.tanh(np.where(data["signal_shift_+1_msignal"] <= -998, data["maxbatch_msignal"], np.where(data["mean_abs_chgbatch_slices2"] <= -998, data["mean_abs_chgbatch_slices2"], np.sin((data["stdbatch_msignal"])) ) )) +

                            0.050000*np.tanh((((-((((((data["rangebatch_slices2"]) * (((np.cos((np.where(((data["medianbatch_slices2"]) * 2.0) <= -998, np.tanh((((data["signal_shift_+1_msignal"]) / 2.0))), ((data["medianbatch_slices2"]) + (data["medianbatch_slices2"])) )))) / 2.0)))) / 2.0))))) * 2.0)) +

                            0.050000*np.tanh(((np.minimum(((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((np.where(data["medianbatch_slices2_msignal"] <= -998, ((data["abs_avgbatch_msignal"]) * (data["maxtominbatch_slices2_msignal"])), data["mean_abs_chgbatch_slices2_msignal"] )))))), ((np.where((((np.sin((((data["meanbatch_slices2_msignal"]) + (data["meanbatch_msignal"]))))) + (data["maxtominbatch_slices2_msignal"]))/2.0) <= -998, data["maxtominbatch_slices2_msignal"], np.sin((((data["maxtominbatch_slices2_msignal"]) + (data["meanbatch_msignal"])))) ))))) * 2.0)) +

                            0.050000*np.tanh(((np.cos((((np.where(data["meanbatch_msignal"] <= -998, ((np.where(data["signal_shift_-1"] > -998, data["minbatch_slices2"], np.cos(((-((data["rangebatch_slices2"]))))) )) * (((data["abs_maxbatch"]) - (np.where((6.0) <= -998, data["minbatch_slices2_msignal"], np.tanh((data["abs_maxbatch"])) ))))), data["maxbatch_msignal"] )) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(((np.sin((((data["stdbatch_slices2_msignal"]) + (np.sin((data["minbatch_slices2"]))))))) * (((((data["stdbatch_slices2_msignal"]) + (np.maximum(((data["medianbatch_slices2"])), (((2.0))))))) * (np.where(data["minbatch_slices2"] > -998, ((data["stdbatch_slices2_msignal"]) + (np.sin((data["minbatch_slices2"])))), data["stdbatch_slices2_msignal"] )))))) +

                            0.050000*np.tanh(np.sin((np.where(np.where(data["rangebatch_msignal"] > -998, data["stdbatch_slices2_msignal"], np.minimum(((data["maxtominbatch_msignal"])), (((-((data["meanbatch_msignal"])))))) ) > -998, data["stdbatch_slices2_msignal"], (11.81794071197509766) )))) +

                            0.050000*np.tanh(((np.sin((np.minimum(((np.minimum((((-((data["maxbatch_slices2"]))))), ((((data["minbatch_msignal"]) * 2.0)))))), ((np.where(np.where(data["rangebatch_msignal"] > -998, data["maxbatch_msignal"], np.sin((np.minimum((((-((data["maxbatch_slices2"]))))), ((data["minbatch_msignal"]))))) ) > -998, ((data["signal_shift_+1_msignal"]) * 2.0), data["meanbatch_slices2_msignal"] ))))))) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((((((np.sin((np.tanh((((np.sin((((data["abs_avgbatch_msignal"]) * 2.0)))) * 2.0)))))) * 2.0)) * 2.0))), ((((np.sin((data["stdbatch_msignal"]))) + (np.sin(((-((data["maxbatch_slices2"]))))))))))) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) * ((-(((((np.sin(((((data["rangebatch_slices2"]) > (data["signal_shift_+1"]))*1.)))) <= ((((((np.where(data["signal_shift_+1"] > -998, data["rangebatch_msignal"], data["stdbatch_msignal"] )) * 2.0)) + (((np.cos(((((data["maxtominbatch"]) <= (data["maxtominbatch"]))*1.)))) / 2.0)))/2.0)))*1.))))))) +

                            0.050000*np.tanh(((np.cos((((((((np.minimum(((np.cos((data["signal_shift_-1"])))), ((np.cos((((((((np.sin((((np.cos(((9.0)))) * 2.0)))) * 2.0)) * 2.0)) * (np.cos((data["signal_shift_-1"])))))))))) * 2.0)) * 2.0)) * (np.cos((data["signal_shift_-1"]))))))) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((data["abs_minbatch_msignal"])), (((((4.0)) * (np.sin((data["stdbatch_msignal"])))))))) +

                            0.050000*np.tanh(np.sin((((data["signal_shift_+1"]) - (np.where(data["signal_shift_+1"] > -998, np.where(data["signal_shift_+1"] > -998, data["abs_avgbatch_msignal"], data["signal_shift_+1"] ), np.cos((((data["abs_avgbatch_msignal"]) + (np.sin((((data["abs_minbatch_slices2"]) - (data["abs_avgbatch_msignal"])))))))) )))))) +

                            0.050000*np.tanh(np.maximum(((np.cos((np.maximum(((data["meanbatch_slices2"])), ((np.cos((((data["medianbatch_slices2"]) * (np.cos((((data["medianbatch_slices2"]) * (np.cos((data["medianbatch_slices2"]))))))))))))))))), ((data["stdbatch_msignal"])))) +

                            0.050000*np.tanh(np.sin((np.maximum(((np.maximum(((np.where(data["rangebatch_slices2"] <= -998, np.minimum(((data["maxtominbatch_slices2"])), ((data["stdbatch_slices2_msignal"]))), data["abs_minbatch_msignal"] ))), ((((data["rangebatch_slices2"]) / 2.0)))))), ((data["stdbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh((-((np.where(data["maxbatch_slices2_msignal"] <= -998, ((data["meanbatch_msignal"]) + (np.sin((np.sin((((np.sin(((-((((np.sin((data["signal_shift_+1"]))) * 2.0))))))) + (data["meanbatch_msignal"])))))))), data["signal_shift_+1"] ))))) +

                            0.050000*np.tanh(((((data["abs_avgbatch_slices2_msignal"]) - ((-((np.cos((((data["signal"]) - ((-(((-((np.cos((data["abs_avgbatch_msignal"])))))))))))))))))) * ((-((np.cos((data["abs_avgbatch_slices2_msignal"])))))))) +

                            0.050000*np.tanh(((data["abs_avgbatch_msignal"]) + (np.minimum(((data["minbatch_slices2_msignal"])), ((((((((data["maxbatch_slices2"]) * (np.where(np.sin((((data["maxbatch_slices2"]) * (np.where(((data["abs_avgbatch_msignal"]) / 2.0) > -998, data["abs_avgbatch_slices2"], data["maxbatch_slices2"] ))))) > -998, data["minbatch_slices2"], np.sin((data["abs_avgbatch_slices2"])) )))) / 2.0)) / 2.0))))))) +

                            0.050000*np.tanh(((np.cos((data["medianbatch_slices2"]))) + (data["maxbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((((((data["abs_minbatch_msignal"]) * (((((data["abs_minbatch_msignal"]) * (((data["medianbatch_msignal"]) * (np.sin((data["stdbatch_slices2_msignal"]))))))) * (data["abs_minbatch_msignal"]))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.sin((((data["minbatch_msignal"]) + (np.maximum(((data["abs_avgbatch_slices2_msignal"])), ((np.maximum((((-((data["abs_avgbatch_slices2_msignal"]))))), ((((data["signal"]) + (((data["signal_shift_+1_msignal"]) - (np.maximum(((data["abs_avgbatch_slices2_msignal"])), ((data["signal"])))))))))))))))))) +

                            0.050000*np.tanh(np.minimum(((((np.sin((data["stdbatch_slices2_msignal"]))) * ((9.0))))), ((((((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, data["meanbatch_slices2_msignal"], data["signal_shift_+1"] )) * (data["mean_abs_chgbatch_slices2_msignal"]))) + (np.maximum(((np.where(data["meanbatch_slices2_msignal"] <= -998, np.tanh((data["abs_maxbatch_msignal"])), data["abs_maxbatch_slices2_msignal"] ))), ((np.maximum(((data["abs_maxbatch_slices2_msignal"])), (((((data["signal_shift_+1"]) + (data["maxtominbatch_slices2_msignal"]))/2.0))))))))))))) +

                            0.050000*np.tanh(np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((np.cos((((data["signal_shift_+1"]) + (np.cos((((((((((((-(((4.55383777618408203))))) <= (data["signal_shift_+1"]))*1.)) + ((4.55383777618408203)))/2.0)) <= (data["signal_shift_+1"]))*1.))))))))))) +

                            0.050000*np.tanh((-((np.cos(((((((-((data["signal_shift_+1"])))) * 2.0)) + ((((data["maxtominbatch"]) > ((((np.cos((((data["signal_shift_+1_msignal"]) + ((((data["maxtominbatch"]) > (np.cos(((((-((data["signal_shift_+1"])))) * 2.0)))))*1.)))))) + (np.sin(((((((-((np.cos(((3.0))))))) / 2.0)) * 2.0)))))/2.0)))*1.))))))))) +

                            0.050000*np.tanh(((np.where(data["signal"] > -998, np.where(((data["signal"]) * 2.0) > -998, data["minbatch_slices2"], np.minimum(((data["stdbatch_slices2"])), ((data["maxbatch_slices2"]))) ), np.where(((data["minbatch_msignal"]) * 2.0) > -998, data["meanbatch_slices2_msignal"], ((data["signal"]) * 2.0) ) )) * (np.cos((((data["signal"]) * 2.0)))))) +

                            0.050000*np.tanh(np.cos((((data["signal_shift_-1"]) - (data["abs_maxbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(np.maximum(((data["abs_maxbatch_msignal"])), ((np.maximum(((np.tanh((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0))))), ((data["meanbatch_msignal"]))))))) +

                            0.050000*np.tanh(((((data["abs_avgbatch_slices2"]) * (((np.sin((data["abs_avgbatch_slices2"]))) * ((((data["maxbatch_slices2"]) + (((np.sin((data["abs_avgbatch_slices2"]))) * ((((data["signal_shift_-1_msignal"]) + (data["abs_avgbatch_slices2"]))/2.0)))))/2.0)))))) * ((6.0)))) +

                            0.050000*np.tanh(np.cos((((((np.cos((np.where((((np.cos((data["signal_shift_-1"]))) > (np.cos((((data["maxbatch_msignal"]) * (((np.cos((((data["signal_shift_-1"]) * 2.0)))) * 2.0)))))))*1.) <= -998, data["meanbatch_msignal"], data["signal_shift_-1"] )))) * 2.0)) * 2.0)))) +

                            0.050000*np.tanh(((np.sin(((-((np.sin((np.sin((data["abs_maxbatch"])))))))))) + (np.sin((np.cos((((data["maxtominbatch_msignal"]) + (np.maximum(((data["abs_maxbatch_slices2_msignal"])), (((-((np.sin((np.tanh((data["mean_abs_chgbatch_slices2_msignal"]))))))))))))))))))) +

                            0.050000*np.tanh(np.where(data["signal_shift_-1"] <= -998, ((((-((((data["maxtominbatch_slices2_msignal"]) / 2.0))))) + (((((((((((((data["minbatch_msignal"]) / 2.0)) / 2.0)) > (data["maxbatch_msignal"]))*1.)) / 2.0)) + (np.sin((data["stdbatch_slices2"]))))/2.0)))/2.0), (-((data["signal_shift_-1"]))) )) +

                            0.050000*np.tanh(((np.where(((((data["abs_avgbatch_slices2_msignal"]) * 2.0)) + (np.where(data["rangebatch_slices2"] > -998, np.tanh((data["minbatch"])), np.tanh((((data["abs_avgbatch_slices2_msignal"]) - (data["abs_avgbatch_slices2_msignal"])))) ))) > -998, data["abs_maxbatch"], ((((data["abs_avgbatch_slices2_msignal"]) - (data["minbatch"]))) / 2.0) )) * (np.cos((((data["abs_avgbatch_slices2_msignal"]) - (data["minbatch"]))))))) +

                            0.050000*np.tanh(np.sin((np.maximum(((data["stdbatch_slices2_msignal"])), ((np.where((((data["maxbatch_slices2_msignal"]) + (data["maxbatch_slices2_msignal"]))/2.0) <= -998, ((data["medianbatch_msignal"]) - ((-((((data["maxtominbatch"]) * 2.0)))))), (-((np.cos((data["minbatch_slices2"]))))) ))))))) +

                            0.050000*np.tanh(np.cos((((np.cos(((((data["signal_shift_+1_msignal"]) + (data["meanbatch_msignal"]))/2.0)))) * (np.maximum(((data["meanbatch_slices2"])), ((data["meanbatch_slices2"])))))))) +

                            0.050000*np.tanh(np.sin((np.where(data["signal_shift_+1_msignal"] <= -998, np.cos((np.sin((((np.where(data["signal_shift_+1_msignal"] <= -998, data["mean_abs_chgbatch_slices2_msignal"], data["minbatch"] )) * 2.0))))), (((data["medianbatch_slices2_msignal"]) > (((data["signal_shift_-1"]) * 2.0)))*1.) )))) +

                            0.050000*np.tanh(((np.where((((np.where(np.where(data["mean_abs_chgbatch_msignal"] > -998, (10.0), data["rangebatch_msignal"] ) > -998, (10.0), ((np.sin(((((data["abs_avgbatch_msignal"]) + (data["mean_abs_chgbatch_msignal"]))/2.0)))) * 2.0) )) + ((2.0)))/2.0) > -998, (10.0), data["abs_avgbatch_msignal"] )) * (np.sin(((((data["abs_avgbatch_msignal"]) + (data["mean_abs_chgbatch_msignal"]))/2.0)))))) +

                            0.050000*np.tanh(((np.sin((np.maximum(((np.maximum(((data["abs_avgbatch_slices2"])), ((data["stdbatch_msignal"]))))), ((data["signal_shift_-1"])))))) * 2.0)) +

                            0.050000*np.tanh(((data["signal_shift_+1_msignal"]) * (((np.where(np.tanh((data["mean_abs_chgbatch_msignal"])) <= -998, ((data["abs_avgbatch_slices2_msignal"]) - ((-((data["mean_abs_chgbatch_msignal"]))))), ((((((((((data["maxbatch_slices2"]) * 2.0)) * (data["minbatch_slices2_msignal"]))) + (data["mean_abs_chgbatch_msignal"]))/2.0)) + (((data["minbatch_slices2_msignal"]) - ((-(((((data["mean_abs_chgbatch_msignal"]) <= (data["maxtominbatch_slices2_msignal"]))*1.))))))))/2.0) )) * 2.0)))) +

                            0.050000*np.tanh(((data["maxtominbatch_msignal"]) * (data["medianbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((np.tanh((data["medianbatch_msignal"]))) * (np.where((((((data["abs_maxbatch_msignal"]) + (data["signal_shift_+1"]))/2.0)) + (np.sin((data["rangebatch_slices2_msignal"])))) <= -998, data["maxtominbatch_msignal"], np.where(data["abs_avgbatch_slices2"] <= -998, data["signal"], ((np.tanh((data["maxtominbatch_msignal"]))) + (data["maxtominbatch_msignal"])) ) )))) +

                            0.050000*np.tanh(np.sin((np.cos(((((((data["minbatch"]) <= (np.where(data["abs_avgbatch_slices2"] <= -998, ((np.sin((((((np.cos((np.cos((np.sin((data["abs_avgbatch_slices2"]))))))) / 2.0)) / 2.0)))) / 2.0), ((np.cos((data["abs_avgbatch_slices2"]))) / 2.0) )))*1.)) / 2.0)))))) +

                            0.050000*np.tanh(((((((9.67606735229492188)) + ((((data["minbatch_slices2"]) <= (data["abs_maxbatch_slices2"]))*1.)))/2.0)) - (((data["abs_maxbatch_slices2_msignal"]) - (np.cos(((((data["abs_avgbatch_slices2"]) <= (data["abs_maxbatch_slices2_msignal"]))*1.)))))))) +

                            0.050000*np.tanh((((data["signal"]) <= ((((data["rangebatch_slices2_msignal"]) + (data["abs_maxbatch"]))/2.0)))*1.)) +

                            0.050000*np.tanh(np.cos((((data["minbatch_slices2_msignal"]) + ((((np.where(data["stdbatch_slices2"] > -998, ((data["meanbatch_slices2"]) - ((((data["mean_abs_chgbatch_slices2"]) > (((np.tanh((data["maxbatch_slices2_msignal"]))) * (data["meanbatch_msignal"]))))*1.))), np.where(data["rangebatch_slices2"] > -998, data["signal_shift_+1_msignal"], data["meanbatch_msignal"] ) )) <= (data["meanbatch_slices2"]))*1.)))))) +

                            0.050000*np.tanh(np.maximum((((-((((np.where(((data["rangebatch_msignal"]) + (((np.tanh((data["abs_minbatch_msignal"]))) + (np.sin((data["stdbatch_slices2_msignal"])))))) > -998, data["meanbatch_slices2"], data["rangebatch_msignal"] )) + (np.sin((data["maxbatch_slices2"]))))))))), ((np.sin((data["stdbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(np.cos((((data["abs_maxbatch"]) - (((data["abs_avgbatch_msignal"]) * ((((data["mean_abs_chgbatch_msignal"]) + (((((-((data["minbatch_msignal"])))) <= ((((data["signal_shift_-1"]) > ((-((data["abs_maxbatch"])))))*1.)))*1.)))/2.0)))))))) +

                            0.050000*np.tanh(np.sin((np.sin((((data["signal_shift_-1"]) - (np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((np.where(np.where(((data["signal"]) - (data["rangebatch_slices2_msignal"])) <= -998, data["abs_maxbatch_slices2_msignal"], ((data["signal"]) - (np.sin((data["signal"])))) ) <= -998, data["rangebatch_slices2_msignal"], data["abs_maxbatch_slices2"] ))))))))))) +

                            0.050000*np.tanh(((((data["abs_avgbatch_slices2"]) * (np.where((1.0) > -998, data["medianbatch_slices2_msignal"], ((np.sin((data["abs_maxbatch_msignal"]))) * ((1.0))) )))) * (np.sin((np.sin((data["stdbatch_slices2_msignal"]))))))) +

                            0.050000*np.tanh(((data["medianbatch_slices2_msignal"]) * (((((((data["medianbatch_slices2_msignal"]) + ((-((data["signal_shift_+1"])))))/2.0)) + (np.where(data["medianbatch_slices2_msignal"] <= -998, data["mean_abs_chgbatch_slices2"], data["maxtominbatch_slices2_msignal"] )))/2.0)))) +

                            0.050000*np.tanh(np.maximum(((np.sin((data["maxbatch_slices2"])))), ((np.maximum(((np.where(data["maxbatch_msignal"] <= -998, data["rangebatch_slices2_msignal"], ((((9.07463359832763672)) > (data["abs_maxbatch"]))*1.) ))), ((data["abs_maxbatch_msignal"]))))))) +

                            0.050000*np.tanh(((((np.sin((np.sin((np.where(np.where(np.sin((np.where(data["abs_minbatch_slices2"] > -998, data["abs_avgbatch_slices2"], data["abs_avgbatch_slices2"] ))) > -998, data["abs_avgbatch_slices2"], data["abs_avgbatch_slices2"] ) > -998, data["abs_avgbatch_slices2"], data["abs_avgbatch_slices2"] )))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.sin((((((((data["minbatch_slices2_msignal"]) - ((((data["mean_abs_chgbatch_msignal"]) > (np.sin((((((np.sin((data["medianbatch_msignal"]))) / 2.0)) * 2.0)))))*1.)))) * 2.0)) * 2.0)))) +

                            0.050000*np.tanh(np.maximum(((data["stdbatch_slices2"])), (((((np.where((((((data["mean_abs_chgbatch_slices2"]) + (np.cos((np.cos(((((((data["rangebatch_slices2"]) * 2.0)) + (np.cos((data["abs_maxbatch"]))))/2.0)))))))/2.0)) * (data["abs_maxbatch_slices2"])) > -998, data["rangebatch_msignal"], np.cos((data["rangebatch_msignal"])) )) + (data["stdbatch_slices2"]))/2.0))))) +

                            0.050000*np.tanh((((((data["signal_shift_+1"]) <= (np.sin((np.maximum(((np.sin(((3.0))))), (((4.44015836715698242))))))))*1.)) * 2.0)) +

                            0.050000*np.tanh(np.where(np.sin((data["minbatch_msignal"])) <= -998, ((data["maxbatch_msignal"]) * 2.0), np.sin((((((np.cos((np.cos((((np.cos((np.minimum(((((data["mean_abs_chgbatch_msignal"]) / 2.0))), ((data["abs_minbatch_msignal"])))))) + (data["abs_minbatch_slices2_msignal"]))))))) + (data["abs_minbatch_slices2_msignal"]))) - (data["maxbatch_msignal"])))) )) +

                            0.050000*np.tanh(data["rangebatch_slices2"]) +

                            0.050000*np.tanh(((((np.cos((((data["minbatch_slices2"]) - (data["abs_avgbatch_msignal"]))))) * (np.where(np.sin(((((((data["minbatch_slices2"]) - (data["maxtominbatch_slices2_msignal"]))) + (np.cos((((((data["minbatch_slices2"]) - (data["abs_avgbatch_msignal"]))) - (((data["abs_avgbatch_slices2"]) - (data["abs_avgbatch_msignal"]))))))))/2.0))) > -998, data["abs_avgbatch_msignal"], data["minbatch_slices2"] )))) * 2.0)) +

                            0.050000*np.tanh((((4.0)) * (np.sin((np.where(((np.sin((((np.sin((data["abs_avgbatch_slices2"]))) + ((-((((data["signal_shift_+1"]) / 2.0))))))))) + ((-((((data["signal_shift_+1"]) / 2.0)))))) <= -998, ((data["abs_avgbatch_msignal"]) + (data["mean_abs_chgbatch_slices2"])), ((data["minbatch_slices2_msignal"]) + (data["abs_avgbatch_msignal"])) )))))) +

                            0.050000*np.tanh(((((np.cos((data["stdbatch_msignal"]))) * (((np.cos((data["meanbatch_slices2"]))) - ((-((data["stdbatch_msignal"])))))))) - (data["meanbatch_slices2"]))) +

                            0.050000*np.tanh(np.where(data["maxbatch_slices2_msignal"] > -998, np.maximum(((((np.maximum(((data["abs_minbatch_slices2_msignal"])), ((np.where(np.tanh((((data["abs_maxbatch_slices2_msignal"]) - (((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0))))) <= -998, data["maxbatch_slices2"], data["abs_maxbatch"] ))))) * 2.0))), ((np.tanh((((data["abs_maxbatch"]) / 2.0)))))), np.where(data["abs_minbatch_msignal"] > -998, data["abs_avgbatch_slices2"], data["maxbatch_msignal"] ) )) +

                            0.050000*np.tanh(((np.minimum(((np.minimum(((data["abs_avgbatch_slices2"])), ((((np.sin((data["abs_avgbatch_slices2"]))) * 2.0)))))), ((data["abs_avgbatch_slices2"])))) * 2.0)) +

                            0.050000*np.tanh((-((np.sin((data["maxbatch_slices2"])))))) +

                            0.050000*np.tanh((((6.0)) - (np.maximum(((data["abs_maxbatch_slices2"])), ((np.maximum(((np.maximum(((np.tanh((((data["signal_shift_+1"]) * 2.0))))), ((np.maximum(((data["abs_avgbatch_slices2"])), ((data["abs_maxbatch_slices2"])))))))), ((data["abs_maxbatch_slices2"]))))))))) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) + (np.maximum(((data["rangebatch_slices2"])), ((np.where(((np.cos((np.where(np.cos((data["stdbatch_slices2_msignal"])) > -998, data["signal"], np.maximum(((data["maxbatch_slices2_msignal"])), ((((((((3.20188593864440918)) + (data["meanbatch_msignal"]))) <= (data["abs_minbatch_slices2"]))*1.)))) )))) / 2.0) <= -998, ((data["rangebatch_slices2"]) * 2.0), data["mean_abs_chgbatch_slices2_msignal"] ))))))) +

                            0.050000*np.tanh(((np.where(((((np.where(data["signal_shift_-1_msignal"] > -998, ((data["minbatch_msignal"]) + (data["stdbatch_slices2_msignal"])), data["signal_shift_-1_msignal"] )) * (((data["minbatch_msignal"]) + (data["stdbatch_slices2_msignal"]))))) * 2.0) > -998, ((data["signal_shift_-1_msignal"]) * 2.0), ((((data["mean_abs_chgbatch_msignal"]) * (data["abs_avgbatch_slices2_msignal"]))) * 2.0) )) * (((data["minbatch_msignal"]) + (data["stdbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(np.cos((np.where(((np.cos((data["signal_shift_-1"]))) / 2.0) > -998, ((((np.cos((data["abs_minbatch_msignal"]))) * 2.0)) + (((np.tanh((((data["signal_shift_+1_msignal"]) * (data["rangebatch_slices2_msignal"]))))) / 2.0))), ((np.sin(((11.49898338317871094)))) / 2.0) )))) +

                            0.050000*np.tanh(np.where(data["abs_maxbatch_slices2_msignal"] > -998, (((data["meanbatch_slices2_msignal"]) <= (data["abs_avgbatch_msignal"]))*1.), ((data["medianbatch_slices2_msignal"]) - (np.cos((((data["medianbatch_slices2_msignal"]) - (data["minbatch_slices2_msignal"])))))) )))  

    

    def GP_class_2(self,data):

        return self.Output( -2.199090 +

                            0.050000*np.tanh(np.minimum(((data["stdbatch_msignal"])), ((np.where(np.minimum(((np.where(((np.where(data["maxbatch_slices2"] <= -998, data["stdbatch_msignal"], ((data["stdbatch_msignal"]) * 2.0) )) - (data["abs_minbatch_msignal"])) <= -998, np.minimum(((np.cos((data["meanbatch_slices2"])))), ((np.cos((data["meanbatch_slices2"]))))), np.cos((data["maxbatch_slices2"])) ))), ((data["abs_minbatch_msignal"]))) <= -998, data["abs_avgbatch_msignal"], ((data["maxbatch_slices2"]) - (data["abs_minbatch_msignal"])) ))))) +

                            0.050000*np.tanh(np.where(((data["signal_shift_+1"]) - (data["minbatch_msignal"])) <= -998, ((((((np.minimum(((data["stdbatch_msignal"])), ((data["maxbatch_slices2"])))) * 2.0)) * (np.cos((data["abs_avgbatch_slices2"]))))) * 2.0), ((np.minimum(((data["stdbatch_msignal"])), ((data["maxbatch_slices2"])))) * 2.0) )) +

                            0.050000*np.tanh(((data["maxbatch_slices2"]) * (((data["abs_maxbatch_slices2_msignal"]) * (np.where(data["mean_abs_chgbatch_msignal"] > -998, data["mean_abs_chgbatch_msignal"], data["mean_abs_chgbatch_msignal"] )))))) +

                            0.050000*np.tanh(((np.minimum(((data["mean_abs_chgbatch_msignal"])), ((((np.minimum(((data["mean_abs_chgbatch_msignal"])), ((((np.minimum(((data["mean_abs_chgbatch_msignal"])), ((np.tanh(((((((((data["meanbatch_slices2"]) + (np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["minbatch"], (9.20059394836425781) )))) > (data["stdbatch_slices2_msignal"]))*1.)) - (data["maxtominbatch_slices2_msignal"])))))))) - (data["maxtominbatch_slices2_msignal"])))))) - (data["maxtominbatch_slices2_msignal"])))))) * 2.0)) +

                            0.050000*np.tanh(((data["mean_abs_chgbatch_msignal"]) * (((((data["maxbatch_slices2"]) + (data["maxbatch_slices2"]))) * 2.0)))) +

                            0.050000*np.tanh(((data["maxbatch_slices2"]) * ((((data["abs_minbatch_slices2_msignal"]) + ((((data["abs_minbatch_slices2_msignal"]) + (((data["abs_maxbatch_slices2_msignal"]) * (data["stdbatch_slices2_msignal"]))))/2.0)))/2.0)))) +

                            0.050000*np.tanh(((((data["minbatch"]) * (((data["mean_abs_chgbatch_slices2_msignal"]) * (np.where(np.where(data["medianbatch_msignal"] <= -998, data["meanbatch_msignal"], data["maxbatch_slices2"] ) > -998, data["maxtominbatch_slices2_msignal"], np.where(data["signal_shift_+1"] > -998, data["maxtominbatch_slices2_msignal"], data["stdbatch_msignal"] ) )))))) + (data["maxbatch_slices2"]))) +

                            0.050000*np.tanh(((((np.sin((np.sin((np.where(data["stdbatch_slices2_msignal"] > -998, data["abs_avgbatch_msignal"], np.cos(((10.52720451354980469))) )))))) * ((10.52720451354980469)))) * ((((((((10.52720451354980469)) + (((np.sin((np.cos(((10.52720451354980469)))))) * (((((10.52720451354980469)) + (np.sin(((10.52720451354980469)))))/2.0)))))/2.0)) + (data["abs_avgbatch_msignal"]))/2.0)))) +

                            0.050000*np.tanh(np.minimum(((((np.minimum(((((data["abs_avgbatch_slices2_msignal"]) * (data["abs_avgbatch_slices2_msignal"])))), ((((((data["mean_abs_chgbatch_msignal"]) * (data["meanbatch_slices2"]))) + (np.tanh((data["maxbatch_slices2"])))))))) * 2.0))), ((((data["stdbatch_slices2_msignal"]) * 2.0))))) +

                            0.050000*np.tanh(np.minimum((((((data["abs_maxbatch_msignal"]) + (data["stdbatch_slices2"]))/2.0))), ((((np.where(np.cos((data["maxbatch_slices2"])) > -998, data["maxbatch_slices2"], ((np.tanh((data["minbatch_msignal"]))) * (data["maxbatch_slices2"])) )) * (data["mean_abs_chgbatch_msignal"])))))) +

                            0.050000*np.tanh(((((data["maxbatch_slices2"]) * 2.0)) * (np.where(data["maxbatch_slices2"] <= -998, np.cos((np.where(data["stdbatch_slices2_msignal"] <= -998, np.tanh((data["rangebatch_slices2"])), data["stdbatch_slices2_msignal"] ))), data["mean_abs_chgbatch_msignal"] )))) +

                            0.050000*np.tanh(((data["stdbatch_msignal"]) * (np.where(data["rangebatch_slices2"] > -998, data["maxbatch_slices2"], ((((data["meanbatch_slices2_msignal"]) * (np.cos((data["maxbatch_slices2"]))))) / 2.0) )))) +

                            0.050000*np.tanh((((2.0)) * (np.cos((np.where(np.where((((data["rangebatch_msignal"]) <= ((((data["rangebatch_msignal"]) <= (np.maximum(((data["abs_minbatch_msignal"])), ((np.where((11.72177124023437500) > -998, data["signal"], (-((data["rangebatch_msignal"]))) ))))))*1.)))*1.) > -998, data["abs_minbatch_msignal"], (((np.cos((data["maxbatch_slices2"]))) + (data["medianbatch_slices2_msignal"]))/2.0) ) > -998, data["abs_minbatch_msignal"], data["stdbatch_msignal"] )))))) +

                            0.050000*np.tanh(np.minimum(((((data["maxbatch_slices2"]) * 2.0))), ((((data["abs_avgbatch_slices2_msignal"]) + (((((data["stdbatch_slices2_msignal"]) + ((((((data["stdbatch_slices2_msignal"]) * (np.minimum(((data["stdbatch_slices2_msignal"])), ((data["mean_abs_chgbatch_msignal"])))))) + (data["abs_minbatch_slices2_msignal"]))/2.0)))) + (((np.minimum(((((((data["abs_minbatch_msignal"]) * 2.0)) / 2.0))), ((data["abs_minbatch_slices2_msignal"])))) * (data["maxbatch_slices2"])))))))))) +

                            0.050000*np.tanh(((((((data["stdbatch_msignal"]) + (((data["abs_maxbatch_msignal"]) * (((((data["abs_minbatch_slices2_msignal"]) * (data["signal_shift_-1"]))) / 2.0)))))) + (np.where(data["abs_avgbatch_msignal"] > -998, data["stdbatch_slices2"], np.cos((data["mean_abs_chgbatch_msignal"])) )))) * 2.0)) +

                            0.050000*np.tanh(((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)) * (np.where(np.where(((data["maxbatch_slices2"]) * (data["maxtominbatch_msignal"])) > -998, np.where(((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)) * 2.0) > -998, data["maxbatch_slices2"], data["meanbatch_slices2_msignal"] ), data["rangebatch_slices2"] ) > -998, np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, data["maxbatch_slices2"], ((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0) ), (((data["maxbatch_slices2"]) + (data["abs_minbatch_slices2_msignal"]))/2.0) )))) +

                            0.050000*np.tanh(((np.cos((((data["abs_minbatch_msignal"]) - ((((data["abs_minbatch_msignal"]) <= ((((data["stdbatch_slices2_msignal"]) > ((((((((((data["maxtominbatch"]) + (data["abs_maxbatch"]))) * (data["mean_abs_chgbatch_slices2_msignal"]))) <= (data["mean_abs_chgbatch_slices2_msignal"]))*1.)) * 2.0)))*1.)))*1.)))))) * 2.0)) +

                            0.050000*np.tanh(((((data["maxbatch_slices2"]) * (data["mean_abs_chgbatch_msignal"]))) * 2.0)) +

                            0.050000*np.tanh(np.where(np.sin((np.maximum(((data["abs_avgbatch_slices2_msignal"])), ((data["abs_avgbatch_slices2_msignal"]))))) <= -998, data["maxbatch_slices2"], ((data["mean_abs_chgbatch_slices2_msignal"]) * (data["maxbatch_slices2"])) )) +

                            0.050000*np.tanh(((data["maxbatch_slices2"]) * (np.where(data["maxbatch_slices2"] <= -998, (((((np.minimum(((data["mean_abs_chgbatch_msignal"])), ((data["maxbatch_slices2"])))) + (((data["abs_minbatch_msignal"]) * 2.0)))/2.0)) * 2.0), np.where((((((data["mean_abs_chgbatch_msignal"]) * 2.0)) <= (np.where(data["abs_maxbatch"] <= -998, data["signal_shift_-1_msignal"], data["abs_minbatch_msignal"] )))*1.) <= -998, data["maxbatch_slices2"], ((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0) ) )))) +

                            0.050000*np.tanh(((data["maxbatch_slices2"]) * (np.where(data["mean_abs_chgbatch_msignal"] <= -998, data["maxbatch_slices2"], data["mean_abs_chgbatch_msignal"] )))) +

                            0.050000*np.tanh(np.minimum(((((np.minimum(((data["rangebatch_slices2"])), ((((data["mean_abs_chgbatch_msignal"]) * 2.0))))) * 2.0))), ((((np.minimum(((((data["mean_abs_chgbatch_msignal"]) * 2.0))), ((((np.sin((np.where(np.cos((((data["abs_avgbatch_msignal"]) * 2.0))) > -998, data["abs_avgbatch_msignal"], (((((data["medianbatch_msignal"]) + (data["stdbatch_slices2_msignal"]))/2.0)) / 2.0) )))) * 2.0))))) * 2.0))))) +

                            0.050000*np.tanh(((np.where((3.88836002349853516) <= -998, data["minbatch_msignal"], np.cos(((-((data["maxtominbatch_msignal"]))))) )) * 2.0)) +

                            0.050000*np.tanh((((-((np.maximum(((np.minimum(((((data["abs_maxbatch_msignal"]) * 2.0))), (((((data["abs_maxbatch"]) + (np.where(np.cos((data["abs_minbatch_slices2"])) > -998, data["maxtominbatch_slices2_msignal"], (5.19431018829345703) )))/2.0)))))), ((((data["maxtominbatch_slices2_msignal"]) * (np.minimum(((((data["maxtominbatch_msignal"]) * 2.0))), ((data["stdbatch_msignal"])))))))))))) + ((3.0)))) +

                            0.050000*np.tanh(np.where(((np.cos((data["stdbatch_msignal"]))) - (((data["abs_maxbatch_slices2"]) + (((data["abs_maxbatch"]) * (((np.tanh((data["abs_avgbatch_slices2_msignal"]))) * (data["maxbatch_msignal"])))))))) <= -998, (-((np.tanh((data["medianbatch_slices2"]))))), ((np.cos((np.where(np.cos((data["abs_maxbatch"])) <= -998, data["maxtominbatch_slices2"], data["maxtominbatch_slices2_msignal"] )))) * (data["rangebatch_slices2_msignal"])) )) +

                            0.050000*np.tanh(((np.maximum(((np.cos((np.tanh((data["stdbatch_slices2_msignal"])))))), ((data["maxbatch_msignal"])))) * (((((np.where(np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) * (data["maxtominbatch_msignal"])))) > -998, ((np.sin((data["abs_avgbatch_slices2_msignal"]))) * (data["mean_abs_chgbatch_slices2_msignal"])), data["mean_abs_chgbatch_slices2_msignal"] )) * 2.0)) * 2.0)))) +

                            0.050000*np.tanh(((data["maxbatch_slices2"]) * (np.where(data["stdbatch_msignal"] <= -998, data["mean_abs_chgbatch_slices2_msignal"], np.where(np.where(((((data["maxbatch_slices2"]) * (data["abs_minbatch_slices2_msignal"]))) - (data["medianbatch_msignal"])) <= -998, data["maxbatch_slices2"], ((data["maxtominbatch_slices2_msignal"]) * (data["medianbatch_slices2_msignal"])) ) <= -998, data["medianbatch_slices2"], data["mean_abs_chgbatch_slices2_msignal"] ) )))) +

                            0.050000*np.tanh(((data["maxbatch_slices2"]) * (((((((data["mean_abs_chgbatch_slices2_msignal"]) + (((data["stdbatch_msignal"]) * (data["abs_minbatch_slices2"]))))) - ((-((((data["maxtominbatch_slices2_msignal"]) + (np.where(np.where(data["maxbatch_slices2"] > -998, (7.0), data["maxbatch_slices2"] ) > -998, np.sin((data["stdbatch_msignal"])), data["maxbatch_slices2"] ))))))))) * 2.0)))) +

                            0.050000*np.tanh(((np.minimum(((((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)) * 2.0))), ((data["maxbatch_slices2"])))) * 2.0)) +

                            0.050000*np.tanh(((((np.where(data["abs_minbatch_msignal"] > -998, ((np.where(np.cos((data["maxbatch_slices2"])) <= -998, data["abs_minbatch_msignal"], np.cos(((-((data["abs_minbatch_msignal"]))))) )) * 2.0), data["maxtominbatch_slices2_msignal"] )) + (((((data["maxtominbatch_slices2_msignal"]) * ((-((data["abs_minbatch_msignal"])))))) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(((np.where(data["abs_minbatch_msignal"] > -998, data["mean_abs_chgbatch_msignal"], data["stdbatch_msignal"] )) * 2.0)) +

                            0.050000*np.tanh(np.where(data["abs_minbatch_msignal"] <= -998, np.where((((data["maxbatch_slices2"]) + (data["stdbatch_slices2"]))/2.0) <= -998, (((data["mean_abs_chgbatch_slices2_msignal"]) > (np.tanh((data["mean_abs_chgbatch_slices2_msignal"]))))*1.), data["mean_abs_chgbatch_slices2_msignal"] ), np.where(data["minbatch"] <= -998, data["rangebatch_slices2_msignal"], ((((data["maxbatch_slices2"]) * (data["mean_abs_chgbatch_slices2_msignal"]))) * 2.0) ) )) +

                            0.050000*np.tanh(((((data["signal"]) * (data["maxtominbatch_slices2_msignal"]))) + (((np.sin((np.where((((7.0)) + (np.sin((data["mean_abs_chgbatch_msignal"])))) <= -998, data["maxtominbatch_slices2_msignal"], ((data["mean_abs_chgbatch_msignal"]) * (data["abs_minbatch_msignal"])) )))) * 2.0)))) +

                            0.050000*np.tanh(((((data["mean_abs_chgbatch_msignal"]) * (np.where((((np.where(((((2.0)) > (np.where(data["abs_avgbatch_slices2_msignal"] <= -998, (2.0), np.tanh((data["maxbatch_slices2"])) )))*1.) > -998, data["maxbatch_slices2"], data["maxbatch_slices2"] )) > (data["signal_shift_+1"]))*1.) > -998, data["maxbatch_slices2"], data["maxbatch_slices2"] )))) * 2.0)) +

                            0.050000*np.tanh(((((np.where(data["abs_avgbatch_slices2"] <= -998, np.maximum((((((data["abs_avgbatch_slices2"]) > (data["maxbatch_slices2_msignal"]))*1.))), ((np.cos((data["abs_maxbatch_msignal"]))))), np.sin((data["abs_avgbatch_slices2_msignal"])) )) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((((((data["abs_avgbatch_msignal"]) * (np.sin((data["abs_avgbatch_msignal"]))))) * (((((np.sin((data["abs_avgbatch_msignal"]))) + (((data["abs_minbatch_slices2_msignal"]) + (data["abs_avgbatch_msignal"]))))) + (data["rangebatch_slices2_msignal"]))))) * (data["mean_abs_chgbatch_msignal"]))) +

                            0.050000*np.tanh(((np.cos((np.minimum(((data["abs_minbatch_msignal"])), ((((data["stdbatch_slices2_msignal"]) - (np.cos((((np.cos((np.where(np.minimum(((data["maxtominbatch_slices2_msignal"])), ((data["maxbatch_msignal"]))) > -998, data["meanbatch_slices2"], data["abs_avgbatch_msignal"] )))) * 2.0))))))))))) * 2.0)) +

                            0.050000*np.tanh((((4.65593338012695312)) * (np.where(((np.cos((data["mean_abs_chgbatch_msignal"]))) * (data["mean_abs_chgbatch_msignal"])) > -998, np.cos((((np.cos((data["mean_abs_chgbatch_msignal"]))) - (data["abs_minbatch_msignal"])))), ((data["abs_minbatch_msignal"]) + ((11.16297435760498047))) )))) +

                            0.050000*np.tanh(np.where(data["stdbatch_slices2_msignal"] <= -998, (-(((-((((data["maxbatch_slices2_msignal"]) + (np.cos((data["maxtominbatch_msignal"])))))))))), ((data["abs_minbatch_msignal"]) + (((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0))) )) +

                            0.050000*np.tanh((((4.0)) * (((data["mean_abs_chgbatch_msignal"]) + (np.where(data["minbatch_msignal"] > -998, ((data["signal"]) * (((data["mean_abs_chgbatch_slices2_msignal"]) + (data["maxtominbatch_msignal"])))), np.minimum(((data["mean_abs_chgbatch_msignal"])), ((np.cos(((4.0)))))) )))))) +

                            0.050000*np.tanh(((((np.cos((np.minimum(((data["maxtominbatch_msignal"])), ((np.maximum((((5.92805910110473633))), ((np.minimum(((((((np.cos((np.minimum(((data["signal"])), ((np.maximum(((data["maxtominbatch_msignal"])), ((np.minimum(((data["mean_abs_chgbatch_msignal"])), ((data["medianbatch_slices2"])))))))))))) * 2.0)) * 2.0))), ((data["maxtominbatch_msignal"])))))))))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.sin((np.where(np.where(data["stdbatch_msignal"] > -998, data["rangebatch_slices2_msignal"], data["stdbatch_msignal"] ) > -998, data["abs_avgbatch_slices2_msignal"], ((data["abs_minbatch_slices2"]) * 2.0) )))) +

                            0.050000*np.tanh(((np.sin((np.where(data["maxbatch_slices2"] <= -998, np.maximum((((((((((data["mean_abs_chgbatch_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))) + (np.cos((data["signal_shift_+1"]))))) + (np.sin((data["maxtominbatch_slices2_msignal"]))))/2.0))), ((np.cos((data["abs_minbatch_msignal"]))))), ((data["mean_abs_chgbatch_msignal"]) * 2.0) )))) * 2.0)) +

                            0.050000*np.tanh(((np.sin((data["abs_avgbatch_msignal"]))) * ((((9.0)) + (data["abs_minbatch_msignal"]))))) +

                            0.050000*np.tanh(((np.sin((np.where(np.sin((np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], np.where(data["maxtominbatch"] > -998, ((data["mean_abs_chgbatch_slices2"]) / 2.0), np.sin((data["mean_abs_chgbatch_slices2"])) ) ))) > -998, data["abs_avgbatch_msignal"], data["rangebatch_msignal"] )))) * 2.0)) +

                            0.050000*np.tanh(((data["maxbatch_slices2_msignal"]) * (np.minimum(((np.sin((data["abs_avgbatch_msignal"])))), ((np.where(np.minimum(((np.sin((data["abs_avgbatch_msignal"])))), ((data["mean_abs_chgbatch_slices2_msignal"]))) > -998, ((data["abs_maxbatch_slices2"]) * (data["abs_minbatch_msignal"])), np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) * (data["rangebatch_msignal"])))) ))))))) +

                            0.050000*np.tanh(((np.where(data["signal"] <= -998, data["meanbatch_slices2"], np.cos(((10.54172039031982422))) )) + (np.cos((data["meanbatch_slices2"]))))) +

                            0.050000*np.tanh(((np.maximum(((np.maximum(((data["maxbatch_slices2"])), ((np.where(np.tanh((data["abs_avgbatch_slices2_msignal"])) > -998, np.sin((data["abs_minbatch_msignal"])), np.sin((data["maxbatch_slices2"])) )))))), ((data["abs_avgbatch_slices2_msignal"])))) * (np.sin((data["abs_avgbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(((data["stdbatch_msignal"]) - (np.where(((data["stdbatch_msignal"]) / 2.0) > -998, data["signal_shift_+1"], ((((((((((data["abs_minbatch_msignal"]) - (data["stdbatch_slices2"]))) / 2.0)) <= (np.sin((data["signal_shift_+1"]))))*1.)) <= (data["rangebatch_slices2_msignal"]))*1.) )))) +

                            0.050000*np.tanh(((data["rangebatch_msignal"]) * (np.sin((((data["abs_avgbatch_slices2"]) + (np.where(((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) * 2.0) <= -998, data["signal"], np.where(data["abs_minbatch_slices2_msignal"] <= -998, data["abs_avgbatch_slices2_msignal"], ((data["signal"]) * 2.0) ) )))))))) +

                            0.050000*np.tanh(((((np.cos((((data["abs_minbatch_slices2_msignal"]) * (data["abs_avgbatch_slices2"]))))) + (((data["signal"]) * ((((-(((((-((((((((data["abs_minbatch_slices2_msignal"]) * 2.0)) * (data["mean_abs_chgbatch_msignal"]))) * 2.0))))) - (data["minbatch_msignal"])))))) / 2.0)))))) / 2.0)) +

                            0.050000*np.tanh(np.cos((np.where(((np.minimum(((np.cos((data["medianbatch_slices2"])))), ((np.sin((np.where((((data["abs_avgbatch_slices2"]) > ((((data["abs_maxbatch"]) + (data["stdbatch_msignal"]))/2.0)))*1.) > -998, data["abs_avgbatch_slices2"], (((((10.0)) / 2.0)) + (np.cos((data["maxtominbatch"])))) ))))))) * 2.0) > -998, data["medianbatch_slices2"], data["meanbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(((((np.where(data["abs_minbatch_msignal"] > -998, data["mean_abs_chgbatch_msignal"], data["abs_avgbatch_msignal"] )) + (((np.cos((data["abs_minbatch_msignal"]))) + (np.minimum(((np.cos((data["abs_minbatch_msignal"])))), ((((data["maxtominbatch_slices2_msignal"]) + ((((data["abs_minbatch_msignal"]) <= (data["abs_minbatch_msignal"]))*1.))))))))))) + (((data["abs_avgbatch_msignal"]) * (np.cos((data["meanbatch_slices2"]))))))) +

                            0.050000*np.tanh(((np.where((((data["abs_minbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0) > -998, (4.0), (6.0) )) * (((np.cos((np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_slices2_msignal"], (6.0) )))) * ((((data["abs_minbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)))))) +

                            0.050000*np.tanh(np.where(data["stdbatch_msignal"] > -998, ((((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) * (((data["stdbatch_msignal"]) + (data["abs_maxbatch_msignal"]))))) + (data["mean_abs_chgbatch_slices2_msignal"])), ((data["maxbatch_slices2_msignal"]) + (data["maxtominbatch"])) )) +

                            0.050000*np.tanh(((((((((((data["abs_maxbatch_slices2_msignal"]) * ((3.0)))) * (np.minimum(((data["abs_maxbatch_slices2"])), ((((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, np.where(((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0) <= -998, data["mean_abs_chgbatch_msignal"], data["abs_maxbatch_slices2"] ), ((((3.0)) > (data["abs_avgbatch_slices2_msignal"]))*1.) )) * (data["abs_maxbatch_slices2"])))))))) * 2.0)) * (data["mean_abs_chgbatch_slices2_msignal"]))) * 2.0)) +

                            0.050000*np.tanh(((((((np.minimum(((((data["abs_minbatch_msignal"]) * 2.0))), ((((np.sin((np.sin((np.maximum(((data["abs_avgbatch_slices2_msignal"])), ((data["abs_avgbatch_slices2_msignal"])))))))) * 2.0))))) * 2.0)) + (np.where(data["abs_avgbatch_slices2_msignal"] > -998, ((np.sin((np.maximum(((data["abs_avgbatch_slices2_msignal"])), ((np.tanh((np.sin((np.maximum(((data["abs_avgbatch_slices2_msignal"])), ((data["signal_shift_+1"]))))))))))))) * 2.0), data["signal_shift_+1"] )))) * 2.0)) +

                            0.050000*np.tanh(((np.minimum(((np.cos((data["mean_abs_chgbatch_msignal"])))), ((((data["abs_maxbatch"]) * (np.where(data["mean_abs_chgbatch_msignal"] > -998, data["mean_abs_chgbatch_msignal"], (-(((((data["mean_abs_chgbatch_slices2_msignal"]) + (data["medianbatch_slices2"]))/2.0)))) ))))))) * (data["abs_maxbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((((data["mean_abs_chgbatch_slices2_msignal"]) - ((((data["mean_abs_chgbatch_msignal"]) <= (data["signal_shift_+1_msignal"]))*1.)))) + (np.tanh((np.where((((data["meanbatch_msignal"]) + (data["meanbatch_msignal"]))/2.0) > -998, data["stdbatch_msignal"], ((((((data["maxtominbatch_msignal"]) > (data["abs_minbatch_msignal"]))*1.)) + (data["stdbatch_slices2"]))/2.0) )))))) +

                            0.050000*np.tanh(((np.sin(((((((data["meanbatch_slices2_msignal"]) + (np.tanh((np.cos((data["signal_shift_-1_msignal"]))))))) + (data["mean_abs_chgbatch_msignal"]))/2.0)))) + (np.cos((np.where(np.minimum(((np.maximum(((data["mean_abs_chgbatch_msignal"])), ((np.minimum(((np.cos((data["maxtominbatch_msignal"])))), ((np.sin((((data["abs_minbatch_slices2_msignal"]) * 2.0))))))))))), ((data["signal_shift_-1_msignal"]))) > -998, data["medianbatch_slices2"], np.cos((data["maxtominbatch_msignal"])) )))))) +

                            0.050000*np.tanh(((np.where(data["stdbatch_msignal"] <= -998, np.cos((np.cos((data["abs_avgbatch_msignal"])))), (5.0) )) * (np.sin((data["abs_avgbatch_msignal"]))))) +

                            0.050000*np.tanh(np.where(np.where(np.sin((((data["rangebatch_slices2_msignal"]) - (data["signal_shift_-1"])))) <= -998, data["abs_maxbatch"], ((data["abs_minbatch_slices2_msignal"]) * 2.0) ) <= -998, np.tanh((data["abs_minbatch_slices2_msignal"])), np.where((-((data["signal_shift_-1"]))) <= -998, data["rangebatch_msignal"], ((data["mean_abs_chgbatch_slices2_msignal"]) * (data["abs_maxbatch"])) ) )) +

                            0.050000*np.tanh(((((data["mean_abs_chgbatch_msignal"]) + (((np.cos((data["stdbatch_slices2_msignal"]))) - (data["meanbatch_msignal"]))))) * 2.0)) +

                            0.050000*np.tanh(((np.minimum((((3.0))), ((np.cos((np.where(np.cos((data["mean_abs_chgbatch_slices2_msignal"])) <= -998, ((np.cos((np.where(data["mean_abs_chgbatch_msignal"] <= -998, data["mean_abs_chgbatch_msignal"], np.cos((data["abs_minbatch_slices2"])) )))) + ((-((np.cos((data["signal_shift_-1_msignal"]))))))), ((data["signal_shift_-1"]) + (np.sin((data["minbatch_slices2_msignal"])))) ))))))) * 2.0)) +

                            0.050000*np.tanh(((data["rangebatch_slices2_msignal"]) * (((np.cos((((data["maxtominbatch_msignal"]) - (np.where(((data["abs_minbatch_msignal"]) * (((((10.0)) <= (np.cos(((10.0)))))*1.))) > -998, (-((data["mean_abs_chgbatch_slices2_msignal"]))), ((data["signal_shift_-1"]) * 2.0) )))))) * 2.0)))) +

                            0.050000*np.tanh(((data["mean_abs_chgbatch_msignal"]) + (np.where(((((data["signal_shift_+1_msignal"]) * 2.0)) + (data["signal_shift_+1"])) > -998, ((data["abs_maxbatch_slices2"]) * (data["abs_minbatch_msignal"])), data["signal_shift_+1_msignal"] )))) +

                            0.050000*np.tanh(np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((((np.where(np.sin((data["abs_avgbatch_msignal"])) <= -998, ((data["rangebatch_slices2"]) - ((((np.tanh((np.where((((((data["medianbatch_slices2"]) / 2.0)) <= (data["stdbatch_msignal"]))*1.) > -998, data["abs_minbatch_slices2_msignal"], (3.0) )))) + (data["abs_minbatch_msignal"]))/2.0))), np.sin((data["abs_avgbatch_msignal"])) )) * 2.0))))) +

                            0.050000*np.tanh(np.cos((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((data["maxtominbatch_msignal"])))))) +

                            0.050000*np.tanh(((data["rangebatch_slices2_msignal"]) * (((((data["maxtominbatch_slices2_msignal"]) * (np.where(np.where(data["medianbatch_msignal"] > -998, data["minbatch_slices2_msignal"], data["maxtominbatch_slices2_msignal"] ) > -998, data["signal_shift_+1"], np.sin((((((data["abs_avgbatch_msignal"]) / 2.0)) - (data["maxbatch_slices2"])))) )))) + (np.where(np.cos((data["minbatch_slices2_msignal"])) > -998, np.cos((data["mean_abs_chgbatch_msignal"])), data["maxbatch_slices2"] )))))) +

                            0.050000*np.tanh(((((np.where(np.sin((data["abs_avgbatch_slices2_msignal"])) > -998, np.where(np.sin((np.minimum(((data["signal_shift_+1"])), ((np.maximum(((np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((data["abs_avgbatch_msignal"]))))), ((data["abs_avgbatch_slices2_msignal"])))))))) > -998, np.sin((data["abs_avgbatch_slices2_msignal"])), ((data["abs_avgbatch_slices2_msignal"]) * 2.0) ), np.sin((((data["signal_shift_+1"]) / 2.0))) )) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((np.where(data["signal_shift_-1_msignal"] <= -998, np.minimum(((data["rangebatch_msignal"])), (((-((((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) * 2.0))))))), data["maxbatch_slices2_msignal"] )) * (np.minimum(((np.minimum(((np.sin((data["abs_avgbatch_slices2"])))), ((np.cos((data["mean_abs_chgbatch_msignal"]))))))), ((np.minimum(((data["mean_abs_chgbatch_msignal"])), (((7.0)))))))))) +

                            0.050000*np.tanh(((np.sin((np.maximum(((data["abs_avgbatch_msignal"])), ((data["abs_avgbatch_slices2"])))))) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((((np.minimum(((np.cos((data["mean_abs_chgbatch_msignal"])))), ((np.maximum(((data["maxtominbatch_slices2"])), ((data["abs_maxbatch_slices2"]))))))) * (data["abs_maxbatch_slices2"])))), ((np.cos((np.maximum(((np.maximum(((data["mean_abs_chgbatch_msignal"])), ((((data["medianbatch_slices2"]) * (np.minimum(((np.cos((data["mean_abs_chgbatch_msignal"])))), ((np.maximum(((data["maxtominbatch_slices2"])), ((data["mean_abs_chgbatch_msignal"]))))))))))))), ((data["mean_abs_chgbatch_msignal"]))))))))) +

                            0.050000*np.tanh(((np.tanh((np.minimum(((data["maxtominbatch"])), ((np.tanh((data["abs_avgbatch_slices2"])))))))) + (((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)))) +

                            0.050000*np.tanh(((((data["mean_abs_chgbatch_slices2_msignal"]) - (np.where((((data["abs_avgbatch_slices2_msignal"]) + (np.cos(((6.17451095581054688)))))/2.0) > -998, data["signal_shift_+1"], (((((np.sin(((((data["minbatch_msignal"]) + (data["minbatch"]))/2.0)))) * 2.0)) > (data["stdbatch_msignal"]))*1.) )))) * 2.0)) +

                            0.050000*np.tanh(((np.minimum(((((((data["mean_abs_chgbatch_msignal"]) + ((6.0)))) + (((((((10.86887836456298828)) + ((-((data["maxtominbatch"])))))) <= ((((data["abs_avgbatch_slices2_msignal"]) > (((data["abs_maxbatch"]) * 2.0)))*1.)))*1.))))), ((np.maximum(((data["abs_avgbatch_slices2_msignal"])), ((np.maximum(((data["abs_avgbatch_slices2"])), ((np.cos((data["mean_abs_chgbatch_slices2_msignal"])))))))))))) * (np.cos((data["maxtominbatch_msignal"]))))) +

                            0.050000*np.tanh(((np.sin((np.where((-((data["medianbatch_slices2"]))) <= -998, data["maxtominbatch_slices2"], ((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0) )))) * (((((np.maximum(((np.cos((data["abs_minbatch_slices2"])))), (((8.0))))) - (data["maxbatch_msignal"]))) + (((data["mean_abs_chgbatch_msignal"]) + (np.where(np.maximum(((data["mean_abs_chgbatch_slices2_msignal"])), ((data["mean_abs_chgbatch_msignal"]))) <= -998, data["signal_shift_+1_msignal"], data["mean_abs_chgbatch_slices2_msignal"] )))))))) +

                            0.050000*np.tanh(((np.sin((np.where(np.where(data["abs_minbatch_msignal"] > -998, np.cos((np.minimum(((data["abs_minbatch_msignal"])), (((-((data["maxtominbatch"])))))))), data["meanbatch_slices2_msignal"] ) > -998, data["abs_avgbatch_slices2_msignal"], data["abs_minbatch_msignal"] )))) * (np.maximum(((((data["abs_avgbatch_slices2_msignal"]) * (np.sin((data["abs_avgbatch_msignal"])))))), ((data["mean_abs_chgbatch_msignal"])))))) +

                            0.050000*np.tanh(((data["rangebatch_slices2_msignal"]) * (np.cos((np.where(((np.tanh((np.cos((np.where(data["mean_abs_chgbatch_slices2"] > -998, np.where(data["meanbatch_slices2"] > -998, data["meanbatch_slices2"], ((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) * 2.0) ), data["signal"] )))))) - (data["abs_minbatch_slices2_msignal"])) > -998, data["meanbatch_slices2"], data["meanbatch_slices2"] )))))) +

                            0.050000*np.tanh(((np.cos((np.where(data["meanbatch_msignal"] > -998, data["meanbatch_msignal"], np.minimum(((data["signal"])), ((np.tanh((((np.minimum(((((np.minimum((((4.0))), ((data["abs_maxbatch_msignal"])))) * (np.cos((np.minimum(((data["medianbatch_slices2"])), ((((data["meanbatch_slices2_msignal"]) + (data["meanbatch_msignal"]))))))))))), ((np.sin((data["signal_shift_+1"])))))) * (data["signal_shift_+1"]))))))) )))) * 2.0)) +

                            0.050000*np.tanh(((data["maxtominbatch_msignal"]) + ((((((data["stdbatch_msignal"]) > (((((data["meanbatch_msignal"]) - (data["abs_avgbatch_msignal"]))) - (data["abs_avgbatch_slices2"]))))*1.)) * 2.0)))) +

                            0.050000*np.tanh(((((np.sin((np.where(((data["stdbatch_slices2_msignal"]) * (np.tanh((data["medianbatch_msignal"])))) <= -998, (((data["rangebatch_slices2"]) + (np.cos((data["mean_abs_chgbatch_msignal"]))))/2.0), ((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0) )))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((np.sin((np.sin((((data["abs_maxbatch_slices2_msignal"]) / 2.0)))))) + (((((((np.cos((data["maxbatch_msignal"]))) * (data["abs_maxbatch_slices2_msignal"]))) + (data["abs_maxbatch_slices2_msignal"]))) + (((((data["maxbatch_msignal"]) * (data["maxtominbatch_slices2_msignal"]))) + (data["maxbatch_msignal"]))))))) +

                            0.050000*np.tanh(np.where(np.where(np.sin((data["abs_maxbatch_msignal"])) <= -998, data["mean_abs_chgbatch_slices2_msignal"], np.cos((np.maximum(((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((data["minbatch_msignal"]))))), ((data["abs_avgbatch_slices2_msignal"]))))) ) > -998, np.cos((np.maximum(((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((data["stdbatch_slices2_msignal"]))))), ((data["meanbatch_slices2"]))))), np.cos((np.sin((data["abs_avgbatch_slices2"])))) )) +

                            0.050000*np.tanh(((np.cos((data["maxtominbatch_msignal"]))) + (np.cos((data["maxtominbatch"]))))) +

                            0.050000*np.tanh(((((np.cos((((np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["minbatch_msignal"], np.where(data["minbatch_msignal"] > -998, data["minbatch_msignal"], (((data["abs_minbatch_slices2"]) + ((((((data["signal"]) * (data["minbatch"]))) > (((((data["minbatch"]) * 2.0)) * 2.0)))*1.)))/2.0) ) )) * 2.0)))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.where(data["medianbatch_msignal"] <= -998, (3.54091382026672363), ((np.tanh((((((data["abs_avgbatch_slices2_msignal"]) * (((np.sin((data["abs_avgbatch_slices2_msignal"]))) * 2.0)))) - (data["mean_abs_chgbatch_slices2_msignal"]))))) * 2.0) )) +

                            0.050000*np.tanh(((np.sin((np.where(np.sin((np.cos((data["meanbatch_slices2"])))) <= -998, np.sin((np.where(data["meanbatch_slices2"] > -998, data["maxtominbatch_msignal"], data["meanbatch_slices2"] ))), data["abs_avgbatch_msignal"] )))) * 2.0)) +

                            0.050000*np.tanh(((data["maxtominbatch_msignal"]) + (((np.cos((((data["maxtominbatch_slices2_msignal"]) - (np.where(data["maxtominbatch_msignal"] <= -998, ((data["abs_minbatch_slices2_msignal"]) / 2.0), data["signal"] )))))) * 2.0)))) +

                            0.050000*np.tanh(((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((np.where(np.sin((((((((data["abs_minbatch_slices2_msignal"]) * (data["abs_minbatch_slices2_msignal"]))) - (data["maxbatch_slices2"]))) * (data["abs_maxbatch_slices2_msignal"])))) <= -998, ((data["maxbatch_slices2"]) * 2.0), np.cos((data["meanbatch_slices2"])) ))))) * (data["abs_maxbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((np.minimum(((np.minimum(((np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, np.sin((np.minimum(((np.where(data["meanbatch_msignal"] <= -998, (13.98129940032958984), ((np.sin((data["abs_avgbatch_slices2_msignal"]))) * (data["abs_avgbatch_msignal"])) ))), ((data["mean_abs_chgbatch_msignal"]))))), ((np.sin((data["abs_avgbatch_slices2_msignal"]))) * (data["abs_avgbatch_msignal"])) ))), ((data["stdbatch_msignal"]))))), ((data["mean_abs_chgbatch_msignal"])))) * 2.0)) +

                            0.050000*np.tanh(((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) * (((((np.cos((data["maxtominbatch_msignal"]))) * (data["mean_abs_chgbatch_slices2_msignal"]))) * (((data["abs_maxbatch"]) + (data["mean_abs_chgbatch_slices2_msignal"]))))))) +

                            0.050000*np.tanh(((((((((np.sin((data["mean_abs_chgbatch_msignal"]))) * 2.0)) + (data["abs_minbatch_slices2_msignal"]))) + (data["abs_minbatch_slices2_msignal"]))) * 2.0)) +

                            0.050000*np.tanh((((((3.0)) + (data["mean_abs_chgbatch_msignal"]))) - ((3.0)))) +

                            0.050000*np.tanh(((np.cos((np.minimum(((data["maxtominbatch_msignal"])), ((((data["signal"]) * (np.minimum(((data["abs_minbatch_slices2_msignal"])), ((((data["abs_avgbatch_slices2_msignal"]) - ((((((((data["abs_avgbatch_slices2_msignal"]) - ((((((((data["stdbatch_msignal"]) * 2.0)) * 2.0)) > ((((data["meanbatch_msignal"]) <= (data["abs_maxbatch"]))*1.)))*1.)))) * 2.0)) > (data["abs_maxbatch"]))*1.)))))))))))))) * 2.0)) +

                            0.050000*np.tanh(np.where(((((((data["maxtominbatch_slices2_msignal"]) * (((data["signal_shift_-1"]) / 2.0)))) * 2.0)) * 2.0) <= -998, data["signal_shift_-1"], ((((data["maxtominbatch_slices2_msignal"]) * (((((data["signal_shift_-1"]) * 2.0)) / 2.0)))) * 2.0) )) +

                            0.050000*np.tanh(np.where(data["abs_avgbatch_msignal"] > -998, np.cos((np.where(data["signal_shift_+1"] <= -998, data["abs_maxbatch"], data["meanbatch_slices2"] ))), data["signal"] )) +

                            0.050000*np.tanh(((((np.minimum(((np.sin((data["abs_avgbatch_msignal"])))), ((data["mean_abs_chgbatch_msignal"])))) * (((((data["abs_maxbatch_slices2_msignal"]) - (((np.sin((data["abs_avgbatch_msignal"]))) * (np.sin((np.sin((data["abs_maxbatch_slices2_msignal"]))))))))) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(((np.where(data["abs_maxbatch"] > -998, (8.97018146514892578), ((data["maxtominbatch"]) * (data["maxtominbatch_msignal"])) )) + (((data["abs_maxbatch"]) * (data["maxtominbatch"]))))) +

                            0.050000*np.tanh(((data["abs_avgbatch_slices2_msignal"]) * (np.sin((np.where(np.cos((((data["abs_avgbatch_slices2_msignal"]) - (data["minbatch_msignal"])))) > -998, np.sin((data["abs_avgbatch_slices2_msignal"])), ((data["meanbatch_slices2_msignal"]) / 2.0) )))))) +

                            0.050000*np.tanh(((np.sin((np.maximum(((data["abs_avgbatch_slices2"])), ((((data["maxtominbatch_msignal"]) + (np.where(data["minbatch_slices2"] <= -998, data["stdbatch_slices2"], (((((data["stdbatch_slices2_msignal"]) > ((((np.maximum(((np.sin((data["stdbatch_slices2_msignal"])))), ((data["maxtominbatch_slices2_msignal"])))) <= (data["stdbatch_slices2_msignal"]))*1.)))*1.)) - (((data["signal"]) * 2.0))) ))))))))) * 2.0)) +

                            0.050000*np.tanh(((((2.40803050994873047)) + (data["stdbatch_msignal"]))/2.0)) +

                            0.050000*np.tanh(((((np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((data["abs_minbatch_slices2"])))) * 2.0)) * (np.cos(((((((data["abs_maxbatch"]) > (np.maximum(((data["abs_avgbatch_slices2"])), ((data["abs_minbatch_msignal"])))))*1.)) - (np.minimum((((((13.53835296630859375)) / 2.0))), ((((data["abs_minbatch_msignal"]) * 2.0))))))))))) +

                            0.050000*np.tanh(((((np.cos((np.where(np.maximum(((data["abs_avgbatch_slices2"])), ((np.cos((data["abs_minbatch_slices2_msignal"]))))) > -998, data["mean_abs_chgbatch_slices2_msignal"], (((data["signal"]) + ((14.09535217285156250)))/2.0) )))) * (data["abs_minbatch_slices2_msignal"]))) * ((((14.09535217285156250)) - (data["mean_abs_chgbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(np.cos((np.where(np.where(data["abs_avgbatch_msignal"] > -998, (0.0), data["maxbatch_slices2"] ) > -998, data["meanbatch_slices2"], np.sin((np.where(np.where(data["abs_minbatch_slices2_msignal"] > -998, np.where(data["signal_shift_+1_msignal"] <= -998, data["mean_abs_chgbatch_msignal"], np.cos((((np.sin((data["maxbatch_slices2"]))) * 2.0))) ), data["meanbatch_slices2"] ) > -998, data["meanbatch_slices2"], data["abs_minbatch_slices2_msignal"] ))) )))) +

                            0.050000*np.tanh(((np.sin((((data["meanbatch_slices2"]) + (np.cos((np.cos(((((((data["medianbatch_slices2"]) * ((((data["signal_shift_+1_msignal"]) > (data["maxtominbatch"]))*1.)))) <= (np.where((((data["maxbatch_slices2_msignal"]) > (data["meanbatch_slices2"]))*1.) <= -998, data["medianbatch_slices2_msignal"], data["abs_avgbatch_slices2_msignal"] )))*1.)))))))))) * 2.0)) +

                            0.050000*np.tanh(((((((data["rangebatch_msignal"]) * ((-((((((data["abs_avgbatch_slices2"]) + ((4.62214088439941406)))) * (np.where(data["maxbatch_slices2_msignal"] > -998, np.where(data["maxtominbatch"] > -998, np.sin((((data["signal_shift_-1"]) - (data["mean_abs_chgbatch_msignal"])))), np.tanh(((1.0))) ), data["stdbatch_slices2"] ))))))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((np.sin((np.sin((data["abs_avgbatch_slices2_msignal"]))))) + (((((np.sin((data["abs_avgbatch_slices2_msignal"]))) + (np.sin((data["abs_avgbatch_slices2_msignal"]))))) + ((((np.sin((data["abs_avgbatch_slices2_msignal"]))) > (np.where(np.cos((((data["meanbatch_slices2"]) * 2.0))) > -998, data["abs_maxbatch"], np.sin((data["abs_avgbatch_slices2_msignal"])) )))*1.)))))) +

                            0.050000*np.tanh(((np.sin((((np.where((-((np.maximum(((data["rangebatch_slices2"])), ((np.cos((data["maxbatch_slices2_msignal"])))))))) > -998, data["mean_abs_chgbatch_slices2_msignal"], data["mean_abs_chgbatch_slices2_msignal"] )) * 2.0)))) * (np.where((10.0) <= -998, ((data["signal_shift_-1"]) * 2.0), np.maximum(((data["rangebatch_slices2"])), ((data["signal_shift_+1"]))) )))) +

                            0.050000*np.tanh(np.sin((((np.maximum(((data["abs_maxbatch_slices2"])), ((((np.where(data["stdbatch_msignal"] > -998, data["minbatch"], data["abs_maxbatch_slices2"] )) / 2.0))))) / 2.0)))) +

                            0.050000*np.tanh(np.minimum(((data["rangebatch_msignal"])), ((((data["stdbatch_slices2"]) - (((np.where(((((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((np.maximum(((np.maximum(((data["stdbatch_slices2_msignal"])), ((((data["medianbatch_slices2"]) - (data["rangebatch_slices2"]))))))), ((data["abs_minbatch_slices2"]))))))) * 2.0)) * 2.0) <= -998, ((data["minbatch"]) - (data["abs_minbatch_msignal"])), ((data["signal_shift_-1"]) * (data["signal_shift_+1"])) )) * 2.0))))))) +

                            0.050000*np.tanh(np.where(data["mean_abs_chgbatch_msignal"] <= -998, (((data["abs_maxbatch_slices2"]) + (((data["abs_avgbatch_msignal"]) * 2.0)))/2.0), ((np.sin((np.sin((data["abs_avgbatch_msignal"]))))) * 2.0) )) +

                            0.050000*np.tanh(((((np.cos((np.where(np.minimum(((data["medianbatch_slices2"])), ((data["maxtominbatch_slices2_msignal"]))) > -998, data["maxtominbatch_msignal"], data["abs_minbatch_slices2"] )))) * 2.0)) + (((np.sin((data["abs_avgbatch_slices2"]))) * (np.maximum(((data["maxbatch_slices2"])), ((np.cos((((data["maxtominbatch_slices2"]) + ((-((((data["maxbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))))))))))))))))) +

                            0.050000*np.tanh(np.where(data["abs_avgbatch_slices2_msignal"] > -998, np.where(np.cos(((8.0))) > -998, ((data["abs_avgbatch_msignal"]) * (np.sin((data["abs_avgbatch_slices2_msignal"])))), data["medianbatch_slices2"] ), data["abs_avgbatch_msignal"] )) +

                            0.050000*np.tanh(((((((((data["signal"]) * (((np.minimum(((np.sin((((data["signal_shift_-1"]) * 2.0))))), ((data["abs_maxbatch"])))) - (data["signal_shift_-1"]))))) * 2.0)) + (np.cos(((-((np.sin(((((((np.sin((data["abs_avgbatch_slices2_msignal"]))) - (data["maxtominbatch_slices2_msignal"]))) > (np.cos((data["maxbatch_slices2"]))))*1.))))))))))) * 2.0)) +

                            0.050000*np.tanh(((((((data["abs_avgbatch_slices2_msignal"]) * 2.0)) - (data["abs_minbatch_slices2"]))) * (((data["mean_abs_chgbatch_msignal"]) - (((((np.minimum(((data["signal"])), ((data["signal_shift_-1"])))) * 2.0)) * (data["signal_shift_-1"]))))))) +

                            0.050000*np.tanh(((np.cos((((np.sin((data["medianbatch_slices2_msignal"]))) - (np.where(np.cos((data["meanbatch_slices2"])) > -998, data["meanbatch_slices2"], ((data["medianbatch_slices2_msignal"]) - (np.sin((data["abs_maxbatch_slices2_msignal"])))) )))))) * 2.0)) +

                            0.050000*np.tanh(((((np.where(np.minimum(((data["medianbatch_slices2"])), ((data["stdbatch_msignal"]))) > -998, np.cos((np.where((((np.tanh((data["medianbatch_slices2_msignal"]))) > (data["maxbatch_msignal"]))*1.) > -998, data["maxtominbatch_slices2"], np.where(data["maxtominbatch_slices2"] > -998, ((data["signal_shift_+1"]) - (data["maxtominbatch_slices2"])), (((data["abs_avgbatch_slices2"]) + (data["medianbatch_slices2"]))/2.0) ) ))), data["abs_avgbatch_slices2"] )) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((data["maxtominbatch_msignal"]) * (((np.where(data["maxtominbatch_slices2"] <= -998, data["meanbatch_slices2"], data["signal_shift_+1"] )) * 2.0)))) +

                            0.050000*np.tanh((((((np.where(((data["abs_maxbatch"]) * (data["meanbatch_slices2"])) > -998, data["stdbatch_slices2_msignal"], np.where(((data["rangebatch_slices2"]) / 2.0) > -998, data["meanbatch_slices2"], data["meanbatch_slices2"] ) )) - (((data["minbatch_slices2"]) / 2.0)))) + (((data["signal_shift_-1"]) * (((data["minbatch_slices2"]) * (np.maximum(((data["abs_maxbatch"])), ((data["abs_minbatch_msignal"])))))))))/2.0)) +

                            0.050000*np.tanh(((np.sin((data["abs_avgbatch_msignal"]))) * (np.where((((data["maxtominbatch_msignal"]) + (data["stdbatch_slices2"]))/2.0) > -998, ((data["stdbatch_slices2_msignal"]) * ((8.0))), ((data["stdbatch_slices2"]) * (data["stdbatch_msignal"])) )))) +

                            0.050000*np.tanh(((np.where((4.60696315765380859) <= -998, np.where(np.sin((np.sin((np.where(np.where(data["medianbatch_msignal"] > -998, data["abs_minbatch_slices2_msignal"], np.maximum(((data["mean_abs_chgbatch_slices2_msignal"])), ((data["abs_minbatch_slices2_msignal"]))) ) > -998, data["medianbatch_slices2_msignal"], data["mean_abs_chgbatch_slices2_msignal"] ))))) > -998, data["mean_abs_chgbatch_slices2_msignal"], data["minbatch_slices2"] ), ((np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)))) * 2.0) )) * 2.0)) +

                            0.050000*np.tanh(((((data["maxbatch_msignal"]) * (((data["maxbatch_msignal"]) * (np.minimum(((((((np.minimum(((data["stdbatch_slices2"])), ((data["abs_minbatch_msignal"])))) * 2.0)) / 2.0))), ((data["abs_minbatch_msignal"])))))))) + (data["stdbatch_slices2"]))) +

                            0.050000*np.tanh(((np.where((((data["rangebatch_slices2"]) + (data["rangebatch_slices2"]))/2.0) > -998, ((((np.cos((((np.tanh((data["meanbatch_slices2_msignal"]))) - (data["signal_shift_+1"]))))) * (np.maximum(((data["mean_abs_chgbatch_msignal"])), ((((np.tanh((data["meanbatch_slices2_msignal"]))) - (data["signal_shift_+1"])))))))) * 2.0), np.cos((np.cos((data["signal_shift_+1"])))) )) * (data["medianbatch_msignal"]))) +

                            0.050000*np.tanh(np.cos((np.where((((((data["mean_abs_chgbatch_slices2_msignal"]) + ((2.0)))) <= (data["medianbatch_slices2"]))*1.) <= -998, data["maxtominbatch_slices2"], np.cos(((-((np.where(((data["abs_avgbatch_slices2_msignal"]) * (data["abs_maxbatch_slices2_msignal"])) <= -998, data["medianbatch_slices2"], ((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)) * (data["stdbatch_msignal"])) )))))) )))) +

                            0.050000*np.tanh(np.cos((np.where(np.cos((data["meanbatch_slices2"])) <= -998, np.sin((np.where(data["maxtominbatch_msignal"] <= -998, ((data["minbatch_slices2_msignal"]) + (data["maxtominbatch"])), ((data["signal_shift_+1"]) * 2.0) ))), ((data["minbatch_slices2_msignal"]) * 2.0) )))) +

                            0.050000*np.tanh(((data["rangebatch_slices2"]) * (np.sin(((((data["stdbatch_slices2_msignal"]) + (np.maximum(((data["abs_maxbatch_slices2"])), ((np.sin((np.tanh((np.cos((((np.where(np.cos((data["medianbatch_msignal"])) > -998, data["abs_maxbatch"], np.sin((np.where(((data["rangebatch_slices2_msignal"]) * 2.0) > -998, data["mean_abs_chgbatch_slices2_msignal"], data["stdbatch_slices2_msignal"] ))) )) * 2.0))))))))))))/2.0)))))) +

                            0.050000*np.tanh((((((10.0)) + (data["mean_abs_chgbatch_slices2_msignal"]))) * (np.where(data["abs_avgbatch_slices2"] <= -998, data["signal_shift_+1_msignal"], np.sin((((np.where(data["abs_avgbatch_slices2"] > -998, data["mean_abs_chgbatch_slices2_msignal"], (((10.0)) + (((np.where(data["signal"] > -998, data["mean_abs_chgbatch_slices2_msignal"], ((data["abs_avgbatch_slices2"]) + (((data["stdbatch_slices2"]) + (data["stdbatch_msignal"])))) )) * 2.0))) )) * 2.0))) )))) +

                            0.050000*np.tanh(np.cos((np.where((((((data["maxtominbatch_slices2_msignal"]) <= (data["meanbatch_slices2"]))*1.)) / 2.0) > -998, data["meanbatch_slices2"], (((data["abs_maxbatch_slices2"]) + (((((data["meanbatch_slices2"]) + (data["medianbatch_slices2"]))) + (((np.sin((np.sin((data["minbatch_slices2_msignal"]))))) + (data["rangebatch_slices2_msignal"]))))))/2.0) )))) +

                            0.050000*np.tanh(np.cos((((np.where(((data["abs_maxbatch_slices2"]) * (np.where(data["abs_minbatch_msignal"] > -998, data["maxtominbatch_msignal"], ((np.where(data["maxtominbatch"] <= -998, data["minbatch_slices2"], data["abs_minbatch_msignal"] )) * (data["abs_maxbatch_slices2_msignal"])) ))) <= -998, data["abs_avgbatch_slices2"], data["abs_minbatch_msignal"] )) * (np.where(data["mean_abs_chgbatch_msignal"] > -998, data["minbatch_slices2_msignal"], data["minbatch_slices2_msignal"] )))))) +

                            0.050000*np.tanh(((np.cos((np.where(data["mean_abs_chgbatch_slices2"] <= -998, np.cos((data["rangebatch_msignal"])), ((np.tanh((((data["abs_minbatch_msignal"]) * (data["signal_shift_+1"]))))) * (data["minbatch"])) )))) * 2.0)) +

                            0.050000*np.tanh(((np.sin((((np.where(np.minimum(((data["minbatch_msignal"])), ((np.where(np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) + (np.tanh((((np.cos((((data["meanbatch_slices2"]) + ((1.0)))))) * 2.0))))))) > -998, (2.0), data["mean_abs_chgbatch_slices2_msignal"] )))) > -998, data["mean_abs_chgbatch_slices2_msignal"], data["abs_avgbatch_slices2_msignal"] )) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(((((np.where((((((data["maxbatch_slices2"]) * 2.0)) <= (((data["maxbatch_slices2"]) * 2.0)))*1.) <= -998, np.sin((data["maxbatch_msignal"])), np.sin((((data["medianbatch_slices2_msignal"]) + (((np.sin((data["maxbatch_slices2"]))) + (np.minimum(((data["rangebatch_slices2_msignal"])), ((data["abs_minbatch_msignal"]))))))))) )) * 2.0)) * 2.0)) +

                            0.050000*np.tanh((((((2.0)) - (np.where(((np.maximum(((np.cos((data["signal_shift_+1"])))), (((-((((np.cos((data["abs_avgbatch_slices2"]))) * (data["meanbatch_slices2"]))))))))) * (data["abs_avgbatch_slices2"])) <= -998, (1.21975803375244141), ((data["minbatch"]) + (data["meanbatch_slices2"])) )))) / 2.0)) +

                            0.050000*np.tanh((((data["abs_maxbatch_slices2_msignal"]) + (((((data["maxbatch_slices2"]) * (((data["maxbatch_slices2"]) * (np.where(((data["signal_shift_+1"]) * 2.0) > -998, (-(((((-((np.where(np.minimum(((((data["abs_minbatch_slices2_msignal"]) * 2.0))), ((data["medianbatch_slices2_msignal"]))) <= -998, data["abs_avgbatch_slices2"], ((data["signal_shift_-1"]) + (data["signal_shift_+1"])) ))))) * (data["abs_minbatch_slices2_msignal"]))))), data["signal_shift_-1"] )))))) * 2.0)))/2.0)) +

                            0.050000*np.tanh(((((((((data["meanbatch_slices2_msignal"]) + (data["maxtominbatch_msignal"]))) * (data["signal_shift_-1"]))) + (np.cos((np.where(data["rangebatch_slices2_msignal"] <= -998, np.where(data["abs_avgbatch_msignal"] <= -998, data["abs_avgbatch_slices2_msignal"], np.cos((((data["mean_abs_chgbatch_msignal"]) * (data["signal_shift_-1"])))) ), np.cos((np.tanh((data["signal_shift_+1"])))) )))))) * 2.0)) +

                            0.050000*np.tanh(((np.sin((((data["stdbatch_slices2"]) * 2.0)))) * (((data["stdbatch_slices2_msignal"]) + (((np.sin((np.maximum(((data["mean_abs_chgbatch_slices2_msignal"])), ((np.tanh((data["abs_minbatch_msignal"])))))))) * (((data["abs_maxbatch_slices2_msignal"]) + (np.maximum(((data["mean_abs_chgbatch_slices2_msignal"])), ((np.tanh((data["mean_abs_chgbatch_slices2_msignal"])))))))))))))) +

                            0.050000*np.tanh(np.cos((np.maximum(((((data["mean_abs_chgbatch_msignal"]) + (np.cos((data["minbatch"])))))), ((np.where(data["stdbatch_msignal"] > -998, np.maximum(((data["meanbatch_slices2"])), ((np.cos((np.minimum(((data["meanbatch_slices2"])), ((((data["stdbatch_slices2"]) - (np.cos((data["stdbatch_msignal"])))))))))))), data["mean_abs_chgbatch_msignal"] ))))))) +

                            0.050000*np.tanh(((data["rangebatch_slices2_msignal"]) - (((data["medianbatch_slices2"]) * (np.minimum(((data["medianbatch_slices2"])), ((data["rangebatch_slices2_msignal"])))))))) +

                            0.050000*np.tanh(((data["maxtominbatch_msignal"]) - (np.where(data["minbatch_msignal"] <= -998, data["maxtominbatch_msignal"], ((np.sin(((-((((((data["minbatch_msignal"]) - ((-((((np.tanh((data["maxtominbatch_slices2_msignal"]))) / 2.0))))))) * 2.0))))))) * ((-(((12.54640769958496094)))))) )))) +

                            0.050000*np.tanh(np.where(np.where(data["maxbatch_slices2_msignal"] <= -998, data["mean_abs_chgbatch_slices2_msignal"], (-(((((data["minbatch"]) + (np.maximum(((data["maxtominbatch_slices2_msignal"])), (((((data["maxbatch_slices2"]) <= (((data["meanbatch_slices2"]) - (data["signal"]))))*1.))))))/2.0)))) ) <= -998, ((data["meanbatch_slices2_msignal"]) + (((data["rangebatch_slices2_msignal"]) * 2.0))), (9.0) )) +

                            0.050000*np.tanh(((((data["rangebatch_slices2_msignal"]) * (np.minimum(((((np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)))) * 2.0))), ((((np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, data["maxbatch_slices2"], np.maximum(((data["medianbatch_msignal"])), (((((-((data["minbatch_slices2"])))) / 2.0)))) )) * 2.0))))))) * 2.0)) +

                            0.050000*np.tanh(np.where(np.sin((np.cos((np.sin((((np.tanh((data["mean_abs_chgbatch_msignal"]))) - (((data["abs_maxbatch"]) - (((np.cos((data["maxtominbatch_slices2"]))) + (data["maxtominbatch_slices2"])))))))))))) > -998, np.cos((data["signal_shift_+1"])), ((np.minimum((((((data["abs_minbatch_slices2_msignal"]) + (data["meanbatch_slices2"]))/2.0))), (((-((data["signal_shift_+1"]))))))) - (data["signal_shift_+1"])) )) +

                            0.050000*np.tanh(((np.cos(((-((data["maxtominbatch_msignal"])))))) * (((np.tanh((((data["rangebatch_slices2"]) - (((data["abs_avgbatch_msignal"]) * 2.0)))))) * (np.where(np.minimum(((data["rangebatch_slices2"])), ((((np.minimum(((np.tanh((data["mean_abs_chgbatch_slices2"])))), (((((data["rangebatch_slices2"]) <= (data["stdbatch_slices2"]))*1.))))) / 2.0)))) > -998, data["rangebatch_slices2"], np.maximum(((data["medianbatch_slices2_msignal"])), ((data["maxtominbatch_msignal"]))) )))))) +

                            0.050000*np.tanh(np.cos((np.maximum(((np.where((((((data["maxtominbatch_msignal"]) - ((-((data["maxbatch_slices2"])))))) > ((((((data["medianbatch_msignal"]) > (np.tanh((data["minbatch"]))))*1.)) / 2.0)))*1.) <= -998, data["stdbatch_msignal"], (-(((-((((data["signal_shift_-1"]) - ((-((((data["signal_shift_-1"]) - (((data["abs_minbatch_slices2_msignal"]) * 2.0)))))))))))))) ))), ((data["medianbatch_slices2"])))))) +

                            0.050000*np.tanh(np.cos((np.where(np.where((((np.sin((data["abs_minbatch_msignal"]))) <= (data["abs_minbatch_msignal"]))*1.) > -998, data["signal_shift_+1"], np.minimum(((data["signal_shift_-1"])), ((data["abs_minbatch_msignal"]))) ) > -998, data["medianbatch_slices2"], data["medianbatch_slices2"] )))) +

                            0.050000*np.tanh(((((((-((((((data["maxtominbatch_msignal"]) * (((data["signal_shift_+1"]) * 2.0)))) * (data["signal_shift_+1"])))))) <= (np.cos((data["maxtominbatch_msignal"]))))*1.)) - ((-((((data["maxtominbatch_msignal"]) * (((data["signal_shift_+1"]) * 2.0))))))))) +

                            0.050000*np.tanh(((data["medianbatch_slices2"]) + (data["maxbatch_msignal"]))) +

                            0.050000*np.tanh(np.minimum((((((np.where(data["stdbatch_slices2"] > -998, data["abs_avgbatch_slices2"], ((np.sin((np.sin((data["abs_minbatch_slices2_msignal"]))))) * 2.0) )) + ((((data["maxtominbatch_slices2_msignal"]) <= (data["mean_abs_chgbatch_slices2_msignal"]))*1.)))/2.0))), ((np.where(data["maxtominbatch_slices2_msignal"] > -998, np.sin((np.sin((data["meanbatch_msignal"])))), ((data["abs_avgbatch_slices2"]) + (data["rangebatch_slices2"])) ))))) +

                            0.050000*np.tanh((((6.81191825866699219)) - (((data["abs_maxbatch"]) * (((data["stdbatch_msignal"]) * (data["mean_abs_chgbatch_slices2"]))))))) +

                            0.050000*np.tanh(((np.sin((np.where(np.sin((np.where(data["abs_avgbatch_slices2"] > -998, data["minbatch_msignal"], ((np.sin((data["stdbatch_slices2"]))) * 2.0) ))) > -998, ((data["minbatch_msignal"]) - (data["stdbatch_slices2"])), ((((-((np.tanh((((data["stdbatch_slices2"]) + (data["stdbatch_slices2"])))))))) <= (data["abs_avgbatch_slices2_msignal"]))*1.) )))) * 2.0)) +

                            0.050000*np.tanh((((8.49771595001220703)) * (np.sin((((data["maxbatch_slices2"]) + (((np.sin((data["abs_minbatch_msignal"]))) + (np.where((2.64441561698913574) > -998, data["meanbatch_msignal"], np.where(data["abs_minbatch_msignal"] > -998, data["maxtominbatch_slices2"], np.cos((data["medianbatch_slices2"])) ) )))))))))) +

                            0.050000*np.tanh(np.cos((np.where(np.cos((data["minbatch"])) > -998, data["meanbatch_slices2"], np.where(data["mean_abs_chgbatch_slices2"] > -998, ((data["signal_shift_-1_msignal"]) * 2.0), np.sin((np.cos((data["meanbatch_slices2_msignal"])))) ) )))) +

                            0.050000*np.tanh(np.cos((np.where((-((((((-((data["abs_minbatch_slices2"])))) > (np.cos((np.where(((((np.sin((((np.tanh((data["maxbatch_slices2"]))) + (data["stdbatch_slices2_msignal"]))))) * 2.0)) / 2.0) > -998, data["rangebatch_slices2_msignal"], data["abs_maxbatch_msignal"] )))))*1.)))) > -998, (-((data["meanbatch_slices2"]))), (((data["stdbatch_slices2_msignal"]) + (((data["signal"]) - (data["rangebatch_slices2_msignal"]))))/2.0) )))) +

                            0.050000*np.tanh(np.where(data["signal_shift_+1"] <= -998, np.where(np.where(data["medianbatch_msignal"] > -998, np.cos((data["signal_shift_+1"])), data["medianbatch_msignal"] ) > -998, np.maximum(((data["medianbatch_msignal"])), ((data["abs_avgbatch_slices2"]))), np.tanh((data["minbatch"])) ), data["abs_avgbatch_slices2"] )) +

                            0.050000*np.tanh(((((((((((data["signal_shift_+1"]) + (np.tanh(((9.14462471008300781)))))) * 2.0)) * 2.0)) * 2.0)) * (np.where(np.tanh((data["signal_shift_+1"])) > -998, np.tanh((np.tanh(((9.14462471008300781))))), (((9.14462471008300781)) + (np.tanh(((((((((np.tanh(((9.14462471008300781)))) / 2.0)) > (data["rangebatch_slices2_msignal"]))*1.)) + (data["maxbatch_slices2_msignal"])))))) )))) +

                            0.050000*np.tanh((-((((((data["signal_shift_+1"]) * (((data["maxbatch_slices2_msignal"]) + (np.where(data["signal_shift_+1"] <= -998, data["rangebatch_slices2_msignal"], (((-((data["minbatch_msignal"])))) + ((((((data["meanbatch_slices2_msignal"]) > ((((data["mean_abs_chgbatch_slices2"]) > ((14.58663940429687500)))*1.)))*1.)) - ((-((data["rangebatch_slices2_msignal"]))))))) )))))) * 2.0))))) +

                            0.050000*np.tanh(np.tanh((np.sin((np.where(((np.where((((data["abs_avgbatch_msignal"]) + ((((data["abs_maxbatch_slices2"]) <= (data["meanbatch_slices2_msignal"]))*1.)))/2.0) > -998, data["signal_shift_-1_msignal"], (-(((((((np.where(data["maxtominbatch_msignal"] > -998, data["meanbatch_msignal"], data["meanbatch_slices2_msignal"] )) * 2.0)) + (data["meanbatch_msignal"]))/2.0)))) )) * 2.0) > -998, data["meanbatch_msignal"], data["meanbatch_msignal"] )))))) +

                            0.050000*np.tanh(np.where(np.sin((((data["abs_maxbatch_slices2"]) - (data["abs_maxbatch_slices2"])))) <= -998, data["meanbatch_msignal"], (((data["medianbatch_msignal"]) <= (np.sin((data["maxbatch_slices2"]))))*1.) )) +

                            0.050000*np.tanh(np.sin((np.where(np.cos((data["signal_shift_-1"])) <= -998, np.where(data["abs_avgbatch_msignal"] <= -998, data["abs_maxbatch_slices2_msignal"], (((((((0.0)) <= ((((data["minbatch_slices2_msignal"]) <= (data["rangebatch_slices2_msignal"]))*1.)))*1.)) <= (data["abs_minbatch_slices2"]))*1.) ), np.minimum(((data["stdbatch_slices2_msignal"])), (((((((((data["abs_maxbatch_slices2"]) > (np.sin(((9.0)))))*1.)) + (data["minbatch_msignal"]))) * 2.0)))) )))) +

                            0.050000*np.tanh(data["abs_maxbatch_slices2"]) +

                            0.050000*np.tanh((((((7.0)) - (data["abs_maxbatch"]))) - (np.where(data["rangebatch_slices2"] <= -998, ((np.tanh((np.sin(((((7.0)) - (np.where(data["abs_minbatch_slices2"] <= -998, ((data["abs_minbatch_slices2"]) / 2.0), data["abs_maxbatch_slices2"] )))))))) / 2.0), np.sin((data["abs_maxbatch"])) )))) +

                            0.050000*np.tanh(((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) + (np.where(data["rangebatch_slices2_msignal"] > -998, ((data["signal_shift_-1_msignal"]) * (data["abs_avgbatch_slices2"])), np.sin((((np.sin((np.minimum(((data["signal_shift_-1_msignal"])), ((data["abs_maxbatch_slices2"])))))) * ((10.54265403747558594))))) )))) +

                            0.050000*np.tanh(((np.cos((data["meanbatch_slices2"]))) * (((data["minbatch_msignal"]) * (np.where(np.minimum(((data["stdbatch_msignal"])), ((data["signal_shift_-1_msignal"]))) <= -998, np.minimum(((data["meanbatch_slices2"])), ((data["signal_shift_-1_msignal"]))), np.where(data["minbatch_slices2_msignal"] <= -998, data["rangebatch_slices2"], np.tanh((((data["minbatch_slices2_msignal"]) + (data["stdbatch_msignal"])))) ) )))))) +

                            0.050000*np.tanh(((np.sin((np.where(np.where(data["medianbatch_slices2_msignal"] <= -998, data["minbatch_msignal"], data["meanbatch_slices2"] ) > -998, data["abs_avgbatch_slices2_msignal"], np.tanh((((data["meanbatch_slices2"]) + (data["signal_shift_-1"])))) )))) * (np.maximum(((data["maxtominbatch_slices2_msignal"])), ((data["medianbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(((np.sin((np.cos((((((data["abs_minbatch_slices2_msignal"]) * (np.where(np.minimum(((((data["abs_minbatch_slices2_msignal"]) * (np.where((14.50428867340087891) > -998, data["abs_avgbatch_slices2_msignal"], data["medianbatch_slices2"] ))))), ((data["meanbatch_msignal"]))) > -998, data["signal_shift_-1"], data["abs_maxbatch"] )))) - (data["stdbatch_slices2"]))))))) * 2.0)) +

                            0.050000*np.tanh(((np.cos((np.where(data["maxbatch_slices2_msignal"] <= -998, data["medianbatch_slices2_msignal"], np.tanh((data["maxbatch_slices2"])) )))) + (((data["abs_maxbatch_msignal"]) * (np.sin((data["medianbatch_msignal"]))))))) +

                            0.050000*np.tanh(((((((np.tanh((np.sin((data["meanbatch_slices2_msignal"]))))) + (((np.cos((np.maximum(((data["stdbatch_slices2_msignal"])), ((np.maximum(((data["abs_minbatch_slices2"])), ((data["stdbatch_msignal"]))))))))) * 2.0)))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((np.where((-((np.where(data["abs_minbatch_slices2_msignal"] > -998, data["abs_minbatch_slices2_msignal"], ((((np.sin((data["medianbatch_msignal"]))) * 2.0)) * 2.0) )))) <= -998, ((data["abs_minbatch_slices2_msignal"]) / 2.0), ((np.where((((data["signal_shift_-1"]) <= ((((data["medianbatch_msignal"]) <= (data["signal_shift_-1"]))*1.)))*1.) > -998, np.sin((data["medianbatch_msignal"])), np.sin((data["medianbatch_msignal"])) )) * 2.0) )) - (data["signal_shift_+1"]))) +

                            0.050000*np.tanh(((data["abs_avgbatch_slices2_msignal"]) - (((np.where(data["maxtominbatch_slices2_msignal"] > -998, ((np.where((1.0) > -998, data["signal"], data["mean_abs_chgbatch_slices2"] )) * 2.0), np.where((-((data["rangebatch_msignal"]))) <= -998, data["maxbatch_slices2"], ((((data["abs_maxbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2_msignal"]))) * (np.tanh(((-((data["maxtominbatch_slices2_msignal"]))))))) ) )) * (((data["maxtominbatch_msignal"]) + (data["abs_maxbatch_slices2_msignal"]))))))) +

                            0.050000*np.tanh(data["maxbatch_slices2"]) +

                            0.050000*np.tanh(((((data["abs_minbatch_slices2"]) + (data["abs_minbatch_msignal"]))) * (((np.maximum(((data["abs_minbatch_msignal"])), ((np.maximum(((data["signal"])), ((data["abs_avgbatch_slices2_msignal"]))))))) * (data["meanbatch_slices2"]))))) +

                            0.050000*np.tanh(np.sin((np.sin((((((data["medianbatch_slices2_msignal"]) - (np.minimum(((data["maxtominbatch_slices2_msignal"])), ((np.minimum(((data["maxtominbatch_slices2_msignal"])), ((np.minimum(((np.sin((((np.cos((np.sin((((data["abs_maxbatch_msignal"]) + ((13.72685527801513672)))))))) + (data["mean_abs_chgbatch_slices2_msignal"])))))), ((np.sin((np.sin((data["signal_shift_+1_msignal"])))))))))))))))) * 2.0)))))) +

                            0.050000*np.tanh(np.cos((((np.where(((data["maxbatch_slices2"]) + (np.sin((np.cos(((-((np.cos((data["meanbatch_slices2"]))))))))))) <= -998, (((data["minbatch_slices2_msignal"]) <= (((np.maximum(((data["abs_avgbatch_slices2"])), ((np.cos((data["meanbatch_slices2"])))))) + (((data["medianbatch_slices2"]) * 2.0)))))*1.), np.minimum(((((data["meanbatch_slices2"]) / 2.0))), (((-((data["abs_minbatch_msignal"])))))) )) * 2.0)))) +

                            0.050000*np.tanh(((((data["stdbatch_slices2_msignal"]) * 2.0)) + (np.tanh((((((((np.maximum(((data["abs_avgbatch_msignal"])), ((np.sin((np.maximum(((data["abs_avgbatch_msignal"])), ((data["abs_avgbatch_slices2"]))))))))) + (data["signal"]))/2.0)) + (data["stdbatch_slices2_msignal"]))/2.0)))))) +

                            0.050000*np.tanh(np.cos((((data["maxtominbatch_msignal"]) + (np.where(data["signal"] <= -998, np.where((((((data["medianbatch_slices2"]) <= (((data["abs_minbatch_msignal"]) - ((-((np.maximum(((data["signal_shift_+1"])), ((data["minbatch_slices2"]))))))))))*1.)) - ((-((np.cos((data["medianbatch_msignal"]))))))) <= -998, (((((data["abs_minbatch_slices2_msignal"]) + (data["medianbatch_msignal"]))/2.0)) / 2.0), data["minbatch_slices2"] ), data["medianbatch_msignal"] )))))) +

                            0.050000*np.tanh(np.maximum(((((data["meanbatch_msignal"]) * ((((((data["meanbatch_slices2_msignal"]) + (((data["meanbatch_msignal"]) * ((((((data["maxtominbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"]))/2.0)) / 2.0)))))/2.0)) / 2.0))))), ((np.cos((((data["signal_shift_-1"]) - (np.where(data["signal_shift_-1"] <= -998, data["rangebatch_slices2_msignal"], np.tanh((data["maxtominbatch_slices2_msignal"])) ))))))))) +

                            0.050000*np.tanh(np.cos((np.where((0.56028139591217041) > -998, ((data["stdbatch_msignal"]) - (((np.tanh((data["rangebatch_slices2_msignal"]))) - (data["rangebatch_slices2_msignal"])))), ((data["stdbatch_slices2"]) * (((np.cos((((((((np.tanh((data["signal_shift_-1_msignal"]))) * 2.0)) * (data["stdbatch_msignal"]))) - (data["medianbatch_slices2"]))))) - (((data["meanbatch_slices2_msignal"]) * (data["minbatch"])))))) )))) +

                            0.050000*np.tanh(np.where(data["stdbatch_slices2"] > -998, ((data["maxbatch_slices2_msignal"]) - (((((data["maxbatch_slices2"]) * ((((((4.00822257995605469)) * (data["signal_shift_-1"]))) * (data["signal"]))))) * (data["maxbatch_msignal"])))), ((((data["meanbatch_slices2_msignal"]) * (((data["abs_maxbatch_slices2"]) * (np.tanh((np.cos((data["abs_maxbatch_slices2"]))))))))) + (((data["abs_maxbatch_slices2"]) - (data["stdbatch_slices2"])))) )) +

                            0.050000*np.tanh(np.where(np.where((-(((((np.sin((np.sin((data["meanbatch_msignal"]))))) > (data["signal_shift_+1_msignal"]))*1.)))) <= -998, data["meanbatch_msignal"], ((((((np.sin((data["meanbatch_msignal"]))) - (np.sin((data["meanbatch_slices2"]))))) * 2.0)) * 2.0) ) <= -998, np.sin((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, data["signal_shift_+1"], data["signal_shift_+1"] ))), ((np.sin((data["meanbatch_msignal"]))) * 2.0) )) +

                            0.050000*np.tanh(((data["rangebatch_slices2_msignal"]) + (np.where(((data["abs_minbatch_msignal"]) * (((data["maxtominbatch_slices2_msignal"]) - (data["meanbatch_msignal"])))) > -998, ((((data["maxtominbatch_slices2_msignal"]) - (data["meanbatch_msignal"]))) * 2.0), ((data["rangebatch_slices2_msignal"]) - (np.where(data["meanbatch_msignal"] > -998, ((((data["maxtominbatch_slices2_msignal"]) - (data["meanbatch_msignal"]))) * 2.0), ((data["meanbatch_msignal"]) - (data["maxtominbatch_slices2_msignal"])) ))) )))) +

                            0.050000*np.tanh((((((data["meanbatch_msignal"]) + (np.where(((data["maxbatch_slices2"]) * 2.0) <= -998, data["meanbatch_slices2_msignal"], data["medianbatch_slices2"] )))) + (((((((data["maxtominbatch_slices2_msignal"]) * (((data["mean_abs_chgbatch_slices2_msignal"]) - (data["meanbatch_slices2_msignal"]))))) * (data["minbatch"]))) * 2.0)))/2.0)) +

                            0.050000*np.tanh(data["maxbatch_slices2_msignal"]) +

                            0.050000*np.tanh((((np.maximum(((data["maxbatch_slices2"])), ((np.where(np.sin((np.where(data["signal_shift_-1"] <= -998, (2.78504920005798340), (((np.where(data["maxbatch_slices2"] > -998, data["rangebatch_slices2_msignal"], np.tanh(((12.91273593902587891))) )) + (data["rangebatch_slices2_msignal"]))/2.0) ))) > -998, data["stdbatch_slices2"], data["maxbatch_slices2"] ))))) + (np.maximum(((data["maxbatch_slices2_msignal"])), ((data["mean_abs_chgbatch_slices2"])))))/2.0)) +

                            0.050000*np.tanh(((data["abs_avgbatch_msignal"]) - (np.where(data["stdbatch_slices2_msignal"] > -998, data["minbatch_slices2"], (-((np.minimum(((data["maxbatch_slices2"])), ((np.minimum(((data["abs_minbatch_slices2"])), ((data["signal"]))))))))) )))) +

                            0.050000*np.tanh(((data["medianbatch_slices2"]) * (np.where(np.maximum(((((data["abs_minbatch_slices2_msignal"]) * (data["medianbatch_slices2"])))), ((data["medianbatch_slices2_msignal"]))) <= -998, data["maxtominbatch_slices2_msignal"], data["maxtominbatch_msignal"] )))) +

                            0.050000*np.tanh(((((np.sin((((np.maximum(((data["abs_maxbatch_slices2"])), (((-((data["meanbatch_slices2"]))))))) - (data["minbatch_msignal"]))))) * (np.maximum(((np.sin(((-((data["meanbatch_slices2"]))))))), ((data["maxbatch_msignal"])))))) * 2.0)) +

                            0.050000*np.tanh(((np.sin((((data["meanbatch_msignal"]) - (np.tanh(((((data["maxtominbatch"]) + ((((data["stdbatch_msignal"]) <= (((data["signal_shift_+1_msignal"]) - (((np.cos((((data["rangebatch_msignal"]) - (np.sin((data["signal_shift_+1"]))))))) * 2.0)))))*1.)))/2.0)))))))) * 2.0)) +

                            0.050000*np.tanh(np.where(np.where((((data["medianbatch_msignal"]) > (data["signal_shift_-1"]))*1.) > -998, data["maxbatch_msignal"], data["stdbatch_slices2_msignal"] ) > -998, np.where(np.maximum(((data["maxbatch_msignal"])), ((data["mean_abs_chgbatch_slices2"]))) <= -998, np.where(data["abs_maxbatch_slices2"] <= -998, (0.0), data["rangebatch_slices2_msignal"] ), data["abs_maxbatch_msignal"] ), np.where(data["rangebatch_msignal"] <= -998, np.cos((data["signal_shift_-1"])), data["maxbatch_msignal"] ) )) +

                            0.050000*np.tanh(np.cos((((data["maxtominbatch_msignal"]) - (np.where((((np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, data["abs_maxbatch"], data["maxtominbatch_msignal"] )) + (((((((data["maxtominbatch_msignal"]) > (np.sin((data["abs_avgbatch_slices2"]))))*1.)) <= (data["abs_avgbatch_slices2"]))*1.)))/2.0) > -998, data["signal_shift_+1"], np.maximum((((((data["abs_avgbatch_msignal"]) <= (data["maxbatch_msignal"]))*1.))), ((data["medianbatch_slices2"]))) )))))) +

                            0.050000*np.tanh(np.cos((((((data["abs_maxbatch_slices2"]) / 2.0)) * (np.minimum(((data["signal_shift_+1"])), (((((np.cos(((((data["medianbatch_slices2"]) <= (np.cos(((((data["rangebatch_slices2"]) <= (data["abs_maxbatch_slices2"]))*1.)))))*1.)))) + (np.cos((data["medianbatch_msignal"]))))/2.0))))))))) +

                            0.050000*np.tanh(data["maxbatch_slices2"]) +

                            0.050000*np.tanh(np.cos((((data["rangebatch_slices2_msignal"]) - (np.where(((data["rangebatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"])) > -998, (-((data["abs_minbatch_slices2_msignal"]))), data["maxtominbatch_msignal"] )))))) +

                            0.050000*np.tanh(np.where(data["minbatch"] <= -998, np.cos((np.tanh((((data["maxtominbatch_slices2_msignal"]) * (data["signal_shift_+1_msignal"])))))), ((((((4.0)) + (np.tanh((np.tanh((data["minbatch"]))))))/2.0)) * (np.cos((((data["maxtominbatch_slices2_msignal"]) + (data["medianbatch_slices2_msignal"])))))) )) +

                            0.050000*np.tanh(((np.maximum(((data["signal_shift_+1_msignal"])), ((np.where(((data["rangebatch_slices2_msignal"]) + (data["meanbatch_slices2"])) <= -998, ((data["rangebatch_slices2_msignal"]) + (data["abs_maxbatch_msignal"])), data["rangebatch_slices2_msignal"] ))))) * (np.maximum((((0.0))), ((data["rangebatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(((data["maxtominbatch_msignal"]) * (np.sin((np.minimum(((data["minbatch_slices2_msignal"])), ((np.minimum(((data["abs_maxbatch_slices2_msignal"])), (((-((np.maximum(((np.minimum(((np.minimum(((data["minbatch_slices2_msignal"])), (((-((np.minimum(((data["medianbatch_slices2"])), ((np.minimum(((data["abs_avgbatch_slices2"])), ((data["maxtominbatch_msignal"])))))))))))))), (((-((np.maximum(((data["maxtominbatch_msignal"])), ((data["abs_avgbatch_slices2"]))))))))))), ((data["abs_avgbatch_slices2"]))))))))))))))))) +

                            0.050000*np.tanh(np.where(data["meanbatch_slices2_msignal"] > -998, np.sin((((((((-((np.minimum(((data["minbatch_msignal"])), ((data["stdbatch_slices2_msignal"]))))))) * (data["maxbatch_msignal"]))) + (((data["mean_abs_chgbatch_slices2_msignal"]) - ((((((7.0)) / 2.0)) + ((((data["maxbatch_msignal"]) <= (data["abs_maxbatch_slices2_msignal"]))*1.)))))))/2.0))), data["mean_abs_chgbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(np.maximum((((((((data["maxbatch_slices2_msignal"]) * ((7.0)))) > ((((data["abs_avgbatch_slices2_msignal"]) + (data["stdbatch_slices2"]))/2.0)))*1.))), ((data["maxtominbatch_slices2"])))) +

                            0.050000*np.tanh((((((np.sin((np.tanh((data["mean_abs_chgbatch_msignal"]))))) <= (((((((((data["signal_shift_-1"]) <= (data["signal_shift_-1"]))*1.)) <= (data["abs_minbatch_slices2"]))*1.)) * 2.0)))*1.)) / 2.0)) +

                            0.050000*np.tanh(((data["signal_shift_+1_msignal"]) * (((((data["mean_abs_chgbatch_msignal"]) * (data["abs_maxbatch"]))) + (np.cos((data["abs_maxbatch_slices2"]))))))))  

    

    def GP_class_3(self,data):

        return self.Output( -2.012666 +

                            0.050000*np.tanh(((np.cos((data["abs_maxbatch_msignal"]))) + (((data["abs_maxbatch_msignal"]) * (np.cos((np.where(np.where(((data["abs_maxbatch_msignal"]) * (np.cos((data["rangebatch_slices2"])))) > -998, data["mean_abs_chgbatch_msignal"], ((data["stdbatch_msignal"]) * (np.maximum((((3.29916071891784668))), ((data["mean_abs_chgbatch_slices2_msignal"]))))) ) > -998, data["stdbatch_msignal"], (-(((((-((data["stdbatch_msignal"])))) * 2.0)))) )))))))) +

                            0.050000*np.tanh(((((np.where((-(((14.68631076812744141)))) > -998, np.where(data["minbatch"] > -998, data["abs_avgbatch_slices2_msignal"], np.where(np.minimum(((data["abs_minbatch_msignal"])), ((data["signal"]))) <= -998, ((data["abs_maxbatch_slices2"]) + ((2.92223882675170898))), data["abs_avgbatch_slices2_msignal"] ) ), data["signal"] )) * (np.where(np.tanh((data["signal"])) <= -998, data["rangebatch_slices2_msignal"], data["signal"] )))) * 2.0)) +

                            0.050000*np.tanh(((((data["abs_avgbatch_msignal"]) * (((data["signal"]) * ((4.58320951461791992)))))) - ((((((((data["abs_avgbatch_msignal"]) + (data["medianbatch_slices2"]))) > (np.where(data["minbatch_msignal"] <= -998, (4.58320951461791992), data["signal"] )))*1.)) + (data["signal"]))))) +

                            0.050000*np.tanh(((((data["abs_avgbatch_msignal"]) * (np.where(np.minimum(((data["signal"])), ((((data["abs_avgbatch_msignal"]) * (data["maxtominbatch_msignal"]))))) > -998, data["signal"], np.minimum(((data["abs_avgbatch_slices2"])), ((np.where(data["signal_shift_-1"] <= -998, ((np.minimum(((data["abs_minbatch_slices2_msignal"])), ((data["mean_abs_chgbatch_msignal"])))) * (data["signal"])), (-((data["signal"]))) )))) )))) * 2.0)) +

                            0.050000*np.tanh((((((3.0)) * (data["signal"]))) * (np.cos(((((((np.minimum(((((np.cos((np.tanh((data["signal"]))))) * (data["signal"])))), (((3.0))))) <= (((data["signal_shift_+1"]) * ((((3.0)) * (data["signal"]))))))*1.)) + (np.sin((data["abs_maxbatch"]))))))))) +

                            0.050000*np.tanh(((((data["signal"]) * (((data["abs_avgbatch_msignal"]) * (np.maximum((((((9.0)) * 2.0))), ((((np.maximum(((((data["abs_minbatch_msignal"]) - (data["abs_maxbatch_slices2"])))), ((data["abs_avgbatch_msignal"])))) * 2.0))))))))) - (((data["signal"]) + (np.maximum(((np.cos((data["signal_shift_+1"])))), ((data["medianbatch_slices2"])))))))) +

                            0.050000*np.tanh(np.where(data["abs_avgbatch_msignal"] <= -998, data["abs_avgbatch_msignal"], ((data["signal"]) * (((np.cos((data["stdbatch_slices2_msignal"]))) + (((data["stdbatch_msignal"]) + (np.cos((data["stdbatch_msignal"])))))))) )) +

                            0.050000*np.tanh(((data["minbatch_slices2"]) * (((data["abs_minbatch_slices2_msignal"]) - ((((((((data["meanbatch_slices2"]) * ((((data["abs_minbatch_slices2_msignal"]) + (np.sin(((((0.0)) * (data["abs_minbatch_msignal"]))))))/2.0)))) <= (np.cos((data["abs_avgbatch_msignal"]))))*1.)) + (((data["signal"]) * ((((data["mean_abs_chgbatch_slices2_msignal"]) + ((-((data["meanbatch_slices2"])))))/2.0)))))))))) +

                            0.050000*np.tanh(((((np.sin((np.where(data["abs_maxbatch_msignal"] <= -998, ((np.where(data["maxbatch_slices2"] > -998, np.cos((data["mean_abs_chgbatch_slices2_msignal"])), data["signal_shift_-1"] )) * (((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) / 2.0))), data["signal"] )))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((((np.cos(((((data["stdbatch_msignal"]) + ((((data["stdbatch_msignal"]) + (((data["stdbatch_msignal"]) * 2.0)))/2.0)))/2.0)))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((np.where(data["minbatch_msignal"] <= -998, data["maxtominbatch_msignal"], np.minimum(((((data["signal"]) - (((np.where(np.sin((data["minbatch_msignal"])) > -998, data["abs_minbatch_msignal"], data["stdbatch_msignal"] )) / 2.0))))), ((((((((np.minimum(((data["signal"])), ((np.sin((data["signal"])))))) * 2.0)) * 2.0)) - (data["abs_minbatch_slices2"]))))) )) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((np.sin((np.maximum(((data["signal"])), ((np.maximum(((((((data["stdbatch_slices2"]) / 2.0)) / 2.0))), ((np.sin((data["signal"])))))))))))), ((data["signal"])))) +

                            0.050000*np.tanh(((((((np.where(np.tanh((data["signal"])) > -998, np.cos((data["stdbatch_slices2"])), (-((data["abs_minbatch_slices2"]))) )) * 2.0)) * (data["maxbatch_msignal"]))) * (np.where(data["signal"] > -998, data["signal"], data["maxtominbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(((((np.minimum(((np.cos((np.where(data["rangebatch_msignal"] > -998, data["stdbatch_slices2_msignal"], ((np.minimum(((((np.sin((data["maxbatch_slices2"]))) * 2.0))), ((np.cos((((((data["rangebatch_slices2"]) / 2.0)) / 2.0))))))) * 2.0) ))))), ((np.cos((data["abs_minbatch_msignal"])))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((data["signal"]) * (((np.where(((data["signal"]) * (np.maximum(((np.sin((((data["abs_avgbatch_slices2_msignal"]) * (data["abs_avgbatch_slices2_msignal"])))))), ((data["maxtominbatch"]))))) > -998, ((np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], data["signal_shift_-1"] )) * 2.0), ((data["mean_abs_chgbatch_slices2"]) / 2.0) )) * 2.0)))) +

                            0.050000*np.tanh(((((((((np.minimum(((((((np.minimum(((data["signal"])), ((np.cos((np.minimum(((data["signal"])), ((np.minimum(((data["abs_minbatch_slices2_msignal"])), ((np.cos((np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((np.cos(((-((data["abs_minbatch_slices2_msignal"])))))))))))))))))))))) * 2.0)) * 2.0))), ((((data["rangebatch_slices2_msignal"]) * 2.0))))) * 2.0)) * 2.0)) * 2.0)) - (np.cos((data["signal"]))))) +

                            0.050000*np.tanh(((((data["signal"]) * (((np.maximum(((data["mean_abs_chgbatch_msignal"])), ((np.cos((((data["signal"]) - (np.cos(((((data["signal"]) + (np.minimum(((((data["signal"]) * (data["abs_maxbatch"])))), ((((data["signal"]) * 2.0))))))/2.0))))))))))) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(((np.minimum(((((np.minimum(((((((((np.minimum(((((np.cos((np.maximum(((data["signal"])), ((data["abs_maxbatch"])))))) - (np.sin((data["maxtominbatch_msignal"])))))), ((np.sin((data["maxtominbatch_msignal"])))))) - (np.sin((data["maxtominbatch_msignal"]))))) * 2.0)) - (np.sin((data["maxtominbatch_msignal"])))))), ((np.sin((data["signal"])))))) * 2.0))), ((data["signal"])))) * 2.0)) +

                            0.050000*np.tanh(((np.where(((((data["maxbatch_slices2"]) - (data["maxtominbatch_slices2_msignal"]))) - (data["maxtominbatch"])) <= -998, np.maximum(((data["signal"])), ((data["rangebatch_slices2_msignal"]))), data["abs_minbatch_msignal"] )) * (((((data["signal"]) - (data["mean_abs_chgbatch_slices2"]))) - (data["abs_maxbatch_slices2"]))))) +

                            0.050000*np.tanh(((((np.where(data["maxtominbatch_slices2_msignal"] <= -998, np.where(data["maxtominbatch_slices2_msignal"] <= -998, ((np.where(data["maxtominbatch_slices2_msignal"] <= -998, (((data["abs_avgbatch_msignal"]) <= (((((np.where(data["abs_minbatch_slices2_msignal"] <= -998, data["abs_minbatch_slices2_msignal"], np.cos((data["abs_minbatch_slices2_msignal"])) )) * 2.0)) * 2.0)))*1.), np.cos((data["abs_minbatch_slices2_msignal"])) )) * 2.0), np.cos((data["abs_minbatch_slices2_msignal"])) ), np.cos((data["abs_minbatch_slices2_msignal"])) )) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) * (np.sin((data["maxtominbatch_msignal"]))))) +

                            0.050000*np.tanh(((np.sin((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((np.minimum(((((np.where(data["signal"] > -998, data["minbatch_msignal"], data["maxbatch_msignal"] )) + (data["mean_abs_chgbatch_slices2_msignal"])))), ((((np.sin((data["minbatch_msignal"]))) * 2.0)))))))))) * 2.0)) +

                            0.050000*np.tanh(((data["medianbatch_slices2"]) + (((data["medianbatch_slices2"]) + (((data["rangebatch_msignal"]) * (np.minimum(((data["signal"])), (((((((((data["stdbatch_slices2_msignal"]) * (np.cos((((data["stdbatch_slices2_msignal"]) / 2.0)))))) <= (np.sin((data["rangebatch_msignal"]))))*1.)) + (((data["rangebatch_msignal"]) * (np.minimum(((np.cos((data["abs_avgbatch_msignal"])))), ((np.cos((data["signal"]))))))))))))))))))) +

                            0.050000*np.tanh(((np.where(data["medianbatch_slices2"] > -998, data["rangebatch_slices2"], ((data["abs_avgbatch_slices2"]) * (((np.sin((data["abs_avgbatch_slices2"]))) * 2.0))) )) * (((((data["signal"]) * (((np.sin((data["abs_avgbatch_slices2"]))) * 2.0)))) * 2.0)))) +

                            0.050000*np.tanh(((((data["rangebatch_slices2_msignal"]) + (((data["stdbatch_slices2_msignal"]) - (np.maximum(((data["maxtominbatch_msignal"])), ((((np.maximum(((data["mean_abs_chgbatch_msignal"])), ((data["medianbatch_slices2_msignal"])))) * 2.0))))))))) + (((data["stdbatch_slices2_msignal"]) - (((data["mean_abs_chgbatch_msignal"]) * (data["maxtominbatch_msignal"]))))))) +

                            0.050000*np.tanh(np.where((((((data["signal_shift_-1"]) * 2.0)) > (((np.cos((data["stdbatch_slices2_msignal"]))) * 2.0)))*1.) <= -998, data["minbatch_slices2"], np.maximum(((data["minbatch_slices2"])), ((np.where(data["stdbatch_slices2_msignal"] <= -998, data["mean_abs_chgbatch_msignal"], ((((np.cos((data["stdbatch_slices2_msignal"]))) * 2.0)) * 2.0) )))) )) +

                            0.050000*np.tanh(np.where(((((data["stdbatch_msignal"]) * (data["medianbatch_slices2"]))) * 2.0) <= -998, data["maxtominbatch_msignal"], ((((((data["stdbatch_slices2_msignal"]) * (data["medianbatch_slices2"]))) - (np.where(data["abs_minbatch_msignal"] <= -998, ((data["stdbatch_msignal"]) * 2.0), np.where(((np.sin((data["maxtominbatch_msignal"]))) * 2.0) <= -998, data["stdbatch_slices2_msignal"], np.sin((data["maxtominbatch_msignal"])) ) )))) * 2.0) )) +

                            0.050000*np.tanh(((((np.minimum((((-((data["mean_abs_chgbatch_slices2_msignal"]))))), ((((np.minimum(((((data["signal"]) * (np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((((data["stdbatch_msignal"]) * (data["maxbatch_slices2"]))))))))), ((data["signal"])))) * (data["maxbatch_slices2"])))))) + (data["maxbatch_slices2"]))) + (data["stdbatch_slices2"]))) +

                            0.050000*np.tanh(np.where(((((np.where(np.tanh((data["abs_minbatch_msignal"])) > -998, data["signal_shift_-1"], data["mean_abs_chgbatch_slices2"] )) * 2.0)) * (data["mean_abs_chgbatch_slices2"])) <= -998, np.where(data["maxbatch_slices2"] <= -998, np.minimum(((np.minimum((((6.92917728424072266))), ((data["signal_shift_-1_msignal"]))))), ((data["stdbatch_msignal"]))), data["minbatch_slices2"] ), ((((((data["signal"]) * (np.cos((data["stdbatch_slices2"]))))) * 2.0)) * (data["abs_maxbatch"])) )) +

                            0.050000*np.tanh((-((((np.where((((((((3.0)) <= (((((((np.cos(((-((((data["meanbatch_slices2"]) * 2.0))))))) + (data["minbatch_msignal"]))/2.0)) > (data["maxtominbatch_msignal"]))*1.)))*1.)) + ((6.0)))/2.0) > -998, np.sin((data["maxtominbatch_msignal"])), data["meanbatch_slices2"] )) * 2.0))))) +

                            0.050000*np.tanh((((5.61153316497802734)) * (((np.minimum((((-((data["mean_abs_chgbatch_slices2_msignal"]))))), ((data["mean_abs_chgbatch_slices2_msignal"])))) + (((np.cos((data["abs_minbatch_slices2_msignal"]))) + (np.cos((data["abs_avgbatch_slices2_msignal"]))))))))) +

                            0.050000*np.tanh(((np.where(data["abs_maxbatch_slices2"] <= -998, data["abs_minbatch_msignal"], np.cos((data["abs_minbatch_msignal"])) )) * 2.0)) +

                            0.050000*np.tanh(np.sin((((np.where(np.minimum(((((data["mean_abs_chgbatch_slices2"]) * 2.0))), ((np.sin((np.sin(((((data["abs_maxbatch"]) + (data["rangebatch_msignal"]))/2.0)))))))) > -998, data["signal"], data["abs_maxbatch"] )) * 2.0)))) +

                            0.050000*np.tanh(((np.where(((data["signal"]) * ((-((((data["rangebatch_slices2"]) * 2.0)))))) > -998, ((data["signal"]) * 2.0), ((data["rangebatch_slices2"]) * (data["maxtominbatch"])) )) * (np.tanh((np.where(((((data["maxtominbatch_msignal"]) * 2.0)) * 2.0) <= -998, np.cos((data["minbatch_msignal"])), (((11.20401573181152344)) + (((((data["maxtominbatch_msignal"]) * 2.0)) * 2.0))) )))))) +

                            0.050000*np.tanh(((((data["abs_avgbatch_slices2"]) * (((np.sin((data["minbatch_msignal"]))) - (np.maximum(((data["abs_avgbatch_slices2_msignal"])), ((data["signal_shift_-1"])))))))) + ((((((np.sin((data["abs_avgbatch_slices2_msignal"]))) + (data["abs_avgbatch_slices2"]))/2.0)) * (((np.sin((data["abs_avgbatch_slices2"]))) * 2.0)))))) +

                            0.050000*np.tanh((((6.0)) * (np.where(data["stdbatch_slices2"] <= -998, data["signal_shift_+1"], np.where(data["abs_maxbatch_slices2_msignal"] <= -998, (6.0), np.sin((np.where(data["signal"] > -998, data["signal"], data["signal"] ))) ) )))) +

                            0.050000*np.tanh(((np.minimum(((((data["abs_maxbatch_slices2"]) * (data["signal"])))), ((data["abs_avgbatch_slices2"])))) - ((((((((((((np.sin((data["stdbatch_slices2_msignal"]))) <= (data["abs_avgbatch_slices2_msignal"]))*1.)) / 2.0)) + (((data["signal_shift_+1"]) * (data["abs_avgbatch_slices2_msignal"]))))/2.0)) + (((data["minbatch"]) * (((data["signal_shift_+1"]) * (data["stdbatch_slices2_msignal"]))))))/2.0)))) +

                            0.050000*np.tanh((((14.12239265441894531)) * (((np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((np.cos((np.where(np.maximum(((data["stdbatch_slices2"])), ((np.cos((data["minbatch"]))))) > -998, ((data["stdbatch_slices2_msignal"]) * 2.0), ((data["stdbatch_slices2_msignal"]) * 2.0) ))))))) / 2.0)))) +

                            0.050000*np.tanh(((data["signal"]) - ((-((((data["maxbatch_slices2"]) * (np.where((-((data["stdbatch_msignal"]))) <= -998, ((np.cos((data["signal"]))) * (data["abs_maxbatch_slices2"])), (((data["abs_minbatch_slices2_msignal"]) + (np.cos((((data["stdbatch_msignal"]) * 2.0)))))/2.0) ))))))))) +

                            0.050000*np.tanh(np.where((-((data["maxtominbatch_msignal"]))) <= -998, ((((data["signal"]) * 2.0)) + (((data["stdbatch_msignal"]) * (((data["signal"]) * (data["signal"])))))), (((10.0)) * (np.cos((data["abs_minbatch_slices2_msignal"])))) )) +

                            0.050000*np.tanh(((np.sin((np.maximum(((data["meanbatch_slices2"])), ((((np.sin((np.maximum(((data["meanbatch_slices2"])), ((np.where(np.maximum(((data["meanbatch_slices2"])), ((np.where(((data["stdbatch_slices2_msignal"]) * (data["minbatch_slices2"])) <= -998, ((data["meanbatch_slices2"]) / 2.0), data["signal"] )))) <= -998, np.sin(((((-((((data["meanbatch_slices2"]) * 2.0))))) / 2.0))), data["signal"] ))))))) * 2.0))))))) * 2.0)) +

                            0.050000*np.tanh(((((data["signal"]) * (((data["rangebatch_slices2_msignal"]) * (((data["signal"]) * (((data["rangebatch_slices2_msignal"]) * (np.sin((np.cos((np.where(data["rangebatch_slices2_msignal"] <= -998, np.tanh((data["signal"])), data["signal"] )))))))))))))) * (((data["signal"]) + (((data["signal"]) * (np.sin(((6.0)))))))))) +

                            0.050000*np.tanh(np.minimum(((np.minimum(((np.tanh((np.where(np.sin((np.cos((((data["abs_avgbatch_slices2_msignal"]) * (np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, data["abs_maxbatch_slices2_msignal"], (0.62379610538482666) ))))))) > -998, ((data["abs_avgbatch_slices2_msignal"]) * 2.0), np.cos((data["stdbatch_slices2"])) ))))), ((np.minimum(((data["signal"])), ((((np.cos((data["medianbatch_slices2"]))) * 2.0))))))))), (((7.0))))) +

                            0.050000*np.tanh(np.minimum(((np.sin(((-((data["maxtominbatch_slices2_msignal"]))))))), (((((((np.where(((np.sin((data["abs_avgbatch_msignal"]))) + ((((data["maxbatch_slices2"]) + ((((data["abs_avgbatch_msignal"]) <= (np.cos((data["maxtominbatch_slices2_msignal"]))))*1.)))/2.0))) <= -998, data["meanbatch_msignal"], ((data["maxbatch_slices2"]) * 2.0) )) * 2.0)) + (np.where(data["abs_avgbatch_msignal"] <= -998, data["meanbatch_msignal"], np.sin((data["abs_avgbatch_msignal"])) )))/2.0))))) +

                            0.050000*np.tanh(np.where((((data["abs_avgbatch_msignal"]) <= (data["maxtominbatch_slices2_msignal"]))*1.) <= -998, ((data["minbatch"]) + (data["abs_maxbatch_slices2_msignal"])), ((np.sin((((np.where(data["abs_maxbatch_msignal"] > -998, data["abs_maxbatch_msignal"], data["stdbatch_msignal"] )) + (((data["maxtominbatch_msignal"]) + ((((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)) > ((((((data["maxtominbatch_slices2"]) / 2.0)) <= (data["abs_maxbatch_slices2_msignal"]))*1.)))*1.)))))))) * 2.0) )) +

                            0.050000*np.tanh((-((np.sin((((np.sin((data["mean_abs_chgbatch_msignal"]))) + (np.minimum(((data["abs_minbatch_slices2_msignal"])), ((np.sin((data["signal_shift_+1"]))))))))))))) +

                            0.050000*np.tanh(np.where(np.where(np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((data["signal_shift_+1"]))) > -998, data["abs_avgbatch_slices2"], data["maxbatch_slices2_msignal"] ) <= -998, data["rangebatch_msignal"], ((np.maximum((((-((np.sin(((9.0)))))))), ((data["abs_maxbatch"])))) * (np.where(data["signal_shift_-1"] <= -998, (8.73901081085205078), (((8.73901081085205078)) * (np.sin((np.where(data["abs_avgbatch_slices2"] > -998, data["abs_avgbatch_slices2"], data["minbatch_slices2"] ))))) ))) )) +

                            0.050000*np.tanh(((data["abs_maxbatch_slices2"]) * (np.where((((data["abs_maxbatch_msignal"]) > (data["meanbatch_slices2"]))*1.) <= -998, ((((data["maxbatch_slices2_msignal"]) * 2.0)) * 2.0), ((np.cos((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, data["abs_maxbatch_slices2"], ((np.cos((data["maxbatch_msignal"]))) + (data["meanbatch_slices2"])) )))) * 2.0) )))) +

                            0.050000*np.tanh(((np.minimum(((np.cos((data["stdbatch_msignal"])))), ((((np.minimum(((((np.minimum(((((np.cos((((data["stdbatch_msignal"]) * 2.0)))) * 2.0))), ((np.cos(((((data["abs_avgbatch_slices2_msignal"]) > (data["abs_avgbatch_slices2_msignal"]))*1.))))))) * 2.0))), ((np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0))))))) * 2.0))))) * 2.0)) +

                            0.050000*np.tanh(np.minimum((((((((data["maxbatch_slices2"]) * (data["stdbatch_slices2_msignal"]))) + ((((((np.cos(((((((((data["stdbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))/2.0)) * (np.cos((data["abs_avgbatch_slices2_msignal"]))))) / 2.0)))) * (data["maxbatch_slices2"]))) > (data["abs_avgbatch_slices2_msignal"]))*1.)))/2.0))), ((np.cos((data["abs_avgbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(((np.minimum(((np.sin((((data["maxbatch_msignal"]) * 2.0))))), ((((np.maximum(((data["maxtominbatch"])), ((((np.sin((np.cos((((data["stdbatch_msignal"]) * 2.0)))))) * 2.0))))) * (np.sin((((data["maxbatch_slices2"]) / 2.0))))))))) * 2.0)) +

                            0.050000*np.tanh(((np.minimum((((4.0))), ((data["abs_maxbatch_slices2_msignal"])))) * (((((np.sin((((data["signal"]) * 2.0)))) * 2.0)) - ((((((((data["abs_avgbatch_msignal"]) * (data["abs_maxbatch_slices2_msignal"]))) + (((data["minbatch_slices2_msignal"]) - (np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["signal_shift_+1"], data["signal_shift_+1_msignal"] )))))/2.0)) / 2.0)))))) +

                            0.050000*np.tanh(np.sin((np.where(np.cos((data["stdbatch_slices2_msignal"])) <= -998, np.cos((data["meanbatch_slices2"])), np.maximum(((data["medianbatch_slices2_msignal"])), ((((data["maxbatch_msignal"]) * 2.0)))) )))) +

                            0.050000*np.tanh((((-((np.cos(((-((np.maximum(((data["abs_maxbatch_msignal"])), ((data["abs_avgbatch_slices2_msignal"])))))))))))) * 2.0)) +

                            0.050000*np.tanh(np.where(np.where(data["abs_maxbatch_slices2"] > -998, (((7.36041975021362305)) - (data["abs_maxbatch"])), ((np.where(data["signal"] > -998, (7.36041975021362305), data["signal_shift_-1"] )) * 2.0) ) > -998, (((7.36041975021362305)) - (data["abs_maxbatch"])), np.tanh((np.where(data["stdbatch_slices2"] > -998, (((7.36041975021362305)) - (data["abs_maxbatch"])), data["abs_maxbatch_slices2"] ))) )) +

                            0.050000*np.tanh(np.sin((np.maximum(((np.where(data["abs_avgbatch_msignal"] <= -998, data["signal_shift_+1"], data["signal_shift_+1"] ))), ((np.maximum(((data["abs_avgbatch_msignal"])), ((np.maximum(((np.sin((data["signal"])))), ((np.maximum(((data["abs_avgbatch_slices2"])), ((data["abs_avgbatch_msignal"]))))))))))))))) +

                            0.050000*np.tanh(((np.sin((((np.where(np.cos((((np.where(data["maxtominbatch"] > -998, data["maxbatch_msignal"], np.where((-((data["maxtominbatch"]))) > -998, np.where(data["maxtominbatch"] > -998, data["rangebatch_msignal"], data["signal"] ), np.where(data["abs_maxbatch_msignal"] > -998, data["stdbatch_slices2_msignal"], ((np.sin((data["maxbatch_msignal"]))) * 2.0) ) ) )) * 2.0))) > -998, data["maxbatch_msignal"], data["maxbatch_msignal"] )) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(((data["maxtominbatch_slices2"]) + (((np.cos((((((data["stdbatch_msignal"]) + (data["stdbatch_slices2_msignal"]))) * (np.cos(((((((data["abs_avgbatch_msignal"]) + (data["medianbatch_slices2"]))) <= ((-((data["abs_maxbatch_slices2_msignal"])))))*1.)))))))) * ((((7.74140071868896484)) + (np.tanh((np.cos((data["abs_minbatch_msignal"]))))))))))) +

                            0.050000*np.tanh(((np.cos((data["abs_minbatch_msignal"]))) * ((((10.0)) * (np.cos((np.where(np.cos((data["mean_abs_chgbatch_slices2_msignal"])) > -998, data["medianbatch_slices2_msignal"], (((((9.0)) + (np.cos((np.cos((data["abs_minbatch_msignal"]))))))) + (np.cos((data["abs_maxbatch_slices2"])))) )))))))) +

                            0.050000*np.tanh(np.sin((np.where(data["maxtominbatch_msignal"] <= -998, np.where(data["rangebatch_slices2"] > -998, data["signal_shift_-1"], (((((data["rangebatch_slices2_msignal"]) <= (np.cos((((np.maximum(((data["medianbatch_slices2"])), ((data["minbatch_msignal"])))) * (data["abs_maxbatch_msignal"]))))))*1.)) / 2.0) ), ((data["medianbatch_slices2"]) - (data["abs_minbatch_slices2_msignal"])) )))) +

                            0.050000*np.tanh(((np.cos((((data["stdbatch_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))))) * (((data["rangebatch_slices2"]) + (((data["meanbatch_slices2"]) * (((((data["mean_abs_chgbatch_slices2_msignal"]) + (((((data["abs_avgbatch_slices2_msignal"]) + (((np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))))) * (data["signal_shift_+1"]))))) + (data["meanbatch_msignal"]))))) + (data["signal_shift_+1"]))))))))) +

                            0.050000*np.tanh(np.where(data["meanbatch_slices2_msignal"] <= -998, (9.69016170501708984), np.cos((np.where(np.cos((data["abs_avgbatch_slices2_msignal"])) <= -998, data["abs_avgbatch_slices2"], np.where(data["stdbatch_slices2_msignal"] <= -998, ((data["minbatch_msignal"]) - (np.cos((np.maximum(((data["abs_avgbatch_slices2"])), ((data["mean_abs_chgbatch_slices2_msignal"]))))))), data["medianbatch_slices2"] ) ))) )) +

                            0.050000*np.tanh(((np.where(((data["rangebatch_slices2_msignal"]) * (data["rangebatch_slices2_msignal"])) <= -998, data["medianbatch_slices2"], np.cos((np.sin((data["abs_avgbatch_slices2"])))) )) - (((data["maxtominbatch_msignal"]) + (((((data["rangebatch_slices2_msignal"]) * (np.sin((data["abs_avgbatch_slices2"]))))) * (data["maxtominbatch_msignal"]))))))) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) * (((((data["stdbatch_slices2_msignal"]) + (((data["maxbatch_msignal"]) * ((((((data["mean_abs_chgbatch_msignal"]) + (data["abs_minbatch_slices2"]))/2.0)) + (np.cos((data["abs_avgbatch_slices2_msignal"]))))))))) + (((data["abs_maxbatch_slices2"]) * (((data["mean_abs_chgbatch_msignal"]) + (np.tanh((((np.sin((data["abs_avgbatch_slices2_msignal"]))) + (np.cos((data["abs_avgbatch_slices2_msignal"]))))))))))))))) +

                            0.050000*np.tanh(((np.where((13.25574779510498047) <= -998, data["medianbatch_slices2"], np.minimum((((((data["mean_abs_chgbatch_msignal"]) + ((((((((4.63609886169433594)) + (data["signal_shift_-1"]))/2.0)) > (data["maxtominbatch_msignal"]))*1.)))/2.0))), ((np.where(((data["medianbatch_slices2"]) * 2.0) <= -998, data["maxtominbatch_msignal"], np.minimum(((np.sin((data["abs_avgbatch_slices2"])))), ((data["signal_shift_-1"]))) )))) )) * 2.0)) +

                            0.050000*np.tanh((((-((data["rangebatch_slices2"])))) * ((((np.cos((np.where(((((data["abs_avgbatch_slices2"]) / 2.0)) / 2.0) > -998, np.sin((((((np.sin((np.cos((((data["abs_avgbatch_slices2_msignal"]) * 2.0)))))) * 2.0)) * 2.0))), data["minbatch_slices2_msignal"] )))) + (((((data["abs_minbatch_slices2_msignal"]) * 2.0)) * (np.sin((data["abs_avgbatch_slices2"]))))))/2.0)))) +

                            0.050000*np.tanh(np.minimum(((np.where(((np.sin((np.minimum(((data["maxbatch_slices2"])), ((data["maxbatch_slices2"])))))) * 2.0) > -998, np.sin((data["abs_avgbatch_slices2"])), data["abs_avgbatch_slices2"] ))), ((np.sin((data["abs_avgbatch_slices2"])))))) +

                            0.050000*np.tanh(((np.where((((data["maxtominbatch"]) > (data["maxtominbatch"]))*1.) > -998, np.maximum(((np.where((-((data["rangebatch_slices2"]))) > -998, (10.04239368438720703), data["abs_minbatch_slices2_msignal"] ))), (((-(((-(((10.04239368438720703)))))))))), data["rangebatch_slices2"] )) * (np.cos((data["abs_minbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(np.where(data["rangebatch_slices2"] <= -998, data["abs_avgbatch_msignal"], ((data["abs_avgbatch_msignal"]) * (((((((np.cos((data["abs_avgbatch_msignal"]))) * (((data["meanbatch_msignal"]) + (data["stdbatch_slices2_msignal"]))))) + (data["abs_minbatch_slices2_msignal"]))) + (((np.cos((data["abs_avgbatch_msignal"]))) * (data["abs_maxbatch"])))))) )) +

                            0.050000*np.tanh(((np.cos((data["medianbatch_slices2_msignal"]))) * (np.where(data["abs_avgbatch_slices2_msignal"] > -998, np.where((8.0) <= -998, data["signal_shift_-1_msignal"], ((data["abs_maxbatch"]) * (((((((14.30874156951904297)) > (((data["maxbatch_slices2_msignal"]) * 2.0)))*1.)) + (data["mean_abs_chgbatch_slices2_msignal"])))) ), ((data["maxbatch_slices2_msignal"]) * 2.0) )))) +

                            0.050000*np.tanh(((np.sin((np.where(np.sin((data["stdbatch_slices2_msignal"])) <= -998, data["abs_maxbatch"], ((data["mean_abs_chgbatch_slices2_msignal"]) - (data["abs_maxbatch"])) )))) * 2.0)) +

                            0.050000*np.tanh(((((np.cos((data["signal_shift_+1"]))) + (np.sin((data["abs_avgbatch_msignal"]))))) + ((((((np.sin((data["abs_avgbatch_msignal"]))) + (np.minimum(((data["abs_avgbatch_msignal"])), ((np.cos((data["mean_abs_chgbatch_slices2"])))))))/2.0)) + (((((data["stdbatch_msignal"]) - (data["signal_shift_+1"]))) * (data["signal_shift_-1"]))))))) +

                            0.050000*np.tanh(((data["rangebatch_msignal"]) + (((data["rangebatch_msignal"]) + (((np.where(data["rangebatch_msignal"] <= -998, np.sin((data["mean_abs_chgbatch_slices2_msignal"])), ((np.where(data["rangebatch_msignal"] > -998, data["abs_minbatch_slices2_msignal"], data["rangebatch_slices2"] )) * (((data["rangebatch_slices2"]) / 2.0))) )) * 2.0)))))) +

                            0.050000*np.tanh((-((((data["rangebatch_msignal"]) * (np.sin((data["maxtominbatch_msignal"])))))))) +

                            0.050000*np.tanh(((np.where((((((((data["maxtominbatch_msignal"]) * (np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["stdbatch_slices2"], data["rangebatch_slices2"] )))) * 2.0)) + (data["stdbatch_slices2"]))/2.0) <= -998, np.cos((data["maxtominbatch_msignal"])), ((np.sin((data["abs_avgbatch_slices2"]))) * 2.0) )) * (data["signal"]))) +

                            0.050000*np.tanh(np.where(np.minimum(((np.where(data["abs_maxbatch_slices2_msignal"] > -998, np.cos((data["medianbatch_slices2"])), data["abs_avgbatch_slices2_msignal"] ))), ((data["abs_avgbatch_msignal"]))) > -998, data["signal"], data["meanbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(((np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((np.minimum(((np.sin((np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((np.sin((np.cos((data["mean_abs_chgbatch_msignal"]))))))))))), ((np.sin((data["abs_avgbatch_slices2"]))))))))) * 2.0)) +

                            0.050000*np.tanh(data["abs_avgbatch_slices2_msignal"]) +

                            0.050000*np.tanh(((np.sin((((data["maxbatch_msignal"]) * (np.where((((data["abs_maxbatch_slices2"]) <= (((data["abs_avgbatch_msignal"]) - (np.where(data["maxbatch_msignal"] > -998, np.maximum(((np.sin((data["abs_minbatch_msignal"])))), ((data["medianbatch_slices2_msignal"]))), data["abs_avgbatch_msignal"] )))))*1.) > -998, data["abs_avgbatch_slices2"], ((np.tanh((data["signal"]))) - (data["abs_avgbatch_slices2_msignal"])) )))))) * (data["rangebatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((((np.where(data["minbatch"] > -998, np.where((6.0) > -998, np.sin((data["abs_avgbatch_slices2"])), data["signal_shift_-1"] ), np.where(data["signal_shift_+1"] > -998, np.maximum(((data["signal_shift_+1"])), ((data["abs_avgbatch_slices2"]))), ((data["maxbatch_slices2"]) + (data["signal_shift_+1"])) ) )) * (data["signal_shift_+1"]))) * 2.0)) +

                            0.050000*np.tanh(((((data["abs_maxbatch_msignal"]) * (np.cos((((np.where(data["signal_shift_+1"] > -998, data["abs_avgbatch_slices2"], np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["stdbatch_slices2"], (((np.sin((np.maximum(((data["stdbatch_msignal"])), ((np.minimum(((data["abs_avgbatch_slices2"])), ((data["signal_shift_+1_msignal"]))))))))) > (data["abs_minbatch_msignal"]))*1.) ) )) / 2.0)))))) * (data["signal_shift_+1"]))) +

                            0.050000*np.tanh(((((((((np.tanh(((((6.0)) * 2.0)))) + (np.cos((np.maximum(((((data["stdbatch_slices2"]) * (data["signal_shift_-1"])))), ((data["stdbatch_slices2"])))))))) * 2.0)) + (data["stdbatch_msignal"]))) * 2.0)) +

                            0.050000*np.tanh(((((np.sin((data["rangebatch_msignal"]))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.sin((np.where(((np.sin((data["maxbatch_msignal"]))) * 2.0) > -998, data["abs_avgbatch_msignal"], (3.0) )))) +

                            0.050000*np.tanh(((((data["minbatch_msignal"]) - (((((data["medianbatch_slices2"]) / 2.0)) + (np.sin((np.minimum(((data["medianbatch_slices2"])), ((((data["abs_avgbatch_slices2_msignal"]) * 2.0))))))))))) * (np.sin((((data["maxtominbatch_msignal"]) + (np.cos((data["medianbatch_slices2"]))))))))) +

                            0.050000*np.tanh(((((((np.sin((((data["rangebatch_msignal"]) * 2.0)))) * (data["rangebatch_msignal"]))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((np.cos((np.where(((data["signal_shift_-1"]) - (np.cos((data["signal_shift_-1"])))) > -998, ((data["stdbatch_slices2_msignal"]) * 2.0), np.cos((data["minbatch"])) )))) * 2.0)) +

                            0.050000*np.tanh(((((data["mean_abs_chgbatch_msignal"]) + (((np.sin((((((data["signal_shift_-1"]) - (data["stdbatch_msignal"]))) * 2.0)))) * (data["abs_avgbatch_slices2_msignal"]))))) * ((((((4.0)) * 2.0)) * 2.0)))) +

                            0.050000*np.tanh((((((-(((-((np.sin((((data["abs_avgbatch_msignal"]) + (((((-(((-((data["maxbatch_slices2_msignal"]))))))) > (np.where(((data["abs_avgbatch_slices2"]) * (np.sin((data["abs_avgbatch_slices2"])))) <= -998, data["abs_avgbatch_slices2"], data["mean_abs_chgbatch_slices2"] )))*1.)))))))))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.sin((np.where(((data["signal"]) - (data["mean_abs_chgbatch_msignal"])) <= -998, ((data["stdbatch_msignal"]) * 2.0), np.where(data["mean_abs_chgbatch_msignal"] <= -998, data["abs_avgbatch_slices2"], ((data["signal_shift_-1"]) - (np.where(data["abs_minbatch_msignal"] > -998, data["mean_abs_chgbatch_msignal"], data["abs_avgbatch_msignal"] ))) ) )))) +

                            0.050000*np.tanh(np.minimum(((np.minimum(((np.minimum(((np.cos((data["abs_avgbatch_slices2_msignal"])))), ((((data["abs_avgbatch_slices2"]) * (np.cos(((((data["mean_abs_chgbatch_slices2"]) <= (data["signal_shift_+1"]))*1.)))))))))), ((data["abs_avgbatch_msignal"]))))), ((data["abs_avgbatch_slices2_msignal"])))) +

                            0.050000*np.tanh(((np.where(data["stdbatch_msignal"] > -998, data["signal_shift_-1"], data["maxtominbatch_msignal"] )) * (((np.cos((((((data["meanbatch_slices2"]) / 2.0)) * (data["signal"]))))) * 2.0)))) +

                            0.050000*np.tanh(((np.cos((np.where(np.where(((data["stdbatch_slices2_msignal"]) * 2.0) > -998, data["abs_avgbatch_msignal"], data["abs_avgbatch_msignal"] ) > -998, np.where((((np.sin((((data["meanbatch_slices2_msignal"]) * 2.0)))) > ((((data["stdbatch_slices2"]) + (data["mean_abs_chgbatch_slices2"]))/2.0)))*1.) <= -998, np.cos((np.where(data["abs_minbatch_slices2"] > -998, (-((data["mean_abs_chgbatch_slices2"]))), data["signal_shift_+1_msignal"] ))), data["abs_avgbatch_msignal"] ), data["abs_avgbatch_msignal"] )))) * 2.0)) +

                            0.050000*np.tanh(((((np.maximum((((6.08415985107421875))), (((((np.cos((data["medianbatch_slices2"]))) <= (data["maxtominbatch_msignal"]))*1.))))) + (data["maxtominbatch_msignal"]))) + (data["maxtominbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((((np.cos((((data["abs_avgbatch_msignal"]) - (np.where(data["signal_shift_+1"] > -998, np.tanh((data["signal_shift_+1"])), np.where(((data["abs_avgbatch_msignal"]) / 2.0) > -998, np.tanh((data["abs_avgbatch_msignal"])), data["signal_shift_-1"] ) )))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((data["mean_abs_chgbatch_msignal"]) + (((np.sin((data["signal_shift_-1"]))) + (((((np.where(((data["abs_avgbatch_slices2"]) * 2.0) > -998, np.sin((np.where(data["abs_avgbatch_slices2"] > -998, data["abs_avgbatch_slices2"], data["abs_avgbatch_slices2"] ))), (((((data["mean_abs_chgbatch_msignal"]) + (data["abs_avgbatch_slices2"]))) + (data["rangebatch_slices2"]))/2.0) )) * 2.0)) * 2.0)))))) +

                            0.050000*np.tanh((((data["abs_avgbatch_msignal"]) + (((data["abs_maxbatch_slices2_msignal"]) * (((data["signal"]) * (np.sin(((-((data["rangebatch_slices2_msignal"])))))))))))/2.0)) +

                            0.050000*np.tanh(((data["abs_avgbatch_slices2_msignal"]) * (np.where(np.where(np.cos((data["maxbatch_slices2_msignal"])) <= -998, ((data["signal_shift_+1"]) / 2.0), data["meanbatch_msignal"] ) <= -998, np.where(data["medianbatch_msignal"] <= -998, data["abs_maxbatch_slices2"], data["signal_shift_-1"] ), np.cos((data["abs_avgbatch_msignal"])) )))) +

                            0.050000*np.tanh((((((5.29374170303344727)) * (((data["stdbatch_msignal"]) + (np.cos((((data["signal_shift_+1"]) * (((data["stdbatch_slices2_msignal"]) + (data["stdbatch_slices2_msignal"]))))))))))) + (np.cos(((5.29374170303344727)))))) +

                            0.050000*np.tanh(np.where(data["mean_abs_chgbatch_slices2"] > -998, ((np.where(((data["signal"]) * (np.sin((data["abs_avgbatch_slices2"])))) > -998, ((np.where(((data["stdbatch_slices2"]) * 2.0) > -998, ((data["signal"]) * (np.sin((data["abs_avgbatch_slices2"])))), np.sin((data["abs_avgbatch_slices2"])) )) * 2.0), data["medianbatch_slices2"] )) * 2.0), np.where(np.sin((data["abs_avgbatch_slices2"])) > -998, data["abs_avgbatch_slices2"], data["abs_avgbatch_slices2_msignal"] ) )) +

                            0.050000*np.tanh(((((data["abs_avgbatch_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))) + ((((data["abs_avgbatch_slices2_msignal"]) + (np.tanh((data["rangebatch_slices2"]))))/2.0)))) +

                            0.050000*np.tanh(((np.sin((((np.minimum(((np.sin((data["abs_avgbatch_slices2"])))), ((np.minimum((((((np.cos((((np.where(data["signal_shift_+1"] > -998, data["meanbatch_slices2"], np.sin(((5.0))) )) * 2.0)))) + (((((np.sin((data["signal_shift_+1"]))) * 2.0)) * 2.0)))/2.0))), ((data["signal_shift_+1"]))))))) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) + (((data["signal"]) * ((((((data["meanbatch_slices2"]) > (data["meanbatch_slices2"]))*1.)) + ((((((data["signal_shift_+1"]) <= (data["abs_avgbatch_slices2"]))*1.)) + ((-((np.where(np.minimum(((data["signal_shift_-1"])), ((data["signal"]))) > -998, data["meanbatch_slices2"], ((np.sin((data["maxtominbatch"]))) * 2.0) ))))))))))))) +

                            0.050000*np.tanh(((np.minimum(((np.maximum(((data["mean_abs_chgbatch_slices2"])), ((np.maximum((((9.0))), ((np.cos((((data["abs_minbatch_msignal"]) * 2.0))))))))))), (((-((np.sin((((data["abs_minbatch_msignal"]) * 2.0)))))))))) * 2.0)) +

                            0.050000*np.tanh(data["abs_avgbatch_msignal"]) +

                            0.050000*np.tanh(((data["minbatch_slices2"]) * (np.cos((np.where(data["abs_minbatch_slices2"] > -998, np.where(np.where(data["abs_maxbatch_msignal"] > -998, data["abs_minbatch_slices2_msignal"], (((data["signal_shift_-1"]) <= (data["abs_maxbatch_msignal"]))*1.) ) > -998, data["abs_maxbatch_msignal"], data["meanbatch_slices2_msignal"] ), ((((((data["signal_shift_-1"]) <= (((data["minbatch"]) - (np.sin((data["minbatch_slices2"]))))))*1.)) <= (data["meanbatch_slices2_msignal"]))*1.) )))))) +

                            0.050000*np.tanh(((data["maxtominbatch_slices2_msignal"]) - (np.minimum(((data["minbatch"])), ((np.where(data["rangebatch_slices2"] <= -998, np.maximum(((data["minbatch"])), ((((((data["abs_avgbatch_msignal"]) + (((data["stdbatch_msignal"]) * 2.0)))) / 2.0)))), data["mean_abs_chgbatch_slices2_msignal"] ))))))) +

                            0.050000*np.tanh(((np.where(data["abs_avgbatch_slices2_msignal"] > -998, np.cos((data["abs_avgbatch_slices2_msignal"])), ((((((np.tanh((np.tanh((np.cos((data["abs_avgbatch_slices2_msignal"]))))))) * 2.0)) * 2.0)) * (np.cos((data["medianbatch_slices2"])))) )) * 2.0)) +

                            0.050000*np.tanh(((((((((data["abs_avgbatch_msignal"]) * 2.0)) * 2.0)) * (np.cos((np.maximum(((data["signal_shift_+1"])), ((np.where((((((np.cos((((data["abs_avgbatch_msignal"]) + (((np.tanh((np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["abs_avgbatch_msignal"], data["meanbatch_slices2_msignal"] )))) / 2.0)))))) + (data["maxtominbatch_msignal"]))/2.0)) * (data["maxbatch_slices2"])) > -998, data["meanbatch_msignal"], data["meanbatch_slices2_msignal"] ))))))))) * 2.0)) +

                            0.050000*np.tanh(((data["abs_avgbatch_slices2"]) + (((np.where(data["stdbatch_msignal"] > -998, ((data["meanbatch_slices2"]) + (np.where(data["stdbatch_slices2_msignal"] <= -998, data["meanbatch_slices2"], data["signal_shift_-1"] ))), data["rangebatch_slices2_msignal"] )) * (np.tanh((np.tanh((((np.minimum(((((data["stdbatch_msignal"]) * (data["abs_minbatch_slices2"])))), (((-((data["minbatch_slices2"]))))))) + (np.tanh((((data["stdbatch_slices2_msignal"]) * 2.0)))))))))))))) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) * (np.where(data["meanbatch_slices2_msignal"] <= -998, data["rangebatch_slices2_msignal"], ((np.where(((((data["minbatch_slices2_msignal"]) * 2.0)) * (data["rangebatch_msignal"])) > -998, ((data["stdbatch_msignal"]) - (data["meanbatch_msignal"])), ((np.where(data["stdbatch_slices2_msignal"] > -998, ((np.sin(((14.30072021484375000)))) - (np.sin((data["stdbatch_msignal"])))), data["meanbatch_slices2_msignal"] )) * 2.0) )) * 2.0) )))) +

                            0.050000*np.tanh(((data["abs_maxbatch_slices2"]) * (np.tanh(((-((np.cos((((data["maxtominbatch_msignal"]) - (np.where(data["maxbatch_msignal"] > -998, np.where(np.tanh((((data["signal_shift_-1_msignal"]) * 2.0))) > -998, data["signal"], np.cos(((((np.sin((data["signal_shift_+1"]))) > (data["meanbatch_msignal"]))*1.))) ), ((data["medianbatch_msignal"]) * (data["maxbatch_msignal"])) ))))))))))))) +

                            0.050000*np.tanh(((np.sin((((data["maxbatch_msignal"]) * 2.0)))) * (np.where(np.sin((data["signal_shift_-1"])) > -998, ((data["rangebatch_slices2_msignal"]) - (data["medianbatch_slices2_msignal"])), np.sin((data["rangebatch_slices2_msignal"])) )))) +

                            0.050000*np.tanh(np.minimum(((data["abs_maxbatch_msignal"])), ((np.sin((np.where(((((((data["maxbatch_msignal"]) / 2.0)) / 2.0)) * 2.0) <= -998, (((data["abs_avgbatch_slices2"]) > (((((data["maxbatch_msignal"]) / 2.0)) * 2.0)))*1.), ((data["maxbatch_msignal"]) * 2.0) ))))))) +

                            0.050000*np.tanh(((data["abs_maxbatch_slices2_msignal"]) * 2.0)) +

                            0.050000*np.tanh(((((np.maximum(((data["abs_avgbatch_msignal"])), ((((np.maximum(((data["stdbatch_slices2_msignal"])), ((np.cos((((data["stdbatch_slices2_msignal"]) - (data["mean_abs_chgbatch_msignal"])))))))) + (data["signal_shift_-1"])))))) * (np.minimum(((((((np.sin((data["signal_shift_-1"]))) * 2.0)) * 2.0))), ((np.sin((np.cos((((data["abs_avgbatch_msignal"]) - (data["signal_shift_-1"])))))))))))) * 2.0)) +

                            0.050000*np.tanh(((np.where(((((np.where(data["stdbatch_msignal"] > -998, data["minbatch_msignal"], data["abs_avgbatch_slices2_msignal"] )) * 2.0)) + ((-((data["stdbatch_slices2_msignal"]))))) > -998, ((np.sin((data["abs_avgbatch_slices2"]))) + (data["stdbatch_msignal"])), (((data["abs_avgbatch_slices2_msignal"]) <= (((np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)))) - (((data["stdbatch_msignal"]) * (data["abs_avgbatch_slices2_msignal"]))))))*1.) )) * 2.0)) +

                            0.050000*np.tanh(np.cos((np.sin((np.cos((data["abs_avgbatch_slices2_msignal"]))))))) +

                            0.050000*np.tanh(np.where(data["signal_shift_+1_msignal"] > -998, np.cos((((data["signal_shift_-1"]) - (np.where(np.tanh((data["abs_avgbatch_slices2_msignal"])) <= -998, np.cos((data["abs_avgbatch_slices2_msignal"])), np.where(data["signal_shift_-1"] > -998, data["abs_avgbatch_slices2_msignal"], (((data["abs_avgbatch_slices2_msignal"]) + (data["minbatch_slices2"]))/2.0) ) ))))), np.tanh((data["minbatch_slices2"])) )) +

                            0.050000*np.tanh(((((np.minimum(((((np.cos((data["abs_avgbatch_slices2_msignal"]))) * 2.0))), ((data["abs_avgbatch_slices2_msignal"])))) + (np.minimum(((np.minimum(((data["abs_maxbatch_msignal"])), ((np.minimum(((data["mean_abs_chgbatch_slices2"])), ((((data["stdbatch_slices2_msignal"]) * (((data["signal_shift_-1"]) + (data["medianbatch_slices2_msignal"])))))))))))), ((np.where(data["meanbatch_msignal"] <= -998, data["signal_shift_+1"], data["abs_maxbatch_msignal"] ))))))) * 2.0)) +

                            0.050000*np.tanh(((((data["abs_maxbatch"]) * (data["stdbatch_msignal"]))) * (np.where(data["signal_shift_+1"] > -998, ((np.maximum(((data["signal_shift_+1"])), ((((data["meanbatch_msignal"]) * 2.0))))) + (data["medianbatch_slices2_msignal"])), ((data["abs_maxbatch"]) * (np.sin((data["rangebatch_msignal"])))) )))) +

                            0.050000*np.tanh(((np.minimum(((data["stdbatch_slices2_msignal"])), ((data["stdbatch_slices2_msignal"])))) * (np.where(((data["meanbatch_msignal"]) + (data["stdbatch_msignal"])) > -998, (((10.0)) * (np.minimum(((((np.cos((data["meanbatch_msignal"]))) * 2.0))), ((data["signal_shift_-1_msignal"]))))), (10.0) )))) +

                            0.050000*np.tanh(((((data["abs_maxbatch"]) * ((((((((((np.cos(((-((data["abs_avgbatch_msignal"])))))) + (np.cos((data["stdbatch_msignal"]))))/2.0)) + (np.cos((data["abs_avgbatch_slices2_msignal"]))))/2.0)) + (np.cos(((-((data["meanbatch_slices2"])))))))/2.0)))) + (data["abs_avgbatch_msignal"]))) +

                            0.050000*np.tanh((((14.70708179473876953)) + (((data["abs_maxbatch"]) * (data["maxtominbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(((np.sin((data["abs_maxbatch_slices2"]))) * (np.where(np.minimum(((np.sin((data["abs_maxbatch_slices2"])))), ((data["rangebatch_slices2_msignal"]))) <= -998, data["mean_abs_chgbatch_slices2"], ((data["abs_minbatch_slices2_msignal"]) * 2.0) )))) +

                            0.050000*np.tanh((-((np.cos((((((data["abs_minbatch_slices2_msignal"]) * 2.0)) - ((((data["medianbatch_slices2_msignal"]) > (data["abs_minbatch_slices2_msignal"]))*1.))))))))) +

                            0.050000*np.tanh(np.where(data["mean_abs_chgbatch_slices2"] > -998, ((data["maxbatch_slices2"]) * (np.sin((np.cos((np.maximum(((data["medianbatch_msignal"])), ((np.where(data["stdbatch_msignal"] <= -998, np.maximum(((np.cos((np.sin((data["abs_avgbatch_slices2"])))))), ((np.sin((data["medianbatch_msignal"]))))), data["signal_shift_+1"] )))))))))), data["abs_avgbatch_slices2"] )) +

                            0.050000*np.tanh((14.91276741027832031)) +

                            0.050000*np.tanh((-((((((((data["signal"]) - ((((data["signal_shift_-1"]) > (((data["meanbatch_slices2"]) * ((((((data["signal_shift_-1"]) > (data["signal"]))*1.)) * 2.0)))))*1.)))) - ((((np.maximum(((np.tanh((data["medianbatch_slices2"])))), ((((data["signal_shift_-1"]) * 2.0))))) > (((data["rangebatch_slices2"]) * (data["mean_abs_chgbatch_slices2_msignal"]))))*1.)))) * 2.0))))) +

                            0.050000*np.tanh((((((3.0)) + (np.minimum(((((data["maxtominbatch_slices2_msignal"]) * (data["signal_shift_-1"])))), ((((data["stdbatch_slices2_msignal"]) + (data["maxbatch_slices2"])))))))) + (np.tanh((data["signal_shift_+1"]))))) +

                            0.050000*np.tanh(data["signal_shift_-1"]) +

                            0.050000*np.tanh(((data["rangebatch_slices2_msignal"]) * (((data["abs_minbatch_msignal"]) * (((data["meanbatch_msignal"]) - ((((data["abs_avgbatch_slices2_msignal"]) > ((-(((((np.maximum(((data["maxbatch_slices2"])), ((data["abs_avgbatch_slices2"])))) <= (data["abs_minbatch_slices2"]))*1.))))))*1.)))))))) +

                            0.050000*np.tanh(((((((np.sin((((np.sin((data["mean_abs_chgbatch_msignal"]))) - (data["signal_shift_+1"]))))) - (((((data["signal_shift_-1"]) * 2.0)) * (data["medianbatch_slices2"]))))) + (np.where(data["signal_shift_+1"] <= -998, data["medianbatch_slices2_msignal"], data["rangebatch_slices2_msignal"] )))) * 2.0)) +

                            0.050000*np.tanh(np.where(data["abs_avgbatch_msignal"] <= -998, (((data["rangebatch_slices2_msignal"]) + (data["medianbatch_slices2_msignal"]))/2.0), (((-((np.sin((data["maxbatch_msignal"])))))) * 2.0) )) +

                            0.050000*np.tanh(((np.cos((data["abs_avgbatch_msignal"]))) + (np.minimum(((np.where(data["abs_avgbatch_msignal"] > -998, data["stdbatch_slices2_msignal"], data["maxbatch_slices2"] ))), ((np.where(data["abs_avgbatch_msignal"] <= -998, data["abs_maxbatch"], data["rangebatch_slices2_msignal"] ))))))) +

                            0.050000*np.tanh(((data["mean_abs_chgbatch_msignal"]) - (np.minimum(((((data["signal_shift_+1"]) + (data["minbatch_slices2"])))), ((((((np.minimum(((data["abs_maxbatch_slices2_msignal"])), ((data["abs_avgbatch_msignal"])))) - (data["mean_abs_chgbatch_msignal"]))) - (np.sin(((((((((((data["signal_shift_+1"]) + (data["minbatch_slices2_msignal"]))) + (data["mean_abs_chgbatch_slices2_msignal"]))) > (data["mean_abs_chgbatch_slices2"]))*1.)) + (data["minbatch_slices2"])))))))))))) +

                            0.050000*np.tanh(((((data["signal_shift_+1_msignal"]) * 2.0)) * (np.where(np.where(((((data["signal_shift_+1_msignal"]) * 2.0)) * (np.where(data["mean_abs_chgbatch_msignal"] > -998, data["stdbatch_slices2_msignal"], data["abs_avgbatch_slices2_msignal"] ))) > -998, data["stdbatch_slices2_msignal"], data["signal_shift_+1_msignal"] ) > -998, ((data["stdbatch_slices2_msignal"]) * 2.0), ((data["signal_shift_+1_msignal"]) * (np.where(data["stdbatch_slices2_msignal"] > -998, data["stdbatch_slices2_msignal"], data["abs_avgbatch_msignal"] ))) )))) +

                            0.050000*np.tanh((((((data["signal_shift_-1"]) * 2.0)) + (((data["meanbatch_msignal"]) * (np.where(((np.tanh((((data["meanbatch_msignal"]) * (np.where((((data["abs_minbatch_slices2_msignal"]) <= ((((data["abs_minbatch_slices2_msignal"]) + (data["meanbatch_msignal"]))/2.0)))*1.) > -998, data["abs_minbatch_slices2_msignal"], np.where(data["signal"] > -998, data["abs_minbatch_slices2_msignal"], data["medianbatch_msignal"] ) )))))) * (data["stdbatch_msignal"])) > -998, data["abs_minbatch_slices2_msignal"], data["signal"] )))))/2.0)) +

                            0.050000*np.tanh(((data["stdbatch_slices2_msignal"]) + ((((data["stdbatch_slices2"]) <= (((np.maximum((((6.0))), ((np.maximum(((np.where(data["stdbatch_slices2_msignal"] > -998, (3.87789225578308105), data["stdbatch_slices2_msignal"] ))), ((data["maxtominbatch_slices2_msignal"]))))))) + (((data["rangebatch_slices2"]) - (data["stdbatch_slices2_msignal"]))))))*1.)))) +

                            0.050000*np.tanh(((((np.cos((data["meanbatch_slices2"]))) + (np.maximum(((data["abs_minbatch_slices2_msignal"])), ((np.cos((np.minimum(((((data["stdbatch_slices2"]) / 2.0))), ((((np.cos((((data["stdbatch_slices2_msignal"]) * 2.0)))) - ((9.0)))))))))))))) * (data["abs_maxbatch"]))) +

                            0.050000*np.tanh(((data["rangebatch_slices2_msignal"]) - (((((data["meanbatch_slices2"]) - (((np.where(np.tanh((((data["abs_maxbatch_msignal"]) - (data["meanbatch_slices2"])))) <= -998, data["abs_avgbatch_slices2_msignal"], np.cos((data["rangebatch_slices2_msignal"])) )) - (data["meanbatch_slices2"]))))) - (data["abs_avgbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(((data["signal"]) + (((((np.cos((np.maximum(((data["signal"])), ((np.maximum(((np.tanh((((((data["medianbatch_msignal"]) + (np.minimum((((4.23611021041870117))), ((data["mean_abs_chgbatch_msignal"])))))) / 2.0))))), ((data["medianbatch_msignal"]))))))))) * (((data["maxbatch_slices2"]) * (data["rangebatch_slices2_msignal"]))))) + (data["signal"]))))) +

                            0.050000*np.tanh((((2.0)) + ((-((np.where(data["maxtominbatch_msignal"] > -998, ((data["abs_minbatch_slices2_msignal"]) * (((np.maximum(((data["signal_shift_+1"])), ((data["abs_avgbatch_msignal"])))) * (data["maxtominbatch"])))), (-(((-((np.where((((2.0)) + (data["signal_shift_+1"])) > -998, data["maxtominbatch"], data["minbatch_msignal"] ))))))) ))))))) +

                            0.050000*np.tanh(np.where(data["stdbatch_slices2_msignal"] <= -998, data["rangebatch_slices2"], ((data["abs_maxbatch_slices2_msignal"]) * (((data["stdbatch_slices2_msignal"]) * (((np.maximum(((((((data["signal_shift_+1_msignal"]) * (data["abs_maxbatch_slices2_msignal"]))) - (np.minimum(((data["medianbatch_slices2_msignal"])), ((data["abs_maxbatch"]))))))), ((((data["minbatch_slices2"]) - (((data["abs_maxbatch_slices2_msignal"]) * 2.0))))))) * 2.0))))) )) +

                            0.050000*np.tanh((((((data["rangebatch_msignal"]) + (data["signal_shift_-1"]))/2.0)) * (np.minimum(((np.where(data["signal_shift_-1"] <= -998, (((data["rangebatch_msignal"]) + ((3.66988515853881836)))/2.0), np.tanh((np.sin((((data["signal_shift_-1"]) - (np.minimum(((data["meanbatch_msignal"])), ((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)))))))))) ))), ((np.tanh((data["abs_maxbatch_msignal"])))))))) +

                            0.050000*np.tanh(((np.sin((np.where(data["abs_maxbatch"] <= -998, data["maxbatch_slices2"], np.minimum(((data["maxbatch_slices2"])), ((data["stdbatch_slices2"]))) )))) * 2.0)) +

                            0.050000*np.tanh(((data["abs_maxbatch_slices2_msignal"]) - (((data["medianbatch_slices2"]) * (np.where(np.where(data["medianbatch_slices2"] <= -998, ((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0), data["mean_abs_chgbatch_slices2_msignal"] ) <= -998, data["signal_shift_-1"], data["signal_shift_-1"] )))))) +

                            0.050000*np.tanh(((((np.sin((data["signal_shift_-1"]))) + (((data["maxbatch_slices2_msignal"]) * (((data["abs_avgbatch_msignal"]) * (((np.sin((data["signal_shift_-1"]))) + (((np.sin((np.cos((data["abs_avgbatch_msignal"]))))) * (np.where(np.cos((data["abs_avgbatch_msignal"])) > -998, data["medianbatch_msignal"], ((data["maxbatch_slices2_msignal"]) * (data["signal"])) )))))))))))) * 2.0)) +

                            0.050000*np.tanh(((((((12.09984111785888672)) + (((data["maxtominbatch_slices2_msignal"]) * (((data["meanbatch_slices2"]) * 2.0)))))/2.0)) * ((((((data["mean_abs_chgbatch_slices2"]) > (((data["rangebatch_slices2_msignal"]) * ((12.09984111785888672)))))*1.)) + ((12.09984111785888672)))))) +

                            0.050000*np.tanh(np.where(np.where(data["abs_avgbatch_msignal"] > -998, data["minbatch_slices2_msignal"], np.where((2.26455736160278320) > -998, np.where(data["meanbatch_slices2"] > -998, ((((data["meanbatch_slices2"]) + (np.tanh((data["medianbatch_slices2"]))))) * (data["abs_avgbatch_msignal"])), data["abs_maxbatch_msignal"] ), data["rangebatch_slices2"] ) ) > -998, ((data["abs_avgbatch_msignal"]) * (((data["abs_maxbatch"]) * (np.cos((data["abs_avgbatch_msignal"])))))), data["medianbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(((data["stdbatch_slices2_msignal"]) + (np.where(data["stdbatch_slices2_msignal"] <= -998, data["stdbatch_msignal"], np.cos(((((np.maximum(((data["signal_shift_+1"])), ((np.where(data["stdbatch_slices2_msignal"] <= -998, data["stdbatch_msignal"], ((data["maxbatch_slices2"]) - (((data["maxbatch_slices2"]) * (np.sin((data["stdbatch_msignal"])))))) ))))) > (data["minbatch"]))*1.))) )))) +

                            0.050000*np.tanh((((((1.24717736244201660)) + ((((((((((1.24717736244201660)) + (((((np.where(((data["stdbatch_slices2"]) * 2.0) <= -998, (((data["abs_minbatch_slices2_msignal"]) + (data["signal_shift_-1"]))/2.0), np.cos((np.maximum(((data["signal_shift_-1"])), ((data["abs_avgbatch_msignal"]))))) )) + (((np.cos((data["abs_maxbatch"]))) * 2.0)))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh((((-((((data["stdbatch_slices2"]) * (((data["abs_maxbatch_slices2"]) * (((np.sin((data["mean_abs_chgbatch_msignal"]))) + (((((np.sin((np.maximum(((np.sin((np.maximum(((((np.sin((data["medianbatch_slices2_msignal"]))) * (data["stdbatch_slices2"])))), ((data["abs_maxbatch_slices2"]))))))), ((data["abs_maxbatch_slices2"])))))) * (data["signal_shift_-1"]))) * 2.0))))))))))) * 2.0)) +

                            0.050000*np.tanh(((np.cos((np.maximum(((data["signal_shift_+1"])), (((((((data["maxbatch_msignal"]) * 2.0)) <= (((((-((np.where(data["mean_abs_chgbatch_slices2"] > -998, ((((np.cos((data["signal_shift_+1"]))) / 2.0)) * 2.0), data["signal_shift_-1_msignal"] ))))) <= (data["mean_abs_chgbatch_slices2"]))*1.)))*1.))))))) * 2.0)) +

                            0.050000*np.tanh(((np.where(data["stdbatch_slices2_msignal"] <= -998, np.maximum(((((data["rangebatch_slices2"]) * (data["minbatch"])))), ((data["abs_maxbatch_msignal"]))), ((np.maximum(((data["meanbatch_msignal"])), ((data["signal_shift_+1"])))) * (data["stdbatch_slices2_msignal"])) )) * (((np.cos((data["meanbatch_msignal"]))) * 2.0)))) +

                            0.050000*np.tanh(((data["abs_avgbatch_slices2_msignal"]) * (((np.where(data["abs_avgbatch_slices2_msignal"] <= -998, ((data["meanbatch_slices2"]) + (data["stdbatch_msignal"])), np.where(data["signal_shift_+1"] > -998, np.where((3.0) > -998, data["signal_shift_+1"], data["signal"] ), ((data["abs_avgbatch_slices2_msignal"]) * (((data["meanbatch_slices2"]) - (data["signal_shift_+1"])))) ) )) - ((-((data["meanbatch_slices2"])))))))) +

                            0.050000*np.tanh(((data["stdbatch_slices2"]) + ((((((((((data["maxtominbatch_msignal"]) * 2.0)) * (data["medianbatch_slices2"]))) + ((7.87292909622192383)))) + (data["medianbatch_slices2"]))/2.0)))) +

                            0.050000*np.tanh(((((np.cos((np.maximum(((((((((((((np.cos((np.maximum(((data["meanbatch_msignal"])), ((data["meanbatch_msignal"])))))) + (data["meanbatch_msignal"]))/2.0)) > (((data["meanbatch_msignal"]) + ((-((data["maxtominbatch"])))))))*1.)) * 2.0)) * 2.0))), ((np.maximum(((((((-((data["medianbatch_slices2"])))) + (data["rangebatch_slices2"]))/2.0))), ((data["meanbatch_msignal"]))))))))) * (data["abs_maxbatch_slices2"]))) * 2.0)) +

                            0.050000*np.tanh(((data["maxbatch_slices2"]) + (np.maximum(((np.cos((data["abs_maxbatch_slices2"])))), ((data["maxbatch_slices2"])))))) +

                            0.050000*np.tanh(np.where(data["meanbatch_slices2"] <= -998, np.minimum(((((((data["minbatch_msignal"]) + (((data["abs_maxbatch_slices2"]) + (np.sin((data["mean_abs_chgbatch_msignal"]))))))) * (data["minbatch_slices2"])))), ((data["mean_abs_chgbatch_msignal"]))), (((12.16635227203369141)) * (np.sin((((data["mean_abs_chgbatch_msignal"]) - (((data["abs_maxbatch_slices2"]) + (np.sin((data["abs_maxbatch_slices2_msignal"])))))))))) )) +

                            0.050000*np.tanh((((((np.maximum(((np.where(data["stdbatch_slices2"] <= -998, np.cos((((np.maximum(((np.tanh((data["stdbatch_slices2"])))), ((data["maxbatch_msignal"])))) * 2.0))), data["rangebatch_slices2_msignal"] ))), ((((np.where(data["abs_minbatch_slices2_msignal"] > -998, data["rangebatch_slices2_msignal"], data["meanbatch_msignal"] )) * 2.0))))) * 2.0)) + (data["maxbatch_slices2"]))/2.0)) +

                            0.050000*np.tanh(np.cos((((data["maxtominbatch"]) + (np.minimum(((data["abs_maxbatch"])), ((data["stdbatch_msignal"])))))))) +

                            0.050000*np.tanh(((np.where(np.where(((np.cos((data["medianbatch_msignal"]))) * (data["abs_maxbatch_msignal"])) <= -998, data["medianbatch_msignal"], ((data["medianbatch_slices2_msignal"]) * 2.0) ) > -998, (((3.15710377693176270)) + (data["medianbatch_msignal"])), np.cos((data["medianbatch_msignal"])) )) * (((((((np.cos((data["medianbatch_msignal"]))) * (data["abs_maxbatch_msignal"]))) * 2.0)) + (data["medianbatch_msignal"]))))) +

                            0.050000*np.tanh(np.where(data["meanbatch_slices2_msignal"] <= -998, data["abs_maxbatch"], np.maximum(((data["signal_shift_+1"])), ((np.tanh((np.cos((np.where(((data["rangebatch_msignal"]) * (data["signal_shift_+1"])) <= -998, (((data["maxtominbatch_slices2_msignal"]) + (data["abs_maxbatch"]))/2.0), (((data["stdbatch_msignal"]) + (data["abs_maxbatch"]))/2.0) )))))))) )) +

                            0.050000*np.tanh(np.maximum(((np.where(((data["stdbatch_msignal"]) * 2.0) <= -998, data["rangebatch_slices2"], ((data["signal_shift_-1"]) * (np.cos((data["medianbatch_slices2"])))) ))), ((data["medianbatch_slices2_msignal"])))) +

                            0.050000*np.tanh(np.where((-((data["maxbatch_msignal"]))) <= -998, ((data["maxbatch_msignal"]) * (((((data["signal_shift_+1"]) + (data["medianbatch_slices2_msignal"]))) * (data["medianbatch_slices2_msignal"])))), ((((np.cos((data["abs_avgbatch_slices2_msignal"]))) * (data["medianbatch_slices2_msignal"]))) * (((data["signal_shift_+1"]) + ((5.0))))) )) +

                            0.050000*np.tanh(np.where(data["meanbatch_msignal"] <= -998, (((data["maxtominbatch"]) <= (((np.minimum(((np.maximum(((data["meanbatch_msignal"])), ((data["abs_maxbatch_slices2"]))))), (((-((data["meanbatch_slices2_msignal"]))))))) * (np.minimum(((data["medianbatch_msignal"])), ((data["mean_abs_chgbatch_slices2"])))))))*1.), ((np.sin((data["rangebatch_msignal"]))) * (data["meanbatch_msignal"])) )) +

                            0.050000*np.tanh(((np.cos((np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, np.where(np.where(data["minbatch_slices2_msignal"] <= -998, data["rangebatch_slices2"], np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, (2.75588107109069824), ((data["signal"]) * ((-((data["abs_avgbatch_slices2_msignal"]))))) ) ) <= -998, data["mean_abs_chgbatch_msignal"], data["abs_avgbatch_slices2_msignal"] ), (((data["maxtominbatch_slices2"]) + (data["minbatch_slices2_msignal"]))/2.0) )))) * (data["meanbatch_slices2_msignal"]))) +

                            0.050000*np.tanh((((np.where(data["abs_maxbatch_slices2"] <= -998, ((data["abs_maxbatch"]) - ((((np.tanh((data["abs_maxbatch"]))) + (data["rangebatch_slices2"]))/2.0))), (8.0) )) + (np.maximum(((data["signal_shift_+1"])), ((np.where(data["rangebatch_slices2"] > -998, (((-((data["rangebatch_slices2"])))) * (data["stdbatch_msignal"])), data["abs_maxbatch_slices2"] ))))))/2.0)) +

                            0.050000*np.tanh(np.where(((data["medianbatch_slices2"]) * 2.0) <= -998, np.where(data["signal_shift_+1_msignal"] > -998, ((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, np.maximum((((((data["signal_shift_+1"]) <= (data["maxbatch_slices2"]))*1.))), ((data["signal_shift_+1"]))), data["mean_abs_chgbatch_msignal"] )) - (data["meanbatch_msignal"])), np.sin((data["stdbatch_slices2"])) ), ((data["medianbatch_slices2"]) * (np.minimum(((data["meanbatch_msignal"])), ((np.cos((data["meanbatch_msignal"]))))))) )) +

                            0.050000*np.tanh(((np.where(data["abs_minbatch_msignal"] > -998, np.where(data["signal_shift_+1"] > -998, data["rangebatch_msignal"], data["meanbatch_slices2_msignal"] ), data["signal_shift_+1"] )) * (data["signal_shift_+1"]))) +

                            0.050000*np.tanh(((data["meanbatch_msignal"]) * (np.tanh((np.sin((np.where((((data["meanbatch_msignal"]) + (data["abs_avgbatch_slices2"]))/2.0) <= -998, data["meanbatch_msignal"], np.where(np.sin((data["medianbatch_msignal"])) <= -998, ((data["meanbatch_msignal"]) * (data["signal"])), data["rangebatch_msignal"] ) )))))))) +

                            0.050000*np.tanh(((((data["maxbatch_slices2_msignal"]) - (data["abs_avgbatch_slices2_msignal"]))) - (((data["meanbatch_slices2"]) * (np.where(((data["signal_shift_+1"]) - (data["abs_avgbatch_slices2_msignal"])) > -998, data["signal_shift_+1"], (((3.0)) * 2.0) )))))) +

                            0.050000*np.tanh(((np.maximum(((((data["signal_shift_-1"]) * (data["stdbatch_msignal"])))), ((np.where(((data["minbatch_slices2"]) * 2.0) <= -998, data["abs_maxbatch_slices2"], data["medianbatch_slices2_msignal"] ))))) * 2.0)) +

                            0.050000*np.tanh(((np.where(data["abs_avgbatch_slices2"] > -998, ((np.cos((((np.where((9.74521446228027344) > -998, data["stdbatch_slices2_msignal"], data["abs_avgbatch_msignal"] )) * 2.0)))) * 2.0), np.where(data["abs_avgbatch_slices2"] <= -998, data["stdbatch_slices2"], data["signal_shift_+1"] ) )) * 2.0)) +

                            0.050000*np.tanh((((((((np.cos((np.where(((np.cos((data["signal_shift_-1"]))) * 2.0) > -998, (((data["mean_abs_chgbatch_msignal"]) + (((data["meanbatch_slices2_msignal"]) * 2.0)))/2.0), data["mean_abs_chgbatch_slices2"] )))) + (np.maximum(((np.sin(((0.0))))), ((np.sin((np.minimum(((data["meanbatch_slices2_msignal"])), ((data["minbatch_slices2_msignal"]))))))))))/2.0)) * 2.0)) * 2.0)) +

                            0.050000*np.tanh((((((((data["medianbatch_slices2_msignal"]) * (np.where((((np.where(np.cos((data["abs_avgbatch_msignal"])) <= -998, ((data["abs_avgbatch_msignal"]) * (data["stdbatch_slices2"])), data["maxbatch_msignal"] )) + (data["abs_maxbatch_msignal"]))/2.0) <= -998, (((data["maxbatch_slices2_msignal"]) > (data["mean_abs_chgbatch_slices2_msignal"]))*1.), data["stdbatch_slices2_msignal"] )))) / 2.0)) + (data["stdbatch_msignal"]))/2.0)) +

                            0.050000*np.tanh(np.maximum(((((data["signal_shift_+1"]) * 2.0))), ((np.cos((data["rangebatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(((((data["rangebatch_slices2_msignal"]) - (data["abs_maxbatch_msignal"]))) * (np.cos((((data["abs_maxbatch_msignal"]) * (np.minimum(((np.maximum(((np.cos((data["medianbatch_slices2"])))), ((data["mean_abs_chgbatch_slices2"]))))), ((np.where(data["maxtominbatch_msignal"] <= -998, np.minimum(((data["medianbatch_slices2"])), ((data["meanbatch_msignal"]))), data["maxtominbatch_slices2"] ))))))))))) +

                            0.050000*np.tanh(((np.sin((np.maximum(((data["abs_avgbatch_slices2"])), ((data["signal_shift_-1"])))))) * (((np.where(data["abs_avgbatch_slices2"] <= -998, ((data["abs_maxbatch"]) / 2.0), data["signal_shift_-1"] )) + (((data["signal_shift_-1"]) + (((((data["signal_shift_-1"]) + (((np.sin((data["maxbatch_slices2"]))) * (data["signal_shift_-1"]))))) + (np.cos((data["minbatch_slices2"]))))))))))) +

                            0.050000*np.tanh((((((((data["maxtominbatch_slices2_msignal"]) / 2.0)) * (data["maxbatch_msignal"]))) + ((((((((data["medianbatch_slices2_msignal"]) + (((data["abs_minbatch_slices2_msignal"]) * (data["medianbatch_slices2"]))))/2.0)) + (data["abs_maxbatch"]))) + (np.cos((data["medianbatch_slices2_msignal"]))))))/2.0)) +

                            0.050000*np.tanh(np.maximum(((((((data["signal_shift_-1"]) * (data["abs_minbatch_slices2_msignal"]))) + (((np.cos((np.maximum(((data["meanbatch_msignal"])), (((((data["maxbatch_slices2"]) <= (data["abs_minbatch_slices2_msignal"]))*1.))))))) * 2.0))))), ((data["abs_minbatch_msignal"])))) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) * (((np.minimum(((((((((6.0)) - (data["abs_maxbatch"]))) <= (data["abs_maxbatch"]))*1.))), (((((6.0)) - (data["abs_maxbatch"])))))) + (np.minimum(((data["signal_shift_+1"])), ((np.tanh((data["signal_shift_+1"])))))))))) +

                            0.050000*np.tanh((((7.0)) * (((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, np.maximum(((data["maxbatch_msignal"])), ((np.sin((((((6.0)) + ((((np.cos((data["signal_shift_-1"]))) <= ((7.0)))*1.)))/2.0)))))), ((np.cos((data["signal_shift_-1"]))) / 2.0) )) * 2.0)))) +

                            0.050000*np.tanh(np.tanh((data["rangebatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((np.maximum(((np.where(data["abs_minbatch_slices2_msignal"] <= -998, data["signal_shift_-1"], ((np.cos((data["meanbatch_msignal"]))) * 2.0) ))), ((((np.where(data["medianbatch_slices2_msignal"] <= -998, ((np.sin((data["stdbatch_msignal"]))) + (data["medianbatch_slices2_msignal"])), ((np.cos((data["meanbatch_msignal"]))) * 2.0) )) + (data["abs_minbatch_slices2_msignal"])))))) * 2.0)) +

                            0.050000*np.tanh(np.maximum(((((np.where(((data["mean_abs_chgbatch_msignal"]) + (np.cos((((np.sin((data["abs_avgbatch_msignal"]))) * 2.0))))) <= -998, data["maxbatch_slices2_msignal"], np.where(np.where(data["minbatch_msignal"] > -998, data["maxbatch_slices2_msignal"], data["abs_avgbatch_msignal"] ) <= -998, data["abs_avgbatch_msignal"], data["abs_minbatch_msignal"] ) )) * 2.0))), ((np.sin((((data["signal_shift_+1"]) * 2.0))))))) +

                            0.050000*np.tanh(((np.where((0.67817705869674683) <= -998, (0.67817705869674683), ((np.sin((np.minimum(((data["minbatch_slices2_msignal"])), (((-((data["stdbatch_msignal"]))))))))) * 2.0) )) * 2.0)) +

                            0.050000*np.tanh(np.where(data["stdbatch_slices2_msignal"] > -998, data["signal_shift_-1"], data["medianbatch_slices2"] )) +

                            0.050000*np.tanh(((np.maximum(((data["signal_shift_+1"])), ((data["medianbatch_msignal"])))) * 2.0)) +

                            0.050000*np.tanh(((((np.where(data["rangebatch_slices2"] <= -998, ((data["meanbatch_slices2_msignal"]) - (data["medianbatch_msignal"])), data["meanbatch_slices2_msignal"] )) - (((((np.sin((data["medianbatch_msignal"]))) * 2.0)) * (data["rangebatch_slices2"]))))) * (data["abs_minbatch_msignal"]))) +

                            0.050000*np.tanh(np.sin((((data["maxtominbatch"]) + (np.where(((((((data["abs_maxbatch_slices2"]) + (((data["maxtominbatch"]) - (data["abs_minbatch_slices2_msignal"]))))/2.0)) + (data["minbatch_slices2_msignal"]))/2.0) <= -998, data["abs_minbatch_slices2_msignal"], np.where(np.where(data["abs_maxbatch_slices2"] <= -998, data["signal"], data["minbatch_slices2_msignal"] ) <= -998, data["minbatch_slices2_msignal"], data["minbatch_slices2_msignal"] ) )))))) +

                            0.050000*np.tanh(((np.cos((data["maxbatch_msignal"]))) * (np.where(((data["medianbatch_slices2_msignal"]) - (data["medianbatch_slices2_msignal"])) <= -998, np.where(data["maxtominbatch_slices2_msignal"] <= -998, np.cos((((data["medianbatch_slices2_msignal"]) * 2.0))), (-((data["maxbatch_msignal"]))) ), (-((data["medianbatch_slices2_msignal"]))) )))) +

                            0.050000*np.tanh(((np.cos((np.where(((data["signal_shift_-1"]) * (((data["maxtominbatch_msignal"]) / 2.0))) <= -998, data["signal_shift_-1"], np.maximum((((((-((data["signal_shift_-1"])))) * (((data["signal_shift_-1"]) / 2.0))))), ((data["maxtominbatch_msignal"]))) )))) * 2.0)) +

                            0.050000*np.tanh(np.sin((np.minimum(((np.minimum((((((((np.tanh((data["stdbatch_msignal"]))) * (np.cos((data["medianbatch_slices2"]))))) > (data["abs_avgbatch_msignal"]))*1.))), ((data["minbatch_msignal"]))))), ((data["minbatch_msignal"])))))) +

                            0.050000*np.tanh(((((data["abs_avgbatch_slices2"]) + (data["medianbatch_msignal"]))) * (np.where(data["minbatch_msignal"] > -998, data["medianbatch_msignal"], ((np.minimum(((np.where(data["signal_shift_-1_msignal"] > -998, data["abs_avgbatch_slices2"], ((data["abs_avgbatch_slices2"]) * (data["medianbatch_msignal"])) ))), ((data["medianbatch_msignal"])))) - ((6.0))) )))) +

                            0.050000*np.tanh(np.sin((np.where(np.sin((np.where(((data["rangebatch_slices2"]) * (((data["maxbatch_slices2"]) - (data["signal_shift_-1"])))) > -998, ((data["stdbatch_slices2"]) * (data["meanbatch_msignal"])), ((data["stdbatch_msignal"]) * (data["mean_abs_chgbatch_msignal"])) ))) > -998, data["minbatch_msignal"], ((data["stdbatch_msignal"]) * (data["mean_abs_chgbatch_msignal"])) )))) +

                            0.050000*np.tanh(np.cos((np.maximum(((np.where(data["abs_avgbatch_slices2_msignal"] <= -998, ((np.sin((data["medianbatch_msignal"]))) + (data["mean_abs_chgbatch_slices2_msignal"])), data["signal_shift_-1"] ))), ((np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, np.cos((((data["medianbatch_msignal"]) / 2.0))), data["abs_avgbatch_slices2_msignal"] ))))))) +

                            0.050000*np.tanh(np.maximum(((((np.where(np.minimum(((data["minbatch_msignal"])), ((np.minimum(((np.maximum(((data["stdbatch_slices2_msignal"])), ((data["abs_maxbatch"]))))), ((np.maximum(((data["minbatch_msignal"])), ((data["medianbatch_msignal"]))))))))) > -998, np.sin((data["abs_avgbatch_slices2"])), np.where(np.minimum(((data["maxbatch_msignal"])), ((((data["abs_avgbatch_slices2"]) * 2.0)))) <= -998, (4.27560997009277344), np.sin((data["abs_avgbatch_slices2"])) ) )) * 2.0))), ((np.sin((data["abs_avgbatch_slices2"])))))) +

                            0.050000*np.tanh(((((np.sin((((((((data["abs_minbatch_slices2_msignal"]) > (np.sin((data["meanbatch_slices2_msignal"]))))*1.)) + (data["rangebatch_msignal"]))/2.0)))) + (np.maximum(((np.sin((data["medianbatch_msignal"])))), ((np.minimum(((data["maxbatch_msignal"])), ((((np.sin((np.maximum(((data["abs_minbatch_slices2_msignal"])), ((data["abs_minbatch_slices2_msignal"])))))) * 2.0)))))))))) * 2.0)))    

      

    def GP_class_4(self,data):

        return self.Output( -2.516509 +

                            0.050000*np.tanh(((np.where(data["meanbatch_msignal"] > -998, data["meanbatch_slices2"], ((np.where(data["meanbatch_msignal"] > -998, data["meanbatch_slices2"], ((np.minimum(((np.maximum(((data["medianbatch_slices2"])), ((((((data["abs_maxbatch_slices2"]) * 2.0)) / 2.0)))))), (((-(((((data["medianbatch_slices2"]) + ((-((data["meanbatch_slices2"])))))/2.0)))))))) / 2.0) )) / 2.0) )) + (data["medianbatch_slices2"]))) +

                            0.050000*np.tanh(np.where(data["medianbatch_slices2"] > -998, data["medianbatch_slices2"], np.where(data["signal"] > -998, data["signal_shift_+1"], data["abs_avgbatch_slices2_msignal"] ) )) +

                            0.050000*np.tanh(np.where(np.maximum(((data["signal_shift_-1"])), (((((data["mean_abs_chgbatch_slices2_msignal"]) + (data["abs_avgbatch_msignal"]))/2.0)))) <= -998, np.tanh((data["signal_shift_-1"])), data["signal"] )) +

                            0.050000*np.tanh(np.where(((data["signal_shift_+1"]) + (data["meanbatch_slices2"])) > -998, data["meanbatch_slices2"], np.where(np.maximum((((((data["meanbatch_slices2"]) > (data["abs_maxbatch_msignal"]))*1.))), ((data["stdbatch_slices2"]))) > -998, np.where((5.59334421157836914) > -998, data["medianbatch_slices2"], data["maxbatch_msignal"] ), data["medianbatch_slices2"] ) )) +

                            0.050000*np.tanh((((((((8.0)) * (data["abs_avgbatch_slices2"]))) / 2.0)) * (np.where(data["signal"] > -998, np.sin((data["medianbatch_slices2"])), np.where(data["meanbatch_slices2"] > -998, np.tanh((np.sin((((np.where(data["maxtominbatch_msignal"] > -998, data["signal"], data["medianbatch_slices2"] )) * 2.0))))), np.cos((data["maxbatch_msignal"])) ) )))) +

                            0.050000*np.tanh(np.where((-((data["minbatch_slices2_msignal"]))) > -998, data["medianbatch_slices2"], np.where(((data["rangebatch_slices2"]) - (data["maxbatch_slices2_msignal"])) <= -998, np.minimum(((data["signal"])), ((data["abs_minbatch_msignal"]))), ((np.where(data["signal_shift_+1"] > -998, ((data["mean_abs_chgbatch_slices2"]) * (data["signal_shift_-1_msignal"])), data["signal_shift_+1"] )) / 2.0) ) )) +

                            0.050000*np.tanh(np.where(np.sin(((((data["signal"]) > (np.where((((data["signal_shift_+1"]) > (data["signal_shift_+1"]))*1.) <= -998, data["signal"], np.minimum(((data["rangebatch_msignal"])), ((((data["meanbatch_slices2"]) / 2.0)))) )))*1.))) <= -998, ((((((data["signal_shift_+1"]) + (data["medianbatch_slices2"]))/2.0)) + (data["signal_shift_+1"]))/2.0), data["medianbatch_slices2"] )) +

                            0.050000*np.tanh(np.where(data["signal_shift_-1"] <= -998, data["abs_avgbatch_slices2_msignal"], np.minimum(((data["signal_shift_-1"])), (((((data["medianbatch_slices2"]) + (np.minimum(((np.sin((np.where(data["signal"] <= -998, ((data["stdbatch_msignal"]) / 2.0), data["signal"] ))))), ((data["abs_maxbatch"])))))/2.0)))) )) +

                            0.050000*np.tanh(np.sin((np.where(((data["meanbatch_slices2_msignal"]) * (np.minimum(((data["meanbatch_slices2"])), ((np.sin((data["meanbatch_slices2"]))))))) > -998, ((np.sin((np.where(data["stdbatch_slices2_msignal"] > -998, data["meanbatch_slices2"], data["abs_maxbatch"] )))) * 2.0), data["abs_maxbatch_slices2"] )))) +

                            0.050000*np.tanh(((((np.sin((data["abs_maxbatch_slices2_msignal"]))) * 2.0)) + ((((data["minbatch_slices2_msignal"]) > (((data["meanbatch_slices2"]) - (np.minimum(((data["abs_maxbatch"])), ((data["minbatch_slices2_msignal"])))))))*1.)))) +

                            0.050000*np.tanh(((data["maxtominbatch_msignal"]) * (np.cos((((data["abs_minbatch_msignal"]) - ((((np.where(np.where((((data["maxtominbatch_msignal"]) > (data["minbatch_msignal"]))*1.) <= -998, data["maxbatch_slices2_msignal"], data["signal_shift_-1"] ) > -998, data["maxtominbatch"], np.sin((data["abs_maxbatch_slices2"])) )) > ((((data["signal_shift_-1"]) + (((data["maxbatch_slices2_msignal"]) + (data["maxtominbatch_msignal"]))))/2.0)))*1.)))))))) +

                            0.050000*np.tanh(((((np.sin((np.maximum(((np.maximum(((np.maximum(((data["abs_maxbatch_msignal"])), ((data["signal"]))))), ((data["abs_maxbatch_msignal"]))))), ((data["medianbatch_msignal"])))))) * (((data["abs_maxbatch_msignal"]) * (data["stdbatch_slices2"]))))) - (np.cos((data["signal_shift_+1"]))))) +

                            0.050000*np.tanh(np.where(((data["abs_maxbatch"]) * (((data["medianbatch_slices2"]) * (data["stdbatch_slices2"])))) <= -998, np.where(np.sin((data["medianbatch_slices2"])) <= -998, data["rangebatch_slices2_msignal"], ((data["meanbatch_msignal"]) * 2.0) ), ((data["rangebatch_slices2_msignal"]) * (np.sin((np.where(((data["minbatch"]) * 2.0) > -998, data["medianbatch_slices2"], data["signal"] ))))) )) +

                            0.050000*np.tanh(((data["abs_maxbatch_slices2_msignal"]) * (np.sin((np.where(data["abs_maxbatch_msignal"] > -998, data["medianbatch_slices2"], ((np.maximum(((data["minbatch"])), ((((((data["signal_shift_+1_msignal"]) / 2.0)) / 2.0))))) / 2.0) )))))) +

                            0.050000*np.tanh(((((data["minbatch_slices2_msignal"]) * (np.where(data["meanbatch_msignal"] > -998, data["stdbatch_slices2_msignal"], np.minimum(((data["signal_shift_+1_msignal"])), (((((((((data["signal_shift_-1_msignal"]) * (data["minbatch_slices2_msignal"]))) * 2.0)) > (np.where(np.cos((data["stdbatch_slices2_msignal"])) <= -998, ((data["minbatch_slices2"]) * 2.0), data["stdbatch_slices2"] )))*1.)))) )))) * 2.0)) +

                            0.050000*np.tanh(((data["stdbatch_msignal"]) * (((np.sin((np.where(((np.sin((((((data["stdbatch_msignal"]) * (np.where(data["maxbatch_msignal"] > -998, data["minbatch_msignal"], np.maximum(((data["minbatch_msignal"])), ((np.tanh((data["abs_maxbatch_slices2"]))))) )))) * 2.0)))) * 2.0) > -998, data["abs_maxbatch_slices2"], data["stdbatch_msignal"] )))) * 2.0)))) +

                            0.050000*np.tanh((((np.sin(((((data["minbatch_slices2"]) <= (np.sin((data["minbatch_slices2"]))))*1.)))) + (data["signal"]))/2.0)) +

                            0.050000*np.tanh((((((-((np.where(((((((data["abs_avgbatch_slices2_msignal"]) * 2.0)) * 2.0)) * 2.0) > -998, data["abs_minbatch_msignal"], np.where(data["meanbatch_slices2"] > -998, ((np.minimum(((((np.tanh((data["abs_avgbatch_slices2_msignal"]))) * 2.0))), ((data["abs_minbatch_msignal"])))) - (data["abs_avgbatch_slices2_msignal"])), ((data["abs_avgbatch_slices2_msignal"]) * 2.0) ) ))))) - (((data["abs_avgbatch_slices2_msignal"]) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh((((((10.0)) * (np.tanh((np.tanh((((data["mean_abs_chgbatch_slices2"]) - (np.maximum(((data["abs_avgbatch_msignal"])), (((((np.sin((np.maximum(((data["minbatch_slices2"])), ((data["abs_avgbatch_msignal"])))))) + (((data["minbatch_slices2"]) * (data["abs_avgbatch_msignal"]))))/2.0))))))))))))) * 2.0)) +

                            0.050000*np.tanh(np.where(data["abs_avgbatch_msignal"] <= -998, ((((((np.sin((data["medianbatch_slices2"]))) > ((((((np.cos(((-((data["medianbatch_slices2"])))))) * (data["medianbatch_slices2"]))) > (((data["meanbatch_slices2"]) / 2.0)))*1.)))*1.)) <= (np.maximum(((data["abs_avgbatch_msignal"])), ((((data["abs_avgbatch_msignal"]) * 2.0))))))*1.), np.sin((data["meanbatch_slices2"])) )) +

                            0.050000*np.tanh(np.minimum(((np.sin((data["signal"])))), ((np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["meanbatch_slices2"], np.where(data["abs_maxbatch"] > -998, data["meanbatch_slices2"], np.where(data["meanbatch_slices2"] <= -998, data["maxtominbatch"], data["signal_shift_-1"] ) ) ))))) +

                            0.050000*np.tanh(((((np.sin((data["abs_maxbatch_msignal"]))) - (data["abs_avgbatch_msignal"]))) * ((((np.maximum(((((((np.sin((data["abs_maxbatch_slices2_msignal"]))) - (data["abs_avgbatch_msignal"]))) - (np.cos((np.where(data["rangebatch_slices2_msignal"] <= -998, data["abs_maxbatch_msignal"], ((data["medianbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2_msignal"])) ))))))), ((data["medianbatch_msignal"])))) + (((data["rangebatch_slices2_msignal"]) * 2.0)))/2.0)))) +

                            0.050000*np.tanh(np.minimum(((np.minimum(((np.minimum(((((np.minimum(((data["medianbatch_slices2"])), ((((data["medianbatch_slices2"]) * (((data["abs_maxbatch"]) * (np.sin((np.sin((data["signal"])))))))))))) * (data["abs_maxbatch_msignal"])))), ((np.minimum(((data["medianbatch_slices2"])), (((((6.26549148559570312)) * (np.sin((data["abs_maxbatch_msignal"])))))))))))), ((data["signal"]))))), (((((6.26549148559570312)) * (data["signal"])))))) +

                            0.050000*np.tanh(np.where((((data["medianbatch_msignal"]) <= (data["signal_shift_+1_msignal"]))*1.) > -998, data["medianbatch_slices2"], ((((np.sin((data["medianbatch_slices2"]))) / 2.0)) * 2.0) )) +

                            0.050000*np.tanh(((np.where(data["minbatch_msignal"] <= -998, data["abs_minbatch_slices2"], np.where(data["abs_maxbatch_msignal"] <= -998, data["abs_maxbatch_msignal"], np.where(data["abs_minbatch_slices2"] <= -998, np.sin((data["signal"])), np.where(np.sin((data["signal_shift_-1"])) <= -998, data["rangebatch_slices2_msignal"], np.where(data["abs_maxbatch_msignal"] <= -998, np.where(data["abs_minbatch_slices2"] <= -998, np.cos((data["signal_shift_+1"])), data["signal"] ), np.sin((data["abs_maxbatch_msignal"])) ) ) ) ) )) * 2.0)) +

                            0.050000*np.tanh(np.sin((np.where(np.sin((data["meanbatch_slices2"])) > -998, data["meanbatch_slices2"], (((data["rangebatch_slices2"]) > (((np.sin((np.sin((data["meanbatch_msignal"]))))) / 2.0)))*1.) )))) +

                            0.050000*np.tanh(((data["abs_maxbatch_slices2_msignal"]) * (np.where(np.tanh((data["maxbatch_slices2"])) <= -998, ((data["meanbatch_slices2"]) * 2.0), np.sin((data["abs_maxbatch_msignal"])) )))) +

                            0.050000*np.tanh(np.sin((np.where(np.where(np.maximum(((data["abs_maxbatch_msignal"])), ((np.where((9.63134956359863281) <= -998, np.sin((data["medianbatch_slices2_msignal"])), (9.63134956359863281) )))) <= -998, data["abs_maxbatch_msignal"], (-((np.maximum(((data["abs_maxbatch_msignal"])), ((np.where((10.77424335479736328) <= -998, (9.63134956359863281), data["medianbatch_slices2_msignal"] ))))))) ) <= -998, data["signal_shift_+1"], data["abs_maxbatch_msignal"] )))) +

                            0.050000*np.tanh(np.where((-((np.sin((data["maxtominbatch_slices2"]))))) <= -998, data["signal_shift_-1"], ((np.sin((((data["abs_minbatch_slices2_msignal"]) + (data["abs_minbatch_msignal"]))))) * 2.0) )) +

                            0.050000*np.tanh(((np.where(data["abs_avgbatch_slices2_msignal"] <= -998, data["medianbatch_slices2"], ((((((data["stdbatch_msignal"]) * (np.sin((data["abs_minbatch_msignal"]))))) - (((data["abs_avgbatch_slices2_msignal"]) * (np.where(data["minbatch_msignal"] <= -998, data["minbatch_msignal"], data["abs_avgbatch_slices2_msignal"] )))))) * 2.0) )) * 2.0)) +

                            0.050000*np.tanh(((np.where(np.sin((data["abs_maxbatch"])) <= -998, data["mean_abs_chgbatch_msignal"], np.where(data["signal_shift_+1"] <= -998, ((((data["mean_abs_chgbatch_msignal"]) * (np.sin((np.sin((data["abs_maxbatch"]))))))) * 2.0), ((data["abs_minbatch_msignal"]) - ((-((data["stdbatch_msignal"]))))) ) )) * (np.sin((data["abs_maxbatch"]))))) +

                            0.050000*np.tanh(((((((((((((np.cos((((((data["abs_avgbatch_msignal"]) * 2.0)) - (data["abs_avgbatch_msignal"]))))) - (data["abs_avgbatch_msignal"]))) * 2.0)) - (((data["meanbatch_slices2_msignal"]) * (data["medianbatch_msignal"]))))) - ((-((data["medianbatch_msignal"])))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((np.cos((np.where(np.maximum(((data["medianbatch_msignal"])), ((((data["minbatch_msignal"]) + ((5.0)))))) > -998, data["rangebatch_msignal"], np.cos((np.minimum(((data["signal"])), ((data["abs_maxbatch_slices2_msignal"]))))) )))) * 2.0)) +

                            0.050000*np.tanh(((((data["signal"]) * (np.sin((np.maximum(((((data["signal_shift_-1_msignal"]) * 2.0))), ((np.maximum(((data["signal"])), ((data["abs_maxbatch_slices2_msignal"]))))))))))) * ((((np.sin((data["signal_shift_-1_msignal"]))) + (data["signal"]))/2.0)))) +

                            0.050000*np.tanh(((np.where(np.tanh((((((np.where(data["rangebatch_slices2"] > -998, np.sin((data["abs_maxbatch_slices2_msignal"])), ((data["signal_shift_+1"]) / 2.0) )) * 2.0)) * 2.0))) > -998, np.sin((data["rangebatch_slices2"])), ((data["maxbatch_slices2"]) * 2.0) )) * 2.0)) +

                            0.050000*np.tanh((((data["minbatch_slices2"]) + (((data["maxbatch_slices2_msignal"]) * ((((np.minimum(((data["minbatch_slices2"])), ((data["stdbatch_slices2_msignal"])))) + (((data["maxbatch_slices2_msignal"]) * ((((-((data["rangebatch_slices2"])))) * (np.sin((((data["abs_maxbatch_msignal"]) * 2.0)))))))))/2.0)))))/2.0)) +

                            0.050000*np.tanh(((((data["abs_maxbatch_slices2"]) * (np.minimum(((data["minbatch"])), ((data["abs_maxbatch_msignal"])))))) * (np.sin((np.where(data["maxbatch_slices2_msignal"] > -998, ((data["maxbatch_slices2_msignal"]) * 2.0), data["maxtominbatch_slices2"] )))))) +

                            0.050000*np.tanh(np.where(data["abs_minbatch_slices2_msignal"] > -998, ((np.sin((data["minbatch_msignal"]))) * 2.0), ((((np.sin((np.sin(((-((np.sin((data["rangebatch_msignal"])))))))))) * 2.0)) + (np.minimum(((data["stdbatch_slices2_msignal"])), ((data["stdbatch_slices2_msignal"]))))) )) +

                            0.050000*np.tanh(np.where(((data["abs_maxbatch"]) / 2.0) > -998, ((np.where((7.26189327239990234) > -998, data["stdbatch_msignal"], data["maxtominbatch_slices2_msignal"] )) * (np.minimum((((((7.26189327239990234)) * ((-((np.sin((((data["abs_maxbatch"]) / 2.0)))))))))), ((data["maxbatch_slices2_msignal"]))))), (((((((data["abs_maxbatch"]) / 2.0)) * 2.0)) <= ((7.26189327239990234)))*1.) )) +

                            0.050000*np.tanh(((data["abs_maxbatch_msignal"]) * (np.sin((np.where(data["abs_maxbatch_msignal"] <= -998, data["abs_maxbatch_msignal"], ((np.sin((np.where(data["signal_shift_+1"] <= -998, data["abs_maxbatch_msignal"], ((data["abs_minbatch_msignal"]) * 2.0) )))) * 2.0) )))))) +

                            0.050000*np.tanh(((((np.sin((np.where(np.maximum(((np.where(((data["rangebatch_slices2_msignal"]) - (np.sin((data["minbatch_slices2_msignal"])))) > -998, np.sin((((np.sin((data["abs_maxbatch_slices2_msignal"]))) * 2.0))), (((np.tanh((data["medianbatch_slices2"]))) + ((((data["signal_shift_-1"]) + (data["abs_maxbatch_slices2_msignal"]))/2.0)))/2.0) ))), ((data["abs_maxbatch_msignal"]))) > -998, np.sin((data["abs_maxbatch_slices2_msignal"])), data["signal_shift_-1"] )))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((((np.where(np.sin((((np.sin((np.sin((data["maxbatch_msignal"]))))) * 2.0))) <= -998, ((np.sin((((data["minbatch_msignal"]) * 2.0)))) * 2.0), np.sin((((np.sin((data["minbatch_msignal"]))) * 2.0))) )) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((data["abs_maxbatch"]) * (np.cos((np.where((7.0) <= -998, data["stdbatch_msignal"], (((((np.cos(((-((((data["signal_shift_-1"]) + (data["abs_maxbatch_slices2"])))))))) > (((np.cos((data["meanbatch_msignal"]))) * ((-((data["abs_maxbatch_slices2_msignal"])))))))*1.)) + (data["abs_maxbatch_slices2"])) )))))) +

                            0.050000*np.tanh(((np.where(np.where(np.minimum(((data["maxbatch_slices2"])), ((np.tanh((data["medianbatch_slices2_msignal"]))))) <= -998, data["stdbatch_msignal"], ((data["abs_maxbatch_slices2"]) / 2.0) ) <= -998, data["medianbatch_slices2_msignal"], ((np.where(data["rangebatch_slices2"] <= -998, np.where(data["signal_shift_-1"] <= -998, data["minbatch_msignal"], ((data["abs_minbatch_msignal"]) * 2.0) ), np.cos((((np.tanh((data["stdbatch_msignal"]))) - (data["abs_maxbatch"])))) )) * 2.0) )) * 2.0)) +

                            0.050000*np.tanh(((((data["meanbatch_slices2"]) + (((data["medianbatch_msignal"]) * (data["medianbatch_msignal"]))))) + (((((data["meanbatch_slices2"]) + (((data["meanbatch_msignal"]) * (data["meanbatch_slices2"]))))) * (((data["medianbatch_msignal"]) * (data["abs_minbatch_slices2_msignal"]))))))) +

                            0.050000*np.tanh(np.where(np.sin((data["maxtominbatch_slices2"])) <= -998, ((data["maxbatch_msignal"]) * (np.tanh(((5.0))))), (((5.0)) + (((data["abs_avgbatch_slices2"]) * (np.where(data["medianbatch_msignal"] > -998, ((((data["mean_abs_chgbatch_msignal"]) + (data["abs_minbatch_slices2"]))) + (np.tanh((data["abs_avgbatch_msignal"])))), np.tanh((data["abs_avgbatch_msignal"])) ))))) )) +

                            0.050000*np.tanh(((np.where(data["abs_avgbatch_slices2"] > -998, (((data["rangebatch_msignal"]) + (np.maximum(((data["maxbatch_slices2_msignal"])), (((((data["signal_shift_+1_msignal"]) + ((((data["signal_shift_-1"]) > (data["abs_maxbatch_msignal"]))*1.)))/2.0))))))/2.0), data["rangebatch_slices2_msignal"] )) * (np.sin((data["abs_maxbatch_msignal"]))))) +

                            0.050000*np.tanh(((((((np.where(data["minbatch"] <= -998, data["abs_avgbatch_slices2"], np.cos((data["maxbatch_slices2_msignal"])) )) + (np.where(data["medianbatch_slices2"] <= -998, data["abs_avgbatch_slices2"], np.cos((data["maxbatch_slices2_msignal"])) )))) * ((-(((10.0))))))) - (np.sin((data["abs_avgbatch_slices2"]))))) +

                            0.050000*np.tanh(((((np.where((-(((7.23746728897094727)))) <= -998, data["abs_minbatch_slices2_msignal"], np.tanh((((np.where((((data["medianbatch_msignal"]) > (data["signal"]))*1.) <= -998, data["medianbatch_msignal"], np.tanh((np.sin((data["abs_minbatch_slices2_msignal"])))) )) * 2.0))) )) * 2.0)) * (data["stdbatch_msignal"]))) +

                            0.050000*np.tanh(((np.sin((np.maximum(((data["abs_maxbatch_msignal"])), ((np.maximum(((data["signal"])), ((((np.sin((np.maximum(((data["abs_maxbatch_msignal"])), ((np.sin((np.maximum(((np.sin((data["abs_maxbatch"])))), ((np.minimum(((data["rangebatch_slices2_msignal"])), ((data["maxbatch_msignal"])))))))))))))) * ((((data["signal"]) > (data["rangebatch_slices2_msignal"]))*1.)))))))))))) * (data["rangebatch_slices2_msignal"]))) +

                            0.050000*np.tanh(np.cos((((data["maxtominbatch"]) - (((np.where(data["minbatch_slices2"] > -998, data["maxbatch_msignal"], data["meanbatch_slices2"] )) * 2.0)))))) +

                            0.050000*np.tanh(((((np.sin((np.where(data["abs_maxbatch"] > -998, data["minbatch_msignal"], np.where(np.where(data["minbatch_slices2_msignal"] > -998, data["signal_shift_+1"], data["medianbatch_slices2"] ) > -998, np.sin((np.where((10.0) > -998, data["maxbatch_slices2_msignal"], data["abs_maxbatch"] ))), np.tanh((data["minbatch_msignal"])) ) )))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((((np.sin(((-((np.where(((((np.sin(((((-((data["minbatch_msignal"])))) * 2.0)))) * 2.0)) * 2.0) <= -998, ((np.sin((data["maxbatch_msignal"]))) * 2.0), ((data["maxbatch_msignal"]) * 2.0) ))))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((np.sin((np.sin((np.where(data["medianbatch_slices2_msignal"] > -998, data["maxbatch_slices2_msignal"], data["abs_maxbatch_slices2_msignal"] ))))))), ((np.where(data["maxbatch_slices2_msignal"] <= -998, np.sin(((-((np.sin((data["abs_avgbatch_slices2_msignal"]))))))), ((np.sin(((-((data["mean_abs_chgbatch_slices2_msignal"])))))) * 2.0) ))))) +

                            0.050000*np.tanh(((np.where(((np.sin((data["rangebatch_slices2"]))) * 2.0) <= -998, ((data["abs_minbatch_msignal"]) * 2.0), np.where(data["minbatch"] <= -998, data["stdbatch_slices2"], np.sin((((data["abs_minbatch_msignal"]) * 2.0))) ) )) * 2.0)) +

                            0.050000*np.tanh(np.sin((np.minimum(((np.where(data["maxtominbatch_slices2"] <= -998, data["abs_maxbatch_slices2_msignal"], (((((data["maxtominbatch_msignal"]) + (data["mean_abs_chgbatch_msignal"]))/2.0)) * 2.0) ))), ((np.where(np.tanh((((np.cos((data["medianbatch_slices2_msignal"]))) / 2.0))) <= -998, data["medianbatch_slices2_msignal"], ((((data["mean_abs_chgbatch_msignal"]) / 2.0)) * 2.0) ))))))) +

                            0.050000*np.tanh(np.minimum(((np.where(((data["abs_avgbatch_msignal"]) - (data["abs_avgbatch_msignal"])) <= -998, np.where(data["stdbatch_slices2_msignal"] <= -998, data["abs_avgbatch_msignal"], data["abs_avgbatch_msignal"] ), (10.01193141937255859) ))), ((((np.where(data["signal_shift_+1"] <= -998, data["abs_maxbatch_slices2"], np.sin((data["signal"])) )) * 2.0))))) +

                            0.050000*np.tanh(((np.cos((((data["maxbatch_slices2"]) - (np.minimum(((data["mean_abs_chgbatch_msignal"])), ((((data["meanbatch_msignal"]) * (((data["maxbatch_msignal"]) + ((((np.minimum(((data["mean_abs_chgbatch_msignal"])), ((np.minimum(((((data["meanbatch_slices2"]) * 2.0))), (((-((np.cos((data["maxbatch_slices2"])))))))))))) <= (data["maxbatch_slices2_msignal"]))*1.))))))))))))) * 2.0)) +

                            0.050000*np.tanh(((np.sin((data["meanbatch_msignal"]))) - (((((np.maximum(((data["abs_minbatch_msignal"])), ((np.cos((np.where(data["abs_minbatch_msignal"] <= -998, ((data["meanbatch_msignal"]) - (np.maximum(((data["abs_minbatch_msignal"])), ((np.cos((data["maxbatch_msignal"]))))))), data["maxbatch_msignal"] ))))))) * 2.0)) * 2.0)))) +

                            0.050000*np.tanh(((data["signal"]) * (((data["minbatch_msignal"]) * (np.where(np.where(data["medianbatch_slices2_msignal"] > -998, ((np.where(data["medianbatch_slices2"] > -998, np.where(data["maxbatch_slices2"] > -998, np.sin((data["maxbatch_slices2"])), data["signal_shift_-1_msignal"] ), data["maxbatch_slices2"] )) * 2.0), data["maxbatch_slices2"] ) > -998, np.sin((data["maxbatch_slices2"])), data["mean_abs_chgbatch_msignal"] )))))) +

                            0.050000*np.tanh(((np.cos((((data["abs_avgbatch_msignal"]) + (np.tanh((np.where(((np.cos((((np.tanh((data["abs_avgbatch_msignal"]))) + (data["abs_avgbatch_msignal"]))))) * 2.0) > -998, data["abs_avgbatch_slices2_msignal"], ((data["abs_avgbatch_msignal"]) + (np.tanh((data["abs_avgbatch_msignal"])))) )))))))) * 2.0)) +

                            0.050000*np.tanh(((np.maximum(((data["abs_maxbatch_msignal"])), ((data["abs_maxbatch_msignal"])))) * (np.minimum(((np.cos((((data["abs_minbatch_slices2_msignal"]) + ((((np.maximum(((((((data["abs_minbatch_slices2_msignal"]) + ((((np.maximum(((np.cos((np.sin((data["maxtominbatch"])))))), ((data["abs_maxbatch_msignal"])))) + (data["abs_maxbatch_msignal"]))/2.0)))) - (data["maxbatch_msignal"])))), ((data["maxbatch_msignal"])))) + (data["medianbatch_slices2"]))/2.0))))))), ((data["signal_shift_+1"])))))) +

                            0.050000*np.tanh(((np.where(np.sin((((data["maxtominbatch"]) * 2.0))) <= -998, data["rangebatch_msignal"], ((data["medianbatch_slices2_msignal"]) + (np.where(data["minbatch_msignal"] > -998, ((np.sin((data["minbatch_msignal"]))) * 2.0), np.where(data["maxbatch_slices2_msignal"] > -998, data["rangebatch_msignal"], np.sin((data["abs_avgbatch_slices2"])) ) ))) )) * 2.0)) +

                            0.050000*np.tanh((((-((((np.cos((data["stdbatch_msignal"]))) * (np.where((((data["medianbatch_msignal"]) + (data["abs_maxbatch"]))/2.0) <= -998, data["medianbatch_msignal"], ((((data["rangebatch_msignal"]) + (data["stdbatch_slices2"]))) + (data["meanbatch_slices2_msignal"])) ))))))) * (np.where(data["rangebatch_slices2_msignal"] <= -998, data["rangebatch_slices2_msignal"], data["meanbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(((data["medianbatch_msignal"]) + (np.minimum(((data["signal_shift_-1"])), ((np.where(data["minbatch_slices2_msignal"] <= -998, np.where(np.minimum(((data["signal_shift_-1_msignal"])), ((data["maxtominbatch"]))) > -998, data["rangebatch_slices2"], np.sin((data["medianbatch_msignal"])) ), np.sin((np.where(np.where(np.cos((np.tanh((data["medianbatch_slices2_msignal"])))) > -998, data["medianbatch_slices2"], np.sin((data["mean_abs_chgbatch_slices2"])) ) > -998, data["minbatch_slices2_msignal"], data["maxbatch_slices2_msignal"] ))) ))))))) +

                            0.050000*np.tanh(np.minimum(((np.sin((((data["abs_avgbatch_msignal"]) + (data["meanbatch_slices2"])))))), (((((data["meanbatch_slices2"]) + (((data["minbatch_msignal"]) * (np.tanh((((data["abs_avgbatch_msignal"]) * (data["abs_avgbatch_msignal"]))))))))/2.0))))) +

                            0.050000*np.tanh(((np.minimum(((np.where(((((((data["meanbatch_slices2"]) <= (data["signal"]))*1.)) > (((((data["medianbatch_msignal"]) + (np.sin((((np.sin((data["mean_abs_chgbatch_msignal"]))) * 2.0)))))) * 2.0)))*1.) <= -998, np.sin((data["meanbatch_slices2"])), np.sin((data["signal"])) ))), ((((((data["medianbatch_msignal"]) + (np.sin((data["abs_avgbatch_slices2"]))))) * 2.0))))) * 2.0)) +

                            0.050000*np.tanh(((((((data["abs_minbatch_msignal"]) + (np.sin((data["abs_minbatch_msignal"]))))) + ((((np.sin((np.maximum(((np.sin((np.maximum(((data["maxbatch_slices2"])), ((data["abs_minbatch_msignal"]))))))), ((np.cos((data["abs_minbatch_msignal"])))))))) <= (data["abs_maxbatch_slices2_msignal"]))*1.)))) * (np.sin((np.maximum(((data["abs_maxbatch"])), ((data["abs_maxbatch"])))))))) +

                            0.050000*np.tanh(((data["medianbatch_msignal"]) + (np.minimum(((((data["medianbatch_msignal"]) + (np.tanh((((data["medianbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2_msignal"])))))))), ((((((((data["rangebatch_slices2_msignal"]) - ((-((np.sin((np.cos((data["medianbatch_msignal"])))))))))) * (data["rangebatch_slices2"]))) * (np.tanh((np.sin((data["maxbatch_slices2_msignal"])))))))))))) +

                            0.050000*np.tanh(((((data["minbatch_msignal"]) * 2.0)) * (np.cos((np.maximum(((data["maxbatch_slices2_msignal"])), ((np.maximum(((np.cos((np.maximum(((data["maxbatch_slices2_msignal"])), ((np.sin((np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((np.cos((np.cos((data["medianbatch_msignal"])))))))))))))))), ((np.sin((((((data["minbatch_msignal"]) * 2.0)) * (np.cos((data["maxbatch_slices2_msignal"]))))))))))))))))) +

                            0.050000*np.tanh(((np.where(data["maxtominbatch_slices2"] > -998, data["abs_minbatch_slices2_msignal"], np.sin((data["abs_maxbatch"])) )) * (np.sin((data["abs_maxbatch"]))))) +

                            0.050000*np.tanh(((data["rangebatch_slices2_msignal"]) * (np.where(((np.cos((np.maximum(((data["maxtominbatch_msignal"])), ((data["maxbatch_msignal"])))))) * (data["maxtominbatch_msignal"])) <= -998, np.minimum(((((np.cos((data["maxbatch_msignal"]))) * (data["maxtominbatch_msignal"])))), ((data["maxtominbatch_msignal"]))), np.minimum(((((np.cos((data["maxbatch_msignal"]))) * (data["maxtominbatch_msignal"])))), ((data["maxbatch_msignal"]))) )))) +

                            0.050000*np.tanh((-((((data["abs_maxbatch"]) * (((np.sin((((np.sin((((np.where(((data["signal"]) - ((-((data["signal"]))))) > -998, data["signal"], data["signal"] )) * 2.0)))) * 2.0)))) * 2.0))))))) +

                            0.050000*np.tanh((((((-((((data["minbatch_slices2_msignal"]) * (np.where(data["minbatch_slices2"] <= -998, ((data["mean_abs_chgbatch_msignal"]) * (data["mean_abs_chgbatch_slices2_msignal"])), (((np.sin((data["abs_maxbatch_slices2_msignal"]))) + (data["medianbatch_msignal"]))/2.0) ))))))) * 2.0)) - (data["medianbatch_msignal"]))) +

                            0.050000*np.tanh(((data["rangebatch_msignal"]) * (np.minimum(((np.cos((np.where((-((data["maxtominbatch_msignal"]))) > -998, data["rangebatch_msignal"], np.minimum(((data["minbatch"])), ((((np.maximum(((data["rangebatch_msignal"])), ((data["minbatch_slices2_msignal"])))) * ((8.39901733398437500)))))) ))))), ((np.sin((data["minbatch_msignal"])))))))) +

                            0.050000*np.tanh(((np.sin((data["medianbatch_slices2"]))) * (((data["rangebatch_msignal"]) - (((data["abs_maxbatch_msignal"]) - (np.tanh((((((data["abs_maxbatch_slices2_msignal"]) + (np.sin((np.sin((data["rangebatch_msignal"]))))))) + ((10.0)))))))))))) +

                            0.050000*np.tanh(((((((((data["medianbatch_msignal"]) - (((np.cos((data["maxbatch_msignal"]))) * 2.0)))) * (((((data["rangebatch_slices2"]) - (np.sin((data["minbatch_slices2_msignal"]))))) - (((data["signal"]) + (np.cos((((((data["minbatch_slices2_msignal"]) - (np.cos((data["meanbatch_slices2_msignal"]))))) * 2.0)))))))))) * (data["abs_maxbatch"]))) - (data["abs_minbatch_msignal"]))) +

                            0.050000*np.tanh(np.where(data["abs_minbatch_slices2"] > -998, (((data["meanbatch_slices2_msignal"]) + (data["minbatch_slices2"]))/2.0), (-((np.tanh((data["abs_minbatch_slices2"]))))) )) +

                            0.050000*np.tanh(((((((np.sin((np.sin((data["abs_minbatch_slices2_msignal"]))))) * (data["meanbatch_msignal"]))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.where(data["maxbatch_slices2"] <= -998, data["maxtominbatch"], ((((np.sin((data["abs_maxbatch_slices2_msignal"]))) * ((((data["maxbatch_slices2_msignal"]) + (((np.where(np.cos((data["abs_maxbatch_slices2_msignal"])) > -998, ((np.minimum((((((data["maxtominbatch"]) + (data["signal_shift_-1_msignal"]))/2.0))), ((((data["medianbatch_msignal"]) * 2.0))))) + (data["maxbatch_msignal"])), data["abs_maxbatch_slices2_msignal"] )) * 2.0)))/2.0)))) * 2.0) )) +

                            0.050000*np.tanh(((np.minimum(((np.sin((data["abs_maxbatch_slices2_msignal"])))), ((np.where((((data["abs_minbatch_slices2_msignal"]) + (np.minimum(((data["abs_maxbatch_slices2_msignal"])), ((np.sin((data["abs_maxbatch_msignal"])))))))/2.0) > -998, data["abs_minbatch_slices2_msignal"], data["stdbatch_slices2"] ))))) + (((((np.where(data["abs_maxbatch_slices2_msignal"] > -998, np.sin((data["abs_maxbatch_slices2_msignal"])), np.where(data["meanbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], data["abs_minbatch_msignal"] ) )) * 2.0)) * 2.0)))) +

                            0.050000*np.tanh(np.where(data["maxtominbatch"] > -998, ((data["meanbatch_slices2"]) * (np.minimum(((np.sin((np.maximum(((data["signal_shift_+1"])), ((data["maxbatch_msignal"]))))))), (((((np.minimum(((data["maxtominbatch"])), ((data["minbatch_slices2"])))) <= (np.minimum((((12.09745597839355469))), ((np.cos((data["maxtominbatch"])))))))*1.)))))), (14.58901405334472656) )) +

                            0.050000*np.tanh(((np.minimum(((np.where(data["medianbatch_slices2_msignal"] > -998, data["medianbatch_slices2_msignal"], data["meanbatch_msignal"] ))), ((((data["minbatch"]) * (np.sin((data["medianbatch_slices2_msignal"])))))))) * 2.0)) +

                            0.050000*np.tanh(np.sin((((np.minimum(((data["maxbatch_msignal"])), ((np.sin((((np.minimum(((np.where(data["stdbatch_slices2"] > -998, data["signal_shift_+1"], ((data["rangebatch_slices2_msignal"]) / 2.0) ))), ((np.where(np.where(data["abs_minbatch_msignal"] > -998, np.sin((data["abs_minbatch_msignal"])), data["meanbatch_slices2_msignal"] ) > -998, data["abs_minbatch_msignal"], data["rangebatch_slices2"] ))))) * 2.0))))))) * 2.0)))) +

                            0.050000*np.tanh(((((data["mean_abs_chgbatch_msignal"]) * (data["abs_maxbatch_slices2"]))) * (np.sin((((data["abs_maxbatch_slices2"]) + ((((np.minimum(((data["abs_maxbatch_slices2"])), ((data["maxbatch_slices2_msignal"])))) <= (np.cos((np.where((-((data["abs_minbatch_slices2_msignal"]))) <= -998, (7.36450386047363281), data["maxbatch_msignal"] )))))*1.)))))))) +

                            0.050000*np.tanh(((data["abs_minbatch_slices2_msignal"]) * (np.sin((data["abs_maxbatch"]))))) +

                            0.050000*np.tanh(np.where(data["mean_abs_chgbatch_msignal"] <= -998, data["mean_abs_chgbatch_slices2"], ((((data["abs_minbatch_slices2"]) + (data["minbatch_msignal"]))) * (np.sin((((np.sin((((np.where(np.sin((np.cos((data["signal"])))) <= -998, data["meanbatch_slices2"], data["maxbatch_msignal"] )) * 2.0)))) * 2.0))))) )) +

                            0.050000*np.tanh(((((data["abs_minbatch_slices2_msignal"]) + ((((7.25954723358154297)) * ((-((np.cos((((np.where(((data["mean_abs_chgbatch_msignal"]) * (data["abs_minbatch_slices2_msignal"])) <= -998, np.where(((data["mean_abs_chgbatch_msignal"]) - (data["abs_avgbatch_slices2_msignal"])) > -998, data["abs_minbatch_slices2_msignal"], (7.25954723358154297) ), data["mean_abs_chgbatch_msignal"] )) * 2.0))))))))))) + (data["mean_abs_chgbatch_msignal"]))) +

                            0.050000*np.tanh(((np.sin((np.sin((((np.sin((((np.where(data["meanbatch_slices2_msignal"] > -998, np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["abs_minbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] ), data["abs_minbatch_msignal"] )) * 2.0)))) * 2.0)))))) * 2.0)) +

                            0.050000*np.tanh((-(((((((((data["stdbatch_slices2_msignal"]) + (((data["maxbatch_msignal"]) + (data["abs_maxbatch_slices2_msignal"]))))/2.0)) + (np.where((-((np.sin((((data["abs_maxbatch_slices2_msignal"]) * 2.0)))))) <= -998, ((data["stdbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2_msignal"])), data["meanbatch_slices2"] )))) * (np.sin((((data["stdbatch_slices2_msignal"]) * 2.0))))))))) +

                            0.050000*np.tanh(np.sin((np.cos((np.where(((((data["stdbatch_slices2"]) * 2.0)) * 2.0) > -998, data["abs_avgbatch_msignal"], ((((data["minbatch_slices2_msignal"]) - (((data["abs_avgbatch_msignal"]) / 2.0)))) * 2.0) )))))) +

                            0.050000*np.tanh(((np.maximum(((np.where(np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["meanbatch_msignal"], data["medianbatch_slices2_msignal"] ) > -998, data["minbatch_msignal"], np.sin((data["maxtominbatch_slices2_msignal"])) ))), ((((np.where(np.tanh((np.cos((data["abs_avgbatch_slices2_msignal"])))) > -998, (-((((np.sin((data["meanbatch_msignal"]))) * 2.0)))), ((np.tanh((data["rangebatch_slices2"]))) * 2.0) )) * 2.0))))) * 2.0)) +

                            0.050000*np.tanh(((((np.where(data["abs_minbatch_msignal"] <= -998, data["medianbatch_slices2_msignal"], data["abs_minbatch_slices2_msignal"] )) * 2.0)) * (np.where(np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((np.where(np.maximum(((data["abs_minbatch_msignal"])), ((np.sin((data["medianbatch_slices2_msignal"]))))) > -998, np.cos((data["mean_abs_chgbatch_slices2"])), (-((data["mean_abs_chgbatch_slices2_msignal"]))) )))) > -998, np.cos((data["abs_minbatch_msignal"])), data["medianbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(np.where(((data["abs_minbatch_msignal"]) + (data["abs_minbatch_msignal"])) <= -998, (10.07492733001708984), ((data["medianbatch_msignal"]) + (((np.sin((data["abs_minbatch_msignal"]))) * (np.where(data["medianbatch_msignal"] <= -998, ((data["medianbatch_msignal"]) + (data["medianbatch_msignal"])), data["abs_minbatch_msignal"] ))))) )) +

                            0.050000*np.tanh((((data["medianbatch_msignal"]) + (np.where((-((((data["medianbatch_slices2_msignal"]) + (np.where(np.sin((data["minbatch_slices2_msignal"])) > -998, data["abs_avgbatch_slices2_msignal"], np.sin((data["medianbatch_msignal"])) )))))) <= -998, np.tanh((np.sin((data["maxbatch_slices2_msignal"])))), (((((np.sin((data["minbatch_slices2_msignal"]))) * 2.0)) <= (data["maxbatch_slices2_msignal"]))*1.) )))/2.0)) +

                            0.050000*np.tanh(((np.where((5.81791639328002930) <= -998, data["maxbatch_msignal"], np.sin((np.sin((np.where(np.sin((data["maxbatch_slices2"])) > -998, np.sin((data["maxbatch_slices2"])), data["mean_abs_chgbatch_slices2"] ))))) )) * (data["mean_abs_chgbatch_msignal"]))) +

                            0.050000*np.tanh(((((np.sin((np.where(np.where(np.tanh((data["mean_abs_chgbatch_slices2_msignal"])) <= -998, data["minbatch_slices2_msignal"], data["abs_maxbatch_msignal"] ) <= -998, data["minbatch_slices2_msignal"], data["abs_maxbatch_msignal"] )))) * 2.0)) + (((data["mean_abs_chgbatch_msignal"]) - ((-(((((((data["mean_abs_chgbatch_msignal"]) > (((data["abs_maxbatch"]) * (data["stdbatch_msignal"]))))*1.)) * 2.0))))))))) +

                            0.050000*np.tanh(((((np.where(data["abs_minbatch_msignal"] > -998, ((data["abs_maxbatch_slices2"]) * (data["meanbatch_slices2"])), np.where(data["meanbatch_slices2"] <= -998, data["mean_abs_chgbatch_slices2"], np.sin((data["minbatch_slices2_msignal"])) ) )) * (np.cos(((((np.sin((data["minbatch_slices2_msignal"]))) + (data["abs_minbatch_msignal"]))/2.0)))))) - (np.maximum(((data["signal_shift_-1"])), ((data["abs_minbatch_msignal"])))))) +

                            0.050000*np.tanh(np.where(data["medianbatch_msignal"] <= -998, ((data["signal_shift_-1_msignal"]) + (np.cos((np.where(np.where(((np.maximum(((data["maxtominbatch_msignal"])), ((data["abs_avgbatch_msignal"])))) * 2.0) <= -998, data["stdbatch_slices2_msignal"], ((np.maximum(((data["maxtominbatch_msignal"])), ((data["rangebatch_slices2"])))) * 2.0) ) <= -998, data["signal_shift_+1"], ((data["maxtominbatch"]) * 2.0) ))))), np.sin((data["abs_maxbatch_slices2_msignal"])) )) +

                            0.050000*np.tanh(((((data["abs_maxbatch_slices2"]) - (data["mean_abs_chgbatch_slices2_msignal"]))) * (((np.where((-(((((((data["meanbatch_msignal"]) + (data["meanbatch_msignal"]))/2.0)) + (data["medianbatch_msignal"]))))) <= -998, data["medianbatch_msignal"], np.cos(((((data["medianbatch_msignal"]) + ((((((data["meanbatch_msignal"]) + (data["meanbatch_msignal"]))/2.0)) + (data["medianbatch_msignal"]))))/2.0))) )) * 2.0)))) +

                            0.050000*np.tanh(((data["mean_abs_chgbatch_slices2_msignal"]) * ((-((np.cos((np.minimum(((data["meanbatch_slices2_msignal"])), ((((((data["mean_abs_chgbatch_slices2_msignal"]) * ((-((np.cos((np.minimum(((data["meanbatch_slices2_msignal"])), ((data["stdbatch_slices2_msignal"]))))))))))) - (np.cos((np.sin((data["meanbatch_slices2_msignal"]))))))))))))))))) +

                            0.050000*np.tanh(((data["medianbatch_msignal"]) - (((((((data["maxtominbatch"]) - (np.where(np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, data["abs_maxbatch_slices2"], ((data["abs_minbatch_msignal"]) * 2.0) ) > -998, ((data["mean_abs_chgbatch_slices2_msignal"]) * (data["signal_shift_-1"])), ((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0) )))) + (((data["mean_abs_chgbatch_msignal"]) - (data["signal_shift_-1"]))))) + (data["maxtominbatch"]))))) +

                            0.050000*np.tanh((((4.67500543594360352)) * (np.cos((((np.maximum(((data["maxbatch_msignal"])), ((data["signal_shift_-1"])))) + (np.minimum(((data["maxtominbatch"])), ((np.minimum(((np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((data["signal_shift_-1"]))))), ((np.minimum(((data["maxtominbatch"])), ((data["maxtominbatch"])))))))))))))))) +

                            0.050000*np.tanh(((((((((((((((data["abs_minbatch_slices2_msignal"]) * 2.0)) + (np.minimum(((((((np.sin((data["minbatch_msignal"]))) * 2.0)) * 2.0))), ((((data["abs_minbatch_slices2_msignal"]) * 2.0))))))/2.0)) > (data["rangebatch_msignal"]))*1.)) + (np.sin((((data["abs_minbatch_slices2_msignal"]) * 2.0)))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.where(np.cos((data["abs_minbatch_msignal"])) > -998, ((data["abs_minbatch_slices2_msignal"]) * (np.cos((((data["maxtominbatch_slices2_msignal"]) - (data["meanbatch_slices2_msignal"])))))), data["mean_abs_chgbatch_msignal"] )) +

                            0.050000*np.tanh(np.sin((np.where(data["abs_maxbatch_slices2_msignal"] > -998, np.where(np.where(data["meanbatch_slices2_msignal"] <= -998, np.maximum(((data["medianbatch_slices2"])), (((((data["abs_minbatch_slices2_msignal"]) + (data["maxbatch_slices2_msignal"]))/2.0)))), np.sin((np.where(data["maxbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], np.maximum(((data["medianbatch_slices2"])), ((np.sin((data["meanbatch_slices2"]))))) ))) ) > -998, data["abs_maxbatch_slices2_msignal"], np.sin((data["abs_maxbatch_slices2_msignal"])) ), data["medianbatch_slices2"] )))) +

                            0.050000*np.tanh((((((data["meanbatch_slices2_msignal"]) + (((data["meanbatch_slices2"]) * (np.sin((data["meanbatch_slices2"]))))))) + (np.where(((data["meanbatch_slices2"]) * (data["meanbatch_slices2"])) <= -998, (((data["rangebatch_slices2"]) + (((data["rangebatch_slices2"]) * (np.minimum(((data["meanbatch_slices2_msignal"])), ((((data["meanbatch_slices2_msignal"]) * 2.0))))))))/2.0), np.cos((((data["meanbatch_slices2_msignal"]) + (np.cos((data["abs_minbatch_slices2_msignal"])))))) )))/2.0)) +

                            0.050000*np.tanh(np.minimum(((((np.sin((((np.minimum((((((data["abs_minbatch_slices2_msignal"]) + (data["medianbatch_msignal"]))/2.0))), ((data["stdbatch_slices2_msignal"])))) + (np.where(np.where(data["meanbatch_slices2"] > -998, data["medianbatch_slices2_msignal"], data["rangebatch_slices2_msignal"] ) <= -998, data["maxbatch_msignal"], data["maxtominbatch_msignal"] )))))) * 2.0))), ((np.where(np.cos((data["signal"])) <= -998, (((data["meanbatch_slices2"]) <= (data["abs_maxbatch_slices2_msignal"]))*1.), (5.01787090301513672) ))))) +

                            0.050000*np.tanh(np.where(data["meanbatch_slices2_msignal"] <= -998, np.sin((((data["abs_minbatch_slices2_msignal"]) - (np.sin((((np.sin((data["rangebatch_slices2"]))) - (data["rangebatch_slices2"])))))))), ((((((np.sin((data["rangebatch_slices2"]))) - (np.sin((data["meanbatch_slices2_msignal"]))))) * 2.0)) + (data["meanbatch_slices2_msignal"])) )) +

                            0.050000*np.tanh((-((np.where((-(((-((data["mean_abs_chgbatch_slices2_msignal"])))))) > -998, np.where((((data["signal"]) + (((np.sin((data["abs_minbatch_slices2"]))) * 2.0)))/2.0) <= -998, data["abs_maxbatch_slices2"], ((data["rangebatch_msignal"]) * (np.cos((((np.sin(((-((data["mean_abs_chgbatch_slices2_msignal"])))))) * 2.0))))) ), data["stdbatch_slices2_msignal"] ))))) +

                            0.050000*np.tanh((((data["stdbatch_msignal"]) + (((((np.where(data["minbatch"] <= -998, (((((data["stdbatch_msignal"]) <= (data["stdbatch_msignal"]))*1.)) + (((data["stdbatch_msignal"]) + (data["minbatch_slices2"])))), np.cos((data["rangebatch_msignal"])) )) * 2.0)) * 2.0)))/2.0)) +

                            0.050000*np.tanh(data["abs_minbatch_slices2_msignal"]) +

                            0.050000*np.tanh(np.where(np.maximum(((data["abs_minbatch_msignal"])), ((data["maxtominbatch_slices2_msignal"]))) <= -998, data["abs_minbatch_msignal"], np.sin(((((((data["abs_maxbatch_msignal"]) + (np.cos((np.where(data["abs_minbatch_msignal"] > -998, data["medianbatch_slices2"], data["abs_avgbatch_msignal"] )))))/2.0)) * 2.0))) )) +

                            0.050000*np.tanh(((np.sin((data["maxbatch_slices2_msignal"]))) * (((data["medianbatch_slices2"]) + (np.sin(((-((((data["abs_minbatch_msignal"]) + (data["signal_shift_+1_msignal"])))))))))))) +

                            0.050000*np.tanh(data["medianbatch_msignal"]) +

                            0.050000*np.tanh(((np.maximum(((data["maxbatch_msignal"])), ((data["abs_avgbatch_slices2"])))) * ((((((np.where(data["maxbatch_slices2_msignal"] > -998, data["meanbatch_msignal"], data["medianbatch_msignal"] )) <= (data["signal_shift_-1_msignal"]))*1.)) + (np.minimum(((data["medianbatch_msignal"])), ((((((np.tanh((data["medianbatch_msignal"]))) - (data["maxbatch_msignal"]))) * (np.sin((data["medianbatch_msignal"])))))))))))) +

                            0.050000*np.tanh((((np.minimum(((data["signal_shift_-1"])), ((((np.sin((np.where(((np.sin((data["maxtominbatch_slices2"]))) / 2.0) > -998, (((((data["medianbatch_msignal"]) + (((data["mean_abs_chgbatch_slices2"]) * 2.0)))/2.0)) * 2.0), (((data["abs_minbatch_slices2"]) + (np.sin((data["minbatch_msignal"]))))/2.0) )))) * 2.0))))) + (data["meanbatch_slices2_msignal"]))/2.0)) +

                            0.050000*np.tanh((((-((np.where((-((np.cos((np.minimum(((np.sin(((((data["abs_minbatch_slices2_msignal"]) > (data["abs_maxbatch_msignal"]))*1.))))), ((data["meanbatch_msignal"])))))))) <= -998, data["stdbatch_slices2_msignal"], data["maxbatch_slices2_msignal"] ))))) * (((data["abs_minbatch_slices2_msignal"]) * (np.cos(((((((data["maxbatch_msignal"]) > (data["maxtominbatch"]))*1.)) - (data["medianbatch_slices2"]))))))))) +

                            0.050000*np.tanh(((data["stdbatch_slices2_msignal"]) * (((np.cos((((data["maxtominbatch"]) + (np.where((-((data["rangebatch_msignal"]))) > -998, data["abs_maxbatch_slices2"], ((((((data["minbatch_slices2"]) + (np.tanh((data["abs_avgbatch_slices2_msignal"]))))) + (np.cos((((data["rangebatch_slices2"]) * (np.cos((((data["rangebatch_slices2"]) * (data["minbatch_slices2"]))))))))))) + (data["stdbatch_slices2_msignal"])) )))))) * 2.0)))) +

                            0.050000*np.tanh(((np.sin((data["abs_maxbatch_slices2_msignal"]))) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((((((((3.0)) * (np.tanh((np.minimum(((np.minimum(((data["minbatch"])), (((((data["stdbatch_msignal"]) <= (np.minimum(((data["minbatch_msignal"])), ((data["meanbatch_slices2"])))))*1.)))))), ((data["signal_shift_-1_msignal"])))))))) + (((data["meanbatch_slices2"]) * (data["stdbatch_msignal"]))))/2.0))), ((data["abs_maxbatch"])))) +

                            0.050000*np.tanh(np.minimum(((((np.tanh(((-((((data["signal"]) * 2.0))))))) - (((data["signal"]) * (np.sin((data["abs_maxbatch"])))))))), ((np.cos((np.cos((np.minimum(((data["abs_minbatch_msignal"])), ((data["mean_abs_chgbatch_slices2_msignal"]))))))))))) +

                            0.050000*np.tanh(np.minimum(((np.sin((np.where(data["abs_avgbatch_slices2_msignal"] <= -998, np.minimum(((((((data["minbatch"]) / 2.0)) - (np.cos((data["abs_avgbatch_slices2_msignal"])))))), ((data["signal_shift_+1_msignal"]))), data["minbatch_msignal"] ))))), ((((data["abs_avgbatch_slices2"]) * (data["rangebatch_msignal"])))))) +

                            0.050000*np.tanh(((np.tanh((np.where(data["rangebatch_msignal"] > -998, np.where((((data["signal_shift_-1"]) + (data["abs_maxbatch_slices2"]))/2.0) > -998, np.sin((data["signal_shift_-1"])), np.where((((data["maxtominbatch_slices2"]) + (((np.tanh((data["abs_avgbatch_slices2_msignal"]))) / 2.0)))/2.0) > -998, data["signal_shift_-1_msignal"], data["signal_shift_-1"] ) ), ((np.cos((np.maximum(((data["rangebatch_msignal"])), ((data["abs_minbatch_slices2"])))))) / 2.0) )))) * 2.0)) +

                            0.050000*np.tanh(((np.where((((data["mean_abs_chgbatch_msignal"]) > (np.minimum(((data["mean_abs_chgbatch_msignal"])), ((data["signal_shift_-1"])))))*1.) > -998, data["signal_shift_-1"], data["medianbatch_msignal"] )) * (np.cos((data["medianbatch_msignal"]))))) +

                            0.050000*np.tanh(np.sin((np.where(data["signal_shift_-1"] <= -998, ((data["minbatch_msignal"]) - (np.where((-((np.where(data["maxbatch_slices2_msignal"] > -998, data["abs_avgbatch_msignal"], np.where(np.sin((data["signal_shift_-1"])) <= -998, data["meanbatch_slices2_msignal"], data["medianbatch_slices2_msignal"] ) )))) > -998, data["abs_avgbatch_msignal"], np.cos(((-((np.sin((np.sin((np.cos((np.sin((data["signal_shift_-1"]))))))))))))) ))), data["signal_shift_-1"] )))) +

                            0.050000*np.tanh((-((np.cos((((np.where(((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)) * 2.0) <= -998, ((data["signal_shift_-1_msignal"]) / 2.0), data["mean_abs_chgbatch_slices2_msignal"] )) * 2.0))))))) +

                            0.050000*np.tanh((-((((np.cos((data["signal_shift_-1_msignal"]))) * 2.0))))) +

                            0.050000*np.tanh(((np.sin((np.sin((data["meanbatch_slices2_msignal"]))))) * (np.where((9.0) > -998, data["minbatch_msignal"], np.maximum(((((data["medianbatch_msignal"]) * 2.0))), ((np.sin(((9.0)))))) )))) +

                            0.050000*np.tanh(((((np.sin((((data["maxbatch_slices2"]) - (((data["mean_abs_chgbatch_msignal"]) - (np.maximum(((data["signal_shift_-1"])), ((np.maximum(((data["signal_shift_-1"])), (((((((np.sin((data["signal_shift_-1"]))) + ((((data["signal_shift_+1"]) > (data["abs_minbatch_msignal"]))*1.)))/2.0)) - (data["signal_shift_-1"]))))))))))))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((np.cos((data["maxtominbatch_slices2_msignal"]))) * (np.minimum(((data["minbatch_slices2"])), ((np.where(np.sin((data["stdbatch_slices2"])) <= -998, ((np.cos((data["maxtominbatch_slices2_msignal"]))) + (((np.minimum(((data["signal_shift_+1"])), ((data["minbatch_slices2"])))) - (data["maxtominbatch_slices2_msignal"])))), data["abs_avgbatch_msignal"] ))))))) +

                            0.050000*np.tanh(((data["medianbatch_slices2"]) * (np.sin((np.where(data["abs_minbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], ((np.where(np.cos((np.sin((data["mean_abs_chgbatch_slices2_msignal"])))) > -998, data["medianbatch_slices2"], data["mean_abs_chgbatch_slices2_msignal"] )) * (np.cos(((((data["abs_minbatch_slices2_msignal"]) + ((((((data["maxtominbatch_msignal"]) > ((((12.99664306640625000)) + (data["maxbatch_slices2_msignal"]))))*1.)) * (data["maxbatch_slices2"]))))/2.0))))) )))))) +

                            0.050000*np.tanh(data["medianbatch_msignal"]) +

                            0.050000*np.tanh(((np.sin((((data["stdbatch_slices2_msignal"]) * (data["maxtominbatch_slices2"]))))) * 2.0)) +

                            0.050000*np.tanh(np.cos((((((data["abs_maxbatch_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))) - (np.where(((np.cos((((((data["abs_maxbatch_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))) - (np.where(data["abs_maxbatch_msignal"] > -998, data["signal_shift_-1"], data["maxtominbatch"] )))))) * (((np.tanh((data["abs_maxbatch_msignal"]))) * 2.0))) > -998, data["signal_shift_-1"], data["abs_avgbatch_slices2_msignal"] )))))) +

                            0.050000*np.tanh(((np.sin((((((data["abs_maxbatch_slices2_msignal"]) - (np.where(np.tanh((((data["maxbatch_msignal"]) / 2.0))) > -998, ((np.minimum(((data["medianbatch_slices2_msignal"])), ((np.tanh((data["signal_shift_+1_msignal"])))))) - ((-((np.tanh((((data["rangebatch_msignal"]) * 2.0)))))))), data["abs_maxbatch_slices2_msignal"] )))) + (np.tanh((((data["rangebatch_msignal"]) * 2.0)))))))) * 2.0)) +

                            0.050000*np.tanh(((((((((data["medianbatch_msignal"]) + (((data["abs_minbatch_slices2"]) * 2.0)))) + ((((data["mean_abs_chgbatch_msignal"]) + (data["minbatch_slices2"]))/2.0)))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((data["minbatch"]) * (np.cos((np.where((((np.minimum(((data["maxbatch_msignal"])), ((data["maxbatch_msignal"])))) <= (np.tanh((data["abs_minbatch_msignal"]))))*1.) > -998, data["maxbatch_msignal"], np.tanh((np.tanh((np.maximum(((data["abs_minbatch_slices2_msignal"])), ((np.tanh((data["minbatch_slices2"]))))))))) )))))) +

                            0.050000*np.tanh(((np.sin((data["abs_maxbatch_msignal"]))) * (((np.sin((data["rangebatch_slices2"]))) + (((data["abs_maxbatch_msignal"]) * (((np.sin((np.sin((((((((data["stdbatch_slices2_msignal"]) * 2.0)) - (((data["abs_avgbatch_slices2_msignal"]) / 2.0)))) * 2.0)))))) + (((np.sin((np.sin((data["signal_shift_+1"]))))) + (data["medianbatch_msignal"]))))))))))) +

                            0.050000*np.tanh(np.sin((((((data["minbatch_msignal"]) * 2.0)) - (np.where(data["stdbatch_slices2"] > -998, data["meanbatch_slices2_msignal"], np.cos((np.where(((data["minbatch_msignal"]) * 2.0) > -998, data["meanbatch_slices2_msignal"], data["abs_maxbatch_slices2"] ))) )))))) +

                            0.050000*np.tanh(((data["abs_minbatch_msignal"]) + (((np.where(data["abs_avgbatch_msignal"] > -998, (((12.93795680999755859)) * (np.tanh((np.cos((data["rangebatch_msignal"])))))), np.where(((data["meanbatch_msignal"]) - (((data["minbatch_slices2_msignal"]) + (data["meanbatch_slices2"])))) > -998, (5.0), data["minbatch_slices2_msignal"] ) )) + (data["minbatch"]))))) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) * 2.0)) +

                            0.050000*np.tanh(((data["mean_abs_chgbatch_slices2_msignal"]) * (np.sin((np.sin((np.where(data["maxbatch_slices2"] > -998, data["maxbatch_slices2"], np.where(((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0) <= -998, (4.0), np.maximum(((np.sin((data["maxbatch_slices2"])))), ((data["abs_avgbatch_msignal"]))) ) )))))))) +

                            0.050000*np.tanh(((data["rangebatch_slices2"]) * (((data["minbatch_slices2_msignal"]) * (((np.cos((((data["signal_shift_+1_msignal"]) - (data["maxbatch_msignal"]))))) / 2.0)))))) +

                            0.050000*np.tanh(((((data["stdbatch_msignal"]) - (np.sin((np.where(data["meanbatch_msignal"] <= -998, (((5.37529850006103516)) - (data["abs_avgbatch_slices2"])), ((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)) * 2.0) )))))) + (((np.sin((data["mean_abs_chgbatch_slices2_msignal"]))) * (data["mean_abs_chgbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(data["medianbatch_msignal"]) +

                            0.050000*np.tanh(np.sin((np.sin((data["minbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) - (np.sin((np.where(((data["rangebatch_slices2"]) + (data["minbatch_slices2"])) > -998, data["maxbatch_msignal"], data["meanbatch_slices2_msignal"] )))))) +

                            0.050000*np.tanh(np.minimum(((np.sin((np.minimum(((data["maxtominbatch_slices2_msignal"])), ((data["minbatch_slices2_msignal"]))))))), ((data["maxbatch_slices2_msignal"])))) +

                            0.050000*np.tanh(np.where(data["signal_shift_-1"] <= -998, np.cos((data["signal_shift_+1"])), np.sin((((np.where(data["signal_shift_-1"] <= -998, data["signal_shift_+1"], np.sin((np.sin((((((data["signal_shift_-1"]) / 2.0)) - (data["abs_avgbatch_msignal"])))))) )) * 2.0))) )) +

                            0.050000*np.tanh(((np.sin((np.where(np.where(np.sin((((np.sin((np.sin((np.where(data["stdbatch_slices2"] > -998, data["signal_shift_-1"], data["maxbatch_msignal"] )))))) * 2.0))) <= -998, data["minbatch_slices2_msignal"], ((data["maxtominbatch_slices2_msignal"]) * 2.0) ) > -998, data["signal_shift_-1"], np.sin((np.minimum(((((data["signal_shift_-1_msignal"]) * 2.0))), ((data["abs_minbatch_slices2_msignal"]))))) )))) * 2.0)) +

                            0.050000*np.tanh(((data["medianbatch_msignal"]) - (((data["signal_shift_-1"]) * (np.sin((((np.maximum(((data["minbatch_slices2"])), (((1.47495067119598389))))) * (((np.minimum(((np.minimum(((data["mean_abs_chgbatch_msignal"])), ((np.where(data["signal_shift_-1"] <= -998, ((data["medianbatch_slices2"]) * 2.0), ((np.maximum(((data["signal_shift_+1"])), ((data["abs_minbatch_slices2_msignal"])))) * 2.0) )))))), ((data["signal_shift_-1"])))) / 2.0)))))))))) +

                            0.050000*np.tanh(np.cos((np.where(data["stdbatch_slices2_msignal"] <= -998, np.cos(((-((((data["signal_shift_-1"]) + (((data["abs_maxbatch"]) * 2.0)))))))), np.where(data["signal_shift_-1"] <= -998, ((data["abs_maxbatch"]) - (data["meanbatch_slices2"])), ((data["maxtominbatch_slices2"]) - (((data["signal_shift_-1"]) + (((data["abs_maxbatch"]) * 2.0))))) ) )))) +

                            0.050000*np.tanh(((((np.sin((np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((np.where(data["abs_maxbatch_slices2_msignal"] > -998, np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((data["signal_shift_-1"]))), ((((np.sin((np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((data["signal_shift_-1"])))))) * 2.0)) * (((data["medianbatch_slices2"]) * 2.0))) )))))))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.where((-((((data["abs_minbatch_slices2_msignal"]) + (data["meanbatch_msignal"]))))) <= -998, data["meanbatch_slices2_msignal"], data["meanbatch_msignal"] )) +

                            0.050000*np.tanh(np.where((-((data["maxtominbatch"]))) <= -998, ((data["stdbatch_slices2_msignal"]) + (np.maximum(((data["minbatch_slices2"])), ((data["medianbatch_slices2"]))))), np.where(np.sin((np.sin((data["maxbatch_slices2_msignal"])))) <= -998, data["meanbatch_msignal"], ((np.sin(((-((np.maximum(((((data["maxbatch_slices2_msignal"]) / 2.0))), ((((data["signal_shift_+1"]) * 2.0)))))))))) * 2.0) ) )) +

                            0.050000*np.tanh(data["medianbatch_msignal"]) +

                            0.050000*np.tanh(np.where(np.sin((data["signal_shift_+1"])) > -998, ((((data["abs_maxbatch_slices2_msignal"]) + ((((((data["abs_avgbatch_msignal"]) / 2.0)) <= (np.sin((data["abs_maxbatch_slices2_msignal"]))))*1.)))) * (np.sin((np.sin((data["signal_shift_+1"])))))), data["abs_avgbatch_msignal"] )) +

                            0.050000*np.tanh(np.sin((np.minimum((((-((((data["signal"]) + (np.maximum(((data["abs_avgbatch_msignal"])), ((data["signal_shift_-1"])))))))))), ((np.minimum(((np.maximum(((((data["signal_shift_-1"]) + (data["minbatch_slices2"])))), ((data["abs_avgbatch_slices2_msignal"]))))), ((np.sin((data["signal"]))))))))))) +

                            0.050000*np.tanh(((np.cos((np.where(data["abs_maxbatch_msignal"] <= -998, data["signal_shift_-1"], data["abs_avgbatch_msignal"] )))) * (np.where(data["abs_avgbatch_msignal"] > -998, data["signal_shift_-1"], ((np.tanh((data["maxtominbatch"]))) / 2.0) )))) +

                            0.050000*np.tanh(((np.sin((np.where(np.where(np.maximum(((data["signal_shift_+1"])), ((data["maxbatch_slices2_msignal"]))) <= -998, data["abs_avgbatch_slices2"], np.maximum(((data["signal_shift_+1"])), ((data["maxbatch_slices2_msignal"]))) ) <= -998, data["stdbatch_slices2"], np.maximum(((data["signal_shift_+1"])), ((data["maxbatch_slices2_msignal"]))) )))) * 2.0)) +

                            0.050000*np.tanh(((((((np.sin((data["abs_maxbatch_slices2_msignal"]))) - ((((data["mean_abs_chgbatch_msignal"]) > (data["medianbatch_msignal"]))*1.)))) - ((((np.cos((np.sin((data["abs_maxbatch_slices2_msignal"]))))) > (data["medianbatch_slices2"]))*1.)))) * (np.where(data["minbatch_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], np.minimum(((data["medianbatch_msignal"])), ((data["abs_maxbatch_slices2_msignal"]))) )))) +

                            0.050000*np.tanh(((np.sin((np.where(((np.minimum(((data["meanbatch_msignal"])), ((data["abs_minbatch_slices2"])))) * (data["abs_maxbatch_slices2_msignal"])) > -998, ((np.sin((((np.cos((np.minimum(((data["meanbatch_msignal"])), ((data["abs_minbatch_slices2"])))))) * (data["abs_maxbatch_slices2_msignal"]))))) * 2.0), data["medianbatch_msignal"] )))) * 2.0)) +

                            0.050000*np.tanh(np.sin((data["minbatch_msignal"]))) +

                            0.050000*np.tanh(np.where(data["maxtominbatch_msignal"] <= -998, data["minbatch"], np.sin((data["rangebatch_slices2"])) )) +

                            0.050000*np.tanh(np.sin((np.minimum(((((np.sin((data["maxbatch_slices2_msignal"]))) * 2.0))), (((8.0))))))) +

                            0.050000*np.tanh(np.cos((((((data["meanbatch_msignal"]) / 2.0)) + (np.where(data["meanbatch_slices2"] > -998, data["meanbatch_msignal"], data["abs_maxbatch_slices2"] )))))) +

                            0.050000*np.tanh(np.minimum(((np.tanh((((((((((data["abs_avgbatch_slices2"]) + (data["stdbatch_msignal"]))) * 2.0)) + (data["signal_shift_-1_msignal"]))) - (data["minbatch_slices2_msignal"])))))), ((np.minimum(((data["stdbatch_msignal"])), ((np.minimum(((((((data["medianbatch_slices2"]) * 2.0)) + (data["stdbatch_slices2"])))), ((data["meanbatch_msignal"])))))))))) +

                            0.050000*np.tanh(np.sin((np.sin((((((data["rangebatch_slices2"]) + (np.minimum(((data["mean_abs_chgbatch_msignal"])), ((np.sin((np.minimum(((data["rangebatch_slices2"])), ((np.sin(((-((data["abs_minbatch_msignal"])))))))))))))))) - (data["abs_minbatch_msignal"]))))))) +

                            0.050000*np.tanh(((((((((((np.sin(((-((((np.sin((data["medianbatch_msignal"]))) - (data["signal_shift_+1"])))))))) - (np.sin((((data["signal_shift_+1"]) * 2.0)))))) * 2.0)) - (np.sin((((((np.sin((((data["signal_shift_+1"]) * 2.0)))) * 2.0)) * 2.0)))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) * ((((((data["stdbatch_slices2_msignal"]) + (np.minimum(((np.where(np.sin(((((((data["meanbatch_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)) * 2.0))) <= -998, (7.0), data["mean_abs_chgbatch_slices2_msignal"] ))), ((data["signal_shift_-1_msignal"])))))/2.0)) - (data["signal_shift_-1_msignal"]))))))) +

                            0.050000*np.tanh(np.sin((np.where(np.where(np.sin((data["meanbatch_slices2"])) <= -998, ((data["abs_minbatch_slices2_msignal"]) * (data["stdbatch_msignal"])), np.sin((data["minbatch_msignal"])) ) <= -998, data["signal_shift_-1_msignal"], data["minbatch_msignal"] )))) +

                            0.050000*np.tanh(np.minimum(((data["medianbatch_msignal"])), ((np.tanh((data["signal_shift_-1_msignal"])))))) +

                            0.050000*np.tanh(((data["abs_minbatch_slices2"]) + (((((np.cos((((data["signal_shift_-1"]) + ((-(((((np.maximum(((data["maxbatch_slices2"])), ((data["abs_maxbatch_msignal"])))) + (np.maximum(((data["meanbatch_slices2"])), ((((((((np.maximum(((data["signal_shift_-1"])), ((data["meanbatch_slices2"])))) + (np.maximum(((data["maxtominbatch_msignal"])), ((data["abs_maxbatch_msignal"])))))/2.0)) + (data["signal_shift_-1"]))/2.0))))))/2.0))))))))) * 2.0)) * 2.0)))) +

                            0.050000*np.tanh(((np.sin((((np.where(np.sin((((np.sin((((data["maxbatch_slices2"]) * 2.0)))) * 2.0))) > -998, data["maxbatch_slices2"], np.where(data["mean_abs_chgbatch_slices2"] > -998, data["maxbatch_slices2"], data["maxbatch_slices2"] ) )) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(((data["medianbatch_msignal"]) + (np.tanh((np.maximum(((np.where(np.minimum((((((data["minbatch_msignal"]) <= (data["medianbatch_slices2"]))*1.))), ((data["mean_abs_chgbatch_slices2_msignal"]))) > -998, data["medianbatch_slices2_msignal"], data["stdbatch_slices2"] ))), ((np.where(((data["medianbatch_slices2_msignal"]) + (np.tanh((data["abs_avgbatch_slices2_msignal"])))) > -998, ((data["rangebatch_slices2"]) / 2.0), ((data["medianbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"])) ))))))))) +

                            0.050000*np.tanh(((np.where(data["medianbatch_slices2_msignal"] <= -998, np.tanh((data["medianbatch_slices2"])), data["medianbatch_slices2"] )) / 2.0)) +

                            0.050000*np.tanh(data["meanbatch_slices2"]) +

                            0.050000*np.tanh(np.where(data["meanbatch_msignal"] > -998, np.minimum(((((np.sin((data["abs_maxbatch"]))) * (np.where(data["medianbatch_slices2_msignal"] > -998, ((data["mean_abs_chgbatch_msignal"]) * 2.0), ((data["abs_avgbatch_slices2_msignal"]) / 2.0) ))))), ((data["abs_avgbatch_slices2_msignal"]))), data["medianbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(np.minimum(((np.where(data["abs_minbatch_msignal"] <= -998, data["mean_abs_chgbatch_msignal"], data["meanbatch_slices2"] ))), ((((data["abs_minbatch_msignal"]) / 2.0))))) +

                            0.050000*np.tanh(((data["rangebatch_slices2_msignal"]) * (np.sin((np.maximum(((((data["maxbatch_slices2_msignal"]) * 2.0))), ((np.where(data["abs_minbatch_msignal"] > -998, data["signal_shift_+1"], ((data["signal"]) * (np.sin((data["stdbatch_slices2"])))) ))))))))) +

                            0.050000*np.tanh(np.minimum(((np.minimum(((((data["rangebatch_slices2"]) - (data["stdbatch_slices2_msignal"])))), ((np.sin((((data["rangebatch_slices2"]) - (data["stdbatch_slices2_msignal"]))))))))), ((((data["medianbatch_slices2_msignal"]) + ((4.06319332122802734))))))) +

                            0.050000*np.tanh(((np.sin((data["abs_maxbatch"]))) * (np.where(((data["rangebatch_slices2"]) * 2.0) > -998, data["mean_abs_chgbatch_slices2_msignal"], ((np.where(data["rangebatch_msignal"] > -998, np.where(((data["abs_maxbatch"]) * 2.0) <= -998, ((np.sin((data["abs_maxbatch"]))) * 2.0), data["abs_avgbatch_slices2_msignal"] ), data["abs_minbatch_msignal"] )) + (np.sin((data["abs_maxbatch"])))) )))) +

                            0.050000*np.tanh(((((((np.minimum(((np.sin((data["medianbatch_slices2"])))), ((np.cos((np.tanh((data["meanbatch_msignal"])))))))) * 2.0)) * 2.0)) + (np.minimum(((data["stdbatch_msignal"])), ((data["meanbatch_msignal"])))))) +

                            0.050000*np.tanh(np.minimum(((data["abs_avgbatch_msignal"])), ((np.minimum(((data["abs_maxbatch"])), ((((data["meanbatch_msignal"]) + (data["minbatch_slices2_msignal"]))))))))) +

                            0.050000*np.tanh(((data["medianbatch_slices2"]) + (data["meanbatch_slices2"]))) +

                            0.050000*np.tanh(np.where((((data["medianbatch_msignal"]) <= ((((np.tanh((data["medianbatch_slices2"]))) + (((((data["abs_avgbatch_msignal"]) / 2.0)) / 2.0)))/2.0)))*1.) > -998, (((np.minimum(((data["signal"])), ((np.cos((np.tanh((data["abs_avgbatch_msignal"])))))))) + (((np.cos((((data["medianbatch_msignal"]) + (data["meanbatch_msignal"]))))) / 2.0)))/2.0), data["medianbatch_msignal"] )) +

                            0.050000*np.tanh(((np.where((((data["abs_avgbatch_slices2_msignal"]) + (data["meanbatch_slices2"]))/2.0) <= -998, np.sin((np.sin(((6.0))))), ((np.where(data["medianbatch_msignal"] > -998, data["meanbatch_slices2"], data["meanbatch_slices2"] )) * 2.0) )) * (((data["abs_avgbatch_slices2_msignal"]) + ((((np.sin((np.tanh(((((data["abs_avgbatch_slices2_msignal"]) <= (np.sin((data["abs_avgbatch_slices2"]))))*1.)))))) <= (data["meanbatch_slices2"]))*1.)))))) +

                            0.050000*np.tanh(((data["medianbatch_msignal"]) + (((((((((((data["medianbatch_msignal"]) + (data["maxtominbatch_slices2_msignal"]))/2.0)) > (data["maxbatch_slices2"]))*1.)) * 2.0)) + (np.sin((data["abs_avgbatch_slices2"]))))))) +

                            0.050000*np.tanh(np.where((((data["stdbatch_slices2_msignal"]) > (np.minimum(((data["rangebatch_slices2"])), ((np.cos((np.minimum(((((np.cos((data["meanbatch_slices2_msignal"]))) + (np.sin((np.sin((data["meanbatch_msignal"])))))))), ((data["signal"]))))))))))*1.) > -998, np.cos((data["minbatch_msignal"])), ((data["medianbatch_slices2"]) * 2.0) )) +

                            0.050000*np.tanh(data["abs_avgbatch_slices2_msignal"]) +

                            0.050000*np.tanh(((((((data["minbatch"]) * (data["abs_minbatch_msignal"]))) * 2.0)) * (np.where(np.where(np.sin(((((np.cos(((((data["abs_minbatch_msignal"]) <= (data["minbatch"]))*1.)))) + (data["minbatch"]))/2.0))) <= -998, data["stdbatch_slices2"], np.cos((data["stdbatch_slices2"])) ) <= -998, (-((np.sin((data["stdbatch_slices2"]))))), np.cos((data["stdbatch_slices2"])) )))) +

                            0.050000*np.tanh(((((data["mean_abs_chgbatch_msignal"]) * (np.cos((np.minimum(((((data["maxtominbatch_slices2"]) - (data["rangebatch_slices2"])))), (((((data["medianbatch_slices2_msignal"]) > (np.minimum(((((data["medianbatch_slices2_msignal"]) * (np.cos((np.tanh((((np.tanh((((data["maxtominbatch_slices2"]) - (data["mean_abs_chgbatch_msignal"]))))) - (data["rangebatch_slices2"])))))))))), ((data["meanbatch_msignal"])))))*1.))))))))) * 2.0)) +

                            0.050000*np.tanh(np.where(data["mean_abs_chgbatch_msignal"] > -998, data["stdbatch_slices2_msignal"], np.sin((data["stdbatch_slices2_msignal"])) )) +

                            0.050000*np.tanh(np.minimum((((-((data["minbatch_slices2"]))))), ((((data["medianbatch_msignal"]) + (np.maximum(((data["medianbatch_msignal"])), ((np.sin((np.minimum(((data["rangebatch_msignal"])), ((((data["abs_maxbatch_msignal"]) + (np.sin((np.cos((data["maxbatch_slices2_msignal"])))))))))))))))))))) +

                            0.050000*np.tanh(np.sin((np.where(data["signal_shift_-1"] > -998, np.where(data["rangebatch_msignal"] > -998, data["minbatch_msignal"], ((data["abs_maxbatch_slices2_msignal"]) + (data["rangebatch_msignal"])) ), np.sin((data["abs_maxbatch_slices2_msignal"])) )))) +

                            0.050000*np.tanh((((data["meanbatch_msignal"]) + ((-((((data["minbatch"]) * (np.sin((((np.where(((((((data["maxbatch_slices2_msignal"]) + (np.sin((data["abs_maxbatch_slices2"]))))/2.0)) <= ((((((data["abs_maxbatch_slices2"]) * ((((data["meanbatch_msignal"]) + (data["abs_maxbatch_slices2"]))/2.0)))) > (data["maxbatch_slices2_msignal"]))*1.)))*1.) > -998, data["abs_maxbatch_slices2"], data["minbatch"] )) * 2.0))))))))))/2.0)) +

                            0.050000*np.tanh(((np.where(np.tanh((np.tanh((((data["rangebatch_msignal"]) / 2.0))))) <= -998, data["rangebatch_slices2"], np.sin((np.sin((data["minbatch_msignal"])))) )) - (np.cos((((np.tanh((data["signal_shift_-1_msignal"]))) - (((data["mean_abs_chgbatch_msignal"]) * 2.0)))))))) +

                            0.050000*np.tanh(np.sin((((data["signal_shift_+1_msignal"]) + (np.where(np.minimum(((data["abs_maxbatch_slices2_msignal"])), ((np.minimum(((data["rangebatch_msignal"])), ((np.maximum(((np.where(data["minbatch_msignal"] <= -998, ((data["signal_shift_+1_msignal"]) * 2.0), data["signal_shift_-1"] ))), ((data["signal_shift_-1_msignal"]))))))))) <= -998, data["abs_minbatch_slices2_msignal"], ((data["abs_maxbatch_msignal"]) + (data["abs_maxbatch_slices2_msignal"])) )))))) +

                            0.050000*np.tanh((((((data["medianbatch_slices2_msignal"]) + (np.sin((np.sin((np.where(np.sin((data["abs_maxbatch_slices2_msignal"])) <= -998, data["meanbatch_slices2"], ((data["medianbatch_msignal"]) + (np.sin((np.cos((data["rangebatch_msignal"])))))) )))))))/2.0)) + (np.sin((data["meanbatch_slices2"]))))))   

   

    def GP_class_5(self,data):

        return self.Output( -2.889212 +

                            0.050000*np.tanh((((data["meanbatch_slices2"]) + (data["signal"]))/2.0)) +

                            0.050000*np.tanh(np.where(data["minbatch_slices2"] <= -998, ((data["maxtominbatch"]) - ((((data["signal_shift_-1"]) <= ((((data["signal"]) <= ((((((np.where(data["medianbatch_slices2_msignal"] <= -998, np.where((4.0) <= -998, data["medianbatch_slices2"], data["abs_maxbatch_slices2_msignal"] ), data["signal_shift_-1"] )) <= (data["abs_maxbatch_slices2_msignal"]))*1.)) + (data["medianbatch_slices2"]))))*1.)))*1.))), data["signal"] )) +

                            0.050000*np.tanh((-((data["abs_minbatch_msignal"])))) +

                            0.050000*np.tanh(np.where(((((np.cos((((data["signal"]) - ((((data["medianbatch_slices2"]) <= (np.cos((np.where((((data["minbatch_slices2_msignal"]) <= (data["mean_abs_chgbatch_slices2"]))*1.) <= -998, ((((data["signal"]) - (data["medianbatch_slices2"]))) * (data["signal"])), data["medianbatch_slices2"] )))))*1.)))))) + (data["abs_minbatch_slices2"]))) * (data["minbatch_slices2"])) <= -998, data["medianbatch_slices2"], ((data["medianbatch_slices2"]) * 2.0) )) +

                            0.050000*np.tanh(((data["signal"]) - ((((((((data["signal_shift_+1"]) - (np.maximum(((data["meanbatch_slices2"])), ((data["medianbatch_slices2"])))))) - (np.cos((((data["signal_shift_+1"]) * (data["stdbatch_msignal"]))))))) + (data["abs_avgbatch_slices2"]))/2.0)))) +

                            0.050000*np.tanh(((data["abs_minbatch_msignal"]) * (np.minimum(((np.where(data["rangebatch_msignal"] <= -998, np.sin((data["minbatch_msignal"])), data["abs_avgbatch_slices2_msignal"] ))), ((((data["signal"]) - (((np.tanh((data["signal"]))) + (((((((data["maxtominbatch_slices2"]) + (data["medianbatch_slices2"]))/2.0)) > ((((data["medianbatch_slices2"]) + (data["minbatch_msignal"]))/2.0)))*1.))))))))))) +

                            0.050000*np.tanh(np.where(np.sin((np.where(data["meanbatch_slices2"] > -998, np.maximum(((data["signal_shift_-1"])), ((data["minbatch_msignal"]))), data["signal_shift_-1"] ))) > -998, data["meanbatch_slices2"], np.cos((np.where(data["meanbatch_slices2"] > -998, data["signal_shift_-1"], data["meanbatch_slices2"] ))) )) +

                            0.050000*np.tanh(data["meanbatch_slices2"]) +

                            0.050000*np.tanh(np.minimum(((data["signal_shift_+1"])), ((np.where(np.tanh(((5.89498901367187500))) <= -998, data["rangebatch_slices2_msignal"], data["signal_shift_+1"] ))))) +

                            0.050000*np.tanh(((data["medianbatch_slices2_msignal"]) - (np.where(np.where(data["medianbatch_slices2"] > -998, np.maximum(((data["signal"])), ((np.sin((np.cos((data["abs_minbatch_slices2"]))))))), np.tanh((data["signal_shift_+1"])) ) > -998, data["abs_minbatch_msignal"], data["abs_minbatch_msignal"] )))) +

                            0.050000*np.tanh(np.minimum((((-((data["abs_avgbatch_msignal"]))))), ((np.where(((data["abs_avgbatch_slices2_msignal"]) + (np.cos((data["meanbatch_slices2_msignal"])))) <= -998, data["mean_abs_chgbatch_msignal"], data["signal_shift_-1"] ))))) +

                            0.050000*np.tanh(((((((((data["stdbatch_slices2"]) - (data["maxtominbatch_slices2_msignal"]))) * 2.0)) * (np.cos((np.maximum(((data["abs_maxbatch_slices2_msignal"])), (((((np.maximum(((np.cos((np.maximum(((data["abs_maxbatch_slices2_msignal"])), (((((data["signal"]) > (((data["maxbatch_slices2"]) * (data["signal"]))))*1.)))))))), ((data["meanbatch_msignal"])))) > (data["maxtominbatch_msignal"]))*1.))))))))) * 2.0)) +

                            0.050000*np.tanh(np.where(data["mean_abs_chgbatch_slices2"] > -998, data["medianbatch_slices2"], np.maximum(((data["signal_shift_+1"])), (((((7.0)) * 2.0)))) )) +

                            0.050000*np.tanh(data["medianbatch_slices2"]) +

                            0.050000*np.tanh((((((((data["abs_minbatch_msignal"]) > ((((((data["signal_shift_-1_msignal"]) * 2.0)) > ((((data["abs_maxbatch_msignal"]) > (data["signal_shift_-1_msignal"]))*1.)))*1.)))*1.)) / 2.0)) - (((np.where((((data["signal_shift_-1_msignal"]) > ((((-((np.maximum((((8.0))), (((((data["meanbatch_slices2_msignal"]) > (data["signal_shift_-1_msignal"]))*1.)))))))) - (data["abs_avgbatch_msignal"]))))*1.) > -998, data["abs_avgbatch_msignal"], data["meanbatch_slices2_msignal"] )) * 2.0)))) +

                            0.050000*np.tanh(np.cos((((((((((data["abs_maxbatch"]) + (data["abs_maxbatch_slices2"]))/2.0)) + (data["abs_maxbatch_slices2"]))/2.0)) - (np.where(data["meanbatch_msignal"] > -998, data["medianbatch_msignal"], np.where(data["minbatch_slices2"] <= -998, data["mean_abs_chgbatch_slices2"], (-((((data["stdbatch_slices2_msignal"]) - (((data["stdbatch_slices2_msignal"]) - (data["meanbatch_msignal"]))))))) ) )))))) +

                            0.050000*np.tanh(((((((data["signal_shift_+1"]) - (data["abs_maxbatch_slices2_msignal"]))) * ((-((data["minbatch_slices2"])))))) - (((((-((((data["abs_maxbatch"]) + ((((data["stdbatch_slices2"]) <= ((((((data["signal_shift_+1_msignal"]) - ((-((data["minbatch_slices2"])))))) + ((10.0)))/2.0)))*1.))))))) <= ((-((((data["signal_shift_+1_msignal"]) - ((-((data["minbatch_slices2"]))))))))))*1.)))) +

                            0.050000*np.tanh(((np.minimum(((data["signal"])), ((np.sin((np.maximum(((data["abs_maxbatch_slices2_msignal"])), (((-(((((9.0)) + (data["signal"])))))))))))))) * 2.0)) +

                            0.050000*np.tanh(((data["medianbatch_slices2"]) - ((((data["signal_shift_-1"]) <= (((data["rangebatch_slices2_msignal"]) * (data["maxbatch_slices2"]))))*1.)))) +

                            0.050000*np.tanh(((data["meanbatch_slices2_msignal"]) - (((data["abs_avgbatch_msignal"]) * (np.where((-((data["abs_avgbatch_slices2_msignal"]))) > -998, ((data["rangebatch_slices2_msignal"]) + (data["rangebatch_slices2_msignal"])), (((data["signal_shift_-1"]) > (data["meanbatch_slices2"]))*1.) )))))) +

                            0.050000*np.tanh(((((np.sin((data["maxbatch_msignal"]))) * (np.where(((np.tanh((np.tanh((data["medianbatch_slices2"]))))) / 2.0) > -998, data["signal_shift_-1"], ((data["maxtominbatch_msignal"]) * 2.0) )))) * 2.0)) +

                            0.050000*np.tanh(np.where(np.cos((((np.maximum(((data["medianbatch_slices2"])), ((np.maximum(((data["abs_maxbatch_msignal"])), ((np.cos((data["signal"]))))))))) * (np.maximum(((np.cos((((data["maxtominbatch_slices2_msignal"]) + (data["maxbatch_msignal"])))))), ((data["mean_abs_chgbatch_slices2_msignal"]))))))) <= -998, (((np.tanh((np.cos((data["medianbatch_slices2"]))))) <= ((-((data["abs_avgbatch_slices2_msignal"])))))*1.), np.sin((data["maxbatch_msignal"])) )) +

                            0.050000*np.tanh(((np.where(np.maximum(((np.cos((np.sin((data["maxbatch_slices2"])))))), ((((data["abs_maxbatch_msignal"]) * 2.0)))) > -998, np.sin((np.maximum(((((np.sin((((np.sin((data["meanbatch_msignal"]))) * 2.0)))) * 2.0))), ((((data["abs_maxbatch_msignal"]) * 2.0)))))), np.maximum(((((data["rangebatch_msignal"]) * 2.0))), ((((data["abs_maxbatch_msignal"]) / 2.0)))) )) - (data["abs_avgbatch_slices2_msignal"]))) +

                            0.050000*np.tanh((-((((np.cos((data["meanbatch_slices2_msignal"]))) * (np.where(data["minbatch"] > -998, data["abs_maxbatch"], ((data["maxbatch_msignal"]) * ((-((((np.cos((data["meanbatch_slices2_msignal"]))) * (data["abs_maxbatch_msignal"]))))))) ))))))) +

                            0.050000*np.tanh(np.where(data["stdbatch_slices2"] <= -998, data["meanbatch_slices2"], ((data["signal"]) * (((np.where(((np.minimum((((-((data["abs_maxbatch"]))))), ((data["rangebatch_slices2_msignal"])))) * 2.0) <= -998, np.cos((data["meanbatch_slices2"])), ((np.minimum(((data["rangebatch_slices2"])), ((np.cos((data["maxbatch_msignal"])))))) * 2.0) )) * 2.0))) )) +

                            0.050000*np.tanh(((np.where(np.cos((data["meanbatch_slices2"])) <= -998, np.maximum(((np.cos((np.cos((data["maxtominbatch"])))))), ((np.sin((data["maxtominbatch"]))))), np.sin((((np.maximum(((data["maxbatch_slices2_msignal"])), ((np.where(np.maximum(((((np.sin((data["abs_maxbatch_slices2_msignal"]))) * 2.0))), ((data["stdbatch_slices2_msignal"]))) > -998, data["stdbatch_slices2_msignal"], data["abs_minbatch_slices2"] ))))) * 2.0))) )) * 2.0)) +

                            0.050000*np.tanh(np.sin((np.where(np.sin((np.where(data["signal"] > -998, ((data["mean_abs_chgbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"])), (((data["abs_maxbatch_slices2"]) > (((data["mean_abs_chgbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2"]))))*1.) ))) > -998, ((data["mean_abs_chgbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2"])), np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, ((data["mean_abs_chgbatch_slices2_msignal"]) + (data["meanbatch_slices2"])), data["mean_abs_chgbatch_slices2_msignal"] ) )))) +

                            0.050000*np.tanh(((((np.tanh((np.sin(((((2.0)) + ((((data["abs_maxbatch_slices2_msignal"]) + ((((data["medianbatch_slices2_msignal"]) + ((((data["rangebatch_slices2_msignal"]) + (np.sin((data["minbatch_slices2_msignal"]))))/2.0)))/2.0)))/2.0)))))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((np.minimum(((((np.cos((data["maxbatch_slices2_msignal"]))) * 2.0))), ((((((((((np.where(((data["meanbatch_slices2"]) * 2.0) <= -998, data["stdbatch_slices2_msignal"], ((data["signal"]) * 2.0) )) + (((np.cos((data["maxbatch_slices2_msignal"]))) * 2.0)))) + (data["abs_maxbatch_slices2_msignal"]))/2.0)) + (((data["signal"]) * 2.0)))/2.0))))) * ((9.0)))) +

                            0.050000*np.tanh(data["medianbatch_slices2"]) +

                            0.050000*np.tanh(((np.cos((data["abs_maxbatch_slices2_msignal"]))) * (np.where(((np.where(data["maxbatch_slices2"] <= -998, data["abs_maxbatch"], data["medianbatch_slices2_msignal"] )) * (np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((np.cos((((data["abs_maxbatch_msignal"]) / 2.0)))))))) > -998, (((data["abs_maxbatch_slices2"]) + ((((data["maxbatch_slices2"]) <= ((1.22195386886596680)))*1.)))/2.0), (-((np.where(data["medianbatch_slices2"] <= -998, data["abs_maxbatch"], data["medianbatch_slices2_msignal"] )))) )))) +

                            0.050000*np.tanh(((((data["maxbatch_msignal"]) * 2.0)) + (((((np.where(data["minbatch_slices2"] <= -998, ((np.sin((np.sin((np.cos((data["maxbatch_msignal"]))))))) * 2.0), ((np.tanh((data["maxbatch_msignal"]))) * (((data["abs_maxbatch_slices2"]) * (data["abs_avgbatch_slices2"])))) )) * 2.0)) * (np.cos((data["maxbatch_msignal"]))))))) +

                            0.050000*np.tanh(np.sin((np.maximum(((data["abs_minbatch_msignal"])), ((np.maximum(((data["abs_maxbatch_msignal"])), ((((np.sin((np.sin((data["maxbatch_slices2_msignal"]))))) * 2.0)))))))))) +

                            0.050000*np.tanh(((np.sin((np.where(((data["maxtominbatch"]) + (np.sin(((((((np.sin((data["abs_maxbatch_slices2_msignal"]))) * 2.0)) <= (data["abs_maxbatch_msignal"]))*1.))))) > -998, ((data["abs_maxbatch_slices2_msignal"]) * 2.0), np.where(data["mean_abs_chgbatch_slices2"] <= -998, np.sin((np.where(data["abs_maxbatch_slices2_msignal"] > -998, np.sin((data["maxbatch_slices2_msignal"])), data["signal_shift_+1_msignal"] ))), data["stdbatch_msignal"] ) )))) * 2.0)) +

                            0.050000*np.tanh(((data["meanbatch_msignal"]) * ((-((np.where(data["rangebatch_slices2_msignal"] > -998, np.sin((data["abs_avgbatch_slices2"])), ((np.where((((2.0)) * (data["meanbatch_msignal"])) > -998, data["meanbatch_msignal"], ((data["signal_shift_+1"]) - (data["mean_abs_chgbatch_msignal"])) )) * (((data["mean_abs_chgbatch_msignal"]) / 2.0))) ))))))) +

                            0.050000*np.tanh(((np.cos((data["maxbatch_msignal"]))) * (((np.maximum(((np.minimum(((data["mean_abs_chgbatch_msignal"])), ((data["minbatch_msignal"]))))), ((((data["medianbatch_slices2"]) - ((((data["signal"]) > (np.minimum(((data["medianbatch_slices2"])), ((data["signal_shift_+1_msignal"])))))*1.))))))) * (data["signal"]))))) +

                            0.050000*np.tanh(((((np.cos((data["signal"]))) * (data["meanbatch_slices2_msignal"]))) - ((((np.sin((np.sin((data["mean_abs_chgbatch_msignal"]))))) > (((np.cos((np.tanh((data["meanbatch_slices2_msignal"]))))) * ((((((((np.cos((data["mean_abs_chgbatch_msignal"]))) <= (data["minbatch"]))*1.)) * (data["medianbatch_msignal"]))) * (data["medianbatch_msignal"]))))))*1.)))) +

                            0.050000*np.tanh((((13.67484474182128906)) * (np.where(np.where((13.67484474182128906) > -998, data["abs_maxbatch_slices2_msignal"], np.sin((data["abs_maxbatch_slices2_msignal"])) ) <= -998, np.tanh((np.sin((data["abs_maxbatch_msignal"])))), np.sin((np.maximum(((data["abs_maxbatch_msignal"])), ((((data["abs_maxbatch"]) / 2.0)))))) )))) +

                            0.050000*np.tanh(((((np.sin((data["abs_maxbatch"]))) * (((data["stdbatch_msignal"]) * 2.0)))) - (np.tanh(((((data["medianbatch_msignal"]) <= (((data["abs_maxbatch"]) - (((data["abs_maxbatch"]) * (((data["stdbatch_msignal"]) * 2.0)))))))*1.)))))) +

                            0.050000*np.tanh(((np.cos(((-((np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], np.where(data["abs_maxbatch_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_msignal"], ((data["medianbatch_slices2"]) / 2.0) ) ) ))))))) * 2.0)) +

                            0.050000*np.tanh((((8.75535774230957031)) - (((((((10.0)) * (data["abs_maxbatch_slices2_msignal"]))) + (np.sin((((np.where(np.tanh((data["signal_shift_+1"])) <= -998, np.where(data["mean_abs_chgbatch_msignal"] <= -998, data["abs_maxbatch_slices2_msignal"], data["mean_abs_chgbatch_msignal"] ), (10.0) )) * 2.0)))))/2.0)))) +

                            0.050000*np.tanh((((((data["abs_maxbatch_msignal"]) + (((data["maxtominbatch_slices2_msignal"]) / 2.0)))/2.0)) * (((data["meanbatch_msignal"]) - (np.cos((data["meanbatch_msignal"]))))))) +

                            0.050000*np.tanh(((data["maxbatch_slices2"]) * (np.sin((np.where(np.cos((data["maxbatch_slices2_msignal"])) > -998, np.where(np.where(data["stdbatch_slices2"] > -998, data["maxbatch_msignal"], data["meanbatch_msignal"] ) > -998, (((np.where(data["maxbatch_slices2_msignal"] > -998, data["maxbatch_msignal"], data["maxbatch_slices2_msignal"] )) + (data["abs_maxbatch_msignal"]))/2.0), np.sin((data["maxbatch_slices2"])) ), data["maxbatch_slices2"] )))))) +

                            0.050000*np.tanh((((((((data["maxtominbatch"]) <= (np.cos((data["meanbatch_msignal"]))))*1.)) - (((np.tanh((np.cos((data["abs_maxbatch_slices2_msignal"]))))) * (np.where(np.cos((data["meanbatch_msignal"])) > -998, data["maxtominbatch_slices2_msignal"], ((data["maxbatch_msignal"]) + (data["signal_shift_-1_msignal"])) )))))) - (((((np.cos((data["meanbatch_msignal"]))) * 2.0)) * 2.0)))) +

                            0.050000*np.tanh(np.where(np.tanh(((((np.where(data["signal_shift_+1"] <= -998, np.sin((data["minbatch"])), ((np.maximum(((data["rangebatch_msignal"])), ((data["medianbatch_slices2"])))) + (data["medianbatch_slices2"])) )) <= (data["meanbatch_slices2_msignal"]))*1.))) > -998, np.cos((data["abs_maxbatch_slices2_msignal"])), data["medianbatch_slices2"] )) +

                            0.050000*np.tanh(np.cos((np.where(((np.where(np.sin((data["minbatch"])) <= -998, data["minbatch"], data["maxbatch_msignal"] )) + (data["abs_avgbatch_slices2_msignal"])) > -998, data["abs_maxbatch_slices2_msignal"], (((data["signal_shift_+1"]) + (data["meanbatch_msignal"]))/2.0) )))) +

                            0.050000*np.tanh(((np.cos((data["abs_maxbatch_slices2_msignal"]))) * 2.0)) +

                            0.050000*np.tanh(((np.where(((data["medianbatch_slices2"]) + (np.sin((data["abs_maxbatch_msignal"])))) <= -998, data["abs_maxbatch_msignal"], np.where(np.maximum(((data["abs_maxbatch_msignal"])), (((-((((data["maxbatch_slices2"]) - (data["abs_maxbatch_msignal"])))))))) <= -998, data["maxtominbatch_msignal"], np.minimum(((np.sin((data["abs_maxbatch_msignal"])))), ((data["medianbatch_slices2"]))) ) )) * 2.0)) +

                            0.050000*np.tanh(np.where(np.cos((np.cos((data["medianbatch_slices2_msignal"])))) <= -998, ((data["abs_maxbatch_slices2"]) - (np.where(data["medianbatch_slices2_msignal"] <= -998, data["minbatch_slices2"], ((data["abs_minbatch_slices2_msignal"]) * (data["medianbatch_slices2_msignal"])) ))), ((np.cos((data["abs_minbatch_slices2_msignal"]))) * (data["medianbatch_slices2_msignal"])) )) +

                            0.050000*np.tanh(((np.where(data["maxbatch_slices2_msignal"] <= -998, data["minbatch_slices2"], np.cos((data["maxbatch_slices2_msignal"])) )) * 2.0)) +

                            0.050000*np.tanh(((np.cos((data["meanbatch_msignal"]))) * (np.minimum(((data["minbatch_msignal"])), (((-((data["meanbatch_msignal"]))))))))) +

                            0.050000*np.tanh(((((data["maxbatch_slices2_msignal"]) * (np.cos((((np.minimum(((data["rangebatch_slices2"])), ((np.cos((data["abs_minbatch_msignal"])))))) - (np.maximum(((data["abs_maxbatch_slices2"])), ((np.sin((np.where(data["abs_avgbatch_msignal"] > -998, data["maxbatch_slices2_msignal"], data["signal"] ))))))))))))) * (data["maxbatch_slices2"]))) +

                            0.050000*np.tanh(np.cos((((np.minimum(((((((((data["medianbatch_slices2"]) > (np.tanh((data["signal_shift_-1"]))))*1.)) > (((np.where(data["signal_shift_+1"] > -998, np.sin(((((data["medianbatch_slices2"]) > (data["minbatch_slices2"]))*1.))), data["mean_abs_chgbatch_slices2_msignal"] )) * 2.0)))*1.))), ((np.where(data["medianbatch_slices2"] > -998, data["mean_abs_chgbatch_slices2_msignal"], data["medianbatch_slices2"] ))))) * 2.0)))) +

                            0.050000*np.tanh(((((np.sin((np.maximum(((np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((data["abs_avgbatch_slices2"]))))), ((np.cos((data["abs_avgbatch_slices2"])))))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((data["maxbatch_slices2"]) * (np.sin((np.maximum(((data["maxbatch_msignal"])), (((-((np.where(np.sin((data["abs_maxbatch_slices2_msignal"])) <= -998, ((data["maxbatch_msignal"]) / 2.0), data["stdbatch_msignal"] )))))))))))) +

                            0.050000*np.tanh(((data["abs_maxbatch"]) * (((data["abs_maxbatch"]) * (((data["mean_abs_chgbatch_msignal"]) * (np.where(np.sin((data["maxbatch_slices2_msignal"])) > -998, ((np.sin((np.maximum(((data["maxbatch_slices2"])), (((-((data["abs_maxbatch_slices2_msignal"]))))))))) + (np.tanh((np.tanh((np.where(data["abs_minbatch_msignal"] > -998, data["abs_minbatch_slices2"], data["signal"] ))))))), data["abs_avgbatch_msignal"] )))))))) +

                            0.050000*np.tanh(((data["stdbatch_slices2_msignal"]) * (np.cos((np.where((13.96799564361572266) <= -998, np.where(((data["signal_shift_+1_msignal"]) * 2.0) <= -998, data["meanbatch_msignal"], np.where(data["stdbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_msignal"], data["abs_avgbatch_msignal"] ) ), np.where(data["signal_shift_+1_msignal"] > -998, data["mean_abs_chgbatch_msignal"], ((((data["stdbatch_slices2_msignal"]) * (((data["medianbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_msignal"]))))) * (data["minbatch_slices2_msignal"])) ) )))))) +

                            0.050000*np.tanh(np.where(data["abs_maxbatch_slices2_msignal"] <= -998, np.sin((np.maximum(((((np.maximum((((((data["abs_maxbatch_slices2_msignal"]) > (data["abs_maxbatch"]))*1.))), ((data["abs_maxbatch"])))) * 2.0))), ((((((data["medianbatch_slices2"]) * 2.0)) * 2.0)))))), ((np.sin((((data["abs_maxbatch_slices2_msignal"]) * 2.0)))) * 2.0) )) +

                            0.050000*np.tanh(((data["medianbatch_slices2_msignal"]) * (np.where(data["medianbatch_slices2_msignal"] <= -998, np.where(data["minbatch_slices2"] <= -998, ((((((data["medianbatch_slices2_msignal"]) + (data["maxbatch_slices2"]))/2.0)) <= ((((data["medianbatch_slices2_msignal"]) + ((((np.sin((data["mean_abs_chgbatch_msignal"]))) <= ((9.99973869323730469)))*1.)))/2.0)))*1.), np.sin((data["maxbatch_slices2"])) ), np.sin((data["maxbatch_slices2"])) )))) +

                            0.050000*np.tanh(((((data["abs_avgbatch_slices2_msignal"]) * (np.where(np.sin((data["abs_maxbatch_slices2"])) > -998, np.sin((data["abs_maxbatch_slices2"])), ((data["stdbatch_slices2_msignal"]) * (data["abs_avgbatch_slices2_msignal"])) )))) * 2.0)) +

                            0.050000*np.tanh(((((data["meanbatch_msignal"]) * (np.cos((data["signal"]))))) - (np.sin(((((data["meanbatch_slices2"]) > (np.where(((data["meanbatch_msignal"]) * (data["mean_abs_chgbatch_slices2_msignal"])) > -998, ((data["meanbatch_msignal"]) * (np.cos((data["mean_abs_chgbatch_slices2_msignal"])))), data["mean_abs_chgbatch_slices2_msignal"] )))*1.)))))) +

                            0.050000*np.tanh(((((data["mean_abs_chgbatch_slices2_msignal"]) * (((np.where(((data["meanbatch_msignal"]) * 2.0) > -998, data["abs_maxbatch_msignal"], data["medianbatch_slices2"] )) * (np.sin((data["maxbatch_slices2"]))))))) * 2.0)) +

                            0.050000*np.tanh(((np.sin((np.maximum(((np.maximum(((data["maxbatch_slices2_msignal"])), ((np.maximum(((np.maximum(((data["maxbatch_slices2_msignal"])), ((data["abs_minbatch_slices2"]))))), ((data["abs_avgbatch_slices2"])))))))), ((np.maximum((((-((np.maximum(((data["maxtominbatch_msignal"])), ((data["abs_maxbatch_msignal"])))))))), ((data["abs_avgbatch_slices2"]))))))))) * 2.0)) +

                            0.050000*np.tanh(((((data["maxbatch_slices2"]) * (np.sin((np.sin((np.sin((np.minimum(((((np.sin((((data["abs_maxbatch_slices2_msignal"]) * 2.0)))) * 2.0))), ((np.sin((((data["abs_maxbatch_slices2_msignal"]) * 2.0))))))))))))))) * 2.0)) +

                            0.050000*np.tanh(((((((((np.cos((data["maxbatch_slices2_msignal"]))) * (((data["abs_maxbatch"]) + (((data["signal_shift_-1"]) - (((np.where(((((np.cos((data["maxbatch_slices2_msignal"]))) * 2.0)) * (np.cos((data["maxbatch_slices2_msignal"])))) > -998, data["rangebatch_slices2_msignal"], data["stdbatch_slices2"] )) / 2.0)))))))) * (data["maxbatch_slices2_msignal"]))) * ((10.12534904479980469)))) * (data["maxbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(np.where(((data["abs_avgbatch_slices2_msignal"]) * (np.cos(((-(((-((data["abs_minbatch_msignal"])))))))))) <= -998, data["abs_avgbatch_slices2_msignal"], ((data["abs_avgbatch_slices2_msignal"]) * (np.cos(((-((data["abs_minbatch_slices2_msignal"]))))))) )) +

                            0.050000*np.tanh(np.minimum(((data["medianbatch_slices2"])), ((np.where(np.where(np.where(data["abs_maxbatch"] > -998, data["medianbatch_slices2"], data["medianbatch_slices2"] ) > -998, data["medianbatch_slices2"], data["medianbatch_slices2"] ) > -998, ((((np.cos((data["signal"]))) * (((data["abs_minbatch_slices2_msignal"]) + (data["medianbatch_slices2"]))))) * 2.0), data["medianbatch_slices2"] ))))) +

                            0.050000*np.tanh(((np.where(np.tanh((((np.cos((data["abs_maxbatch_slices2_msignal"]))) * 2.0))) <= -998, ((np.cos((((data["meanbatch_slices2"]) - (data["abs_maxbatch_slices2_msignal"]))))) - (data["abs_maxbatch_slices2_msignal"])), ((np.maximum(((data["mean_abs_chgbatch_msignal"])), ((data["abs_maxbatch"])))) + ((8.0))) )) * (np.cos((data["abs_maxbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(((((((np.maximum(((((data["meanbatch_slices2"]) * (np.sin((data["maxbatch_slices2_msignal"])))))), ((data["signal_shift_+1_msignal"])))) - (data["stdbatch_msignal"]))) * (data["stdbatch_msignal"]))) * (data["abs_avgbatch_slices2"]))) +

                            0.050000*np.tanh(((np.sin((np.where(np.sin((data["abs_maxbatch_msignal"])) > -998, np.sin((np.where(data["abs_maxbatch_msignal"] > -998, data["abs_maxbatch_msignal"], ((np.where(np.sin((data["mean_abs_chgbatch_msignal"])) > -998, ((data["abs_maxbatch_msignal"]) * 2.0), np.sin((data["maxbatch_slices2_msignal"])) )) * 2.0) ))), data["abs_avgbatch_msignal"] )))) * 2.0)) +

                            0.050000*np.tanh(np.where(np.sin((data["abs_maxbatch_slices2"])) <= -998, np.sin((data["abs_maxbatch_slices2"])), ((data["meanbatch_slices2_msignal"]) * (np.sin((data["abs_maxbatch_slices2"])))) )) +

                            0.050000*np.tanh(np.where(data["maxbatch_slices2"] > -998, ((data["stdbatch_msignal"]) * (np.sin((data["maxbatch_slices2"])))), np.where(((data["mean_abs_chgbatch_slices2_msignal"]) * (np.sin((data["maxbatch_slices2"])))) > -998, data["abs_avgbatch_msignal"], data["stdbatch_slices2"] ) )) +

                            0.050000*np.tanh(((data["medianbatch_slices2_msignal"]) * (np.where(((data["medianbatch_slices2_msignal"]) * (np.cos((np.where(data["stdbatch_msignal"] > -998, data["mean_abs_chgbatch_msignal"], np.cos((np.where(data["signal"] > -998, data["medianbatch_msignal"], np.cos((data["signal"])) ))) ))))) > -998, np.cos((data["signal"])), data["mean_abs_chgbatch_msignal"] )))) +

                            0.050000*np.tanh(((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, data["meanbatch_slices2_msignal"], np.cos((np.where(np.where(data["abs_maxbatch"] <= -998, (((data["abs_maxbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"]))/2.0), data["abs_maxbatch_msignal"] ) <= -998, data["abs_maxbatch"], data["abs_maxbatch_slices2_msignal"] ))) )) * 2.0)) +

                            0.050000*np.tanh(((((data["maxbatch_slices2_msignal"]) + (data["stdbatch_msignal"]))) * (np.sin((np.where((((data["mean_abs_chgbatch_msignal"]) <= (((data["mean_abs_chgbatch_msignal"]) + (data["mean_abs_chgbatch_msignal"]))))*1.) > -998, data["stdbatch_msignal"], data["maxbatch_slices2_msignal"] )))))) +

                            0.050000*np.tanh(np.where(data["signal_shift_+1_msignal"] <= -998, ((np.cos((((((data["minbatch_msignal"]) * (np.cos(((((-((np.cos((data["minbatch_msignal"])))))) + (data["medianbatch_slices2_msignal"]))))))) + (data["abs_avgbatch_slices2_msignal"]))))) + (data["medianbatch_slices2_msignal"])), ((data["minbatch_msignal"]) * (np.cos(((((-((np.cos((data["minbatch_msignal"])))))) + (data["abs_avgbatch_slices2_msignal"])))))) )) +

                            0.050000*np.tanh(((np.sin((np.maximum(((data["meanbatch_slices2"])), ((data["abs_maxbatch_msignal"])))))) * (np.maximum(((((data["rangebatch_msignal"]) - (((((data["abs_avgbatch_slices2"]) + (data["rangebatch_slices2"]))) + (data["abs_maxbatch_msignal"])))))), ((data["abs_avgbatch_slices2"])))))) +

                            0.050000*np.tanh(np.where(np.maximum(((data["signal_shift_+1_msignal"])), ((((np.sin((np.maximum(((data["signal_shift_+1_msignal"])), ((((data["abs_maxbatch_msignal"]) * 2.0))))))) * 2.0)))) <= -998, ((data["signal"]) * 2.0), ((np.where((2.0) <= -998, data["signal"], np.sin((np.maximum(((np.sin((((data["signal"]) / 2.0))))), ((((data["signal"]) * 2.0)))))) )) * 2.0) )) +

                            0.050000*np.tanh(np.where(np.sin((data["stdbatch_msignal"])) > -998, np.sin((((data["stdbatch_msignal"]) * 2.0))), data["stdbatch_msignal"] )) +

                            0.050000*np.tanh(((np.where(np.where(data["signal_shift_-1_msignal"] > -998, data["abs_minbatch_slices2_msignal"], data["stdbatch_slices2_msignal"] ) <= -998, np.cos((np.where(((data["abs_maxbatch_slices2_msignal"]) * 2.0) <= -998, data["abs_maxbatch_slices2_msignal"], np.where(((data["abs_maxbatch_slices2_msignal"]) * 2.0) <= -998, data["abs_minbatch_slices2_msignal"], data["abs_maxbatch_slices2_msignal"] ) ))), np.sin((((np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((data["abs_maxbatch_msignal"])))) * 2.0))) )) * 2.0)) +

                            0.050000*np.tanh(((np.sin((np.where(np.sin((data["abs_avgbatch_msignal"])) > -998, np.sin((data["abs_avgbatch_msignal"])), data["signal_shift_+1"] )))) * (((((np.sin((data["abs_avgbatch_msignal"]))) * 2.0)) + (((data["minbatch_msignal"]) - (np.where(data["rangebatch_slices2"] > -998, np.sin((np.sin((data["abs_avgbatch_msignal"])))), (((np.sin((data["meanbatch_msignal"]))) <= (data["stdbatch_slices2"]))*1.) )))))))) +

                            0.050000*np.tanh((((((data["stdbatch_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)) * (((data["maxbatch_msignal"]) - (((np.sin((data["maxbatch_msignal"]))) * 2.0)))))) +

                            0.050000*np.tanh(data["meanbatch_msignal"]) +

                            0.050000*np.tanh(((((np.where(data["minbatch"] <= -998, ((data["rangebatch_slices2"]) - (data["mean_abs_chgbatch_slices2"])), ((np.sin((np.sin((data["abs_maxbatch"]))))) * 2.0) )) * (((data["stdbatch_msignal"]) * (data["medianbatch_slices2"]))))) * 2.0)) +

                            0.050000*np.tanh(((np.where(np.tanh(((((data["abs_minbatch_slices2"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0))) <= -998, np.cos((data["minbatch_slices2_msignal"])), np.cos((np.where((((data["abs_maxbatch_msignal"]) <= (data["meanbatch_msignal"]))*1.) > -998, data["mean_abs_chgbatch_msignal"], np.cos((data["rangebatch_slices2"])) ))) )) * (data["meanbatch_msignal"]))) +

                            0.050000*np.tanh(((data["maxtominbatch_msignal"]) + (((data["rangebatch_slices2"]) * (((((data["stdbatch_slices2_msignal"]) * (np.cos((data["mean_abs_chgbatch_slices2_msignal"]))))) * (np.cos((np.where(np.sin((np.maximum(((data["stdbatch_slices2"])), ((data["abs_maxbatch"]))))) > -998, (((data["maxtominbatch_msignal"]) > (data["maxbatch_slices2_msignal"]))*1.), np.cos((data["mean_abs_chgbatch_slices2_msignal"])) )))))))))) +

                            0.050000*np.tanh(((np.where(data["meanbatch_slices2_msignal"] <= -998, np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], np.sin((((data["maxbatch_msignal"]) * 2.0))) ), data["meanbatch_msignal"] )) * (np.cos((data["abs_minbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(((((np.minimum(((np.cos((data["abs_maxbatch_slices2_msignal"])))), ((np.sin((((data["abs_maxbatch_slices2_msignal"]) * 2.0))))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((data["meanbatch_msignal"]) * (np.minimum(((np.where(data["meanbatch_msignal"] <= -998, np.cos((data["meanbatch_msignal"])), np.sin((np.where(data["meanbatch_msignal"] <= -998, np.cos((data["meanbatch_msignal"])), np.sin((data["stdbatch_slices2_msignal"])) ))) ))), ((np.where(data["abs_maxbatch"] <= -998, np.maximum(((np.sin((data["meanbatch_msignal"])))), ((((data["maxbatch_msignal"]) / 2.0)))), np.sin((data["maxbatch_slices2"])) ))))))) +

                            0.050000*np.tanh(((data["medianbatch_msignal"]) * (((np.cos((np.minimum(((data["stdbatch_slices2"])), ((np.where(data["medianbatch_slices2"] > -998, np.where(data["stdbatch_slices2"] > -998, data["abs_minbatch_slices2_msignal"], ((data["abs_minbatch_slices2_msignal"]) * (((np.cos((data["stdbatch_slices2"]))) * 2.0))) ), data["medianbatch_msignal"] ))))))) * 2.0)))) +

                            0.050000*np.tanh(((((np.where((7.0) > -998, ((data["meanbatch_slices2_msignal"]) + (((data["abs_minbatch_msignal"]) * (np.cos((data["mean_abs_chgbatch_msignal"])))))), ((data["meanbatch_slices2_msignal"]) + (((data["meanbatch_slices2_msignal"]) * (np.cos((data["stdbatch_msignal"])))))) )) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.cos(((((((data["abs_avgbatch_slices2_msignal"]) + (data["rangebatch_slices2"]))/2.0)) - (np.where(np.sin((np.minimum(((data["rangebatch_slices2"])), ((np.where(data["abs_maxbatch_msignal"] > -998, data["stdbatch_slices2"], (3.0) )))))) > -998, data["stdbatch_slices2"], np.sin((data["abs_avgbatch_slices2_msignal"])) )))))) +

                            0.050000*np.tanh(((np.cos((data["mean_abs_chgbatch_msignal"]))) * (np.where(np.tanh((((data["mean_abs_chgbatch_msignal"]) * (np.where(data["signal_shift_+1_msignal"] > -998, data["mean_abs_chgbatch_msignal"], ((np.cos((data["mean_abs_chgbatch_msignal"]))) * (data["meanbatch_slices2_msignal"])) ))))) > -998, data["medianbatch_slices2_msignal"], data["mean_abs_chgbatch_msignal"] )))) +

                            0.050000*np.tanh(((np.where(data["medianbatch_slices2"] <= -998, np.where(data["abs_minbatch_slices2_msignal"] <= -998, data["maxbatch_slices2"], ((data["stdbatch_msignal"]) * (np.where(data["signal_shift_-1_msignal"] <= -998, data["maxtominbatch_msignal"], ((np.cos((data["maxtominbatch_msignal"]))) - (data["minbatch_msignal"])) ))) ), ((data["abs_avgbatch_msignal"]) * (np.cos((((data["medianbatch_msignal"]) * 2.0))))) )) * 2.0)) +

                            0.050000*np.tanh(((np.where(((data["mean_abs_chgbatch_slices2_msignal"]) - (((data["maxtominbatch_slices2_msignal"]) / 2.0))) > -998, np.cos((np.maximum(((data["stdbatch_msignal"])), ((data["maxtominbatch_slices2_msignal"]))))), ((data["mean_abs_chgbatch_slices2_msignal"]) + (data["medianbatch_slices2_msignal"])) )) * (((data["mean_abs_chgbatch_slices2_msignal"]) + (data["medianbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(((data["meanbatch_slices2_msignal"]) * (np.where(np.where(data["mean_abs_chgbatch_msignal"] > -998, data["medianbatch_slices2_msignal"], (-((data["rangebatch_msignal"]))) ) <= -998, np.where((((np.cos((np.where(data["meanbatch_slices2_msignal"] <= -998, (((data["medianbatch_slices2_msignal"]) > (data["signal_shift_-1"]))*1.), np.cos((data["abs_minbatch_msignal"])) )))) > (data["signal_shift_-1"]))*1.) <= -998, data["abs_minbatch_msignal"], data["stdbatch_slices2"] ), np.cos((data["abs_minbatch_msignal"])) )))) +

                            0.050000*np.tanh(((data["stdbatch_msignal"]) * (np.sin((((data["abs_maxbatch_slices2"]) + (np.sin((((data["abs_maxbatch_slices2"]) * (np.sin((data["signal_shift_+1"]))))))))))))) +

                            0.050000*np.tanh((((data["meanbatch_msignal"]) + (((((data["medianbatch_slices2"]) + (((((((((data["abs_minbatch_msignal"]) + (np.sin((data["meanbatch_msignal"]))))) + (data["abs_avgbatch_slices2_msignal"]))) + (((np.cos((data["signal"]))) + ((((data["abs_avgbatch_slices2_msignal"]) <= (((((np.cos((data["stdbatch_slices2_msignal"]))) / 2.0)) * 2.0)))*1.)))))) * 2.0)))) * (np.cos((data["stdbatch_slices2_msignal"]))))))/2.0)) +

                            0.050000*np.tanh(np.where(data["meanbatch_slices2_msignal"] <= -998, np.maximum((((-((np.maximum((((-((data["meanbatch_slices2_msignal"]))))), ((data["abs_maxbatch_slices2_msignal"])))))))), ((data["abs_maxbatch_slices2_msignal"]))), np.minimum(((np.maximum((((8.0))), ((np.sin((data["maxtominbatch_slices2"]))))))), ((((np.minimum((((8.0))), ((np.sin((np.maximum((((-((data["meanbatch_msignal"]))))), ((data["abs_maxbatch_slices2_msignal"]))))))))) * 2.0)))) )) +

                            0.050000*np.tanh(np.minimum(((((((data["stdbatch_msignal"]) * (((data["signal"]) + (((data["abs_avgbatch_slices2_msignal"]) * (np.sin((data["signal"]))))))))) * (np.sin((data["maxbatch_slices2"])))))), ((((np.sin((np.where(data["stdbatch_msignal"] <= -998, data["signal"], ((np.where(((data["maxbatch_slices2"]) * (data["abs_maxbatch_msignal"])) > -998, data["signal"], data["abs_maxbatch"] )) * 2.0) )))) * 2.0))))) +

                            0.050000*np.tanh(((np.sin((data["meanbatch_msignal"]))) * (((data["stdbatch_slices2_msignal"]) + (((np.sin((np.sin((data["stdbatch_slices2_msignal"]))))) - (np.sin((((data["abs_avgbatch_slices2"]) - (np.sin((data["rangebatch_slices2"]))))))))))))) +

                            0.050000*np.tanh(((((data["stdbatch_msignal"]) * (((((((np.sin((data["abs_maxbatch_slices2"]))) * (((((data["abs_maxbatch_slices2"]) + (data["mean_abs_chgbatch_msignal"]))) * 2.0)))) * 2.0)) * 2.0)))) + (((((3.0)) + (np.minimum(((data["abs_maxbatch_slices2"])), ((data["stdbatch_msignal"])))))/2.0)))) +

                            0.050000*np.tanh(((np.cos((data["signal"]))) * (np.where((4.17584276199340820) > -998, ((np.tanh((data["meanbatch_msignal"]))) + (data["meanbatch_msignal"])), np.sin((((data["meanbatch_msignal"]) + (data["meanbatch_msignal"])))) )))) +

                            0.050000*np.tanh(((((data["signal_shift_+1"]) * (data["meanbatch_msignal"]))) * (np.minimum(((np.sin((((np.sin((data["minbatch_msignal"]))) * 2.0))))), ((((np.where(np.where(data["minbatch_msignal"] <= -998, data["signal_shift_+1"], ((((data["signal_shift_+1"]) * (data["abs_avgbatch_slices2_msignal"]))) * (np.cos((data["signal"])))) ) <= -998, data["signal"], data["signal_shift_+1"] )) * 2.0))))))) +

                            0.050000*np.tanh(((np.cos((((np.where(data["minbatch"] <= -998, data["medianbatch_msignal"], data["medianbatch_slices2_msignal"] )) - ((-((((data["maxtominbatch_slices2"]) - ((((data["stdbatch_slices2"]) + (((data["rangebatch_slices2_msignal"]) + (np.cos((np.tanh(((((data["medianbatch_msignal"]) <= (data["meanbatch_slices2_msignal"]))*1.)))))))))/2.0))))))))))) * 2.0)) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) * (np.tanh(((((((((data["stdbatch_slices2_msignal"]) + ((10.58995056152343750)))) > (data["abs_maxbatch"]))*1.)) - (((data["meanbatch_slices2_msignal"]) * (np.sin((data["abs_maxbatch"]))))))))))) +

                            0.050000*np.tanh(np.where(((((data["medianbatch_slices2_msignal"]) * 2.0)) / 2.0) <= -998, (((data["stdbatch_slices2_msignal"]) + (np.sin((data["stdbatch_slices2"]))))/2.0), np.minimum(((data["stdbatch_slices2_msignal"])), ((np.maximum((((13.43751239776611328))), ((((data["stdbatch_msignal"]) * (np.sin((np.minimum(((data["abs_maxbatch"])), ((data["medianbatch_msignal"]))))))))))))) )) +

                            0.050000*np.tanh(((((np.maximum(((np.maximum(((np.cos((data["maxtominbatch_msignal"])))), (((13.34599208831787109)))))), ((data["maxtominbatch_slices2"])))) - (data["medianbatch_slices2"]))) * (np.where(np.cos((np.cos((data["maxbatch_slices2_msignal"])))) <= -998, data["abs_minbatch_slices2"], np.cos((data["maxbatch_slices2_msignal"])) )))) +

                            0.050000*np.tanh(((data["medianbatch_msignal"]) + (np.where((((((np.sin((data["meanbatch_msignal"]))) + ((14.61012458801269531)))/2.0)) * (data["signal_shift_-1"])) > -998, data["abs_minbatch_msignal"], ((data["abs_minbatch_msignal"]) * 2.0) )))) +

                            0.050000*np.tanh(np.where(data["minbatch"] > -998, ((np.sin(((((data["maxtominbatch_slices2_msignal"]) + (((np.minimum(((data["abs_maxbatch"])), ((((np.minimum(((data["mean_abs_chgbatch_slices2"])), ((data["rangebatch_slices2_msignal"])))) * (data["minbatch_msignal"])))))) / 2.0)))/2.0)))) * 2.0), (((data["abs_maxbatch"]) > (data["meanbatch_slices2_msignal"]))*1.) )) +

                            0.050000*np.tanh(((np.where(data["signal_shift_-1"] <= -998, np.where(np.tanh((np.minimum(((data["mean_abs_chgbatch_msignal"])), ((data["stdbatch_slices2_msignal"]))))) > -998, data["maxtominbatch"], data["stdbatch_slices2_msignal"] ), ((data["stdbatch_msignal"]) * 2.0) )) * (np.cos((data["mean_abs_chgbatch_msignal"]))))) +

                            0.050000*np.tanh(((np.minimum((((((data["stdbatch_slices2"]) > (np.cos((data["signal"]))))*1.))), ((data["signal_shift_+1_msignal"])))) * 2.0)) +

                            0.050000*np.tanh(((np.cos((data["abs_minbatch_msignal"]))) * (np.where(data["abs_avgbatch_slices2"] > -998, data["meanbatch_msignal"], ((((np.cos((data["abs_minbatch_msignal"]))) * (np.where(data["minbatch_slices2_msignal"] > -998, data["meanbatch_msignal"], np.cos((data["abs_minbatch_msignal"])) )))) * (((data["maxbatch_slices2_msignal"]) * (data["meanbatch_msignal"])))) )))) +

                            0.050000*np.tanh(((data["medianbatch_msignal"]) * (np.where(data["abs_avgbatch_slices2"] <= -998, data["medianbatch_msignal"], np.minimum(((np.where(data["medianbatch_msignal"] <= -998, data["medianbatch_msignal"], np.cos((data["mean_abs_chgbatch_msignal"])) ))), ((data["rangebatch_slices2"]))) )))) +

                            0.050000*np.tanh(np.where(np.where(data["signal_shift_+1"] <= -998, np.maximum(((data["signal_shift_-1_msignal"])), ((((((data["medianbatch_msignal"]) * (data["abs_maxbatch_slices2"]))) - (data["maxbatch_msignal"]))))), data["medianbatch_slices2"] ) <= -998, np.sin((data["minbatch_msignal"])), np.minimum(((np.sin((data["maxbatch_msignal"])))), ((((data["medianbatch_msignal"]) * (np.sin((data["minbatch_msignal"]))))))) )) +

                            0.050000*np.tanh(np.sin((np.maximum(((data["meanbatch_msignal"])), ((np.maximum(((((data["abs_maxbatch_msignal"]) - (data["meanbatch_msignal"])))), ((data["abs_maxbatch_msignal"]))))))))) +

                            0.050000*np.tanh(((np.tanh((np.sin((((data["meanbatch_msignal"]) - (data["signal_shift_-1"]))))))) + (np.minimum(((((np.sin((np.sin((((data["medianbatch_slices2_msignal"]) - (data["signal_shift_-1"]))))))) * 2.0))), ((((data["signal_shift_-1"]) / 2.0))))))) +

                            0.050000*np.tanh(((((data["medianbatch_slices2_msignal"]) * (((data["maxtominbatch"]) * (((data["abs_minbatch_slices2_msignal"]) * (data["medianbatch_slices2"]))))))) * (np.cos((data["abs_minbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(((np.sin((np.where(np.minimum(((np.maximum(((np.sin((np.minimum(((np.sin((data["abs_maxbatch_msignal"])))), ((data["minbatch_msignal"]))))))), (((((((data["medianbatch_slices2_msignal"]) * 2.0)) + (data["signal_shift_-1"]))/2.0)))))), ((data["meanbatch_slices2"]))) > -998, data["minbatch_msignal"], data["meanbatch_slices2_msignal"] )))) * (((data["meanbatch_slices2_msignal"]) * 2.0)))) +

                            0.050000*np.tanh(((np.sin(((10.50283241271972656)))) + (((np.cos((data["mean_abs_chgbatch_msignal"]))) * (((((data["mean_abs_chgbatch_slices2"]) + (data["mean_abs_chgbatch_msignal"]))) + (((data["abs_minbatch_msignal"]) + (data["meanbatch_msignal"]))))))))) +

                            0.050000*np.tanh(((np.cos((np.where(((data["minbatch_msignal"]) * 2.0) > -998, data["minbatch_msignal"], np.where(data["minbatch_msignal"] > -998, np.sin((data["signal_shift_-1_msignal"])), data["minbatch_msignal"] ) )))) * 2.0)) +

                            0.050000*np.tanh(((data["medianbatch_slices2_msignal"]) + (((((data["abs_maxbatch"]) + (data["medianbatch_slices2_msignal"]))) * (np.sin((np.where(((data["stdbatch_slices2_msignal"]) + (np.sin((np.sin((np.where(data["maxbatch_slices2"] > -998, data["abs_maxbatch_msignal"], ((data["maxtominbatch"]) * 2.0) ))))))) > -998, data["abs_maxbatch_msignal"], ((data["meanbatch_msignal"]) + (data["minbatch_slices2_msignal"])) )))))))) +

                            0.050000*np.tanh(((data["stdbatch_msignal"]) * (np.sin((data["abs_maxbatch"]))))) +

                            0.050000*np.tanh(((data["signal"]) * (((data["mean_abs_chgbatch_msignal"]) * ((((data["abs_maxbatch_slices2"]) + (np.minimum(((np.minimum(((data["medianbatch_msignal"])), ((((data["mean_abs_chgbatch_msignal"]) * (data["meanbatch_msignal"]))))))), ((data["rangebatch_slices2"])))))/2.0)))))) +

                            0.050000*np.tanh(((np.where(data["medianbatch_msignal"] <= -998, np.maximum(((data["meanbatch_slices2_msignal"])), ((np.where(np.maximum(((data["abs_minbatch_slices2_msignal"])), ((((data["stdbatch_slices2_msignal"]) * 2.0)))) <= -998, np.maximum(((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0))), ((data["meanbatch_msignal"]))), data["stdbatch_msignal"] )))), np.maximum(((data["stdbatch_msignal"])), ((data["meanbatch_msignal"]))) )) * (np.sin((data["stdbatch_msignal"]))))) +

                            0.050000*np.tanh((-((np.cos((np.where(data["abs_minbatch_slices2_msignal"] > -998, data["rangebatch_slices2_msignal"], np.cos((data["meanbatch_slices2_msignal"])) ))))))) +

                            0.050000*np.tanh(np.sin((np.where((-((np.where(data["abs_minbatch_msignal"] > -998, np.minimum(((data["medianbatch_slices2_msignal"])), ((data["minbatch"]))), data["abs_avgbatch_slices2_msignal"] )))) > -998, ((np.maximum(((data["medianbatch_slices2_msignal"])), ((data["medianbatch_slices2_msignal"])))) - (data["signal_shift_+1"])), ((data["abs_avgbatch_msignal"]) * 2.0) )))) +

                            0.050000*np.tanh(((data["stdbatch_msignal"]) * (np.where(((data["signal"]) * (((data["stdbatch_msignal"]) / 2.0))) <= -998, data["minbatch_slices2"], data["signal"] )))) +

                            0.050000*np.tanh(((np.cos((((data["abs_maxbatch"]) - (np.where(np.where(((data["minbatch_slices2"]) - ((5.48352003097534180))) > -998, data["maxbatch_slices2"], (((data["medianbatch_slices2_msignal"]) <= (data["maxbatch_slices2"]))*1.) ) > -998, data["medianbatch_slices2_msignal"], ((data["medianbatch_slices2_msignal"]) * 2.0) )))))) * 2.0)) +

                            0.050000*np.tanh(((((data["mean_abs_chgbatch_slices2"]) - (((((data["mean_abs_chgbatch_slices2"]) - (np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, (((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)) <= (data["signal_shift_-1_msignal"]))*1.), data["mean_abs_chgbatch_slices2_msignal"] )))) * (((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)))))) * (((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)))) +

                            0.050000*np.tanh(((((np.minimum(((data["meanbatch_slices2"])), ((np.minimum(((data["abs_maxbatch"])), ((np.sin((np.where(((((data["maxbatch_slices2_msignal"]) * 2.0)) + ((-((data["signal_shift_-1_msignal"]))))) <= -998, data["medianbatch_slices2"], ((((data["maxbatch_slices2_msignal"]) * 2.0)) + ((-((data["signal_shift_-1_msignal"]))))) )))))))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((data["meanbatch_slices2_msignal"]) * (((np.minimum(((np.sin((np.sin((data["minbatch_msignal"])))))), ((np.sin((np.sin((data["minbatch_msignal"])))))))) * ((((11.18099498748779297)) - (((np.minimum((((11.18099498748779297))), ((np.sin((data["minbatch_msignal"])))))) * (np.minimum(((data["mean_abs_chgbatch_msignal"])), (((11.18099498748779297))))))))))))) +

                            0.050000*np.tanh(((((np.where(np.sin((((data["abs_avgbatch_msignal"]) - (data["signal_shift_-1"])))) <= -998, data["stdbatch_msignal"], np.sin((((data["abs_avgbatch_msignal"]) - (data["signal_shift_-1"])))) )) * 2.0)) * 2.0)) +

                            0.050000*np.tanh((-((np.where(data["abs_minbatch_slices2_msignal"] <= -998, ((data["signal_shift_-1"]) + (np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, data["stdbatch_slices2"], data["abs_minbatch_msignal"] ))), ((((np.sin((data["signal_shift_-1"]))) * 2.0)) * (((data["minbatch_slices2"]) * (data["stdbatch_slices2_msignal"])))) ))))) +

                            0.050000*np.tanh(np.where(data["stdbatch_msignal"] <= -998, ((((np.where(data["abs_maxbatch"] <= -998, data["medianbatch_slices2_msignal"], data["maxbatch_msignal"] )) * 2.0)) * 2.0), ((np.minimum(((((data["mean_abs_chgbatch_msignal"]) * 2.0))), ((((data["stdbatch_msignal"]) * (np.where(data["signal_shift_-1"] > -998, data["abs_avgbatch_slices2_msignal"], ((data["maxbatch_slices2_msignal"]) * (((data["meanbatch_slices2_msignal"]) * 2.0))) ))))))) * 2.0) )) +

                            0.050000*np.tanh(((((data["medianbatch_msignal"]) + (np.where(data["maxtominbatch"] > -998, ((np.cos((((data["signal"]) * 2.0)))) * (data["signal_shift_+1"])), ((np.cos((((data["signal"]) * 2.0)))) * (data["rangebatch_slices2_msignal"])) )))) * 2.0)) +

                            0.050000*np.tanh(((np.minimum(((data["meanbatch_slices2"])), ((np.cos((np.where(np.sin((data["abs_maxbatch_msignal"])) > -998, np.maximum(((data["abs_minbatch_msignal"])), ((((data["signal_shift_-1_msignal"]) - (data["abs_maxbatch_slices2_msignal"]))))), np.where(data["abs_maxbatch_msignal"] <= -998, (((data["abs_maxbatch_msignal"]) + (data["abs_maxbatch_msignal"]))/2.0), np.sin((data["abs_maxbatch_msignal"])) ) ))))))) * 2.0)) +

                            0.050000*np.tanh(np.where(np.minimum(((data["signal_shift_+1"])), ((((np.tanh((data["meanbatch_slices2_msignal"]))) * 2.0)))) <= -998, (-((data["rangebatch_msignal"]))), ((data["meanbatch_slices2_msignal"]) * (np.cos((((data["signal_shift_+1"]) - (data["abs_maxbatch_slices2_msignal"])))))) )) +

                            0.050000*np.tanh(np.where(np.tanh((np.sin(((-(((((data["rangebatch_slices2"]) + ((((np.sin(((-((data["signal_shift_-1"])))))) + (np.cos((data["minbatch_msignal"]))))/2.0)))/2.0)))))))) > -998, ((np.cos((data["minbatch_msignal"]))) * 2.0), (((data["signal_shift_-1"]) + (data["stdbatch_slices2_msignal"]))/2.0) )) +

                            0.050000*np.tanh(np.sin((((data["abs_avgbatch_slices2_msignal"]) - (np.maximum(((data["signal_shift_-1"])), ((np.sin((np.minimum(((np.maximum(((((data["abs_avgbatch_slices2_msignal"]) - (np.maximum(((data["signal_shift_-1"])), (((((data["signal_shift_-1"]) <= (data["signal_shift_-1"]))*1.)))))))), ((data["signal_shift_-1"]))))), ((data["signal"]))))))))))))) +

                            0.050000*np.tanh((-((np.where(data["stdbatch_slices2_msignal"] > -998, np.cos((data["rangebatch_slices2_msignal"])), (((data["minbatch_msignal"]) > (np.maximum(((data["minbatch_msignal"])), ((data["maxbatch_slices2"])))))*1.) ))))) +

                            0.050000*np.tanh((((np.where((-((np.cos((np.maximum(((np.cos((((data["maxtominbatch_slices2"]) * 2.0))))), ((np.where(data["abs_avgbatch_msignal"] > -998, data["meanbatch_msignal"], data["meanbatch_msignal"] ))))))))) > -998, ((data["medianbatch_slices2_msignal"]) / 2.0), ((np.where(data["meanbatch_msignal"] > -998, data["medianbatch_slices2_msignal"], data["meanbatch_msignal"] )) / 2.0) )) + (data["medianbatch_msignal"]))/2.0)) +

                            0.050000*np.tanh(np.cos(((((data["maxbatch_msignal"]) + (((np.cos((np.maximum(((data["maxtominbatch_slices2_msignal"])), ((data["rangebatch_msignal"])))))) - (data["minbatch_slices2"]))))/2.0)))) +

                            0.050000*np.tanh(((data["stdbatch_slices2_msignal"]) * ((-((np.where(np.minimum(((data["rangebatch_slices2"])), ((data["signal_shift_+1_msignal"]))) <= -998, data["abs_minbatch_msignal"], np.minimum(((np.sin((data["rangebatch_slices2"])))), (((-(((-(((((14.85157394409179688)) - (data["signal_shift_-1"]))))))))))) ))))))) +

                            0.050000*np.tanh((((-((data["mean_abs_chgbatch_msignal"])))) * (np.sin(((-((((data["minbatch"]) - ((-((data["mean_abs_chgbatch_msignal"]))))))))))))) +

                            0.050000*np.tanh(np.where(np.tanh((data["medianbatch_slices2_msignal"])) > -998, np.where(((data["medianbatch_msignal"]) * 2.0) > -998, np.where(data["meanbatch_slices2_msignal"] > -998, data["medianbatch_slices2"], data["meanbatch_slices2_msignal"] ), np.cos((((data["medianbatch_slices2_msignal"]) - (np.where(((data["medianbatch_msignal"]) / 2.0) > -998, data["medianbatch_slices2"], data["medianbatch_slices2_msignal"] ))))) ), data["meanbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(((data["stdbatch_msignal"]) * (((np.maximum(((np.maximum(((data["rangebatch_slices2"])), ((((((np.sin((data["abs_maxbatch"]))) * (data["stdbatch_msignal"]))) * 2.0)))))), ((np.sin(((-((data["rangebatch_slices2"]))))))))) * (np.sin((((np.sin(((-((data["rangebatch_slices2"])))))) * 2.0)))))))) +

                            0.050000*np.tanh(((data["maxbatch_slices2"]) * (np.where(data["maxbatch_slices2_msignal"] > -998, ((((((data["meanbatch_msignal"]) + (data["maxtominbatch_slices2"]))) * (data["medianbatch_slices2_msignal"]))) * (data["signal_shift_-1_msignal"])), ((((((((((data["medianbatch_slices2_msignal"]) + (data["maxtominbatch_slices2"]))) * (data["medianbatch_slices2_msignal"]))) + (data["maxtominbatch_slices2"]))) * (data["medianbatch_slices2_msignal"]))) * (data["signal_shift_-1_msignal"])) )))) +

                            0.050000*np.tanh(((((np.sin((((data["medianbatch_slices2_msignal"]) - (np.maximum(((data["signal_shift_-1"])), (((((np.where((3.0) <= -998, data["abs_avgbatch_slices2"], data["medianbatch_msignal"] )) + (data["signal_shift_-1"]))/2.0))))))))) * ((13.39701461791992188)))) + (data["medianbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((((data["signal"]) * (data["stdbatch_msignal"]))) * (np.where(data["signal_shift_+1"] > -998, ((((data["abs_maxbatch_slices2_msignal"]) - (data["medianbatch_slices2_msignal"]))) - (data["signal_shift_+1_msignal"])), np.cos((((data["signal_shift_+1_msignal"]) * (data["stdbatch_msignal"])))) )))) +

                            0.050000*np.tanh(np.where(np.where(data["minbatch_msignal"] > -998, data["abs_avgbatch_slices2"], np.sin((np.where(((data["minbatch_msignal"]) * 2.0) > -998, np.minimum(((data["signal"])), ((data["rangebatch_slices2"]))), ((np.cos((np.maximum(((data["rangebatch_slices2"])), ((data["meanbatch_slices2"])))))) * 2.0) ))) ) > -998, ((data["stdbatch_slices2_msignal"]) * (np.sin((np.sin((data["abs_maxbatch"])))))), ((((data["rangebatch_slices2"]) * 2.0)) * 2.0) )) +

                            0.050000*np.tanh(((np.cos((np.minimum(((np.minimum(((np.cos((data["signal_shift_+1_msignal"])))), ((np.minimum(((data["minbatch_msignal"])), ((np.cos((data["maxbatch_slices2_msignal"])))))))))), ((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((data["signal_shift_-1"]))))))))) * 2.0)) +

                            0.050000*np.tanh(((data["minbatch_slices2_msignal"]) * (np.where(np.tanh((data["maxtominbatch_slices2"])) <= -998, data["stdbatch_slices2_msignal"], np.cos((np.where(((data["maxtominbatch_slices2"]) * (np.where(data["minbatch_slices2_msignal"] <= -998, data["maxtominbatch_slices2"], np.cos((data["maxtominbatch_msignal"])) ))) <= -998, data["mean_abs_chgbatch_slices2"], data["medianbatch_slices2"] ))) )))) +

                            0.050000*np.tanh(np.sin((((data["maxbatch_slices2"]) + (np.where(((((data["signal_shift_+1"]) + (np.sin((data["stdbatch_slices2_msignal"]))))) + ((9.0))) > -998, data["signal_shift_+1"], data["abs_minbatch_slices2_msignal"] )))))) +

                            0.050000*np.tanh(np.minimum(((np.where(np.where(data["medianbatch_msignal"] > -998, data["meanbatch_msignal"], data["stdbatch_msignal"] ) <= -998, ((np.maximum(((data["minbatch"])), ((data["stdbatch_msignal"])))) / 2.0), data["medianbatch_slices2_msignal"] ))), ((np.minimum(((data["meanbatch_msignal"])), ((np.sin((np.where(data["stdbatch_slices2_msignal"] <= -998, (6.52855157852172852), data["abs_maxbatch_msignal"] )))))))))) +

                            0.050000*np.tanh(((np.where(np.sin(((((data["signal_shift_+1"]) + (data["minbatch_msignal"]))/2.0))) > -998, np.cos(((((data["minbatch_msignal"]) + (data["minbatch_msignal"]))/2.0))), data["minbatch_msignal"] )) * 2.0)) +

                            0.050000*np.tanh(((np.where(data["signal"] > -998, data["meanbatch_msignal"], data["maxbatch_slices2"] )) - (((data["abs_minbatch_msignal"]) * (np.maximum(((((data["meanbatch_msignal"]) * 2.0))), ((((data["maxtominbatch_slices2"]) * (((data["abs_minbatch_msignal"]) * (np.maximum(((((data["signal_shift_+1_msignal"]) * 2.0))), ((data["mean_abs_chgbatch_slices2_msignal"]))))))))))))))) +

                            0.050000*np.tanh(np.where(((data["abs_maxbatch_slices2_msignal"]) / 2.0) <= -998, data["medianbatch_slices2"], data["medianbatch_slices2"] )) +

                            0.050000*np.tanh(np.minimum(((((np.minimum(((data["medianbatch_slices2_msignal"])), (((((((4.0)) - ((4.0)))) - ((-((np.minimum(((data["stdbatch_msignal"])), ((data["medianbatch_msignal"])))))))))))) * 2.0))), ((((data["medianbatch_slices2_msignal"]) * 2.0))))) +

                            0.050000*np.tanh(np.where(np.tanh((data["signal_shift_+1"])) <= -998, ((data["signal_shift_+1"]) + (data["signal_shift_+1"])), data["signal_shift_-1"] )) +

                            0.050000*np.tanh(np.cos((np.where((((((data["medianbatch_slices2"]) + (np.cos((data["medianbatch_slices2"]))))) + (np.where(((data["abs_avgbatch_slices2"]) + (data["meanbatch_msignal"])) <= -998, data["signal"], data["abs_avgbatch_slices2"] )))/2.0) > -998, ((np.sin((data["abs_maxbatch_slices2"]))) - (data["minbatch_msignal"])), data["abs_maxbatch_slices2"] )))) +

                            0.050000*np.tanh(np.minimum(((((np.minimum(((((np.minimum(((((data["signal_shift_-1"]) * 2.0))), ((data["minbatch_msignal"])))) * (np.cos((data["meanbatch_slices2"])))))), ((((((data["signal_shift_-1"]) * (data["meanbatch_slices2"]))) * (np.sin((data["abs_maxbatch_msignal"])))))))) + (data["meanbatch_slices2"])))), ((((data["rangebatch_slices2_msignal"]) * (np.sin((np.sin((data["abs_maxbatch_msignal"])))))))))) +

                            0.050000*np.tanh(np.minimum(((np.minimum((((((((((np.cos((np.where(np.minimum(((data["medianbatch_slices2"])), ((data["signal_shift_+1"]))) > -998, np.sin((data["maxtominbatch_slices2_msignal"])), data["medianbatch_slices2"] )))) > (data["medianbatch_slices2"]))*1.)) * (data["minbatch"]))) / 2.0))), ((data["stdbatch_slices2"]))))), (((2.77612638473510742))))) +

                            0.050000*np.tanh(((data["signal_shift_-1_msignal"]) + (np.where(data["minbatch_msignal"] > -998, ((np.cos((((data["signal_shift_-1"]) * 2.0)))) * (data["abs_maxbatch_slices2_msignal"])), data["meanbatch_msignal"] )))) +

                            0.050000*np.tanh(((((np.sin((((data["medianbatch_slices2"]) - (data["abs_maxbatch_msignal"]))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.minimum((((((((((((data["signal_shift_-1"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)) * 2.0)) * (data["signal_shift_-1"]))) / 2.0))), ((np.sin((((((data["mean_abs_chgbatch_slices2_msignal"]) + ((((np.where(np.maximum(((data["mean_abs_chgbatch_slices2_msignal"])), ((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))))) > -998, data["signal_shift_-1"], data["mean_abs_chgbatch_slices2_msignal"] )) <= (np.cos((data["rangebatch_slices2"]))))*1.)))) * 2.0))))))) +

                            0.050000*np.tanh(np.where((4.46286773681640625) <= -998, data["abs_minbatch_slices2"], np.sin((((data["maxbatch_msignal"]) + (np.tanh(((((((data["medianbatch_msignal"]) * (np.sin(((((data["maxbatch_msignal"]) + (data["medianbatch_slices2_msignal"]))/2.0)))))) <= ((((4.46286773681640625)) + ((((data["meanbatch_slices2_msignal"]) <= ((((data["meanbatch_slices2_msignal"]) <= (((data["abs_minbatch_slices2"]) * (data["signal_shift_+1"]))))*1.)))*1.)))))*1.))))))) )) +

                            0.050000*np.tanh(((data["abs_minbatch_msignal"]) * (np.minimum(((data["abs_avgbatch_slices2"])), ((np.cos((np.where(((((data["medianbatch_msignal"]) + (((np.tanh((data["meanbatch_slices2_msignal"]))) + (np.maximum(((data["stdbatch_slices2"])), ((data["abs_minbatch_msignal"])))))))) * 2.0) <= -998, data["abs_minbatch_slices2"], np.maximum(((data["medianbatch_slices2"])), ((((data["signal_shift_-1"]) + (data["meanbatch_slices2_msignal"]))))) ))))))))) +

                            0.050000*np.tanh(((data["medianbatch_slices2_msignal"]) + (np.minimum(((np.sin((data["abs_maxbatch_msignal"])))), ((np.minimum(((np.maximum((((((((np.sin((data["abs_maxbatch_msignal"]))) / 2.0)) <= (data["abs_minbatch_slices2"]))*1.))), ((np.cos((np.maximum(((np.tanh((np.tanh((np.maximum(((data["stdbatch_slices2_msignal"])), ((data["stdbatch_slices2_msignal"]))))))))), ((np.tanh((data["rangebatch_msignal"])))))))))))), ((np.sin((data["signal_shift_+1_msignal"]))))))))))) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) * (np.sin((((np.maximum(((data["meanbatch_slices2"])), ((np.maximum(((np.sin((((np.maximum(((data["meanbatch_slices2"])), ((np.cos((np.sin((((data["meanbatch_slices2"]) * (data["meanbatch_slices2"])))))))))) * 2.0))))), ((data["meanbatch_slices2_msignal"]))))))) * 2.0)))))) +

                            0.050000*np.tanh(((np.where(data["signal_shift_-1"] <= -998, np.cos((np.tanh((((data["stdbatch_slices2_msignal"]) * 2.0))))), ((np.cos((((data["mean_abs_chgbatch_msignal"]) - (data["signal_shift_-1"]))))) * 2.0) )) * (data["abs_maxbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(np.sin((np.maximum(((((data["rangebatch_msignal"]) - (data["abs_avgbatch_slices2"])))), ((np.maximum(((((data["rangebatch_msignal"]) - (data["abs_avgbatch_slices2"])))), ((data["abs_avgbatch_slices2"]))))))))) +

                            0.050000*np.tanh(np.where(data["stdbatch_slices2"] <= -998, ((((np.minimum(((data["maxtominbatch_msignal"])), ((((np.maximum(((data["maxtominbatch"])), ((data["medianbatch_msignal"])))) / 2.0))))) + (data["medianbatch_slices2"]))) + (data["mean_abs_chgbatch_msignal"])), data["medianbatch_msignal"] )) +

                            0.050000*np.tanh(np.where(np.maximum(((np.cos((data["signal_shift_+1_msignal"])))), ((np.where(np.maximum(((data["signal_shift_-1_msignal"])), (((((((data["abs_maxbatch_slices2"]) * (data["meanbatch_slices2_msignal"]))) <= (data["maxtominbatch_slices2"]))*1.)))) <= -998, np.cos((data["minbatch_msignal"])), np.tanh((data["signal_shift_-1_msignal"])) )))) <= -998, np.cos((data["minbatch_msignal"])), np.tanh((((np.cos((data["minbatch_msignal"]))) * 2.0))) )) +

                            0.050000*np.tanh(np.sin((np.where(data["signal_shift_+1"] <= -998, np.cos((data["meanbatch_slices2"])), np.minimum(((data["signal_shift_+1"])), ((np.maximum(((data["meanbatch_slices2"])), ((data["medianbatch_slices2_msignal"])))))) )))) +

                            0.050000*np.tanh(np.sin((((data["medianbatch_slices2"]) - (np.where(((data["meanbatch_slices2"]) * 2.0) > -998, data["maxtominbatch_slices2_msignal"], np.cos((np.where(data["rangebatch_msignal"] > -998, np.where(data["meanbatch_slices2_msignal"] > -998, data["medianbatch_slices2"], np.maximum(((data["signal_shift_-1_msignal"])), ((data["signal_shift_-1"]))) ), np.sin((data["minbatch"])) ))) )))))) +

                            0.050000*np.tanh(((np.where(((data["medianbatch_slices2"]) / 2.0) > -998, np.where(data["meanbatch_slices2"] <= -998, data["abs_avgbatch_msignal"], ((data["stdbatch_msignal"]) * (np.sin((np.minimum(((data["maxtominbatch_slices2"])), ((data["meanbatch_msignal"]))))))) ), data["maxtominbatch_slices2_msignal"] )) + (data["meanbatch_msignal"]))) +

                            0.050000*np.tanh(((data["abs_minbatch_slices2"]) + ((((data["meanbatch_slices2_msignal"]) + (((data["signal_shift_+1_msignal"]) + (np.tanh((data["mean_abs_chgbatch_msignal"]))))))/2.0)))) +

                            0.050000*np.tanh(np.where(np.where((((data["minbatch_msignal"]) <= (((((((((data["minbatch_msignal"]) <= (data["minbatch_msignal"]))*1.)) / 2.0)) <= (data["abs_maxbatch_slices2_msignal"]))*1.)))*1.) > -998, data["abs_maxbatch_slices2_msignal"], data["signal"] ) <= -998, data["meanbatch_slices2"], ((data["meanbatch_slices2"]) * (((data["maxbatch_slices2"]) * (np.cos((((data["abs_maxbatch_slices2_msignal"]) - ((((((data["signal_shift_-1"]) <= (data["maxbatch_slices2"]))*1.)) / 2.0))))))))) )) +

                            0.050000*np.tanh(((((data["abs_avgbatch_slices2_msignal"]) * (data["meanbatch_slices2_msignal"]))) * (np.where(np.tanh((data["signal_shift_-1"])) > -998, data["signal_shift_-1"], ((np.maximum(((data["meanbatch_slices2"])), ((data["minbatch"])))) / 2.0) )))) +

                            0.050000*np.tanh(np.maximum(((data["signal_shift_+1_msignal"])), ((np.minimum(((np.sin((((data["medianbatch_slices2_msignal"]) + (data["medianbatch_slices2_msignal"])))))), ((data["medianbatch_slices2_msignal"]))))))) +

                            0.050000*np.tanh(data["medianbatch_slices2_msignal"]) +

                            0.050000*np.tanh(((data["abs_maxbatch_slices2"]) * (np.tanh((np.sin((np.where(data["medianbatch_slices2"] <= -998, np.maximum((((((((data["maxtominbatch"]) <= (((data["meanbatch_slices2"]) * 2.0)))*1.)) / 2.0))), ((data["abs_minbatch_msignal"]))), ((((data["maxbatch_msignal"]) * 2.0)) * 2.0) )))))))) +

                            0.050000*np.tanh(((np.cos((((np.cos((data["signal_shift_-1"]))) + (((data["medianbatch_msignal"]) - (data["abs_maxbatch"]))))))) * 2.0)) +

                            0.050000*np.tanh(np.sin((((np.where(np.where(data["abs_maxbatch"] > -998, ((data["signal"]) + (data["medianbatch_slices2"])), data["medianbatch_slices2"] ) <= -998, data["rangebatch_slices2_msignal"], np.where(data["rangebatch_msignal"] <= -998, ((data["minbatch_slices2"]) - (data["abs_maxbatch_msignal"])), data["rangebatch_msignal"] ) )) - (np.maximum(((data["signal"])), ((data["meanbatch_msignal"])))))))) +

                            0.050000*np.tanh(((data["medianbatch_slices2"]) + (np.minimum(((np.where(data["maxtominbatch_slices2_msignal"] <= -998, np.tanh((np.sin((data["maxbatch_msignal"])))), data["medianbatch_slices2"] ))), ((np.maximum(((data["maxtominbatch_msignal"])), ((np.maximum(((data["maxtominbatch_msignal"])), ((data["maxtominbatch"])))))))))))) +

                            0.050000*np.tanh(np.cos((np.maximum(((((data["signal_shift_+1"]) - (np.tanh(((((((np.cos((data["signal_shift_+1"]))) / 2.0)) + (((data["abs_minbatch_slices2"]) * 2.0)))/2.0))))))), ((data["rangebatch_slices2"])))))) +

                            0.050000*np.tanh(data["medianbatch_slices2"]) +

                            0.050000*np.tanh(((data["medianbatch_msignal"]) - (np.where(np.sin(((12.89390754699707031))) > -998, data["signal_shift_+1"], data["maxtominbatch"] )))) +

                            0.050000*np.tanh(((((((((data["signal_shift_+1"]) * 2.0)) * (np.sin((np.sin((np.where(np.minimum(((data["medianbatch_msignal"])), (((((((6.0)) * (np.sin((((data["signal_shift_+1"]) * 2.0)))))) * ((-((data["signal_shift_-1_msignal"])))))))) > -998, data["medianbatch_msignal"], data["stdbatch_slices2"] )))))))) * ((-((data["signal_shift_-1_msignal"])))))) * 2.0)) +

                            0.050000*np.tanh(((data["medianbatch_msignal"]) - (np.where((((np.tanh((np.tanh((np.sin((data["medianbatch_msignal"]))))))) <= (np.where(data["abs_avgbatch_slices2_msignal"] <= -998, data["maxtominbatch"], data["stdbatch_slices2_msignal"] )))*1.) <= -998, ((((data["abs_maxbatch_msignal"]) * (data["medianbatch_msignal"]))) * (np.where(data["signal_shift_-1"] <= -998, (-((data["medianbatch_slices2"]))), data["abs_avgbatch_slices2_msignal"] ))), data["stdbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) * (((data["rangebatch_msignal"]) * (np.sin((data["abs_maxbatch_msignal"]))))))) +

                            0.050000*np.tanh(np.where(data["maxtominbatch_slices2"] > -998, np.where(data["minbatch"] > -998, np.minimum(((data["signal_shift_-1"])), ((np.where(data["stdbatch_msignal"] > -998, data["stdbatch_msignal"], data["medianbatch_slices2_msignal"] )))), (((data["medianbatch_msignal"]) <= (data["stdbatch_msignal"]))*1.) ), ((data["abs_minbatch_msignal"]) / 2.0) )) +

                            0.050000*np.tanh(((data["meanbatch_slices2"]) - ((((np.where((((data["meanbatch_slices2"]) > (((data["signal_shift_+1"]) + (data["meanbatch_slices2"]))))*1.) > -998, data["abs_maxbatch_msignal"], ((data["signal_shift_+1"]) + ((-((np.tanh((data["meanbatch_slices2"]))))))) )) > (((data["medianbatch_msignal"]) + (data["signal_shift_-1"]))))*1.)))) +

                            0.050000*np.tanh(np.cos((np.minimum(((np.minimum(((data["minbatch_msignal"])), ((np.maximum(((data["meanbatch_slices2_msignal"])), ((np.cos((np.minimum(((data["minbatch_msignal"])), ((np.maximum(((np.sin((np.cos((data["minbatch"])))))), ((data["mean_abs_chgbatch_msignal"])))))))))))))))), ((np.cos((np.minimum(((np.cos((data["maxbatch_slices2"])))), ((data["meanbatch_slices2_msignal"]))))))))))) +

                            0.050000*np.tanh(np.where((((data["meanbatch_slices2_msignal"]) > (((np.maximum(((data["meanbatch_slices2_msignal"])), ((data["meanbatch_slices2_msignal"])))) - (data["meanbatch_slices2"]))))*1.) <= -998, (((data["medianbatch_slices2_msignal"]) <= ((((data["signal_shift_+1_msignal"]) <= (data["meanbatch_slices2"]))*1.)))*1.), np.maximum(((data["signal_shift_-1_msignal"])), ((data["meanbatch_slices2_msignal"]))) )) +

                            0.050000*np.tanh((((data["signal_shift_+1"]) + (np.maximum(((np.where(data["abs_maxbatch_slices2_msignal"] > -998, np.sin(((((((((np.maximum(((data["mean_abs_chgbatch_msignal"])), ((data["abs_maxbatch_slices2_msignal"])))) > (data["abs_avgbatch_slices2_msignal"]))*1.)) + (data["abs_avgbatch_msignal"]))) / 2.0))), data["minbatch_msignal"] ))), ((data["meanbatch_slices2_msignal"])))))/2.0)) +

                            0.050000*np.tanh(np.minimum(((data["maxbatch_slices2_msignal"])), ((data["medianbatch_slices2_msignal"])))) +

                            0.050000*np.tanh((((-((((((data["abs_maxbatch"]) * (np.sin((np.where(data["medianbatch_msignal"] > -998, data["medianbatch_msignal"], (-((np.where(((((data["abs_maxbatch"]) * ((1.0)))) * 2.0) > -998, data["medianbatch_msignal"], data["medianbatch_msignal"] )))) )))))) * 2.0))))) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((data["medianbatch_slices2_msignal"])), (((((np.where(data["signal_shift_-1_msignal"] > -998, data["mean_abs_chgbatch_slices2"], data["meanbatch_slices2"] )) > (data["maxtominbatch"]))*1.))))))  

    

    def GP_class_6(self,data):

        return self.Output( -3.281070 +

                            0.050000*np.tanh(((((np.where(data["meanbatch_msignal"] > -998, data["maxbatch_msignal"], np.sin((np.maximum(((data["signal_shift_-1_msignal"])), ((data["meanbatch_slices2"]))))) )) * (data["meanbatch_slices2"]))) - (((data["abs_maxbatch"]) - (data["minbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(np.minimum(((np.minimum(((data["abs_maxbatch"])), ((((data["stdbatch_slices2"]) + (((((data["mean_abs_chgbatch_slices2"]) * 2.0)) / 2.0)))))))), ((np.maximum(((data["minbatch_slices2_msignal"])), ((data["signal"]))))))) +

                            0.050000*np.tanh(data["meanbatch_slices2"]) +

                            0.050000*np.tanh(np.where((((data["meanbatch_slices2_msignal"]) <= (data["rangebatch_msignal"]))*1.) <= -998, (((data["stdbatch_slices2_msignal"]) + ((((data["rangebatch_slices2"]) <= ((((((data["meanbatch_slices2_msignal"]) - (data["meanbatch_msignal"]))) + ((((data["maxtominbatch_slices2_msignal"]) > (((data["abs_avgbatch_msignal"]) + (np.minimum(((data["meanbatch_slices2"])), ((data["meanbatch_slices2_msignal"])))))))*1.)))/2.0)))*1.)))/2.0), data["meanbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(np.where(data["abs_minbatch_msignal"] > -998, data["signal"], np.where(data["medianbatch_msignal"] > -998, data["maxbatch_msignal"], np.where(data["stdbatch_slices2"] > -998, np.maximum(((np.where(((data["meanbatch_slices2"]) * 2.0) > -998, data["signal_shift_+1"], data["meanbatch_slices2"] ))), ((data["medianbatch_msignal"]))), np.tanh((((data["meanbatch_slices2"]) / 2.0))) ) ) )) +

                            0.050000*np.tanh(data["signal_shift_+1"]) +

                            0.050000*np.tanh(np.tanh((np.where(data["meanbatch_slices2"] <= -998, data["abs_avgbatch_slices2"], (((np.sin((data["stdbatch_slices2"]))) + (data["signal"]))/2.0) )))) +

                            0.050000*np.tanh(((np.where(data["minbatch_slices2"] <= -998, np.where(((((data["meanbatch_msignal"]) / 2.0)) / 2.0) > -998, data["meanbatch_msignal"], data["medianbatch_slices2"] ), data["minbatch_slices2"] )) + (data["maxbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(np.where(data["meanbatch_slices2_msignal"] > -998, data["meanbatch_slices2_msignal"], np.where(data["abs_minbatch_slices2"] > -998, ((((data["maxbatch_slices2"]) + (data["meanbatch_slices2_msignal"]))) * (np.maximum(((((((data["maxbatch_slices2"]) + (data["signal_shift_+1_msignal"]))) * 2.0))), ((np.maximum((((((9.0)) * 2.0))), ((data["meanbatch_slices2_msignal"])))))))), data["signal_shift_-1"] ) )) +

                            0.050000*np.tanh(np.where(data["stdbatch_slices2_msignal"] > -998, data["meanbatch_slices2_msignal"], np.tanh((np.sin((((np.where(data["stdbatch_slices2_msignal"] > -998, data["medianbatch_slices2"], np.where(data["meanbatch_slices2"] > -998, ((data["medianbatch_slices2"]) * 2.0), (-((data["abs_avgbatch_msignal"]))) ) )) + (data["meanbatch_slices2_msignal"])))))) )) +

                            0.050000*np.tanh(data["meanbatch_slices2_msignal"]) +

                            0.050000*np.tanh((((((data["medianbatch_slices2_msignal"]) + ((((((12.08343982696533203)) * (np.tanh((data["meanbatch_msignal"]))))) * (((((((data["signal"]) / 2.0)) * 2.0)) * 2.0)))))) + (((data["minbatch_slices2_msignal"]) * 2.0)))/2.0)) +

                            0.050000*np.tanh(((data["signal"]) * (np.maximum(((data["meanbatch_slices2_msignal"])), ((((((((data["medianbatch_msignal"]) / 2.0)) * 2.0)) * (np.where(((np.cos((data["maxbatch_slices2_msignal"]))) - (((((data["signal_shift_+1"]) * (data["medianbatch_slices2"]))) + (data["abs_minbatch_slices2"])))) > -998, data["abs_maxbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] ))))))))) +

                            0.050000*np.tanh(np.where(data["medianbatch_slices2_msignal"] > -998, np.minimum(((((data["signal_shift_+1"]) + (np.where(((data["medianbatch_slices2"]) + (data["maxtominbatch_slices2_msignal"])) > -998, np.minimum(((data["signal_shift_+1"])), ((data["abs_avgbatch_slices2_msignal"]))), data["maxtominbatch_slices2"] ))))), ((data["medianbatch_msignal"]))), data["medianbatch_msignal"] )) +

                            0.050000*np.tanh(np.sin((np.sin((data["meanbatch_msignal"]))))) +

                            0.050000*np.tanh(np.minimum(((data["signal_shift_-1"])), ((np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((np.where(np.minimum(((data["signal_shift_+1"])), ((data["abs_avgbatch_slices2"]))) <= -998, data["signal_shift_-1"], np.minimum(((data["meanbatch_slices2"])), ((data["medianbatch_msignal"]))) )))))))) +

                            0.050000*np.tanh(np.minimum(((np.where(data["meanbatch_msignal"] > -998, np.where(data["maxbatch_slices2_msignal"] <= -998, data["meanbatch_slices2_msignal"], data["medianbatch_slices2_msignal"] ), (((((data["signal"]) * (((((((1.0)) * ((((data["stdbatch_msignal"]) + (data["medianbatch_slices2_msignal"]))/2.0)))) + (data["medianbatch_msignal"]))/2.0)))) + (data["medianbatch_msignal"]))/2.0) ))), ((data["signal"])))) +

                            0.050000*np.tanh(np.where((((data["meanbatch_slices2"]) + (np.cos((data["signal_shift_+1"]))))/2.0) > -998, data["signal_shift_+1"], np.cos((data["rangebatch_msignal"])) )) +

                            0.050000*np.tanh(((data["medianbatch_slices2"]) - (((np.where(data["abs_maxbatch_slices2_msignal"] > -998, (((np.where(data["medianbatch_slices2"] > -998, data["meanbatch_msignal"], data["meanbatch_msignal"] )) > (data["signal_shift_-1_msignal"]))*1.), data["maxbatch_msignal"] )) * 2.0)))) +

                            0.050000*np.tanh(np.where(((((np.tanh((np.tanh((((np.where((((data["meanbatch_msignal"]) + (((data["meanbatch_slices2_msignal"]) / 2.0)))/2.0) > -998, ((data["abs_maxbatch"]) * (data["abs_maxbatch"])), np.tanh((((((data["medianbatch_slices2_msignal"]) * 2.0)) * (data["medianbatch_msignal"])))) )) * (data["rangebatch_slices2"]))))))) / 2.0)) * 2.0) > -998, data["medianbatch_msignal"], data["medianbatch_msignal"] )) +

                            0.050000*np.tanh(np.where(((data["minbatch_msignal"]) * (data["signal_shift_+1"])) > -998, data["medianbatch_slices2"], data["meanbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(((np.where(np.where(np.cos((data["maxbatch_msignal"])) > -998, data["medianbatch_msignal"], data["medianbatch_slices2_msignal"] ) > -998, data["signal"], data["mean_abs_chgbatch_slices2"] )) * (data["medianbatch_msignal"]))) +

                            0.050000*np.tanh(np.minimum(((data["medianbatch_slices2"])), ((data["signal"])))) +

                            0.050000*np.tanh(((np.where(np.maximum(((((data["signal"]) / 2.0))), ((data["signal"]))) > -998, data["meanbatch_msignal"], (((((((np.where(np.where(data["rangebatch_slices2"] > -998, (-((np.maximum(((np.sin((data["meanbatch_msignal"])))), ((data["stdbatch_slices2_msignal"])))))), ((data["meanbatch_msignal"]) * 2.0) ) > -998, data["meanbatch_slices2"], data["signal"] )) * 2.0)) + (data["meanbatch_msignal"]))/2.0)) * 2.0) )) * 2.0)) +

                            0.050000*np.tanh(((np.cos((data["maxtominbatch"]))) * ((-((np.where(data["signal"] > -998, (14.11495685577392578), (((((np.cos((np.where(((data["rangebatch_slices2"]) * ((-((((((data["minbatch_slices2"]) * ((-((data["maxbatch_slices2_msignal"])))))) + (data["signal_shift_+1_msignal"]))))))) <= -998, data["meanbatch_slices2_msignal"], data["signal_shift_+1_msignal"] )))) + (data["stdbatch_slices2"]))/2.0)) + (data["meanbatch_slices2"])) ))))))) +

                            0.050000*np.tanh(np.where(np.where(np.tanh((data["rangebatch_slices2"])) <= -998, np.tanh((data["stdbatch_slices2_msignal"])), np.tanh((np.tanh((data["signal_shift_+1"])))) ) > -998, data["meanbatch_slices2_msignal"], data["minbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(np.minimum(((((data["signal"]) * ((6.30654382705688477))))), ((((data["meanbatch_msignal"]) + (((np.tanh((data["meanbatch_slices2"]))) * (((data["medianbatch_slices2_msignal"]) / 2.0))))))))) +

                            0.050000*np.tanh(np.minimum((((9.0))), ((((data["abs_maxbatch_slices2"]) * 2.0))))) +

                            0.050000*np.tanh(((np.where(data["meanbatch_slices2_msignal"] > -998, data["meanbatch_slices2_msignal"], data["abs_avgbatch_slices2"] )) * 2.0)) +

                            0.050000*np.tanh(np.sin((np.maximum(((data["abs_maxbatch_slices2"])), ((np.minimum((((((data["maxtominbatch"]) > (np.cos((data["maxbatch_msignal"]))))*1.))), (((3.0)))))))))) +

                            0.050000*np.tanh(np.minimum(((data["mean_abs_chgbatch_slices2"])), ((data["medianbatch_msignal"])))) +

                            0.050000*np.tanh(((data["meanbatch_slices2_msignal"]) * (data["meanbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((((data["signal"]) - ((1.03515052795410156)))) * (((((data["maxtominbatch_slices2"]) + (((((np.tanh((((((data["maxbatch_slices2_msignal"]) / 2.0)) + (data["maxbatch_slices2_msignal"]))))) / 2.0)) + (((data["abs_avgbatch_slices2_msignal"]) * 2.0)))))) * 2.0)))) +

                            0.050000*np.tanh(np.where((((data["signal_shift_-1"]) > ((-((data["maxbatch_slices2"])))))*1.) <= -998, data["rangebatch_msignal"], ((np.minimum(((data["abs_maxbatch"])), ((data["maxbatch_slices2_msignal"])))) * (np.sin((np.where(np.sin((data["abs_minbatch_slices2_msignal"])) <= -998, data["maxbatch_slices2_msignal"], np.sin((data["abs_maxbatch"])) ))))) )) +

                            0.050000*np.tanh(np.where(np.where(data["meanbatch_slices2_msignal"] > -998, ((data["abs_avgbatch_slices2_msignal"]) * (data["rangebatch_slices2"])), data["signal_shift_-1"] ) > -998, np.minimum(((np.sin((data["meanbatch_msignal"])))), ((data["meanbatch_msignal"]))), data["abs_avgbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(((np.where(data["abs_avgbatch_msignal"] <= -998, data["mean_abs_chgbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] )) * 2.0)) +

                            0.050000*np.tanh(((np.where(data["meanbatch_slices2_msignal"] <= -998, np.cos((data["maxbatch_msignal"])), np.cos((data["maxbatch_msignal"])) )) * 2.0)) +

                            0.050000*np.tanh(((data["medianbatch_msignal"]) + (np.minimum(((((np.where(data["medianbatch_slices2_msignal"] > -998, np.where(data["abs_maxbatch_msignal"] <= -998, data["medianbatch_slices2"], np.sin((data["abs_maxbatch"])) ), data["stdbatch_slices2"] )) * 2.0))), ((((data["medianbatch_msignal"]) * 2.0))))))) +

                            0.050000*np.tanh(np.where(((np.maximum(((np.where(data["maxbatch_slices2"] > -998, data["meanbatch_slices2_msignal"], np.maximum(((data["meanbatch_slices2_msignal"])), ((data["meanbatch_slices2_msignal"]))) ))), ((data["meanbatch_slices2_msignal"])))) * (data["rangebatch_msignal"])) <= -998, (-((np.where(data["signal"] > -998, (8.77815341949462891), data["minbatch"] )))), data["medianbatch_msignal"] )) +

                            0.050000*np.tanh(np.cos((np.where((((np.maximum(((data["medianbatch_msignal"])), ((data["signal_shift_+1"])))) > (data["meanbatch_slices2"]))*1.) > -998, data["stdbatch_msignal"], (-((np.tanh((np.where(np.cos((np.where(data["meanbatch_slices2"] > -998, np.sin((data["abs_avgbatch_slices2"])), (-((data["stdbatch_msignal"]))) ))) > -998, data["meanbatch_msignal"], data["abs_minbatch_msignal"] )))))) )))) +

                            0.050000*np.tanh(np.maximum(((np.cos((data["maxbatch_slices2_msignal"])))), ((data["minbatch_slices2_msignal"])))) +

                            0.050000*np.tanh(((((data["medianbatch_slices2_msignal"]) * (data["signal"]))) - (np.cos((np.where(data["minbatch_slices2_msignal"] <= -998, ((data["signal"]) / 2.0), data["meanbatch_slices2_msignal"] )))))) +

                            0.050000*np.tanh(np.cos((np.maximum(((data["maxbatch_msignal"])), ((((np.sin((data["maxbatch_msignal"]))) - (data["meanbatch_msignal"])))))))) +

                            0.050000*np.tanh(data["signal_shift_-1"]) +

                            0.050000*np.tanh(data["meanbatch_msignal"]) +

                            0.050000*np.tanh(((((np.cos((data["maxbatch_msignal"]))) + (data["medianbatch_msignal"]))) - ((((np.cos((np.maximum((((-((data["medianbatch_msignal"]))))), (((-((data["medianbatch_msignal"]))))))))) > (data["maxtominbatch_slices2"]))*1.)))) +

                            0.050000*np.tanh(((np.where(np.maximum(((data["abs_minbatch_slices2_msignal"])), ((data["maxbatch_slices2_msignal"]))) <= -998, data["maxbatch_slices2_msignal"], np.cos((np.maximum(((data["meanbatch_slices2"])), ((np.maximum(((data["maxbatch_slices2_msignal"])), ((((np.tanh((data["abs_maxbatch_slices2"]))) * 2.0))))))))) )) * (np.where(data["medianbatch_slices2"] > -998, data["abs_maxbatch_slices2"], ((np.cos((np.maximum(((data["medianbatch_slices2_msignal"])), ((data["medianbatch_slices2"])))))) * 2.0) )))) +

                            0.050000*np.tanh(np.sin((np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((data["medianbatch_msignal"])))))) +

                            0.050000*np.tanh(((data["medianbatch_msignal"]) * (np.where(data["mean_abs_chgbatch_msignal"] <= -998, np.cos((data["signal"])), ((np.minimum((((-((data["mean_abs_chgbatch_slices2_msignal"]))))), ((((data["maxbatch_msignal"]) * 2.0))))) * (data["medianbatch_slices2_msignal"])) )))) +

                            0.050000*np.tanh(np.minimum(((data["meanbatch_slices2_msignal"])), ((np.where(np.where(data["maxbatch_slices2_msignal"] <= -998, np.sin((data["meanbatch_slices2_msignal"])), ((data["meanbatch_slices2_msignal"]) * (((data["medianbatch_msignal"]) * (data["meanbatch_msignal"])))) ) <= -998, data["meanbatch_slices2_msignal"], ((data["abs_maxbatch"]) * (np.sin((data["abs_avgbatch_slices2_msignal"])))) ))))) +

                            0.050000*np.tanh(((np.sin(((4.0)))) * (np.where(data["signal"] > -998, (-((data["signal"]))), np.maximum((((7.53076505661010742))), ((data["abs_minbatch_slices2_msignal"]))) )))) +

                            0.050000*np.tanh(((np.cos((data["maxbatch_msignal"]))) + (np.cos((((data["maxbatch_msignal"]) + (data["maxbatch_msignal"]))))))) +

                            0.050000*np.tanh(np.cos((np.where((-((np.tanh(((((np.tanh((np.sin((((np.maximum(((((data["medianbatch_msignal"]) * 2.0))), ((data["mean_abs_chgbatch_slices2_msignal"])))) - (data["medianbatch_msignal"]))))))) > (np.cos(((((data["rangebatch_msignal"]) + (np.cos((data["medianbatch_msignal"]))))/2.0)))))*1.)))))) > -998, data["abs_maxbatch_slices2_msignal"], data["abs_maxbatch_slices2"] )))) +

                            0.050000*np.tanh(np.minimum(((((data["meanbatch_msignal"]) * 2.0))), ((((np.minimum((((-((np.tanh((data["mean_abs_chgbatch_msignal"]))))))), ((data["signal"])))) * (data["abs_maxbatch_slices2"])))))) +

                            0.050000*np.tanh(np.minimum(((data["meanbatch_msignal"])), ((np.cos((np.maximum(((data["maxbatch_slices2_msignal"])), ((data["maxbatch_slices2_msignal"]))))))))) +

                            0.050000*np.tanh(((((np.cos((np.where(data["abs_maxbatch_msignal"] > -998, data["maxbatch_msignal"], data["meanbatch_slices2_msignal"] )))) * (np.where(data["maxbatch_msignal"] > -998, data["abs_maxbatch_msignal"], np.tanh((((np.where(data["minbatch_slices2_msignal"] > -998, data["stdbatch_slices2_msignal"], data["abs_maxbatch_slices2_msignal"] )) / 2.0))) )))) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((np.minimum((((12.71473979949951172))), ((np.where(data["medianbatch_msignal"] > -998, np.where(data["rangebatch_slices2_msignal"] > -998, np.minimum(((data["meanbatch_slices2_msignal"])), ((np.cos((data["maxbatch_msignal"]))))), (10.0) ), np.maximum(((data["signal"])), ((np.where(data["medianbatch_msignal"] <= -998, data["signal"], data["signal"] )))) )))))), ((data["signal"])))) +

                            0.050000*np.tanh(((np.where(data["meanbatch_msignal"] <= -998, data["maxbatch_slices2"], np.minimum(((((np.where(data["signal_shift_-1"] <= -998, ((np.sin((np.maximum(((np.maximum(((data["maxtominbatch_slices2_msignal"])), ((data["abs_maxbatch_slices2_msignal"]))))), ((np.cos((data["maxbatch_slices2_msignal"])))))))) * 2.0), np.minimum(((data["meanbatch_slices2_msignal"])), ((np.cos((data["maxbatch_slices2_msignal"]))))) )) * 2.0))), ((data["abs_minbatch_slices2"]))) )) * 2.0)) +

                            0.050000*np.tanh(np.minimum((((((-((data["mean_abs_chgbatch_msignal"])))) * 2.0))), ((((data["abs_avgbatch_msignal"]) + (data["abs_avgbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(((np.tanh((data["meanbatch_slices2_msignal"]))) * (np.where(data["minbatch_msignal"] > -998, ((data["abs_maxbatch_msignal"]) * (np.where(data["signal_shift_+1_msignal"] > -998, (6.65409708023071289), ((((data["signal_shift_+1_msignal"]) * 2.0)) * (data["abs_maxbatch_slices2_msignal"])) ))), np.cos((data["meanbatch_slices2_msignal"])) )))) +

                            0.050000*np.tanh(((((data["abs_maxbatch_slices2_msignal"]) * (np.cos((np.cos((np.where(np.sin((data["abs_avgbatch_slices2_msignal"])) > -998, np.cos((((data["abs_maxbatch_msignal"]) - (np.where(data["medianbatch_slices2_msignal"] > -998, data["meanbatch_msignal"], np.sin((data["maxtominbatch_msignal"])) ))))), data["meanbatch_msignal"] )))))))) * 2.0)) +

                            0.050000*np.tanh(((np.where(np.cos((data["maxbatch_msignal"])) <= -998, data["maxbatch_msignal"], ((((np.cos((data["minbatch_msignal"]))) * ((((((data["abs_maxbatch_msignal"]) / 2.0)) + (data["abs_maxbatch_msignal"]))/2.0)))) * 2.0) )) - (((data["maxbatch_msignal"]) / 2.0)))) +

                            0.050000*np.tanh((-((np.where(((np.sin((((data["maxbatch_slices2_msignal"]) * (data["maxbatch_slices2"]))))) - ((((data["meanbatch_msignal"]) <= (((data["signal"]) - (data["abs_avgbatch_slices2_msignal"]))))*1.))) <= -998, ((data["maxbatch_slices2_msignal"]) + (data["stdbatch_msignal"])), ((data["mean_abs_chgbatch_slices2_msignal"]) * (data["abs_avgbatch_slices2_msignal"])) ))))) +

                            0.050000*np.tanh(((data["maxtominbatch_msignal"]) * (np.where(data["stdbatch_msignal"] > -998, np.cos((data["abs_avgbatch_msignal"])), np.where(data["maxtominbatch_slices2"] <= -998, data["abs_minbatch_msignal"], data["maxtominbatch_msignal"] ) )))) +

                            0.050000*np.tanh(np.where(data["medianbatch_msignal"] <= -998, ((((((np.cos((data["maxbatch_msignal"]))) * (data["maxbatch_slices2"]))) * (np.sin((data["signal_shift_+1_msignal"]))))) * (data["maxbatch_msignal"])), ((data["minbatch_slices2_msignal"]) + (np.where(data["maxbatch_slices2"] <= -998, data["medianbatch_msignal"], ((data["meanbatch_msignal"]) * (data["maxbatch_slices2"])) ))) )) +

                            0.050000*np.tanh(((np.where(np.cos((data["abs_maxbatch_msignal"])) > -998, ((data["rangebatch_slices2"]) * 2.0), (-((data["abs_maxbatch_msignal"]))) )) * 2.0)) +

                            0.050000*np.tanh(((data["signal"]) * (np.cos((((np.where(data["stdbatch_msignal"] <= -998, np.maximum(((np.where(np.tanh((data["minbatch_slices2_msignal"])) <= -998, data["minbatch_msignal"], data["signal"] ))), ((data["signal"]))), data["stdbatch_msignal"] )) * 2.0)))))) +

                            0.050000*np.tanh(np.where(((((8.0)) + (np.minimum(((data["rangebatch_slices2_msignal"])), ((data["meanbatch_slices2"])))))/2.0) > -998, ((np.where(data["meanbatch_slices2_msignal"] <= -998, data["maxbatch_msignal"], (-((((data["mean_abs_chgbatch_slices2_msignal"]) * (data["medianbatch_slices2_msignal"]))))) )) * 2.0), np.cos((data["mean_abs_chgbatch_slices2_msignal"])) )) +

                            0.050000*np.tanh(((((np.minimum(((data["signal_shift_-1"])), ((((np.sin((data["medianbatch_msignal"]))) - (np.where(((np.cos((data["stdbatch_msignal"]))) * (np.tanh((data["mean_abs_chgbatch_msignal"])))) <= -998, ((data["abs_avgbatch_msignal"]) * (data["abs_maxbatch_slices2_msignal"])), ((data["meanbatch_slices2"]) * (np.tanh((data["mean_abs_chgbatch_msignal"])))) ))))))) * (((np.cos((data["abs_maxbatch_slices2_msignal"]))) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(((data["mean_abs_chgbatch_msignal"]) * (np.where(data["abs_avgbatch_msignal"] <= -998, ((((data["stdbatch_msignal"]) * (data["minbatch_msignal"]))) * (((data["mean_abs_chgbatch_msignal"]) * (np.sin((data["abs_maxbatch"])))))), (-((np.where((((data["abs_minbatch_msignal"]) + (np.maximum(((data["signal"])), ((np.where(data["rangebatch_slices2_msignal"] > -998, data["medianbatch_msignal"], data["minbatch_msignal"] ))))))/2.0) > -998, data["medianbatch_msignal"], data["abs_minbatch_msignal"] )))) )))) +

                            0.050000*np.tanh((((2.66262960433959961)) * (np.cos((np.minimum(((np.maximum(((np.minimum(((data["stdbatch_slices2_msignal"])), ((((data["signal"]) * 2.0)))))), ((data["abs_minbatch_slices2_msignal"]))))), ((np.where(data["abs_maxbatch"] <= -998, data["abs_minbatch_slices2_msignal"], ((data["stdbatch_slices2_msignal"]) * 2.0) ))))))))) +

                            0.050000*np.tanh(((np.sin((((data["abs_maxbatch_slices2"]) - (np.minimum(((np.minimum(((np.sin((data["meanbatch_slices2_msignal"])))), ((np.minimum(((data["abs_maxbatch_msignal"])), ((data["mean_abs_chgbatch_slices2_msignal"])))))))), ((np.minimum(((np.minimum(((((data["signal"]) + (data["signal_shift_+1"])))), ((np.minimum(((data["signal"])), ((data["mean_abs_chgbatch_slices2_msignal"])))))))), ((data["mean_abs_chgbatch_slices2_msignal"]))))))))))) * (data["abs_maxbatch_msignal"]))) +

                            0.050000*np.tanh(((np.cos((data["abs_avgbatch_msignal"]))) * (((data["mean_abs_chgbatch_slices2_msignal"]) * (np.where((((((data["minbatch"]) + (data["rangebatch_msignal"]))) <= (np.minimum(((np.minimum(((((data["abs_avgbatch_msignal"]) * (data["medianbatch_slices2_msignal"])))), ((data["abs_avgbatch_msignal"]))))), ((data["mean_abs_chgbatch_slices2_msignal"])))))*1.) > -998, data["rangebatch_msignal"], np.cos((np.cos((data["abs_avgbatch_msignal"])))) )))))) +

                            0.050000*np.tanh(np.where(data["abs_avgbatch_slices2"] <= -998, np.where(data["maxbatch_msignal"] <= -998, np.cos((data["medianbatch_msignal"])), data["maxbatch_msignal"] ), np.cos((np.where(np.where(data["abs_avgbatch_slices2"] > -998, data["maxbatch_slices2_msignal"], np.cos((np.cos((data["abs_avgbatch_slices2"])))) ) <= -998, data["signal"], data["maxbatch_msignal"] ))) )) +

                            0.050000*np.tanh((((data["signal"]) + ((-((((data["abs_maxbatch_slices2"]) * (data["mean_abs_chgbatch_msignal"])))))))/2.0)) +

                            0.050000*np.tanh(((((((data["mean_abs_chgbatch_slices2"]) * (((((np.sin((((np.cos((data["meanbatch_msignal"]))) - (data["abs_minbatch_msignal"]))))) * (np.maximum(((data["meanbatch_msignal"])), ((data["stdbatch_msignal"])))))) * 2.0)))) * (data["meanbatch_msignal"]))) - ((((((np.sin((data["meanbatch_msignal"]))) * (data["abs_minbatch_msignal"]))) <= (data["abs_maxbatch_slices2_msignal"]))*1.)))) +

                            0.050000*np.tanh(((((np.sin((np.minimum(((np.sin((((np.maximum((((((data["abs_maxbatch_msignal"]) + (np.where(data["abs_maxbatch_msignal"] > -998, data["maxtominbatch_slices2_msignal"], data["maxtominbatch_msignal"] )))/2.0))), ((data["abs_maxbatch_msignal"])))) + (data["medianbatch_msignal"])))))), ((data["meanbatch_slices2"])))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((((data["medianbatch_msignal"]) * (data["signal"]))) - (((((((np.cos((np.sin((data["maxtominbatch_msignal"]))))) > (data["signal_shift_+1_msignal"]))*1.)) <= ((-((np.minimum(((data["abs_avgbatch_msignal"])), ((data["maxbatch_msignal"]))))))))*1.)))) +

                            0.050000*np.tanh(((((data["meanbatch_msignal"]) * ((-((np.tanh((np.where(data["stdbatch_msignal"] > -998, data["mean_abs_chgbatch_msignal"], ((((data["medianbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_msignal"]))) * 2.0) ))))))))) * 2.0)) +

                            0.050000*np.tanh(np.where((((data["meanbatch_msignal"]) <= (data["abs_avgbatch_msignal"]))*1.) > -998, ((((data["medianbatch_slices2_msignal"]) * 2.0)) * ((-((data["mean_abs_chgbatch_slices2_msignal"]))))), data["abs_maxbatch"] )) +

                            0.050000*np.tanh(np.where((((((((data["signal"]) * 2.0)) > (data["maxtominbatch_slices2_msignal"]))*1.)) + (data["signal"])) <= -998, ((data["rangebatch_slices2"]) - ((((np.minimum((((((data["signal"]) <= (data["stdbatch_slices2_msignal"]))*1.))), ((data["stdbatch_slices2_msignal"])))) + (data["rangebatch_slices2"]))/2.0))), ((data["rangebatch_slices2"]) * (np.cos((((data["stdbatch_slices2_msignal"]) * 2.0))))) )) +

                            0.050000*np.tanh(np.where(((data["signal_shift_+1"]) * (data["signal_shift_-1"])) <= -998, data["abs_minbatch_slices2_msignal"], ((data["mean_abs_chgbatch_slices2_msignal"]) * ((((np.minimum(((data["minbatch_slices2"])), ((data["minbatch_slices2"])))) + ((-((data["stdbatch_slices2_msignal"])))))/2.0))) )) +

                            0.050000*np.tanh(((((data["mean_abs_chgbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_msignal"]))) * (((data["minbatch_slices2_msignal"]) - (((((((data["medianbatch_msignal"]) + (np.cos((data["maxbatch_msignal"]))))) - (data["signal_shift_-1"]))) * 2.0)))))) +

                            0.050000*np.tanh(((np.cos((((np.where(data["stdbatch_msignal"] <= -998, (((np.where(data["meanbatch_slices2_msignal"] <= -998, data["meanbatch_msignal"], data["abs_maxbatch_msignal"] )) + ((-((data["abs_avgbatch_msignal"])))))/2.0), data["abs_maxbatch_msignal"] )) + (data["stdbatch_msignal"]))))) * 2.0)) +

                            0.050000*np.tanh(((np.sin((((np.where(((data["maxbatch_msignal"]) * 2.0) > -998, data["abs_maxbatch_msignal"], data["abs_avgbatch_slices2"] )) + (data["meanbatch_msignal"]))))) * 2.0)) +

                            0.050000*np.tanh(np.cos((np.where(data["abs_maxbatch_msignal"] > -998, data["maxbatch_slices2_msignal"], np.where(data["minbatch_msignal"] <= -998, (-((np.where(((np.cos((data["abs_maxbatch_slices2_msignal"]))) * 2.0) <= -998, data["abs_maxbatch"], data["minbatch_slices2_msignal"] )))), np.cos((np.tanh((data["maxbatch_slices2_msignal"])))) ) )))) +

                            0.050000*np.tanh(((np.where(data["signal_shift_-1"] <= -998, np.cos((((data["meanbatch_slices2"]) * 2.0))), np.where((((np.where(((data["stdbatch_msignal"]) * ((-((((np.cos(((-((data["minbatch_msignal"])))))) * 2.0)))))) <= -998, np.tanh((data["minbatch_msignal"])), data["meanbatch_msignal"] )) + (data["signal_shift_-1"]))/2.0) <= -998, data["mean_abs_chgbatch_slices2"], data["meanbatch_msignal"] ) )) / 2.0)) +

                            0.050000*np.tanh(np.where(data["meanbatch_msignal"] > -998, np.where(np.sin(((3.0))) <= -998, data["abs_maxbatch_slices2_msignal"], (((((((data["medianbatch_slices2_msignal"]) > (data["maxtominbatch_msignal"]))*1.)) + (data["abs_maxbatch_slices2_msignal"]))) * (np.sin((data["maxtominbatch_msignal"])))) ), np.sin((data["maxtominbatch_msignal"])) )) +

                            0.050000*np.tanh(((data["rangebatch_slices2_msignal"]) * ((((((((-((data["mean_abs_chgbatch_msignal"])))) * 2.0)) * (((data["mean_abs_chgbatch_msignal"]) + (data["meanbatch_msignal"]))))) + (np.tanh((data["signal_shift_-1"]))))))) +

                            0.050000*np.tanh(((np.minimum(((((data["signal_shift_+1"]) - (np.tanh((((data["abs_minbatch_slices2_msignal"]) + (data["rangebatch_msignal"])))))))), ((((np.where(data["maxtominbatch_slices2_msignal"] > -998, np.tanh((np.cos((data["minbatch_msignal"])))), data["meanbatch_slices2_msignal"] )) * (((((((((data["mean_abs_chgbatch_msignal"]) - (data["maxtominbatch"]))) <= (data["abs_maxbatch_msignal"]))*1.)) + (data["stdbatch_slices2"]))/2.0))))))) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((((((data["signal_shift_+1"]) + (np.minimum(((data["abs_maxbatch_msignal"])), ((data["meanbatch_msignal"])))))) * (np.where(data["abs_avgbatch_msignal"] > -998, ((((np.cos((data["abs_maxbatch_msignal"]))) * 2.0)) * 2.0), data["meanbatch_msignal"] ))))), ((np.where(data["abs_maxbatch_msignal"] > -998, data["meanbatch_msignal"], np.cos((data["abs_maxbatch_msignal"])) ))))) +

                            0.050000*np.tanh(np.sin((np.minimum(((data["meanbatch_msignal"])), ((np.minimum((((((data["rangebatch_slices2_msignal"]) <= (((np.sin((np.sin((data["abs_maxbatch"]))))) * 2.0)))*1.))), (((-((data["rangebatch_slices2"])))))))))))) +

                            0.050000*np.tanh(((((((data["mean_abs_chgbatch_slices2_msignal"]) - (data["abs_maxbatch_msignal"]))) * 2.0)) * (((((np.cos((np.where(data["abs_avgbatch_slices2"] <= -998, data["mean_abs_chgbatch_slices2"], data["medianbatch_msignal"] )))) - (np.where(data["abs_avgbatch_slices2"] <= -998, ((((np.cos((((data["abs_maxbatch_slices2_msignal"]) / 2.0)))) - (data["medianbatch_msignal"]))) * 2.0), data["medianbatch_msignal"] )))) * 2.0)))) +

                            0.050000*np.tanh(((np.minimum(((data["medianbatch_msignal"])), ((((np.sin((data["maxtominbatch_msignal"]))) * (np.maximum(((data["abs_minbatch_slices2"])), ((np.where(((np.cos((data["stdbatch_slices2"]))) / 2.0) <= -998, data["signal_shift_+1"], data["maxbatch_slices2"] )))))))))) * 2.0)) +

                            0.050000*np.tanh(np.cos((np.where(np.where(data["minbatch_slices2_msignal"] > -998, np.tanh((data["signal_shift_+1"])), ((((np.cos((data["maxtominbatch_msignal"]))) / 2.0)) * (data["abs_minbatch_msignal"])) ) > -998, data["abs_maxbatch_slices2_msignal"], (((data["signal"]) > (data["abs_minbatch_msignal"]))*1.) )))) +

                            0.050000*np.tanh(np.cos((np.where(((data["meanbatch_slices2_msignal"]) + (data["maxtominbatch_slices2_msignal"])) > -998, ((data["maxbatch_slices2_msignal"]) * 2.0), np.sin((np.where(data["rangebatch_msignal"] <= -998, ((((np.sin((data["meanbatch_msignal"]))) + (data["stdbatch_slices2"]))) * (np.minimum(((data["maxbatch_msignal"])), ((data["minbatch_slices2_msignal"]))))), np.sin((data["meanbatch_slices2_msignal"])) ))) )))) +

                            0.050000*np.tanh(np.minimum(((np.where(np.cos((data["maxbatch_slices2"])) > -998, np.cos((data["maxbatch_msignal"])), data["meanbatch_msignal"] ))), ((((data["mean_abs_chgbatch_slices2_msignal"]) * (((np.where(data["maxbatch_slices2"] > -998, data["medianbatch_msignal"], (((data["maxbatch_msignal"]) <= (data["minbatch_slices2_msignal"]))*1.) )) * (data["minbatch_slices2_msignal"])))))))) +

                            0.050000*np.tanh(np.minimum(((data["meanbatch_msignal"])), ((np.minimum(((data["meanbatch_msignal"])), ((data["signal"]))))))) +

                            0.050000*np.tanh(((np.maximum(((data["signal"])), ((np.where(data["signal_shift_+1"] > -998, data["signal"], np.maximum(((((((np.maximum(((data["minbatch_msignal"])), ((data["abs_maxbatch_msignal"])))) * (np.cos((np.sin((data["abs_maxbatch_slices2"]))))))) / 2.0))), ((data["abs_maxbatch_slices2_msignal"]))) ))))) * (((np.cos((np.sin(((0.0)))))) * (data["meanbatch_msignal"]))))) +

                            0.050000*np.tanh(np.cos((((np.where(np.sin((data["abs_maxbatch_slices2"])) <= -998, data["minbatch"], data["stdbatch_msignal"] )) * 2.0)))) +

                            0.050000*np.tanh(((((np.tanh((np.where((-((data["medianbatch_slices2_msignal"]))) <= -998, data["abs_avgbatch_slices2"], data["mean_abs_chgbatch_slices2_msignal"] )))) * ((-((np.where(data["minbatch_msignal"] > -998, ((data["mean_abs_chgbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2_msignal"])), np.tanh((data["meanbatch_slices2_msignal"])) ))))))) * 2.0)) +

                            0.050000*np.tanh(np.where(np.where(np.sin((data["signal_shift_+1"])) > -998, ((np.where((((data["maxbatch_slices2"]) <= (np.where(data["maxtominbatch_slices2"] > -998, data["meanbatch_msignal"], np.sin((data["meanbatch_msignal"])) )))*1.) > -998, np.cos((data["meanbatch_slices2_msignal"])), data["stdbatch_slices2"] )) * 2.0), data["minbatch"] ) <= -998, data["minbatch"], np.cos((((data["stdbatch_msignal"]) * 2.0))) )) +

                            0.050000*np.tanh(np.cos((np.maximum(((data["maxbatch_slices2_msignal"])), ((data["signal"])))))) +

                            0.050000*np.tanh(np.where(np.sin((np.sin((data["meanbatch_msignal"])))) <= -998, ((data["signal"]) * 2.0), ((data["meanbatch_msignal"]) * ((-((np.where(np.where(data["abs_avgbatch_msignal"] <= -998, data["signal_shift_-1"], ((((data["abs_avgbatch_msignal"]) * ((-((np.where(np.tanh((data["abs_avgbatch_msignal"])) > -998, data["mean_abs_chgbatch_msignal"], data["abs_avgbatch_msignal"] ))))))) * 2.0) ) > -998, data["mean_abs_chgbatch_msignal"], data["medianbatch_slices2"] )))))) )) +

                            0.050000*np.tanh(np.minimum(((np.minimum(((np.sin((np.sin(((-((np.sin((((np.cos((data["minbatch_slices2"]))) + (np.where(((data["maxtominbatch_slices2_msignal"]) * 2.0) <= -998, data["signal_shift_+1"], np.sin((((data["abs_avgbatch_msignal"]) - (np.cos(((-((data["meanbatch_slices2"]))))))))) )))))))))))))), ((data["abs_maxbatch"]))))), ((np.sin((np.sin((data["maxtominbatch_slices2_msignal"])))))))) +

                            0.050000*np.tanh(np.sin((data["signal"]))) +

                            0.050000*np.tanh(((data["medianbatch_slices2_msignal"]) * (np.tanh((np.cos((np.maximum(((data["maxbatch_msignal"])), ((((np.cos(((2.39422607421875000)))) + (np.tanh((data["signal"])))))))))))))) +

                            0.050000*np.tanh(((np.minimum(((np.sin((data["abs_maxbatch"])))), ((np.sin((((np.maximum(((np.sin((((np.minimum(((data["maxbatch_msignal"])), ((data["abs_maxbatch_slices2_msignal"])))) - (((np.sin((data["stdbatch_slices2_msignal"]))) / 2.0))))))), ((data["meanbatch_slices2"])))) - (np.sin((np.sin((data["signal"])))))))))))) * 2.0)) +

                            0.050000*np.tanh(np.sin((data["signal_shift_-1"]))) +

                            0.050000*np.tanh(np.where(np.where(np.cos((data["signal"])) <= -998, np.where((((data["meanbatch_slices2"]) > (((data["meanbatch_slices2_msignal"]) / 2.0)))*1.) > -998, data["minbatch"], ((data["medianbatch_slices2"]) * 2.0) ), np.maximum((((7.32007217407226562))), ((data["rangebatch_msignal"]))) ) <= -998, data["medianbatch_slices2_msignal"], ((data["maxtominbatch_slices2_msignal"]) * (np.sin((np.maximum(((data["medianbatch_slices2"])), ((data["rangebatch_msignal"]))))))) )) +

                            0.050000*np.tanh(np.sin((((data["maxtominbatch_msignal"]) - (data["signal_shift_+1_msignal"]))))) +

                            0.050000*np.tanh(np.where(data["maxtominbatch"] > -998, np.cos((data["maxbatch_slices2_msignal"])), (-((data["abs_minbatch_msignal"]))) )) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) * ((((((data["mean_abs_chgbatch_msignal"]) - (np.tanh((((data["minbatch_slices2_msignal"]) / 2.0)))))) <= (data["mean_abs_chgbatch_msignal"]))*1.)))) +

                            0.050000*np.tanh(np.tanh((((np.where(np.maximum(((((data["meanbatch_msignal"]) * 2.0))), ((data["abs_avgbatch_slices2"]))) > -998, ((np.cos((data["maxbatch_msignal"]))) * 2.0), np.sin((np.sin((data["rangebatch_slices2"])))) )) * 2.0)))) +

                            0.050000*np.tanh(np.minimum(((data["medianbatch_msignal"])), ((data["signal_shift_-1"])))) +

                            0.050000*np.tanh(np.where(np.where((((data["abs_avgbatch_slices2_msignal"]) > (data["mean_abs_chgbatch_slices2_msignal"]))*1.) > -998, np.sin((data["mean_abs_chgbatch_slices2_msignal"])), (((np.cos(((((data["signal"]) <= (data["minbatch_slices2_msignal"]))*1.)))) <= (data["mean_abs_chgbatch_slices2_msignal"]))*1.) ) > -998, (-((((data["mean_abs_chgbatch_slices2_msignal"]) * (data["meanbatch_msignal"]))))), data["meanbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(np.where((-((data["medianbatch_slices2_msignal"]))) > -998, np.cos((data["minbatch_msignal"])), np.where(((data["signal_shift_-1"]) / 2.0) <= -998, (-((np.cos((data["meanbatch_slices2"]))))), np.tanh((((np.where(np.sin((data["maxbatch_msignal"])) > -998, np.cos((data["minbatch_msignal"])), np.where(data["stdbatch_slices2_msignal"] <= -998, data["meanbatch_msignal"], np.tanh((data["meanbatch_msignal"])) ) )) + (data["minbatch_msignal"])))) ) )) +

                            0.050000*np.tanh((-((((((data["mean_abs_chgbatch_slices2_msignal"]) * (data["rangebatch_slices2"]))) * 2.0))))) +

                            0.050000*np.tanh(data["medianbatch_msignal"]) +

                            0.050000*np.tanh(np.where(np.minimum(((data["stdbatch_slices2_msignal"])), ((np.where(np.where(data["maxtominbatch_msignal"] <= -998, np.tanh((data["minbatch_slices2_msignal"])), data["minbatch"] ) <= -998, data["meanbatch_msignal"], data["meanbatch_slices2_msignal"] )))) <= -998, data["minbatch"], np.minimum((((6.63431692123413086))), ((data["minbatch_slices2_msignal"]))) )) +

                            0.050000*np.tanh(np.where(data["rangebatch_slices2"] <= -998, np.cos((np.sin((np.sin((data["meanbatch_msignal"])))))), data["medianbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(np.cos((np.where((((data["minbatch_msignal"]) <= (data["medianbatch_msignal"]))*1.) <= -998, (((np.tanh((data["signal_shift_+1"]))) + (np.cos((np.minimum(((data["signal_shift_+1"])), ((np.tanh((np.sin((data["medianbatch_slices2"])))))))))))/2.0), ((data["signal_shift_+1"]) + (np.where(data["signal_shift_+1"] <= -998, np.minimum(((data["meanbatch_msignal"])), ((data["maxtominbatch_msignal"]))), ((data["medianbatch_msignal"]) * (data["abs_minbatch_msignal"])) ))) )))) +

                            0.050000*np.tanh(((np.minimum(((np.where(np.sin((((np.sin((((data["medianbatch_msignal"]) / 2.0)))) * 2.0))) <= -998, (((np.maximum(((data["minbatch_slices2"])), ((np.cos((data["maxbatch_slices2_msignal"])))))) + (data["abs_avgbatch_slices2_msignal"]))/2.0), ((((data["abs_minbatch_slices2"]) - ((-((data["signal_shift_+1"])))))) / 2.0) ))), ((np.cos((np.cos((data["mean_abs_chgbatch_msignal"])))))))) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((((np.cos((data["stdbatch_slices2_msignal"]))) * (data["signal_shift_-1"])))), ((data["signal_shift_-1"])))) +

                            0.050000*np.tanh((-((((data["abs_minbatch_slices2_msignal"]) * (data["maxtominbatch_slices2"])))))) +

                            0.050000*np.tanh(np.cos((((data["signal_shift_-1"]) * 2.0)))) +

                            0.050000*np.tanh(np.cos((np.where(data["abs_avgbatch_slices2"] > -998, data["maxbatch_slices2_msignal"], data["medianbatch_msignal"] )))) +

                            0.050000*np.tanh(np.minimum(((data["meanbatch_msignal"])), ((((np.sin((np.where((((data["minbatch_slices2"]) <= (((data["abs_maxbatch"]) * 2.0)))*1.) <= -998, np.where(data["meanbatch_slices2_msignal"] <= -998, (0.25848990678787231), np.sin((data["signal"])) ), np.sin((np.where(np.sin((data["minbatch_msignal"])) <= -998, data["maxbatch_slices2_msignal"], data["abs_maxbatch"] ))) )))) * 2.0))))) +

                            0.050000*np.tanh(np.minimum(((((np.where(data["signal_shift_-1"] <= -998, ((data["signal_shift_+1"]) - (((((np.sin((np.tanh((data["medianbatch_slices2_msignal"]))))) / 2.0)) - (((data["medianbatch_msignal"]) + (data["abs_minbatch_msignal"])))))), ((np.cos((data["signal_shift_+1"]))) * (np.sin((data["signal_shift_-1"])))) )) + (data["signal_shift_-1"])))), ((data["rangebatch_slices2"])))) +

                            0.050000*np.tanh(np.where(data["abs_maxbatch_slices2_msignal"] <= -998, data["minbatch_msignal"], np.minimum(((data["signal_shift_-1"])), ((np.tanh((data["abs_minbatch_slices2_msignal"]))))) )) +

                            0.050000*np.tanh(((((((np.where(data["stdbatch_slices2_msignal"] > -998, data["signal_shift_-1"], ((data["rangebatch_slices2_msignal"]) * 2.0) )) * 2.0)) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((np.where(data["stdbatch_slices2_msignal"] <= -998, np.cos((((np.sin((data["minbatch"]))) * 2.0))), data["medianbatch_slices2"] )) * (np.cos((data["stdbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(np.sin((np.minimum(((np.minimum(((data["maxtominbatch_msignal"])), ((data["mean_abs_chgbatch_slices2"]))))), (((-((np.where(data["signal_shift_-1"] <= -998, ((data["rangebatch_msignal"]) * 2.0), np.tanh((data["abs_avgbatch_msignal"])) )))))))))) +

                            0.050000*np.tanh(np.where(data["meanbatch_msignal"] > -998, data["minbatch_slices2"], (((np.cos((data["abs_minbatch_slices2"]))) + (((((data["maxtominbatch_msignal"]) + (data["maxbatch_msignal"]))) - (data["signal_shift_-1"]))))/2.0) )) +

                            0.050000*np.tanh((-((((((((((((((data["signal_shift_-1"]) - (data["abs_maxbatch_slices2_msignal"]))) / 2.0)) + (data["rangebatch_msignal"]))) * (np.cos((((data["abs_maxbatch_slices2_msignal"]) - (np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["medianbatch_slices2"], np.cos((((data["medianbatch_msignal"]) - (data["maxbatch_slices2_msignal"])))) )))))))) / 2.0)) * 2.0))))) +

                            0.050000*np.tanh(((np.tanh((data["signal"]))) * (np.where(data["signal_shift_-1"] <= -998, data["abs_minbatch_msignal"], data["abs_maxbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(np.minimum(((np.tanh((np.sin((np.sin((((data["signal_shift_-1"]) - ((((np.maximum(((data["abs_avgbatch_slices2_msignal"])), ((np.sin((np.sin((data["abs_avgbatch_slices2_msignal"])))))))) + (np.where(data["abs_avgbatch_slices2_msignal"] <= -998, data["signal_shift_-1"], data["abs_avgbatch_slices2_msignal"] )))/2.0))))))))))), ((((((data["signal_shift_-1"]) / 2.0)) + (data["signal_shift_-1"])))))) +

                            0.050000*np.tanh(((data["minbatch_slices2_msignal"]) * (np.sin((((((np.maximum(((data["signal"])), ((((((4.24774360656738281)) <= ((((((((data["signal_shift_+1"]) * 2.0)) * (np.cos((data["rangebatch_msignal"]))))) <= (data["signal_shift_+1"]))*1.)))*1.))))) * 2.0)) + (((data["signal_shift_-1"]) / 2.0)))))))) +

                            0.050000*np.tanh(np.sin((np.where(data["maxtominbatch_slices2_msignal"] <= -998, np.where(data["abs_maxbatch"] <= -998, (9.0), np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["signal_shift_-1"], data["rangebatch_slices2"] ) ), np.sin((np.where(data["medianbatch_slices2"] <= -998, np.where(np.tanh((data["signal_shift_+1"])) <= -998, np.sin((np.cos((np.cos(((((data["stdbatch_slices2"]) + (data["medianbatch_slices2_msignal"]))/2.0))))))), data["abs_avgbatch_msignal"] ), data["abs_maxbatch"] ))) )))) +

                            0.050000*np.tanh(np.cos((data["maxbatch_msignal"]))) +

                            0.050000*np.tanh(np.minimum(((np.where(np.sin((np.tanh((((data["signal"]) + (np.minimum(((data["abs_maxbatch"])), ((data["abs_avgbatch_slices2_msignal"]))))))))) > -998, data["signal"], ((data["abs_minbatch_slices2_msignal"]) * (data["stdbatch_slices2_msignal"])) ))), ((((data["signal_shift_-1_msignal"]) * (data["stdbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(((data["signal_shift_+1_msignal"]) * (((np.cos((np.where(data["stdbatch_msignal"] > -998, data["minbatch_msignal"], data["stdbatch_msignal"] )))) - (np.maximum(((np.minimum(((np.minimum(((np.maximum(((((data["signal_shift_+1"]) - (data["medianbatch_msignal"])))), ((((data["minbatch_msignal"]) * (data["minbatch_msignal"]))))))), ((data["maxtominbatch_msignal"]))))), ((data["signal_shift_+1_msignal"]))))), ((((data["minbatch_msignal"]) * (np.tanh((data["stdbatch_msignal"])))))))))))) +

                            0.050000*np.tanh(np.minimum(((data["meanbatch_slices2_msignal"])), ((np.cos((data["meanbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(data["meanbatch_msignal"]) +

                            0.050000*np.tanh(((data["medianbatch_msignal"]) - (data["abs_minbatch_slices2"]))) +

                            0.050000*np.tanh(((np.sin((np.sin((data["signal_shift_-1"]))))) + (np.where(np.cos((data["rangebatch_slices2"])) > -998, np.sin((data["abs_maxbatch_slices2"])), ((np.sin((data["maxtominbatch_msignal"]))) * ((1.0))) )))) +

                            0.050000*np.tanh(((data["meanbatch_slices2"]) * (data["meanbatch_slices2_msignal"]))) +

                            0.050000*np.tanh((-((np.maximum(((data["rangebatch_msignal"])), ((np.sin(((7.61515092849731445)))))))))) +

                            0.050000*np.tanh(np.sin((np.where((((np.sin((data["abs_minbatch_msignal"]))) + ((0.53171408176422119)))/2.0) > -998, data["maxtominbatch_slices2_msignal"], (((((((np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["abs_maxbatch_slices2"], np.cos((data["medianbatch_msignal"])) )) * 2.0)) + (np.cos((data["abs_maxbatch_slices2"]))))/2.0)) * (data["abs_maxbatch_slices2"])) )))) +

                            0.050000*np.tanh(((data["stdbatch_msignal"]) * (((np.minimum(((data["signal_shift_-1"])), ((((data["signal_shift_+1"]) - (((data["signal_shift_-1"]) - (data["signal_shift_-1"])))))))) - (data["stdbatch_slices2"]))))) +

                            0.050000*np.tanh(np.sin((np.sin((np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2"], (((data["rangebatch_slices2"]) + (data["meanbatch_slices2_msignal"]))/2.0) )))))) +

                            0.050000*np.tanh(np.where(np.cos((data["minbatch_msignal"])) <= -998, data["minbatch_msignal"], ((((data["abs_maxbatch_slices2_msignal"]) - (((data["maxtominbatch_slices2"]) - (data["abs_maxbatch_slices2_msignal"]))))) * (np.cos((data["minbatch_msignal"])))) )) +

                            0.050000*np.tanh(((((data["signal_shift_-1"]) * 2.0)) * (np.where(data["abs_maxbatch"] <= -998, ((data["medianbatch_msignal"]) - (np.sin((data["maxtominbatch"])))), ((np.tanh((data["medianbatch_msignal"]))) / 2.0) )))) +

                            0.050000*np.tanh(np.minimum(((((((np.cos((data["abs_maxbatch_msignal"]))) / 2.0)) * (np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["rangebatch_slices2_msignal"], np.sin((((((((data["maxtominbatch_slices2"]) > (data["mean_abs_chgbatch_slices2_msignal"]))*1.)) <= (data["maxbatch_slices2"]))*1.))) ))))), ((np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["rangebatch_slices2_msignal"], data["abs_maxbatch_msignal"] ))))) +

                            0.050000*np.tanh(np.minimum((((((((data["maxtominbatch"]) + (np.cos((((data["abs_maxbatch_msignal"]) + (data["signal_shift_+1_msignal"]))))))/2.0)) * (np.where(((np.sin((data["signal_shift_+1"]))) * (data["stdbatch_slices2_msignal"])) > -998, data["abs_minbatch_slices2_msignal"], data["mean_abs_chgbatch_slices2"] ))))), ((data["abs_avgbatch_slices2_msignal"])))) +

                            0.050000*np.tanh(((data["abs_maxbatch"]) * (((np.cos((data["medianbatch_msignal"]))) * (np.tanh((np.where(np.minimum(((data["signal_shift_-1"])), (((((8.06406688690185547)) * (((data["maxbatch_slices2"]) * (np.tanh((np.where(data["maxbatch_msignal"] <= -998, np.cos(((8.0))), (((8.06406688690185547)) * (data["mean_abs_chgbatch_slices2_msignal"])) )))))))))) <= -998, data["abs_maxbatch_slices2_msignal"], np.cos((data["maxbatch_slices2"])) )))))))) +

                            0.050000*np.tanh(((np.where((((np.maximum(((data["abs_maxbatch_msignal"])), ((data["abs_maxbatch_msignal"])))) + (np.sin((np.tanh((np.maximum(((((data["abs_maxbatch_msignal"]) + (data["abs_maxbatch_msignal"])))), ((np.minimum(((data["abs_avgbatch_msignal"])), (((-((data["abs_avgbatch_slices2_msignal"])))))))))))))))/2.0) > -998, np.sin((data["abs_maxbatch_slices2"])), np.tanh((np.tanh((data["abs_maxbatch_slices2_msignal"])))) )) * 2.0)) +

                            0.050000*np.tanh(np.where(data["stdbatch_slices2"] > -998, data["meanbatch_msignal"], (((((data["meanbatch_msignal"]) * (data["meanbatch_msignal"]))) <= (np.sin((data["meanbatch_msignal"]))))*1.) )) +

                            0.050000*np.tanh(((((data["signal"]) * (data["meanbatch_msignal"]))) - ((((((data["meanbatch_msignal"]) * (data["signal"]))) > ((-((((((data["signal"]) * (data["meanbatch_msignal"]))) - ((((data["signal"]) > (data["medianbatch_msignal"]))*1.))))))))*1.)))) +

                            0.050000*np.tanh(np.where(data["signal"] <= -998, data["signal"], data["signal"] )) +

                            0.050000*np.tanh(((data["signal_shift_-1_msignal"]) * (np.sin((((np.maximum(((data["stdbatch_msignal"])), ((np.minimum(((((((data["stdbatch_msignal"]) - (((((data["stdbatch_msignal"]) * 2.0)) * 2.0)))) - (data["abs_maxbatch_slices2"])))), ((np.minimum(((data["stdbatch_msignal"])), ((((((data["signal_shift_-1_msignal"]) * (np.sin((((data["stdbatch_msignal"]) * 2.0)))))) * (data["stdbatch_msignal"])))))))))))) * 2.0)))))) +

                            0.050000*np.tanh(data["meanbatch_msignal"]) +

                            0.050000*np.tanh(np.sin((data["medianbatch_slices2"]))) +

                            0.050000*np.tanh(np.where(data["abs_maxbatch_slices2_msignal"] <= -998, (((np.where(data["signal_shift_-1"] <= -998, data["abs_maxbatch_slices2_msignal"], data["signal_shift_-1"] )) + (data["maxbatch_msignal"]))/2.0), ((data["meanbatch_msignal"]) * (np.where(np.sin((data["abs_avgbatch_msignal"])) <= -998, data["abs_maxbatch_slices2_msignal"], ((data["signal_shift_-1"]) / 2.0) ))) )) +

                            0.050000*np.tanh((((np.where(data["abs_avgbatch_msignal"] <= -998, data["abs_minbatch_slices2"], np.sin((((data["signal_shift_-1"]) - (data["abs_avgbatch_slices2_msignal"])))) )) + (data["meanbatch_msignal"]))/2.0)) +

                            0.050000*np.tanh(np.where(np.sin((np.where(((np.cos((np.cos((data["maxtominbatch_slices2_msignal"]))))) * (data["minbatch_msignal"])) > -998, data["abs_avgbatch_slices2"], data["mean_abs_chgbatch_slices2_msignal"] ))) <= -998, np.cos(((-((data["stdbatch_msignal"]))))), data["maxtominbatch_slices2_msignal"] )) +

                            0.050000*np.tanh((((np.maximum(((data["maxbatch_slices2"])), ((data["mean_abs_chgbatch_slices2"])))) + (np.cos(((-((np.where(np.maximum(((data["meanbatch_msignal"])), ((np.sin(((9.0)))))) > -998, data["signal_shift_-1"], (-((data["signal_shift_+1"]))) ))))))))/2.0)) +

                            0.050000*np.tanh(np.minimum(((np.where(((np.where(data["signal_shift_-1_msignal"] <= -998, ((data["stdbatch_slices2_msignal"]) * (((data["signal"]) * (np.tanh((data["signal"])))))), data["abs_avgbatch_slices2_msignal"] )) + (((data["signal"]) * (data["stdbatch_msignal"])))) <= -998, ((((data["signal_shift_-1_msignal"]) * (data["stdbatch_msignal"]))) * (data["signal_shift_-1_msignal"])), ((data["signal_shift_-1_msignal"]) * (data["stdbatch_slices2_msignal"])) ))), ((data["signal"])))) +

                            0.050000*np.tanh(np.minimum(((data["signal_shift_-1"])), ((data["medianbatch_msignal"])))) +

                            0.050000*np.tanh(np.where(data["medianbatch_msignal"] > -998, data["meanbatch_msignal"], data["meanbatch_msignal"] )) +

                            0.050000*np.tanh(np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) * (((data["maxbatch_slices2_msignal"]) + (((data["stdbatch_msignal"]) * (np.minimum(((data["maxbatch_msignal"])), ((((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0))))))))))))) +

                            0.050000*np.tanh(((((np.sin((data["signal_shift_+1"]))) * 2.0)) * (data["medianbatch_msignal"]))) +

                            0.050000*np.tanh(np.where(np.where(np.minimum(((data["abs_avgbatch_slices2_msignal"])), (((-((data["abs_avgbatch_slices2"])))))) <= -998, data["maxbatch_slices2_msignal"], np.sin((data["maxbatch_slices2_msignal"])) ) <= -998, ((((data["medianbatch_slices2_msignal"]) * 2.0)) - (data["mean_abs_chgbatch_slices2_msignal"])), ((np.cos(((((((5.27607488632202148)) - (((data["maxtominbatch"]) * (np.sin((data["maxbatch_slices2_msignal"]))))))) * 2.0)))) * (data["abs_avgbatch_slices2"])) )) +

                            0.050000*np.tanh(np.sin((np.maximum(((data["maxbatch_slices2"])), ((np.maximum(((np.maximum((((((data["maxtominbatch"]) <= (data["minbatch_slices2"]))*1.))), ((np.where(((data["abs_maxbatch_slices2_msignal"]) + (((data["minbatch"]) * (data["abs_maxbatch_slices2"])))) > -998, data["abs_maxbatch_slices2"], data["abs_avgbatch_msignal"] )))))), ((data["minbatch_slices2_msignal"]))))))))) +

                            0.050000*np.tanh(np.where(data["maxbatch_msignal"] <= -998, (10.0), data["signal"] )) +

                            0.050000*np.tanh(np.cos((((data["stdbatch_msignal"]) - (((data["rangebatch_slices2"]) - (((data["abs_minbatch_slices2_msignal"]) - ((((np.minimum(((data["stdbatch_slices2"])), ((np.where(((((np.where(data["rangebatch_slices2"] > -998, data["rangebatch_msignal"], ((data["abs_minbatch_slices2"]) + (data["abs_minbatch_slices2_msignal"])) )) * (data["maxtominbatch"]))) + (data["mean_abs_chgbatch_slices2_msignal"])) > -998, data["signal"], data["maxtominbatch"] ))))) > (data["rangebatch_slices2"]))*1.)))))))))) +

                            0.050000*np.tanh(np.minimum(((((np.minimum(((data["medianbatch_slices2"])), ((((np.cos((((np.minimum(((((((data["medianbatch_slices2"]) * 2.0)) / 2.0))), ((data["medianbatch_slices2"])))) * 2.0)))) * 2.0))))) * 2.0))), ((((np.cos((((data["medianbatch_slices2"]) * 2.0)))) * 2.0))))) +

                            0.050000*np.tanh(((data["minbatch"]) * (np.sin((((np.maximum(((data["minbatch"])), ((data["rangebatch_slices2"])))) + (np.cos(((((data["maxtominbatch"]) <= (np.sin((((np.cos(((((7.44715881347656250)) - (np.cos((np.cos((data["medianbatch_slices2_msignal"]))))))))) + (data["minbatch"]))))))*1.)))))))))) +

                            0.050000*np.tanh(np.where(data["stdbatch_slices2_msignal"] <= -998, ((np.maximum(((data["stdbatch_slices2_msignal"])), ((np.cos((np.tanh((np.tanh((data["abs_maxbatch_msignal"])))))))))) / 2.0), np.cos((np.where(np.where(data["mean_abs_chgbatch_msignal"] <= -998, ((np.sin((data["stdbatch_slices2_msignal"]))) * 2.0), np.cos((data["minbatch_msignal"])) ) <= -998, data["maxtominbatch_slices2"], data["stdbatch_slices2_msignal"] ))) )) +

                            0.050000*np.tanh(np.sin((data["abs_maxbatch"]))) +

                            0.050000*np.tanh((((((data["minbatch_msignal"]) * 2.0)) <= (data["abs_minbatch_slices2"]))*1.)) +

                            0.050000*np.tanh(np.where(np.where((-(((((((data["maxbatch_slices2"]) + (data["meanbatch_slices2_msignal"]))/2.0)) * 2.0)))) > -998, data["maxbatch_msignal"], np.minimum(((np.sin(((-((data["rangebatch_msignal"]))))))), ((data["maxbatch_slices2"]))) ) > -998, np.sin((data["abs_maxbatch_slices2"])), np.minimum(((data["signal_shift_+1_msignal"])), ((np.sin((data["abs_minbatch_slices2_msignal"]))))) )) +

                            0.050000*np.tanh(np.minimum(((((((data["maxbatch_slices2_msignal"]) * (((data["abs_avgbatch_slices2_msignal"]) + ((((np.tanh(((8.0)))) <= (data["signal_shift_+1"]))*1.)))))) / 2.0))), ((((np.cos((data["maxbatch_slices2_msignal"]))) / 2.0))))) +

                            0.050000*np.tanh(((np.where(np.sin((data["abs_maxbatch"])) > -998, ((((data["abs_maxbatch_slices2_msignal"]) * 2.0)) * 2.0), (((((data["meanbatch_msignal"]) + (((np.minimum(((np.sin((data["abs_avgbatch_slices2_msignal"])))), ((np.where(np.tanh((data["minbatch"])) > -998, data["signal_shift_+1"], data["abs_minbatch_slices2"] ))))) * 2.0)))) <= (data["abs_minbatch_slices2"]))*1.) )) / 2.0)) +

                            0.050000*np.tanh((-((((np.where(((np.sin(((((data["abs_maxbatch"]) <= (data["abs_minbatch_slices2"]))*1.)))) * (data["abs_minbatch_slices2"])) <= -998, (-((data["maxbatch_msignal"]))), (-((np.cos((data["abs_maxbatch_slices2"]))))) )) * 2.0))))) +

                            0.050000*np.tanh(np.minimum(((np.where(np.where(data["meanbatch_msignal"] <= -998, (((data["medianbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2"]))/2.0), ((data["abs_avgbatch_msignal"]) * (np.minimum(((data["signal_shift_+1"])), ((data["abs_avgbatch_slices2"]))))) ) > -998, data["signal_shift_+1"], (((data["medianbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2"]))/2.0) ))), ((data["medianbatch_msignal"])))) +

                            0.050000*np.tanh(data["stdbatch_msignal"]) +

                            0.050000*np.tanh(((((((data["stdbatch_slices2_msignal"]) > (np.maximum((((((data["stdbatch_slices2_msignal"]) + (np.cos((np.minimum(((((((((data["abs_avgbatch_slices2_msignal"]) / 2.0)) - (((data["mean_abs_chgbatch_slices2_msignal"]) - (data["meanbatch_msignal"]))))) / 2.0))), ((data["minbatch_slices2_msignal"])))))))/2.0))), ((data["abs_maxbatch"])))))*1.)) + (np.cos((np.minimum(((((((data["abs_maxbatch"]) - (data["mean_abs_chgbatch_msignal"]))) / 2.0))), ((data["minbatch_slices2_msignal"])))))))/2.0)) +

                            0.050000*np.tanh(((np.where((((((data["medianbatch_slices2"]) + (data["abs_maxbatch_msignal"]))) <= (data["stdbatch_slices2_msignal"]))*1.) <= -998, (2.51731467247009277), ((data["medianbatch_slices2"]) * (np.cos((((data["abs_maxbatch_msignal"]) + (np.where(data["signal_shift_+1_msignal"] <= -998, data["medianbatch_msignal"], np.sin((data["medianbatch_slices2"])) ))))))) )) * (data["abs_maxbatch_msignal"]))) +

                            0.050000*np.tanh(np.sin((np.sin((data["abs_minbatch_slices2"]))))) +

                            0.050000*np.tanh(np.where((7.47483921051025391) > -998, np.cos((data["maxbatch_slices2_msignal"])), data["medianbatch_slices2"] )) +

                            0.050000*np.tanh(np.sin((((data["signal_shift_-1_msignal"]) * ((((np.sin((np.sin((data["maxbatch_msignal"]))))) + ((((data["meanbatch_slices2_msignal"]) > (np.maximum(((data["stdbatch_slices2"])), (((-(((-((((((0.0)) > ((0.0)))*1.))))))))))))*1.)))/2.0)))))) +

                            0.050000*np.tanh(((((((((data["medianbatch_slices2_msignal"]) + ((-((data["meanbatch_slices2_msignal"])))))/2.0)) * 2.0)) <= (np.sin((np.sin((data["signal"]))))))*1.)) +

                            0.050000*np.tanh(np.cos((np.maximum(((np.where((((data["abs_maxbatch_msignal"]) <= (((data["signal_shift_-1_msignal"]) * (data["medianbatch_msignal"]))))*1.) > -998, data["signal_shift_-1_msignal"], data["stdbatch_slices2"] ))), ((np.where(((data["signal_shift_-1_msignal"]) * (data["mean_abs_chgbatch_slices2_msignal"])) > -998, data["abs_maxbatch_msignal"], (((data["maxtominbatch_slices2"]) <= (data["maxtominbatch_msignal"]))*1.) ))))))) +

                            0.050000*np.tanh(np.where((5.0) > -998, ((data["stdbatch_msignal"]) * (data["signal_shift_+1_msignal"])), np.cos((np.minimum(((data["signal_shift_+1_msignal"])), ((np.where(np.tanh((((data["medianbatch_slices2_msignal"]) * 2.0))) <= -998, data["abs_minbatch_slices2_msignal"], np.where(data["stdbatch_slices2"] > -998, ((data["stdbatch_msignal"]) / 2.0), np.maximum(((data["meanbatch_msignal"])), ((((data["minbatch_slices2_msignal"]) / 2.0)))) ) )))))) )) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) * 2.0)) +

                            0.050000*np.tanh(np.where((((((data["meanbatch_msignal"]) / 2.0)) <= ((((data["maxbatch_msignal"]) + (((data["maxbatch_slices2"]) / 2.0)))/2.0)))*1.) > -998, data["abs_avgbatch_msignal"], data["maxbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(np.minimum(((data["medianbatch_msignal"])), ((((data["signal"]) + (np.minimum(((data["stdbatch_slices2_msignal"])), ((np.sin((np.where(data["signal"] <= -998, ((data["medianbatch_msignal"]) + (data["medianbatch_msignal"])), np.sin((np.where(data["meanbatch_msignal"] > -998, data["maxtominbatch_slices2_msignal"], np.where(data["abs_maxbatch_slices2"] > -998, data["abs_maxbatch_slices2"], data["abs_minbatch_slices2"] ) ))) )))))))))))) +

                            0.050000*np.tanh(((np.cos((((((data["meanbatch_slices2_msignal"]) * 2.0)) + ((((data["signal_shift_-1_msignal"]) > (np.sin(((((data["medianbatch_slices2"]) > (np.where(data["medianbatch_slices2_msignal"] <= -998, data["signal"], (((((((data["meanbatch_slices2_msignal"]) * 2.0)) * 2.0)) > (data["signal_shift_+1_msignal"]))*1.) )))*1.)))))*1.)))))) * 2.0)) +

                            0.050000*np.tanh(data["meanbatch_slices2"]))     

    

    def GP_class_7(self,data):

        return self.Output( -2.939938 +

                            0.050000*np.tanh(np.where(((data["signal_shift_-1"]) * (np.tanh(((((np.sin((data["maxtominbatch_slices2"]))) + (((data["abs_avgbatch_slices2_msignal"]) - ((((np.sin((data["maxtominbatch_slices2"]))) + (((data["abs_avgbatch_slices2_msignal"]) - (data["abs_maxbatch_msignal"]))))/2.0)))))/2.0))))) <= -998, data["meanbatch_slices2"], data["signal_shift_-1"] )) +

                            0.050000*np.tanh(np.minimum(((np.where(((data["mean_abs_chgbatch_slices2"]) * (data["maxbatch_slices2_msignal"])) > -998, data["medianbatch_slices2"], data["medianbatch_slices2_msignal"] ))), ((((np.maximum(((data["mean_abs_chgbatch_slices2"])), ((data["signal_shift_+1"])))) + (((data["stdbatch_slices2"]) + (data["medianbatch_slices2"])))))))) +

                            0.050000*np.tanh((((((data["signal"]) + (((data["meanbatch_slices2_msignal"]) * (((((((data["signal_shift_+1"]) - ((((data["minbatch_slices2_msignal"]) + (data["meanbatch_msignal"]))/2.0)))) - (np.where(((np.cos((data["meanbatch_slices2"]))) - (data["maxbatch_slices2"])) <= -998, data["signal_shift_+1"], data["meanbatch_slices2_msignal"] )))) - (data["maxbatch_slices2"]))))))/2.0)) * 2.0)) +

                            0.050000*np.tanh(((((((data["abs_maxbatch_msignal"]) * (np.sin((data["abs_maxbatch_slices2"]))))) - ((3.22629404067993164)))) - (np.maximum((((((-((data["abs_maxbatch_msignal"])))) + (((((data["mean_abs_chgbatch_slices2"]) * (data["medianbatch_slices2"]))) - (data["abs_maxbatch_slices2_msignal"])))))), ((np.where(data["rangebatch_msignal"] <= -998, data["medianbatch_slices2_msignal"], np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((data["abs_maxbatch_msignal"]))) ))))))) +

                            0.050000*np.tanh(((((data["meanbatch_slices2"]) + (data["signal"]))) + (((data["abs_avgbatch_slices2_msignal"]) * (np.where(((data["maxbatch_slices2"]) / 2.0) > -998, data["signal"], data["stdbatch_msignal"] )))))) +

                            0.050000*np.tanh(((data["signal"]) - ((((data["meanbatch_slices2_msignal"]) + (data["stdbatch_slices2"]))/2.0)))) +

                            0.050000*np.tanh(((np.minimum(((((data["stdbatch_slices2"]) * (data["medianbatch_slices2"])))), ((data["mean_abs_chgbatch_slices2_msignal"])))) + (((data["maxtominbatch"]) + ((-((((np.where(data["signal_shift_+1"] > -998, data["stdbatch_slices2_msignal"], np.where((-((data["signal"]))) <= -998, data["signal"], ((data["signal_shift_-1_msignal"]) * 2.0) ) )) * (data["maxbatch_msignal"])))))))))) +

                            0.050000*np.tanh(np.where(data["abs_maxbatch_slices2"] > -998, data["signal_shift_+1"], (((data["medianbatch_slices2"]) > ((((data["signal_shift_-1"]) > (data["signal_shift_+1_msignal"]))*1.)))*1.) )) +

                            0.050000*np.tanh(np.where(data["signal"] <= -998, np.where(data["signal"] <= -998, data["meanbatch_slices2"], np.where(data["meanbatch_slices2"] <= -998, data["mean_abs_chgbatch_slices2"], data["abs_minbatch_msignal"] ) ), data["signal"] )) +

                            0.050000*np.tanh(np.minimum(((np.minimum(((data["abs_maxbatch_slices2"])), ((np.where((-((data["abs_avgbatch_slices2_msignal"]))) > -998, data["signal_shift_+1"], np.minimum(((data["medianbatch_slices2"])), ((np.minimum(((data["stdbatch_slices2"])), ((data["medianbatch_slices2"])))))) )))))), ((data["medianbatch_slices2"])))) +

                            0.050000*np.tanh(((data["signal"]) - ((((np.where((((((data["signal_shift_+1"]) - ((-((data["signal_shift_+1"])))))) <= (np.cos((np.cos((data["signal"]))))))*1.) <= -998, data["signal_shift_+1_msignal"], ((data["signal_shift_+1"]) / 2.0) )) <= (((data["signal_shift_+1"]) / 2.0)))*1.)))) +

                            0.050000*np.tanh((-((np.where(data["signal_shift_+1"] > -998, data["stdbatch_msignal"], data["signal_shift_+1"] ))))) +

                            0.050000*np.tanh((((data["maxbatch_slices2"]) + (((np.where(((data["medianbatch_slices2"]) / 2.0) <= -998, np.cos(((((data["abs_maxbatch"]) + (data["maxtominbatch_msignal"]))/2.0))), data["abs_maxbatch"] )) - ((12.97748088836669922)))))/2.0)) +

                            0.050000*np.tanh(np.minimum(((np.sin((data["maxbatch_slices2"])))), ((((np.cos((np.where(data["medianbatch_slices2_msignal"] > -998, data["maxbatch_slices2"], data["meanbatch_slices2_msignal"] )))) - (data["abs_minbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(((data["signal"]) + (np.minimum(((((data["abs_maxbatch_slices2"]) * (data["meanbatch_slices2_msignal"])))), ((data["maxtominbatch"])))))) +

                            0.050000*np.tanh(((data["medianbatch_slices2"]) * 2.0)) +

                            0.050000*np.tanh(((data["abs_avgbatch_slices2_msignal"]) * ((-((np.where(np.where(data["stdbatch_slices2_msignal"] > -998, data["minbatch_slices2_msignal"], ((np.tanh((data["abs_avgbatch_slices2_msignal"]))) / 2.0) ) <= -998, data["stdbatch_msignal"], data["stdbatch_slices2_msignal"] ))))))) +

                            0.050000*np.tanh(((((data["abs_maxbatch_slices2"]) * (data["medianbatch_slices2"]))) + (np.minimum(((np.minimum(((np.minimum(((((data["meanbatch_slices2"]) * (data["abs_avgbatch_msignal"])))), ((np.minimum(((data["meanbatch_msignal"])), ((((np.tanh((data["signal"]))) * 2.0))))))))), ((np.minimum(((np.cos((((data["rangebatch_msignal"]) * (data["abs_maxbatch_slices2"])))))), ((data["meanbatch_slices2"])))))))), ((data["meanbatch_slices2"])))))) +

                            0.050000*np.tanh(((data["abs_avgbatch_slices2_msignal"]) * (((data["signal"]) + (np.minimum((((((np.sin((data["maxbatch_msignal"]))) <= (data["signal"]))*1.))), ((np.sin((((np.tanh((((data["medianbatch_slices2"]) / 2.0)))) * ((((data["medianbatch_slices2"]) > ((((data["minbatch_slices2_msignal"]) + (data["medianbatch_slices2"]))/2.0)))*1.))))))))))))) +

                            0.050000*np.tanh(np.where(data["abs_minbatch_slices2"] <= -998, np.where(data["abs_avgbatch_slices2"] <= -998, (-((data["mean_abs_chgbatch_slices2_msignal"]))), ((data["rangebatch_slices2"]) * 2.0) ), (-((data["mean_abs_chgbatch_slices2_msignal"]))) )) +

                            0.050000*np.tanh(((((data["minbatch_slices2"]) * (np.cos((((data["signal"]) / 2.0)))))) + (np.cos((data["meanbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(((data["medianbatch_slices2_msignal"]) - ((((((data["abs_minbatch_msignal"]) - (np.tanh((data["maxtominbatch_slices2_msignal"]))))) + (((((data["meanbatch_msignal"]) * 2.0)) * (np.minimum(((data["abs_avgbatch_msignal"])), ((np.minimum(((data["maxbatch_slices2_msignal"])), ((data["abs_avgbatch_msignal"]))))))))))/2.0)))) +

                            0.050000*np.tanh(((((((data["medianbatch_slices2_msignal"]) * 2.0)) - (((data["mean_abs_chgbatch_msignal"]) * ((10.36975574493408203)))))) - (((data["mean_abs_chgbatch_msignal"]) * (data["abs_avgbatch_msignal"]))))) +

                            0.050000*np.tanh(((np.sin((((np.where(data["stdbatch_slices2_msignal"] > -998, np.sin((data["abs_avgbatch_msignal"])), ((data["maxbatch_slices2_msignal"]) * (np.sin((data["signal_shift_-1_msignal"])))) )) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(((((data["meanbatch_slices2"]) * (np.sin((np.where(np.where(data["medianbatch_slices2_msignal"] > -998, data["abs_maxbatch_slices2"], np.maximum(((np.tanh((data["signal"])))), ((data["abs_maxbatch_slices2"]))) ) > -998, (-((data["minbatch_msignal"]))), ((data["minbatch_msignal"]) * 2.0) )))))) - (((data["medianbatch_slices2_msignal"]) * (data["meanbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(np.minimum(((((data["maxbatch_slices2_msignal"]) - ((4.0))))), ((np.maximum(((((np.where(data["mean_abs_chgbatch_msignal"] > -998, ((((((data["maxbatch_slices2_msignal"]) * (((data["maxbatch_slices2_msignal"]) - ((4.0)))))) * 2.0)) * (data["meanbatch_slices2_msignal"])), data["meanbatch_slices2"] )) * (data["mean_abs_chgbatch_msignal"])))), (((((data["meanbatch_slices2_msignal"]) + (((data["signal_shift_+1"]) * 2.0)))/2.0)))))))) +

                            0.050000*np.tanh(((data["abs_maxbatch"]) * (((np.where(((np.where((-((data["abs_maxbatch"]))) > -998, ((data["abs_minbatch_slices2"]) + (data["signal"])), data["minbatch_msignal"] )) + (data["signal"])) <= -998, data["abs_maxbatch"], np.sin((data["abs_maxbatch"])) )) * 2.0)))) +

                            0.050000*np.tanh(np.where((((((12.75586605072021484)) / 2.0)) * 2.0) <= -998, np.minimum(((data["signal_shift_-1"])), ((np.minimum(((data["signal"])), ((np.minimum(((np.where((12.75586605072021484) <= -998, data["signal_shift_+1"], ((np.sin((data["abs_maxbatch_slices2"]))) * 2.0) ))), ((data["minbatch"]))))))))), ((np.sin((data["abs_maxbatch_slices2"]))) * 2.0) )) +

                            0.050000*np.tanh(((np.minimum(((np.maximum(((((np.minimum(((data["abs_avgbatch_msignal"])), ((data["meanbatch_slices2"])))) * 2.0))), ((data["meanbatch_slices2"]))))), ((((data["signal_shift_-1_msignal"]) + (data["abs_avgbatch_slices2_msignal"])))))) + (np.sin(((((data["abs_minbatch_slices2"]) <= (data["minbatch"]))*1.)))))) +

                            0.050000*np.tanh(np.where(data["abs_maxbatch_slices2_msignal"] > -998, np.where((-((data["maxtominbatch_msignal"]))) <= -998, data["abs_avgbatch_msignal"], np.where(((data["abs_avgbatch_slices2_msignal"]) * (data["meanbatch_slices2"])) > -998, data["signal_shift_-1"], np.tanh(((((data["stdbatch_msignal"]) > (data["abs_maxbatch"]))*1.))) ) ), np.sin(((((data["abs_avgbatch_slices2_msignal"]) <= (data["maxtominbatch_slices2_msignal"]))*1.))) )) +

                            0.050000*np.tanh(np.tanh((((np.sin((np.where(data["meanbatch_slices2_msignal"] > -998, data["rangebatch_slices2_msignal"], data["meanbatch_msignal"] )))) * 2.0)))) +

                            0.050000*np.tanh(np.minimum(((data["signal"])), ((((np.cos((np.cos((((np.minimum(((data["abs_minbatch_msignal"])), ((((data["medianbatch_slices2"]) - (data["medianbatch_slices2"])))))) * 2.0)))))) + (((np.minimum(((np.minimum(((data["signal_shift_+1"])), ((np.cos((data["medianbatch_slices2_msignal"]))))))), ((data["rangebatch_slices2_msignal"])))) * 2.0))))))) +

                            0.050000*np.tanh(data["abs_avgbatch_msignal"]) +

                            0.050000*np.tanh(((data["medianbatch_slices2"]) * (np.maximum(((data["abs_minbatch_msignal"])), ((np.where((((data["abs_avgbatch_slices2_msignal"]) + (data["abs_avgbatch_msignal"]))/2.0) > -998, data["abs_avgbatch_msignal"], np.minimum(((data["minbatch_slices2"])), ((data["abs_avgbatch_msignal"]))) ))))))) +

                            0.050000*np.tanh(((data["abs_avgbatch_msignal"]) * (np.where((((data["minbatch_slices2"]) <= (np.cos((data["meanbatch_msignal"]))))*1.) > -998, data["medianbatch_slices2"], (-((((np.where(((data["abs_avgbatch_msignal"]) * (data["meanbatch_slices2"])) <= -998, (-((data["abs_avgbatch_msignal"]))), data["abs_avgbatch_msignal"] )) * ((-((((((data["abs_maxbatch_slices2_msignal"]) - ((-(((-((data["meanbatch_slices2"]))))))))) * 2.0))))))))) )))) +

                            0.050000*np.tanh(((np.sin((np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((((np.sin((np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["rangebatch_slices2_msignal"], np.where(data["abs_minbatch_slices2_msignal"] > -998, data["maxbatch_slices2"], data["signal_shift_+1_msignal"] ) )))) * 2.0))))))) * 2.0)) +

                            0.050000*np.tanh(((np.sin((data["rangebatch_slices2_msignal"]))) * (((np.where(data["meanbatch_slices2_msignal"] <= -998, np.minimum(((data["maxbatch_msignal"])), ((data["maxtominbatch_msignal"]))), data["abs_maxbatch_slices2"] )) + (data["rangebatch_msignal"]))))) +

                            0.050000*np.tanh(np.where(np.where((((-((data["stdbatch_slices2_msignal"])))) * (data["abs_maxbatch_slices2"])) > -998, data["abs_minbatch_slices2_msignal"], data["abs_minbatch_slices2_msignal"] ) > -998, np.sin((data["abs_maxbatch_slices2"])), data["abs_avgbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(((((((np.sin((data["signal"]))) * ((((data["minbatch_slices2_msignal"]) + (data["maxbatch_slices2"]))/2.0)))) + (data["abs_avgbatch_msignal"]))) * ((((-(((-((data["medianbatch_msignal"]))))))) + (data["abs_maxbatch_slices2"]))))) +

                            0.050000*np.tanh(((np.cos((((data["maxbatch_msignal"]) - (np.sin((np.where(((data["maxbatch_msignal"]) * (((data["minbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"])))) > -998, (4.10039758682250977), ((((np.where(data["minbatch_slices2_msignal"] > -998, (4.10039758682250977), ((data["abs_avgbatch_msignal"]) / 2.0) )) * 2.0)) * ((-((np.tanh((data["abs_avgbatch_msignal"]))))))) )))))))) * 2.0)) +

                            0.050000*np.tanh(((((np.sin((data["rangebatch_slices2_msignal"]))) * (np.maximum(((((((((data["signal"]) - (data["rangebatch_slices2_msignal"]))) / 2.0)) - (np.sin((data["rangebatch_slices2_msignal"])))))), ((data["rangebatch_slices2_msignal"])))))) * (np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((np.maximum(((data["maxbatch_msignal"])), ((np.sin((((data["signal"]) - (data["minbatch_msignal"]))))))))))))) +

                            0.050000*np.tanh(((np.where(data["meanbatch_slices2_msignal"] <= -998, data["mean_abs_chgbatch_slices2_msignal"], ((np.where(data["signal"] > -998, data["stdbatch_slices2_msignal"], ((np.where(data["signal"] > -998, data["stdbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] )) * (data["abs_maxbatch_msignal"])) )) * (data["signal_shift_+1"])) )) - (((np.where(data["stdbatch_slices2_msignal"] > -998, data["stdbatch_slices2_msignal"], data["abs_maxbatch_msignal"] )) * (((data["meanbatch_slices2_msignal"]) + (data["abs_maxbatch_msignal"]))))))) +

                            0.050000*np.tanh(((((np.tanh((np.minimum(((data["abs_maxbatch_slices2"])), ((np.cos((np.where(data["abs_maxbatch_slices2"] > -998, ((data["abs_minbatch_slices2_msignal"]) * 2.0), np.tanh((np.minimum(((data["meanbatch_msignal"])), ((np.cos((np.sin((np.cos((np.minimum(((((data["abs_maxbatch_slices2"]) / 2.0))), ((np.cos((data["abs_maxbatch_slices2"])))))))))))))))) ))))))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.where(np.where(data["medianbatch_slices2_msignal"] <= -998, ((np.sin((data["signal_shift_-1"]))) / 2.0), data["signal_shift_-1"] ) <= -998, data["signal_shift_-1"], ((data["abs_avgbatch_slices2_msignal"]) * (data["signal_shift_+1"])) )) +

                            0.050000*np.tanh(np.where(data["abs_maxbatch_msignal"] <= -998, np.maximum(((data["stdbatch_slices2_msignal"])), ((data["minbatch_slices2_msignal"]))), ((data["meanbatch_slices2_msignal"]) + (np.sin((np.sin((data["rangebatch_slices2_msignal"])))))) )) +

                            0.050000*np.tanh(np.sin((np.maximum(((data["minbatch_msignal"])), ((((np.sin((np.maximum((((-((data["minbatch_msignal"]))))), (((-((data["minbatch_msignal"]))))))))) + (np.sin((np.tanh((np.minimum(((data["abs_avgbatch_msignal"])), ((np.sin(((-((data["minbatch_msignal"])))))))))))))))))))) +

                            0.050000*np.tanh(np.minimum(((((data["signal_shift_+1"]) * (((data["abs_minbatch_slices2_msignal"]) * (np.where(data["signal_shift_-1"] <= -998, data["abs_minbatch_slices2_msignal"], np.where(np.where(data["abs_minbatch_slices2_msignal"] <= -998, data["abs_maxbatch_slices2_msignal"], np.cos(((((-((data["meanbatch_slices2"])))) * 2.0))) ) <= -998, data["signal"], np.cos((data["mean_abs_chgbatch_slices2_msignal"])) ) ))))))), ((((data["abs_avgbatch_msignal"]) * 2.0))))) +

                            0.050000*np.tanh(np.minimum(((((data["abs_avgbatch_slices2_msignal"]) * (data["rangebatch_slices2_msignal"])))), (((((((-((data["stdbatch_slices2_msignal"])))) - (np.sin((((((data["abs_avgbatch_slices2_msignal"]) * 2.0)) * 2.0)))))) - (np.sin((np.maximum((((-((data["stdbatch_slices2_msignal"]))))), ((data["meanbatch_slices2"]))))))))))) +

                            0.050000*np.tanh(((data["minbatch_slices2"]) - (((((data["minbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))) * (np.cos((np.where(np.where(np.maximum(((((data["abs_avgbatch_slices2_msignal"]) * 2.0))), ((np.where(data["maxbatch_slices2_msignal"] > -998, data["abs_maxbatch_msignal"], data["stdbatch_slices2"] )))) > -998, data["minbatch_slices2_msignal"], data["maxtominbatch"] ) > -998, data["minbatch_slices2_msignal"], data["minbatch_slices2_msignal"] )))))))) +

                            0.050000*np.tanh(np.where((5.0) <= -998, (((-((np.where(data["maxbatch_msignal"] <= -998, data["maxbatch_msignal"], ((data["rangebatch_msignal"]) / 2.0) ))))) * (np.sin((data["meanbatch_msignal"])))), (-((np.where(data["minbatch_msignal"] <= -998, data["maxbatch_msignal"], ((((data["abs_avgbatch_slices2_msignal"]) * (data["stdbatch_slices2_msignal"]))) * 2.0) )))) )) +

                            0.050000*np.tanh(np.where(data["abs_maxbatch_slices2_msignal"] <= -998, ((data["maxtominbatch_slices2_msignal"]) - ((-((np.sin((data["maxbatch_slices2_msignal"]))))))), ((data["abs_maxbatch_slices2_msignal"]) * ((-((np.sin((np.maximum(((data["maxbatch_msignal"])), (((2.0))))))))))) )) +

                            0.050000*np.tanh(np.where((((data["rangebatch_slices2"]) + (np.maximum(((data["rangebatch_slices2"])), ((data["rangebatch_msignal"])))))/2.0) <= -998, data["mean_abs_chgbatch_slices2"], ((np.cos((data["maxtominbatch_slices2_msignal"]))) * 2.0) )) +

                            0.050000*np.tanh((((-((data["rangebatch_slices2_msignal"])))) + (((data["maxbatch_slices2_msignal"]) + (((data["minbatch_msignal"]) * (np.sin((np.sin((np.where(data["abs_avgbatch_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], (((data["signal_shift_+1"]) > ((-(((((np.tanh((data["minbatch_msignal"]))) + (np.sin((data["signal_shift_+1"]))))/2.0))))))*1.) )))))))))))) +

                            0.050000*np.tanh(np.where(data["maxbatch_msignal"] <= -998, ((data["maxbatch_msignal"]) / 2.0), ((data["maxtominbatch_msignal"]) * (np.cos((np.where(np.where(data["medianbatch_slices2"] <= -998, data["maxtominbatch_msignal"], np.where(np.cos(((-((np.minimum(((data["maxtominbatch_msignal"])), ((data["mean_abs_chgbatch_msignal"])))))))) <= -998, data["maxbatch_slices2_msignal"], ((data["mean_abs_chgbatch_msignal"]) * (((data["medianbatch_slices2"]) / 2.0))) ) ) <= -998, data["maxbatch_msignal"], data["mean_abs_chgbatch_msignal"] ))))) )) +

                            0.050000*np.tanh((-((((data["mean_abs_chgbatch_slices2"]) - (((data["maxtominbatch_slices2_msignal"]) * (np.maximum(((np.sin((data["maxbatch_slices2_msignal"])))), ((np.minimum(((np.minimum(((np.sin((data["mean_abs_chgbatch_slices2_msignal"])))), ((data["mean_abs_chgbatch_slices2"]))))), ((((data["mean_abs_chgbatch_slices2_msignal"]) - (((data["maxtominbatch_slices2_msignal"]) + (data["medianbatch_slices2"])))))))))))))))))) +

                            0.050000*np.tanh(((((np.cos((((np.maximum(((data["signal"])), ((data["signal"])))) * 2.0)))) * 2.0)) * (data["signal"]))) +

                            0.050000*np.tanh((-((((data["mean_abs_chgbatch_slices2_msignal"]) * (((data["abs_avgbatch_msignal"]) - (((((((((((data["abs_avgbatch_msignal"]) * (data["mean_abs_chgbatch_slices2_msignal"]))) - (((data["mean_abs_chgbatch_slices2_msignal"]) * (np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)))))))) - ((((((data["signal_shift_-1_msignal"]) > ((((3.0)) * 2.0)))*1.)) / 2.0)))) / 2.0)) / 2.0))))))))) +

                            0.050000*np.tanh(((data["mean_abs_chgbatch_slices2_msignal"]) * (((data["abs_avgbatch_slices2_msignal"]) * (np.minimum(((((np.minimum(((data["stdbatch_msignal"])), ((data["meanbatch_slices2"])))) * 2.0))), ((((((-((data["abs_maxbatch_slices2_msignal"])))) + ((((np.minimum(((data["maxtominbatch_msignal"])), (((((data["mean_abs_chgbatch_msignal"]) + (data["mean_abs_chgbatch_msignal"]))/2.0))))) <= (np.cos((np.where(data["meanbatch_slices2_msignal"] <= -998, data["maxbatch_msignal"], np.tanh((data["maxtominbatch_msignal"])) )))))*1.)))/2.0))))))))) +

                            0.050000*np.tanh(((((data["maxtominbatch_slices2_msignal"]) * (np.sin((data["minbatch_msignal"]))))) - (np.tanh((((((((data["stdbatch_msignal"]) > (data["minbatch_msignal"]))*1.)) > (np.where(data["abs_maxbatch_slices2_msignal"] <= -998, data["minbatch_msignal"], (-((np.maximum(((data["maxtominbatch_slices2_msignal"])), ((data["abs_avgbatch_slices2_msignal"])))))) )))*1.)))))) +

                            0.050000*np.tanh(((np.where(((((((((np.sin((((data["medianbatch_slices2"]) * 2.0)))) * (data["abs_maxbatch_slices2"]))) * 2.0)) - ((((np.tanh((np.tanh((((data["abs_maxbatch_slices2"]) * ((7.0)))))))) + (data["rangebatch_slices2"]))/2.0)))) * 2.0) > -998, ((data["abs_avgbatch_slices2_msignal"]) * (data["medianbatch_slices2"])), (-((data["maxtominbatch_msignal"]))) )) * 2.0)) +

                            0.050000*np.tanh(((np.where(data["stdbatch_msignal"] > -998, ((((np.where(np.sin((((data["abs_avgbatch_slices2_msignal"]) * 2.0))) <= -998, ((np.cos((data["maxbatch_msignal"]))) * (data["maxbatch_msignal"])), np.sin((data["abs_avgbatch_slices2_msignal"])) )) - (np.sin((data["maxbatch_slices2_msignal"]))))) - ((((data["mean_abs_chgbatch_slices2_msignal"]) > (data["signal_shift_+1_msignal"]))*1.))), np.sin((data["maxbatch_slices2"])) )) * (data["rangebatch_msignal"]))) +

                            0.050000*np.tanh(np.where(np.tanh((data["maxtominbatch"])) <= -998, np.where((((data["maxbatch_slices2_msignal"]) > (data["stdbatch_slices2_msignal"]))*1.) > -998, data["stdbatch_msignal"], data["maxbatch_slices2_msignal"] ), (-((((np.cos((((data["stdbatch_slices2_msignal"]) * 2.0)))) * 2.0)))) )) +

                            0.050000*np.tanh(((np.cos((np.where((((((((data["rangebatch_msignal"]) * 2.0)) > (data["maxtominbatch_slices2"]))*1.)) * (data["medianbatch_slices2"])) <= -998, np.cos((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, data["rangebatch_msignal"], data["medianbatch_slices2"] ))), ((data["abs_minbatch_msignal"]) * 2.0) )))) * (((data["abs_maxbatch_slices2_msignal"]) * 2.0)))) +

                            0.050000*np.tanh(((data["maxtominbatch_msignal"]) * (np.cos((np.where(np.sin((data["maxbatch_slices2_msignal"])) > -998, data["mean_abs_chgbatch_msignal"], data["maxtominbatch_msignal"] )))))) +

                            0.050000*np.tanh(np.sin((np.minimum(((((np.where(((data["medianbatch_slices2_msignal"]) * 2.0) > -998, data["mean_abs_chgbatch_msignal"], (-((data["mean_abs_chgbatch_msignal"]))) )) * 2.0))), ((data["maxtominbatch_msignal"])))))) +

                            0.050000*np.tanh((-((((np.sin((data["minbatch_msignal"]))) * ((((((data["abs_maxbatch_slices2"]) <= ((((-((((np.sin((data["minbatch_msignal"]))) / 2.0))))) - ((-((((((data["abs_minbatch_msignal"]) * 2.0)) + (data["minbatch_msignal"])))))))))*1.)) + (data["abs_maxbatch_slices2"])))))))) +

                            0.050000*np.tanh(((np.sin((((data["abs_minbatch_msignal"]) + ((((((np.sin((np.cos((data["abs_maxbatch"]))))) * 2.0)) + (data["abs_minbatch_msignal"]))/2.0)))))) * (data["rangebatch_slices2_msignal"]))) +

                            0.050000*np.tanh(np.cos((((np.maximum(((data["signal"])), ((np.cos((((np.maximum(((np.cos((data["signal"])))), ((np.where(((np.cos((np.sin((data["signal"]))))) * 2.0) > -998, data["signal"], np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["meanbatch_slices2"], np.cos((np.cos((data["medianbatch_msignal"])))) ) ))))) * 2.0))))))) * 2.0)))) +

                            0.050000*np.tanh(((((np.minimum(((((((((np.minimum(((data["signal_shift_+1"])), (((((((data["rangebatch_slices2"]) + ((((data["maxtominbatch_msignal"]) + (data["stdbatch_slices2_msignal"]))/2.0)))) <= (((data["medianbatch_msignal"]) * 2.0)))*1.))))) * 2.0)) - (data["stdbatch_msignal"]))) * 2.0))), ((((data["medianbatch_msignal"]) * (data["signal_shift_+1"])))))) * 2.0)) - ((((data["mean_abs_chgbatch_msignal"]) + (data["medianbatch_msignal"]))/2.0)))) +

                            0.050000*np.tanh(((np.minimum(((((np.tanh((np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((np.cos((np.maximum(((data["abs_avgbatch_slices2_msignal"])), ((data["medianbatch_slices2_msignal"]))))))))))) * 2.0))), ((((((((((((((data["abs_maxbatch_slices2"]) <= (np.minimum((((((((((data["medianbatch_slices2_msignal"]) / 2.0)) * 2.0)) + (data["maxtominbatch_msignal"]))/2.0))), ((data["minbatch_slices2_msignal"])))))*1.)) / 2.0)) * 2.0)) / 2.0)) > (data["maxtominbatch_msignal"]))*1.))))) * 2.0)) +

                            0.050000*np.tanh(((((data["rangebatch_slices2_msignal"]) * ((-((data["stdbatch_msignal"])))))) * 2.0)) +

                            0.050000*np.tanh(((((data["abs_maxbatch_slices2"]) * (np.sin((np.minimum(((data["maxtominbatch_msignal"])), (((((np.sin((np.sin((((np.sin((data["maxtominbatch_msignal"]))) - (data["abs_maxbatch_slices2"]))))))) > (np.sin((data["maxtominbatch_msignal"]))))*1.))))))))) - (data["abs_maxbatch_slices2"]))) +

                            0.050000*np.tanh(((data["abs_maxbatch_slices2_msignal"]) * (np.cos((((data["abs_minbatch_msignal"]) * 2.0)))))) +

                            0.050000*np.tanh(((data["abs_avgbatch_slices2_msignal"]) * 2.0)) +

                            0.050000*np.tanh(((np.cos((np.minimum((((0.54778587818145752))), ((np.where(data["maxtominbatch_msignal"] > -998, data["maxtominbatch_msignal"], (((data["stdbatch_slices2"]) + (data["abs_avgbatch_slices2_msignal"]))/2.0) ))))))) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((data["signal_shift_-1"])), (((((((-((data["mean_abs_chgbatch_msignal"])))) - (((data["meanbatch_msignal"]) * (np.minimum(((data["minbatch"])), ((np.minimum(((np.sin((((data["abs_avgbatch_slices2_msignal"]) * 2.0))))), ((((data["maxtominbatch_slices2_msignal"]) / 2.0)))))))))))) * 2.0))))) +

                            0.050000*np.tanh(((((np.where(data["minbatch"] <= -998, data["signal"], np.where(data["abs_maxbatch_msignal"] > -998, data["minbatch"], np.where(data["stdbatch_slices2"] > -998, (6.60688304901123047), np.cos((data["maxtominbatch_msignal"])) ) ) )) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.where(((data["signal"]) * (data["abs_avgbatch_slices2_msignal"])) > -998, ((np.sin((((data["abs_minbatch_msignal"]) + ((((data["rangebatch_msignal"]) + ((((data["medianbatch_msignal"]) <= (((data["abs_avgbatch_slices2_msignal"]) * 2.0)))*1.)))/2.0)))))) * 2.0), np.tanh((data["abs_maxbatch_msignal"])) )) +

                            0.050000*np.tanh(np.where(data["minbatch_msignal"] <= -998, (((np.tanh(((((((data["abs_minbatch_slices2_msignal"]) * 2.0)) + (np.minimum(((data["meanbatch_msignal"])), ((data["abs_minbatch_slices2_msignal"])))))/2.0)))) + (data["abs_minbatch_slices2_msignal"]))/2.0), np.sin((np.where(data["abs_minbatch_msignal"] <= -998, data["abs_avgbatch_slices2_msignal"], ((data["maxtominbatch_msignal"]) * 2.0) ))) )) +

                            0.050000*np.tanh(((((data["maxtominbatch_msignal"]) * (np.sin((data["abs_maxbatch_slices2_msignal"]))))) - ((((np.sin((data["medianbatch_msignal"]))) <= (np.maximum(((((data["maxtominbatch_msignal"]) * (data["abs_maxbatch_slices2_msignal"])))), ((data["abs_maxbatch_slices2_msignal"])))))*1.)))) +

                            0.050000*np.tanh(np.sin(((-((np.where(np.where(data["mean_abs_chgbatch_msignal"] > -998, np.where(np.sin((np.sin(((-((np.cos((((data["abs_minbatch_msignal"]) / 2.0)))))))))) <= -998, data["abs_minbatch_msignal"], data["maxbatch_slices2_msignal"] ), data["stdbatch_slices2"] ) <= -998, data["abs_minbatch_msignal"], data["maxbatch_slices2_msignal"] ))))))) +

                            0.050000*np.tanh(((np.maximum(((((data["abs_avgbatch_msignal"]) * (((data["mean_abs_chgbatch_msignal"]) * (data["mean_abs_chgbatch_msignal"])))))), ((np.where(np.cos(((-((data["minbatch_slices2"]))))) > -998, np.minimum(((data["mean_abs_chgbatch_msignal"])), ((data["mean_abs_chgbatch_slices2_msignal"]))), np.where(np.cos((np.cos((data["abs_maxbatch"])))) <= -998, data["signal_shift_-1"], (3.0) ) ))))) + (data["maxtominbatch"]))) +

                            0.050000*np.tanh(((np.sin((data["maxbatch_msignal"]))) * (((data["minbatch"]) - (np.minimum(((data["signal_shift_-1"])), ((np.minimum(((data["medianbatch_slices2_msignal"])), ((np.sin((data["abs_maxbatch"]))))))))))))) +

                            0.050000*np.tanh(np.minimum(((data["abs_avgbatch_msignal"])), ((np.cos((np.where(np.minimum((((((10.31050777435302734)) * (data["abs_avgbatch_slices2_msignal"])))), (((((10.31050777435302734)) * (np.cos((((data["rangebatch_slices2"]) * (data["minbatch_slices2"]))))))))) > -998, data["maxtominbatch_slices2_msignal"], ((data["minbatch"]) - (data["minbatch"])) ))))))) +

                            0.050000*np.tanh(((((np.sin(((((data["mean_abs_chgbatch_slices2_msignal"]) > (np.where(data["stdbatch_msignal"] <= -998, np.sin((((data["abs_maxbatch"]) - (data["mean_abs_chgbatch_slices2_msignal"])))), data["stdbatch_msignal"] )))*1.)))) - (np.sin((data["minbatch_msignal"]))))) * 2.0)) +

                            0.050000*np.tanh(((data["stdbatch_slices2"]) + ((-((np.minimum(((data["medianbatch_slices2_msignal"])), (((-((data["maxtominbatch_msignal"])))))))))))) +

                            0.050000*np.tanh(((np.tanh((np.sin((data["abs_minbatch_slices2_msignal"]))))) - (np.maximum(((data["minbatch"])), ((np.where(data["minbatch"] <= -998, data["medianbatch_slices2_msignal"], ((data["stdbatch_msignal"]) * (data["abs_avgbatch_slices2_msignal"])) ))))))) +

                            0.050000*np.tanh(((np.sin((data["maxbatch_slices2_msignal"]))) * (np.where(np.cos((data["abs_maxbatch_slices2_msignal"])) > -998, data["abs_minbatch_slices2_msignal"], ((((np.sin((np.sin((data["abs_maxbatch_slices2_msignal"]))))) * (np.where(np.cos((((((data["meanbatch_slices2_msignal"]) / 2.0)) / 2.0))) > -998, data["stdbatch_msignal"], data["abs_avgbatch_slices2_msignal"] )))) - (data["signal_shift_+1_msignal"])) )))) +

                            0.050000*np.tanh(((np.cos((data["abs_avgbatch_slices2_msignal"]))) * (np.sin(((-((np.where(data["abs_maxbatch"] <= -998, np.sin((np.sin(((-((np.sin((np.cos((((data["minbatch_slices2"]) * (data["abs_maxbatch"]))))))))))))), data["minbatch_msignal"] ))))))))) +

                            0.050000*np.tanh(((((data["maxtominbatch_msignal"]) + (np.sin((np.cos((((data["maxbatch_slices2"]) * 2.0)))))))) * (np.where(np.cos(((((np.minimum(((data["minbatch_msignal"])), ((data["abs_maxbatch_msignal"])))) > (data["maxtominbatch"]))*1.))) <= -998, data["abs_minbatch_msignal"], data["meanbatch_slices2"] )))) +

                            0.050000*np.tanh(((data["stdbatch_msignal"]) * (data["minbatch"]))) +

                            0.050000*np.tanh(((np.where(data["meanbatch_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], data["minbatch"] )) * (((data["signal_shift_-1"]) + ((-((np.maximum(((np.sin(((((data["stdbatch_slices2_msignal"]) > (np.where(np.where(data["maxbatch_slices2_msignal"] > -998, ((data["mean_abs_chgbatch_slices2_msignal"]) + (np.cos((data["signal_shift_-1"])))), data["abs_avgbatch_slices2_msignal"] ) > -998, data["signal_shift_-1"], data["minbatch"] )))*1.))))), ((((data["abs_avgbatch_slices2_msignal"]) * 2.0)))))))))))) +

                            0.050000*np.tanh(((data["maxtominbatch"]) * (np.cos((((data["maxtominbatch"]) * (np.minimum(((data["stdbatch_slices2_msignal"])), (((-((((np.sin((np.sin((((data["medianbatch_msignal"]) + ((((np.cos((((data["maxtominbatch"]) * (data["maxtominbatch"]))))) > (np.sin((((data["medianbatch_msignal"]) + (data["minbatch_msignal"]))))))*1.)))))))) * (data["mean_abs_chgbatch_slices2_msignal"]))))))))))))))) +

                            0.050000*np.tanh((-((((np.where((9.0) <= -998, np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["minbatch_msignal"], data["stdbatch_slices2_msignal"] ), data["mean_abs_chgbatch_msignal"] )) * (data["maxbatch_msignal"])))))) +

                            0.050000*np.tanh(np.sin((np.where(((data["medianbatch_slices2_msignal"]) - (np.tanh((data["mean_abs_chgbatch_slices2_msignal"])))) > -998, data["abs_avgbatch_msignal"], np.minimum(((np.sin((data["signal_shift_+1"])))), ((data["abs_maxbatch"]))) )))) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) * (((np.sin((data["stdbatch_msignal"]))) + (np.cos((np.where(data["signal_shift_+1_msignal"] > -998, data["mean_abs_chgbatch_slices2"], np.where(np.sin((data["maxbatch_slices2"])) <= -998, data["mean_abs_chgbatch_msignal"], data["rangebatch_slices2"] ) )))))))) +

                            0.050000*np.tanh((-((((data["abs_avgbatch_slices2_msignal"]) * (np.minimum(((((data["abs_maxbatch_slices2_msignal"]) * (np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((data["abs_avgbatch_slices2_msignal"]))))))), ((((data["abs_avgbatch_msignal"]) * ((((np.cos((data["medianbatch_msignal"]))) > (data["mean_abs_chgbatch_slices2_msignal"]))*1.)))))))))))) +

                            0.050000*np.tanh(((np.where(data["medianbatch_slices2"] <= -998, np.where(data["abs_avgbatch_msignal"] <= -998, ((data["stdbatch_slices2_msignal"]) * (data["abs_avgbatch_msignal"])), np.cos((data["minbatch_msignal"])) ), np.minimum((((-((data["stdbatch_msignal"]))))), ((np.sin((data["abs_avgbatch_msignal"]))))) )) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((data["meanbatch_slices2_msignal"])), ((((np.minimum(((data["mean_abs_chgbatch_slices2"])), ((((((((((data["mean_abs_chgbatch_slices2"]) / 2.0)) * 2.0)) * 2.0)) * 2.0))))) / 2.0))))) +

                            0.050000*np.tanh(np.sin((((np.sin((np.where(((((((np.cos((data["maxbatch_slices2"]))) + (data["maxbatch_slices2"]))/2.0)) <= (data["signal_shift_-1_msignal"]))*1.) > -998, (-((data["maxbatch_slices2_msignal"]))), ((data["meanbatch_slices2_msignal"]) * ((0.0))) )))) - (np.where(data["stdbatch_msignal"] <= -998, np.sin(((-((data["medianbatch_slices2"]))))), data["maxbatch_msignal"] )))))) +

                            0.050000*np.tanh(((np.cos((data["medianbatch_msignal"]))) + (np.minimum(((np.sin((np.where(data["maxtominbatch_slices2_msignal"] > -998, data["meanbatch_slices2_msignal"], data["abs_maxbatch_slices2"] ))))), ((((data["abs_avgbatch_msignal"]) * 2.0))))))) +

                            0.050000*np.tanh((((((((data["rangebatch_slices2"]) * (data["abs_avgbatch_msignal"]))) * 2.0)) + (data["minbatch"]))/2.0)) +

                            0.050000*np.tanh(np.minimum(((np.where(data["maxbatch_slices2_msignal"] <= -998, ((np.minimum(((np.sin(((-((data["stdbatch_slices2_msignal"]))))))), ((data["meanbatch_slices2"])))) * 2.0), np.sin(((-((data["abs_minbatch_slices2"]))))) ))), (((((data["rangebatch_slices2"]) > (data["maxbatch_msignal"]))*1.))))) +

                            0.050000*np.tanh(np.where(np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((data["mean_abs_chgbatch_slices2_msignal"]))) > -998, data["signal"], np.where(data["signal_shift_-1_msignal"] <= -998, data["abs_avgbatch_slices2"], ((data["abs_avgbatch_msignal"]) * (data["minbatch_msignal"])) ) )) +

                            0.050000*np.tanh(np.where(np.cos((data["minbatch"])) > -998, data["maxtominbatch"], np.minimum(((data["abs_minbatch_slices2_msignal"])), ((data["signal_shift_-1"]))) )) +

                            0.050000*np.tanh(np.sin((data["abs_minbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((np.where(data["abs_maxbatch"] > -998, data["rangebatch_msignal"], np.where(data["meanbatch_slices2_msignal"] > -998, np.sin((data["signal_shift_-1"])), data["abs_minbatch_slices2_msignal"] ) )) * (np.cos((((np.where(data["abs_minbatch_slices2_msignal"] > -998, data["signal_shift_-1"], data["signal_shift_-1"] )) - (data["abs_minbatch_slices2_msignal"]))))))) +

                            0.050000*np.tanh((((((np.where(((((data["signal_shift_-1"]) + (data["maxbatch_slices2"]))) * (data["stdbatch_slices2_msignal"])) <= -998, (-(((((data["stdbatch_slices2_msignal"]) <= (((data["minbatch"]) + (data["signal_shift_-1"]))))*1.)))), data["minbatch"] )) * 2.0)) + (((data["signal_shift_-1"]) - (((((data["signal_shift_-1"]) + (((data["signal_shift_-1"]) + (data["maxbatch_slices2"]))))) * (data["stdbatch_slices2_msignal"]))))))/2.0)) +

                            0.050000*np.tanh(((np.where(np.where(data["maxbatch_slices2"] <= -998, data["maxtominbatch_msignal"], (((np.where(data["signal_shift_-1"] > -998, np.cos((data["rangebatch_msignal"])), data["abs_avgbatch_msignal"] )) > (np.where(((data["maxtominbatch_msignal"]) * (np.sin((data["maxtominbatch"])))) <= -998, data["rangebatch_msignal"], data["minbatch_slices2_msignal"] )))*1.) ) > -998, np.cos((data["rangebatch_msignal"])), np.cos((data["abs_maxbatch"])) )) * (data["maxtominbatch_msignal"]))) +

                            0.050000*np.tanh(np.sin((np.maximum(((data["minbatch_slices2"])), (((-((np.maximum(((data["minbatch_msignal"])), (((-((((np.cos((data["medianbatch_slices2_msignal"]))) * 2.0)))))))))))))))) +

                            0.050000*np.tanh(np.minimum(((np.cos((data["minbatch_slices2"])))), (((-((data["minbatch_slices2_msignal"]))))))) +

                            0.050000*np.tanh(np.minimum(((np.sin(((1.85122179985046387))))), ((((np.cos((np.maximum(((data["signal_shift_+1"])), ((np.where(np.cos((np.cos((data["abs_avgbatch_msignal"])))) > -998, np.cos((data["rangebatch_slices2_msignal"])), np.where(data["meanbatch_slices2_msignal"] > -998, np.cos((data["abs_maxbatch_msignal"])), data["mean_abs_chgbatch_slices2"] ) ))))))) * (data["maxtominbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh((((-((np.where(data["medianbatch_msignal"] > -998, np.cos((data["signal_shift_-1"])), ((data["signal_shift_-1_msignal"]) * 2.0) ))))) + (((data["signal_shift_-1_msignal"]) * (((data["meanbatch_slices2_msignal"]) * 2.0)))))) +

                            0.050000*np.tanh(np.tanh((np.tanh((data["signal"]))))) +

                            0.050000*np.tanh(np.minimum(((np.minimum(((np.sin((data["medianbatch_msignal"])))), ((np.sin((data["maxbatch_slices2"]))))))), ((np.where((-((data["medianbatch_slices2"]))) > -998, data["abs_avgbatch_slices2_msignal"], ((np.sin((data["abs_avgbatch_slices2_msignal"]))) * 2.0) ))))) +

                            0.050000*np.tanh((((-((data["signal"])))) / 2.0)) +

                            0.050000*np.tanh(np.where(data["signal_shift_+1_msignal"] > -998, np.sin((np.where(np.sin((((data["signal_shift_+1_msignal"]) * 2.0))) > -998, np.sin((np.sin((np.sin((np.sin((data["abs_maxbatch_slices2"])))))))), (((data["maxbatch_msignal"]) > ((((data["abs_maxbatch_slices2_msignal"]) + (np.sin((data["abs_maxbatch_slices2"]))))/2.0)))*1.) ))), data["abs_maxbatch_slices2"] )) +

                            0.050000*np.tanh(((np.where(np.cos((data["maxbatch_slices2_msignal"])) <= -998, np.cos((data["meanbatch_msignal"])), data["signal_shift_-1"] )) * (((np.sin((data["maxtominbatch_msignal"]))) * 2.0)))) +

                            0.050000*np.tanh(((data["minbatch_slices2_msignal"]) + (((np.maximum(((np.where(np.minimum(((data["minbatch_slices2_msignal"])), ((data["minbatch_slices2_msignal"]))) > -998, data["signal"], ((data["signal_shift_-1"]) + (data["signal_shift_+1"])) ))), ((np.tanh((data["maxbatch_slices2_msignal"])))))) * ((((((data["signal_shift_-1"]) <= (np.maximum(((data["abs_maxbatch_msignal"])), ((np.tanh((data["minbatch_slices2_msignal"])))))))*1.)) * (data["signal_shift_+1"]))))))) +

                            0.050000*np.tanh(np.cos((data["abs_avgbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(data["abs_avgbatch_msignal"]) +

                            0.050000*np.tanh(np.where(np.maximum(((np.maximum(((np.maximum(((data["meanbatch_msignal"])), ((data["rangebatch_msignal"]))))), ((np.maximum(((data["abs_maxbatch_msignal"])), ((data["signal"])))))))), ((np.minimum((((10.0))), ((data["maxtominbatch_slices2"])))))) <= -998, data["rangebatch_msignal"], ((((((data["signal_shift_-1_msignal"]) * (data["meanbatch_msignal"]))) * 2.0)) * 2.0) )) +

                            0.050000*np.tanh(((np.where(data["minbatch_slices2"] > -998, data["signal_shift_+1"], ((np.minimum(((data["rangebatch_slices2"])), ((np.sin((data["abs_avgbatch_slices2_msignal"])))))) + ((((((-((data["minbatch"])))) - (((data["abs_minbatch_slices2"]) * 2.0)))) * 2.0))) )) - (data["abs_avgbatch_msignal"]))) +

                            0.050000*np.tanh(((np.cos((np.maximum(((np.sin((np.cos((np.maximum(((data["signal_shift_-1"])), ((np.maximum(((np.maximum(((((((data["signal"]) + (data["signal_shift_-1"]))) + (data["signal_shift_+1"])))), ((data["signal"]))))), ((((data["signal"]) + (data["signal_shift_-1"])))))))))))))), ((((data["signal"]) + (data["signal_shift_-1"])))))))) * 2.0)) +

                            0.050000*np.tanh(np.where((((data["stdbatch_msignal"]) <= (data["abs_maxbatch"]))*1.) <= -998, ((((data["stdbatch_msignal"]) * 2.0)) * 2.0), ((((data["medianbatch_msignal"]) * (data["signal_shift_-1_msignal"]))) * (np.maximum(((data["medianbatch_msignal"])), ((((np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["signal_shift_-1_msignal"], (-(((-((((data["maxbatch_slices2"]) * ((-(((-((data["medianbatch_slices2"])))))))))))))) )) / 2.0)))))) )) +

                            0.050000*np.tanh((((-((np.sin((np.sin((np.where(data["stdbatch_slices2_msignal"] > -998, data["abs_maxbatch_msignal"], ((np.cos(((((-((data["abs_maxbatch_msignal"])))) * 2.0)))) + ((((-((np.sin((data["abs_avgbatch_msignal"])))))) * 2.0))) ))))))))) * 2.0)) +

                            0.050000*np.tanh(((data["meanbatch_msignal"]) - (((((data["maxbatch_msignal"]) * (((np.where(data["abs_avgbatch_slices2_msignal"] <= -998, np.minimum(((data["signal_shift_-1_msignal"])), ((np.where(data["abs_avgbatch_slices2_msignal"] <= -998, data["meanbatch_msignal"], np.minimum(((data["medianbatch_slices2_msignal"])), ((data["abs_minbatch_msignal"]))) )))), np.sin((data["abs_minbatch_msignal"])) )) * (data["signal_shift_+1_msignal"]))))) * 2.0)))) +

                            0.050000*np.tanh(np.sin((((data["stdbatch_slices2_msignal"]) - (np.maximum(((np.maximum(((np.maximum(((data["signal_shift_+1"])), ((np.sin((np.maximum(((data["stdbatch_slices2_msignal"])), ((np.maximum(((np.maximum(((data["signal_shift_+1"])), ((np.sin((np.sin((data["stdbatch_slices2_msignal"]))))))))), ((data["signal_shift_+1"]))))))))))))), ((data["minbatch_slices2"]))))), ((np.sin((data["stdbatch_slices2_msignal"])))))))))) +

                            0.050000*np.tanh(np.sin((np.where(data["maxbatch_slices2_msignal"] > -998, (((-((data["rangebatch_msignal"])))) - (data["signal_shift_-1"])), np.minimum(((data["abs_maxbatch_msignal"])), ((((data["maxbatch_slices2_msignal"]) - (np.where(data["maxtominbatch_slices2_msignal"] > -998, data["maxtominbatch_slices2_msignal"], ((data["maxbatch_slices2_msignal"]) - (data["abs_maxbatch_msignal"])) )))))) )))) +

                            0.050000*np.tanh(np.sin((np.where(((np.where(np.tanh((data["abs_maxbatch_msignal"])) <= -998, data["mean_abs_chgbatch_msignal"], ((data["signal_shift_-1"]) - (((data["abs_maxbatch_msignal"]) * 2.0))) )) * 2.0) <= -998, data["mean_abs_chgbatch_msignal"], ((data["abs_maxbatch_msignal"]) - (data["signal_shift_-1"])) )))) +

                            0.050000*np.tanh(((((data["signal_shift_+1"]) * 2.0)) * (((data["medianbatch_msignal"]) * (np.where(data["medianbatch_msignal"] > -998, data["signal_shift_-1_msignal"], np.where(data["meanbatch_slices2"] > -998, data["signal_shift_+1"], ((data["medianbatch_msignal"]) * (np.where((((data["abs_minbatch_slices2_msignal"]) + (data["rangebatch_slices2_msignal"]))/2.0) > -998, data["signal_shift_-1"], (((data["medianbatch_msignal"]) <= ((((data["medianbatch_msignal"]) + (data["abs_maxbatch"]))/2.0)))*1.) ))) ) )))))) +

                            0.050000*np.tanh(((data["stdbatch_msignal"]) * (np.where(((data["stdbatch_msignal"]) * (((data["abs_avgbatch_msignal"]) * (data["stdbatch_slices2_msignal"])))) <= -998, data["abs_avgbatch_slices2_msignal"], data["minbatch_slices2"] )))) +

                            0.050000*np.tanh(((((data["stdbatch_msignal"]) * (((data["signal_shift_+1"]) - (((((((data["abs_maxbatch_slices2_msignal"]) + (data["medianbatch_msignal"]))) + (data["stdbatch_slices2_msignal"]))) + (data["medianbatch_slices2"]))))))) - (np.tanh((np.minimum((((((((data["meanbatch_slices2"]) + (data["abs_maxbatch_slices2_msignal"]))/2.0)) - ((10.77968311309814453))))), ((data["minbatch"])))))))) +

                            0.050000*np.tanh(((((np.where(data["stdbatch_msignal"] <= -998, data["abs_avgbatch_slices2"], data["maxbatch_slices2_msignal"] )) * (((np.sin(((-((data["maxbatch_slices2_msignal"])))))) * ((((np.sin((data["maxbatch_slices2_msignal"]))) <= (data["signal_shift_-1"]))*1.)))))) + (np.where(data["maxtominbatch_slices2"] > -998, ((np.sin((data["maxbatch_slices2_msignal"]))) * (((data["minbatch_slices2"]) + (data["maxtominbatch_msignal"])))), data["abs_maxbatch_slices2"] )))) +

                            0.050000*np.tanh(np.where(data["abs_avgbatch_slices2_msignal"] <= -998, data["minbatch_slices2"], np.minimum(((np.where((((((-((data["mean_abs_chgbatch_msignal"])))) - (data["abs_minbatch_slices2_msignal"]))) / 2.0) <= -998, np.maximum(((np.sin((((np.sin((data["maxtominbatch_msignal"]))) * 2.0))))), ((np.sin((data["minbatch_slices2"]))))), np.minimum(((data["signal_shift_+1"])), (((-((np.sin((data["minbatch_msignal"])))))))) ))), ((data["stdbatch_slices2"]))) )) +

                            0.050000*np.tanh(np.minimum(((np.where(data["signal_shift_+1_msignal"] <= -998, np.sin((np.cos(((((data["signal_shift_-1"]) <= (data["mean_abs_chgbatch_slices2"]))*1.))))), (((data["maxtominbatch_slices2"]) > (data["abs_avgbatch_msignal"]))*1.) ))), ((data["signal_shift_+1"])))) +

                            0.050000*np.tanh(np.sin((((np.sin((data["abs_maxbatch"]))) / 2.0)))) +

                            0.050000*np.tanh(((data["signal_shift_-1_msignal"]) * (np.where((((np.tanh((data["signal"]))) <= (((data["abs_avgbatch_slices2_msignal"]) * 2.0)))*1.) <= -998, np.tanh((data["meanbatch_msignal"])), np.minimum(((data["signal"])), ((data["meanbatch_msignal"]))) )))) +

                            0.050000*np.tanh(((data["signal_shift_+1_msignal"]) * (np.minimum(((np.where(((data["maxtominbatch_msignal"]) + (data["abs_avgbatch_slices2_msignal"])) > -998, np.cos((data["stdbatch_slices2_msignal"])), data["meanbatch_slices2_msignal"] ))), ((((data["signal_shift_+1"]) * (np.cos((np.cos((data["stdbatch_slices2_msignal"])))))))))))) +

                            0.050000*np.tanh(np.sin((np.where(data["minbatch_msignal"] > -998, np.sin(((-((((np.sin((((((7.0)) + (data["medianbatch_slices2_msignal"]))/2.0)))) / 2.0)))))), np.minimum(((data["maxbatch_slices2"])), ((np.sin((data["abs_avgbatch_slices2_msignal"]))))) )))) +

                            0.050000*np.tanh(np.tanh((np.minimum(((data["medianbatch_msignal"])), ((np.where((-((data["abs_minbatch_msignal"]))) <= -998, data["abs_maxbatch_msignal"], np.minimum(((np.cos((((np.minimum(((data["signal_shift_-1_msignal"])), ((np.maximum(((data["signal_shift_-1"])), (((-((data["signal_shift_+1"])))))))))) * (data["abs_maxbatch"])))))), ((np.sin((data["meanbatch_msignal"]))))) ))))))) +

                            0.050000*np.tanh(np.minimum(((((np.minimum(((data["minbatch"])), ((np.sin((np.maximum(((data["mean_abs_chgbatch_msignal"])), ((data["abs_maxbatch"]))))))))) - (data["maxtominbatch_msignal"])))), ((np.cos((((((np.sin((np.sin((((data["mean_abs_chgbatch_msignal"]) + (np.where(data["stdbatch_slices2"] <= -998, data["maxtominbatch_msignal"], data["mean_abs_chgbatch_msignal"] )))))))) * 2.0)) - (data["maxtominbatch_msignal"])))))))) +

                            0.050000*np.tanh(np.cos((np.minimum(((data["maxtominbatch_msignal"])), ((((data["rangebatch_slices2_msignal"]) - (np.cos((data["abs_avgbatch_msignal"])))))))))) +

                            0.050000*np.tanh(data["meanbatch_msignal"]) +

                            0.050000*np.tanh(((((data["abs_maxbatch_slices2_msignal"]) * (np.cos((np.where(np.cos((data["signal_shift_-1_msignal"])) > -998, data["abs_maxbatch_slices2_msignal"], ((data["minbatch"]) * (data["signal_shift_-1_msignal"])) )))))) + (((data["signal_shift_-1_msignal"]) * ((((5.82869577407836914)) * (np.cos((np.where(np.tanh((data["medianbatch_msignal"])) > -998, data["mean_abs_chgbatch_msignal"], np.tanh((data["maxbatch_slices2_msignal"])) )))))))))) +

                            0.050000*np.tanh(((np.where(data["maxbatch_slices2"] > -998, np.sin((data["maxbatch_slices2"])), ((((np.where((6.96738290786743164) > -998, data["maxbatch_slices2"], ((data["maxbatch_slices2"]) - (np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, np.sin((data["maxbatch_slices2"])), (8.34637451171875000) ))) )) * 2.0)) - (np.where(data["abs_avgbatch_msignal"] > -998, data["abs_avgbatch_msignal"], (0.0) ))) )) * 2.0)) +

                            0.050000*np.tanh(((data["abs_maxbatch"]) - (np.where(data["abs_maxbatch"] > -998, ((((13.51512145996093750)) + (np.sin(((((data["abs_maxbatch"]) + (data["maxtominbatch"]))/2.0)))))/2.0), data["medianbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(np.where(data["signal_shift_-1"] <= -998, ((data["abs_avgbatch_msignal"]) - (data["rangebatch_slices2"])), np.minimum(((np.cos((data["meanbatch_msignal"])))), ((((data["meanbatch_msignal"]) + (((((data["signal_shift_+1_msignal"]) + (np.maximum(((np.sin((data["abs_avgbatch_msignal"])))), ((np.cos((data["meanbatch_msignal"])))))))) * (np.tanh((data["meanbatch_msignal"]))))))))) )) +

                            0.050000*np.tanh(((data["signal_shift_-1_msignal"]) * (((np.cos((np.where(((data["abs_maxbatch_slices2_msignal"]) * 2.0) <= -998, data["abs_maxbatch_slices2_msignal"], np.where(np.minimum(((np.maximum(((data["abs_minbatch_slices2"])), ((data["signal_shift_-1_msignal"]))))), ((np.sin((np.sin((data["rangebatch_msignal"]))))))) > -998, data["abs_maxbatch_slices2_msignal"], data["abs_maxbatch_slices2_msignal"] ) )))) * (((data["maxbatch_msignal"]) / 2.0)))))) +

                            0.050000*np.tanh(((((data["signal_shift_-1_msignal"]) * (np.where(((data["signal_shift_-1_msignal"]) / 2.0) > -998, data["meanbatch_msignal"], ((np.where(np.cos((data["signal_shift_+1"])) > -998, data["meanbatch_msignal"], ((data["signal_shift_-1_msignal"]) * (data["maxtominbatch_slices2_msignal"])) )) * (data["abs_minbatch_slices2"])) )))) * (data["signal_shift_+1"]))) +

                            0.050000*np.tanh(np.minimum(((((data["medianbatch_msignal"]) + (np.maximum(((np.minimum(((data["stdbatch_slices2"])), (((-((data["abs_minbatch_slices2"])))))))), ((data["abs_avgbatch_slices2_msignal"]))))))), ((data["signal_shift_+1"])))) +

                            0.050000*np.tanh(np.where(np.maximum(((np.cos(((-((data["maxtominbatch_slices2"]))))))), ((np.where((((data["minbatch_msignal"]) > (data["maxbatch_slices2_msignal"]))*1.) > -998, data["stdbatch_slices2"], np.where(data["signal_shift_+1"] <= -998, ((data["maxbatch_slices2"]) / 2.0), data["rangebatch_slices2"] ) )))) <= -998, np.tanh((data["meanbatch_slices2_msignal"])), np.sin((((data["medianbatch_slices2"]) - (((data["abs_minbatch_slices2_msignal"]) - (data["maxbatch_slices2"])))))) )) +

                            0.050000*np.tanh(np.minimum(((((np.cos((data["maxtominbatch_msignal"]))) + (data["meanbatch_msignal"])))), ((((np.sin((np.sin((data["maxtominbatch_msignal"]))))) + (np.minimum(((((np.minimum(((np.cos((data["mean_abs_chgbatch_slices2"])))), ((data["mean_abs_chgbatch_slices2"])))) / 2.0))), ((np.sin(((((data["minbatch_msignal"]) + (np.sin((data["minbatch_msignal"]))))/2.0)))))))))))) +

                            0.050000*np.tanh(np.tanh((((np.maximum(((data["meanbatch_slices2_msignal"])), ((data["medianbatch_slices2"])))) * (np.sin(((((data["signal"]) + (data["abs_minbatch_slices2"]))/2.0)))))))) +

                            0.050000*np.tanh(((data["meanbatch_msignal"]) * (((np.where(data["signal_shift_+1"] > -998, ((((np.where(data["meanbatch_msignal"] > -998, np.tanh((data["signal_shift_+1_msignal"])), data["medianbatch_slices2_msignal"] )) * 2.0)) / 2.0), data["mean_abs_chgbatch_msignal"] )) * 2.0)))) +

                            0.050000*np.tanh(((np.minimum(((((data["signal_shift_+1"]) * 2.0))), ((np.sin(((((data["meanbatch_slices2_msignal"]) <= (data["signal_shift_+1"]))*1.))))))) + (data["abs_maxbatch"]))) +

                            0.050000*np.tanh(np.where((((np.sin((np.sin((np.tanh((data["abs_avgbatch_slices2"]))))))) + (data["rangebatch_slices2_msignal"]))/2.0) > -998, np.sin((np.sin((np.maximum(((data["abs_maxbatch"])), ((data["medianbatch_slices2_msignal"]))))))), data["minbatch_slices2"] )) +

                            0.050000*np.tanh(((data["abs_maxbatch_msignal"]) * (np.cos((np.where(data["stdbatch_slices2"] > -998, data["minbatch_slices2_msignal"], data["mean_abs_chgbatch_slices2_msignal"] )))))) +

                            0.050000*np.tanh(np.minimum(((((data["meanbatch_msignal"]) + (data["medianbatch_slices2"])))), ((((data["stdbatch_slices2_msignal"]) * (((data["maxtominbatch_slices2_msignal"]) - ((((data["abs_maxbatch_slices2_msignal"]) + ((((((np.sin((data["meanbatch_slices2"]))) + (np.minimum(((data["signal"])), ((data["abs_avgbatch_msignal"])))))) + (data["abs_maxbatch_slices2_msignal"]))/2.0)))/2.0))))))))) +

                            0.050000*np.tanh(np.minimum(((np.minimum(((data["maxbatch_slices2_msignal"])), ((((np.minimum(((((((np.minimum(((data["mean_abs_chgbatch_slices2"])), (((((data["abs_avgbatch_slices2"]) > (data["maxtominbatch_slices2_msignal"]))*1.))))) * 2.0)) / 2.0))), ((((data["signal_shift_-1"]) * 2.0))))) / 2.0)))))), ((np.tanh((np.minimum(((np.minimum(((((data["mean_abs_chgbatch_slices2"]) * 2.0))), (((((data["minbatch"]) > (data["abs_avgbatch_msignal"]))*1.)))))), ((np.cos((data["maxtominbatch_slices2_msignal"]))))))))))) +

                            0.050000*np.tanh(np.sin((np.sin((np.maximum(((data["abs_maxbatch_slices2"])), (((((np.maximum(((data["abs_maxbatch_slices2"])), ((data["abs_maxbatch_slices2"])))) <= (np.maximum(((np.maximum(((data["abs_maxbatch"])), ((data["abs_maxbatch_slices2"]))))), ((np.maximum(((data["abs_maxbatch_slices2"])), ((data["abs_maxbatch_slices2"]))))))))*1.))))))))) +

                            0.050000*np.tanh(np.sin((((data["abs_minbatch_slices2_msignal"]) - (((np.maximum((((((np.sin((((data["abs_minbatch_slices2_msignal"]) - (data["abs_minbatch_slices2_msignal"]))))) + (data["maxbatch_slices2"]))/2.0))), (((((0.11005285382270813)) / 2.0))))) * 2.0)))))) +

                            0.050000*np.tanh(np.where(data["abs_maxbatch_slices2"] > -998, np.sin((((data["abs_maxbatch_slices2"]) * 2.0))), data["abs_maxbatch_slices2"] )) +

                            0.050000*np.tanh(np.where(np.minimum(((data["meanbatch_msignal"])), ((data["maxtominbatch_slices2_msignal"]))) <= -998, data["meanbatch_slices2"], np.where(data["abs_avgbatch_msignal"] > -998, ((np.sin((((data["abs_maxbatch_slices2_msignal"]) * (data["signal"]))))) * (data["signal_shift_-1"])), data["maxtominbatch_slices2_msignal"] ) )) +

                            0.050000*np.tanh(np.cos((((data["rangebatch_slices2_msignal"]) + ((((((data["abs_maxbatch_msignal"]) + (np.maximum(((data["signal"])), ((np.minimum(((data["meanbatch_slices2_msignal"])), ((np.where(np.cos((np.cos((data["signal_shift_-1"])))) <= -998, ((data["maxbatch_slices2"]) * 2.0), data["abs_avgbatch_slices2"] )))))))))/2.0)) * (data["medianbatch_slices2"]))))))) +

                            0.050000*np.tanh(((np.where((((np.where(data["medianbatch_msignal"] <= -998, (((data["stdbatch_slices2"]) > (data["rangebatch_slices2"]))*1.), data["medianbatch_msignal"] )) > (data["maxbatch_slices2_msignal"]))*1.) <= -998, data["minbatch_slices2_msignal"], data["medianbatch_msignal"] )) - (((data["medianbatch_msignal"]) + (((np.where((((data["medianbatch_msignal"]) + (data["medianbatch_msignal"]))/2.0) <= -998, data["abs_avgbatch_slices2"], data["medianbatch_msignal"] )) * (data["stdbatch_msignal"]))))))) +

                            0.050000*np.tanh((-((((data["maxtominbatch_msignal"]) - (np.where((((data["signal_shift_+1"]) > (((np.minimum(((data["signal_shift_+1"])), ((((data["maxtominbatch_msignal"]) / 2.0))))) / 2.0)))*1.) > -998, data["minbatch"], (((((data["minbatch"]) / 2.0)) <= (((data["maxtominbatch_msignal"]) - ((-((((data["minbatch"]) - (data["signal_shift_+1"])))))))))*1.) ))))))) +

                            0.050000*np.tanh(np.sin((((data["signal_shift_+1"]) * 2.0)))) +

                            0.050000*np.tanh(((np.sin((((data["abs_maxbatch"]) * ((((data["medianbatch_slices2_msignal"]) <= (data["abs_maxbatch"]))*1.)))))) * 2.0)) +

                            0.050000*np.tanh(((np.where(data["mean_abs_chgbatch_msignal"] <= -998, (-((data["meanbatch_msignal"]))), data["signal_shift_-1_msignal"] )) / 2.0)) +

                            0.050000*np.tanh((((((((0.0)) <= (data["abs_maxbatch_msignal"]))*1.)) > ((-((data["meanbatch_msignal"])))))*1.)) +

                            0.050000*np.tanh(np.cos(((-((((data["abs_maxbatch_slices2_msignal"]) + (((data["meanbatch_msignal"]) * (np.maximum(((data["abs_maxbatch_msignal"])), ((np.sin((np.minimum(((((data["minbatch_slices2"]) + (data["abs_maxbatch_slices2_msignal"])))), ((data["meanbatch_msignal"])))))))))))))))))) +

                            0.050000*np.tanh(data["medianbatch_msignal"]) +

                            0.050000*np.tanh(np.cos((np.where(data["meanbatch_slices2_msignal"] > -998, data["maxtominbatch_msignal"], np.where(data["minbatch_slices2"] <= -998, np.where(data["abs_maxbatch"] <= -998, data["minbatch_slices2"], np.cos(((((np.cos((data["abs_minbatch_slices2"]))) <= (data["maxtominbatch_msignal"]))*1.))) ), data["signal_shift_-1_msignal"] ) )))) +

                            0.050000*np.tanh(np.minimum(((data["abs_avgbatch_msignal"])), ((np.sin((np.where(data["maxtominbatch_msignal"] <= -998, ((np.where(data["meanbatch_slices2"] > -998, np.sin((data["maxtominbatch_msignal"])), np.sin((data["abs_avgbatch_msignal"])) )) * 2.0), data["abs_avgbatch_msignal"] ))))))) +

                            0.050000*np.tanh(np.sin((np.where(data["medianbatch_slices2"] <= -998, data["meanbatch_slices2"], data["maxtominbatch_msignal"] )))) +

                            0.050000*np.tanh(((((((np.cos((((np.where(data["abs_minbatch_msignal"] <= -998, data["signal_shift_+1_msignal"], data["meanbatch_msignal"] )) * (data["minbatch_msignal"]))))) * (np.where(data["abs_minbatch_msignal"] > -998, data["maxtominbatch_slices2_msignal"], (-((data["abs_minbatch_msignal"]))) )))) * 2.0)) * (data["minbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) * (np.cos(((((((data["maxtominbatch_msignal"]) * 2.0)) + (np.where(((data["meanbatch_slices2"]) + (data["maxtominbatch_msignal"])) > -998, data["minbatch_slices2"], (6.0) )))/2.0)))))) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) * (np.sin((np.maximum(((data["maxbatch_slices2_msignal"])), ((np.sin(((((np.cos((data["maxbatch_slices2"]))) + (np.maximum((((((((data["minbatch_msignal"]) <= (data["minbatch_msignal"]))*1.)) * (data["maxbatch_msignal"])))), ((data["abs_avgbatch_slices2"])))))/2.0))))))))))) +

                            0.050000*np.tanh((-((((np.minimum(((data["abs_minbatch_slices2"])), (((-((((data["abs_avgbatch_slices2_msignal"]) / 2.0)))))))) / 2.0))))) +

                            0.050000*np.tanh(np.minimum(((data["abs_avgbatch_msignal"])), ((data["abs_avgbatch_msignal"])))) +

                            0.050000*np.tanh(np.where(((np.tanh((data["signal"]))) - (np.where(data["abs_minbatch_slices2"] <= -998, (((data["mean_abs_chgbatch_slices2"]) > ((((data["medianbatch_slices2_msignal"]) > (np.sin((data["maxbatch_slices2"]))))*1.)))*1.), data["meanbatch_msignal"] ))) > -998, np.sin((data["maxbatch_slices2"])), np.where(data["maxbatch_slices2"] > -998, np.sin((data["maxbatch_slices2"])), np.cos((data["abs_maxbatch_msignal"])) ) )) +

                            0.050000*np.tanh(np.maximum(((((np.sin((data["minbatch_slices2"]))) + (((data["abs_minbatch_msignal"]) * 2.0))))), ((np.where(data["meanbatch_slices2_msignal"] > -998, np.where(data["maxtominbatch_slices2"] > -998, data["meanbatch_slices2_msignal"], (11.69006729125976562) ), np.sin((((np.where(data["abs_maxbatch_msignal"] > -998, data["meanbatch_msignal"], data["maxtominbatch"] )) * (np.sin((np.sin((data["abs_minbatch_msignal"])))))))) ))))) +

                            0.050000*np.tanh(np.cos((((np.where(((data["abs_maxbatch"]) - (np.maximum(((data["rangebatch_slices2_msignal"])), ((data["mean_abs_chgbatch_slices2"]))))) <= -998, data["abs_maxbatch"], data["abs_maxbatch"] )) - ((((((data["abs_minbatch_slices2"]) / 2.0)) > (data["maxtominbatch_msignal"]))*1.)))))) +

                            0.050000*np.tanh(((np.minimum(((np.cos((data["maxbatch_slices2_msignal"])))), ((((data["meanbatch_slices2_msignal"]) * (((((data["meanbatch_slices2_msignal"]) * (((data["signal_shift_-1"]) + (np.tanh((((((data["meanbatch_slices2_msignal"]) / 2.0)) * 2.0)))))))) * ((((10.02487373352050781)) * 2.0))))))))) * (np.cos((data["maxbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(np.where((((((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((data["abs_minbatch_msignal"])))) + (np.sin((data["abs_avgbatch_msignal"]))))/2.0)) + (data["mean_abs_chgbatch_slices2_msignal"])) <= -998, data["abs_maxbatch_slices2"], np.cos((((data["meanbatch_slices2"]) * (data["abs_avgbatch_msignal"])))) )) +

                            0.050000*np.tanh(np.cos((np.minimum(((((np.sin((np.tanh((data["abs_avgbatch_msignal"]))))) * (data["minbatch_slices2_msignal"])))), ((((data["minbatch"]) * (((np.minimum(((np.cos((np.minimum(((np.where(data["abs_maxbatch"] <= -998, data["mean_abs_chgbatch_slices2_msignal"], data["minbatch"] ))), ((data["medianbatch_msignal"]))))))), ((data["abs_avgbatch_slices2_msignal"])))) * (data["abs_avgbatch_slices2_msignal"])))))))))) +

                            0.050000*np.tanh(np.sin((np.where(np.minimum((((((data["abs_maxbatch_slices2"]) > (data["minbatch_slices2"]))*1.))), ((((data["abs_avgbatch_slices2"]) / 2.0)))) > -998, data["abs_maxbatch"], np.cos((np.sin((data["signal_shift_-1"])))) )))) +

                            0.050000*np.tanh(np.sin((np.maximum(((np.minimum((((((data["signal"]) > (data["signal"]))*1.))), ((np.minimum(((np.sin((data["abs_avgbatch_msignal"])))), ((np.minimum(((np.cos((np.sin((data["abs_minbatch_slices2_msignal"])))))), ((data["meanbatch_slices2_msignal"]))))))))))), ((np.sin((data["abs_avgbatch_msignal"])))))))) +

                            0.050000*np.tanh(np.where(((data["abs_avgbatch_msignal"]) / 2.0) <= -998, ((data["meanbatch_slices2_msignal"]) * 2.0), np.minimum(((np.tanh((((data["abs_avgbatch_msignal"]) / 2.0))))), ((data["signal_shift_+1"]))) )) +

                            0.050000*np.tanh(((data["minbatch"]) * 2.0)) +

                            0.050000*np.tanh(np.where(data["signal_shift_+1_msignal"] > -998, (((data["abs_avgbatch_msignal"]) + (((((data["abs_avgbatch_slices2_msignal"]) / 2.0)) / 2.0)))/2.0), np.sin((data["abs_avgbatch_msignal"])) )) +

                            0.050000*np.tanh(np.cos((np.where(data["rangebatch_slices2_msignal"] <= -998, np.cos((data["medianbatch_slices2_msignal"])), np.where(((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0) > -998, data["signal_shift_+1_msignal"], data["abs_minbatch_msignal"] ) )))) +

                            0.050000*np.tanh(np.where(data["signal"] <= -998, (((data["mean_abs_chgbatch_slices2_msignal"]) <= (data["signal"]))*1.), ((data["signal"]) * 2.0) )) +

                            0.050000*np.tanh(((np.cos((np.minimum(((np.where((((6.0)) * 2.0) <= -998, data["mean_abs_chgbatch_msignal"], data["medianbatch_slices2_msignal"] ))), ((((data["mean_abs_chgbatch_msignal"]) * ((6.0))))))))) * 2.0)) +

                            0.050000*np.tanh((((((((np.sin((np.where(np.tanh((data["abs_maxbatch"])) <= -998, ((((data["mean_abs_chgbatch_slices2"]) - (data["signal_shift_-1"]))) * 2.0), data["abs_maxbatch_slices2_msignal"] )))) - ((-(((((data["abs_maxbatch"]) > (((data["medianbatch_msignal"]) * 2.0)))*1.))))))) / 2.0)) <= ((0.83761829137802124)))*1.)) +

                            0.050000*np.tanh(np.sin((((data["abs_maxbatch_slices2"]) - (np.where(((np.where(data["abs_maxbatch_slices2"] > -998, data["medianbatch_msignal"], (5.0) )) - (((data["abs_maxbatch_slices2"]) - (data["abs_avgbatch_slices2_msignal"])))) > -998, np.where(np.sin((data["meanbatch_msignal"])) > -998, data["meanbatch_msignal"], data["abs_avgbatch_slices2_msignal"] ), data["abs_maxbatch_slices2_msignal"] )))))) +

                            0.050000*np.tanh(np.minimum(((data["signal_shift_-1"])), ((np.where(data["signal_shift_-1"] > -998, data["abs_avgbatch_slices2_msignal"], data["rangebatch_slices2"] ))))) +

                            0.050000*np.tanh((((((data["rangebatch_slices2_msignal"]) / 2.0)) <= (data["abs_maxbatch_msignal"]))*1.)) +

                            0.050000*np.tanh((((((np.sin(((((((-((np.minimum(((np.maximum(((data["signal_shift_-1_msignal"])), (((((data["maxbatch_slices2_msignal"]) <= (data["maxtominbatch_msignal"]))*1.)))))), ((data["maxbatch_slices2"]))))))) * 2.0)) * 2.0)))) <= (np.maximum(((data["maxtominbatch"])), ((data["meanbatch_slices2_msignal"])))))*1.)) * 2.0)))   

        

    def GP_class_8(self,data):

        return self.Output( -3.014039 +

                            0.050000*np.tanh(((np.cos((np.where(data["mean_abs_chgbatch_slices2"] > -998, data["abs_maxbatch_slices2_msignal"], (((np.where(np.cos((data["signal_shift_+1"])) > -998, data["signal_shift_+1"], data["signal_shift_-1"] )) <= (((data["medianbatch_slices2"]) * 2.0)))*1.) )))) + ((-((data["stdbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(((((np.minimum(((np.maximum(((((np.sin(((((data["mean_abs_chgbatch_slices2"]) <= (((data["signal"]) * (data["rangebatch_slices2_msignal"]))))*1.)))) * 2.0))), ((np.sin((data["maxbatch_slices2_msignal"]))))))), ((np.minimum(((((data["maxtominbatch_slices2"]) / 2.0))), ((data["minbatch"]))))))) + (np.minimum(((np.tanh((data["minbatch_msignal"])))), ((data["stdbatch_msignal"])))))) - (((data["stdbatch_msignal"]) * (data["rangebatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(np.where(data["abs_minbatch_slices2"] <= -998, (-((((data["signal_shift_-1"]) + (data["medianbatch_slices2"]))))), ((data["signal"]) / 2.0) )) +

                            0.050000*np.tanh(data["medianbatch_slices2"]) +

                            0.050000*np.tanh((((-((data["stdbatch_msignal"])))) - ((((((((((-((data["stdbatch_msignal"])))) + (data["signal"]))) <= (((data["abs_maxbatch_slices2_msignal"]) - (data["abs_maxbatch_msignal"]))))*1.)) > (data["mean_abs_chgbatch_slices2_msignal"]))*1.)))) +

                            0.050000*np.tanh(((data["maxtominbatch"]) + (np.where(data["medianbatch_slices2"] > -998, data["signal"], ((np.where((((((data["medianbatch_slices2_msignal"]) / 2.0)) <= (np.tanh((data["stdbatch_slices2"]))))*1.) > -998, data["medianbatch_slices2"], np.where(data["medianbatch_slices2"] <= -998, data["stdbatch_slices2"], data["medianbatch_slices2"] ) )) * 2.0) )))) +

                            0.050000*np.tanh(np.cos((np.where(np.where(data["abs_maxbatch_slices2"] > -998, (((data["minbatch_slices2"]) + (np.cos(((((data["minbatch_slices2"]) > (data["signal_shift_+1"]))*1.)))))/2.0), np.sin((data["abs_avgbatch_slices2"])) ) > -998, data["maxtominbatch_msignal"], data["signal_shift_+1"] )))) +

                            0.050000*np.tanh(np.where((((data["signal"]) <= ((((data["rangebatch_slices2"]) + (np.minimum(((data["medianbatch_slices2"])), ((data["signal_shift_+1"])))))/2.0)))*1.) > -998, data["signal_shift_-1"], (((data["medianbatch_slices2"]) + (((np.where(data["signal"] <= -998, np.cos((data["abs_maxbatch_slices2"])), data["signal_shift_+1"] )) - (data["signal"]))))/2.0) )) +

                            0.050000*np.tanh(((data["abs_avgbatch_slices2_msignal"]) * (((data["minbatch_slices2_msignal"]) - (np.where(((data["abs_minbatch_slices2"]) * (np.sin((data["signal_shift_-1_msignal"])))) > -998, data["abs_maxbatch_slices2_msignal"], (11.76613903045654297) )))))) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) + (np.where(data["abs_maxbatch_msignal"] <= -998, np.sin(((-((data["abs_maxbatch_msignal"]))))), data["signal_shift_+1"] )))) +

                            0.050000*np.tanh(((((data["medianbatch_slices2"]) - ((((data["signal_shift_+1"]) > (((data["signal_shift_+1"]) + (data["medianbatch_slices2"]))))*1.)))) - ((((data["abs_avgbatch_slices2"]) <= (np.minimum(((data["rangebatch_msignal"])), ((data["mean_abs_chgbatch_slices2"])))))*1.)))) +

                            0.050000*np.tanh(np.where(((((((9.66260623931884766)) > ((1.0)))*1.)) / 2.0) > -998, data["signal"], np.sin((data["abs_maxbatch_slices2"])) )) +

                            0.050000*np.tanh(np.where(((np.minimum(((data["mean_abs_chgbatch_msignal"])), ((np.where(((data["signal_shift_+1"]) - ((((data["maxbatch_slices2_msignal"]) > (data["abs_maxbatch"]))*1.))) > -998, data["abs_avgbatch_slices2_msignal"], np.sin((data["maxbatch_slices2_msignal"])) ))))) + (data["abs_avgbatch_slices2"])) > -998, ((data["meanbatch_slices2"]) * 2.0), (((data["signal_shift_+1"]) + (((data["maxtominbatch_msignal"]) * 2.0)))/2.0) )) +

                            0.050000*np.tanh(((np.sin((data["rangebatch_msignal"]))) * ((-(((((data["abs_minbatch_msignal"]) + (data["signal_shift_-1_msignal"]))/2.0))))))) +

                            0.050000*np.tanh((-((((data["stdbatch_msignal"]) - ((-((np.where(((data["abs_avgbatch_slices2_msignal"]) * 2.0) > -998, data["stdbatch_slices2_msignal"], np.maximum((((((data["signal"]) > ((10.39941024780273438)))*1.))), ((np.maximum(((data["stdbatch_msignal"])), ((((data["abs_maxbatch_slices2"]) * 2.0))))))) )))))))))) +

                            0.050000*np.tanh(np.where(data["signal_shift_+1"] > -998, (-((data["stdbatch_slices2_msignal"]))), np.tanh((np.sin((np.cos((data["medianbatch_slices2"])))))) )) +

                            0.050000*np.tanh(np.where(np.where(data["abs_maxbatch"] <= -998, data["signal_shift_+1_msignal"], data["rangebatch_slices2"] ) <= -998, np.cos((np.sin((data["meanbatch_slices2"])))), ((np.sin(((-((data["abs_avgbatch_slices2"])))))) * 2.0) )) +

                            0.050000*np.tanh(np.where((((data["signal_shift_-1"]) > (np.sin((data["maxtominbatch_slices2_msignal"]))))*1.) <= -998, ((data["medianbatch_slices2"]) + (data["abs_avgbatch_slices2_msignal"])), ((((((data["signal_shift_-1"]) / 2.0)) + ((((data["medianbatch_slices2"]) <= (data["stdbatch_slices2"]))*1.)))) - (((((((((data["stdbatch_msignal"]) + (data["stdbatch_msignal"]))) / 2.0)) + (data["abs_avgbatch_slices2_msignal"]))) + (data["abs_avgbatch_slices2_msignal"])))) )) +

                            0.050000*np.tanh(np.where(data["abs_minbatch_msignal"] <= -998, ((data["maxbatch_slices2_msignal"]) - ((((np.where(((data["stdbatch_slices2"]) * 2.0) <= -998, data["maxbatch_msignal"], np.sin((data["rangebatch_msignal"])) )) > (data["medianbatch_slices2"]))*1.))), data["signal_shift_+1"] )) +

                            0.050000*np.tanh(np.sin((np.maximum(((data["rangebatch_msignal"])), ((np.sin((np.maximum(((((np.maximum(((data["rangebatch_msignal"])), ((data["rangebatch_msignal"])))) / 2.0))), ((np.sin((np.maximum(((data["medianbatch_slices2_msignal"])), ((np.maximum(((np.where(data["mean_abs_chgbatch_slices2"] <= -998, data["mean_abs_chgbatch_slices2"], np.maximum(((data["mean_abs_chgbatch_slices2"])), ((data["rangebatch_msignal"]))) ))), ((data["rangebatch_msignal"]))))))))))))))))))) +

                            0.050000*np.tanh(((data["minbatch"]) - (((data["maxtominbatch_msignal"]) + ((((data["medianbatch_slices2"]) > (((data["maxtominbatch_msignal"]) - (data["minbatch"]))))*1.)))))) +

                            0.050000*np.tanh(((((data["medianbatch_slices2"]) - (np.where(data["maxbatch_msignal"] > -998, ((np.where(data["stdbatch_msignal"] <= -998, data["abs_avgbatch_msignal"], data["abs_avgbatch_slices2_msignal"] )) * 2.0), data["signal_shift_+1_msignal"] )))) + (np.minimum(((data["medianbatch_slices2"])), ((data["signal_shift_-1_msignal"])))))) +

                            0.050000*np.tanh(np.where(((((data["mean_abs_chgbatch_msignal"]) / 2.0)) / 2.0) > -998, (((-((data["mean_abs_chgbatch_msignal"])))) - ((2.0))), (((((-((data["mean_abs_chgbatch_msignal"])))) - (((data["mean_abs_chgbatch_msignal"]) * 2.0)))) - (((np.where(((data["signal"]) * (data["abs_avgbatch_msignal"])) <= -998, (2.55029869079589844), data["signal_shift_-1_msignal"] )) - (data["stdbatch_slices2_msignal"])))) )) +

                            0.050000*np.tanh(np.where(((data["minbatch_slices2_msignal"]) / 2.0) <= -998, data["minbatch_slices2"], ((np.sin((np.where(np.where(((data["signal"]) / 2.0) > -998, np.where(data["stdbatch_msignal"] <= -998, ((data["medianbatch_msignal"]) * 2.0), ((np.sin((np.sin((data["abs_maxbatch"]))))) * 2.0) ), data["minbatch_slices2"] ) > -998, data["abs_maxbatch"], ((data["maxbatch_msignal"]) * 2.0) )))) * 2.0) )) +

                            0.050000*np.tanh((-((np.where(data["maxtominbatch_slices2_msignal"] > -998, data["stdbatch_slices2_msignal"], ((np.where(data["minbatch_slices2"] <= -998, data["signal_shift_+1"], np.where((-((data["stdbatch_slices2_msignal"]))) > -998, (-((data["mean_abs_chgbatch_slices2_msignal"]))), ((data["medianbatch_slices2"]) / 2.0) ) )) + (data["minbatch_msignal"])) ))))) +

                            0.050000*np.tanh(((np.sin((data["abs_maxbatch"]))) * (np.where((((data["stdbatch_slices2"]) + (np.where(np.minimum(((np.cos((data["minbatch_msignal"])))), ((data["mean_abs_chgbatch_slices2_msignal"]))) > -998, data["abs_avgbatch_slices2_msignal"], (((2.0)) / 2.0) )))/2.0) > -998, np.maximum(((np.sin((data["abs_maxbatch"])))), ((data["abs_maxbatch_slices2_msignal"]))), (((data["minbatch_slices2_msignal"]) > (data["mean_abs_chgbatch_msignal"]))*1.) )))) +

                            0.050000*np.tanh(((np.sin((np.where(np.where(data["abs_maxbatch_slices2_msignal"] > -998, np.where(np.where(np.sin((data["abs_minbatch_msignal"])) > -998, data["mean_abs_chgbatch_slices2_msignal"], data["mean_abs_chgbatch_slices2_msignal"] ) > -998, data["minbatch_slices2"], data["minbatch_slices2"] ), ((data["signal_shift_-1_msignal"]) - (np.maximum(((np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["minbatch_slices2"], data["medianbatch_msignal"] ))), ((data["stdbatch_msignal"]))))) ) > -998, data["abs_maxbatch_slices2"], data["minbatch_slices2"] )))) - (data["abs_avgbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(np.maximum(((data["minbatch_slices2"])), ((((data["minbatch"]) * (np.minimum(((data["stdbatch_msignal"])), ((data["rangebatch_msignal"]))))))))) +

                            0.050000*np.tanh(np.sin((data["abs_maxbatch"]))) +

                            0.050000*np.tanh(np.sin((np.where(((data["rangebatch_slices2"]) * 2.0) <= -998, data["minbatch_msignal"], np.where((((((((data["abs_avgbatch_slices2"]) + ((6.0)))) * 2.0)) <= (((data["maxbatch_slices2"]) * 2.0)))*1.) > -998, data["abs_minbatch_slices2_msignal"], data["abs_maxbatch_slices2"] ) )))) +

                            0.050000*np.tanh(np.minimum(((np.minimum(((np.cos((np.where(data["mean_abs_chgbatch_slices2"] > -998, data["maxtominbatch_msignal"], (11.48959922790527344) ))))), ((np.maximum((((((data["abs_maxbatch_msignal"]) <= (np.where(data["abs_minbatch_slices2_msignal"] > -998, data["maxbatch_msignal"], data["minbatch"] )))*1.))), ((((np.where(data["medianbatch_slices2"] > -998, data["maxtominbatch_msignal"], data["minbatch"] )) * (((data["rangebatch_msignal"]) / 2.0))))))))))), (((-((data["minbatch"]))))))) +

                            0.050000*np.tanh(np.sin((np.sin((np.maximum(((np.maximum(((np.maximum(((data["minbatch"])), ((data["rangebatch_msignal"]))))), ((data["mean_abs_chgbatch_slices2_msignal"]))))), ((data["rangebatch_msignal"])))))))) +

                            0.050000*np.tanh(np.sin((((data["rangebatch_msignal"]) - ((((data["minbatch_slices2_msignal"]) <= (np.cos((np.tanh((data["maxtominbatch_slices2_msignal"]))))))*1.)))))) +

                            0.050000*np.tanh(((data["minbatch"]) * (np.maximum(((data["meanbatch_msignal"])), ((((((data["medianbatch_slices2_msignal"]) * ((-((data["signal"])))))) + (((np.sin((data["minbatch_slices2_msignal"]))) - (data["abs_maxbatch_slices2_msignal"])))))))))) +

                            0.050000*np.tanh(np.where(np.maximum(((np.sin((((((data["rangebatch_msignal"]) * 2.0)) * (data["minbatch_msignal"])))))), ((data["maxbatch_slices2"]))) > -998, ((data["abs_avgbatch_slices2_msignal"]) * (data["minbatch_msignal"])), (((data["abs_avgbatch_msignal"]) + (((data["medianbatch_msignal"]) * (data["maxtominbatch_slices2"]))))/2.0) )) +

                            0.050000*np.tanh(np.where(np.where(((np.sin((np.where(data["rangebatch_msignal"] <= -998, data["minbatch_slices2"], (-((((data["mean_abs_chgbatch_slices2_msignal"]) * (np.cos((data["mean_abs_chgbatch_slices2_msignal"]))))))) )))) * (data["abs_minbatch_slices2_msignal"])) <= -998, data["maxtominbatch_slices2_msignal"], data["maxbatch_slices2_msignal"] ) <= -998, data["maxbatch_slices2_msignal"], (-((((data["maxbatch_slices2_msignal"]) * (np.cos((data["mean_abs_chgbatch_slices2_msignal"]))))))) )) +

                            0.050000*np.tanh(((((((np.sin((data["maxbatch_msignal"]))) * (np.minimum(((np.minimum(((np.maximum(((data["minbatch_slices2"])), ((((np.where(data["abs_minbatch_msignal"] <= -998, data["abs_maxbatch_msignal"], (((data["maxtominbatch_slices2"]) <= (np.cos((data["signal"]))))*1.) )) - (data["abs_maxbatch_msignal"]))))))), ((np.cos((data["abs_maxbatch_msignal"]))))))), ((data["abs_minbatch_msignal"])))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh((-((((data["abs_minbatch_slices2_msignal"]) * (((np.minimum(((((data["minbatch"]) - (data["abs_minbatch_slices2_msignal"])))), ((((data["minbatch"]) - (np.tanh((((data["abs_avgbatch_slices2_msignal"]) - ((((((data["abs_maxbatch_slices2_msignal"]) * (data["stdbatch_slices2_msignal"]))) > (np.cos((np.sin((data["abs_minbatch_slices2_msignal"]))))))*1.))))))))))) * (np.cos((data["minbatch_msignal"])))))))))) +

                            0.050000*np.tanh(np.where(np.minimum(((np.sin((np.sin((data["abs_maxbatch"])))))), ((np.where(((data["meanbatch_msignal"]) + (np.cos((data["abs_minbatch_msignal"])))) <= -998, ((data["maxtominbatch_msignal"]) * 2.0), np.cos((data["maxtominbatch_msignal"])) )))) > -998, np.sin((data["abs_minbatch_msignal"])), data["signal_shift_+1_msignal"] )) +

                            0.050000*np.tanh(((np.cos((data["maxtominbatch_msignal"]))) * 2.0)) +

                            0.050000*np.tanh(((np.cos((data["maxtominbatch_msignal"]))) * 2.0)) +

                            0.050000*np.tanh(((((data["maxtominbatch_slices2_msignal"]) * (np.sin((data["signal"]))))) + (np.where((-((data["minbatch"]))) > -998, data["minbatch"], np.where((-((data["minbatch"]))) <= -998, np.tanh((np.where(data["signal"] > -998, (-((np.sin((data["signal"]))))), data["stdbatch_slices2"] ))), ((data["maxtominbatch_slices2_msignal"]) * (np.sin((data["signal"])))) ) )))) +

                            0.050000*np.tanh(((((((np.minimum(((data["maxbatch_msignal"])), ((data["abs_maxbatch_msignal"])))) + (np.sin((((np.sin((data["maxbatch_msignal"]))) + (((((((data["maxbatch_msignal"]) > (data["signal"]))*1.)) <= (data["signal_shift_-1"]))*1.)))))))) + (((data["maxtominbatch_slices2"]) - (data["rangebatch_slices2"]))))) * (np.sin((data["maxbatch_msignal"]))))) +

                            0.050000*np.tanh(np.minimum(((((((data["minbatch_msignal"]) * (data["abs_avgbatch_msignal"]))) * (np.maximum((((((data["abs_maxbatch"]) + (data["stdbatch_slices2"]))/2.0))), ((np.where(((data["minbatch_msignal"]) * (data["meanbatch_msignal"])) <= -998, (((data["abs_minbatch_slices2_msignal"]) + (data["stdbatch_slices2"]))/2.0), data["stdbatch_slices2"] )))))))), ((np.maximum(((data["signal_shift_-1"])), (((-((np.where(data["stdbatch_slices2"] > -998, data["medianbatch_msignal"], data["meanbatch_msignal"] ))))))))))) +

                            0.050000*np.tanh(((np.sin((np.where(np.tanh((np.sin((np.where(data["abs_maxbatch_msignal"] > -998, data["abs_minbatch_slices2_msignal"], data["maxtominbatch_slices2_msignal"] ))))) > -998, data["rangebatch_msignal"], ((data["abs_maxbatch_slices2_msignal"]) / 2.0) )))) * 2.0)) +

                            0.050000*np.tanh(((np.where(data["maxtominbatch"] > -998, np.sin((np.where((-((data["abs_maxbatch_slices2_msignal"]))) <= -998, np.where(data["maxtominbatch"] > -998, np.cos((data["abs_minbatch_msignal"])), data["meanbatch_msignal"] ), np.where(np.cos((data["meanbatch_msignal"])) <= -998, data["abs_minbatch_slices2"], data["rangebatch_msignal"] ) ))), ((data["abs_minbatch_slices2"]) - (data["abs_minbatch_msignal"])) )) * 2.0)) +

                            0.050000*np.tanh(((((((data["abs_maxbatch"]) * (np.cos((((np.where(((data["meanbatch_slices2"]) * 2.0) > -998, data["abs_maxbatch_slices2_msignal"], data["signal_shift_+1"] )) * 2.0)))))) - (np.where(((((data["minbatch_slices2_msignal"]) * 2.0)) * (data["minbatch"])) > -998, data["abs_maxbatch_slices2_msignal"], np.where(data["signal_shift_-1"] <= -998, data["stdbatch_slices2_msignal"], data["rangebatch_msignal"] ) )))) * 2.0)) +

                            0.050000*np.tanh(np.sin((np.where(np.sin((np.cos((data["mean_abs_chgbatch_slices2"])))) <= -998, data["rangebatch_slices2"], data["abs_maxbatch_slices2"] )))) +

                            0.050000*np.tanh(np.where(data["medianbatch_slices2_msignal"] > -998, ((data["abs_maxbatch"]) * (np.sin((np.sin((np.sin((((((np.minimum(((data["medianbatch_slices2"])), ((data["abs_minbatch_slices2_msignal"])))) * 2.0)) / 2.0))))))))), np.maximum(((data["abs_minbatch_msignal"])), ((((data["maxbatch_slices2_msignal"]) * (((data["signal_shift_-1"]) / 2.0)))))) )) +

                            0.050000*np.tanh(np.sin((data["abs_minbatch_msignal"]))) +

                            0.050000*np.tanh(((((np.where(data["abs_maxbatch_slices2_msignal"] <= -998, np.where(data["maxtominbatch"] <= -998, data["rangebatch_msignal"], data["minbatch_msignal"] ), ((np.cos((np.where(data["rangebatch_msignal"] > -998, data["stdbatch_msignal"], (10.96751308441162109) )))) * 2.0) )) / 2.0)) * (np.where(data["minbatch_slices2"] > -998, data["minbatch_slices2"], data["abs_maxbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(np.minimum(((((data["medianbatch_slices2_msignal"]) * (data["maxtominbatch_msignal"])))), ((np.minimum(((((data["medianbatch_msignal"]) * (data["meanbatch_slices2_msignal"])))), ((np.where((((data["meanbatch_slices2_msignal"]) + (data["meanbatch_slices2_msignal"]))/2.0) > -998, ((((data["abs_maxbatch_slices2_msignal"]) + (data["medianbatch_slices2_msignal"]))) + (data["medianbatch_slices2_msignal"])), data["meanbatch_slices2"] )))))))) +

                            0.050000*np.tanh(((data["medianbatch_slices2"]) + ((-((np.where(data["abs_avgbatch_msignal"] > -998, ((((data["abs_maxbatch_slices2"]) * (data["abs_avgbatch_msignal"]))) * (data["abs_avgbatch_msignal"])), (((data["medianbatch_slices2"]) + (data["medianbatch_slices2"]))/2.0) ))))))) +

                            0.050000*np.tanh(np.where(data["signal"] <= -998, np.maximum(((data["maxbatch_msignal"])), (((((data["abs_avgbatch_slices2_msignal"]) <= (((data["maxtominbatch_msignal"]) / 2.0)))*1.)))), np.where(data["rangebatch_slices2_msignal"] <= -998, data["maxtominbatch_msignal"], np.cos((data["maxtominbatch_msignal"])) ) )) +

                            0.050000*np.tanh(((np.where(np.where(data["stdbatch_msignal"] > -998, data["stdbatch_msignal"], ((np.sin(((2.0)))) * 2.0) ) > -998, np.sin((((np.where(np.sin((((((data["stdbatch_msignal"]) * 2.0)) + (((np.where(data["stdbatch_msignal"] > -998, data["rangebatch_slices2"], data["stdbatch_msignal"] )) / 2.0))))) > -998, data["stdbatch_msignal"], np.cos((data["maxtominbatch"])) )) * 2.0))), (2.0) )) * 2.0)) +

                            0.050000*np.tanh(((np.cos(((((data["stdbatch_slices2_msignal"]) + (((np.where(np.where(data["signal_shift_-1"] > -998, data["abs_minbatch_slices2_msignal"], data["minbatch_slices2_msignal"] ) > -998, data["abs_minbatch_slices2_msignal"], np.where((2.0) <= -998, data["abs_minbatch_slices2_msignal"], ((((data["stdbatch_slices2"]) * (np.minimum(((data["abs_minbatch_msignal"])), ((np.cos((data["abs_minbatch_slices2_msignal"])))))))) * 2.0) ) )) * 2.0)))/2.0)))) * (np.maximum(((data["signal_shift_-1"])), ((data["signal_shift_-1"])))))) +

                            0.050000*np.tanh(((np.sin((np.where(np.sin((np.where(data["maxtominbatch_slices2"] > -998, np.tanh((data["medianbatch_msignal"])), np.tanh((((((((data["abs_avgbatch_msignal"]) + (data["medianbatch_msignal"]))) - (data["rangebatch_msignal"]))) + (data["abs_maxbatch"])))) ))) > -998, data["rangebatch_msignal"], ((data["abs_avgbatch_msignal"]) - (data["medianbatch_msignal"])) )))) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((np.where((1.77388834953308105) <= -998, data["abs_avgbatch_msignal"], np.sin((np.where(((data["abs_minbatch_slices2_msignal"]) + (data["signal_shift_-1"])) > -998, np.maximum(((np.where(data["signal_shift_-1"] > -998, np.sin((data["abs_minbatch_slices2_msignal"])), data["mean_abs_chgbatch_slices2"] ))), ((np.sin((data["abs_minbatch_slices2_msignal"]))))), ((data["rangebatch_msignal"]) * (np.sin((data["rangebatch_msignal"])))) ))) ))), ((data["rangebatch_msignal"])))) +

                            0.050000*np.tanh(np.where(data["minbatch_slices2"] <= -998, np.where(data["signal_shift_+1"] > -998, data["maxtominbatch_msignal"], (((np.cos((np.where(np.minimum(((data["maxtominbatch_msignal"])), (((9.0)))) <= -998, data["mean_abs_chgbatch_slices2"], data["abs_avgbatch_msignal"] )))) <= (data["signal_shift_-1"]))*1.) ), np.cos((((np.where(data["maxtominbatch_msignal"] > -998, data["abs_avgbatch_msignal"], data["maxtominbatch_slices2"] )) * 2.0))) )) +

                            0.050000*np.tanh(np.sin((np.maximum((((4.23392868041992188))), ((np.where(data["signal_shift_+1"] > -998, data["rangebatch_msignal"], (((np.maximum(((data["rangebatch_msignal"])), ((np.where(data["signal_shift_+1"] > -998, data["rangebatch_msignal"], (-((data["maxbatch_slices2"]))) ))))) <= (np.cos((np.tanh(((((((data["signal_shift_-1_msignal"]) <= (data["signal_shift_-1"]))*1.)) / 2.0)))))))*1.) ))))))) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) * (np.where(np.cos((data["mean_abs_chgbatch_slices2_msignal"])) > -998, np.cos((data["abs_avgbatch_slices2_msignal"])), data["minbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(((data["stdbatch_slices2_msignal"]) * (np.cos((np.where(np.where(data["medianbatch_slices2"] > -998, data["minbatch_msignal"], (((data["maxbatch_slices2_msignal"]) + (np.cos((np.where(data["stdbatch_slices2_msignal"] > -998, np.cos((data["stdbatch_slices2_msignal"])), data["minbatch_msignal"] )))))/2.0) ) > -998, data["minbatch_msignal"], data["meanbatch_slices2"] )))))) +

                            0.050000*np.tanh(((np.cos((((np.minimum(((data["abs_minbatch_slices2_msignal"])), ((np.cos((((np.minimum(((data["abs_minbatch_slices2_msignal"])), ((data["rangebatch_msignal"])))) * 2.0))))))) * 2.0)))) * (np.where(data["maxbatch_slices2_msignal"] > -998, ((data["stdbatch_slices2_msignal"]) * 2.0), np.maximum(((np.cos((data["abs_minbatch_slices2_msignal"])))), ((data["minbatch_slices2_msignal"]))) )))) +

                            0.050000*np.tanh(((data["medianbatch_slices2"]) * (np.cos((((np.where(data["abs_minbatch_slices2_msignal"] <= -998, ((data["signal_shift_-1_msignal"]) / 2.0), data["maxtominbatch_msignal"] )) * 2.0)))))) +

                            0.050000*np.tanh(((((data["minbatch"]) / 2.0)) * (((np.cos((((data["maxtominbatch_msignal"]) + (data["abs_maxbatch_msignal"]))))) * 2.0)))) +

                            0.050000*np.tanh(((np.sin((((np.sin((data["abs_maxbatch_msignal"]))) * 2.0)))) * (((data["abs_maxbatch_msignal"]) * (np.minimum(((data["rangebatch_msignal"])), ((data["minbatch_msignal"])))))))) +

                            0.050000*np.tanh(((np.where(np.where(data["maxbatch_slices2"] > -998, data["abs_minbatch_slices2_msignal"], data["stdbatch_msignal"] ) <= -998, data["maxbatch_slices2"], data["maxtominbatch_slices2"] )) * (np.maximum(((np.sin((np.sin((data["abs_avgbatch_slices2"])))))), ((np.sin((data["minbatch_msignal"])))))))) +

                            0.050000*np.tanh(((((np.sin((np.sin((((np.maximum(((data["maxbatch_msignal"])), ((np.sin((((np.maximum(((data["abs_maxbatch_slices2_msignal"])), ((((np.sin((data["mean_abs_chgbatch_msignal"]))) - ((((np.sin((np.cos((data["maxtominbatch_msignal"]))))) + (((data["abs_maxbatch_slices2_msignal"]) + (np.sin((data["abs_minbatch_slices2_msignal"]))))))/2.0))))))) * 2.0))))))) * 2.0)))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.cos((((np.where(np.maximum(((np.sin((((data["minbatch_slices2"]) / 2.0))))), ((((data["maxbatch_slices2"]) - (data["minbatch_msignal"]))))) > -998, data["abs_maxbatch_slices2_msignal"], ((np.sin((((((data["minbatch_slices2"]) * 2.0)) - (data["minbatch_slices2"]))))) / 2.0) )) * 2.0)))) +

                            0.050000*np.tanh((-((np.sin((np.cos((np.minimum((((((-((data["medianbatch_slices2_msignal"])))) - (((data["maxtominbatch"]) / 2.0))))), ((((data["mean_abs_chgbatch_slices2_msignal"]) + (np.minimum(((data["maxtominbatch_msignal"])), (((((data["minbatch_msignal"]) + (((np.sin((np.cos((np.minimum(((data["signal"])), (((-((((data["abs_avgbatch_msignal"]) - (data["mean_abs_chgbatch_slices2_msignal"]))))))))))))) / 2.0)))/2.0))))))))))))))))) +

                            0.050000*np.tanh(((((data["meanbatch_msignal"]) * (np.maximum(((data["rangebatch_slices2_msignal"])), ((((np.sin((data["maxbatch_msignal"]))) * (((np.sin((((data["rangebatch_msignal"]) * 2.0)))) * (((np.sin((data["maxbatch_msignal"]))) * 2.0))))))))))) * (np.sin((data["maxbatch_msignal"]))))) +

                            0.050000*np.tanh(np.where(np.where((3.11168861389160156) <= -998, data["signal_shift_-1_msignal"], ((data["rangebatch_slices2_msignal"]) * (np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0))))) ) <= -998, np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0))), (((((14.36763572692871094)) * (np.cos((np.where(data["abs_avgbatch_msignal"] <= -998, data["rangebatch_slices2_msignal"], ((np.sin((data["maxtominbatch_msignal"]))) * 2.0) )))))) * 2.0) )) +

                            0.050000*np.tanh(((((((data["abs_minbatch_slices2_msignal"]) * (data["meanbatch_slices2"]))) * (np.sin((np.maximum(((((data["abs_minbatch_slices2_msignal"]) * (data["meanbatch_slices2"])))), ((data["maxbatch_msignal"])))))))) - ((((data["meanbatch_slices2"]) > (data["abs_maxbatch_slices2_msignal"]))*1.)))) +

                            0.050000*np.tanh(np.sin((np.sin((((data["abs_minbatch_slices2_msignal"]) - (np.cos(((((((data["signal_shift_-1"]) <= (np.tanh((np.tanh((np.tanh((data["maxtominbatch_slices2_msignal"]))))))))*1.)) * 2.0)))))))))) +

                            0.050000*np.tanh((((((-((np.where(((data["maxtominbatch_slices2_msignal"]) * (np.tanh((((data["stdbatch_msignal"]) * 2.0))))) <= -998, ((data["stdbatch_msignal"]) * 2.0), ((data["abs_minbatch_msignal"]) * 2.0) ))))) * 2.0)) * (np.sin((np.where(np.sin((((data["stdbatch_msignal"]) * 2.0))) <= -998, data["signal_shift_-1_msignal"], ((data["stdbatch_msignal"]) * 2.0) )))))) +

                            0.050000*np.tanh((-((((np.sin((data["stdbatch_msignal"]))) + (np.where(data["signal_shift_+1"] <= -998, (-((np.where(data["stdbatch_slices2"] > -998, (((data["stdbatch_slices2"]) > (data["minbatch_slices2_msignal"]))*1.), np.where(data["minbatch_msignal"] <= -998, np.sin((np.cos((data["abs_maxbatch_slices2"])))), data["stdbatch_msignal"] ) )))), ((data["abs_maxbatch_slices2_msignal"]) - (data["signal"])) ))))))) +

                            0.050000*np.tanh(np.minimum(((((np.where((-(((-((data["abs_minbatch_msignal"])))))) <= -998, data["minbatch_slices2"], np.minimum(((np.sin((np.where(data["rangebatch_msignal"] <= -998, data["minbatch_slices2"], data["maxbatch_slices2"] ))))), ((np.sin((data["maxtominbatch_slices2"]))))) )) + (data["rangebatch_msignal"])))), ((data["abs_minbatch_msignal"])))) +

                            0.050000*np.tanh(((np.minimum(((data["maxtominbatch_slices2"])), ((data["minbatch_msignal"])))) / 2.0)) +

                            0.050000*np.tanh(((np.where(np.sin((((data["maxtominbatch_msignal"]) * 2.0))) <= -998, np.tanh((np.cos((np.minimum(((data["maxtominbatch_msignal"])), (((((((-((data["abs_avgbatch_slices2_msignal"])))) - (data["maxtominbatch_msignal"]))) - (((data["abs_avgbatch_slices2"]) * 2.0)))))))))), np.where(((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0) <= -998, data["abs_avgbatch_slices2_msignal"], np.cos(((-((data["abs_avgbatch_slices2_msignal"]))))) ) )) * 2.0)) +

                            0.050000*np.tanh(((((np.sin((np.where(data["medianbatch_msignal"] > -998, data["abs_maxbatch_msignal"], data["maxbatch_msignal"] )))) * (data["medianbatch_msignal"]))) * (data["rangebatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((np.sin((((np.where(np.where(data["medianbatch_slices2"] > -998, data["stdbatch_msignal"], ((((np.maximum(((data["abs_minbatch_msignal"])), ((data["minbatch"])))) + (data["stdbatch_slices2"]))) + (np.where(data["maxbatch_slices2"] > -998, data["stdbatch_msignal"], data["minbatch"] ))) ) > -998, (-((((data["stdbatch_slices2"]) + (data["stdbatch_msignal"]))))), ((data["signal_shift_-1"]) + (data["stdbatch_slices2"])) )) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(((data["abs_minbatch_slices2_msignal"]) * (np.cos((((data["minbatch_msignal"]) + (np.sin((((data["minbatch"]) * ((-((data["maxtominbatch"])))))))))))))) +

                            0.050000*np.tanh(((data["abs_minbatch_slices2_msignal"]) * (np.cos((((data["abs_maxbatch_msignal"]) - ((((np.maximum(((data["maxtominbatch_slices2_msignal"])), ((data["mean_abs_chgbatch_slices2"])))) > (((np.where(np.cos((((data["rangebatch_slices2_msignal"]) - (data["abs_minbatch_slices2_msignal"])))) <= -998, np.sin((data["signal"])), ((data["abs_minbatch_slices2_msignal"]) * (data["signal"])) )) + (data["minbatch_slices2"]))))*1.)))))))) +

                            0.050000*np.tanh(np.where(np.sin((data["abs_minbatch_msignal"])) > -998, ((data["mean_abs_chgbatch_slices2"]) - ((((-((((data["minbatch_slices2"]) * (np.cos((data["rangebatch_slices2_msignal"])))))))) * (data["stdbatch_msignal"])))), ((np.where(data["maxtominbatch"] <= -998, data["abs_minbatch_msignal"], ((np.sin((data["mean_abs_chgbatch_msignal"]))) * 2.0) )) + (np.cos((data["signal_shift_+1"])))) )) +

                            0.050000*np.tanh(np.sin((np.where(data["abs_minbatch_msignal"] > -998, data["abs_minbatch_msignal"], np.maximum(((data["rangebatch_msignal"])), ((np.minimum(((np.sin((data["abs_minbatch_slices2_msignal"])))), ((np.where(np.maximum(((data["abs_maxbatch_msignal"])), ((np.sin((data["maxtominbatch_slices2"]))))) <= -998, np.sin((data["abs_minbatch_msignal"])), data["abs_minbatch_msignal"] ))))))) )))) +

                            0.050000*np.tanh((((-((np.cos((np.where(data["abs_minbatch_slices2"] <= -998, data["abs_avgbatch_slices2_msignal"], np.where(np.cos((np.cos((np.sin((data["maxtominbatch_slices2_msignal"])))))) > -998, (-((((np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["medianbatch_msignal"], data["abs_minbatch_msignal"] )) * 2.0)))), data["maxtominbatch_slices2_msignal"] ) ))))))) * 2.0)) +

                            0.050000*np.tanh(((np.sin((data["maxbatch_msignal"]))) * (np.where(np.sin((data["maxbatch_msignal"])) <= -998, np.where(data["abs_minbatch_msignal"] <= -998, np.sin((data["abs_maxbatch"])), np.sin((np.sin((np.minimum(((np.cos((((np.sin((np.sin((data["maxbatch_msignal"]))))) * (data["minbatch_msignal"])))))), ((data["signal_shift_-1"]))))))) ), data["mean_abs_chgbatch_msignal"] )))) +

                            0.050000*np.tanh(np.sin((np.where(data["mean_abs_chgbatch_msignal"] <= -998, np.where(np.tanh((data["abs_minbatch_msignal"])) <= -998, data["abs_maxbatch_slices2_msignal"], np.where(np.where(data["rangebatch_slices2"] <= -998, ((np.where(np.where(data["mean_abs_chgbatch_msignal"] <= -998, data["abs_maxbatch"], data["abs_maxbatch"] ) <= -998, data["abs_avgbatch_slices2_msignal"], np.minimum(((data["maxbatch_slices2"])), ((data["meanbatch_slices2"]))) )) * 2.0), data["minbatch"] ) <= -998, data["abs_maxbatch"], data["abs_maxbatch"] ) ), data["abs_maxbatch"] )))) +

                            0.050000*np.tanh(((data["minbatch"]) * (np.where((((data["minbatch"]) <= (data["rangebatch_slices2"]))*1.) <= -998, np.where(data["maxtominbatch_slices2"] <= -998, ((np.where(data["maxbatch_slices2"] <= -998, ((np.where(data["maxbatch_msignal"] <= -998, data["minbatch"], data["meanbatch_msignal"] )) + (data["meanbatch_msignal"])), data["maxbatch_slices2"] )) * 2.0), data["maxbatch_msignal"] ), ((data["abs_avgbatch_slices2_msignal"]) + (data["meanbatch_msignal"])) )))) +

                            0.050000*np.tanh(((data["abs_maxbatch_slices2_msignal"]) * (np.tanh((np.sin((np.where(np.tanh((data["minbatch_slices2_msignal"])) > -998, data["abs_minbatch_slices2_msignal"], ((np.cos((data["maxtominbatch_msignal"]))) * (np.tanh((data["abs_minbatch_slices2_msignal"])))) )))))))) +

                            0.050000*np.tanh(np.where(data["minbatch"] <= -998, (0.0), data["stdbatch_slices2"] )) +

                            0.050000*np.tanh(((((data["minbatch"]) / 2.0)) * (np.maximum(((np.minimum(((data["minbatch_slices2"])), ((((((data["minbatch_msignal"]) * 2.0)) / 2.0)))))), ((((data["signal_shift_+1"]) - (np.minimum(((np.where(np.minimum(((data["abs_maxbatch_msignal"])), ((data["mean_abs_chgbatch_msignal"]))) > -998, np.minimum(((((np.tanh((data["stdbatch_slices2_msignal"]))) * 2.0))), (((-((data["maxbatch_msignal"])))))), data["maxtominbatch_slices2"] ))), ((data["abs_avgbatch_slices2_msignal"]))))))))))) +

                            0.050000*np.tanh(((np.minimum(((((np.sin((((data["signal_shift_+1"]) / 2.0)))) - (np.minimum(((data["abs_avgbatch_slices2"])), ((((((data["abs_avgbatch_msignal"]) * 2.0)) + (np.tanh(((((data["abs_maxbatch_slices2_msignal"]) > (data["abs_avgbatch_msignal"]))*1.)))))))))))), ((((data["abs_avgbatch_msignal"]) + (np.tanh((((data["abs_maxbatch"]) + (data["minbatch"])))))))))) * 2.0)) +

                            0.050000*np.tanh(((np.cos((np.where(np.cos((((np.cos((np.where(((((13.23457622528076172)) <= (np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)))))*1.) > -998, data["abs_avgbatch_slices2_msignal"], (-((np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)))))) )))) * 2.0))) > -998, ((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0), (13.23457622528076172) )))) * 2.0)) +

                            0.050000*np.tanh(((np.cos((((data["abs_avgbatch_slices2_msignal"]) - ((((data["maxtominbatch_msignal"]) > (np.maximum((((((np.tanh((((((data["mean_abs_chgbatch_slices2"]) - ((((np.tanh((data["abs_avgbatch_slices2_msignal"]))) > (data["medianbatch_slices2_msignal"]))*1.)))) * 2.0)))) > (data["maxbatch_slices2_msignal"]))*1.))), ((np.cos((((data["minbatch_slices2"]) + (data["abs_avgbatch_slices2_msignal"])))))))))*1.)))))) * 2.0)) +

                            0.050000*np.tanh(((np.where(np.where((((data["meanbatch_slices2_msignal"]) > ((((data["meanbatch_slices2"]) > (data["abs_maxbatch_slices2"]))*1.)))*1.) <= -998, data["minbatch_slices2_msignal"], np.sin(((((data["rangebatch_slices2"]) > ((3.0)))*1.))) ) <= -998, data["abs_maxbatch_slices2_msignal"], np.sin((data["abs_minbatch_msignal"])) )) * (data["abs_maxbatch_slices2"]))) +

                            0.050000*np.tanh(np.cos((np.minimum(((np.minimum(((data["maxtominbatch_msignal"])), ((np.cos((np.minimum(((np.cos((data["maxtominbatch_msignal"])))), ((data["maxtominbatch_msignal"])))))))))), ((np.minimum(((np.minimum(((data["maxtominbatch_msignal"])), ((np.tanh((data["medianbatch_slices2"]))))))), ((np.where(np.cos((np.minimum(((data["signal"])), ((data["maxtominbatch_msignal"]))))) > -998, data["medianbatch_msignal"], data["signal_shift_+1"] )))))))))) +

                            0.050000*np.tanh(((data["signal_shift_-1_msignal"]) - (np.where(data["maxtominbatch"] > -998, np.where(((data["stdbatch_msignal"]) * 2.0) > -998, data["abs_minbatch_msignal"], data["minbatch_slices2_msignal"] ), np.minimum(((np.sin((data["mean_abs_chgbatch_slices2"])))), ((((data["signal_shift_-1_msignal"]) + ((-((data["stdbatch_msignal"])))))))) )))) +

                            0.050000*np.tanh(np.sin((((np.where(((data["meanbatch_msignal"]) * (((((((((((np.sin((data["maxbatch_msignal"]))) / 2.0)) > (((np.where(data["maxbatch_slices2_msignal"] > -998, data["maxbatch_msignal"], data["maxbatch_msignal"] )) + (data["abs_minbatch_slices2"]))))*1.)) + (data["maxbatch_msignal"]))) + (data["minbatch"]))/2.0))) > -998, data["minbatch_msignal"], np.tanh((((data["abs_minbatch_slices2"]) + (data["maxbatch_msignal"])))) )) * 2.0)))) +

                            0.050000*np.tanh(np.where((-((((data["minbatch_msignal"]) + (data["minbatch_msignal"]))))) > -998, np.cos((data["minbatch"])), np.sin(((((((np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((data["abs_avgbatch_slices2_msignal"])))) + ((((((data["minbatch_slices2_msignal"]) * (((data["signal_shift_-1_msignal"]) + (data["meanbatch_slices2_msignal"]))))) <= (data["abs_minbatch_msignal"]))*1.)))) + (np.cos((data["abs_avgbatch_slices2_msignal"]))))/2.0))) )) +

                            0.050000*np.tanh(np.minimum(((((data["stdbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2"])))), ((((data["signal_shift_-1"]) + (np.where(data["stdbatch_slices2_msignal"] > -998, data["minbatch_slices2"], data["stdbatch_slices2"] ))))))) +

                            0.050000*np.tanh(((np.cos((data["abs_avgbatch_msignal"]))) - (np.cos((np.where(data["abs_avgbatch_slices2_msignal"] > -998, np.sin((data["abs_avgbatch_msignal"])), data["signal"] )))))) +

                            0.050000*np.tanh(((np.sin((((data["rangebatch_msignal"]) - (np.sin((data["abs_maxbatch"]))))))) * 2.0)) +

                            0.050000*np.tanh(np.sin((np.minimum(((np.where(data["meanbatch_msignal"] > -998, np.sin((np.minimum((((-((data["abs_maxbatch_msignal"]))))), (((-((np.cos((np.minimum(((data["abs_minbatch_msignal"])), ((np.minimum(((data["maxtominbatch_msignal"])), ((data["signal_shift_-1"])))))))))))))))), data["rangebatch_msignal"] ))), ((data["abs_minbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(((((data["signal_shift_-1"]) - (np.where(np.cos((data["signal_shift_-1"])) > -998, data["medianbatch_msignal"], ((data["minbatch"]) * (data["meanbatch_slices2_msignal"])) )))) + ((-((data["maxbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(((np.sin((np.where(np.sin((np.where(data["maxtominbatch_slices2_msignal"] <= -998, data["maxbatch_slices2_msignal"], data["abs_minbatch_slices2_msignal"] ))) <= -998, data["minbatch_slices2"], np.minimum(((data["abs_minbatch_slices2_msignal"])), ((np.maximum(((data["maxbatch_slices2_msignal"])), ((((data["signal_shift_+1"]) * 2.0))))))) )))) * 2.0)) +

                            0.050000*np.tanh(((data["maxtominbatch_slices2_msignal"]) * (data["maxtominbatch_slices2_msignal"]))) +

                            0.050000*np.tanh((-((np.where(np.where(np.cos((data["abs_avgbatch_slices2_msignal"])) <= -998, data["maxtominbatch"], data["minbatch_slices2_msignal"] ) <= -998, np.cos((np.cos((data["signal_shift_-1"])))), data["stdbatch_slices2_msignal"] ))))) +

                            0.050000*np.tanh(np.where(np.minimum(((((data["meanbatch_slices2_msignal"]) - (data["meanbatch_slices2_msignal"])))), ((data["signal_shift_+1_msignal"]))) > -998, np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, np.sin((((data["abs_avgbatch_msignal"]) - (data["signal_shift_-1"])))), ((data["abs_avgbatch_msignal"]) - (data["signal_shift_-1"])) ), np.sin(((-((((data["signal_shift_-1"]) - (data["signal_shift_-1"]))))))) )) +

                            0.050000*np.tanh(((np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, data["maxtominbatch"], data["meanbatch_slices2_msignal"] )) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((np.where(np.cos((np.minimum(((data["abs_maxbatch_slices2"])), ((np.sin((data["maxbatch_slices2_msignal"]))))))) <= -998, data["minbatch"], ((data["maxbatch_slices2_msignal"]) - (((data["signal_shift_+1"]) / 2.0))) ))), ((np.sin((((np.maximum(((np.sin((data["stdbatch_msignal"])))), ((data["signal_shift_-1"])))) - (((data["maxbatch_slices2_msignal"]) - (((np.cos((np.sin((data["rangebatch_slices2_msignal"]))))) / 2.0))))))))))) +

                            0.050000*np.tanh(np.where((((-((np.minimum(((data["mean_abs_chgbatch_msignal"])), (((13.93050289154052734)))))))) * 2.0) > -998, np.cos((data["maxtominbatch_msignal"])), np.where(data["abs_minbatch_slices2_msignal"] > -998, data["abs_minbatch_slices2_msignal"], data["medianbatch_slices2"] ) )) +

                            0.050000*np.tanh(np.cos((np.where(data["abs_avgbatch_msignal"] <= -998, np.maximum(((data["abs_avgbatch_msignal"])), ((data["stdbatch_slices2"]))), np.where(data["abs_avgbatch_slices2"] > -998, data["abs_avgbatch_msignal"], np.cos((data["meanbatch_slices2"])) ) )))) +

                            0.050000*np.tanh(np.cos(((((((data["signal_shift_+1"]) * 2.0)) + (np.where(((((data["minbatch_slices2"]) - (np.where(data["maxtominbatch"] > -998, data["signal"], ((data["signal_shift_+1"]) * (((data["maxtominbatch_slices2"]) / 2.0))) )))) * 2.0) > -998, data["signal_shift_+1"], np.tanh((((data["maxtominbatch"]) * ((((((data["signal_shift_+1"]) * 2.0)) + (data["signal"]))/2.0))))) )))/2.0)))) +

                            0.050000*np.tanh(((np.minimum(((((data["maxtominbatch_slices2"]) - (np.sin((np.maximum(((data["maxbatch_slices2_msignal"])), ((data["minbatch_msignal"]))))))))), ((data["signal_shift_-1"])))) + (np.minimum(((np.where(data["abs_maxbatch"] > -998, np.maximum(((data["signal_shift_-1"])), ((data["signal_shift_-1"]))), ((data["maxtominbatch_slices2"]) - (np.sin((data["abs_maxbatch_slices2_msignal"])))) ))), ((np.maximum(((data["maxbatch_slices2_msignal"])), ((data["maxbatch_slices2_msignal"]))))))))) +

                            0.050000*np.tanh(np.sin((np.where(data["signal_shift_+1"] <= -998, data["signal_shift_+1_msignal"], np.sin((((np.sin((data["mean_abs_chgbatch_msignal"]))) + (data["minbatch_slices2"])))) )))) +

                            0.050000*np.tanh(((((np.sin((np.where(((data["meanbatch_msignal"]) * 2.0) <= -998, ((data["mean_abs_chgbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2"])), np.where(np.cos((data["minbatch_slices2_msignal"])) <= -998, np.maximum(((((data["signal_shift_+1"]) - (data["meanbatch_slices2_msignal"])))), ((data["mean_abs_chgbatch_slices2"]))), (-((((data["signal_shift_+1"]) - (np.cos((data["minbatch_msignal"]))))))) ) )))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.sin(((-((np.where(((data["maxtominbatch_msignal"]) * ((((((np.tanh((np.minimum(((data["rangebatch_slices2"])), ((np.where(data["mean_abs_chgbatch_msignal"] > -998, data["abs_maxbatch"], data["maxtominbatch_slices2_msignal"] ))))))) + (np.minimum(((data["signal_shift_-1"])), ((data["maxbatch_slices2"])))))) > (np.tanh((data["signal_shift_-1"]))))*1.))) > -998, data["maxtominbatch_slices2_msignal"], data["medianbatch_slices2"] ))))))) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) - (((data["maxbatch_msignal"]) * ((((((((((np.maximum((((((data["stdbatch_slices2_msignal"]) > (data["abs_maxbatch_slices2_msignal"]))*1.))), ((np.sin((data["signal_shift_-1"])))))) + (data["maxtominbatch_slices2_msignal"]))) * (data["signal_shift_-1"]))) + (data["rangebatch_msignal"]))) > (data["rangebatch_msignal"]))*1.)))))) +

                            0.050000*np.tanh(((np.sin((data["abs_avgbatch_slices2_msignal"]))) * (data["signal_shift_-1_msignal"]))) +

                            0.050000*np.tanh(((np.where(data["meanbatch_slices2"] > -998, np.where(((np.where(data["maxtominbatch_msignal"] > -998, data["minbatch_slices2_msignal"], data["signal"] )) * 2.0) > -998, data["minbatch_slices2_msignal"], ((data["minbatch_slices2_msignal"]) / 2.0) ), np.sin((np.cos((data["maxtominbatch_msignal"])))) )) + (((data["signal"]) * ((((np.sin((np.cos((data["maxtominbatch_msignal"]))))) + (data["maxbatch_msignal"]))/2.0)))))) +

                            0.050000*np.tanh(((data["minbatch_slices2_msignal"]) * (np.maximum(((((data["medianbatch_msignal"]) / 2.0))), ((((((data["abs_minbatch_slices2"]) * 2.0)) + ((((((data["minbatch_msignal"]) + ((-((((data["minbatch_msignal"]) * (((data["minbatch_slices2"]) / 2.0))))))))/2.0)) - ((((data["medianbatch_msignal"]) <= (data["abs_minbatch_slices2"]))*1.))))))))))) +

                            0.050000*np.tanh(np.minimum(((data["signal_shift_-1_msignal"])), ((((np.where(data["signal_shift_+1_msignal"] <= -998, data["minbatch"], np.minimum(((((np.minimum(((data["maxbatch_msignal"])), ((data["maxbatch_slices2_msignal"])))) * (data["rangebatch_msignal"])))), ((data["signal_shift_-1_msignal"]))) )) * 2.0))))) +

                            0.050000*np.tanh(((np.tanh((((data["minbatch_slices2"]) + (np.where(data["stdbatch_slices2"] > -998, data["abs_minbatch_msignal"], np.cos((data["abs_minbatch_msignal"])) )))))) - (((np.sin((((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) * 2.0)))) * 2.0)))) +

                            0.050000*np.tanh(np.where(data["signal"] <= -998, ((((-((data["minbatch"])))) + ((1.68646967411041260)))/2.0), ((((data["abs_avgbatch_slices2_msignal"]) * (data["medianbatch_msignal"]))) * (data["maxtominbatch_slices2"])) )) +

                            0.050000*np.tanh(np.minimum(((data["abs_maxbatch_msignal"])), ((((np.minimum(((np.sin((((data["signal"]) + ((((np.maximum(((np.tanh((data["abs_avgbatch_slices2"])))), ((((data["signal_shift_+1"]) / 2.0))))) + (data["signal_shift_+1"]))/2.0))))))), ((((np.minimum(((data["signal_shift_-1"])), ((((((data["signal_shift_+1"]) / 2.0)) * 2.0))))) - (np.tanh((data["stdbatch_slices2"])))))))) * 2.0))))) +

                            0.050000*np.tanh((-((data["abs_minbatch_msignal"])))) +

                            0.050000*np.tanh(((np.sin(((((np.minimum(((data["maxtominbatch_msignal"])), ((np.minimum(((((data["minbatch_slices2"]) / 2.0))), ((data["meanbatch_slices2_msignal"]))))))) + (data["minbatch_slices2_msignal"]))/2.0)))) - (data["rangebatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((data["medianbatch_msignal"]) * ((((((data["signal_shift_-1"]) + (data["signal_shift_-1"]))) + (((data["medianbatch_slices2"]) - (np.maximum(((data["medianbatch_msignal"])), ((((data["abs_maxbatch_slices2_msignal"]) * ((((np.maximum(((((data["maxbatch_slices2_msignal"]) - (data["abs_minbatch_slices2_msignal"])))), ((((data["abs_maxbatch_slices2_msignal"]) * (((data["medianbatch_msignal"]) * (data["maxbatch_slices2_msignal"])))))))) + (data["maxbatch_msignal"]))/2.0))))))))))/2.0)))) +

                            0.050000*np.tanh(np.sin((np.where(np.tanh((data["abs_minbatch_slices2"])) > -998, data["signal_shift_-1_msignal"], np.tanh((np.where(data["signal"] > -998, data["abs_minbatch_slices2"], data["abs_minbatch_msignal"] ))) )))) +

                            0.050000*np.tanh(((data["minbatch_slices2_msignal"]) * (np.cos((((((data["minbatch"]) - (np.maximum(((((np.where((((((data["abs_minbatch_slices2_msignal"]) > (data["minbatch"]))*1.)) * (data["minbatch_slices2_msignal"])) <= -998, data["minbatch"], data["stdbatch_slices2"] )) / 2.0))), (((((data["minbatch_slices2_msignal"]) > ((-((data["signal_shift_+1"])))))*1.))))))) - (data["signal_shift_+1"]))))))) +

                            0.050000*np.tanh(((((data["maxtominbatch_msignal"]) * (np.cos((data["minbatch_msignal"]))))) * (np.where(np.maximum(((((data["signal"]) * ((((data["mean_abs_chgbatch_msignal"]) <= (np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, data["signal_shift_+1"], data["abs_maxbatch_msignal"] )))*1.))))), ((data["minbatch"]))) <= -998, data["maxbatch_slices2"], data["signal_shift_+1"] )))) +

                            0.050000*np.tanh(np.where(np.cos((data["abs_avgbatch_msignal"])) <= -998, data["abs_avgbatch_msignal"], np.sin((np.maximum(((data["abs_avgbatch_msignal"])), ((data["mean_abs_chgbatch_slices2"]))))) )) +

                            0.050000*np.tanh(((((data["maxbatch_slices2_msignal"]) - ((((data["signal_shift_-1"]) + (np.where(data["abs_maxbatch_slices2"] > -998, (((data["signal"]) > ((2.0)))*1.), data["mean_abs_chgbatch_slices2_msignal"] )))/2.0)))) * (data["meanbatch_slices2"]))) +

                            0.050000*np.tanh(((((((np.sin((data["abs_minbatch_msignal"]))) * (data["abs_maxbatch_msignal"]))) + (data["stdbatch_slices2_msignal"]))) + (np.tanh((np.sin(((((((data["signal_shift_+1"]) / 2.0)) <= ((((data["abs_maxbatch_msignal"]) + ((6.15981245040893555)))/2.0)))*1.)))))))) +

                            0.050000*np.tanh(((np.where(data["minbatch_slices2"] > -998, data["signal_shift_+1"], np.where(np.where(data["signal"] <= -998, np.tanh((data["minbatch_slices2"])), data["abs_minbatch_slices2"] ) <= -998, data["minbatch_slices2"], np.maximum(((data["abs_maxbatch_msignal"])), ((data["abs_maxbatch"]))) ) )) + (data["minbatch_slices2"]))) +

                            0.050000*np.tanh((((data["medianbatch_slices2_msignal"]) + (((data["maxbatch_slices2_msignal"]) * (np.where(np.minimum(((((data["abs_maxbatch_slices2"]) - (data["signal_shift_-1_msignal"])))), (((-(((((data["minbatch_slices2"]) <= (np.sin((data["medianbatch_slices2_msignal"]))))*1.))))))) <= -998, data["maxbatch_slices2_msignal"], np.sin((np.minimum(((data["abs_maxbatch_slices2"])), ((data["abs_minbatch_slices2_msignal"]))))) )))))/2.0)) +

                            0.050000*np.tanh((-((np.maximum(((np.sin((data["signal_shift_-1"])))), ((((data["abs_avgbatch_slices2_msignal"]) - (np.sin(((((-((np.maximum((((-((np.minimum(((data["maxtominbatch"])), (((-(((((((data["signal_shift_-1_msignal"]) - (data["abs_avgbatch_slices2_msignal"]))) + (np.tanh((((data["signal_shift_-1"]) * 2.0)))))/2.0)))))))))))), ((data["maxtominbatch"]))))))) * ((5.0)))))))))))))) +

                            0.050000*np.tanh(np.minimum(((((data["meanbatch_msignal"]) - (data["abs_maxbatch"])))), ((np.cos((data["signal_shift_-1"])))))) +

                            0.050000*np.tanh(np.where(data["signal_shift_-1"] > -998, ((np.sin(((-((((data["signal_shift_-1"]) - (data["maxtominbatch_msignal"])))))))) * 2.0), (-((((data["signal_shift_-1"]) - (data["maxtominbatch_msignal"]))))) )) +

                            0.050000*np.tanh(np.sin((((data["signal_shift_-1_msignal"]) - (np.where((((data["abs_maxbatch_slices2"]) > (np.sin((((data["meanbatch_slices2"]) + (data["signal_shift_-1"]))))))*1.) > -998, data["maxbatch_msignal"], np.where(np.sin((np.minimum(((data["abs_maxbatch_slices2"])), (((((data["abs_maxbatch_slices2"]) > (data["abs_maxbatch_msignal"]))*1.)))))) <= -998, ((data["rangebatch_slices2_msignal"]) - (data["rangebatch_slices2_msignal"])), data["stdbatch_slices2"] ) )))))) +

                            0.050000*np.tanh(((((data["signal"]) * (data["abs_minbatch_slices2_msignal"]))) * (np.cos((np.where(data["maxtominbatch"] <= -998, data["abs_minbatch_slices2_msignal"], (((data["abs_maxbatch_msignal"]) + (np.cos((np.where((((data["abs_avgbatch_msignal"]) <= (data["maxtominbatch"]))*1.) <= -998, data["abs_maxbatch_msignal"], (((data["signal_shift_+1"]) + (data["stdbatch_msignal"]))/2.0) )))))/2.0) )))))) +

                            0.050000*np.tanh(np.sin((np.where((-((((data["mean_abs_chgbatch_slices2"]) * 2.0)))) > -998, data["abs_maxbatch"], (((data["mean_abs_chgbatch_slices2"]) <= (data["maxtominbatch_msignal"]))*1.) )))) +

                            0.050000*np.tanh(data["mean_abs_chgbatch_msignal"]) +

                            0.050000*np.tanh(np.cos((np.sin((np.cos((data["abs_maxbatch_slices2_msignal"]))))))) +

                            0.050000*np.tanh(((np.sin(((((((-((data["abs_maxbatch_slices2_msignal"])))) * 2.0)) * 2.0)))) * ((((4.24157476425170898)) * (np.minimum(((data["abs_maxbatch"])), ((((((data["abs_maxbatch_slices2_msignal"]) + ((((data["medianbatch_msignal"]) <= (np.tanh((data["medianbatch_msignal"]))))*1.)))) - (((data["medianbatch_msignal"]) * (np.sin((data["medianbatch_msignal"])))))))))))))) +

                            0.050000*np.tanh(np.where(np.where(((data["signal_shift_-1"]) * (data["abs_avgbatch_slices2_msignal"])) <= -998, data["mean_abs_chgbatch_slices2"], data["signal_shift_-1"] ) <= -998, np.where(data["minbatch_slices2_msignal"] <= -998, data["minbatch_slices2_msignal"], ((data["maxbatch_slices2_msignal"]) / 2.0) ), ((data["minbatch_slices2"]) * (np.cos((data["signal_shift_-1"])))) )) +

                            0.050000*np.tanh((-(((3.67669796943664551))))) +

                            0.050000*np.tanh(np.sin(((((data["abs_maxbatch_slices2"]) + (np.maximum(((data["mean_abs_chgbatch_msignal"])), ((np.where(data["abs_maxbatch"] > -998, ((data["signal_shift_+1"]) - (data["mean_abs_chgbatch_slices2_msignal"])), (((((np.where(data["signal"] > -998, data["stdbatch_slices2"], data["stdbatch_msignal"] )) > (data["mean_abs_chgbatch_slices2_msignal"]))*1.)) - (data["mean_abs_chgbatch_msignal"])) ))))))/2.0)))) +

                            0.050000*np.tanh(np.where(((data["mean_abs_chgbatch_slices2"]) * (data["signal_shift_+1"])) <= -998, data["signal_shift_-1_msignal"], data["signal_shift_+1"] )) +

                            0.050000*np.tanh(((data["maxbatch_slices2"]) * (np.sin((np.where(np.where(data["signal_shift_+1"] > -998, ((data["abs_avgbatch_slices2_msignal"]) - (data["abs_avgbatch_slices2_msignal"])), data["abs_avgbatch_slices2"] ) <= -998, data["minbatch_slices2_msignal"], ((data["abs_avgbatch_slices2_msignal"]) - (np.where(data["signal_shift_-1"] <= -998, data["mean_abs_chgbatch_msignal"], data["signal_shift_+1"] ))) )))))) +

                            0.050000*np.tanh(np.where(((data["minbatch_msignal"]) * 2.0) <= -998, data["stdbatch_slices2"], np.minimum(((np.minimum(((data["rangebatch_msignal"])), ((np.minimum(((data["maxtominbatch_msignal"])), ((data["minbatch"])))))))), ((np.tanh((data["minbatch_slices2"]))))) )) +

                            0.050000*np.tanh(np.sin((np.where(data["signal_shift_-1"] > -998, np.sin((np.tanh((np.sin((np.sin((np.where(np.sin((np.sin((data["signal_shift_-1_msignal"])))) > -998, data["abs_maxbatch"], np.where(data["stdbatch_msignal"] > -998, np.sin((((data["abs_maxbatch"]) + (data["abs_maxbatch"])))), data["mean_abs_chgbatch_msignal"] ) ))))))))), data["meanbatch_slices2"] )))) +

                            0.050000*np.tanh(((data["abs_avgbatch_slices2_msignal"]) * (((data["signal_shift_-1_msignal"]) * (((np.where(np.sin((np.where(np.sin((data["signal_shift_-1"])) > -998, data["abs_minbatch_slices2"], data["abs_avgbatch_slices2_msignal"] ))) > -998, data["abs_minbatch_slices2"], data["meanbatch_slices2"] )) + (((np.sin((((data["abs_maxbatch"]) - ((((data["abs_maxbatch_msignal"]) > (data["abs_maxbatch_slices2_msignal"]))*1.)))))) * 2.0)))))))) +

                            0.050000*np.tanh(((np.sin((((np.maximum(((data["abs_maxbatch_msignal"])), ((((np.sin((data["abs_minbatch_slices2_msignal"]))) * (data["abs_maxbatch_msignal"])))))) * (data["abs_avgbatch_slices2_msignal"]))))) * (np.where((9.0) <= -998, ((data["abs_maxbatch"]) - (data["abs_maxbatch_msignal"])), data["stdbatch_slices2"] )))) +

                            0.050000*np.tanh(((np.tanh((np.minimum(((data["signal"])), ((np.minimum(((data["maxbatch_msignal"])), ((np.where(((((data["medianbatch_msignal"]) - (data["maxtominbatch_slices2"]))) * 2.0) <= -998, ((((((((data["medianbatch_msignal"]) - (data["maxtominbatch_slices2"]))) * 2.0)) - (data["minbatch_slices2"]))) * 2.0), ((((data["medianbatch_msignal"]) - (data["maxtominbatch_slices2"]))) * 2.0) )))))))))) * 2.0)) +

                            0.050000*np.tanh(((data["signal"]) * (np.cos(((-((np.minimum(((data["maxtominbatch_msignal"])), ((np.sin(((((4.57688331604003906)) - (data["maxtominbatch_msignal"]))))))))))))))) +

                            0.050000*np.tanh(((np.cos((data["abs_avgbatch_msignal"]))) * ((((8.0)) - (((((data["maxtominbatch_msignal"]) - (np.sin((((data["abs_minbatch_slices2_msignal"]) * 2.0)))))) + ((-(((-((data["maxbatch_slices2_msignal"]))))))))))))) +

                            0.050000*np.tanh(((np.maximum(((data["abs_avgbatch_msignal"])), (((8.78546714782714844))))) * (np.sin((((np.where(np.where(np.sin((((data["abs_avgbatch_slices2_msignal"]) * (data["abs_maxbatch_slices2_msignal"])))) > -998, data["abs_maxbatch"], data["abs_avgbatch_slices2_msignal"] ) <= -998, data["minbatch_slices2_msignal"], np.where(data["stdbatch_slices2"] > -998, data["meanbatch_slices2"], np.minimum(((((data["abs_avgbatch_slices2_msignal"]) * (data["abs_maxbatch_slices2_msignal"])))), ((data["abs_maxbatch_slices2_msignal"]))) ) )) * (data["abs_avgbatch_slices2_msignal"]))))))) +

                            0.050000*np.tanh(np.sin((np.maximum(((data["abs_maxbatch"])), ((((data["abs_maxbatch_msignal"]) / 2.0))))))) +

                            0.050000*np.tanh(np.minimum(((np.tanh((np.where(data["rangebatch_slices2_msignal"] > -998, np.where(np.sin((np.where(data["abs_maxbatch"] > -998, data["mean_abs_chgbatch_slices2"], np.sin((data["abs_maxbatch"])) ))) <= -998, data["signal_shift_+1_msignal"], (((data["abs_maxbatch"]) + (data["rangebatch_slices2_msignal"]))/2.0) ), ((((data["signal_shift_+1"]) / 2.0)) * 2.0) ))))), ((np.sin((data["abs_maxbatch"])))))) +

                            0.050000*np.tanh(((np.where(data["meanbatch_slices2_msignal"] > -998, data["signal_shift_-1"], data["abs_maxbatch"] )) + (data["maxbatch_slices2"]))) +

                            0.050000*np.tanh(np.minimum(((np.minimum(((np.where(np.maximum(((data["signal_shift_-1_msignal"])), ((((((-((data["medianbatch_msignal"])))) + (data["abs_minbatch_slices2_msignal"]))/2.0)))) > -998, data["meanbatch_slices2_msignal"], np.where(data["minbatch_msignal"] > -998, data["meanbatch_slices2_msignal"], data["minbatch_msignal"] ) ))), (((-((np.where(data["meanbatch_slices2_msignal"] <= -998, data["medianbatch_slices2"], data["meanbatch_slices2_msignal"] ))))))))), ((data["abs_minbatch_slices2"])))) +

                            0.050000*np.tanh(np.sin((np.minimum(((np.minimum(((((data["signal_shift_-1"]) * 2.0))), ((data["abs_minbatch_slices2_msignal"]))))), ((((data["signal_shift_+1"]) / 2.0))))))) +

                            0.050000*np.tanh((((np.minimum((((((data["abs_avgbatch_msignal"]) <= (data["minbatch_msignal"]))*1.))), (((((data["meanbatch_slices2"]) > (data["abs_minbatch_slices2"]))*1.))))) <= ((-((data["meanbatch_msignal"])))))*1.)) +

                            0.050000*np.tanh(np.where(np.minimum(((data["minbatch_msignal"])), ((np.tanh(((-((data["abs_maxbatch_slices2"])))))))) > -998, data["medianbatch_slices2_msignal"], data["minbatch"] )) +

                            0.050000*np.tanh((((-((((np.maximum(((data["stdbatch_slices2"])), ((((((np.sin((((data["maxbatch_msignal"]) * 2.0)))) * 2.0)) * 2.0))))) - (data["minbatch_msignal"])))))) * (np.sin((((data["maxbatch_msignal"]) * 2.0)))))) +

                            0.050000*np.tanh(np.minimum(((np.cos(((((data["signal_shift_-1"]) + (np.maximum(((np.minimum((((-((((data["stdbatch_msignal"]) - (data["signal_shift_-1_msignal"]))))))), ((np.minimum(((data["meanbatch_slices2"])), ((data["meanbatch_slices2"])))))))), ((((data["signal"]) * 2.0))))))/2.0))))), ((data["abs_maxbatch"])))) +

                            0.050000*np.tanh(np.where(data["medianbatch_slices2"] > -998, np.sin(((-((np.sin((data["minbatch"]))))))), np.cos((data["abs_avgbatch_slices2_msignal"])) )) +

                            0.050000*np.tanh(np.sin((np.minimum(((np.maximum(((data["meanbatch_msignal"])), ((data["abs_maxbatch"]))))), ((((((np.where(data["mean_abs_chgbatch_slices2"] <= -998, data["abs_minbatch_slices2"], (-((np.where(data["minbatch"] > -998, data["abs_maxbatch_slices2_msignal"], data["signal_shift_-1"] )))) )) * 2.0)) * 2.0))))))) +

                            0.050000*np.tanh(np.cos((np.where(((data["abs_maxbatch_slices2"]) - (np.cos((data["abs_avgbatch_msignal"])))) > -998, data["abs_avgbatch_msignal"], data["rangebatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(((np.minimum((((((((np.sin((data["stdbatch_slices2_msignal"]))) <= (data["signal_shift_+1_msignal"]))*1.)) * (np.sin((data["medianbatch_slices2_msignal"])))))), (((-((np.cos((data["signal_shift_+1"]))))))))) * 2.0)) +

                            0.050000*np.tanh(np.where(data["abs_avgbatch_slices2"] > -998, data["signal_shift_+1"], ((((((data["stdbatch_slices2"]) + (data["signal_shift_+1"]))/2.0)) <= (data["maxbatch_slices2_msignal"]))*1.) )) +

                            0.050000*np.tanh(((np.sin((((data["rangebatch_slices2"]) + (((np.sin(((9.0)))) - (((np.where((((((np.tanh(((12.98753833770751953)))) - (data["maxtominbatch"]))) <= (np.tanh((data["rangebatch_slices2"]))))*1.) <= -998, data["signal_shift_-1"], data["abs_maxbatch_msignal"] )) * (np.tanh((data["meanbatch_slices2"]))))))))))) * 2.0)) +

                            0.050000*np.tanh(((np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((np.where(((data["minbatch_slices2_msignal"]) * (data["minbatch_slices2_msignal"])) <= -998, ((np.where(data["abs_avgbatch_slices2_msignal"] <= -998, (6.0), data["abs_avgbatch_msignal"] )) * (data["abs_avgbatch_slices2_msignal"])), np.where(data["abs_avgbatch_slices2_msignal"] <= -998, data["signal"], np.sin((np.sin((data["abs_minbatch_slices2_msignal"])))) ) ))))) * ((((6.0)) * (data["signal"]))))) +

                            0.050000*np.tanh(((np.where(np.where(data["maxtominbatch_slices2"] > -998, data["abs_minbatch_slices2_msignal"], (((((np.sin((np.sin((np.sin((np.tanh(((-(((((8.85292625427246094)) * 2.0))))))))))))) + (np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((data["abs_avgbatch_slices2_msignal"])))))/2.0)) * (data["abs_minbatch_slices2_msignal"])) ) <= -998, data["signal_shift_+1"], ((np.sin((data["abs_maxbatch_slices2"]))) * 2.0) )) * 2.0)) +

                            0.050000*np.tanh(np.where(np.minimum(((np.minimum(((((data["medianbatch_slices2_msignal"]) * 2.0))), ((data["medianbatch_slices2_msignal"]))))), ((data["rangebatch_slices2"]))) <= -998, data["medianbatch_slices2_msignal"], data["meanbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(np.cos((np.where(((((((((data["abs_minbatch_msignal"]) > (data["maxbatch_slices2"]))*1.)) * (((np.sin((data["minbatch_msignal"]))) * 2.0)))) + (data["mean_abs_chgbatch_msignal"]))/2.0) > -998, ((data["minbatch_msignal"]) * 2.0), np.sin((data["minbatch_msignal"])) )))) +

                            0.050000*np.tanh(np.where(data["abs_maxbatch_slices2"] > -998, data["signal_shift_-1"], np.sin((data["abs_maxbatch_slices2"])) )) +

                            0.050000*np.tanh(np.where(data["abs_avgbatch_slices2_msignal"] <= -998, np.tanh((np.minimum(((data["meanbatch_slices2_msignal"])), ((((((data["meanbatch_slices2_msignal"]) * (data["abs_avgbatch_slices2"]))) / 2.0)))))), np.sin(((((data["abs_avgbatch_slices2"]) + (data["abs_minbatch_msignal"]))/2.0))) )) +

                            0.050000*np.tanh(((data["maxbatch_slices2_msignal"]) - ((((data["signal_shift_+1"]) > (((data["signal_shift_+1"]) / 2.0)))*1.)))) +

                            0.050000*np.tanh(np.sin((((((data["signal"]) + (data["signal_shift_-1"]))) - ((((np.cos((np.where(data["rangebatch_slices2"] > -998, (((data["signal"]) + (np.sin((data["signal_shift_-1"]))))/2.0), data["maxbatch_slices2_msignal"] )))) <= (((data["signal"]) + (data["signal_shift_-1"]))))*1.)))))) +

                            0.050000*np.tanh(np.cos((((np.maximum(((data["meanbatch_slices2"])), ((data["meanbatch_slices2"])))) - (data["stdbatch_msignal"]))))) +

                            0.050000*np.tanh((-((np.cos((np.where(data["meanbatch_slices2_msignal"] > -998, data["minbatch_msignal"], data["abs_avgbatch_slices2_msignal"] ))))))) +

                            0.050000*np.tanh(((np.where(data["medianbatch_slices2"] <= -998, np.sin((data["meanbatch_slices2_msignal"])), ((((np.sin((data["meanbatch_slices2_msignal"]))) + (np.cos((np.sin((data["maxtominbatch_slices2_msignal"]))))))) * (data["signal_shift_-1_msignal"])) )) * (data["signal"]))) +

                            0.050000*np.tanh(((np.minimum(((np.sin((data["mean_abs_chgbatch_slices2_msignal"])))), ((np.minimum(((data["meanbatch_msignal"])), ((np.sin((np.cos(((((data["mean_abs_chgbatch_slices2_msignal"]) + ((((data["maxtominbatch_slices2"]) + (data["stdbatch_msignal"]))/2.0)))/2.0)))))))))))) - (((np.minimum(((data["meanbatch_slices2"])), ((np.sin((data["signal_shift_+1"])))))) * 2.0)))) +

                            0.050000*np.tanh(np.sin((((np.sin((data["meanbatch_msignal"]))) * ((((np.sin(((((((data["abs_avgbatch_msignal"]) / 2.0)) + (np.where((((data["mean_abs_chgbatch_slices2"]) + (np.sin((np.sin((data["minbatch"]))))))/2.0) > -998, data["maxtominbatch_slices2"], np.sin(((-((np.minimum(((data["signal_shift_+1"])), ((data["meanbatch_msignal"])))))))) )))/2.0)))) + (data["signal_shift_-1_msignal"]))/2.0)))))) +

                            0.050000*np.tanh(np.sin((np.sin((((data["meanbatch_slices2"]) - (np.minimum(((data["abs_minbatch_msignal"])), ((data["abs_minbatch_msignal"])))))))))) +

                            0.050000*np.tanh(np.cos((((data["medianbatch_slices2"]) - (np.where(((data["medianbatch_slices2"]) + (data["abs_avgbatch_msignal"])) > -998, data["minbatch_msignal"], np.minimum(((data["medianbatch_msignal"])), (((-((np.minimum(((((data["medianbatch_slices2"]) * (np.minimum(((data["medianbatch_msignal"])), ((data["medianbatch_slices2"]))))))), ((np.minimum(((data["abs_maxbatch_msignal"])), ((data["maxbatch_msignal"])))))))))))) )))))) +

                            0.050000*np.tanh(((np.cos((((np.minimum(((np.where(data["mean_abs_chgbatch_slices2"] > -998, data["medianbatch_slices2"], data["maxbatch_msignal"] ))), ((np.minimum(((data["meanbatch_slices2_msignal"])), ((data["meanbatch_slices2_msignal"]))))))) * (np.where(np.maximum(((data["rangebatch_slices2_msignal"])), ((data["medianbatch_slices2"]))) > -998, np.maximum(((data["mean_abs_chgbatch_slices2"])), ((data["medianbatch_slices2"]))), np.cos((np.tanh((data["abs_maxbatch"])))) )))))) * 2.0)) +

                            0.050000*np.tanh(np.tanh((np.where(np.minimum(((np.minimum(((np.tanh((data["meanbatch_slices2_msignal"])))), ((data["minbatch_slices2_msignal"]))))), ((data["stdbatch_slices2_msignal"]))) > -998, np.tanh((data["maxtominbatch"])), np.minimum(((data["meanbatch_slices2_msignal"])), ((data["meanbatch_slices2_msignal"]))) )))) +

                            0.050000*np.tanh(np.sin(((3.0)))) +

                            0.050000*np.tanh(np.where(data["abs_avgbatch_slices2"] > -998, np.sin((data["abs_maxbatch"])), data["maxtominbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(((((((((data["medianbatch_msignal"]) - (np.sin((np.tanh((data["medianbatch_msignal"]))))))) <= (((np.cos(((((data["minbatch_slices2_msignal"]) > (np.tanh((data["stdbatch_slices2"]))))*1.)))) * ((((data["mean_abs_chgbatch_slices2"]) + (data["signal"]))/2.0)))))*1.)) + (np.cos((data["maxtominbatch_msignal"]))))/2.0)) +

                            0.050000*np.tanh(np.minimum(((data["meanbatch_msignal"])), ((data["signal"])))) +

                            0.050000*np.tanh(np.minimum(((data["signal_shift_-1"])), ((np.maximum(((((data["signal_shift_+1"]) * 2.0))), ((np.minimum(((data["stdbatch_slices2"])), (((((((data["abs_minbatch_msignal"]) / 2.0)) > (data["abs_avgbatch_slices2_msignal"]))*1.))))))))))) +

                            0.050000*np.tanh((-(((((data["abs_avgbatch_msignal"]) > (data["signal_shift_+1"]))*1.))))) +

                            0.050000*np.tanh(((data["abs_maxbatch_slices2"]) + (np.where((((data["maxbatch_slices2"]) + (data["abs_maxbatch_slices2"]))/2.0) <= -998, (((data["abs_maxbatch"]) <= (data["maxtominbatch_msignal"]))*1.), (-((np.maximum((((7.0))), (((((data["maxbatch_slices2"]) + (data["mean_abs_chgbatch_msignal"]))/2.0))))))) )))) +

                            0.050000*np.tanh(np.tanh((data["meanbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(np.where(data["signal_shift_-1"] <= -998, np.where(data["signal_shift_-1"] <= -998, data["maxbatch_slices2"], np.sin((np.where(data["meanbatch_slices2_msignal"] > -998, data["abs_maxbatch"], ((data["stdbatch_slices2"]) * 2.0) ))) ), np.sin((np.where(np.sin((data["abs_maxbatch"])) > -998, data["abs_maxbatch"], ((data["maxtominbatch_slices2"]) * 2.0) ))) )))  

    

    def GP_class_9(self,data):

        return self.Output( -3.603840 +

                            0.050000*np.tanh(np.where(data["meanbatch_msignal"] <= -998, np.minimum((((-(((((-((data["stdbatch_msignal"])))) * 2.0)))))), ((data["signal_shift_+1"]))), (-((data["stdbatch_msignal"]))) )) +

                            0.050000*np.tanh(((data["stdbatch_msignal"]) * (data["minbatch_msignal"]))) +

                            0.050000*np.tanh(((((data["signal"]) - (data["medianbatch_slices2_msignal"]))) - (((data["maxbatch_msignal"]) + (data["meanbatch_slices2"]))))) +

                            0.050000*np.tanh(np.where(np.tanh(((((((data["medianbatch_slices2"]) / 2.0)) > (data["rangebatch_msignal"]))*1.))) > -998, data["signal"], data["mean_abs_chgbatch_msignal"] )) +

                            0.050000*np.tanh(np.tanh((np.where(data["signal"] > -998, (-((data["abs_avgbatch_slices2_msignal"]))), (((((data["signal_shift_+1"]) - ((((data["maxbatch_slices2"]) > (data["abs_avgbatch_slices2"]))*1.)))) + (((np.cos((data["signal"]))) + (((np.cos((data["signal"]))) + (np.tanh((data["signal_shift_+1"]))))))))/2.0) )))) +

                            0.050000*np.tanh(((data["medianbatch_slices2"]) * ((((data["rangebatch_msignal"]) + (np.maximum(((np.minimum(((data["abs_maxbatch"])), ((((data["signal_shift_+1"]) + (data["minbatch_msignal"]))))))), ((np.minimum((((4.0))), ((np.where(data["maxbatch_slices2_msignal"] <= -998, data["medianbatch_slices2"], np.minimum(((data["maxbatch_slices2"])), ((data["abs_minbatch_msignal"]))) )))))))))/2.0)))) +

                            0.050000*np.tanh(np.where(data["signal_shift_+1"] > -998, data["medianbatch_slices2"], data["signal_shift_+1"] )) +

                            0.050000*np.tanh(((np.where(data["stdbatch_slices2"] > -998, data["abs_avgbatch_msignal"], np.maximum(((data["abs_avgbatch_msignal"])), (((-((np.cos(((((data["rangebatch_slices2"]) > (data["maxtominbatch_msignal"]))*1.))))))))) )) * (np.minimum(((data["minbatch"])), (((((data["medianbatch_slices2"]) + ((((((data["minbatch_slices2"]) - (np.cos((data["minbatch"]))))) + (data["abs_avgbatch_msignal"]))/2.0)))/2.0))))))) +

                            0.050000*np.tanh((((((np.where(np.where(data["meanbatch_slices2"] <= -998, data["meanbatch_msignal"], ((data["signal"]) - (((data["rangebatch_msignal"]) / 2.0))) ) > -998, data["meanbatch_slices2"], (((data["meanbatch_slices2"]) <= (((data["abs_maxbatch_slices2"]) - (data["maxtominbatch_msignal"]))))*1.) )) + (data["signal"]))/2.0)) - (np.sin((data["medianbatch_msignal"]))))) +

                            0.050000*np.tanh(np.minimum(((np.where(data["signal"] > -998, data["signal_shift_-1"], data["signal_shift_+1_msignal"] ))), ((((np.where(((((data["maxbatch_slices2_msignal"]) * (np.minimum((((-((data["maxtominbatch"]))))), ((data["meanbatch_slices2"])))))) / 2.0) > -998, (((data["signal"]) <= (data["abs_maxbatch"]))*1.), ((np.where(data["rangebatch_slices2"] <= -998, ((data["signal_shift_-1"]) - (data["abs_avgbatch_msignal"])), data["signal"] )) / 2.0) )) * 2.0))))) +

                            0.050000*np.tanh(((data["signal_shift_-1_msignal"]) - (data["stdbatch_msignal"]))) +

                            0.050000*np.tanh(np.minimum(((((((data["rangebatch_msignal"]) - (data["abs_maxbatch_slices2"]))) * (data["abs_avgbatch_msignal"])))), (((-((np.minimum((((-((np.where(data["rangebatch_slices2"] > -998, data["meanbatch_msignal"], data["signal"] )))))), ((np.minimum(((data["abs_avgbatch_msignal"])), ((data["abs_avgbatch_msignal"]))))))))))))) +

                            0.050000*np.tanh(((((data["signal"]) - (np.maximum(((np.where(data["signal"] > -998, data["signal_shift_+1"], np.cos((data["signal"])) ))), (((-((np.where(data["maxtominbatch"] > -998, data["abs_avgbatch_slices2_msignal"], np.sin((data["signal_shift_-1"])) )))))))))) / 2.0)) +

                            0.050000*np.tanh(((((data["maxbatch_slices2"]) * 2.0)) + (np.cos(((6.0)))))) +

                            0.050000*np.tanh(np.where((((((data["abs_maxbatch_msignal"]) <= (((np.sin((data["stdbatch_slices2"]))) * 2.0)))*1.)) * 2.0) <= -998, np.sin((((np.where(data["abs_minbatch_slices2_msignal"] <= -998, ((((data["signal"]) - (data["maxbatch_msignal"]))) * 2.0), ((data["minbatch_msignal"]) * 2.0) )) * (data["signal_shift_-1"])))), ((data["minbatch_slices2"]) - (np.where(data["mean_abs_chgbatch_msignal"] > -998, data["maxtominbatch_slices2_msignal"], data["signal_shift_-1"] ))) )) +

                            0.050000*np.tanh(np.maximum(((((data["signal_shift_-1_msignal"]) + (((data["maxbatch_slices2"]) - (np.where(data["meanbatch_slices2"] > -998, data["mean_abs_chgbatch_msignal"], (12.20609951019287109) ))))))), ((np.minimum(((data["maxtominbatch_slices2"])), ((((((-((data["maxbatch_slices2"])))) <= ((-(((((data["meanbatch_slices2"]) <= (np.sin(((-((data["signal_shift_-1_msignal"])))))))*1.))))))*1.)))))))) +

                            0.050000*np.tanh((-((np.where((-(((((-((np.where((-((data["minbatch_slices2_msignal"]))) > -998, data["abs_avgbatch_msignal"], np.where(data["maxtominbatch_slices2"] > -998, np.where(data["abs_minbatch_msignal"] > -998, data["abs_avgbatch_msignal"], data["signal"] ), np.where(((data["minbatch_slices2_msignal"]) - (data["signal"])) <= -998, data["maxbatch_msignal"], data["meanbatch_slices2"] ) ) ))))) * 2.0)))) > -998, data["abs_avgbatch_msignal"], data["abs_avgbatch_msignal"] ))))) +

                            0.050000*np.tanh(np.where(np.sin((data["maxtominbatch_msignal"])) <= -998, ((data["maxbatch_slices2"]) / 2.0), data["medianbatch_slices2"] )) +

                            0.050000*np.tanh(data["signal_shift_+1"]) +

                            0.050000*np.tanh(np.where(data["meanbatch_slices2"] <= -998, data["meanbatch_msignal"], (((-(((((data["minbatch"]) > (data["meanbatch_slices2_msignal"]))*1.))))) - (((data["abs_avgbatch_slices2_msignal"]) * 2.0))) )) +

                            0.050000*np.tanh(((np.minimum(((data["minbatch_slices2"])), ((data["maxtominbatch_slices2_msignal"])))) * (np.cos((np.where(data["maxtominbatch"] > -998, data["meanbatch_slices2_msignal"], (((-((data["signal_shift_+1_msignal"])))) * (data["abs_minbatch_slices2_msignal"])) )))))) +

                            0.050000*np.tanh(((((((np.sin((data["maxbatch_slices2_msignal"]))) <= ((-((data["medianbatch_msignal"])))))*1.)) <= (((data["signal_shift_+1_msignal"]) * (data["medianbatch_msignal"]))))*1.)) +

                            0.050000*np.tanh((-((np.where(data["maxtominbatch_slices2_msignal"] <= -998, np.minimum(((data["minbatch_msignal"])), ((np.cos((data["medianbatch_slices2_msignal"]))))), ((data["abs_minbatch_slices2_msignal"]) * (np.sin((data["mean_abs_chgbatch_msignal"])))) ))))) +

                            0.050000*np.tanh(((data["abs_avgbatch_msignal"]) * (np.minimum(((data["abs_avgbatch_msignal"])), (((((((((data["minbatch"]) + (data["maxtominbatch"]))/2.0)) * (data["medianbatch_msignal"]))) - (np.maximum(((np.where(data["maxbatch_slices2_msignal"] > -998, ((((((data["stdbatch_slices2"]) > (data["maxtominbatch"]))*1.)) > (data["minbatch"]))*1.), data["abs_avgbatch_slices2"] ))), ((data["rangebatch_slices2_msignal"]))))))))))) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) * (np.sin((data["signal"]))))) +

                            0.050000*np.tanh((((-((data["abs_avgbatch_slices2_msignal"])))) - ((((((data["abs_avgbatch_slices2_msignal"]) / 2.0)) + (np.cos(((((-((data["rangebatch_msignal"])))) - ((((data["minbatch_slices2_msignal"]) + (data["signal_shift_-1"]))/2.0)))))))/2.0)))) +

                            0.050000*np.tanh(np.where(np.sin((data["abs_avgbatch_slices2"])) <= -998, data["abs_avgbatch_slices2"], np.sin((data["maxbatch_slices2"])) )) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) - (((((((np.tanh(((((((((np.sin((data["mean_abs_chgbatch_slices2_msignal"]))) * 2.0)) * (np.where(data["medianbatch_slices2_msignal"] > -998, data["signal"], ((data["signal"]) - (data["abs_maxbatch"])) )))) <= (((data["abs_maxbatch_msignal"]) - (((((0.31871804594993591)) <= (data["abs_maxbatch"]))*1.)))))*1.)))) * 2.0)) * (data["abs_maxbatch"]))) * 2.0)))) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) - (((data["abs_avgbatch_slices2_msignal"]) + (data["signal_shift_-1"]))))) +

                            0.050000*np.tanh((-((((data["abs_avgbatch_msignal"]) - (((np.tanh((((data["abs_avgbatch_msignal"]) - (((np.tanh((((data["minbatch_slices2"]) * (((((((data["maxbatch_slices2"]) / 2.0)) / 2.0)) - (np.sin((np.sin((data["abs_maxbatch_slices2"]))))))))))) / 2.0)))))) / 2.0))))))) +

                            0.050000*np.tanh(data["signal"]) +

                            0.050000*np.tanh(np.where((((((data["abs_maxbatch"]) > (((((data["abs_avgbatch_msignal"]) / 2.0)) + (data["abs_avgbatch_slices2_msignal"]))))*1.)) * 2.0) > -998, np.where(data["medianbatch_msignal"] > -998, (-((data["abs_avgbatch_msignal"]))), np.where(np.where(data["meanbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_msignal"], data["meanbatch_msignal"] ) <= -998, np.where(data["meanbatch_msignal"] <= -998, data["rangebatch_slices2"], np.cos((data["medianbatch_msignal"])) ), data["maxtominbatch_slices2"] ) ), data["stdbatch_slices2"] )) +

                            0.050000*np.tanh(np.sin((np.where(np.sin((np.where((((np.sin((data["mean_abs_chgbatch_msignal"]))) <= (data["medianbatch_msignal"]))*1.) > -998, data["mean_abs_chgbatch_msignal"], data["abs_maxbatch_slices2_msignal"] ))) <= -998, (((data["medianbatch_msignal"]) + (np.cos((np.cos((data["medianbatch_slices2"]))))))/2.0), data["mean_abs_chgbatch_msignal"] )))) +

                            0.050000*np.tanh(data["signal_shift_+1"]) +

                            0.050000*np.tanh((-((data["medianbatch_msignal"])))) +

                            0.050000*np.tanh(np.minimum(((data["signal_shift_-1"])), ((np.where(data["minbatch"] <= -998, (((np.where(data["abs_avgbatch_slices2_msignal"] <= -998, (-((data["abs_avgbatch_slices2_msignal"]))), (((data["abs_avgbatch_slices2_msignal"]) > (data["medianbatch_slices2"]))*1.) )) > (data["medianbatch_slices2"]))*1.), ((data["abs_avgbatch_slices2_msignal"]) * (data["abs_minbatch_msignal"])) ))))) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) * (np.sin((np.where(data["abs_minbatch_msignal"] > -998, data["abs_maxbatch_slices2_msignal"], data["rangebatch_slices2_msignal"] )))))) +

                            0.050000*np.tanh(((data["minbatch"]) + ((-(((-((np.where(((data["minbatch"]) - ((((((data["abs_maxbatch_msignal"]) - (data["minbatch"]))) <= (data["abs_minbatch_slices2_msignal"]))*1.))) > -998, data["signal"], data["abs_maxbatch"] )))))))))) +

                            0.050000*np.tanh(np.where((10.0) > -998, np.cos((data["abs_minbatch_slices2_msignal"])), np.cos((np.sin((np.where(data["maxbatch_slices2"] > -998, data["stdbatch_msignal"], np.sin((np.cos((data["abs_avgbatch_msignal"])))) ))))) )) +

                            0.050000*np.tanh(((np.cos((np.maximum(((data["medianbatch_slices2_msignal"])), ((data["stdbatch_msignal"])))))) * ((((((((-(((((data["stdbatch_msignal"]) + ((-((((data["maxbatch_msignal"]) / 2.0))))))/2.0))))) <= (data["maxbatch_msignal"]))*1.)) + (data["mean_abs_chgbatch_msignal"]))/2.0)))) +

                            0.050000*np.tanh((((-((np.where(data["rangebatch_slices2_msignal"] > -998, data["abs_avgbatch_msignal"], (-((np.where(((data["rangebatch_msignal"]) + ((((data["abs_maxbatch_msignal"]) > (np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], np.minimum(((np.where(data["minbatch_slices2_msignal"] > -998, (((data["stdbatch_slices2_msignal"]) > (data["minbatch_slices2"]))*1.), data["abs_avgbatch_msignal"] ))), ((data["rangebatch_msignal"]))) )))*1.))) > -998, data["maxbatch_slices2_msignal"], data["abs_avgbatch_msignal"] )))) ))))) * 2.0)) +

                            0.050000*np.tanh(np.where(data["stdbatch_slices2"] <= -998, data["maxtominbatch_slices2_msignal"], ((((((data["minbatch_msignal"]) * 2.0)) + ((((np.cos((((data["maxtominbatch_slices2_msignal"]) / 2.0)))) + (((data["abs_avgbatch_msignal"]) * 2.0)))/2.0)))) * (np.cos((data["meanbatch_msignal"])))) )) +

                            0.050000*np.tanh(((np.where(data["maxbatch_slices2_msignal"] <= -998, ((data["abs_avgbatch_slices2_msignal"]) * (data["maxbatch_slices2_msignal"])), data["minbatch_slices2"] )) + (((np.sin((np.where(np.maximum(((data["signal_shift_-1"])), ((np.where(data["stdbatch_slices2"] > -998, data["maxbatch_slices2_msignal"], data["mean_abs_chgbatch_msignal"] )))) > -998, data["maxbatch_msignal"], np.sin((data["mean_abs_chgbatch_msignal"])) )))) - (((data["abs_avgbatch_slices2_msignal"]) * (data["maxbatch_slices2_msignal"]))))))) +

                            0.050000*np.tanh(((np.where(np.where(np.cos((np.minimum(((data["abs_avgbatch_msignal"])), ((((data["mean_abs_chgbatch_slices2"]) * (data["meanbatch_slices2_msignal"]))))))) > -998, data["medianbatch_slices2_msignal"], data["meanbatch_msignal"] ) > -998, ((data["abs_avgbatch_slices2_msignal"]) * (data["minbatch"])), (((-((data["meanbatch_msignal"])))) - (data["signal_shift_+1"])) )) + ((((data["medianbatch_msignal"]) + (np.sin((data["medianbatch_msignal"]))))/2.0)))) +

                            0.050000*np.tanh(np.sin(((((-((np.where(((data["maxtominbatch_slices2"]) - (data["abs_avgbatch_msignal"])) <= -998, ((np.minimum((((8.0))), ((data["signal_shift_+1_msignal"])))) + (data["maxtominbatch_slices2"])), data["abs_avgbatch_slices2_msignal"] ))))) * (((((-(((-((data["abs_avgbatch_msignal"]))))))) > (data["medianbatch_msignal"]))*1.)))))) +

                            0.050000*np.tanh(((np.where((((np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_slices2_msignal"], data["minbatch_msignal"] )) > (((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0)))*1.) <= -998, np.sin((data["mean_abs_chgbatch_slices2_msignal"])), ((np.sin((np.where((11.62695693969726562) > -998, data["mean_abs_chgbatch_slices2_msignal"], (-((data["maxbatch_msignal"]))) )))) * 2.0) )) * 2.0)) +

                            0.050000*np.tanh(((np.where(data["minbatch"] <= -998, ((data["minbatch_msignal"]) / 2.0), (-((data["abs_avgbatch_msignal"]))) )) - ((((data["medianbatch_msignal"]) > (np.tanh((((np.where(data["abs_maxbatch_msignal"] <= -998, (3.0), (-((data["abs_avgbatch_msignal"]))) )) * (data["minbatch_msignal"]))))))*1.)))) +

                            0.050000*np.tanh(((np.where(data["abs_minbatch_slices2_msignal"] <= -998, ((((data["maxbatch_msignal"]) + (((np.where(np.where(data["mean_abs_chgbatch_slices2"] <= -998, np.maximum(((((data["maxtominbatch_slices2"]) * 2.0))), ((data["abs_maxbatch_slices2_msignal"]))), data["abs_maxbatch_slices2"] ) > -998, ((data["abs_minbatch_slices2_msignal"]) - (data["abs_avgbatch_slices2"])), data["abs_maxbatch"] )) / 2.0)))) / 2.0), np.cos((data["abs_minbatch_slices2_msignal"])) )) * 2.0)) +

                            0.050000*np.tanh((((data["minbatch"]) + (np.where(((data["mean_abs_chgbatch_msignal"]) * ((((data["minbatch"]) + (((data["medianbatch_msignal"]) * (data["abs_minbatch_slices2_msignal"]))))/2.0))) > -998, ((data["stdbatch_msignal"]) * (data["stdbatch_msignal"])), data["medianbatch_msignal"] )))/2.0)) +

                            0.050000*np.tanh(np.where(data["abs_minbatch_slices2_msignal"] <= -998, np.where(data["abs_maxbatch_slices2_msignal"] <= -998, ((((np.cos((data["signal"]))) * 2.0)) * ((-(((((data["abs_maxbatch_msignal"]) <= (np.maximum(((data["signal"])), (((((7.0)) + (data["signal"])))))))*1.)))))), ((np.cos((data["signal_shift_-1_msignal"]))) * 2.0) ), ((np.cos((data["signal"]))) * 2.0) )) +

                            0.050000*np.tanh(((data["minbatch"]) * (np.where(np.sin((data["rangebatch_slices2_msignal"])) <= -998, data["stdbatch_msignal"], np.cos((data["meanbatch_msignal"])) )))) +

                            0.050000*np.tanh(((data["abs_maxbatch"]) - (np.where(data["maxbatch_slices2_msignal"] > -998, np.where((((-((data["abs_maxbatch"])))) + (data["signal"])) > -998, data["rangebatch_msignal"], data["maxbatch_slices2"] ), data["meanbatch_slices2"] )))) +

                            0.050000*np.tanh(np.where(np.where(np.sin((data["stdbatch_slices2_msignal"])) <= -998, np.sin((np.minimum(((np.cos((((((np.sin((data["stdbatch_slices2_msignal"]))) * 2.0)) + (np.sin((data["meanbatch_slices2"])))))))), ((data["mean_abs_chgbatch_msignal"]))))), np.tanh((np.sin((np.minimum(((data["rangebatch_msignal"])), ((data["mean_abs_chgbatch_msignal"]))))))) ) <= -998, data["stdbatch_slices2_msignal"], ((np.sin((np.sin((data["stdbatch_slices2_msignal"]))))) * 2.0) )) +

                            0.050000*np.tanh(np.sin((np.where((((data["meanbatch_slices2"]) + (np.sin((np.minimum(((data["stdbatch_msignal"])), ((np.sin((np.where((6.0) > -998, data["mean_abs_chgbatch_msignal"], ((data["rangebatch_slices2_msignal"]) - (data["signal"])) ))))))))))/2.0) > -998, data["mean_abs_chgbatch_msignal"], data["abs_maxbatch"] )))) +

                            0.050000*np.tanh(((data["minbatch"]) + (np.where(np.sin((((data["minbatch"]) + (np.where(data["abs_maxbatch_slices2"] > -998, data["signal"], data["minbatch"] ))))) > -998, np.maximum(((data["mean_abs_chgbatch_slices2_msignal"])), ((data["signal"]))), ((((data["minbatch"]) + (data["minbatch"]))) / 2.0) )))) +

                            0.050000*np.tanh(np.sin((np.sin((((((np.sin(((((data["abs_maxbatch_msignal"]) + (np.maximum(((data["maxbatch_slices2"])), ((data["medianbatch_slices2"])))))/2.0)))) / 2.0)) / 2.0)))))) +

                            0.050000*np.tanh((-((((data["signal"]) * (np.cos((np.minimum(((((((-((((data["maxbatch_slices2_msignal"]) * 2.0))))) > ((((data["abs_maxbatch_msignal"]) > ((((np.where(np.sin((data["maxbatch_slices2_msignal"])) > -998, data["abs_maxbatch_slices2"], data["maxbatch_msignal"] )) + (data["maxbatch_slices2"]))/2.0)))*1.)))*1.))), ((data["minbatch_msignal"]))))))))))) +

                            0.050000*np.tanh(np.where(np.minimum(((np.sin((data["stdbatch_slices2_msignal"])))), (((((0.67998307943344116)) + (np.sin((data["stdbatch_slices2_msignal"]))))))) > -998, ((np.sin((data["stdbatch_slices2_msignal"]))) * 2.0), np.sin((np.sin((data["stdbatch_slices2_msignal"])))) )) +

                            0.050000*np.tanh(((np.sin((data["mean_abs_chgbatch_slices2_msignal"]))) * (np.where(np.sin((data["mean_abs_chgbatch_slices2_msignal"])) > -998, (2.0), np.where((-((np.where((2.0) > -998, np.where(np.sin((data["mean_abs_chgbatch_slices2_msignal"])) > -998, (2.0), ((data["abs_maxbatch_slices2"]) / 2.0) ), data["maxbatch_slices2_msignal"] )))) > -998, data["maxbatch_slices2_msignal"], data["abs_avgbatch_msignal"] ) )))) +

                            0.050000*np.tanh((((-((np.where(np.cos((((data["maxtominbatch"]) * 2.0))) <= -998, (-((data["abs_maxbatch_slices2"]))), ((np.minimum(((np.cos((data["abs_maxbatch_slices2_msignal"])))), ((np.cos((np.maximum(((np.maximum(((np.tanh((data["abs_avgbatch_slices2_msignal"])))), ((data["abs_maxbatch_slices2"]))))), ((data["meanbatch_slices2_msignal"]))))))))) + (((data["abs_avgbatch_slices2_msignal"]) * (data["signal_shift_-1"])))) ))))) + (data["minbatch"]))) +

                            0.050000*np.tanh(((np.sin((data["mean_abs_chgbatch_slices2_msignal"]))) * 2.0)) +

                            0.050000*np.tanh(((data["minbatch_slices2_msignal"]) - (((data["medianbatch_slices2"]) * (np.minimum(((data["stdbatch_msignal"])), ((((data["abs_maxbatch_slices2_msignal"]) + (data["abs_maxbatch_slices2_msignal"])))))))))) +

                            0.050000*np.tanh(((np.minimum(((np.where(np.where(data["rangebatch_slices2_msignal"] <= -998, data["minbatch_msignal"], data["abs_maxbatch_msignal"] ) <= -998, (-((((np.sin((data["mean_abs_chgbatch_msignal"]))) * 2.0)))), np.cos((data["signal"])) ))), ((np.where(np.where(data["rangebatch_slices2_msignal"] <= -998, data["minbatch_slices2_msignal"], np.tanh((data["signal_shift_+1"])) ) <= -998, data["mean_abs_chgbatch_msignal"], np.cos((data["signal"])) ))))) * 2.0)) +

                            0.050000*np.tanh(np.minimum(((((data["signal"]) * 2.0))), ((np.where(data["medianbatch_msignal"] > -998, np.sin((data["mean_abs_chgbatch_slices2_msignal"])), ((data["rangebatch_msignal"]) + (((data["maxtominbatch_slices2"]) - (np.where(((data["maxbatch_slices2"]) / 2.0) <= -998, data["maxtominbatch_slices2_msignal"], data["mean_abs_chgbatch_slices2_msignal"] ))))) ))))) +

                            0.050000*np.tanh(np.minimum(((np.sin((np.where(data["maxbatch_slices2"] <= -998, data["mean_abs_chgbatch_msignal"], np.sin((np.where(np.tanh((np.tanh((data["maxbatch_msignal"])))) <= -998, (3.0), data["mean_abs_chgbatch_slices2_msignal"] ))) ))))), ((data["maxbatch_msignal"])))) +

                            0.050000*np.tanh(((np.sin((data["stdbatch_slices2_msignal"]))) * (np.where(np.sin((((np.sin((np.sin((data["stdbatch_slices2_msignal"]))))) * (np.where(data["abs_avgbatch_msignal"] <= -998, np.sin((np.minimum(((data["rangebatch_slices2"])), ((data["abs_avgbatch_msignal"]))))), data["signal_shift_-1"] ))))) <= -998, ((data["stdbatch_slices2_msignal"]) / 2.0), data["signal_shift_-1"] )))) +

                            0.050000*np.tanh(((((np.sin((np.where(data["rangebatch_msignal"] <= -998, np.tanh((np.tanh((data["mean_abs_chgbatch_slices2_msignal"])))), data["mean_abs_chgbatch_slices2_msignal"] )))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.where(((np.where(data["mean_abs_chgbatch_slices2"] > -998, data["signal"], ((np.cos((data["abs_minbatch_slices2_msignal"]))) * 2.0) )) * (np.sin((np.sin((data["signal_shift_-1"])))))) <= -998, (-((data["abs_maxbatch_slices2_msignal"]))), ((data["signal"]) * (np.cos((data["abs_minbatch_slices2_msignal"])))) )) +

                            0.050000*np.tanh(np.where(data["minbatch_slices2"] > -998, ((data["minbatch_slices2"]) + (data["signal_shift_+1"])), (((np.minimum(((data["medianbatch_slices2"])), ((data["meanbatch_slices2"])))) > (np.where(((data["maxbatch_slices2_msignal"]) + (((data["medianbatch_msignal"]) + ((4.0))))) <= -998, data["maxbatch_slices2"], data["meanbatch_slices2"] )))*1.) )) +

                            0.050000*np.tanh(((((-((data["abs_avgbatch_slices2_msignal"])))) + (np.tanh((data["maxbatch_slices2"]))))/2.0)) +

                            0.050000*np.tanh(((np.where(((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0) <= -998, (5.0), np.sin((((np.maximum(((data["stdbatch_msignal"])), ((np.cos((data["abs_minbatch_msignal"])))))) * 2.0))) )) * 2.0)) +

                            0.050000*np.tanh(((np.maximum((((10.0))), ((data["mean_abs_chgbatch_slices2_msignal"])))) * (np.sin((data["stdbatch_msignal"]))))) +

                            0.050000*np.tanh(data["signal"]) +

                            0.050000*np.tanh(((np.sin((np.minimum((((((((((np.sin((np.tanh((((np.sin((data["maxbatch_slices2"]))) * 2.0)))))) > (data["abs_maxbatch"]))*1.)) / 2.0)) * (((np.tanh((data["stdbatch_msignal"]))) * (data["maxtominbatch"])))))), ((data["minbatch_slices2"])))))) + (((np.sin((np.sin((data["mean_abs_chgbatch_slices2_msignal"]))))) * 2.0)))) +

                            0.050000*np.tanh(np.where(data["rangebatch_msignal"] > -998, np.tanh((data["maxtominbatch"])), np.sin((((data["rangebatch_msignal"]) * ((-((np.maximum(((np.cos((np.where(np.sin((np.minimum((((-((data["maxtominbatch_msignal"]))))), ((data["signal"]))))) <= -998, np.cos((data["maxtominbatch_msignal"])), data["abs_maxbatch_slices2"] ))))), ((((data["maxtominbatch"]) + (data["signal"])))))))))))) )) +

                            0.050000*np.tanh((-((np.sin((np.where(((data["maxtominbatch_slices2_msignal"]) * (np.where(((data["stdbatch_msignal"]) * (np.sin((data["maxtominbatch"])))) <= -998, np.sin((data["signal_shift_+1"])), ((data["signal_shift_-1"]) * 2.0) ))) <= -998, np.sin((((data["abs_maxbatch"]) / 2.0))), ((data["abs_maxbatch_slices2_msignal"]) * 2.0) ))))))) +

                            0.050000*np.tanh(((((((np.tanh((data["abs_minbatch_slices2"]))) - (np.maximum(((data["medianbatch_slices2"])), ((np.tanh((((data["minbatch_slices2_msignal"]) * (data["medianbatch_msignal"])))))))))) * 2.0)) * (np.sin((((data["maxtominbatch"]) * (np.minimum(((data["medianbatch_msignal"])), ((((((np.sin((data["medianbatch_msignal"]))) + (data["rangebatch_msignal"]))) + (data["maxtominbatch"])))))))))))) +

                            0.050000*np.tanh(((((((((data["abs_maxbatch_msignal"]) + (np.cos((data["signal"]))))) * ((-((np.sin((data["mean_abs_chgbatch_msignal"])))))))) * (((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0)))) - (data["maxbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(((data["mean_abs_chgbatch_msignal"]) * (np.sin((np.sin((np.maximum(((np.maximum(((data["signal"])), ((np.sin((np.sin((np.maximum(((data["signal"])), ((np.sin((np.minimum(((data["stdbatch_slices2"])), ((data["minbatch_msignal"]))))))))))))))))), ((((((data["meanbatch_msignal"]) * (np.maximum(((data["signal"])), ((((data["abs_avgbatch_slices2_msignal"]) * 2.0))))))) + (np.sin((data["rangebatch_slices2_msignal"])))))))))))))) +

                            0.050000*np.tanh(((np.sin((((np.where(data["abs_minbatch_slices2_msignal"] > -998, np.where(data["abs_minbatch_slices2_msignal"] > -998, (-((data["stdbatch_slices2_msignal"]))), data["rangebatch_slices2_msignal"] ), data["maxtominbatch"] )) * 2.0)))) * (data["medianbatch_slices2"]))) +

                            0.050000*np.tanh(np.sin((np.where(np.cos((data["maxbatch_slices2"])) > -998, data["stdbatch_slices2_msignal"], np.cos((((data["signal_shift_+1_msignal"]) / 2.0))) )))) +

                            0.050000*np.tanh((((-(((((((np.tanh((data["rangebatch_slices2"]))) <= (data["meanbatch_slices2_msignal"]))*1.)) + ((((((data["signal"]) <= (np.tanh((data["rangebatch_slices2"]))))*1.)) * 2.0))))))) - (((np.cos((data["minbatch_msignal"]))) * 2.0)))) +

                            0.050000*np.tanh((((data["signal_shift_+1"]) + (np.where((-(((((data["signal_shift_+1_msignal"]) <= ((((data["mean_abs_chgbatch_slices2"]) > (data["rangebatch_slices2_msignal"]))*1.)))*1.)))) <= -998, (((-((data["abs_maxbatch_slices2_msignal"])))) + (data["meanbatch_msignal"])), (((-((np.cos(((((((np.tanh((data["abs_maxbatch"]))) / 2.0)) > (data["rangebatch_slices2"]))*1.))))))) * 2.0) )))/2.0)) +

                            0.050000*np.tanh(((np.sin(((-((np.cos((((np.where(data["signal_shift_+1_msignal"] > -998, data["mean_abs_chgbatch_slices2_msignal"], np.sin((np.sin((data["minbatch_slices2"])))) )) * 2.0))))))))) * (data["rangebatch_msignal"]))) +

                            0.050000*np.tanh(((np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, np.sin((data["abs_minbatch_slices2_msignal"])), ((np.maximum((((-((np.sin((np.maximum((((-((np.sin((np.maximum(((data["mean_abs_chgbatch_slices2_msignal"])), ((data["meanbatch_slices2_msignal"])))))))))), ((data["rangebatch_msignal"])))))))))), ((data["minbatch_slices2_msignal"])))) * 2.0) )) * (data["signal_shift_-1"]))) +

                            0.050000*np.tanh((-(((((((((data["minbatch_slices2_msignal"]) / 2.0)) + (((((((np.maximum(((data["maxtominbatch_slices2_msignal"])), ((data["stdbatch_slices2"])))) + ((-((np.maximum((((((data["stdbatch_slices2_msignal"]) + (np.cos((data["abs_minbatch_slices2_msignal"]))))/2.0))), ((data["stdbatch_slices2"]))))))))/2.0)) + (np.sin((data["abs_minbatch_msignal"]))))/2.0)))) <= (data["minbatch_slices2"]))*1.))))) +

                            0.050000*np.tanh(((np.where(data["minbatch"] <= -998, ((data["minbatch"]) * 2.0), data["minbatch"] )) * ((-((np.sin((np.sin((((data["meanbatch_msignal"]) * 2.0))))))))))) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) * (np.sin((((((data["abs_minbatch_slices2_msignal"]) * 2.0)) - (np.where(data["abs_avgbatch_slices2_msignal"] <= -998, np.minimum(((((np.sin((data["abs_minbatch_slices2"]))) - (np.sin((data["abs_minbatch_slices2_msignal"])))))), ((data["abs_minbatch_slices2"]))), np.where(data["meanbatch_slices2"] <= -998, ((data["abs_minbatch_msignal"]) * 2.0), np.sin((data["abs_minbatch_slices2"])) ) )))))))) +

                            0.050000*np.tanh(np.sin((((data["abs_minbatch_slices2_msignal"]) + (np.where(((data["abs_minbatch_slices2_msignal"]) + (data["minbatch"])) <= -998, np.maximum(((data["abs_minbatch_msignal"])), ((((data["mean_abs_chgbatch_msignal"]) * (data["rangebatch_slices2"]))))), data["abs_minbatch_msignal"] )))))) +

                            0.050000*np.tanh(((((((data["abs_minbatch_slices2"]) - (data["rangebatch_msignal"]))) - (((((((data["meanbatch_slices2"]) - (data["rangebatch_msignal"]))) * 2.0)) * (np.sin((np.sin((data["rangebatch_msignal"]))))))))) * (np.cos((data["abs_maxbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(((np.sin((data["rangebatch_msignal"]))) * (((data["maxbatch_msignal"]) - (np.where(np.sin((np.minimum(((data["minbatch"])), ((data["stdbatch_slices2"]))))) > -998, np.where(data["minbatch_slices2_msignal"] <= -998, np.cos((np.sin(((-((data["maxbatch_slices2"]))))))), data["minbatch"] ), data["stdbatch_slices2_msignal"] )))))) +

                            0.050000*np.tanh((-((np.where(data["maxbatch_slices2_msignal"] <= -998, np.maximum(((np.minimum(((data["maxbatch_slices2_msignal"])), ((data["minbatch_msignal"]))))), ((data["meanbatch_slices2_msignal"]))), np.sin((((data["maxbatch_slices2_msignal"]) * 2.0))) ))))) +

                            0.050000*np.tanh(((np.sin(((((-(((-((data["meanbatch_slices2_msignal"]))))))) + (((data["signal"]) * (np.sin((((data["mean_abs_chgbatch_msignal"]) * (data["mean_abs_chgbatch_msignal"]))))))))))) + (((data["minbatch_msignal"]) + (((data["signal"]) * (data["abs_maxbatch_msignal"]))))))) +

                            0.050000*np.tanh(((np.sin((((data["abs_minbatch_slices2_msignal"]) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(np.sin((np.where(data["abs_minbatch_slices2_msignal"] <= -998, data["minbatch_slices2_msignal"], ((np.sin((data["abs_minbatch_msignal"]))) * 2.0) )))) +

                            0.050000*np.tanh(np.cos((((data["stdbatch_slices2_msignal"]) + (np.where(data["minbatch"] > -998, data["meanbatch_msignal"], ((((np.sin((data["meanbatch_msignal"]))) * (np.cos((((data["stdbatch_msignal"]) + (np.where(data["minbatch_slices2_msignal"] > -998, data["minbatch"], data["mean_abs_chgbatch_msignal"] )))))))) * 2.0) )))))) +

                            0.050000*np.tanh(np.where((((((data["minbatch_slices2"]) * 2.0)) <= (np.minimum(((data["abs_minbatch_slices2"])), (((-((np.maximum(((data["rangebatch_msignal"])), (((-((data["rangebatch_msignal"]))))))))))))))*1.) <= -998, np.tanh((data["rangebatch_msignal"])), ((data["abs_maxbatch_slices2"]) * (np.tanh((np.sin((data["rangebatch_msignal"])))))) )) +

                            0.050000*np.tanh(((data["maxtominbatch"]) - (data["meanbatch_slices2"]))) +

                            0.050000*np.tanh(((((np.sin((np.where(np.minimum(((data["rangebatch_msignal"])), ((data["signal"]))) > -998, np.where(np.minimum(((data["rangebatch_msignal"])), ((data["signal"]))) > -998, data["rangebatch_msignal"], np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0))) ), data["abs_avgbatch_msignal"] )))) * 2.0)) - (((np.cos((((data["mean_abs_chgbatch_slices2_msignal"]) * 2.0)))) * 2.0)))) +

                            0.050000*np.tanh(np.cos((((data["meanbatch_msignal"]) + (np.where(((data["maxbatch_msignal"]) * (np.cos(((((7.0)) + (data["medianbatch_msignal"])))))) <= -998, (7.0), data["minbatch_msignal"] )))))) +

                            0.050000*np.tanh(((((((((((np.minimum(((data["maxtominbatch"])), (((-((np.sin((data["signal"]))))))))) > (np.maximum((((((data["rangebatch_slices2"]) > ((-((np.sin((data["rangebatch_slices2"])))))))*1.))), ((data["signal"])))))*1.)) - (data["stdbatch_slices2_msignal"]))) + (data["signal_shift_-1_msignal"]))/2.0)) * (((data["rangebatch_slices2"]) * ((-((np.sin((data["signal"])))))))))) +

                            0.050000*np.tanh(np.where(np.sin((data["minbatch_slices2_msignal"])) > -998, ((data["mean_abs_chgbatch_slices2_msignal"]) * (np.sin((data["signal"])))), np.maximum(((((np.tanh((data["minbatch_slices2_msignal"]))) / 2.0))), ((np.tanh((np.maximum(((data["abs_avgbatch_slices2"])), ((np.tanh((data["abs_maxbatch"])))))))))) )) +

                            0.050000*np.tanh(np.sin((np.where(((np.cos((data["stdbatch_slices2_msignal"]))) * (np.minimum(((np.minimum(((np.cos(((5.0))))), (((5.0)))))), ((((data["abs_minbatch_msignal"]) * 2.0)))))) <= -998, np.sin((((np.cos(((5.0)))) * 2.0))), ((data["abs_minbatch_msignal"]) * 2.0) )))) +

                            0.050000*np.tanh(((np.sin((((np.where((-((data["meanbatch_msignal"]))) <= -998, data["meanbatch_msignal"], np.minimum(((data["stdbatch_msignal"])), ((((data["minbatch"]) / 2.0)))) )) * 2.0)))) * (np.where((-((data["meanbatch_msignal"]))) <= -998, ((np.sin((np.sin((((data["stdbatch_msignal"]) * 2.0)))))) * (data["meanbatch_slices2"])), np.minimum(((data["stdbatch_msignal"])), ((data["maxtominbatch_msignal"]))) )))) +

                            0.050000*np.tanh(((data["rangebatch_msignal"]) * (np.sin((((np.sin((((np.minimum(((np.where(np.sin((np.sin((data["rangebatch_msignal"])))) <= -998, data["maxbatch_slices2"], data["abs_minbatch_slices2_msignal"] ))), ((data["minbatch"])))) * 2.0)))) * 2.0)))))) +

                            0.050000*np.tanh((((-((((((data["abs_maxbatch_slices2_msignal"]) * (data["abs_avgbatch_slices2_msignal"]))) - ((((-((((((data["abs_maxbatch_slices2_msignal"]) * (data["abs_avgbatch_slices2_msignal"]))) - (np.minimum(((np.cos((data["meanbatch_slices2"])))), ((((np.where(data["meanbatch_slices2_msignal"] > -998, data["maxtominbatch"], data["maxtominbatch_slices2_msignal"] )) * 2.0)))))))))) / 2.0))))))) / 2.0)) +

                            0.050000*np.tanh(((((data["signal_shift_-1"]) - ((12.30045986175537109)))) - (np.minimum(((data["abs_maxbatch"])), ((np.minimum(((data["maxtominbatch_slices2_msignal"])), ((np.where(np.where(data["mean_abs_chgbatch_slices2"] <= -998, data["mean_abs_chgbatch_slices2_msignal"], data["mean_abs_chgbatch_msignal"] ) > -998, np.maximum(((data["abs_maxbatch"])), ((np.maximum((((12.30045986175537109))), ((np.sin((data["signal_shift_-1"])))))))), (-((data["minbatch"]))) )))))))))) +

                            0.050000*np.tanh(((np.where(np.maximum(((((np.where(data["abs_minbatch_slices2_msignal"] <= -998, (0.0), np.cos((np.cos((data["abs_minbatch_slices2_msignal"])))) )) * (data["stdbatch_msignal"])))), (((((data["signal_shift_-1_msignal"]) > ((-((data["stdbatch_msignal"])))))*1.)))) <= -998, data["minbatch"], (-((data["medianbatch_slices2"]))) )) * (np.cos((data["stdbatch_msignal"]))))) +

                            0.050000*np.tanh(data["stdbatch_slices2_msignal"]) +

                            0.050000*np.tanh(((data["rangebatch_msignal"]) * 2.0)) +

                            0.050000*np.tanh(data["signal_shift_+1_msignal"]) +

                            0.050000*np.tanh(((data["signal_shift_-1_msignal"]) - (np.where(data["signal_shift_-1_msignal"] > -998, data["medianbatch_msignal"], data["mean_abs_chgbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(data["minbatch_slices2_msignal"]) +

                            0.050000*np.tanh((((-((data["abs_maxbatch_slices2"])))) * (np.cos((data["maxbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(np.minimum(((np.where(data["minbatch_msignal"] <= -998, ((((data["stdbatch_slices2_msignal"]) * (data["signal"]))) + (data["signal_shift_-1_msignal"])), (-((np.sin((data["signal"]))))) ))), ((((np.sin((np.where(data["signal_shift_-1_msignal"] <= -998, np.cos((np.cos((data["maxtominbatch_msignal"])))), data["signal_shift_-1_msignal"] )))) + (((data["abs_minbatch_slices2"]) + (np.cos((data["maxtominbatch_msignal"])))))))))) +

                            0.050000*np.tanh(((np.sin((((np.sin((data["abs_minbatch_slices2_msignal"]))) - (np.sin((data["signal_shift_+1"]))))))) * 2.0)) +

                            0.050000*np.tanh(((((data["mean_abs_chgbatch_msignal"]) + (data["signal_shift_+1_msignal"]))) * (((np.sin((data["maxtominbatch"]))) - (np.maximum((((((((((data["minbatch_slices2_msignal"]) > (np.sin((np.sin((((data["abs_minbatch_slices2_msignal"]) / 2.0)))))))*1.)) * 2.0)) - (data["minbatch_slices2_msignal"])))), ((data["meanbatch_msignal"])))))))) +

                            0.050000*np.tanh(np.sin((np.minimum((((11.73060512542724609))), (((-((((data["signal_shift_-1"]) + (np.cos((np.where(data["abs_maxbatch_msignal"] > -998, data["mean_abs_chgbatch_msignal"], np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_msignal"], data["maxbatch_slices2_msignal"] ) )))))))))))))) +

                            0.050000*np.tanh(np.cos((np.maximum(((data["abs_avgbatch_slices2_msignal"])), (((((((data["signal"]) + (data["signal_shift_+1"]))/2.0)) - (data["abs_avgbatch_slices2_msignal"])))))))) +

                            0.050000*np.tanh(np.where(data["signal_shift_-1_msignal"] > -998, data["signal_shift_-1_msignal"], np.where(np.minimum(((np.minimum(((np.cos((data["rangebatch_msignal"])))), ((data["minbatch_msignal"]))))), ((np.where((-((np.minimum(((data["minbatch_slices2_msignal"])), ((data["medianbatch_slices2"])))))) > -998, data["signal_shift_-1_msignal"], data["signal_shift_+1"] )))) <= -998, np.sin(((((data["meanbatch_slices2"]) > (data["rangebatch_slices2_msignal"]))*1.))), data["rangebatch_msignal"] ) )) +

                            0.050000*np.tanh(((np.cos((data["signal_shift_-1"]))) * 2.0)) +

                            0.050000*np.tanh((-((((np.sin((((data["signal"]) + (((data["signal"]) + (np.sin((data["signal_shift_-1_msignal"]))))))))) * (((((((data["signal"]) + ((-((data["signal_shift_+1"])))))) + (np.sin((data["signal_shift_+1"]))))) + (((data["signal"]) + (data["signal_shift_+1"])))))))))) +

                            0.050000*np.tanh(((((((data["signal_shift_-1_msignal"]) + (((data["minbatch_msignal"]) * (np.cos(((((-((data["signal_shift_+1"])))) + (data["abs_maxbatch_slices2_msignal"]))))))))/2.0)) + (((data["mean_abs_chgbatch_slices2_msignal"]) * (np.cos(((((-((data["signal_shift_+1"])))) + (data["abs_maxbatch_slices2_msignal"]))))))))/2.0)) +

                            0.050000*np.tanh(np.minimum(((np.sin((np.where(np.maximum(((data["abs_minbatch_msignal"])), (((((data["stdbatch_slices2_msignal"]) + (data["maxtominbatch_slices2"]))/2.0)))) > -998, data["mean_abs_chgbatch_msignal"], np.tanh((((((np.sin((data["rangebatch_msignal"]))) * (data["stdbatch_slices2"]))) - (data["medianbatch_slices2"])))) ))))), ((np.maximum(((np.maximum(((data["mean_abs_chgbatch_msignal"])), ((data["medianbatch_slices2"]))))), ((data["abs_avgbatch_slices2_msignal"]))))))) +

                            0.050000*np.tanh(((np.minimum(((data["stdbatch_slices2_msignal"])), ((np.cos((np.sin((((np.minimum(((data["stdbatch_slices2_msignal"])), ((data["stdbatch_slices2_msignal"])))) * 2.0))))))))) * (np.sin((((np.minimum(((data["stdbatch_slices2_msignal"])), ((data["stdbatch_slices2_msignal"])))) * 2.0)))))) +

                            0.050000*np.tanh(((np.cos((((((data["stdbatch_slices2_msignal"]) + ((((((data["minbatch_slices2"]) - ((3.65924572944641113)))) <= (((np.cos((((((data["stdbatch_slices2_msignal"]) + ((((data["maxtominbatch_slices2"]) <= (np.cos(((((data["signal_shift_+1"]) > (data["maxbatch_slices2_msignal"]))*1.)))))*1.)))) * 2.0)))) * 2.0)))*1.)))) * 2.0)))) * 2.0)) +

                            0.050000*np.tanh(np.where(data["abs_maxbatch_slices2"] <= -998, (((data["abs_avgbatch_slices2_msignal"]) + ((-((data["maxtominbatch_slices2_msignal"])))))/2.0), ((data["signal_shift_+1_msignal"]) - (np.minimum(((data["abs_avgbatch_slices2"])), ((data["abs_avgbatch_msignal"]))))) )) +

                            0.050000*np.tanh(data["signal_shift_-1_msignal"]) +

                            0.050000*np.tanh(np.cos((np.where((-((((np.cos((np.where(data["stdbatch_slices2_msignal"] > -998, data["signal"], ((data["signal"]) * (data["maxbatch_slices2_msignal"])) )))) / 2.0)))) > -998, data["signal"], np.maximum((((1.0))), ((((((1.0)) + (((np.cos(((-((data["signal_shift_+1_msignal"])))))) * 2.0)))/2.0)))) )))) +

                            0.050000*np.tanh(np.minimum(((data["signal_shift_-1"])), (((-((((np.sin((data["signal_shift_+1"]))) * (np.maximum(((data["abs_maxbatch"])), ((np.where(data["abs_maxbatch"] <= -998, (((((-(((-((np.cos((data["signal_shift_+1"]))))))))) * (data["abs_maxbatch"]))) + (data["signal_shift_-1_msignal"])), data["meanbatch_msignal"] ))))))))))))) +

                            0.050000*np.tanh(((data["signal_shift_-1_msignal"]) + (np.where(data["signal_shift_+1"] > -998, np.cos((np.where(data["abs_avgbatch_slices2_msignal"] <= -998, ((data["stdbatch_msignal"]) * (data["maxtominbatch_slices2_msignal"])), data["minbatch"] ))), data["maxtominbatch_slices2"] )))) +

                            0.050000*np.tanh(np.sin((np.sin((np.where(((np.where(np.sin((data["rangebatch_msignal"])) > -998, data["stdbatch_slices2"], (((((((((((data["minbatch_slices2"]) * 2.0)) - (data["mean_abs_chgbatch_slices2_msignal"]))) + (data["minbatch_msignal"]))/2.0)) * 2.0)) - (data["mean_abs_chgbatch_slices2_msignal"])) )) - ((9.0))) > -998, (-((np.sin((data["signal_shift_-1_msignal"]))))), data["maxbatch_slices2_msignal"] )))))) +

                            0.050000*np.tanh(np.sin((((data["minbatch"]) + (((data["meanbatch_slices2_msignal"]) / 2.0)))))) +

                            0.050000*np.tanh(np.cos((((data["signal_shift_-1"]) - (np.where(data["stdbatch_msignal"] <= -998, (((data["signal_shift_-1"]) > (data["rangebatch_slices2_msignal"]))*1.), np.where(data["stdbatch_msignal"] > -998, data["maxtominbatch_msignal"], data["mean_abs_chgbatch_slices2_msignal"] ) )))))) +

                            0.050000*np.tanh(((np.where(data["mean_abs_chgbatch_msignal"] <= -998, (((np.tanh(((-((np.maximum(((((data["medianbatch_slices2"]) * (data["abs_maxbatch_msignal"])))), ((data["abs_avgbatch_slices2"]))))))))) > (data["maxbatch_msignal"]))*1.), data["signal_shift_+1"] )) - (data["abs_avgbatch_slices2"]))) +

                            0.050000*np.tanh(((((((data["meanbatch_slices2_msignal"]) + ((-((data["maxbatch_slices2_msignal"])))))/2.0)) + ((-((np.where(data["rangebatch_msignal"] <= -998, data["meanbatch_slices2_msignal"], ((data["abs_avgbatch_msignal"]) * (np.where(data["meanbatch_slices2_msignal"] > -998, data["abs_maxbatch_msignal"], data["minbatch_slices2"] ))) ))))))/2.0)) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) - (np.tanh((np.tanh(((((data["abs_maxbatch_msignal"]) > (np.maximum(((np.sin((data["mean_abs_chgbatch_msignal"])))), ((data["maxbatch_slices2_msignal"])))))*1.)))))))) +

                            0.050000*np.tanh(np.cos((data["abs_minbatch_msignal"]))) +

                            0.050000*np.tanh(np.sin((((((np.minimum(((((np.where(data["mean_abs_chgbatch_slices2"] > -998, data["mean_abs_chgbatch_slices2_msignal"], data["minbatch_slices2"] )) * 2.0))), ((((((np.cos((data["minbatch_slices2"]))) * 2.0)) * 2.0))))) - (data["abs_maxbatch_msignal"]))) * 2.0)))) +

                            0.050000*np.tanh(np.minimum(((data["medianbatch_slices2_msignal"])), ((((data["minbatch_slices2"]) * 2.0))))) +

                            0.050000*np.tanh(((data["minbatch_slices2"]) * (((np.where(np.minimum(((data["signal_shift_-1_msignal"])), ((np.cos((data["minbatch_slices2_msignal"]))))) > -998, data["signal_shift_-1"], data["maxtominbatch_slices2_msignal"] )) * (np.sin((np.sin((data["signal_shift_-1"]))))))))) +

                            0.050000*np.tanh(np.where(data["medianbatch_slices2_msignal"] <= -998, ((data["minbatch_msignal"]) - (np.tanh(((((((data["signal_shift_-1_msignal"]) * (data["abs_maxbatch_slices2_msignal"]))) + (np.sin((((np.where(data["signal_shift_-1_msignal"] > -998, np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_slices2"], data["signal_shift_-1_msignal"] ), np.maximum(((data["abs_avgbatch_msignal"])), ((data["signal_shift_-1_msignal"]))) )) / 2.0)))))/2.0))))), data["signal_shift_+1_msignal"] )) +

                            0.050000*np.tanh(((np.sin((data["abs_maxbatch"]))) / 2.0)) +

                            0.050000*np.tanh(np.where((((np.where(data["mean_abs_chgbatch_slices2"] <= -998, ((data["abs_avgbatch_slices2_msignal"]) / 2.0), np.where(((((data["abs_avgbatch_slices2_msignal"]) - (data["signal_shift_-1_msignal"]))) / 2.0) <= -998, data["mean_abs_chgbatch_slices2"], data["mean_abs_chgbatch_slices2_msignal"] ) )) > ((((data["abs_avgbatch_slices2_msignal"]) + (data["medianbatch_slices2"]))/2.0)))*1.) <= -998, data["signal_shift_-1_msignal"], (-((((data["abs_avgbatch_slices2_msignal"]) - (((data["signal_shift_-1_msignal"]) / 2.0)))))) )) +

                            0.050000*np.tanh((((((-((data["maxbatch_slices2_msignal"])))) * (((data["signal_shift_-1_msignal"]) * (np.sin((((data["stdbatch_msignal"]) - (((np.where(np.tanh(((-(((7.0)))))) <= -998, data["stdbatch_msignal"], np.tanh((data["maxtominbatch"])) )) / 2.0)))))))))) * 2.0)) +

                            0.050000*np.tanh(((np.cos((((np.where(np.maximum((((10.35262870788574219))), ((np.cos((data["maxbatch_msignal"]))))) <= -998, np.sin((data["maxbatch_msignal"])), data["maxbatch_msignal"] )) - (data["signal_shift_+1"]))))) * (((data["abs_minbatch_msignal"]) + (np.cos((data["maxbatch_msignal"]))))))) +

                            0.050000*np.tanh(((np.cos((((data["maxbatch_slices2_msignal"]) * 2.0)))) * (np.where(np.minimum(((((data["medianbatch_slices2"]) * (data["maxtominbatch"])))), ((data["abs_maxbatch"]))) > -998, data["signal_shift_-1_msignal"], (((2.0)) + (data["signal_shift_-1"])) )))) +

                            0.050000*np.tanh((((0.0)) - (np.cos((((((np.where(data["abs_minbatch_slices2"] > -998, data["signal_shift_+1"], ((((0.0)) <= (((np.where(data["signal_shift_+1"] > -998, data["signal_shift_+1"], (((data["maxbatch_slices2"]) <= (np.maximum(((data["maxtominbatch_slices2_msignal"])), ((data["signal_shift_-1_msignal"])))))*1.) )) * 2.0)))*1.) )) * ((((data["medianbatch_msignal"]) + ((((data["signal"]) <= (data["stdbatch_msignal"]))*1.)))/2.0)))) / 2.0)))))) +

                            0.050000*np.tanh(np.where(data["meanbatch_slices2_msignal"] > -998, np.sin((data["minbatch_msignal"])), ((np.where(data["maxbatch_msignal"] > -998, data["meanbatch_slices2_msignal"], ((np.sin((data["minbatch"]))) - (data["mean_abs_chgbatch_msignal"])) )) - (np.cos((np.where(data["maxbatch_msignal"] > -998, data["abs_minbatch_slices2"], data["maxbatch_slices2_msignal"] ))))) )) +

                            0.050000*np.tanh(data["minbatch_msignal"]) +

                            0.050000*np.tanh(((data["signal_shift_-1_msignal"]) - (np.maximum(((data["abs_minbatch_msignal"])), ((np.maximum(((((data["abs_avgbatch_slices2"]) * (((data["maxbatch_msignal"]) - ((-((data["abs_minbatch_msignal"]))))))))), ((((((((data["maxtominbatch_slices2_msignal"]) + (((((data["abs_minbatch_slices2_msignal"]) / 2.0)) * 2.0)))/2.0)) + (data["stdbatch_msignal"]))/2.0)))))))))) +

                            0.050000*np.tanh(((data["maxtominbatch_slices2"]) * (np.sin((((np.where((((-((data["stdbatch_slices2"])))) * (np.cos((np.minimum(((np.cos((np.sin((data["abs_maxbatch"])))))), ((((((np.cos((data["medianbatch_slices2"]))) * 2.0)) - (data["rangebatch_slices2_msignal"]))))))))) <= -998, np.maximum((((((data["maxtominbatch_msignal"]) <= (data["maxtominbatch_msignal"]))*1.))), ((data["signal_shift_-1_msignal"]))), data["minbatch_msignal"] )) * 2.0)))))) +

                            0.050000*np.tanh(np.where(np.where(data["maxtominbatch"] > -998, data["abs_maxbatch_slices2"], np.cos((data["maxbatch_slices2_msignal"])) ) <= -998, np.sin((data["maxtominbatch"])), np.sin((data["minbatch_msignal"])) )) +

                            0.050000*np.tanh(np.sin((data["abs_maxbatch"]))) +

                            0.050000*np.tanh(((np.sin((data["signal_shift_+1"]))) * (np.where(np.where(np.cos((((np.sin((data["signal_shift_+1"]))) * (((((np.tanh((np.cos((np.cos((data["rangebatch_slices2"]))))))) - (data["minbatch"]))) - (np.sin((data["signal_shift_+1"])))))))) > -998, data["maxtominbatch_slices2_msignal"], data["abs_maxbatch"] ) > -998, data["maxtominbatch_msignal"], ((data["maxbatch_slices2"]) / 2.0) )))) +

                            0.050000*np.tanh(np.sin((np.where(data["abs_maxbatch_slices2_msignal"] > -998, np.sin((np.where(np.where(data["abs_avgbatch_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], np.where(np.cos(((5.75612592697143555))) > -998, data["abs_avgbatch_slices2"], ((data["abs_maxbatch_slices2_msignal"]) * (data["abs_avgbatch_msignal"])) ) ) > -998, data["abs_maxbatch"], data["stdbatch_slices2_msignal"] ))), np.minimum(((data["abs_avgbatch_slices2_msignal"])), ((np.minimum(((data["abs_avgbatch_msignal"])), ((np.cos((data["signal_shift_+1"])))))))) )))) +

                            0.050000*np.tanh(np.sin(((((((data["signal"]) - (np.sin((data["signal"]))))) > ((-((data["signal_shift_+1_msignal"])))))*1.)))) +

                            0.050000*np.tanh(((np.cos((np.sin((data["maxtominbatch_msignal"]))))) - (((np.sin((np.sin((np.sin((np.sin((data["maxtominbatch_msignal"]))))))))) * (np.where(data["abs_avgbatch_slices2_msignal"] > -998, (9.0), np.cos((((data["signal_shift_-1_msignal"]) - (np.sin((data["maxtominbatch_msignal"])))))) )))))) +

                            0.050000*np.tanh(np.sin((np.tanh((((((data["stdbatch_slices2"]) * (((np.cos((data["stdbatch_slices2"]))) * (((data["medianbatch_slices2_msignal"]) * 2.0)))))) * 2.0)))))) +

                            0.050000*np.tanh(((((data["signal_shift_-1"]) - (data["maxbatch_slices2_msignal"]))) * (np.sin((((data["signal_shift_-1"]) - (np.minimum(((np.where(data["abs_avgbatch_msignal"] <= -998, np.sin((((data["signal_shift_-1"]) - (np.minimum(((data["signal_shift_-1"])), (((((data["mean_abs_chgbatch_msignal"]) <= (((data["signal_shift_-1"]) - (data["signal_shift_-1"]))))*1.)))))))), data["meanbatch_msignal"] ))), ((data["abs_maxbatch_msignal"])))))))))) +

                            0.050000*np.tanh(((np.maximum(((np.sin((data["stdbatch_slices2"])))), ((((data["mean_abs_chgbatch_slices2_msignal"]) / 2.0))))) + ((((-((data["meanbatch_msignal"])))) * 2.0)))) +

                            0.050000*np.tanh(np.where(np.sin((np.minimum(((np.sin(((((((data["abs_avgbatch_slices2"]) + ((-((((data["signal_shift_-1"]) + (data["abs_minbatch_slices2"])))))))/2.0)) * (data["maxtominbatch_msignal"])))))), (((14.36639404296875000)))))) > -998, np.sin((data["abs_maxbatch_slices2"])), (((((data["rangebatch_slices2"]) + (np.sin((data["maxtominbatch"]))))/2.0)) - (np.maximum(((np.tanh((data["abs_maxbatch_msignal"])))), ((data["minbatch_slices2"]))))) )) +

                            0.050000*np.tanh(np.where(data["maxtominbatch_slices2"] <= -998, data["signal_shift_-1_msignal"], np.sin((((data["stdbatch_msignal"]) * (np.maximum(((data["abs_maxbatch_slices2"])), ((data["stdbatch_msignal"]))))))) )) +

                            0.050000*np.tanh(np.where(np.cos((np.cos((data["signal_shift_-1_msignal"])))) > -998, np.cos(((((6.0)) * ((((data["maxbatch_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0))))), np.maximum(((data["signal_shift_+1_msignal"])), ((((data["mean_abs_chgbatch_slices2_msignal"]) + ((((((data["minbatch_msignal"]) * 2.0)) <= (data["signal_shift_+1_msignal"]))*1.)))))) )) +

                            0.050000*np.tanh(np.sin((np.sin((((np.sin((np.minimum((((((data["medianbatch_slices2_msignal"]) <= ((((-(((-((data["signal_shift_+1_msignal"]))))))) * 2.0)))*1.))), ((np.tanh((data["medianbatch_slices2"])))))))) - (data["signal_shift_+1"]))))))) +

                            0.050000*np.tanh(np.cos((np.minimum(((np.sin((data["maxtominbatch_msignal"])))), ((((np.where((-((np.minimum((((9.0))), ((data["stdbatch_msignal"])))))) > -998, data["minbatch_msignal"], data["maxbatch_slices2"] )) + (data["meanbatch_msignal"])))))))) +

                            0.050000*np.tanh((((data["maxbatch_slices2"]) <= ((((-((np.maximum(((np.cos((np.minimum(((np.where(data["maxbatch_msignal"] > -998, np.sin((data["maxbatch_slices2"])), ((data["meanbatch_slices2_msignal"]) * 2.0) ))), ((((data["minbatch"]) - (data["signal"]))))))))), (((4.0)))))))) * 2.0)))*1.)) +

                            0.050000*np.tanh(np.where((((data["meanbatch_slices2_msignal"]) <= (np.cos((data["mean_abs_chgbatch_slices2_msignal"]))))*1.) > -998, ((data["maxtominbatch_msignal"]) + (np.minimum((((-((data["abs_maxbatch"]))))), ((((data["signal_shift_+1_msignal"]) * 2.0)))))), data["maxbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(((np.minimum(((np.where((((data["signal_shift_-1"]) + (data["signal_shift_-1_msignal"]))/2.0) > -998, data["abs_maxbatch_slices2_msignal"], (-((np.where((((data["maxbatch_slices2_msignal"]) <= (((np.tanh((data["signal"]))) + ((-((data["abs_minbatch_slices2_msignal"])))))))*1.) <= -998, np.cos((data["abs_maxbatch_slices2_msignal"])), data["signal_shift_+1"] )))) ))), ((data["signal_shift_+1"])))) - (np.cos((data["abs_maxbatch_slices2_msignal"]))))) +

                            0.050000*np.tanh(((data["maxtominbatch_slices2_msignal"]) * 2.0)) +

                            0.050000*np.tanh(((np.sin((np.where(((data["maxtominbatch_slices2_msignal"]) + (data["maxtominbatch"])) > -998, np.where(data["abs_avgbatch_slices2_msignal"] > -998, data["abs_maxbatch"], data["abs_maxbatch"] ), data["abs_maxbatch_slices2"] )))) * (np.where((((((-((data["abs_avgbatch_slices2_msignal"])))) - (np.sin(((-((data["maxbatch_msignal"])))))))) - (data["abs_maxbatch"])) > -998, (-((data["medianbatch_msignal"]))), data["abs_maxbatch_slices2"] )))) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) * ((((data["abs_avgbatch_slices2_msignal"]) + (((data["abs_maxbatch_slices2_msignal"]) - ((-((((((data["abs_maxbatch_msignal"]) / 2.0)) * 2.0))))))))/2.0)))) +

                            0.050000*np.tanh(np.tanh((np.cos((data["signal_shift_+1"]))))) +

                            0.050000*np.tanh(np.cos((np.where(np.sin((np.cos((data["signal_shift_-1_msignal"])))) > -998, np.maximum(((data["signal_shift_-1"])), ((data["signal"]))), ((np.sin((data["mean_abs_chgbatch_msignal"]))) + ((-(((-((np.maximum((((-((((data["minbatch_msignal"]) * 2.0)))))), ((data["signal"]))))))))))) )))) +

                            0.050000*np.tanh(np.minimum(((((data["maxtominbatch_slices2_msignal"]) - (np.where(data["maxbatch_slices2_msignal"] <= -998, data["rangebatch_slices2"], np.where(data["abs_minbatch_slices2"] > -998, data["rangebatch_slices2"], ((np.maximum(((np.sin(((((data["maxtominbatch"]) + ((-((data["rangebatch_slices2"])))))/2.0))))), ((data["meanbatch_msignal"])))) / 2.0) ) ))))), ((data["maxtominbatch"])))) +

                            0.050000*np.tanh(((data["minbatch"]) * (np.cos((np.where(data["abs_minbatch_msignal"] > -998, ((data["abs_minbatch_msignal"]) * (data["abs_avgbatch_msignal"])), ((data["abs_avgbatch_slices2_msignal"]) * (np.cos((np.where(data["meanbatch_slices2"] > -998, ((data["abs_minbatch_msignal"]) * (data["rangebatch_slices2"])), data["rangebatch_slices2"] ))))) )))))) +

                            0.050000*np.tanh(np.sin((np.cos((np.where(((((data["abs_minbatch_msignal"]) + (np.where(np.sin((np.cos(((((data["signal"]) + (data["stdbatch_slices2"]))/2.0))))) > -998, data["stdbatch_msignal"], (((data["signal_shift_-1"]) + (data["medianbatch_slices2"]))/2.0) )))) + (np.minimum(((data["maxtominbatch_slices2"])), ((np.sin((data["maxtominbatch_slices2"]))))))) <= -998, data["stdbatch_msignal"], data["abs_minbatch_slices2_msignal"] )))))) +

                            0.050000*np.tanh(((data["abs_minbatch_slices2"]) - (((np.sin((data["signal_shift_-1"]))) * (np.where(np.maximum((((11.95872116088867188))), ((data["signal_shift_-1_msignal"]))) <= -998, np.minimum(((((((data["signal_shift_-1_msignal"]) + ((-((data["maxbatch_slices2"])))))) - (data["rangebatch_msignal"])))), ((((np.sin((data["signal_shift_-1"]))) * (data["signal"]))))), data["meanbatch_slices2"] )))))) +

                            0.050000*np.tanh((((-((np.tanh((np.sin((np.where(np.cos(((-((data["rangebatch_slices2_msignal"]))))) > -998, (((np.tanh((data["abs_maxbatch_msignal"]))) > (np.tanh((np.minimum(((data["meanbatch_slices2_msignal"])), ((data["rangebatch_slices2_msignal"])))))))*1.), np.tanh((np.sin((np.where(data["minbatch_slices2"] > -998, np.tanh((data["rangebatch_slices2_msignal"])), data["signal_shift_-1"] ))))) ))))))))) * 2.0)) +

                            0.050000*np.tanh(np.where((((-((data["meanbatch_msignal"])))) * 2.0) > -998, ((data["signal_shift_-1"]) - ((3.0))), np.where(data["maxtominbatch_slices2"] > -998, data["signal_shift_+1"], (((data["abs_maxbatch_slices2_msignal"]) <= ((((data["signal_shift_-1"]) > (((data["signal_shift_+1"]) / 2.0)))*1.)))*1.) ) )) +

                            0.050000*np.tanh(((np.sin((data["maxtominbatch_msignal"]))) * ((((data["mean_abs_chgbatch_msignal"]) + (np.minimum(((data["maxtominbatch_msignal"])), ((((((np.sin((np.where(data["maxtominbatch_msignal"] > -998, data["maxtominbatch_msignal"], data["mean_abs_chgbatch_msignal"] )))) + (((data["stdbatch_slices2"]) - (data["maxtominbatch_msignal"]))))) + (data["maxtominbatch_msignal"])))))))/2.0)))) +

                            0.050000*np.tanh(np.sin((np.where(((data["minbatch"]) - (np.where(data["meanbatch_slices2_msignal"] <= -998, data["maxbatch_msignal"], np.sin((data["medianbatch_slices2_msignal"])) ))) <= -998, np.sin((((data["rangebatch_slices2_msignal"]) * 2.0))), np.where(np.cos((data["rangebatch_slices2_msignal"])) <= -998, data["abs_minbatch_msignal"], data["mean_abs_chgbatch_slices2_msignal"] ) )))) +

                            0.050000*np.tanh(np.minimum(((np.tanh((np.sin((((((((data["minbatch_slices2_msignal"]) > (((((data["stdbatch_slices2_msignal"]) / 2.0)) * 2.0)))*1.)) <= (data["stdbatch_msignal"]))*1.))))))), (((((data["minbatch"]) > ((-((np.maximum(((data["maxtominbatch_msignal"])), (((((data["stdbatch_msignal"]) <= ((-((data["signal_shift_+1"])))))*1.)))))))))*1.))))) +

                            0.050000*np.tanh((((data["abs_maxbatch"]) > (data["meanbatch_msignal"]))*1.)) +

                            0.050000*np.tanh(np.sin((((data["mean_abs_chgbatch_msignal"]) * (np.sin((data["rangebatch_msignal"]))))))) +

                            0.050000*np.tanh(np.sin((((((((-((np.cos((data["medianbatch_slices2"])))))) + ((((((data["maxbatch_slices2"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)) + (data["medianbatch_slices2_msignal"]))))/2.0)) * (data["rangebatch_slices2"]))))) +

                            0.050000*np.tanh(np.where(np.where(np.where(data["signal_shift_+1_msignal"] > -998, data["meanbatch_msignal"], ((data["medianbatch_slices2"]) / 2.0) ) <= -998, data["maxtominbatch_msignal"], np.minimum(((data["abs_maxbatch"])), ((((data["minbatch_slices2"]) / 2.0)))) ) <= -998, (-((((data["abs_maxbatch"]) * 2.0)))), np.sin((np.maximum(((data["abs_maxbatch"])), ((np.cos((np.maximum(((data["minbatch_slices2_msignal"])), ((data["meanbatch_msignal"])))))))))) )) +

                            0.050000*np.tanh(np.sin((((((np.minimum(((data["meanbatch_msignal"])), (((((((data["abs_avgbatch_slices2_msignal"]) - ((1.46488344669342041)))) <= (data["mean_abs_chgbatch_slices2"]))*1.))))) * 2.0)) * ((-((data["mean_abs_chgbatch_msignal"])))))))) +

                            0.050000*np.tanh(((np.where(data["signal_shift_+1"] <= -998, data["rangebatch_msignal"], np.sin((data["abs_maxbatch_slices2"])) )) * 2.0)) +

                            0.050000*np.tanh(np.where(((data["signal_shift_+1"]) * (((data["signal"]) * (data["maxbatch_slices2_msignal"])))) <= -998, data["meanbatch_slices2_msignal"], np.sin(((-((np.where(np.minimum(((data["abs_avgbatch_slices2"])), ((data["minbatch"]))) <= -998, data["rangebatch_slices2"], data["mean_abs_chgbatch_slices2"] )))))) )) +

                            0.050000*np.tanh(np.where(data["signal"] > -998, data["abs_maxbatch"], np.maximum(((((((data["maxbatch_slices2"]) / 2.0)) / 2.0))), ((data["abs_avgbatch_slices2_msignal"]))) )) +

                            0.050000*np.tanh(((np.maximum(((((((np.where(data["minbatch"] <= -998, data["minbatch"], (((data["minbatch_msignal"]) + (data["mean_abs_chgbatch_msignal"]))/2.0) )) * (np.sin((np.where(data["signal_shift_-1"] > -998, data["abs_maxbatch_msignal"], np.where(data["maxtominbatch"] > -998, data["abs_maxbatch_msignal"], ((data["maxtominbatch"]) - ((5.0))) ) )))))) * 2.0))), ((data["minbatch"])))) * 2.0)) +

                            0.050000*np.tanh(np.cos((np.maximum(((data["signal"])), ((np.where(data["medianbatch_slices2"] <= -998, np.where((((((data["meanbatch_slices2"]) > (data["meanbatch_slices2"]))*1.)) * ((((7.0)) * (data["meanbatch_msignal"])))) <= -998, np.maximum(((data["minbatch_slices2_msignal"])), ((data["signal"]))), data["signal"] ), data["abs_avgbatch_slices2"] ))))))) +

                            0.050000*np.tanh((-((np.sin((np.minimum(((((data["abs_maxbatch"]) * (np.sin((data["abs_avgbatch_msignal"])))))), ((data["maxtominbatch_msignal"]))))))))) +

                            0.050000*np.tanh(np.sin((data["signal_shift_-1"]))) +

                            0.050000*np.tanh(np.sin((np.where(np.cos((np.minimum(((data["abs_maxbatch_slices2_msignal"])), ((data["medianbatch_slices2"]))))) > -998, np.sin((data["maxbatch_slices2"])), (-((((((np.where(data["maxtominbatch_slices2"] > -998, data["minbatch_slices2"], data["abs_maxbatch_msignal"] )) - (data["meanbatch_slices2"]))) + (np.where(data["rangebatch_slices2"] <= -998, data["minbatch_slices2"], data["signal_shift_-1"] )))))) )))) +

                            0.050000*np.tanh(np.where(np.where(data["abs_minbatch_msignal"] <= -998, np.cos((data["abs_avgbatch_slices2_msignal"])), np.where(data["rangebatch_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], np.tanh((((data["abs_minbatch_msignal"]) + (((np.sin((data["minbatch_msignal"]))) - (((np.sin((data["meanbatch_slices2_msignal"]))) * 2.0))))))) ) ) > -998, np.sin((data["minbatch_msignal"])), data["abs_minbatch_msignal"] )) +

                            0.050000*np.tanh(((data["meanbatch_slices2"]) - (((np.where((((data["medianbatch_msignal"]) + (data["meanbatch_slices2_msignal"]))/2.0) > -998, (9.0), np.maximum((((((data["medianbatch_slices2"]) <= ((9.0)))*1.))), (((((data["signal_shift_-1_msignal"]) > (data["medianbatch_slices2_msignal"]))*1.)))) )) * (np.cos((((data["medianbatch_slices2"]) * (((data["medianbatch_slices2_msignal"]) + (data["medianbatch_slices2_msignal"]))))))))))) +

                            0.050000*np.tanh(np.minimum(((np.minimum(((data["signal"])), ((np.tanh((np.cos((data["signal"]))))))))), ((data["maxbatch_msignal"])))) +

                            0.050000*np.tanh((((data["abs_avgbatch_msignal"]) > (((np.tanh((data["stdbatch_slices2"]))) * (np.tanh((np.tanh(((-((data["signal_shift_-1_msignal"])))))))))))*1.)))    

    

    def GP_class_10(self,data):

        return self.Output( -4.940363 +

                            0.050000*np.tanh(np.cos((np.tanh(((((-((np.sin((((((data["abs_minbatch_slices2"]) * 2.0)) + ((((data["stdbatch_slices2_msignal"]) + (data["signal"]))/2.0))))))))) - (data["signal_shift_-1_msignal"]))))))) +

                            0.050000*np.tanh((-((np.where(((np.cos((np.sin((data["abs_avgbatch_msignal"]))))) / 2.0) <= -998, ((((data["meanbatch_msignal"]) / 2.0)) - (data["maxbatch_slices2_msignal"])), np.cos((data["medianbatch_slices2_msignal"])) ))))) +

                            0.050000*np.tanh(np.minimum(((np.where(data["rangebatch_slices2"] <= -998, np.cos((((np.tanh((data["abs_minbatch_msignal"]))) / 2.0))), data["minbatch_msignal"] ))), ((data["minbatch_msignal"])))) +

                            0.050000*np.tanh(((((((((data["abs_minbatch_slices2_msignal"]) > (((((((((data["maxbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2"]))) <= (np.tanh(((((np.tanh((np.sin((((data["signal"]) * 2.0)))))) > (data["meanbatch_slices2"]))*1.)))))*1.)) + ((14.98032379150390625)))/2.0)))*1.)) <= (data["signal"]))*1.)) - (data["signal_shift_+1_msignal"]))) +

                            0.050000*np.tanh(data["rangebatch_slices2"]) +

                            0.050000*np.tanh((((data["medianbatch_slices2"]) + ((((np.tanh((data["rangebatch_slices2_msignal"]))) + (np.where(np.sin((data["mean_abs_chgbatch_slices2"])) > -998, data["abs_maxbatch_msignal"], data["minbatch_slices2_msignal"] )))/2.0)))/2.0)) +

                            0.050000*np.tanh(np.where(data["maxtominbatch"] <= -998, ((np.cos((data["minbatch_slices2"]))) / 2.0), (-((((data["minbatch_slices2"]) * 2.0)))) )) +

                            0.050000*np.tanh(((data["abs_maxbatch_slices2_msignal"]) * (data["minbatch_slices2"]))) +

                            0.050000*np.tanh(np.maximum(((data["mean_abs_chgbatch_slices2"])), ((np.sin((data["abs_maxbatch"])))))) +

                            0.050000*np.tanh((((((data["signal_shift_+1_msignal"]) <= (data["rangebatch_slices2"]))*1.)) * (data["signal_shift_+1"]))) +

                            0.050000*np.tanh(((((((np.where(data["meanbatch_slices2_msignal"] <= -998, data["signal_shift_+1"], np.tanh((data["minbatch_msignal"])) )) * 2.0)) * 2.0)) * 2.0)) +

                            0.050000*np.tanh((-((data["maxtominbatch_slices2_msignal"])))) +

                            0.050000*np.tanh((((data["medianbatch_slices2"]) + (((data["abs_maxbatch"]) * 2.0)))/2.0)) +

                            0.050000*np.tanh(data["abs_minbatch_msignal"]) +

                            0.050000*np.tanh(data["rangebatch_slices2"]) +

                            0.050000*np.tanh(np.maximum(((np.minimum((((-((data["abs_minbatch_slices2_msignal"]))))), (((0.0)))))), ((np.cos((((np.cos((data["abs_minbatch_slices2_msignal"]))) - ((((np.tanh((np.where((7.0) > -998, ((data["signal"]) / 2.0), data["maxtominbatch"] )))) + (data["abs_maxbatch_slices2_msignal"]))/2.0))))))))) +

                            0.050000*np.tanh(data["signal_shift_+1"]) +

                            0.050000*np.tanh(((((((((((data["signal"]) - ((((((data["abs_maxbatch_msignal"]) + ((((((((data["signal"]) + (data["signal"]))/2.0)) * 2.0)) * 2.0)))/2.0)) * 2.0)))) > (data["mean_abs_chgbatch_slices2"]))*1.)) + (data["signal"]))/2.0)) * 2.0)) +

                            0.050000*np.tanh((((-((data["minbatch_slices2"])))) * ((((data["minbatch_slices2_msignal"]) > ((((data["abs_avgbatch_msignal"]) <= (data["stdbatch_slices2"]))*1.)))*1.)))) +

                            0.050000*np.tanh(np.sin((data["signal_shift_-1"]))) +

                            0.050000*np.tanh(np.cos((((data["rangebatch_slices2"]) + (((np.where(data["signal"] > -998, np.where(data["meanbatch_slices2_msignal"] > -998, np.where(data["signal"] > -998, (((np.maximum(((data["abs_maxbatch"])), ((data["rangebatch_slices2"])))) <= (np.tanh((data["signal_shift_+1_msignal"]))))*1.), np.minimum((((((data["abs_maxbatch_slices2"]) > (data["maxbatch_msignal"]))*1.))), (((-((data["signal_shift_+1_msignal"])))))) ), data["rangebatch_slices2_msignal"] ), data["rangebatch_slices2"] )) / 2.0)))))) +

                            0.050000*np.tanh(data["maxbatch_slices2_msignal"]) +

                            0.050000*np.tanh(((((((data["maxtominbatch_slices2_msignal"]) <= (((((data["abs_minbatch_slices2"]) + (np.minimum(((data["abs_maxbatch_msignal"])), (((((-((np.maximum(((data["maxbatch_slices2"])), ((data["abs_minbatch_slices2_msignal"]))))))) + (data["maxbatch_slices2"])))))))) / 2.0)))*1.)) <= (data["maxbatch_msignal"]))*1.)) +

                            0.050000*np.tanh(data["maxtominbatch_slices2_msignal"]) +

                            0.050000*np.tanh(np.sin((np.cos(((-((data["abs_maxbatch_slices2"])))))))) +

                            0.050000*np.tanh(((data["rangebatch_slices2"]) / 2.0)) +

                            0.050000*np.tanh(np.where((5.0) > -998, data["medianbatch_msignal"], data["abs_avgbatch_slices2"] )) +

                            0.050000*np.tanh(((np.maximum(((((np.maximum(((data["rangebatch_slices2_msignal"])), ((data["mean_abs_chgbatch_msignal"])))) * 2.0))), ((data["signal"])))) * 2.0)) +

                            0.050000*np.tanh(np.where(np.cos((data["maxbatch_slices2"])) <= -998, data["meanbatch_slices2_msignal"], data["signal"] )) +

                            0.050000*np.tanh((-(((((5.0)) + (data["abs_minbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(data["maxtominbatch_slices2"]) +

                            0.050000*np.tanh(np.cos((data["mean_abs_chgbatch_msignal"]))) +

                            0.050000*np.tanh(data["minbatch_slices2_msignal"]) +

                            0.050000*np.tanh((-((((data["signal_shift_-1_msignal"]) + (data["abs_avgbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(((((data["signal_shift_-1"]) - (data["meanbatch_msignal"]))) / 2.0)) +

                            0.050000*np.tanh(((data["rangebatch_slices2"]) / 2.0)) +

                            0.050000*np.tanh((((data["rangebatch_slices2_msignal"]) <= (np.where((((data["abs_avgbatch_msignal"]) <= (np.where(data["stdbatch_slices2"] > -998, (9.0), data["signal"] )))*1.) > -998, data["signal_shift_+1"], np.maximum(((data["maxtominbatch_msignal"])), ((data["stdbatch_slices2"]))) )))*1.)) +

                            0.050000*np.tanh((((((((((data["maxtominbatch_slices2"]) + (((data["signal_shift_+1"]) * (data["rangebatch_msignal"]))))/2.0)) + (data["signal"]))/2.0)) > (((((data["signal"]) / 2.0)) - (data["abs_maxbatch_msignal"]))))*1.)) +

                            0.050000*np.tanh(data["abs_maxbatch_slices2_msignal"]) +

                            0.050000*np.tanh(np.where(np.cos((((data["signal"]) * (np.where(data["minbatch"] > -998, data["signal_shift_+1"], data["meanbatch_msignal"] ))))) > -998, (-((data["abs_avgbatch_msignal"]))), data["meanbatch_msignal"] )) +

                            0.050000*np.tanh(np.cos((data["maxbatch_msignal"]))) +

                            0.050000*np.tanh((((data["abs_minbatch_msignal"]) <= (((data["abs_minbatch_slices2_msignal"]) / 2.0)))*1.)) +

                            0.050000*np.tanh(np.where(((((-(((11.69715595245361328))))) <= (data["stdbatch_slices2_msignal"]))*1.) <= -998, data["signal_shift_-1"], np.where(data["signal_shift_-1"] <= -998, ((((6.0)) <= (data["meanbatch_msignal"]))*1.), (-((data["stdbatch_slices2_msignal"]))) ) )) +

                            0.050000*np.tanh(((data["signal_shift_-1_msignal"]) - (data["minbatch"]))) +

                            0.050000*np.tanh(data["rangebatch_msignal"]) +

                            0.050000*np.tanh((-(((((-(((-((data["abs_maxbatch_slices2"]))))))) + (((((((((data["signal_shift_-1"]) - (np.maximum(((data["abs_maxbatch_slices2"])), ((((data["signal_shift_-1_msignal"]) / 2.0))))))) / 2.0)) - (data["maxtominbatch_slices2"]))) * ((((((4.0)) - (data["signal_shift_-1_msignal"]))) * (data["meanbatch_msignal"])))))))))) +

                            0.050000*np.tanh(data["signal"]) +

                            0.050000*np.tanh((-((np.where(data["maxtominbatch_slices2_msignal"] > -998, np.cos((((data["stdbatch_slices2_msignal"]) / 2.0))), (((((((data["minbatch"]) <= (data["abs_maxbatch_msignal"]))*1.)) + (np.where(data["abs_avgbatch_slices2"] <= -998, np.cos((((data["stdbatch_slices2_msignal"]) / 2.0))), ((data["stdbatch_slices2_msignal"]) / 2.0) )))) / 2.0) ))))) +

                            0.050000*np.tanh(((((data["medianbatch_slices2"]) - (np.sin((((((((((data["signal_shift_-1"]) - (data["signal_shift_+1"]))) / 2.0)) / 2.0)) - (((((data["abs_maxbatch_slices2"]) - (data["maxbatch_msignal"]))) * ((((((((data["maxbatch_msignal"]) + ((-((data["rangebatch_msignal"])))))/2.0)) / 2.0)) / 2.0)))))))))) - (data["maxbatch_msignal"]))) +

                            0.050000*np.tanh(((np.maximum(((((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))) * ((1.19759941101074219))))), ((((np.cos((((data["signal_shift_+1"]) * ((((-((data["meanbatch_slices2_msignal"])))) * (data["abs_maxbatch_msignal"]))))))) / 2.0))))) * (data["signal"]))) +

                            0.050000*np.tanh((((-((((data["stdbatch_msignal"]) - (data["minbatch"])))))) - (data["minbatch"]))) +

                            0.050000*np.tanh((((((-((((((((((((np.sin(((((((((((np.cos((((data["maxbatch_slices2"]) / 2.0)))) * 2.0)) > (data["meanbatch_slices2_msignal"]))*1.)) * 2.0)) * 2.0)))) * 2.0)) > (data["signal_shift_+1"]))*1.)) <= (data["abs_avgbatch_msignal"]))*1.)) + (data["abs_avgbatch_msignal"])))))) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(data["stdbatch_slices2"]) +

                            0.050000*np.tanh(np.where(((data["medianbatch_slices2"]) / 2.0) <= -998, data["maxbatch_slices2_msignal"], np.where(np.minimum(((((data["signal"]) * (np.sin(((((-((np.sin((data["mean_abs_chgbatch_msignal"])))))) / 2.0))))))), (((((((data["rangebatch_msignal"]) + (data["mean_abs_chgbatch_msignal"]))) <= (data["maxbatch_slices2_msignal"]))*1.)))) > -998, np.sin((np.sin((data["abs_maxbatch"])))), data["minbatch_slices2"] ) )) +

                            0.050000*np.tanh(((np.maximum(((np.cos((np.cos(((((data["stdbatch_slices2_msignal"]) <= (data["signal"]))*1.))))))), ((data["minbatch_slices2"])))) * (data["abs_maxbatch_slices2"]))) +

                            0.050000*np.tanh((((((((data["abs_maxbatch"]) + (data["abs_maxbatch_slices2"]))) > (((((((((-((((data["signal"]) * 2.0))))) * 2.0)) > (((data["abs_maxbatch_slices2"]) / 2.0)))*1.)) - ((-(((4.97659683227539062))))))))*1.)) - (((data["mean_abs_chgbatch_slices2_msignal"]) - ((-((np.where(data["maxtominbatch_msignal"] <= -998, ((data["stdbatch_slices2_msignal"]) * (data["mean_abs_chgbatch_msignal"])), data["abs_maxbatch_slices2_msignal"] ))))))))) +

                            0.050000*np.tanh((((((((-((data["signal_shift_+1"])))) > (data["stdbatch_slices2_msignal"]))*1.)) <= (((data["maxbatch_slices2_msignal"]) * ((-((((((data["maxbatch_msignal"]) * (np.tanh((data["maxtominbatch"]))))) - (data["maxbatch_msignal"])))))))))*1.)) +

                            0.050000*np.tanh(np.maximum(((data["abs_minbatch_msignal"])), (((((data["meanbatch_slices2"]) > ((((((data["stdbatch_slices2_msignal"]) / 2.0)) + (data["abs_minbatch_slices2"]))/2.0)))*1.))))) +

                            0.050000*np.tanh((((data["maxtominbatch"]) <= ((-((((data["signal_shift_+1"]) + (np.tanh((np.minimum(((data["mean_abs_chgbatch_slices2"])), ((np.minimum(((np.minimum(((data["signal_shift_+1"])), ((data["mean_abs_chgbatch_slices2"]))))), ((data["abs_minbatch_slices2"])))))))))))))))*1.)) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) - (((data["signal_shift_+1"]) / 2.0)))) +

                            0.050000*np.tanh(((((data["abs_minbatch_slices2_msignal"]) * (data["meanbatch_msignal"]))) - (np.where(((data["mean_abs_chgbatch_slices2"]) + (((data["stdbatch_slices2"]) + (data["minbatch_slices2_msignal"])))) > -998, data["abs_avgbatch_slices2_msignal"], ((data["signal_shift_-1_msignal"]) + (((data["abs_avgbatch_msignal"]) * (data["stdbatch_msignal"])))) )))) +

                            0.050000*np.tanh(((((((((data["abs_avgbatch_slices2_msignal"]) * (data["meanbatch_msignal"]))) * 2.0)) - (data["abs_avgbatch_slices2_msignal"]))) - (data["abs_avgbatch_slices2"]))) +

                            0.050000*np.tanh(np.where(data["abs_minbatch_slices2_msignal"] <= -998, np.sin(((9.93030166625976562))), ((np.tanh((np.sin(((((3.0)) - (np.maximum(((data["abs_maxbatch_slices2_msignal"])), (((-((np.maximum(((data["maxtominbatch_slices2_msignal"])), ((data["abs_minbatch_slices2"])))))))))))))))) * (((data["abs_maxbatch_slices2_msignal"]) * 2.0))) )) +

                            0.050000*np.tanh(((np.where((-((data["abs_minbatch_slices2"]))) > -998, np.where(data["medianbatch_msignal"] <= -998, data["meanbatch_slices2_msignal"], (-((data["meanbatch_slices2_msignal"]))) ), data["abs_minbatch_msignal"] )) / 2.0)) +

                            0.050000*np.tanh((-((((data["maxbatch_msignal"]) - (((data["signal_shift_+1_msignal"]) - (data["mean_abs_chgbatch_slices2_msignal"])))))))) +

                            0.050000*np.tanh(((data["abs_minbatch_slices2"]) - (np.where((((np.sin((np.where(data["signal_shift_-1"] > -998, data["abs_avgbatch_slices2_msignal"], data["abs_minbatch_slices2"] )))) <= (((data["signal_shift_+1"]) / 2.0)))*1.) > -998, data["abs_avgbatch_slices2_msignal"], np.where(((np.cos((((data["abs_maxbatch_slices2"]) / 2.0)))) / 2.0) <= -998, ((data["rangebatch_slices2_msignal"]) - (np.where(data["abs_minbatch_slices2"] <= -998, data["signal_shift_-1"], data["abs_minbatch_slices2"] ))), data["meanbatch_slices2"] ) )))) +

                            0.050000*np.tanh(data["stdbatch_slices2"]) +

                            0.050000*np.tanh((((((((((((data["mean_abs_chgbatch_slices2_msignal"]) > (data["signal_shift_-1_msignal"]))*1.)) > (data["signal_shift_-1_msignal"]))*1.)) + (data["maxtominbatch_slices2"]))/2.0)) - (data["mean_abs_chgbatch_msignal"]))) +

                            0.050000*np.tanh(((((((data["medianbatch_msignal"]) * (data["medianbatch_msignal"]))) + (np.where(((data["maxbatch_slices2_msignal"]) + (np.minimum(((((data["abs_avgbatch_slices2"]) * 2.0))), ((data["abs_avgbatch_slices2"]))))) > -998, (-(((((data["abs_avgbatch_slices2"]) + (data["abs_avgbatch_msignal"]))/2.0)))), np.where(data["maxtominbatch_slices2"] > -998, data["maxtominbatch"], data["maxbatch_slices2_msignal"] ) )))) - (np.minimum(((data["maxbatch_slices2_msignal"])), ((data["abs_avgbatch_slices2"])))))) +

                            0.050000*np.tanh((((((data["abs_avgbatch_slices2_msignal"]) * (data["minbatch_slices2"]))) + (np.minimum(((data["abs_maxbatch_slices2"])), (((((np.where((((((((data["minbatch_slices2"]) / 2.0)) / 2.0)) + ((9.89667701721191406)))/2.0) > -998, data["minbatch_slices2"], ((((data["abs_avgbatch_slices2_msignal"]) * (data["minbatch_slices2"]))) * 2.0) )) <= (np.where(data["abs_minbatch_slices2_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], ((data["abs_avgbatch_slices2_msignal"]) * 2.0) )))*1.))))))/2.0)) +

                            0.050000*np.tanh(np.cos((np.where(np.where(data["signal_shift_-1"] > -998, (((data["rangebatch_msignal"]) + (np.minimum(((np.tanh((((((data["abs_maxbatch_slices2"]) - ((-((data["signal_shift_+1"])))))) * 2.0))))), ((data["abs_minbatch_slices2_msignal"])))))/2.0), np.tanh((data["signal_shift_+1"])) ) > -998, data["abs_minbatch_slices2_msignal"], data["stdbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh((-((((((data["abs_avgbatch_slices2_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))) + (np.maximum(((((((data["stdbatch_msignal"]) + (data["abs_avgbatch_slices2_msignal"]))) / 2.0))), ((np.cos((np.minimum((((((-((((data["mean_abs_chgbatch_slices2_msignal"]) + (np.cos((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((data["abs_avgbatch_slices2_msignal"]))))))))))) / 2.0))), ((data["signal_shift_-1"])))))))))))))) +

                            0.050000*np.tanh(((((data["stdbatch_slices2"]) - (data["maxbatch_msignal"]))) * 2.0)) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) - ((((((data["abs_maxbatch_slices2_msignal"]) > (((np.tanh((np.cos((np.where(data["rangebatch_msignal"] > -998, np.cos((data["mean_abs_chgbatch_msignal"])), ((data["stdbatch_slices2"]) * (((np.sin((data["mean_abs_chgbatch_msignal"]))) - (data["rangebatch_msignal"])))) )))))) - (data["signal_shift_-1"]))))*1.)) - (((data["abs_avgbatch_slices2"]) * ((((data["maxbatch_slices2"]) > (data["abs_maxbatch_slices2"]))*1.)))))))) +

                            0.050000*np.tanh((((data["maxtominbatch_msignal"]) + (((data["mean_abs_chgbatch_slices2_msignal"]) * (((data["medianbatch_slices2_msignal"]) + ((((data["maxtominbatch"]) <= (data["rangebatch_slices2_msignal"]))*1.)))))))/2.0)) +

                            0.050000*np.tanh(((data["abs_minbatch_msignal"]) * ((((data["abs_maxbatch"]) + (np.minimum(((data["maxtominbatch_msignal"])), ((((data["meanbatch_msignal"]) - ((((-((np.where(data["signal_shift_-1_msignal"] <= -998, data["medianbatch_slices2_msignal"], data["abs_avgbatch_slices2_msignal"] ))))) - (data["stdbatch_msignal"])))))))))/2.0)))) +

                            0.050000*np.tanh(((np.where(((data["maxbatch_msignal"]) * 2.0) <= -998, data["signal_shift_-1"], np.sin((np.cos((data["mean_abs_chgbatch_slices2_msignal"])))) )) * 2.0)) +

                            0.050000*np.tanh(np.where(np.where(data["minbatch"] > -998, data["abs_avgbatch_slices2_msignal"], data["abs_avgbatch_slices2_msignal"] ) > -998, (-((data["abs_avgbatch_slices2_msignal"]))), (-(((((data["abs_maxbatch_slices2"]) + (np.where(data["medianbatch_slices2"] > -998, np.where(data["minbatch_slices2_msignal"] > -998, data["abs_avgbatch_slices2_msignal"], np.sin((data["stdbatch_slices2_msignal"])) ), data["medianbatch_slices2"] )))/2.0)))) )) +

                            0.050000*np.tanh(np.where(((np.maximum(((((np.cos((data["meanbatch_msignal"]))) / 2.0))), ((data["signal_shift_-1"])))) * 2.0) <= -998, data["minbatch_msignal"], data["rangebatch_slices2_msignal"] )) +

                            0.050000*np.tanh(np.where(((((-((data["signal"])))) > (data["signal_shift_+1"]))*1.) > -998, ((data["signal_shift_+1"]) + (data["minbatch_slices2"])), data["signal_shift_+1"] )) +

                            0.050000*np.tanh(np.sin((np.where(data["signal_shift_-1"] > -998, data["stdbatch_msignal"], np.where(data["signal_shift_-1"] > -998, data["stdbatch_msignal"], (((np.where(data["mean_abs_chgbatch_slices2"] > -998, data["stdbatch_msignal"], np.tanh((np.cos((data["abs_minbatch_slices2_msignal"])))) )) <= (np.where(data["stdbatch_slices2"] <= -998, data["abs_maxbatch_msignal"], data["mean_abs_chgbatch_slices2"] )))*1.) ) )))) +

                            0.050000*np.tanh((-((data["abs_avgbatch_msignal"])))) +

                            0.050000*np.tanh(((np.where(np.where(data["abs_maxbatch_msignal"] <= -998, (((np.where(data["abs_avgbatch_slices2"] > -998, data["mean_abs_chgbatch_msignal"], data["maxbatch_msignal"] )) + (data["abs_maxbatch_msignal"]))/2.0), data["abs_maxbatch_msignal"] ) <= -998, np.where(((data["abs_maxbatch_msignal"]) - (data["abs_maxbatch_msignal"])) <= -998, data["abs_avgbatch_slices2"], data["minbatch_slices2"] ), data["minbatch_slices2"] )) - (data["stdbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(np.where(np.where(np.minimum(((data["stdbatch_msignal"])), ((((data["minbatch_slices2"]) * 2.0)))) > -998, np.cos(((-((np.sin((data["rangebatch_msignal"]))))))), np.sin((data["abs_avgbatch_msignal"])) ) <= -998, data["abs_maxbatch_slices2_msignal"], ((data["maxtominbatch"]) + ((((-((data["abs_avgbatch_slices2_msignal"])))) * 2.0))) )) +

                            0.050000*np.tanh(data["abs_maxbatch_slices2"]) +

                            0.050000*np.tanh(np.where((((((((data["medianbatch_msignal"]) + (data["abs_maxbatch_msignal"]))) > ((((-((((data["abs_maxbatch_msignal"]) / 2.0))))) * 2.0)))*1.)) - (((data["stdbatch_slices2_msignal"]) * (data["minbatch"])))) <= -998, ((np.tanh((data["minbatch"]))) / 2.0), ((((data["medianbatch_msignal"]) + (data["abs_maxbatch_msignal"]))) * (data["medianbatch_slices2_msignal"])) )) +

                            0.050000*np.tanh(np.where(((data["abs_avgbatch_slices2_msignal"]) + (np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, data["medianbatch_slices2_msignal"], data["maxbatch_msignal"] ))) <= -998, np.where(data["abs_avgbatch_slices2"] <= -998, data["minbatch_slices2"], ((data["minbatch"]) - ((((((data["abs_avgbatch_slices2"]) + (data["abs_avgbatch_slices2_msignal"]))) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0))) ), ((data["minbatch"]) - ((((data["mean_abs_chgbatch_slices2_msignal"]) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0))) )) +

                            0.050000*np.tanh(((data["maxbatch_msignal"]) + ((((np.minimum((((-((data["abs_avgbatch_msignal"]))))), ((np.maximum(((data["maxbatch_slices2_msignal"])), ((data["maxbatch_slices2_msignal"]))))))) <= (((((data["maxtominbatch"]) * (data["abs_avgbatch_msignal"]))) + (data["maxbatch_msignal"]))))*1.)))) +

                            0.050000*np.tanh(((data["abs_minbatch_slices2_msignal"]) * (((data["maxbatch_slices2_msignal"]) + (((data["abs_avgbatch_slices2_msignal"]) - (np.cos(((((data["minbatch_slices2"]) > (((np.where(data["maxtominbatch_slices2"] <= -998, data["meanbatch_msignal"], data["abs_minbatch_msignal"] )) - ((((np.where(data["abs_maxbatch_msignal"] > -998, data["stdbatch_slices2_msignal"], data["maxtominbatch"] )) + (np.cos((((data["abs_avgbatch_slices2_msignal"]) * 2.0)))))/2.0)))))*1.)))))))))) +

                            0.050000*np.tanh(((np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, data["medianbatch_slices2"], (0.31864652037620544) )) + ((-((((np.where(data["abs_minbatch_slices2_msignal"] <= -998, np.minimum(((data["meanbatch_slices2_msignal"])), ((np.where(np.where(np.minimum(((data["abs_avgbatch_slices2"])), ((np.cos((data["mean_abs_chgbatch_slices2_msignal"]))))) <= -998, data["abs_avgbatch_slices2"], data["maxbatch_msignal"] ) > -998, data["abs_minbatch_slices2"], data["abs_maxbatch"] )))), data["abs_maxbatch_slices2_msignal"] )) * 2.0))))))) +

                            0.050000*np.tanh((((data["meanbatch_msignal"]) + (((data["minbatch_slices2_msignal"]) + (((((np.where((-((data["mean_abs_chgbatch_slices2"]))) > -998, data["stdbatch_slices2_msignal"], data["rangebatch_slices2_msignal"] )) * 2.0)) * (((data["meanbatch_msignal"]) / 2.0)))))))/2.0)) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) * (((((((((data["maxtominbatch_slices2"]) <= (data["rangebatch_slices2_msignal"]))*1.)) + (data["abs_avgbatch_slices2_msignal"]))) + (((((-((np.minimum(((data["rangebatch_slices2_msignal"])), (((((data["medianbatch_slices2_msignal"]) + ((((data["stdbatch_slices2_msignal"]) <= ((-(((-((data["medianbatch_msignal"]))))))))*1.)))/2.0)))))))) <= (((data["abs_maxbatch_slices2"]) - (((data["minbatch_slices2"]) + (data["signal_shift_+1"]))))))*1.)))/2.0)))) +

                            0.050000*np.tanh(((((np.where((((((-((data["abs_avgbatch_msignal"])))) + (data["medianbatch_msignal"]))) + (np.where(data["rangebatch_slices2_msignal"] <= -998, ((((((data["rangebatch_slices2_msignal"]) + (np.cos((((data["signal"]) - (data["stdbatch_slices2_msignal"]))))))/2.0)) + (data["rangebatch_slices2_msignal"]))/2.0), data["abs_avgbatch_msignal"] ))) <= -998, data["abs_maxbatch"], np.sin((data["stdbatch_slices2_msignal"])) )) * 2.0)) * 2.0)) +

                            0.050000*np.tanh((-((((data["rangebatch_msignal"]) - (((np.where(data["signal_shift_-1_msignal"] <= -998, np.maximum(((np.maximum(((data["medianbatch_slices2_msignal"])), ((((data["meanbatch_slices2_msignal"]) * (((data["abs_avgbatch_slices2_msignal"]) * 2.0)))))))), ((((((data["meanbatch_slices2_msignal"]) * 2.0)) * (data["rangebatch_msignal"]))))), ((data["meanbatch_slices2_msignal"]) * (data["stdbatch_slices2_msignal"])) )) / 2.0))))))) +

                            0.050000*np.tanh(((np.cos((data["maxtominbatch_msignal"]))) * (((data["abs_minbatch_slices2_msignal"]) + (np.where(data["stdbatch_slices2"] <= -998, ((data["minbatch_slices2"]) + (data["medianbatch_slices2_msignal"])), np.cos((data["maxtominbatch_msignal"])) )))))) +

                            0.050000*np.tanh(np.where(((np.cos((((((((data["abs_maxbatch"]) * (data["meanbatch_msignal"]))) + (data["maxbatch_msignal"]))) * 2.0)))) * 2.0) <= -998, ((data["maxbatch_msignal"]) + (((data["maxbatch_msignal"]) + (data["meanbatch_msignal"])))), (-((((((data["maxbatch_msignal"]) + (data["meanbatch_msignal"]))) * (np.where(data["signal_shift_-1"] <= -998, data["minbatch_slices2_msignal"], data["abs_maxbatch"] )))))) )) +

                            0.050000*np.tanh(np.where(data["meanbatch_slices2_msignal"] > -998, ((data["signal"]) + (((data["signal"]) - (np.maximum(((np.where(data["signal"] > -998, data["medianbatch_slices2"], data["abs_minbatch_slices2"] ))), (((8.37151908874511719)))))))), np.cos((data["stdbatch_msignal"])) )) +

                            0.050000*np.tanh((-((((data["abs_maxbatch_msignal"]) + (np.where(np.where(data["abs_maxbatch_msignal"] <= -998, data["abs_maxbatch_msignal"], ((data["abs_avgbatch_slices2"]) / 2.0) ) > -998, data["meanbatch_msignal"], ((((-((((data["abs_maxbatch_msignal"]) + (data["stdbatch_slices2_msignal"])))))) > (((np.where(data["meanbatch_msignal"] > -998, data["abs_maxbatch_slices2"], np.where(data["abs_maxbatch_msignal"] <= -998, data["mean_abs_chgbatch_msignal"], (9.0) ) )) * (data["meanbatch_slices2"]))))*1.) ))))))) +

                            0.050000*np.tanh(((data["minbatch_msignal"]) * (np.where(data["abs_avgbatch_msignal"] > -998, np.where(np.where(data["abs_maxbatch_slices2_msignal"] > -998, data["stdbatch_slices2_msignal"], (((data["signal_shift_-1"]) > (data["stdbatch_msignal"]))*1.) ) <= -998, ((data["abs_avgbatch_msignal"]) - (np.cos((data["abs_avgbatch_msignal"])))), np.cos((data["abs_avgbatch_msignal"])) ), data["abs_maxbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, data["signal_shift_+1"], ((np.sin((((((np.sin((np.where(data["signal_shift_-1"] > -998, data["minbatch_slices2"], data["signal_shift_-1"] )))) * 2.0)) * 2.0)))) - (((data["meanbatch_msignal"]) + (np.cos((np.sin(((((data["signal_shift_-1_msignal"]) + (data["signal_shift_-1"]))/2.0))))))))) )) +

                            0.050000*np.tanh(((np.sin((data["signal"]))) * (data["signal"]))) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) - (((np.where(np.where((3.0) <= -998, data["meanbatch_slices2"], data["abs_minbatch_slices2"] ) > -998, np.where((((((((((data["signal_shift_-1"]) + (np.where(data["signal_shift_-1"] > -998, data["abs_avgbatch_slices2"], data["signal"] )))) / 2.0)) <= (data["maxbatch_slices2_msignal"]))*1.)) - (data["stdbatch_slices2"])) <= -998, data["abs_maxbatch"], data["maxbatch_msignal"] ), data["signal_shift_+1"] )) * 2.0)))) +

                            0.050000*np.tanh((-((np.where(np.maximum(((data["stdbatch_slices2"])), ((data["abs_maxbatch_slices2"]))) > -998, data["medianbatch_msignal"], data["maxbatch_slices2"] ))))) +

                            0.050000*np.tanh(np.where((5.29157781600952148) > -998, (-((data["abs_avgbatch_msignal"]))), ((((((data["signal_shift_-1"]) + (data["abs_avgbatch_msignal"]))) + (np.sin((data["abs_avgbatch_msignal"]))))) - (data["abs_avgbatch_msignal"])) )) +

                            0.050000*np.tanh((((-((data["abs_avgbatch_msignal"])))) - ((((data["maxbatch_msignal"]) + (((np.tanh((((((((-((data["abs_avgbatch_msignal"])))) * 2.0)) + (data["rangebatch_slices2"]))/2.0)))) - ((-((np.where((-((data["abs_avgbatch_msignal"]))) <= -998, data["meanbatch_slices2_msignal"], np.minimum(((data["abs_avgbatch_slices2_msignal"])), (((-((data["abs_avgbatch_msignal"])))))) ))))))))/2.0)))) +

                            0.050000*np.tanh(np.where(data["signal_shift_-1"] <= -998, np.minimum(((data["abs_maxbatch_slices2_msignal"])), ((((data["signal_shift_+1"]) * (data["signal_shift_-1"]))))), (((((((((((((data["medianbatch_slices2"]) > ((-((data["medianbatch_slices2"])))))*1.)) + (data["medianbatch_slices2"]))/2.0)) - (data["abs_maxbatch_msignal"]))) + (data["signal_shift_-1"]))/2.0)) - (data["abs_maxbatch_msignal"])) )) +

                            0.050000*np.tanh(((((np.where(np.cos((data["mean_abs_chgbatch_msignal"])) <= -998, data["abs_maxbatch_slices2"], np.cos((np.where(((data["rangebatch_slices2_msignal"]) + (data["signal_shift_-1"])) > -998, data["mean_abs_chgbatch_slices2_msignal"], ((data["abs_maxbatch_slices2"]) - ((-((np.maximum(((data["rangebatch_slices2_msignal"])), ((data["signal_shift_-1"])))))))) ))) )) - (np.where(data["stdbatch_slices2_msignal"] > -998, data["mean_abs_chgbatch_slices2_msignal"], data["mean_abs_chgbatch_slices2_msignal"] )))) - (data["rangebatch_slices2_msignal"]))) +

                            0.050000*np.tanh(np.where((((data["rangebatch_slices2_msignal"]) + (data["signal"]))/2.0) <= -998, np.maximum(((((np.cos((data["abs_minbatch_slices2_msignal"]))) * 2.0))), (((-((np.sin((data["rangebatch_slices2_msignal"])))))))), np.where(np.maximum(((data["signal"])), (((-((data["signal"])))))) <= -998, data["meanbatch_slices2"], (((data["signal"]) + ((-((data["rangebatch_slices2_msignal"])))))/2.0) ) )) +

                            0.050000*np.tanh(((np.minimum(((((((np.sin((((data["maxtominbatch_msignal"]) - (data["maxtominbatch_msignal"]))))) - ((-((data["mean_abs_chgbatch_slices2_msignal"])))))) + (((((data["abs_minbatch_msignal"]) + (((data["minbatch_slices2"]) * ((2.14036393165588379)))))) + (((data["stdbatch_slices2_msignal"]) + ((-((data["abs_minbatch_msignal"]))))))))))), ((data["abs_minbatch_msignal"])))) * (np.sin((data["abs_minbatch_msignal"]))))) +

                            0.050000*np.tanh(((np.sin((data["stdbatch_msignal"]))) * (np.maximum(((data["signal_shift_+1"])), ((data["abs_maxbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(((data["maxbatch_slices2"]) * (np.cos((np.where((((data["signal_shift_+1_msignal"]) > (data["abs_maxbatch"]))*1.) > -998, data["mean_abs_chgbatch_msignal"], np.where(np.maximum((((((-((data["abs_maxbatch"])))) * 2.0))), ((data["rangebatch_msignal"]))) <= -998, data["maxbatch_slices2"], np.where(data["medianbatch_slices2"] > -998, ((((data["mean_abs_chgbatch_msignal"]) + (data["rangebatch_slices2"]))) * (np.cos((data["abs_maxbatch"])))), data["medianbatch_slices2_msignal"] ) ) )))))) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) - ((((data["rangebatch_slices2_msignal"]) + (np.tanh((np.maximum(((((((((((-((((((12.83940505981445312)) + (data["signal_shift_+1"]))/2.0))))) + (data["minbatch_slices2_msignal"]))/2.0)) - (((data["signal_shift_+1_msignal"]) - (data["rangebatch_slices2_msignal"]))))) - (data["signal_shift_+1"])))), ((data["abs_avgbatch_slices2_msignal"])))))))/2.0)))) +

                            0.050000*np.tanh(((np.where(data["minbatch"] > -998, data["minbatch"], np.maximum(((((np.where(data["stdbatch_slices2"] <= -998, data["medianbatch_slices2"], data["mean_abs_chgbatch_slices2_msignal"] )) / 2.0))), ((data["mean_abs_chgbatch_slices2"]))) )) - (np.where(data["abs_minbatch_msignal"] > -998, np.where(np.maximum(((data["mean_abs_chgbatch_slices2_msignal"])), ((data["minbatch"]))) <= -998, data["signal_shift_-1"], data["mean_abs_chgbatch_msignal"] ), data["mean_abs_chgbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh((((-(((((((data["abs_maxbatch_slices2_msignal"]) * 2.0)) + (np.where(((((data["maxbatch_msignal"]) * 2.0)) - (data["abs_avgbatch_slices2"])) > -998, data["medianbatch_msignal"], ((((((((data["medianbatch_msignal"]) * 2.0)) + (data["maxbatch_msignal"]))/2.0)) + ((((8.0)) - (data["maxbatch_msignal"]))))/2.0) )))/2.0))))) * 2.0)) +

                            0.050000*np.tanh(((data["minbatch_slices2"]) * (((data["abs_avgbatch_slices2_msignal"]) + (((np.tanh((np.where(np.where(data["minbatch_slices2"] > -998, np.maximum((((((data["mean_abs_chgbatch_slices2_msignal"]) <= (data["minbatch_slices2"]))*1.))), ((np.maximum(((data["signal"])), ((np.maximum(((data["signal"])), ((data["minbatch_slices2"]))))))))), data["medianbatch_slices2"] ) <= -998, data["maxtominbatch_slices2"], data["signal_shift_+1"] )))) * 2.0)))))) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) - (np.where((-((np.maximum(((data["abs_avgbatch_msignal"])), (((-((data["abs_avgbatch_msignal"]))))))))) > -998, data["rangebatch_slices2_msignal"], data["medianbatch_slices2"] )))) +

                            0.050000*np.tanh(np.cos(((((((data["signal_shift_-1"]) + ((((data["abs_maxbatch_slices2_msignal"]) + (data["meanbatch_slices2"]))/2.0)))/2.0)) + (np.tanh((np.where(np.where(data["abs_maxbatch_slices2"] <= -998, data["minbatch"], data["medianbatch_msignal"] ) <= -998, ((np.where((3.0) > -998, (3.0), data["meanbatch_msignal"] )) - (data["mean_abs_chgbatch_slices2_msignal"])), (-((data["minbatch"]))) )))))))) +

                            0.050000*np.tanh((((((np.minimum(((data["meanbatch_slices2"])), ((data["medianbatch_slices2_msignal"])))) * (((data["stdbatch_msignal"]) * (np.where(data["mean_abs_chgbatch_msignal"] > -998, data["signal"], (((-((data["signal_shift_+1"])))) * 2.0) )))))) + (((data["medianbatch_slices2_msignal"]) * (((np.minimum(((data["medianbatch_slices2_msignal"])), ((data["signal"])))) * (np.sin((np.maximum(((data["signal"])), ((data["maxbatch_slices2"])))))))))))/2.0)) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) + ((-((((data["rangebatch_msignal"]) + (np.where(np.where(data["abs_maxbatch_slices2_msignal"] <= -998, (((((((data["signal"]) <= (data["signal_shift_-1"]))*1.)) + ((-((data["maxbatch_msignal"])))))) - (data["mean_abs_chgbatch_slices2_msignal"])), (((data["maxbatch_slices2"]) <= (((data["abs_maxbatch_slices2"]) / 2.0)))*1.) ) > -998, data["abs_avgbatch_slices2_msignal"], data["signal"] ))))))))) +

                            0.050000*np.tanh((((-((np.where(((data["mean_abs_chgbatch_slices2_msignal"]) + (data["rangebatch_slices2_msignal"])) > -998, ((data["mean_abs_chgbatch_slices2_msignal"]) + (data["rangebatch_slices2_msignal"])), np.maximum(((data["mean_abs_chgbatch_slices2_msignal"])), ((np.maximum(((np.cos((data["minbatch_slices2_msignal"])))), ((data["mean_abs_chgbatch_slices2_msignal"])))))) ))))) * 2.0)) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) - (((((data["maxbatch_slices2_msignal"]) - (np.cos((data["signal_shift_-1"]))))) + ((((data["signal_shift_+1"]) > (np.sin((np.minimum(((((data["maxtominbatch_slices2"]) - ((((data["signal_shift_-1"]) <= ((-((data["signal_shift_-1"])))))*1.))))), (((-((data["maxbatch_slices2_msignal"]))))))))))*1.)))))) +

                            0.050000*np.tanh(np.minimum(((np.minimum(((data["medianbatch_slices2"])), ((np.maximum(((data["abs_avgbatch_msignal"])), ((data["medianbatch_slices2"])))))))), ((np.maximum((((((data["signal_shift_+1"]) + ((((data["maxtominbatch_slices2"]) + (data["maxbatch_slices2"]))/2.0)))/2.0))), ((data["maxtominbatch_slices2"]))))))) +

                            0.050000*np.tanh(((data["abs_avgbatch_slices2"]) * (((((((((((-((data["minbatch_msignal"])))) + (data["meanbatch_slices2"]))/2.0)) - (data["signal_shift_-1_msignal"]))) - (data["stdbatch_msignal"]))) * (np.sin((data["signal"]))))))) +

                            0.050000*np.tanh(np.cos((np.where(data["rangebatch_slices2_msignal"] <= -998, data["maxtominbatch_slices2_msignal"], np.minimum(((data["mean_abs_chgbatch_msignal"])), ((data["abs_maxbatch"]))) )))) +

                            0.050000*np.tanh(data["abs_maxbatch_slices2"]) +

                            0.050000*np.tanh(((np.sin((data["stdbatch_msignal"]))) - (np.where(((data["abs_minbatch_msignal"]) + (data["maxtominbatch_slices2_msignal"])) > -998, ((data["rangebatch_slices2_msignal"]) + (data["stdbatch_msignal"])), (((1.03578352928161621)) + (data["stdbatch_slices2_msignal"])) )))) +

                            0.050000*np.tanh(np.sin((np.where(((np.sin((np.where(data["maxtominbatch_msignal"] <= -998, data["medianbatch_msignal"], data["medianbatch_msignal"] )))) * 2.0) <= -998, data["signal_shift_-1"], data["medianbatch_msignal"] )))) +

                            0.050000*np.tanh((((-((np.minimum((((((np.minimum(((((data["rangebatch_slices2_msignal"]) + (np.sin((((data["maxtominbatch_slices2_msignal"]) - (data["maxtominbatch_slices2_msignal"])))))))), ((data["rangebatch_slices2_msignal"])))) + (np.tanh((np.maximum(((data["maxbatch_msignal"])), ((data["minbatch_msignal"])))))))/2.0))), ((data["rangebatch_slices2_msignal"]))))))) - (data["stdbatch_msignal"]))) +

                            0.050000*np.tanh((((((((data["signal"]) * (((np.sin((np.where(data["maxtominbatch_msignal"] <= -998, np.sin((data["rangebatch_msignal"])), data["signal"] )))) * 2.0)))) + (((np.sin((np.where(data["maxbatch_slices2"] <= -998, data["abs_minbatch_msignal"], (-((data["mean_abs_chgbatch_slices2"]))) )))) - (data["stdbatch_slices2_msignal"]))))/2.0)) * 2.0)) +

                            0.050000*np.tanh(((((np.where(np.cos((((((np.where(((((data["mean_abs_chgbatch_msignal"]) * 2.0)) * 2.0) > -998, np.cos((data["stdbatch_slices2_msignal"])), data["stdbatch_slices2_msignal"] )) * 2.0)) * 2.0))) > -998, np.cos((data["stdbatch_slices2_msignal"])), ((np.cos((data["stdbatch_slices2_msignal"]))) * 2.0) )) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(np.cos((np.minimum(((data["mean_abs_chgbatch_msignal"])), (((((data["signal_shift_-1"]) <= (((np.cos((np.minimum(((np.minimum(((data["signal_shift_-1"])), ((data["abs_minbatch_slices2_msignal"]))))), ((data["maxbatch_slices2_msignal"])))))) + (np.cos((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), (((((((data["medianbatch_msignal"]) + ((-((((data["signal_shift_-1"]) * (data["rangebatch_slices2_msignal"])))))))) <= (data["mean_abs_chgbatch_msignal"]))*1.))))))))))*1.))))))) +

                            0.050000*np.tanh(((data["stdbatch_slices2_msignal"]) * (np.where(np.sin((np.sin((data["abs_minbatch_msignal"])))) <= -998, ((data["stdbatch_slices2_msignal"]) * (np.sin((data["abs_minbatch_msignal"])))), ((np.sin((data["abs_minbatch_msignal"]))) / 2.0) )))) +

                            0.050000*np.tanh((((((data["stdbatch_slices2_msignal"]) * (data["meanbatch_msignal"]))) + ((-((np.where(np.sin((data["meanbatch_msignal"])) > -998, (8.85711002349853516), (-((data["meanbatch_msignal"]))) ))))))/2.0)) +

                            0.050000*np.tanh(np.cos((np.maximum(((np.minimum(((data["maxtominbatch_msignal"])), ((data["medianbatch_slices2_msignal"]))))), ((data["signal_shift_-1"])))))) +

                            0.050000*np.tanh(np.where(data["abs_maxbatch_msignal"] <= -998, data["minbatch_slices2"], np.minimum(((np.where(data["abs_maxbatch_msignal"] <= -998, ((data["minbatch_slices2"]) - (data["rangebatch_msignal"])), np.minimum(((((np.maximum(((data["maxbatch_slices2"])), ((data["signal"])))) - ((((data["rangebatch_msignal"]) + (((data["abs_minbatch_slices2_msignal"]) - (data["rangebatch_msignal"]))))/2.0))))), ((data["signal_shift_+1"]))) ))), ((data["rangebatch_msignal"]))) )) +

                            0.050000*np.tanh(np.maximum(((((((np.where(((data["mean_abs_chgbatch_slices2"]) - (data["minbatch_slices2_msignal"])) > -998, np.cos((data["mean_abs_chgbatch_slices2_msignal"])), (-((data["rangebatch_slices2"]))) )) / 2.0)) * 2.0))), ((data["minbatch_slices2_msignal"])))) +

                            0.050000*np.tanh(((((data["signal_shift_-1"]) - (data["abs_maxbatch"]))) - (np.where(((((data["signal_shift_+1"]) - (((data["maxbatch_msignal"]) - (data["meanbatch_msignal"]))))) * 2.0) <= -998, data["signal_shift_-1"], ((((data["maxbatch_msignal"]) - (data["signal_shift_+1"]))) + ((((np.sin((data["meanbatch_msignal"]))) <= (data["medianbatch_msignal"]))*1.))) )))) +

                            0.050000*np.tanh(((np.sin((data["abs_minbatch_msignal"]))) * (((data["mean_abs_chgbatch_slices2_msignal"]) - (np.sin((data["maxbatch_msignal"]))))))) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) * ((((((data["medianbatch_msignal"]) * (data["abs_avgbatch_msignal"]))) + (data["mean_abs_chgbatch_slices2_msignal"]))/2.0)))) +

                            0.050000*np.tanh(((((np.where(((np.sin((data["signal_shift_-1"]))) * 2.0) > -998, data["signal_shift_-1"], data["signal_shift_-1"] )) - (data["rangebatch_msignal"]))) * 2.0)) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) - (np.where(((data["meanbatch_msignal"]) * 2.0) > -998, (6.0), (((6.0)) - (data["abs_maxbatch_slices2"])) )))) +

                            0.050000*np.tanh(((((((np.where(np.cos((data["mean_abs_chgbatch_slices2"])) > -998, ((data["signal_shift_-1_msignal"]) - (data["medianbatch_msignal"])), data["abs_minbatch_slices2"] )) - (data["abs_avgbatch_msignal"]))) * (np.where((((data["signal_shift_+1"]) + (data["mean_abs_chgbatch_slices2"]))/2.0) > -998, ((data["signal_shift_-1_msignal"]) - (data["medianbatch_msignal"])), data["abs_maxbatch_slices2_msignal"] )))) - (np.where(data["signal_shift_-1_msignal"] > -998, data["rangebatch_slices2_msignal"], data["medianbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(((((((data["abs_avgbatch_msignal"]) - (data["maxtominbatch"]))) * (data["mean_abs_chgbatch_msignal"]))) * ((5.33403539657592773)))) +

                            0.050000*np.tanh(np.sin((np.where(np.sin((np.where(((data["maxtominbatch_slices2_msignal"]) - (data["abs_maxbatch_slices2_msignal"])) > -998, np.sin((data["signal"])), ((data["signal"]) + (data["abs_maxbatch"])) ))) > -998, np.where(data["mean_abs_chgbatch_msignal"] > -998, data["maxbatch_slices2"], data["stdbatch_slices2_msignal"] ), data["signal_shift_+1_msignal"] )))) +

                            0.050000*np.tanh(((data["maxbatch_slices2"]) * (np.where(data["signal_shift_-1_msignal"] <= -998, ((data["medianbatch_slices2"]) + (np.where(((data["signal_shift_-1"]) + (data["signal_shift_-1"])) <= -998, data["abs_minbatch_slices2_msignal"], ((data["signal_shift_-1"]) + (data["signal_shift_-1"])) ))), np.sin((data["medianbatch_msignal"])) )))) +

                            0.050000*np.tanh(np.maximum(((((data["medianbatch_msignal"]) * 2.0))), ((np.cos((data["abs_maxbatch_slices2_msignal"])))))) +

                            0.050000*np.tanh(np.where(np.where(data["signal_shift_-1"] > -998, data["abs_maxbatch_slices2"], np.minimum(((((data["medianbatch_slices2"]) * ((((0.20071868598461151)) / 2.0))))), ((data["maxtominbatch"]))) ) > -998, data["signal_shift_-1"], np.minimum(((((data["maxtominbatch_slices2"]) / 2.0))), ((data["maxtominbatch"]))) )) +

                            0.050000*np.tanh((-((((np.sin((np.where(data["stdbatch_msignal"] > -998, data["abs_minbatch_slices2_msignal"], np.where(data["minbatch"] > -998, data["abs_minbatch_slices2_msignal"], data["maxbatch_msignal"] ) )))) * 2.0))))) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) - (np.maximum(((data["signal_shift_+1"])), ((((np.maximum(((data["maxbatch_slices2_msignal"])), ((((np.cos((data["signal_shift_-1"]))) * (np.sin((np.cos((data["stdbatch_msignal"])))))))))) * 2.0))))))) +

                            0.050000*np.tanh(((np.cos((np.where(data["signal"] > -998, data["signal_shift_-1"], np.maximum(((np.where((3.0) > -998, data["signal_shift_-1"], data["minbatch_slices2_msignal"] ))), ((data["medianbatch_slices2"]))) )))) * (data["signal_shift_-1"]))) +

                            0.050000*np.tanh(np.where(((data["signal"]) * 2.0) <= -998, data["abs_maxbatch_msignal"], np.where(((data["abs_maxbatch"]) + ((((data["signal"]) <= (data["rangebatch_msignal"]))*1.))) <= -998, data["minbatch_slices2"], ((data["signal"]) - (np.minimum(((data["rangebatch_msignal"])), ((data["rangebatch_msignal"]))))) ) )) +

                            0.050000*np.tanh(np.maximum(((np.minimum(((data["abs_maxbatch"])), ((data["signal_shift_-1"]))))), ((data["abs_maxbatch_slices2_msignal"])))) +

                            0.050000*np.tanh((((((((((-(((-((np.minimum((((((-((data["rangebatch_slices2_msignal"])))) * 2.0))), (((11.00670146942138672))))))))))) / 2.0)) <= (np.cos((data["stdbatch_slices2"]))))*1.)) <= (np.where((9.0) <= -998, ((((((data["stdbatch_slices2"]) > (data["meanbatch_msignal"]))*1.)) + (data["signal_shift_-1_msignal"]))/2.0), data["abs_minbatch_msignal"] )))*1.)) +

                            0.050000*np.tanh(np.where(data["abs_maxbatch_slices2_msignal"] > -998, ((((9.0)) + (data["maxtominbatch"]))/2.0), ((np.cos((np.where(data["signal_shift_+1"] <= -998, data["meanbatch_slices2"], data["abs_maxbatch_slices2_msignal"] )))) * 2.0) )) +

                            0.050000*np.tanh(np.cos((np.where(np.cos((((((data["mean_abs_chgbatch_msignal"]) / 2.0)) + (((((data["signal_shift_+1_msignal"]) + (data["signal_shift_+1_msignal"]))) / 2.0))))) <= -998, np.where(((((10.83226490020751953)) + (data["abs_avgbatch_msignal"]))/2.0) > -998, data["mean_abs_chgbatch_slices2_msignal"], np.tanh((((data["abs_maxbatch"]) * (np.minimum(((data["abs_avgbatch_slices2"])), ((np.cos((data["signal_shift_+1"]))))))))) ), data["mean_abs_chgbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(((np.sin((((data["abs_maxbatch"]) - (np.where(data["meanbatch_slices2"] > -998, data["signal_shift_+1"], ((np.where(data["abs_maxbatch_slices2"] > -998, data["signal_shift_+1"], ((np.maximum((((-((data["meanbatch_slices2"]))))), (((-((data["minbatch_slices2"]))))))) / 2.0) )) / 2.0) )))))) * 2.0)) +

                            0.050000*np.tanh(data["meanbatch_slices2"]) +

                            0.050000*np.tanh(np.where(data["mean_abs_chgbatch_slices2_msignal"] <= -998, (((data["meanbatch_slices2"]) > (np.where(np.where(np.cos((np.cos((data["mean_abs_chgbatch_slices2_msignal"])))) > -998, np.tanh((data["medianbatch_slices2"])), data["minbatch_slices2_msignal"] ) <= -998, np.cos((data["minbatch_slices2"])), data["meanbatch_msignal"] )))*1.), np.cos(((((data["mean_abs_chgbatch_slices2_msignal"]) > (data["signal_shift_-1_msignal"]))*1.))) )) +

                            0.050000*np.tanh(np.where(np.cos(((((data["maxtominbatch"]) + (((((data["abs_avgbatch_slices2_msignal"]) - ((-((((data["minbatch_slices2_msignal"]) + ((-((np.tanh((data["abs_avgbatch_slices2_msignal"]))))))))))))) * (data["signal"]))))/2.0))) > -998, ((np.sin((data["signal"]))) * (data["rangebatch_slices2_msignal"])), data["rangebatch_msignal"] )) +

                            0.050000*np.tanh(((data["signal_shift_-1"]) - (((((7.94279861450195312)) + ((((((data["signal_shift_-1_msignal"]) - (np.sin((np.cos((np.cos((data["abs_avgbatch_msignal"]))))))))) + (np.maximum(((data["signal_shift_+1"])), ((data["signal_shift_-1_msignal"])))))/2.0)))/2.0)))) +

                            0.050000*np.tanh(((((np.maximum(((np.maximum(((data["signal_shift_+1"])), ((data["minbatch"]))))), (((2.0))))) + (data["minbatch"]))) - ((((((-((data["meanbatch_msignal"])))) / 2.0)) * ((((((data["minbatch"]) <= ((-((data["abs_maxbatch"])))))*1.)) / 2.0)))))) +

                            0.050000*np.tanh(np.sin((np.where(data["maxbatch_slices2"] <= -998, ((data["maxbatch_msignal"]) + (data["abs_maxbatch_slices2_msignal"])), data["abs_maxbatch"] )))) +

                            0.050000*np.tanh((-((np.maximum(((data["mean_abs_chgbatch_msignal"])), ((((np.cos((data["signal_shift_+1"]))) * (np.where(np.maximum(((data["abs_maxbatch_slices2_msignal"])), (((((data["abs_maxbatch_slices2_msignal"]) > (data["minbatch_slices2"]))*1.)))) > -998, ((data["minbatch_slices2"]) + (data["mean_abs_chgbatch_msignal"])), np.sin((data["maxbatch_slices2"])) )))))))))) +

                            0.050000*np.tanh(np.where(((np.tanh(((((((((data["abs_maxbatch"]) * 2.0)) / 2.0)) <= (data["medianbatch_slices2"]))*1.)))) * 2.0) > -998, np.cos((data["mean_abs_chgbatch_msignal"])), data["abs_maxbatch_slices2_msignal"] )) +

                            0.050000*np.tanh(data["signal_shift_+1"]) +

                            0.050000*np.tanh(((((np.cos((data["signal_shift_-1"]))) * (data["abs_minbatch_slices2_msignal"]))) * (data["minbatch_slices2_msignal"]))) +

                            0.050000*np.tanh(np.where(data["abs_minbatch_msignal"] <= -998, data["signal_shift_+1_msignal"], data["signal_shift_-1_msignal"] )) +

                            0.050000*np.tanh(np.where(((np.where(np.sin((data["maxbatch_msignal"])) <= -998, data["maxbatch_slices2_msignal"], data["medianbatch_slices2"] )) + (data["maxbatch_slices2"])) <= -998, data["abs_maxbatch"], np.where(data["maxbatch_slices2_msignal"] > -998, data["signal_shift_-1_msignal"], np.minimum(((((((((((((data["minbatch_slices2_msignal"]) + (np.maximum(((data["signal_shift_+1_msignal"])), ((data["signal_shift_-1_msignal"])))))) + (data["abs_minbatch_slices2"]))/2.0)) + (data["meanbatch_slices2"]))/2.0)) / 2.0))), ((data["stdbatch_msignal"]))) ) )) +

                            0.050000*np.tanh(np.cos((np.where(data["rangebatch_slices2_msignal"] <= -998, ((((np.where(np.where(((data["minbatch_slices2"]) / 2.0) > -998, np.cos((data["mean_abs_chgbatch_msignal"])), np.where(data["rangebatch_slices2_msignal"] <= -998, data["mean_abs_chgbatch_msignal"], data["mean_abs_chgbatch_slices2_msignal"] ) ) <= -998, ((data["signal"]) * (data["signal_shift_-1"])), data["mean_abs_chgbatch_slices2_msignal"] )) / 2.0)) + (np.cos((np.cos((np.cos((data["maxtominbatch_msignal"])))))))), data["mean_abs_chgbatch_slices2_msignal"] )))) +

                            0.050000*np.tanh(data["rangebatch_slices2_msignal"]) +

                            0.050000*np.tanh(np.sin((np.sin((np.where(data["meanbatch_msignal"] > -998, (-((np.where(data["maxtominbatch_msignal"] > -998, data["abs_minbatch_slices2_msignal"], data["minbatch_slices2_msignal"] )))), data["meanbatch_msignal"] )))))) +

                            0.050000*np.tanh(((np.where(data["abs_minbatch_slices2_msignal"] <= -998, ((np.where(np.where(data["abs_maxbatch_slices2_msignal"] <= -998, data["rangebatch_slices2"], data["minbatch_msignal"] ) <= -998, data["signal_shift_-1_msignal"], data["abs_avgbatch_slices2"] )) * (((data["meanbatch_slices2_msignal"]) - (data["signal_shift_-1"])))), (-((((data["rangebatch_msignal"]) - (data["signal_shift_-1"]))))) )) * 2.0)) +

                            0.050000*np.tanh(((((((data["signal_shift_+1"]) * 2.0)) - ((12.72679519653320312)))) - (np.maximum(((np.where(data["mean_abs_chgbatch_slices2_msignal"] > -998, data["signal_shift_+1_msignal"], data["maxtominbatch"] ))), ((((((((((data["signal_shift_+1"]) * 2.0)) - ((12.72679519653320312)))) - (np.maximum(((data["signal_shift_+1"])), ((data["maxtominbatch"])))))) * 2.0))))))) +

                            0.050000*np.tanh(((np.minimum(((data["signal_shift_+1_msignal"])), ((data["signal_shift_-1"])))) * 2.0)) +

                            0.050000*np.tanh(np.where(data["signal_shift_+1"] > -998, (((np.cos((data["signal_shift_+1_msignal"]))) + (data["rangebatch_slices2"]))/2.0), (-((np.cos(((-((data["signal_shift_+1_msignal"])))))))) )) +

                            0.050000*np.tanh(((((data["signal_shift_-1"]) - ((((((data["abs_maxbatch"]) + (np.where(data["maxbatch_msignal"] > -998, np.sin((np.sin((data["abs_maxbatch"])))), data["minbatch_msignal"] )))/2.0)) * 2.0)))) - (((data["abs_avgbatch_msignal"]) - (np.sin((data["abs_maxbatch"]))))))) +

                            0.050000*np.tanh(((np.sin((np.cos((data["stdbatch_slices2_msignal"]))))) * (((data["signal"]) * 2.0)))) +

                            0.050000*np.tanh(np.where(np.sin((((np.minimum(((((data["signal_shift_+1_msignal"]) / 2.0))), ((data["rangebatch_slices2"])))) * ((((data["maxtominbatch_slices2_msignal"]) <= (data["minbatch_slices2"]))*1.))))) <= -998, data["rangebatch_slices2"], np.sin((np.where(data["signal_shift_+1"] > -998, (((np.sin((((np.cos((data["stdbatch_msignal"]))) * 2.0)))) + ((((data["minbatch_msignal"]) > (data["rangebatch_slices2"]))*1.)))/2.0), data["minbatch_msignal"] ))) )) +

                            0.050000*np.tanh(((data["maxbatch_slices2"]) + (((((((((3.92767763137817383)) > ((((data["abs_maxbatch_msignal"]) > (np.sin((np.sin((data["maxtominbatch_msignal"]))))))*1.)))*1.)) - ((7.16431427001953125)))) + (data["meanbatch_slices2"]))))) +

                            0.050000*np.tanh(np.sin(((((np.sin((data["abs_minbatch_msignal"]))) + (np.where((((((6.67216062545776367)) - (data["signal_shift_-1"]))) + (data["stdbatch_msignal"])) <= -998, np.cos((np.maximum(((np.sin((data["signal"])))), ((((((data["minbatch_slices2"]) - (np.tanh((((data["abs_minbatch_msignal"]) / 2.0)))))) + (data["abs_maxbatch"]))))))), np.minimum(((data["abs_minbatch_slices2_msignal"])), ((data["stdbatch_msignal"]))) )))/2.0)))) +

                            0.050000*np.tanh((((((data["minbatch_slices2_msignal"]) + (np.maximum((((((data["abs_avgbatch_msignal"]) + (data["meanbatch_slices2"]))/2.0))), (((((data["mean_abs_chgbatch_slices2_msignal"]) + (np.sin((((data["mean_abs_chgbatch_slices2_msignal"]) - (((data["minbatch_slices2_msignal"]) * (data["meanbatch_slices2"]))))))))/2.0))))))) + (((data["mean_abs_chgbatch_slices2_msignal"]) * (data["abs_avgbatch_msignal"]))))/2.0)) +

                            0.050000*np.tanh(np.cos((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), ((((((((np.minimum(((data["mean_abs_chgbatch_slices2_msignal"])), (((((data["signal_shift_+1_msignal"]) <= (((data["signal_shift_+1_msignal"]) / 2.0)))*1.))))) > (data["signal_shift_+1_msignal"]))*1.)) <= (data["rangebatch_slices2_msignal"]))*1.))))))) +

                            0.050000*np.tanh(((np.where(np.sin((data["mean_abs_chgbatch_slices2_msignal"])) > -998, ((np.where(data["signal_shift_+1"] > -998, np.sin((data["signal"])), ((data["meanbatch_slices2_msignal"]) - (data["maxtominbatch_slices2"])) )) * 2.0), data["meanbatch_slices2_msignal"] )) * 2.0)) +

                            0.050000*np.tanh(np.where(data["maxbatch_slices2_msignal"] <= -998, np.cos((data["maxbatch_slices2_msignal"])), np.cos((data["signal_shift_+1"])) )) +

                            0.050000*np.tanh(np.where(data["signal_shift_-1_msignal"] <= -998, data["signal_shift_-1_msignal"], data["signal_shift_-1_msignal"] )) +

                            0.050000*np.tanh(((((-((data["abs_minbatch_slices2"])))) <= (np.where(data["abs_maxbatch_slices2"] <= -998, data["minbatch"], np.cos((data["signal_shift_+1"])) )))*1.)) +

                            0.050000*np.tanh(((data["signal_shift_-1_msignal"]) * (np.where(np.where((((((data["signal"]) + (data["maxbatch_msignal"]))/2.0)) * 2.0) <= -998, data["signal"], data["maxbatch_msignal"] ) <= -998, np.sin((data["signal_shift_+1_msignal"])), data["signal"] )))) +

                            0.050000*np.tanh((((((((data["signal_shift_+1"]) > ((-((((data["abs_avgbatch_slices2_msignal"]) * 2.0))))))*1.)) * 2.0)) * 2.0)) +

                            0.050000*np.tanh(((data["signal_shift_+1"]) - (np.where(((data["medianbatch_slices2_msignal"]) - (np.cos((np.maximum(((data["signal_shift_+1"])), ((data["signal_shift_+1"]))))))) <= -998, (((((data["signal"]) * 2.0)) <= (((((data["signal"]) + (data["signal"]))) - (np.where(np.sin((data["maxtominbatch_slices2"])) <= -998, data["mean_abs_chgbatch_msignal"], (4.0) )))))*1.), (4.0) )))) +

                            0.050000*np.tanh((((data["medianbatch_slices2"]) > (((((((((((((np.cos(((1.0)))) * 2.0)) / 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)))*1.)) +

                            0.050000*np.tanh(np.cos((np.minimum(((np.minimum(((data["minbatch_msignal"])), ((((((13.14470481872558594)) <= (((np.cos((data["abs_minbatch_slices2"]))) * 2.0)))*1.)))))), ((np.sin((data["abs_maxbatch_msignal"])))))))) +

                            0.050000*np.tanh(np.where(data["minbatch_slices2"] <= -998, data["signal_shift_-1_msignal"], np.where(np.sin((data["signal_shift_-1_msignal"])) <= -998, data["signal_shift_-1_msignal"], ((np.sin((np.where(np.sin((data["stdbatch_msignal"])) > -998, data["stdbatch_msignal"], data["meanbatch_slices2_msignal"] )))) * 2.0) ) )) +

                            0.050000*np.tanh((-((((((((-((data["abs_avgbatch_slices2"])))) + (np.cos((data["signal"]))))/2.0)) - (data["signal_shift_+1_msignal"])))))) +

                            0.050000*np.tanh(np.sin(((5.0)))) +

                            0.050000*np.tanh(np.maximum((((-(((((12.85178279876708984)) + (np.where((((12.85178279876708984)) + (np.where(data["abs_maxbatch"] > -998, data["medianbatch_slices2"], ((data["medianbatch_slices2"]) * 2.0) ))) > -998, ((data["meanbatch_slices2"]) * (data["stdbatch_msignal"])), (12.85178279876708984) )))))))), ((data["minbatch_slices2"])))) +

                            0.050000*np.tanh(np.where(data["stdbatch_slices2"] > -998, data["signal_shift_-1_msignal"], np.where(np.maximum(((data["minbatch_slices2"])), ((np.where(data["maxtominbatch_msignal"] > -998, data["abs_maxbatch"], data["maxbatch_slices2"] )))) > -998, data["meanbatch_slices2"], np.maximum(((((data["signal_shift_+1"]) * 2.0))), ((((data["stdbatch_slices2_msignal"]) / 2.0)))) ) )) +

                            0.050000*np.tanh(data["meanbatch_slices2"]) +

                            0.050000*np.tanh(((np.sin((np.where(data["minbatch_slices2"] > -998, data["signal_shift_+1"], (((data["maxbatch_slices2_msignal"]) + (np.sin((np.where(data["meanbatch_slices2"] > -998, data["stdbatch_slices2_msignal"], data["signal_shift_+1_msignal"] )))))/2.0) )))) * 2.0)) +

                            0.050000*np.tanh(((data["signal_shift_+1_msignal"]) + (np.where(data["signal_shift_+1"] > -998, data["abs_maxbatch"], np.cos(((-((data["abs_minbatch_slices2_msignal"]))))) )))) +

                            0.050000*np.tanh(np.sin(((((data["abs_maxbatch"]) + (data["signal_shift_-1"]))/2.0))))) 
gp = GP()
train = pd.read_csv('../input/data-without-drift/train_clean.csv')

test = pd.read_csv('../input/data-without-drift/test_clean.csv')
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
base = '../input/data-without-drift/'

train = pd.read_csv(os.path.join(base + 'train_clean.csv'))

test  = pd.read_csv(os.path.join(base + 'test_clean.csv'))
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



col = [c for c in train.columns if c not in ['time', 'open_channels', 'batch', 'batch_index', 'batch_slices',

                                             'batch_slices2',

                                             'mean_abs_chgbatch', 'meanbatch', 'rangebatch', 'stdbatch',

                                             'maxbatch', 'medianbatch', 'abs_minbatch', 'abs_avgbatch']]

target = train['open_channels']

train = train[col]
train.replace(np.inf,np.nan,inplace=True)

train.replace(-np.inf,np.nan,inplace=True)

test.replace(np.inf,np.nan,inplace=True)

test.replace(-np.inf,np.nan,inplace=True)
train_preds = gp.GrabPredictions(train[col].fillna(-999))

test_preds = gp.GrabPredictions(test[col].fillna(-999))
print('Score:',f1_score(target.values,np.argmax(train_preds.values,axis=1),average='macro'))
test['open_channels'] = np.argmax(test_preds.values,axis=1)

test[['time','open_channels']].to_csv('gpsubmission.csv', index=False, float_format='%.4f')
def MacroF1Metric(preds, dtrain):

    labels = dtrain.get_label()

    preds = preds.reshape((-1,11))

    preds = np.argmax(preds,axis=1)

    score = f1_score(labels, preds, average = 'macro')

    return ('MacroF1Metric', score, True)
import gc

import lightgbm as lgb

from sklearn import model_selection



for c in range(11):

    train['gp_'+str(int(c))] = train_preds.values[:,c]

    test['gp_'+str(int(c))] = test_preds.values[:,c]

x1, x2, y1, y2 = model_selection.train_test_split(train, target, test_size=0.2, random_state=7)

del train

gc.collect()

col = x1.columns



params = {'learning_rate': 0.1,

          'max_depth': -1,

          'num_leaves':2**7+1,

          'objective': 'multiclass',

          'num_class': 11,

          'metric': 'multi_logloss',

          'random_state': 7,

          'n_jobs':4} 

model = lgb.train(params, lgb.Dataset(x1, y1), 22222,  lgb.Dataset(x2, y2), verbose_eval=0, early_stopping_rounds=250,feval=MacroF1Metric)

preds_lgb = (model.predict(test[col], num_iteration=model.best_iteration))

oof_lgb = (model.predict(x2, num_iteration=model.best_iteration))

print('f1_score',f1_score(y2,np.argmax(oof_lgb,axis=1),average='macro'))

test['open_channels'] = np.argmax(preds_lgb,axis=1)

test[['time','open_channels']].to_csv('lgbsubmission.csv', index=False, float_format='%.4f')