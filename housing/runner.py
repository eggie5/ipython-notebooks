from model import HouseModel
import pandas as pd
import numpy as np

#train model
house_model = HouseModel("single_family_home_values.csv")

#!!!!!!!!!!!!!!!!!!!!!!! PUT THE RESERVE DATASET IN test.csv !!!!!!!!!!!
reserve = HouseModel.preprocess("test.csv").values
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
features = reserve[:,0:-1]
labels = reserve[:,-1] # last column is labels/targets



print "\n***** TEST ERROR STATS ********"
residuals = pd.Series(house_model.residuals(features,labels))
mean_diff = np.mean(residuals)
p_mean_diff = np.mean(mean_diff/np.exp(labels))
print "R^2:\t %0.5f" % house_model.score(features,labels)
print "%%Err:\t %0.5f%%" % (p_mean_diff*100)
print residuals.describe()
print "***** TEST ERROR STATS ********"