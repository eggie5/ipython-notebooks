from sklearn import preprocessing
from scipy import stats
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from mlxtend.regressor import StackingRegressor
import warnings
warnings.filterwarnings('ignore')

class HouseModel(object):
  
  def __init__(self, path):
    super(HouseModel, self).__init__()
    
    self.build(path)

  def build(self, path):
    df = HouseModel.preprocess(path).values
    features = df[:,0:-1]
    labels = df[:,-1] 
    regressor = self.train(features, labels)
    self.model = regressor
    
  @staticmethod
  def preprocess(path):
    csv = pd.read_csv(path, parse_dates=["last_sale_date"])

    # fill missing values (0's) w/ the median for the column
    cols = ["square_footage", "lot_size", "num_rooms", "num_bedrooms", "num_baths", "year_built", 
         "last_sale_amount", "estimated_value"]

    #impune missing values
    for col in cols:
        csv.loc[csv[col]==0, col] = csv[col].median()
    
    model_cols = ["square_footage", "lot_size", "num_rooms", "num_bedrooms", "num_baths", "year_built", 
                  "last_sale_amount", "last_sale_date", "estimated_value"]
    housing=csv[model_cols]
    housing.head()

    #transform dates

    housing["day"] = (housing.last_sale_date - pd.datetime(1900,1,1))/np.timedelta64(1,'D')
    housing.drop("last_sale_date", 1, inplace=True)

    #outliers

    housing_sans_outliers = housing[(np.abs(stats.zscore(housing)) < 3).all(axis=1)]

    #log transform skewed numeric features:
    numeric_feats = housing_sans_outliers.dtypes[housing_sans_outliers.dtypes != "object"].index

    skewed_feats = housing_sans_outliers[numeric_feats].apply(lambda x: stats.skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]

    skewed_feats = skewed_feats.index
  
    housing_sans_outliers[skewed_feats] = np.log1p(housing_sans_outliers[skewed_feats])

    final = housing_sans_outliers[["square_footage", "lot_size", "num_rooms", "num_bedrooms", "num_baths", "year_built", 
             "last_sale_amount", "day", "estimated_value"]]
           
    return final


  def train(self, X,y):
    features = X
    labels = y

    #test train split
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=4)

    #Ridge
    regcv = linear_model.RidgeCV(alphas=[0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75])
    regcv.fit(features, labels)
    regcv.alpha_  
    reg = linear_model.Ridge(alpha=regcv.alpha_)
    reg.fit(features, labels)

    # GB
    params = {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 2,
              'learning_rate': 0.1, 'loss': 'ls'}
    gbr = ensemble.GradientBoostingRegressor(**params)
    gbr.fit(features, labels)


    #blended model
    meta = linear_model.LinearRegression()
    blender = StackingRegressor(regressors=[reg, gbr], meta_regressor=meta)
    _=blender.fit(features, labels)
    y_pred = blender.predict(X_test)

    print "***** TRAINING STATS ********"
    scores = cross_val_score(blender, features, labels, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    mean_diff = np.mean(np.abs(np.exp(Y_test)-np.exp(y_pred)))
    p_mean_diff = np.mean(mean_diff/np.exp(Y_test))
    print "Mean Error:\t %.0f/%0.3f%%" % (mean_diff, p_mean_diff*100)
    print "***** TRAINING STATS ********"
    
    return blender
      
  def predict(self, X):
    return self.model.predict(X)
  
  def score(self, X, Y):
    return self.model.score(X,Y)
    
  def residuals(self, X,y):
    y_pred = self.predict(X)
    residuals = np.abs(np.exp(y)-np.exp(y_pred))
    mean_diff = np.mean(residuals)
    p_mean_diff = np.mean(mean_diff/np.exp(y))
    # return mean_diff, p_mean_diff
    return residuals
    
      
