import numpy as np
from sklearn.utils import check_X_y, check_array
from scipy.stats import multivariate_normal
import loader as loader
from sklearn.covariance import EmpiricalCovariance

class GaussianClassifier(object):
    def __init__(self, c=1, cov_algo="numpy"):
        super(GaussianClassifier, self).__init__()
        self.c=c
        self.cov_algo = cov_algo

    def _examples_for_class(self, klass, X_train, Y_train):
        examples = []
        for i, example in enumerate(X_train):
            if Y_train[i]==klass:
                examples.append(example)
        
        examples = np.matrix(examples)
        return examples

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        print("c=%s, cov_algo=%s"%(self.c,self.cov_algo))
        
        classes=np.unique(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.class_prior_ = np.zeros(n_classes)
        self.class_count_ = np.zeros(n_classes)
        unique_y = np.unique(y)
         
        for y_i in unique_y:
            i = classes.searchsorted(y_i)
            X_i = X[y == y_i, :]
            sw_i = None
            N_i = X_i.shape[0]
            
            self.class_count_[i] += N_i
        
        self.class_prior_[:] = self.class_count_ / np.sum(self.class_count_)
        self.priors = self.class_prior_
        
        self.posteriors=[]

        for klass in self.classes_:
            examples = self._examples_for_class(klass, X, y)
            mean = np.array(examples.mean(0))[0]
            cov = self._cov(examples)
            cov_smoothed = cov + (self.c * np.eye(mean.shape[0]))
            p_x = multivariate_normal(mean=mean, cov=cov_smoothed)
            self.posteriors.append(p_x)
        return self
    
    def predict(self, X):
        Y = []
        for x in X:
            bayes_probs = []
            for klass in self.classes_:
                prob = [klass, np.log(self.priors[klass]) + self.posteriors[klass].logpdf(x)]
                bayes_probs.append(prob)
            prediction = max(bayes_probs, key= lambda a: a[1])
            Y.append(prediction[0])
        return Y
        
    def _cov(self, examples):
        if self.cov_algo =="numpy":
            return np.cov(examples, rowvar=0)
        elif self.cov_algo == "EmpiricalCovariance":
            return EmpiricalCovariance().fit(examples).covariance_
        else:
            return None
            
    
    def get_params(self, deep=True):
        params = {'c': 1.0, 'cov_algo': 'numpy'}
        return params
    
    def set_params(self, **params):
        for key, value in params.iteritems():
            setattr(self, key, value)
        
if __name__ == "__main__":
    import pprint 
    from sklearn import datasets
    from sklearn.cross_validation import train_test_split
    pp = pprint.PrettyPrinter(depth=6)
    
    digits = datasets.load_digits()
    X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.4, random_state=4)
    X_train.shape
    
    clf =  GaussianClassifier(c=90000)
    clf.fit(X_train,Y_train)
    Y = clf.predict(X_test)

    errors = (Y_test != Y).sum()
    total = X_test.shape[0]
    print("Success rate:\t %d/%d = %f" % ((total-errors,total,((total-errors)/float(total)))))
    print("Error rate:\t %d/%d = %f" % ((errors,total,(errors/float(total)))))