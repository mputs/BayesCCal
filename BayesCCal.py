#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:45:57 2021

This model estimates the proportions of different classes in a dataset and
calibrates the outcome of the classifier based on that. 

Only binary classificication is implemented at the moment, 

@author: mputs
"""

import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize
from scipy.integrate import quad
from inspect import isclass

class __DensEst__():
    def __init__(self, radius=.1):
        self.radius = radius
    def init(self, scores):
        # initialize with scores
        self.scores = np.sort(scores,axis=0)
        self.area = quad(self.__freqs__,0,1)[0]
    def pdf(self,X):
        return self.__freqs__(X)/self.area
    def __freqs__(self,X):
        if not hasattr(X, "shape"):
            X = np.array(X).reshape(1,)
        r = self.radius
        # calculate values
        length = X.shape[0]
        s = self.scores.flatten();
        ls = s.shape[0]
        s = s.reshape(ls,1)
        x = X.flatten();
        lx = x.shape[0]
        x = x.reshape(1,lx)
        ss = np.repeat(s, lx, axis=1)
        xx = np.repeat(x, ls, axis=0)
        xxl = xx - r
        xxl [xxl<0]=0
        xxh = xx + r
        xxh [xxh>1]=1
        idxl = ((xx - r - ss)**2).argmin(axis=0)
        idxh = ((xx + r - ss)**2).argmin(axis=0)
        size = xxh[idxh,np.arange(lx)]-xxl[idxl,np.arange(lx)].flatten()
        n = ((idxh-idxl)/size)
        return n
    
    
class myEst1():
    def __init__(self, N=1000, radius=.1, eps=1e-20):
        self.N = N
        self.R = radius*N
        self.eps = eps;
    def init(self, scores):
        hist = np.histogram(scores, range=(0,1), bins=self.N)
        cumhist = np.cumsum(hist[0])
        index = np.arange(0,self.N,1)
        indexL = (index - self.R).astype(int)
        indexH = (index + self.R).astype(int)
        indexL[indexL<0]=0
        indexH[indexH>=self.N]=self.N-1
        H = (cumhist[indexH]-cumhist[indexL])/(indexH-indexL)
        H += np.sum(H)*self.eps
        
        self.H = H / np.sum(H)
    def pdf(self, X):
        idx = (X*(self.N-1)).astype(int)
        return self.H[idx]
    
class __HistEst__():
    def __init__(self, bins = 3):
        self.bins = bins
    def init(self,scores):
        self.scores = scores
        # calculate histograms
        hx = np.histogram(scores,bins = np.linspace(0,1,self.bins+1), density=True)[0]
        # make sure the sum to 1
        self.hx = hx/self.bins
    def pdf(self,X):
        idx = np.floor(X*self.bins).astype(int)
        l = idx.shape[0]
        # make sure indices are in the right range
        idx[idx>=self.bins]=self.bins-1
        return self.hx[idx].reshape(1,l)




def checkattr(classifier):
    try:
        assert(hasattr(classifier, "predict_proba"))
        assert(hasattr(classifier, "fit"))
    except:
        return -1
    return 0

def DKL(hx0, hx1):
    epsilon = min(np.min(hx0[hx0>0]), np.min(hx1[hx1>0]))/1000.0
    dkl = np.sum(hx1 * np.log((hx1+epsilon) / (hx0+epsilon)))
    return dkl




class calibrator_binary():
    """
    class calibrator_binary: the binary calibration class. Only a calibration based on histograms is implemented.
    
    Parameters:
    -----------
    classifier: 
        the to be calibrated classifier. This classifier needs to have the methods "predict_proba" and "fit" to be implemented.
    bins: int
        number of bins in the histograms of positive and negative scores
    pisamples:
        number of discrete cells in the discretized distribution of the proportion positives (pi)
        
    Attributes:
    ----------
    classifier:
        the used classifier in the estimates
    hxt, hxf: 
        normalized histograms of positive and negative scores
    pi:
        proportion positives in the last presented dataset
    threshold:
        calibrated threshold for the last presented dataset
    
    
    Methods:
    --------
    fit(X,y):
        Fit the model according to the  given training data
    predict(X[, new_threshold = True]):
        predict class labels for the samples in X.
    
    getProportion(X):
        Determine the proportion positives for samples in X
    determineThreshold(X):
        Determine the calibrated threshold for samples in X
        
    
    Example:
        from sklearn.linear_model import LogisticRegression

        from BayesCCal import calibrator_binary
        import numpy as np
        
        def genData(d_prime, N, ppos):
            X = np.random.normal(0, 1, N)
            y = np.random.rand(N)<=ppos
            X[y] += d_prime
            X = X.reshape(-1,1)
            return X,y
        
        
        X, y = genData(2,400,.5)
        clf = LogisticRegression(random_state=0, fit_intercept=True)
        cal = calibrator_binary(clf)
        cal.fit(X,y)
        Xtest, ytest = genData(2,100,.2)
        print(np.sum(cal.predict(Xtest)))
        print(cal.getProportion(Xtest))

    """
    
    def __init__(self, classifier, bins = 3, radius = .1, pisamples=1001, density="dens"):
        if not checkattr(classifier):   
            self.classifier = classifier
            self.bins = bins;
            self.radius = radius;
            self.pisamples = pisamples
            if density == "hist":
                self.density_t = __HistEst__(bins = bins)
                self.density_f = __HistEst__(bins = bins)
            elif density == "dens":
                self.density_t = __DensEst__(radius = radius)
                self.density_f = __DensEst__(radius = radius)
            elif density == "test":
                self.density_t = myEst1()
                self.density_f = myEst1()
            elif isinstance(density, tuple):
                if len(density) == 2:
                    def __checkattr__(density):
                        try:
                            assert(hasattr(density, "pdf"))
                            assert(hasattr(density, "init"))
                        except:
                            return -1
                        return 0
                    if not __checkattr__(density[0]):
                        self.density_t = density[0]
                    else: 
                        raise Exception("first object in density argument is not an expected object" )
                    if not __checkattr__(density[1]):
                        self.density_f = density[1]
                    else:
                        raise Exception("second object in density argument is not an expected object")
                else:
                    raise Exception("density must be \"hist\", \"est\", or a tuple of objects having an init() method and a pdf() method")
                
            else:
                raise Exception("non valid argument for density: {}".format(density))
            
        else:
            raise Exception("Classifier has not all the needed methods");
    def __getDensities__(self, p):
            l = p.shape[0]
            return self.density_t.pdf(p).reshape(1,l), self.density_f.pdf(p).reshape(1,l)
            
    def __maxLike__(self,p):
        def __f__(pi):
            
            dt, df = self.__getDensities__(p)
            return -np.sum(np.log(dt * pi + df * (1-pi)), axis = 1)
        def __fprime__(pi):
            dt, df = self.__getDensities__(p)            
            num = dt-df
            den = dt * pi + df * (1-pi)
            return -np.sum(num/den, axis = 1)
        result = minimize(__f__,   .5,  method = "L-BFGS-B", bounds = [(0,1)], jac = __fprime__)
        if(result.success==True):
            return result.x[0];
        else:
            print(result)
            return result.x[0];
              
    
    def calcDensities(self, X, y):
        p = self.classifier.predict_proba(X);
        pxt = p[y==True,1]
        pxf = p[y==False,1]        
        self.density_t.init(pxt);
        self.density_f.init(pxf);

        
    def fit(self, X, y, **kwargs):
        """
        Fit the data according to the training data. The method cals the fit 
        method of the classifier and determines the distributions of 
        positive and negative scores.
        
        Parameters
        ----------
        X: same shape as needed for classifier
            Training vector
        y: same shape as needed for classifier
        
        returns
        -------
        self
            
        """
        self.kwargs = kwargs;
        self.classifier.fit(X,y, **kwargs);
        self.calcDensities(X,y)
        self.n = X.shape[0]
        return self
        
    def getProportion(self, X):
        """
        Get proportion positives in dataset
        
        Parameters
        ----------
        X: same shape as needed for classifier
            samples to be classified
        
        Returns
        -------
        Proportion positives in dataset (float)
        """
        proba = self.classifier.predict_proba(X)
        pi = self.__maxLike__(proba[:,1])
        self.pi = pi
        return pi
    
    def determineThreshold(self,X):
        """
        Determine the Threshold for calibrated classification for a dataset
        
        Parameters
        ----------
        X: same shape as needed for classifier
            dataset for which the optimal threshold needs to be determined
        
        Returns
        -------
        Threshold (float)
        """
        proba = self.classifier.predict_proba(X)
        pi = self.__maxLike__(proba[:,1])
        self.pi = pi
        s = np.sort(proba[:,1])
        
        # Find index belonging to threshold
        idx = int((1-pi)*s.shape[0])
        if idx >= s.shape[0]:
            idx-= 1
        self.threshold = s[idx]
        self.__proba__ = proba
        return self.threshold        
    
    def predict(self, X, new_threshold = True):
        """
        Predict class labels of X
        
        Parameters
        ----------
        X: same shape as needed for classifier
            dataset for which the class labels need to be predicted
        new_threshold: Boolean (default = False)
            determines if a new threshold needs to be calculated if already exists. 
            
        Example:
        from sklearn.linear_model import LogisticRegression

        from BayesCCal import calibrator_binary
        import numpy as np
        
        def genData(d_prime, N, ppos):
            X = np.random.normal(0, 1, N)
            y = np.random.rand(N)<=ppos
            X[y] += d_prime
            X = X.reshape(-1,1)
            return X,y
        
        
        X, y = genData(2,400,.5) #training set: 50% positive, 50% negative
        clf = LogisticRegression(random_state=0, fit_intercept=True)
        cal = calibrator_binary(clf)
        cal.fit(X,y)
        Xcal, _ = genData(2,100,.2) #calibration set: 20% positives
        print("Treshold: {}".format(cal.determineThreshold(Xcal)))
        Xtest, ytest = genData(2,10,.2) # small dataset to be tested
        ypred = cal.predict(Xtest)
        """
        if new_threshold | (not hasattr(self, "threshold")):
            self.determineThreshold(X);
            proba = self.__proba__
        else:
            proba = self.classifier.predict_proba(X)
        return proba[:,1]>=self.threshold
    
    def predict_proba(self, X):
        """
        Calculate probabilities class labels of X
        
        Parameters
        ----------
        X: same shape as needed for classifier
            dataset for which the class labels need to be predicted
        new_threshold: Boolean (default = False)
            determines if a new threshold needs to be calculated if already exists. 
            
        Example:
        from sklearn.linear_model import LogisticRegression

        from BayesCCal import calibrator_binary
        import numpy as np
        
        def genData(d_prime, N, ppos):
            X = np.random.normal(0, 1, N)
            y = np.random.rand(N)<=ppos
            X[y] += d_prime
            X = X.reshape(-1,1)
            return X,y
        
        
        X, y = genData(2,400,.5) #training set: 50% positive, 50% negative
        clf = LogisticRegression(random_state=0, fit_intercept=True)
        cal = calibrator_binary(clf)
        cal.fit(X,y)
        Xtest, ytest = genData(2,200,.2) 
        ppred_proba = cal.predict_proba(Xtest)
        """
        proba = self.classifier.predict_proba(X)[:,1]
        pt = self.density_t.pdf(proba)
        pf = self.density_f.pdf(proba)
        self.pi = self.__maxLike__(proba)
        p = self.pi*pt/(self.pi*pt+(1-self.pi)*pf)
        p = p.reshape(p.shape[0],1)
        return np.hstack([1-p,p]);

    def compareDists(self,X):
        """
        Compare distribution of dataset (X) with th training set with respect to the proportion positives
        
        Parameters
        ----------
        X: same shape as needed for classifier
            dataset for which the class labels need to be predicted
        
        Returns
        -------
        Dictionary with:
            cs: cosine similarity
            K-S: Kolmogorov Smirnov statistic.
            D_KL: Symetric Kulback Leibler divergence
            
        Example:
        from sklearn.linear_model import LogisticRegression

        from BayesCCal import calibrator_binary
        import numpy as np
        
        def genData(d_prime, N, ppos):
            X = np.random.normal(0, 1, N)
            y = np.random.rand(N)<=ppos
            X[y] += d_prime
            X = X.reshape(-1,1)
            return X,y
        
        
        X, y = genData(2,400,.5)
        clf = LogisticRegression(random_state=0, fit_intercept=True)
        cal = calibrator_binary(clf, bins=10)
        cal.fit(X,y)
        Xtest, ytest = genData(2,100,.2)
        print(cal.compareDists(Xtest))
        Xtest = Xtest +2
        print(cal.compareDists(Xtest))
        """

        # Calculate histogram for X        
        m = X.shape[0]
        Ng = self.bins;
        p = self.classifier.predict_proba(X);
        pi = self.getProportion(X)
        dx = 1/Ng;
        hx1 = np.histogram(p[:,1],bins = np.linspace(0,1,Ng+1), density=True)[0]
        hx1 *= dx
        
        # calculate the training histogram for proportion pi
        hx0 = pi*self.hxt + (1-pi)*self.hxf
        
        
        ## Kolmogorov Smirnov Statistic
        ks = np.max(np.abs(np.cumsum(hx0)-np.cumsum(hx1)))
        
        ## Symetric Kulback Leibler Divergence
        # calculate epsilon to prevent division by zero
        # we test how many nats would be gained when we would 
        # use the real histogram instead of our model
        # Dkl(h1||h0)
        dkl = (DKL(hx0,hx1) + DKL(hx1,hx0))/2 
        
        ## Chi Squared test
        Hx1 = hx1*m+.0001
        Hx0 = hx0*m+.0001
        chisqr = np.sum(((Hx1-Hx0)**2)/Hx0)
        
        return {"K-S": ks, "D_KL": dkl, "chi2": chisqr, "chi2 sig": chi2.cdf(chisqr, Ng)}
         
        
