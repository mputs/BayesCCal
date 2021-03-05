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
from scipy.stats import beta

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
    
    def __init__(self, classifier, bins = 3, pisamples=1001, density="hist"):
        if not checkattr(classifier):   
            self.classifier = classifier
            self.bins = bins;
            self.pisamples = pisamples
            self.density = density
        else:
            raise Exception("Classifier has not all the needed methods");
    def __getDensities__(self, p):
        if self.density == "hist":
            idx = np.floor(p*self.bins).astype(int)
            l = idx.shape[0]
            # make sure inices are in the right range
            idx[idx>=self.bins]=self.bins-1
            return self.hxt[idx].reshape(1,l), self.hxf[idx].reshape(1,l)
        else:
            l = p.shape[0]
            return beta.pdf(p,self.betat[0], self.betat[1]), beta.pdf(p, self.betaf[0], self.betaf[1]).reshape(1,l)
            
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
        

    def __getdens__(self, X,y):
        
        # define basis for histograms
        Ng = self.bins;
        p = self.classifier.predict_proba(X);
        dx = 1/Ng;
    
        # Separate positives from negatives
        pxt = p[y==True,1]
        pxf = p[y==False,1]
    
        # calculate histograms
        hxt = np.histogram(pxt,bins = np.linspace(0,1,Ng+1), density=True)[0]
        hxf = np.histogram(pxf,bins = np.linspace(0,1,Ng+1), density=True)[0]
    
        # make sure the sum to 1
        hxt *= dx
        hxf *= dx
        return hxt, hxf
    def __estbetas__(self,X,y):
        p = self.classifier.predict_proba(X);
        def estbeta(p):
            mn = np.mean(p)
            vr = np.var(p)
            q = (mn*(1-mn)/vr)-1
            alpha = mn*q
            beta = (1-mn)*q
            return (alpha,beta)
        pxt = p[y==True,1]
        pxf = p[y==False,1]
        return estbeta(pxt), estbeta(pxf)
    
    def calcHistogram(self, X, y, bins = 3):
        self.bins = bins;
        self.hxt, self.hxf = __getdens__(X,y)
        self.betat, self.betaf = __estbetas__(X,y)
        return self.hxt, self.hxf
    
    
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
        self.hxt, self.hxf = self.__getdens__(X,y)
        self.betat, self.betaf = self.__estbetas__(X,y)
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
    
    def predict(self, X, new_threshold = False):
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
         
        
