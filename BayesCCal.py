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

def checkattr(classifier):
    try:
        assert(hasattr(classifier, "predict_proba"))
        assert(hasattr(classifier, "fit"))
    except:
        return -1
    return 0

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
    
    def __init__(self, classifier, bins = 3, pisamples=1001):
        if not checkattr(classifier):   
            self.classifier = classifier
            self.bins = bins;
            self.pisamples = pisamples
        else:
            raise Exception("Classifier has not all the needed methods");
    def __maxLike__(self,p):
        # prepare grid
        pi = np.linspace(0,1,self.pisamples).reshape(self.pisamples,1)
        # convert probabilities to grid cells
        idx = np.floor(p*self.bins).astype(int)
        l = idx.shape[0]
        
        # make sure inices are in the right range
        idx[idx>=self.bins]=self.bins-1
        
        # define log likelihood function
        logL = np.sum(np.log(self.hxt[idx].reshape(1,l) * pi + 
                             self.hxf[idx].reshape(1,l) * (1-pi)), axis = 1)
        
        # get maximum likelihood
        maxL = pi[np.argmax(logL)][0]
        return maxL
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
            D_KL: Kulback Leibler divergence
            
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
        Ng = self.bins;
        p = self.classifier.predict_proba(X);
        pi = self.getProportion(X)
        dx = 1/Ng;
        hx1 = np.histogram(p[:,1],bins = np.linspace(0,1,Ng+1), density=True)[0]
        hx1 *= dx
        
        # calculate the training histogram for proportion pi
        hx0 = pi*self.hxt + (1-pi)*self.hxf
        
        ## cosine similarity
        cs = np.sum(hx0*hx1)/np.sqrt(np.sum(hx0*hx0)*np.sum(hx1*hx1))
        
        ## Kolmogorov Smirnov Statistic
        ks = np.max(np.abs(np.cumsum(hx0)-np.cumsum(hx1)))
        
        ## Kulback Leibler Divergence
        # calculate epsilon to prevent division by zero
        epsilon = min(np.min(hx0[hx0>0]), np.min(hx1[hx1>0]))/1000.0
        dkl = np.sum(hx0 * np.log((hx0+epsilon) / (hx1+epsilon)))
        
        return {"cs": cs, "K-S": ks, "D_KL": dkl}
         
        
