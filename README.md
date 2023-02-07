# BayesCCal: Bayesian Calibration of classifiers

This python module calibrates (binary) classifiers using a Bayesian method. First the "proportion positives" is estimated. After that, the probability of an item being positive is estimated. There will be a paper published about this subject. This is now under review.

For now, the module can be installed by:
```bash
pip install git+https://github.com/mputs/BayesCCal.git
```

The module is compatible with classifiers in scikit-learn, as long as predict_proba is implemented.

## a simple example

For the example, we will use the Banknote Authentication dataset. First we will import the data and only select two features:
```python
import pandas as pd
columns = ["var", "skew", "curt", "entr", "class"]
data = pd.read_csv("data_banknote_authentication.txt", names = columns)
data = data[["skew", "curt", "class"]]
```
we create a balanced dataset by selecting the first 200 positive and and the first 200 negative items:

```python
dataPos = data[data["class"]==1]
dataNeg = data[data["class"]==0]
Training = pd.concat([dataPos.iloc[0:200], dataNeg.iloc[0:200]])
X = Training[["skew", "curt"]].to_numpy()
y = Training["class"].to_numpy()
```

We now initialize the model. calibrator_binary accepts any classifier, as long as the methods "predict_proba" and "fit" are implemented. 

```python
from BayesCCal import calibrator_binary
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0)
cclf = calibrator_binary(clf).fit(X,y)
```

Under the hood, the logistic regression model is fitted and the calibrator calculates the distributions of the scores of the positives as well as the negatives. Based on these distributions, the model will correct for bias and calculate the true probability of an item being positive. Even when the dataset is not balanced. To show this, we will create a test set with 20% positives and 80% negatives:

```python
Test = pd.concat([dataPos.iloc[200:].sample(n=100), dataNeg.iloc[200:].sample(n=400)])
Xtest = Test[["skew", "curt"]].to_numpy()
ytest = Test["class"].to_numpy()
```

Now, we predict, and count the number of predicted positives:

```python
print("predicted positives: ", sum(cclf.predict(Xtest)))
```

compared with the logistic regression model, this is a much better result:
```python
print("originally predicted:", sum(clf.predict(Xtest)))

