import sys
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib
from math import pi
from os import chdir
import pandas as pd
from BayesCCal import calibrator_binary
from sklearn.metrics import confusion_matrix, precision_score, recall_score

PPOS = [.9, .8, .5, .2, .1];

x = np.arange(-4,4,.1);
def cumnorm(x):
    return .5+(erf(x/np.sqrt(2))/2)
def norm(x):
    return np.exp(-(x**2)/2)/np.sqrt(2*pi)
t = x
trueneg = cumnorm(t+1)
truepos = 1-cumnorm(t-1)


fig = plt.figure(figsize=(30, 10))
fig.add_gridspec(3, len(PPOS), hspace=0, wspace=0)
plt.rcParams.update({'font.size': 30})
ax = fig.subplots(3, len(PPOS), sharex='col') ;
fig.subplots_adjust(hspace=0)


## Figure 1
for idx, ppos in enumerate(PPOS):
    acc = ((1-ppos)*trueneg+ppos*truepos)
    bias =  (1-ppos)*(1-trueneg) - ppos*(1-truepos)
    minbias = x[np.argmin(bias**2)]
    yneg  = (1-ppos)*norm(x+1);
    ypos  = ppos*norm(x-1);
    mx = np.max(np.concatenate([yneg,ypos]))

    for i in range(3):
        for axis in ['top','bottom','left','right']:
            ax[i][idx].spines[axis].set_linewidth(4)
    
    
    #ax[0][idx].set_title("$c_1 = {:.2f}$\n$c_2 = {:.2f}$".format(ppos,1-ppos), fontsize = 25)
    ax[0][idx].text(-3.5,0.50, "proportion$(c_1) = {:.1f}$".format(ppos),c="green", fontsize=25)
    ax[0][idx].text(-3.5,0.43, "proportion$(c_2) = {:.1f}$".format(1-ppos), c="red", fontsize=25)
    
    ax[0][idx].plot(x, yneg, "red", linewidth=4);
    ax[0][idx].plot(x, ypos, "green", linewidth=4);
    ax[0][idx].set_ylim(-.1,.4)
    ax[0][idx].vlines([x[np.argmax(acc)]], ymin=-.4, ymax=.4, color="grey", clip_on=False,linewidth=2)
    ax[0][idx].set_ylabel("probability", fontsize=30)


    ax[1][idx].plot(t,acc, "blue", linewidth=4)
    ax[1][idx].set_ylim(-.1,1.1)
    ax[1][idx].vlines([x[np.argmax(acc)]], ymin=-.1, ymax=1.2, color="grey", clip_on = False, linewidth=2)
    ax[1][idx].set_ylabel("accuracy", fontsize=30)

    
    ax[2][idx].plot(t, bias, "maroon", linewidth=4)
    ax[2][idx].set_ylim(-1,1.1)
    ax[2][idx].hlines([0], xmin = -4, xmax=4, color="lightgrey", linewidth=2)
    ax[2][idx].vlines([x[np.argmax(acc)]], ymin=-1, ymax=1.2, color="grey", clip_on = False, linewidth=2)
    ax[2][idx].vlines([minbias], ymin = -1, ymax = 1, color = "orange", linewidth=2)
    ax[2][idx].set_ylabel("bias", fontsize=30)
    ax[2][idx].set_xlabel("$x$")
   

for x in ax.flatten():
        x.label_outer()
fig.savefig("Figure1.eps")
print("Figure 1 is generated");


from BayesCCal import calibrator_binary
import numpy as np
import pandas as pd
def genData(d_prime, N, ppos):
    X = np.random.normal(0, 1, N)
    y = np.random.rand(N)<=ppos
    X[y] += d_prime
    X = X.reshape(-1,1)
    u = X.reshape(-1,1)
    return X,y
    
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
import pandas as pd
d_prime = 2
NG = 3
Ntrain = 500
Ntest = 500
columns = ["p_train", "p_test", "S", "S_t", "S_p", "S_ml", "FPR", "FNR"]
df = pd.DataFrame(columns = columns)
for k in range(100):
    if (k%10==0):
        print(k)
    for p_train in np.linspace(0.25,0.75,3):
        X, y = genData(d_prime,Ntrain,p_train)
        clf = LogisticRegression(random_state=0).fit(X, y)
        cclf = calibrator_binary(clf).fit(X,y)

        for p_test in np.linspace(0.1,0.9,9):
            X_t, y_t = genData(d_prime,Ntest,p_test)
            pred = clf.predict(X_t)
            pred_proba = clf.predict_proba(X_t)
            cpred = cclf.predict(X_t)
            cpred_proba = cclf.predict_proba(X_t)
            tn, fp, fn, tp = confusion_matrix(y_t, cpred).ravel()
            fprate = fp/(fp+tn)
            fnrate = fn/(fn+tp)
            S = np.sum(y_t)
            S_t = np.mean(pred)
            S_p = np.mean(pred_proba[:,1])
            S_ML = np.mean(cpred_proba[:,1])
            #Slogit_ml = maxLikeScore(getScore(clf, X_t), hlxt, hlxf)
            df = df.append(pd.DataFrame([[p_train, p_test, S, S_t, S_p, S_ML, fprate, fnrate]], columns = columns))

        
def conf(x):
    return np.std(x)*2

plt.rcParams.update({
    "text.usetex": False,
    "font.size": "12"})
print(df.columns)

def plotit(S_t, ax, legend = True):
    S = df.groupby(["p_train", "p_test"]).agg(["mean", conf])[S_t];
    pd.pivot_table(S, values =["mean", "conf"], index = "p_test", columns = "p_train").plot(y = "mean", yerr="conf", alpha = .6, ax=ax)
    ax.plot([0,1],[0,1], color = "k", alpha = .3)
    ax.grid(True)
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_xlabel("true proportion positives")
    ax.set_ylabel("predicted proportion positives")
    if legend:
        ax.legend()
    else:
        ax.get_legend().remove()
    ax.set(aspect=1)

f, sax = plt.subplots(1,1,figsize=(8,4))
plotit("S_t", sax)
plt.savefig("Figure2.eps")
print("Figure 2 is generated")

f, axs = plt.subplots(1,2,figsize=(8,4))
axs[0].text(0,-.2, "(a)")
axs[1].text(0,-.2, "(b)")


print(f)
plotit("S_t", axs[0])
plotit("S_ml", axs[1])
axs[1].set_ylabel("")
plt.savefig("Figure3.eps")
print("Figure 3 is generated");

columns = ["var", "skew", "curt", "entr", "class"]
data = pd.read_csv("data_banknote_authentication.txt", names = columns)
data = data[["skew", "curt", "class"]]

data = data.iloc[np.random.permutation(len(data))]
print("total size: {}".format(data["class"].count()))
dataPos = data[data["class"]==1]
dataNeg = data[data["class"]==0]
Training = pd.concat([dataPos.iloc[0:200], dataNeg.iloc[0:200]])
print("positive: {}".format(sum(Training["class"]==1)))
print("negative: {}".format(sum(Training["class"]==0)))
test = {"pos": dataPos.iloc[200:], "neg": dataNeg[200:]}
print("size testset: {}".format(test["pos"]["class"].count()+test["neg"]["class"].count()))
def df_to_Xy(df, target="class"):
    X = df[set(df.columns).difference({target})].to_numpy()
    y = df[target].to_numpy()
    return X,y
def genBanknotes(test,N,ppos, target="class"):
    npos = round(ppos*N)
    nneg = N-npos
    df = pd.concat([
        test["pos"].sample(n = npos),
        test["neg"].sample(n = nneg)])
    return df_to_Xy(df)

import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
import pandas as pd
d_prime = 2
NG = 3
Ntrain = 100
Ntest = 300
columns = ["p_train", "p_test", "S", "S_t", "S_p", "S_ml"]
df = pd.DataFrame(columns = columns)
X, y = df_to_Xy(Training)
clf = LogisticRegression(random_state=0).fit(X, y)
cclf = calibrator_binary(clf).fit(X,y)

for k in range(100):
    if (k%10==0):
        print(k)
        for p_test in np.linspace(0.1,0.9,9):
            X_t, y_t = genBanknotes(test,Ntest,p_test)
            pred = clf.predict(X_t)
            pred_proba = clf.predict_proba(X_t)
            cpred = cclf.predict(X_t)
            cpred_proba = cclf.predict_proba(X_t)
            S = np.sum(y_t)
            S_t = np.mean(pred)
            S_p = np.mean(pred_proba[:,1])
            S_ML = np.mean(cpred_proba[:,1])
            #Slogit_ml = maxLikeScore(getScore(clf, X_t), hlxt, hlxf)
            df = df.append(pd.DataFrame([[p_train, p_test, S, S_t, S_p, S_ML]], columns = columns))

        
f, axs = plt.subplots(1,2,figsize=(8,4))
axs[0].text(0,-.2, "(a)")
axs[1].text(0,-.2, "(b)")


print(f)
plotit("S_t", axs[0], legend = False)
plotit("S_ml", axs[1], legend = False)
axs[1].set_ylabel("")

plt.savefig("Figure4.eps")
print("Figure 4 is generated");

columns = [x.strip() for x in "class, lepton 1 pT, lepton 1 eta, lepton 1 phi, lepton 2 pT, lepton 2 eta, lepton 2 phi, missing energy magnitude, missing energy phi, MET_rel, axial MET, M_R, M_TR_2, R, MT2, S_R, M_Delta_R, dPhi_r_b, cos(theta_r1)".split(",")]
data = pd.read_csv("SUSY.csv", names = columns)
data = data.astype({"class": int}) 

data = data.iloc[np.random.permutation(len(data))]
print("total size: {}".format(data["class"].count()))
dataPos = data[data["class"]==1]
dataNeg = data[data["class"]==0]
Train = {
    "pos": dataPos.iloc[0:1000000],
    "neg": dataNeg.iloc[0:1000000]
}
Test = {
    "pos": dataPos.iloc[1000000:],
    "neg": dataNeg.iloc[1000000:]
}

def df_to_Xy(df, target="class"):
    X = df[set(df.columns).difference({target})].to_numpy()
    y = df[target].to_numpy()
    return X,y
def genSUSY(data,N,ppos, target="class"):
    npos = round(ppos*N)
    nneg = N-npos
    df = pd.concat([
        data["pos"].sample(n = npos),
        data["neg"].sample(n = nneg)])
    return df_to_Xy(df)

import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
import pandas as pd
Ntrain = 500
Ntest = 500
columns = ["p_train", "p_test", "S", "S_t", "S_p", "S_ml"]
df = pd.DataFrame(columns = columns)
for k in range(100):
    if (k%10==0):
        print(k)
    for p_train in np.linspace(0.25,0.75,3):
        X, y = genSUSY(Train,Ntrain,p_train)
        clf = LogisticRegression(random_state=0).fit(X, y)
        cclf = calibrator_binary(clf).fit(X,y)
        
        for p_test in np.linspace(0.1,0.9,9):
            X_t, y_t = genSUSY(Test,Ntest,p_test)
            pred = clf.predict(X_t)
            pred_proba = clf.predict_proba(X_t)
            cpred = cclf.predict(X_t)
            cpred_proba = cclf.predict_proba(X_t)
            S = np.sum(y_t)
            S_t = np.mean(pred)
            S_p = np.mean(pred_proba[:,1])
            S_ML = np.mean(cpred_proba[:,1])
            #Slogit_ml = maxLikeScore(getScore(clf, X_t), hlxt, hlxf)
            df = df.append(pd.DataFrame([[p_train, p_test, S, S_t, S_p, S_ML]], columns = columns))

f, axs = plt.subplots(1,2,figsize=(8,4))
axs[0].text(0,-.2, "(a)")
axs[1].text(0,-.2, "(b)")


print(f)
plotit("S_t", axs[0])
plotit("S_ml", axs[1])
axs[1].set_ylabel("")

plt.savefig("Figure5.eps")
print("Figure 5 is generated");

import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
Ntrain = 500
Ntest = 500
columns = ["p_train", "p_test", "S", "S_t", "S_p", "S_ml"]
df = pd.DataFrame(columns = columns)
for k in range(10):
    if (k%10==0):
        print(k)
    for p_train in np.linspace(0.25,0.75,3):
        X, y = genSUSY(Train,Ntrain,p_train)
        clf = make_pipeline(StandardScaler(), SVC(kernel='linear', gamma='auto', probability=True))
        cclf = calibrator_binary(clf).fit(X,y)
        
        for p_test in np.linspace(0.1,0.9,9):
            X_t, y_t = genSUSY(Test,Ntest,p_test)
            pred = clf.predict(X_t)
            pred_proba = clf.predict_proba(X_t)
            cpred = cclf.predict(X_t)
            cpred_proba = cclf.predict_proba(X_t)
            S = np.sum(y_t)
            S_t = np.mean(pred)
            S_p = np.mean(pred_proba[:,1])
            S_ML = np.mean(cpred_proba[:,1])
            #Slogit_ml = maxLikeScore(getScore(clf, X_t), hlxt, hlxf)
            df = df.append(pd.DataFrame([[p_train, p_test, S, S_t, S_p, S_ML]], columns = columns))

f, axs = plt.subplots(1,2,figsize=(8,4))
axs[0].text(0,-.2, "(a)")
axs[1].text(0,-.2, "(b)")


print(f)
plotit("S_p", axs[0])
plotit("S_ml", axs[1])
axs[1].set_ylabel("")

plt.savefig("Figure6.eps")
print("Figure6 is generated")
