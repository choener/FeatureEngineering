#!/usr/bin/env python3

import argparse
from pandas.plotting import scatter_matrix
from pymc3 import *
import arviz as az
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pymc3 as pm
import seaborn as sb
import sys
import random
import scipy as scipy


# create a mixture of two distributions in 'X' form. With features along the principle axis, this
# does not allow additive separation.

size = 200
ax = np.random.normal (scale=1, size=size)
ay = (-1) * ax
ae = np.random.exponential (scale=0.5, size=size)
al = np.full(size, 0)

bx = np.random.normal (scale=1, size=size)
by = bx
be = np.random.exponential (scale=2, size=size)
bl = np.full(size, 1)

x = np.append(ax,bx)
y = np.append(ay,by)
e = np.append(ae,be)
l = np.append(al,bl)

ff, lmbd = scipy.stats.boxcox(e)

data = dict(xx=x,yy=y,ee=e, zz=x*y,ff=ff,lmbd=lmbd, ll=l)

fig, axs = plt.subplots(3,2, figsize=(20,20)) # figure(figsize=(7,7))
axs[0,0].hist(x)
axs[1,0].hist(y)
axs[2,0].hist(e)
axs[0,1].hist(data["zz"])
axs[2,1].hist(data["ff"])
#axs[2,1].hist(data["ll"])
axs[0,0].set_title('x')
axs[1,0].set_title('y')
axs[2,0].set_title('e')
axs[0,1].set_title('x*y')
axs[2,1].set_title('ng.log(e)')
#axs[2,1].set_title('classes')
fig.savefig("histogram.png")

fig, axs = plt.subplots(3,3, figsize=(20,20)) # figure(figsize=(7,7))
axs[0,0].plot(ax)
axs[1,0].plot(bx)
axs[2,0].plot(ax)
axs[2,0].plot(bx)
axs[0,1].plot(ay)
axs[1,1].plot(by)
axs[2,1].plot(ay)
axs[2,1].plot(by)
axs[0,2].plot(ae)
axs[1,2].plot(be)
axs[2,2].plot(ae)
axs[2,2].plot(be)
axs[0,0].set_title('x coordinates')
axs[0,1].set_title('y coordinates')
fig.savefig("single-axes.png")


fig, axs = plt.subplots(figsize=(20,20)) # figure(figsize=(7,7))
axs.plot(ax,ay)
axs.plot(bx,by)
axs.set_title('x,y coordinates')
fig.savefig("both-axes.png")

fig, axs = plt.subplots(figsize=(20,20)) # figure(figsize=(7,7))
bins=np.linspace(0,12,27)
axs.hist(ae, bins, alpha=0.5, label="a")
axs.hist(be, bins, alpha=0.5, label="b")
axs.set_title('x,y coordinates')
fig.savefig("both-exponential.png")

def partsort(xs):
    ls = np.sort(xs[0:size-1])
    rs = np.sort(xs[size:2*size-1])
    return(np.append(ls,rs))


# simple GLM

with Model() as simple:
    priors = { "Intercept": Normal.dist(mu=0, sigma=3)
             , "xx": Normal.dist(mu=0, sigma=5)
             , "yy": Normal.dist(mu=0, sigma=5)
             }
    glm.GLM.from_formula("ll ~ xx + yy", data, family=glm.families.Binomial(), priors = priors)
    trace = sample(3000, cores=1, init="adapt_diag")
    traceDF = trace_to_dataframe(trace)
    print(traceDF.describe())
    scatter_matrix(traceDF, figsize=(8,8))
    plt.savefig("glm_additive_binom.png")
    traceplot(trace)
    plt.savefig("glm_additive_traces.png")
    predict = 1 / (1 + np.exp (- (
        traceDF["Intercept"].mean()
        + traceDF["xx"].mean() * data["xx"]
        + traceDF["yy"].mean() * data["yy"]
    )))
    pred, axs = plt.subplots(figsize=(20,20))
    axs.plot(partsort(predict))
    axs.plot(data["ll"])
    pred.savefig("glm_additive_prediction.png")

with Model() as simple:
    priors = { "Intercept": Normal.dist(mu=0, sigma=3)
             , "zz": Normal.dist(mu=0, sigma=5)
             }
    glm.GLM.from_formula("ll ~ zz", data, family=glm.families.Binomial(), priors = priors)
    trace = sample(3000, cores=1, init="adapt_diag")
    traceDF = trace_to_dataframe(trace)
    print(traceDF.describe())
    scatter_matrix(traceDF, figsize=(8,8))
    plt.savefig("glm_multiplicative_binom.png")
    traceplot(trace)
    plt.savefig("glm_multiplicative_traces.png")
    predict = 1 / (1 + np.exp (- (
        traceDF["Intercept"].mean()
        + traceDF["zz"].mean() * data["zz"]
    )))
    pred, axs = plt.subplots(figsize=(20,20))
    axs.plot(partsort(predict))
    axs.plot(data["ll"])
    pred.savefig("glm_multiplicative_prediction.png")

with Model() as simple:
    priors = { "Intercept": Normal.dist(mu=0, sigma=3)
             , "ee": Normal.dist(mu=0, sigma=5)
             }
    glm.GLM.from_formula("ll ~ ee", data, family=glm.families.Binomial(), priors = priors)
    trace = sample(3000, cores=1, init="adapt_diag")
    traceDF = trace_to_dataframe(trace)
    print(traceDF.describe())
    scatter_matrix(traceDF, figsize=(8,8))
    plt.savefig("glm_ee_binom.png")
    traceplot(trace)
    plt.savefig("glm_ee_traces.png")
    predict = 1 / (1 + np.exp (- (
        traceDF["Intercept"].mean()
        + traceDF["ee"].mean() * data["ee"]
    )))
    pred, axs = plt.subplots(figsize=(20,20))
    axs.plot(partsort(predict))
    axs.plot(data["ll"])
    pred.savefig("glm_ee_prediction.png")

with Model() as simple:
    priors = { "Intercept": Normal.dist(mu=0, sigma=3)
             , "ff": Normal.dist(mu=0, sigma=5)
             }
    glm.GLM.from_formula("ll ~ ff", data, family=glm.families.Binomial(), priors = priors)
    trace = sample(3000, cores=1, init="adapt_diag")
    traceDF = trace_to_dataframe(trace)
    print(traceDF.describe())
    scatter_matrix(traceDF, figsize=(8,8))
    plt.savefig("glm_ff_binom.png")
    traceplot(trace)
    plt.savefig("glm_ff_traces.png")
    predict = 1 / (1 + np.exp (- (
        traceDF["Intercept"].mean()
        + traceDF["ff"].mean() * data["ff"]
    )))
    pred, axs = plt.subplots(figsize=(20,20))
    axs.plot(partsort(predict))
    axs.plot(data["ll"])
    pred.savefig("glm_ff_prediction.png")
    print(data["lmbd"])

