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

az = ax*ay
bz = bx*by

ff, lmbd = scipy.stats.boxcox(e)

data = dict(xx=x,yy=y,ee=e, zz=x*y,ff=ff,lmbd=lmbd, ll=l)

fig, axs = plt.subplots(2,2, figsize=(10,10))
axs[0,0].hist(ax, alpha=0.5)
axs[0,0].hist(bx, alpha=0.5)
axs[1,0].hist(ay, alpha=0.5)
axs[1,0].hist(by, alpha=0.5)
axs[0,1].hist(az, alpha=0.5)
axs[0,1].hist(bz, alpha=0.5)
axs[0,0].set_title('x')
axs[1,0].set_title('y')
axs[0,1].set_title('x*y')
fig.savefig("xy-histogram.png")



fig, axs = plt.subplots(figsize=(10,10))
bins=np.linspace(0,12,27)
axs.hist(ae, bins, alpha=0.5, label="a")
axs.hist(be, bins, alpha=0.5, label="b")
axs.set_title('mixture of exponential distributions: λ=0.5 vs λ=2.0')
fig.savefig("e-histogram.png")

fig, axs = plt.subplots(figsize=(10,10))
bins=np.linspace(-5,5,41)
axs.hist(ff[0:size-1], bins, alpha=0.5, label="a")
axs.hist(ff[size:-1], bins, alpha=0.5, label="b")
axs.set_title('mixture of boxcox transformed exponentials')
fig.savefig("f-histogram.png")



fig, axs = plt.subplots(figsize=(10,10)) # figure(figsize=(7,7))
axs.plot(ax,ay)
axs.plot(bx,by)
axs.set_title('x,y predictors together')
fig.savefig("xy-2d.png")

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
    scatter_matrix(traceDF, figsize=(10,10))
    plt.savefig("xy_glm_additive_binom.png")
    traceplot(trace)
    plt.savefig("xy_glm_additive_traces.png")
    predict = 1 / (1 + np.exp (- (
        traceDF["Intercept"].mean()
        + traceDF["xx"].mean() * data["xx"]
        + traceDF["yy"].mean() * data["yy"]
    )))
    pred, axs = plt.subplots(figsize=(10,10))
    axs.plot(partsort(predict))
    axs.plot(data["ll"])
    pred.savefig("xy_glm_additive_prediction.png")

with Model() as simple:
    priors = { "Intercept": Normal.dist(mu=0, sigma=3)
             , "zz": Normal.dist(mu=0, sigma=5)
             }
    glm.GLM.from_formula("ll ~ zz", data, family=glm.families.Binomial(), priors = priors)
    trace = sample(3000, cores=1, init="adapt_diag")
    traceDF = trace_to_dataframe(trace)
    print(traceDF.describe())
    scatter_matrix(traceDF, figsize=(10,10))
    plt.savefig("xy_glm_multiplicative_binom.png")
    traceplot(trace)
    plt.savefig("xy_glm_multiplicative_traces.png")
    predict = 1 / (1 + np.exp (- (
        traceDF["Intercept"].mean()
        + traceDF["zz"].mean() * data["zz"]
    )))
    pred, axs = plt.subplots(figsize=(10,10))
    axs.plot(partsort(predict))
    axs.plot(data["ll"])
    pred.savefig("xy_glm_multiplicative_prediction.png")

with Model() as simple:
    priors = { "Intercept": Normal.dist(mu=0, sigma=3)
             , "ee": Normal.dist(mu=0, sigma=5)
             }
    glm.GLM.from_formula("ll ~ ee", data, family=glm.families.Binomial(), priors = priors)
    trace = sample(3000, cores=1, init="adapt_diag")
    traceDF = trace_to_dataframe(trace)
    print(traceDF.describe())
    scatter_matrix(traceDF, figsize=(10,10))
    plt.savefig("e_glm_simple_binom.png")
    traceplot(trace)
    plt.savefig("e_glm_simple_traces.png")
    predict = 1 / (1 + np.exp (- (
        traceDF["Intercept"].mean()
        + traceDF["ee"].mean() * data["ee"]
    )))
    pred, axs = plt.subplots(figsize=(10,10))
    axs.plot(partsort(predict))
    axs.plot(data["ll"])
    pred.savefig("e_glm_simple_prediction.png")

with Model() as simple:
    priors = { "Intercept": Normal.dist(mu=0, sigma=3)
             , "ff": Normal.dist(mu=0, sigma=5)
             }
    glm.GLM.from_formula("ll ~ ff", data, family=glm.families.Binomial(), priors = priors)
    trace = sample(3000, cores=1, init="adapt_diag")
    traceDF = trace_to_dataframe(trace)
    print(traceDF.describe())
    scatter_matrix(traceDF, figsize=(10,10))
    plt.savefig("e_glm_boxcox_binom.png")
    traceplot(trace)
    plt.savefig("e_glm_boxcox_traces.png")
    predict = 1 / (1 + np.exp (- (
        traceDF["Intercept"].mean()
        + traceDF["ff"].mean() * data["ff"]
    )))
    pred, axs = plt.subplots(figsize=(10,10))
    axs.plot(partsort(predict))
    axs.plot(data["ll"])
    pred.savefig("e_glm_boxcox_prediction.png")
    print(data["lmbd"])

