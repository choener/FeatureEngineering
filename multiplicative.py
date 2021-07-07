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


# create a mixture of two distributions in 'X' form. With features along the principle axis, this
# does not allow additive separation.

size = 200
ax = np.random.normal (scale=1, size=size)
ay = (-1) * ax
al = np.full(size, 0)

bx = np.random.normal (scale=1, size=size)
by = bx
bl = np.full(size, 1)

x = np.append(ax,bx)
y = np.append(ay,by)
l = np.append(al,bl)

data = dict(xx=x,yy=y, zz=x*y, ll=l)

fig, axs = plt.subplots(3,2, figsize=(20,20)) # figure(figsize=(7,7))
axs[0,0].plot(ax)
axs[1,0].plot(bx)
axs[2,0].plot(ax)
axs[2,0].plot(bx)
axs[0,1].plot(ay)
axs[1,1].plot(by)
axs[2,1].plot(ay)
axs[2,1].plot(by)
axs[0,0].set_title('x coordinates')
axs[0,1].set_title('y coordinates')
fig.savefig("single-axes.png")


fig, axs = plt.subplots(figsize=(20,20)) # figure(figsize=(7,7))
axs.plot(ax,ay)
axs.plot(bx,by)
axs.set_title('x,y coordinates')
fig.savefig("both-axes.png")


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
    axs.plot(predict)
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
    axs.plot(predict)
    axs.plot(data["ll"])
    pred.savefig("glm_multiplicative_prediction.png")

