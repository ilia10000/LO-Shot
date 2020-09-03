#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:39:25 2020

@author: ilia10000
"""
import numpy as np

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

class SoftKNN:
    def __init__(self,k=None):
        self.x=[]
        self.y=[]
        self.k=k
    def fit(self,x,y):
        self.x=x
        self.y=y
    def calc_dists(self, point):
        dists=[]
        for prototype in self.x:
            dist = np.linalg.norm(point-prototype)
            dists.append(dist)
        return dists
    def calc_lab(self, dists):
        
        label=np.zeros_like(self.y[0])
        tups = zip(self.y, dists)
        if self.k is None:
            for prototype, dist in tups:
                label+=prototype/dist
        else:
            tups=list(tups)
            res = sorted(tups, key = lambda x: x[1])[:self.k]
            for prototype, dist in res:
                label+=prototype/dist
        return label
            
    def predict(self, points):
        preds=[]
        for point in points:
            dists = self.calc_dists(point)
            label = self.calc_lab(dists)
            pred = np.argmax(label)
            preds.append(pred)
        return np.array(preds)
    def probabilities(self,points):
        preds=[]
        for point in points:
            dists = self.calc_dists(point)
            label = self.calc_lab(dists)
            preds.append(label)
        return np.array(preds)