#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:39:25 2020

@author: ilia10000
"""
import numpy as np



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