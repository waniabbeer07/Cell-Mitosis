from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd

class CombinedKDEModel:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.normal_kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        self.abnormal_kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)

    def train(self, normal_data, abnormal_data):
        """Train the model with normal and abnormal data."""
        self.normal_kde.fit(normal_data)
        self.abnormal_kde.fit(abnormal_data)

    def classify(self, input_data):
        """Classify new data as normal or abnormal based on KDE scores."""
        normal_score = self.normal_kde.score_samples(input_data)
        abnormal_score = self.abnormal_kde.score_samples(input_data)
        
        if normal_score > abnormal_score:
            return "Normal"
        else:
            return "Abnormal"
