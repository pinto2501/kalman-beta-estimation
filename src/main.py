import numpy as np
import pandas as pd

class KalmanBetaFilter:
    def __init__(self, x, P, Q, R, F=None):
        """
        x: system state of alpha and beta
        P: system state of covariance 
        Q: process noise covariance 
        R: measurement noise variance, scalar
        F: state transition matrix
        """

        self.x = np.asarray(x, dtype=float)
        self.P = np.asarray(P, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = float(R)
        self.F = np.eye(2) if F is None else np.asarray(F, dtype=float) # Random Walk
 
    def predict(self):

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P
 
    def update(self, r_i_t, r_m_t):

        H = np.array([1.0, r_m_t])  # observation matrix            
        y_pred = H @ self.x                       
        S = H @ self.P @ H.T + self.R             
        K = self.P @ H.T / S                      
        innovation = r_i_t - y_pred
        self.x = self.x + K * innovation
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        return self.x, self.P

    """
    def run_filter(initial_state, R, r_stock, r_market,):

        kf = KalmanBetaFilter(initial_state, )


        for t in r_stock:
    """


        

    

