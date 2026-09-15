import numpy as np
import pandas as pd

class KalmanBetaFilter:
    def __init__(self, x0, P0, Q, R, Phi=None):
        """
        x0: initial state of alpha and beta [alpha, beta]
        P0: initial state covariance 
        Q:  process noise covariance 
        R:  measurement noise variance, scalar
        Phi: state transition matrix
        """

        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = float(R)
        self.Phi = np.eye(2) if Phi is None else np.asarray(Phi, dtype=float)
 
    def predict(self):

        self.x = self.Phi @ self.x
        self.P = self.Phi @ self.P @ self.Phi.T + self.Q
        return self.x, self.P
 
    def update(self, r_i_t, r_m_t):

        H = np.array([1.0, r_m_t])              
        y_pred = H @ self.x                       
        S = H @ self.P @ H.T + self.R             
        K = self.P @ H.T / S                      
        innovation = r_i_t - y_pred
        self.x = self.x + K * innovation
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        return self.x, self.P



    def run_filter(beta_hat, sigma_eps2, excess_stock, excess_market, warmup=60,
                   sigma_u2=1e-5, sigma_z2=1e-4, R=None):
    
       if R is None:
           R = sigma_eps2
    
       Q = np.diag([sigma_u2, sigma_z2])
       P0 = np.diag([sigma_eps2, sigma_eps2])  # generic starting uncertainty
       kf = KalmanBetaFilter(x0=beta_hat, P0=P0, Q=Q, R=R)
    
       idx = excess_stock.index[warmup:]
       records = []
       for t in idx:
           kf.predict()
           x, P = kf.update(excess_stock.loc[t], excess_market.loc[t])
           records.append({"date": t, "alpha": x[0], "beta": x[1],
                            "P_alpha": P[0, 0], "P_beta": P[1, 1]})
    
       return pd.DataFrame(records).set_index("date")

