import numpy as np

class KalmanFilter():

    def __init__(self, initial_state, state_covar, noise_var, mean_revert=False):
        self.x = np.asarray(initial_state)
        self.P = np.asarray(state_covar)
        self.Q = np.diag([0, 1e-3])
        self.R = float(noise_var)

        if mean_revert:
            self.F = np.identity(2)
        else:
            self.F = np.identity(2)


    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, stock_return, market_return):
        H = np.array([1, market_return])

        K = (self.P @ H.T) / (H @ self.P @ H.T + self.R)
        self.x = self.x + K * (stock_return - H @ self.x)
        self.P = (np.identity(2) - np.outer(K, H)) @ self.P

        return self.x






    

