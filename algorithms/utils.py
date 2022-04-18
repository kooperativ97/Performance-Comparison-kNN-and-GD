'''
K-Nearest-Neighbour implementation
'''
class PerformanceMetrics:

    @staticmethod
    def mse(y_true, y_pred):
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def mae(y_true, y_pred):
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def r2(y_true, y_pred):
        from sklearn.metrics import r2_score
        return r2_score(y_true, y_pred)