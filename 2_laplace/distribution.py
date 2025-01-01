import numpy as np

class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        # Do not change the class outside of this block
        # Your code here
        mad = np.median(np.abs(x - np.median(x), axis=1))
        return mad 
        ####
    

    def __init__(self, features, is_fix_b=False):
        '''
        Args:
        объект: массив фигур numpy (n_objects, n_features). Каждый столбец представляет все доступные значения для выбранного объекта.        '''
        ####
        # Do not change the class outside of this block
        n = features.shape[0]

        if len(features.shape) <= 1:
            self.loc = np.median(features)
            self.scale = 1/n * np.sum((np.abs(features - self.loc)))
            if is_fix_b:
                self.scale = self.scale * n / (n - 2)
        else:
            self.loc = np.median(features, axis=0)
            self.scale = 1/n * np.sum((np.abs(features - self.loc)), axis=0)
            if is_fix_b:
                self.scale = self.scale * n / (n - 2)
        ####


    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        value = np.log(1/(2 * self.scale) * np.exp(- np.abs(values - self.loc) / self.scale))
        return value
        ####
        
    
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))
