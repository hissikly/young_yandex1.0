import numpy as np
"""
Credits: the original code belongs to Stanford CS231n course assignment1. Source link: http://cs231n.github.io/assignments2019/assignment1/
"""

class KNearestNeighbor:
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################                                                     #
                # Вычислите расстояние l2 между i-й тестовой точкой и j-й тренировочной 
                # точкой и сохраните результат в формате dists[i, j]. Вы должны
                # не используйте цикл по измерению и не используйте np.linalg.norm().          #
                #####################################################################
                # *****НАЧАЛО ВАШЕГО КОДА (НЕ УДАЛЯЙТЕ/НЕ ИЗМЕНЯЙТЕ ЭТУ СТРОКУ)*****

                dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))

                # *****КОНЕЦ ВАШЕГО КОДА (НЕ УДАЛЯЙТЕ/НЕ ИЗМЕНЯЙТЕ ЭТУ СТРОКУ)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Вычислите расстояние l2 между i-й контрольной точкой и всеми обучающими точками #
            # и сохраните результат в формате dists[i, :].
            # Не используйте np.linalg.norm().
            #######################################################################
            # *****НАЧАЛО ВАШЕГО КОДА (НЕ УДАЛЯЙТЕ/НЕ ИЗМЕНЯЙТЕ ЭТУ СТРОКУ)*****
            dists[i, :] = np.sqrt(np.sum((X[i] - self.X_train) ** 2, axis=1))

            # *****КОНЕЦ ВАШЕГО КОДА (НЕ УДАЛЯЙТЕ/НЕ ИЗМЕНЯЙТЕ ЭТУ СТРОКУ)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Вычислите расстояние l2 между всеми тестовыми точками и всеми обучающими #
        # точками без использования каких-либо явных циклов и сохраните результат в #
        # dists.                                                                #
        # Вам следует реализовать эту функцию, используя только базовые операции с массивами.; #
        # в частности, вам не следует использовать функции из scipy, #
        # а также использовать np.linalg.norm().                                             #
        # ПОДСКАЗКА: Попробуйте сформулировать расстояние l2, используя матричное умножение #
        # и две широковещательные суммы.
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Квадраты норм тестовых точек (X^2)
        X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1) # (num_test, 1)
        # Квадраты норм обучающих точек (Y^2)
        X_train_sq = np.sum(self.X_train ** 2, axis=1).reshape(1, -1) # (1, num_train)

        cross_term = 2 * np.dot(X, self.X_train.T) # (num_test, num_train)
        dists = np.sqrt(X_sq + X_train_sq - cross_term) # (num_test, num_train)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """  
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: числовой массив формы (num_test,), содержащий предсказанные метки для
        тестовых данных, где y[i] - предсказанная метка для тестовой точки X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # Список длиной k, содержащий метки k ближайших соседей к
            # i-й контрольной точке.
            #########################################################################
            # Используйте матрицу расстояний, чтобы найти k ближайших соседей i-й #
            # точки тестирования, и используйте self.y_train, чтобы найти метки этих #
            # соседей. Сохраните эти метки в closest_y.                           #
            # Подсказка: Найдите функцию numpy.argsort.                             #
            #########################################################################
            # *****НАЧАЛО ВАШЕГО КОДА (НЕ УДАЛЯЙТЕ/НЕИЗМЕНЯЙТЕ ЭТУ СТРОКУ)*****

            # Получаем индексы k ближайших соседей
            clsosest_idxs = np.argsort(dists[i])[:k]
            # извлекаем метки этих соседей
            closest_y = self.y_train[clsosest_idxs]

            # *****КОНЕЦ ВАШЕГО КОДА (НЕ УДАЛЯЙТЕ/НЕ ИЗМЕНЯЙТЕ ЭТУ СТРОКУ)*****
            #########################################################################
            # Теперь, когда вы нашли ярлыки k ближайших соседей, вы можете    #
            # нужно найти наиболее распространенную метку в списке самых близких к вам меток.   #
            # Сохраните эту метку в y_pred[i]. Разорвите связи, выбрав метку меньшего размера #

            #########################################################################
            # *****НАЧАЛО ВАШЕГО КОДА (НЕ УДАЛЯЙТЕ/НЕ ИЗМЕНЯЙТЕ ЭТУ СТРОКУ)*****

            y_pred[i] = np.argmax(np.bincount(closest_y))

            # *****КОНЕЦ ВАШЕГО КОДА (НЕ УДАЛЯЙТЕ/НЕ ИЗМЕНЯЙТЕ ЭТУ СТРОКУ)*****

        return y_pred
