# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import scipy.stats
import sklearn.naive_bayes as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier


def show_sample(X, index):
    '''Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index
        Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        index (int): index of digit to show
    '''
    characteristics = X[index]
    characteristics = np.reshape(characteristics, [16, 16])
    plt.imshow(characteristics, cmap='gray')
    

def plot_digits_samples(X, y):
    '''Takes a dataset and selects one example from each label and plots it in subplots

    Args:
    X (np.ndarray): Digits data (nsamples x nfeatures)
    y (np.ndarray): Labels for dataset (nsamples)
    '''
    samples_check = np.zeros(10) #array to check if we have seen a number before
    i = 1 #counter for all the numbers (terminates when i=11)
    for k in range(len(y)):
        if (samples_check[int(y[k])] == 0):
            samples_check[int(y[k])] = 1 #we check the number we see
            fig = plt.figure()
            fig.add_subplot(10, 1, i)
            fig.set_figheight(30)
            fig.set_figwidth(30)
            plt.title(y[k])
            show_sample(X, k)
            i = i + 1
        if (i==11) :
            break


def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the mean for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select.

    Returns:
        (float): The mean value of the digits for the specified pixels
    '''
    (p1, p2) = pixel
    values = []
    for k in range(len(y)): 
        if (int(y[k]) == digit) : #we check only the pixels of a specific digit
            characteristics = X[k]
            characteristics = np.reshape(characteristics, [16, 16])
            value = characteristics[p1][p2] #we take the value of the current pixel
            values.append(value)
    return np.mean(values) #return the mean number of the array of all the pixels of a specific digit


def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the variance for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select

    Returns:
        (float): The variance value of the digits for the specified pixels
    '''
    
    #we have the same method as in the computation of the mean number 
    
    (p1, p2) = pixel
    values = []
    for k in range(len(y)):
        if (int(y[k]) == digit) :
            characteristics = X[k]
            characteristics = np.reshape(characteristics, [16, 16])
            value = characteristics[p1][p2]
            values.append(value)
    return np.var(values)


def digit_mean(X, y, digit):
    '''Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    '''
    
    #for every digit we save the mean number of every pixel in an array named mean_digit 
    
    mean_digit = []
    for i in range(16):
        for j in range (16):
            X_pixel = i
            Y_pixel = j
            mean_digit.append(digit_mean_at_pixel(X, y, digit, (X_pixel, Y_pixel)))
    return np.array(mean_digit)


def digit_variance(X, y, digit):
    '''Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
    
    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    '''
    
    #for every digit we save the variance number of every pixel in an array named variance_digit 
    
    variance_digit = []
    for i in range(16):
        for j in range (16):
            X_pixel = i
            Y_pixel = j
            variance_digit.append(digit_variance_at_pixel(X, y, digit, (X_pixel, Y_pixel)))
    return np.array(variance_digit)


def euclidean_distance(s, m):
    '''Calculates the euclidean distance between a sample s and a mean template m

    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)

    Returns:
        (float) The Euclidean distance between s and m
    '''
    #calculation of the euclidean distance
    return (np.linalg.norm(s - m))


def euclidean_distance_classifier(X, X_mean):
    '''Classifiece based on the euclidean distance between samples in X and template vectors in X_mean

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)

    Returns:
        (np.ndarray) predictions (nsamples)
    '''
    y = []
    for i in range(len(X)):
        t = X[i]  #we take every digit so as we can compute the euclidean distance 
        distances = np.zeros(10)
        for i in range (10):
            distances[i] = euclidean_distance(t, X_mean[i]) #we compute the euclidean distance of every digit (0-9)
        y.append(np.argmin(distances)) #we save in our array only the one with the minimum distance
    return np.array(y)


class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self):
        self.X_mean_ = None


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        # Get the sizes of the tables and the number of classes and initialize X_mean_
        [n_samples, n_features] = X.shape
        classes = set(y)
        n_classes = len(classes)
        self.X_mean_ = np.zeros([n_classes, n_features])
        
        # collect the mean values of all features for every class
        for cl in set(classes):
            class_vectors = []
            for sample_index in range(n_samples):
                if y[sample_index] == cl:
                    class_vectors.append(X[sample_index])
            class_vectors = np.asarray(class_vectors)
            for f in range(n_features):
                self.X_mean_[int(cl), f] = np.mean(class_vectors[:,f])
        return self


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        # get the number of classes
        [n_classes, n_features] = self.X_mean_.shape
        # create the list of results
        y = np.zeros(len(X))
        # for every vector v in X we append argmin{d(v, X_mean_[cl]) : for every cl in classes}
        for i in range(len(X)):
            v = X[i, :] 
            distances = np.zeros(n_classes)
            for cl in range (n_classes):
                distances[cl] = euclidean_distance(v, self.X_mean_[cl]) #we compute the euclidean distance of every digit (0-9)
            y[i] = np.argmin(distances) #we save in our array only the one with the minimum distance
        return y

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        predictions = self.predict(X)
        return 1-zero_one_loss(y, predictions)


def evaluate_classifier(clf, X, y, folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y
    
    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (float): The 5-fold classification score (accuracy)
    """
    #we use the function cross_val_score so we have the proper score
    scores = cross_val_score(clf, X, y,
                         cv = KFold(folds),
                         scoring="accuracy")
    return np.mean(scores)

    
def calculate_priors(X, y):
    """Return the a-priori probabilities for every class
    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    classes = set(y)
    priors = np.zeros(len(classes))
    i = 0
    for cl in classes:
        priors[i] = len(y[y == cl])
        i = i + 1
    return priors / len(y)



class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""

    def __init__(self, use_unit_variance=False):
        self.means = None
        self.stds = None
        self.priors = None
        self.use_unit_variance = use_unit_variance


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        Calculates Naive Bayesian Model parameters based on the
        feature values in X for each class.
        We preserve for every class and feature
        -means : (n_classes)x(x_features)
        -stds (standard deviations) : (n_classes)x(x_features)
        and for every class
        -priors : (n_classes)
        fit always returns self.
        """
        # For our NB model to work we make the following assumptions
        # -for every class c p(x|c) is a normal distribution (μ, σ^2) pdf
        # -all classes c are independent
        # -we assume that the samples of X are identically independently distributed
        # -for every class the parameters (μ, σ^2) are fixed
        
        # For the Naive assumption
        # -all the features are independent for every class
        # This means that p(x|class) = Π{p(x[f]|class) : for every feature f}
        
        [n_samples, n_features] = X.shape
        classes = set(y)
        n_classes = len(classes)
        self.means = np.zeros([n_classes, n_features])        
        self.stds = np.ones([n_classes, n_features])
        
        # First we calculate μ for every class and every feature
        # In this case μ is the mean of the vectors of each class
        for cl in range(n_classes):
            class_samples = X[y == cl]
            self.means[cl] = class_samples.mean(axis=0) # we use axis=0 to calculate mean in the vertical dimension
        
        # Then we calculate σ^2 for every class and every feature
        # In this case σ^2 for each class is the Mean squared error
        small_std = 1e-5 # in case variance is zero
        if self.use_unit_variance != True:
            for cl in range(n_classes):
                class_samples = X[y == cl]
                self.stds[cl] = class_samples.std(axis=0) + small_std # we use axis=0 to calculate std in the vertical dimension
        # Finally we calculate the classes' priors
        self.priors = calculate_priors(X, y)        
        return self
    
    
    def predict(self, X):
        """
        Make predictions for X based on the
        self.mean
        self.variances
        self.priors
        """
        n_samples = len(X)
        [n_classes, n_features] = self.means.shape
        predictions = np.zeros(n_samples)
                
        # We will find the class with the highest a-posteriori value
        # for every class i.e. prediction(x) = argmax{N(μ, σ^2)(x)*p(class)}
        # For computation reason we use log function
        # prediction(x) = argmax{N(μ, σ^2)(x)*p(class)} = argmax{Σlog(pdf(x[features]|class))+log(prior(class)}
        for i in range(n_samples):
            a_posteriories = np.zeros(n_classes)
            for cl in range(n_classes):
                vector_logpdf = scipy.stats.norm.logpdf(X[i], loc=self.means[cl], scale=self.stds[cl])
                a_post = np.sum(vector_logpdf) + np.log(self.priors[cl])
                a_posteriories[cl] = a_post
            predictions[i] = np.argmax(a_posteriories)
        return predictions
    

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        predictions = self.predict(X)
        return 1-zero_one_loss(y, predictions)

    
def evaluate_sklearn_nb_classifier(X, y, folds=5):
    """ Create an sklearn naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    nb = sk.GaussianNB()
    nb.fit(X, y)
    score = evaluate_classifier(nb, X, y, folds) * 100
    return score
 
    
def evaluate_custom_nb_classifier(X, y, folds=5):
    """ Create a custom naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    custom = CustomNBClassifier()
    custom.fit(X, y)
    score = evaluate_classifier(custom, X, y, folds) * 100
    return score
 
    
def evaluate_knn_classifier(X, y, folds=5):
    """ Create a knn and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    n1 = KNeighborsClassifier(n_neighbors = 1)
    n1.fit(X, y)
    n3 = KNeighborsClassifier(n_neighbors = 3)
    n3.fit(X, y)
    n5 = KNeighborsClassifier(n_neighbors = 5)
    n5.fit(X, y)
    n7 = KNeighborsClassifier(n_neighbors = 7)
    n7.fit(X, y)
    n9 = KNeighborsClassifier(n_neighbors = 9)
    n9.fit(X, y)
    score1 = evaluate_classifier(n1, X, y, folds) * 100
    score2 = evaluate_classifier(n3, X, y, folds) * 100
    score3 = evaluate_classifier(n5, X, y, folds) * 100
    score4 = evaluate_classifier(n7, X, y, folds) * 100
    score5 = evaluate_classifier(n9, X, y, folds) * 100
    return(score1, score2, score3, score4, score5)
    
    
def evaluate_linear_svm_classifier(X, y, folds=5):
    """ Create an svm with linear kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    cl_linear = SVC(kernel = 'linear')
    cl_linear.fit(X, y)
    score = evaluate_classifier(cl_linear, X, y, folds) * 100
    return score

def evaluate_rbf_svm_classifier(X, y, folds=5):
    """ Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    cl_rbf = SVC(kernel = 'rbf')
    cl_rbf.fit(X, y)
    score = evaluate_classifier(cl_rbf, X, y, folds) * 100
    return score    
 
    
def evaluate_euclidean_classifier(X, y, folds=5):
    """ Create a euclidean classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    euclid = EuclideanDistanceClassifier()
    euclid.fit(X, y)
    score = evaluate_classifier(euclid, X, y, folds) * 100
    return score 

def evaluate_voting_classifier(X, y, folds=5):
    """ Create a voting ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    nb = sk.GaussianNB()
    n3 = KNeighborsClassifier(n_neighbors = 3)
    cl_rbf = SVC(kernel = 'rbf', probability=True)
    i = 0
    methods = ['hard', 'soft']
    score = np.zeros(2)
    for votemethod in methods:
        voting = VotingClassifier(estimators=[('nb', nb), ('n3', n3), ('cl_rbf', cl_rbf)], voting = votemethod)
        voting.fit(X,y)
        score[i] = evaluate_classifier(voting, X, y, folds)
        i = i+1
    return score


def evaluate_bagging_classifier(X, y, folds=5):
    """ Create a bagging ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    bagg = BaggingClassifier(SVC(kernel = 'rbf'), n_estimators = 10)
    bagg.fit(X,y)
    score = evaluate_classifier(bagg, X, y, folds)
    return score


'''
class PytorchNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, *args, **kwargs):
        # WARNING: Make sure predict returns the expected (nsamples) numpy array not a torch tensor.
        # TODO: initialize model, criterion and optimizer
        self.model = ...
        self.criterion = ...
        self.optimizer = ...
        raise NotImplementedError

    def fit(self, X, y):
        # TODO: split X, y in train and validation set and wrap in pytorch dataloaders
        train_loader = ...
        val_loader = ...
        # TODO: Train model
        raise NotImplementedError

    def predict(self, X):
        # TODO: wrap X in a test loader and evaluate
        test_loader = ...
        raise NotImplementedError

    def score(self, X, y):
        # Return accuracy score.
        raise NotImplementedError

        


def evaluate_nn_classifier(X, y, folds=5):
    """ Create a pytorch nn classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError    

 
'''