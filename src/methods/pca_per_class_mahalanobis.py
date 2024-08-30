from oodeel.methods.base import OODBaseDetector
import numpy as np
from IPython.display import clear_output
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import MinCovDet
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
import cupy as cp

def calculate_distance_for_single_test_example(W_train, test_example, MCD):
    N = W_train.shape[0]
    distances = np.zeros(N)
    for j in range(N):
        # diff = W_train[j, :] - test_example
        distance = mahalanobis(W_train[j, :], test_example, MCD.precision_)
        distances[j] = distance
    return distances

def calculate_mahalanobis_distance_parallel(W_train, W_test, MCD):
    M = W_test.shape[0]
    # Utiliser joblib pour paralléliser le calcul des distances
    results = Parallel(n_jobs=-1)(delayed(calculate_distance_for_single_test_example)(W_train, W_test[i, :], MCD) for i in range(M))
    distance_matrix = np.array(results)
    return distance_matrix


class PCA_Unique_Class_Mahalanobis(OODBaseDetector):
    def __init__(
        self,
        n_components=9

    ):
      super().__init__()
      self.A_train = None
      self.A_in = None
      self.labels_train = None
      self.Scaler = None
      self.n_components=n_components

    def _fit_to_dataset(self, fit_dataset):
      # Calculate the activations_matrix A_train for the training dataset, to calculate the PCs
      training_features = self.feature_extractor.predict(fit_dataset)
      # The activations_matrix A_train
      A_train = training_features[0][0]
      A_train = self.op.convert_to_numpy(A_train)
      if len(A_train.shape) > 2: 
        A_train =A_train[:,:, 0, 0]
      # Standardizing the features
      self.Scaler = StandardScaler()
      A_train_scaled = self.Scaler.fit_transform(A_train)
      # print("after scaling : ", A_train_scaled.shape)
      self.A_in = A_train_scaled
      
      
      # print("the shape of A_train is : ", self.A_in.shape)
      # The training labels
      self.labels_train = training_features[1]["labels"]
      # print("the shape of labels_train is : ", self.labels_train.shape)
      self.classes_ = np.unique(self.labels_train)  # Obtenir les classes uniques
      # print("the unique classes are : ", self.classes_)
      self.PCAs = {}  # Dictionnaire pour stocker PCA par classe
      self.H_Bases = {}  # Dictionnaire pour stocker H_base par classe
      self.W_trains = {} # Dictionnaire pour stocker W_train par classe

      for class_label in self.classes_:

        # Sélectionner les données appartenant à la classe actuelle
        A_train_class = self.A_in[self.labels_train == class_label]
        # Appliquer PCA à A_train_class
        pca = PCA(n_components=self.n_components)
        W_train_class = pca.fit_transform(A_train_class)
        # print("the shape of W_train for class {} is  : {}".format(class_label, W_train_class.shape))
        H_Base_class = pca.components_
        # print("the shape of H_base for class {} is  : {}".format(class_label, H_Base_class.shape))
        # Stocker les résultats
        self.PCAs[class_label] = pca
        self.H_Bases[class_label] = H_Base_class
        self.W_trains[class_label] = W_train_class

    # After fitting PCA for each class, calculate and store the inverse of the covariance matrix for Mahalanobis distance
      self.MCDs = {}
      for class_label, W_train_class in self.W_trains.items():
          # Calculate the empirical covariance for the current class
          cov = MinCovDet().fit(W_train_class)
          self.MCDs[class_label] = cov  # Store the precision (inverse covariance) matrix


      return

    def _score_tensor(self, inputs):

      features, logits = self.feature_extractor.predict_tensor(inputs)
      if len(features[0].shape) > 2:
         features[0] = features[0][:,:, 0, 0]

      A_test = features[0].cpu()
      A_test = self.op.convert_to_numpy(A_test) # la matrice des données de test A_test
      A_test_scaled = self.Scaler.transform(A_test)

      min_distances = np.inf * np.ones(A_test_scaled.shape[0])
        
      for class_label in self.classes_:
          pca = self.PCAs[class_label]
          W_test_class = pca.transform(A_test_scaled)
          W_train_class = self.W_trains[class_label]
          MCD = self.MCDs[class_label]  # Get the precision matrix for the class
          # Calculate Mahalanobis distance using the parallel function
          distance_matrix = cdist(W_test_class, W_train_class, 'mahalanobis', VI=MCD.precision_)
          # For each test sample, find the minimum Mahalanobis distance across classes
          min_distance = np.min(distance_matrix, axis=1)
          min_distances = np.minimum(min_distances, min_distance)
      
      return min_distances

    @property
    def requires_to_fit_dataset(self) -> bool:
        """
        Whether an OOD detector needs a `fit_dataset` argument in the fit function.


        Returns:
            bool: True if `fit_dataset` is required else False.
        """
        return True

    @property
    def requires_internal_features(self) -> bool:
        """
        Whether an OOD detector acts on internal model features.

        Returns:
            bool: True if the detector perform computations on an intermediate layer
            else False.
        """
        return True



pca_per_class_mahalanobis = PCA_Unique_Class_Mahalanobis()

