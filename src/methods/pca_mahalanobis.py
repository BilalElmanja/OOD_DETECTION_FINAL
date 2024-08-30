from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import MinCovDet
import torch
from scipy.spatial.distance import mahalanobis
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
import cupy as cp

# def calculate_distance_for_single_test_example(W_train, test_example, MCD):
#     N = W_train.shape[0]
#     distances = np.zeros(N)
#     for j in range(N):
#         # diff = W_train[j, :] - test_example
#         distance = mahalanobis(W_train[j, :], test_example, MCD.precision_)
#         distances[j] = distance
#     return distances

# def calculate_mahalanobis_distance_parallel(W_train, W_test, MCD):
#     M = W_test.shape[0]
#     # Utiliser joblib pour paralléliser le calcul des distances
#     results = Parallel(n_jobs=-1)(delayed(calculate_distance_for_single_test_example)(W_train, W_test[i, :], MCD) for i in range(M))
#     distance_matrix = np.array(results)
#     return distance_matrix
    
  


class PCA_MAHALANOBIS(OODBaseDetector):
    def __init__(
        self,
        n_components=9

    ):
      super().__init__()

      self.n_components = n_components
      self.W_train = None
      self.H_Base = None
      self.PCA = None
      self.Scaler = None
      self.MCD = None


    def _fit_to_dataset(self, fit_dataset):

      # Calculate the activations_matrix A_train for the training dataset, to calculate the PCs
      training_features = self.feature_extractor.predict(fit_dataset)
      # The activations_matrix A_train
      A_train = training_features[0][0]
      A_train = self.op.convert_to_numpy(A_train)
      if len(A_train.shape) > 2:
         A_train = A_train[:,:, 0, 0]
      # Standardizing the features
      self.Scaler = StandardScaler()
      A_train_scaled = self.Scaler.fit_transform(A_train)
      # print("after scaling : ", A_train_scaled.shape)
      self.A_in = A_train_scaled
      # Appliquer PCA
      pca = PCA(n_components=self.n_components)
      self.W_train = pca.fit_transform(self.A_in)   # La matrice des coefficients W (N , K) de A_train dans la base H_base (K , L)
      self.H_Base = pca.components_ # la matrice de covariance ou notamment la base H_base (K , L)
      self.PCA = pca
      print("the shape of W_train is  : ", self.W_train.shape)
      print("the shape of H_base is  : ", self.H_Base.shape)
      self.MCD = MinCovDet().fit(self.W_train)

      return

    def _score_tensor(self, inputs):

      features, logits = self.feature_extractor.predict_tensor(inputs)
      if len(features[0].shape) > 2:
         features[0] = features[0][:,:, 0, 0]
         
      A_test = features[0].cpu()
      A_test = self.op.convert_to_numpy(A_test) # la matrice des données de test A_test
      A_test_scaled = self.Scaler.transform(A_test)
    #   A_test_scaled = self.Scaler.transform(A_test)
      W_test = self.PCA.transform(A_test_scaled) # la matrice des coefficients de A_test dans la base H avec taille (N_test, K)
      # calculer la distance mahalanobis entre W_test et W_train
    #   distance_matrix = calculate_mahalanobis_distance_parallel(self.W_train, W_test, self.MCD)
      # Calculate the mahalanobis distance between each sample and the centroids
      distance_matrix = cdist(W_test, self.W_train, 'mahalanobis', VI=self.MCD.precision_)
      min_distance = np.min(distance_matrix, axis=1)
      

      return min_distance

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


pca_mahalanobis = PCA_MAHALANOBIS()