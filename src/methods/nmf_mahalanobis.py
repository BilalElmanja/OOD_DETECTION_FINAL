
from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import minimize
from oodeel.methods.base import OODBaseDetector
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import MinCovDet
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


# def reconstruction_loss(W_flat, A_test, H_base):
#     """Calculer la perte de reconstruction ||A_test - W_test * H_base||_2."""
#     W_test = W_flat.reshape(A_test.shape[0], -1)
#     reconstruction = np.dot(W_test, H_base)
#     return np.linalg.norm(A_test - reconstruction)


class NMF_MAHALANOBIS(OODBaseDetector):
    def __init__(
        self,
        n_components=9
    ):
      super().__init__()
      self.n_components=n_components
      self.A_train = None
      self.W_train = None
      self.H_Base = None
      self.NMF = None
      self.Scaler = None
      self.MCD = None

    def _fit_to_dataset(self, fit_dataset):

      # Calculate the activations_matrix A_train for the training dataset, to calculate the PCs
      training_features = self.feature_extractor.predict(fit_dataset)
      # The activations_matrix A_train
      A_train = training_features[0][0]
      if len(A_train.shape) > 2:
         A_train = A_train[:,:, 0, 0]

      A_train = self.op.convert_to_numpy(A_train)

      self.NMF = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=400)
      self.W_train = self.NMF.fit_transform(A_train)  # project des données entrainement sur nouvelle base H_base
      self.H_Base = self.NMF.components_  # La matrice des composantes (ou la base)
      print("the shape of H_base is : ", self.H_Base.shape)
      print("the shape of W_train is  : ", self.W_train.shape)
      self.MCD = MinCovDet().fit(self.W_train)
  
      return

    def _score_tensor(self, inputs):

      features, logits = self.feature_extractor.predict_tensor(inputs)
      if len(features[0].shape) > 2:
         features[0] = features[0][:,:, 0, 0]

      A_test = features[0].cpu()
      A_test = self.op.convert_to_numpy(A_test) # la matrice des données de test A_test
    #   A_test = self.Scaler.transform(A_test)
    #   A_test = A_test - np.min(A_test) + 1e-5
      # Initialisation de W_test comme une matrice aplatie (pour l'optimisation)
    #   initial_W_test_flat = np.random.randn(A_test.shape[0] * self.W_train.shape[1])
      # Minimiser la perte de reconstruction
    #   result = minimize(reconstruction_loss, initial_W_test_flat, args=(A_test, self.H_Base), method='L-BFGS-B')
      # Remodeler W_test dans sa forme originale (M, K)
    #   W_test_optimized = result.x.reshape(A_test.shape[0], self.W_train.shape[1])
      W_test = self.NMF.transform(A_test)
    #   print("w_test shape is : ", W_test.shape)
      # calculer la distance mahalanobis entre W_test et W_train
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

nmf_mahalanobis = NMF_MAHALANOBIS()