from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from IPython.display import clear_output
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors



class PCA_KNN(OODBaseDetector):
    def __init__(
        self,
        n_components=9

    ):
      super().__init__()

      self.W_train = None
      self.H_Base = None
      self.PCA = None
      self.Scaler = None
      self.n_components = n_components

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
  
      return

    def _score_tensor(self, inputs):

      features, logits = self.feature_extractor.predict_tensor(inputs)
      if len(features[0].shape) > 2:
         features[0] = features[0][:,:, 0, 0]
      A_test = features[0].cpu()      
      # la matrice des données de test A_test
      A_test = self.op.convert_to_numpy(A_test) 
      A_test_scaled = self.Scaler.transform(A_test)
      # la matrice des coefficients de A_test dans la base H avec taille (N_test, K)
      W_test = self.PCA.transform(A_test_scaled) 
      # Définir le nombre de voisins à considérer
      k = 50
      # Créer et ajuster le modèle kNN
      neigh = NearestNeighbors(n_neighbors=k)
      neigh.fit(self.W_train)
      # Trouver les k plus proches voisins de W_test
      distances, indices = neigh.kneighbors(W_test)
      min_distance = np.min(distances, axis=1)
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


pca_knn = PCA_KNN()