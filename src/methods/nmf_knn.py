
from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF
from scipy.optimize import minimize


# def reconstruction_loss(W_flat, A_test, H_base):
#     """Calculer la perte de reconstruction ||A_test - W_test * H_base||_2."""
#     W_test = W_flat.reshape(A_test.shape[0], -1)
#     reconstruction = np.dot(W_test, H_base)
#     return np.linalg.norm(A_test - reconstruction)


class NMF_KNN(OODBaseDetector):
    def __init__(
        self,
        n_components=16

    ):
      super().__init__()
      self.n_components = n_components
      self.n_neighbors = 50
      self.W_train = None
      self.H_Base = None
      self.NMF = None
      self.knn = None
      self.min_A_train = None

    def _fit_to_dataset(self, fit_dataset):

      # Calculate the activations_matrix A_train for the training dataset, to calculate the PCs
      print("extracting features .................................")
      training_features = self.feature_extractor.predict(fit_dataset)
      # The activations_matrix A_train
      A_train = training_features[0][0]
      A_train = self.op.convert_to_numpy(A_train)
      # if len(A_train.shape) > 2:
      A_train = A_train[:,:, 0, 0]
    
      # self.A_in = A_train - np.min(A_train) + 1e-5
      self.A_in = A_train
      # Appliquer NMF
      print("applying NMF .................................")
      self.NMF = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=400)
      self.W_train = self.NMF.fit_transform(self.A_in)  # La matrice des coefficients (ou des caractéristiques latentes)
      self.H_Base = self.NMF.components_  # La matrice des composantes (ou la base)
      print("the shape of H_base is : ", self.H_Base.shape)
      print("the shape of W_train is  : ", self.W_train.shape)

      # Créer et ajuster le modèle kNN
      print("fitting the KNN algorithm .................................")
      self.knn = NearestNeighbors(n_neighbors=self.n_neighbors)
      self.knn.fit(self.W_train)
      print("Done fitting ...........................")
      return
      
      # else:
      #   self.min_A_train = np.min(A_train)
      #   self.A_in = A_train - self.min_A_train + 1e-5
      #   # Appliquer NMF
      #   self.NMF = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=400)
      #   self.W_train = self.NMF.fit_transform(self.A_in)  # La matrice des coefficients (ou des caractéristiques latentes)
      #   self.H_Base = self.NMF.components_  # La matrice des composantes (ou la base)
      #   print("the shape of H_base is : ", self.H_Base.shape)
      #   print("the shape of W_train is  : ", self.W_train.shape)
      #   return

    def _score_tensor(self, inputs):

      features, logits = self.feature_extractor.predict_tensor(inputs)
      # if len(features[0].shape) > 2:
      A_test = features[0][:,:, 0, 0]

      A_test = A_test.cpu()
      A_test = self.op.convert_to_numpy(A_test) # la matrice des données de test A_test
      
      W_test = self.NMF.transform(A_test)

      # Trouver les k plus proches voisins de W_test
      distances, indices = self.knn.kneighbors(W_test)
      # print("shape of W_test is : ", W_test.shape)

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



