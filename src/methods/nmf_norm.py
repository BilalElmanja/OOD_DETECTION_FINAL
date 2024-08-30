
from oodeel.methods.base import OODBaseDetector
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF
from scipy.optimize import minimize


# def reconstruction_loss(W_flat, A_test, H_base):
#     """Calculer la perte de reconstruction ||A_test - W_test * H_base||_2."""
#     W_test = W_flat.reshape(A_test.shape[0], -1)
#     reconstruction = np.dot(W_test, H_base)
#     return np.linalg.norm(A_test - reconstruction)


class NMF_NORM(OODBaseDetector):
    def __init__(
        self,
        n_components=16

    ):
      super().__init__()
      self.n_components = n_components
      self.W_train = None
      self.H_Base = None
      self.NMF = None

    def _fit_to_dataset(self, fit_dataset):
        # Calculate the activations_matrix A_train for the training dataset, to calculate the PCs
        training_features = self.feature_extractor.predict(fit_dataset)
        # The activations_matrix A_train
        A_train = training_features[0][0]
        if len(A_train.shape) > 2:
            A_train_shape = A_train.shape
            A_train = A_train.reshape(A_train_shape[0], -1)

        A_train = self.op.convert_to_numpy(A_train)
        self.A_in = A_train
        # Appliquer NMF
        self.NMF = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=400)
        self.W_train = self.NMF.fit_transform(self.A_in)  # La matrice des coefficients (ou des caractéristiques latentes)
        self.H_Base = self.NMF.components_  # La matrice des composantes (ou la base)
        print("the shape of H_base is : ", self.H_Base.shape)
        print("the shape of W_train is  : ", self.W_train.shape)
        return
        
     

    def _score_tensor(self, inputs):

      features, logits = self.feature_extractor.predict_tensor(inputs)
      if len(features[0].shape) > 2:
        A_test_shape = features[0].shape
        A_test = features[0].reshape(A_test_shape[0], -1)

      A_test = A_test.cpu()
      A_test = self.op.convert_to_numpy(A_test) # la matrice des données de test A_test
      
      W_test = self.NMF.transform(A_test)

      # calculer la norme des vecteurs
      W_test_norms = np.linalg.norm(W_test, axis=1)
    
      return np.negative(W_test_norms)

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


nmf_norm = NMF_NORM()

