
from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF


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
      self.U = None
      self.V = None
      self.NMF = None
      self.KNN = None
      self.A_train = None
      self.Labels_train = None

    def _fit_to_dataset(self, fit_dataset):
      # extraire les features à partir de la couche penultimate
      print("extracting features .................................")
      training_features = self.feature_extractor.predict(fit_dataset)
      self.A_train = self.op.convert_to_numpy(training_features[0][0])
      # aplatir toutes les features dans une dimension (batch, features)
      self.A_train = self.A_train.reshape(self.A_train.shape[0], -1)
      self.Labels_train = self.op.convert_to_numpy(training_features[1]["labels"])
      # Appliquer NMF
      print("applying NMF .................................")
      self.NMF = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=400)
      self.U = self.NMF.fit_transform(self.A_train)  # La matrice des coefficients (ou des caractéristiques latentes)
      self.V = self.NMF.components_  # La matrice des composantes (ou la base)
      # Créer et ajuster le modèle kNN
      print("fitting the KNN algorithm .................................")
      self.KNN = NearestNeighbors(n_neighbors=self.n_neighbors)
      self.KNN.fit(self.U)
      print("Done fitting ...........................")
      return


    def _score_tensor(self, inputs):
      # extraire les features à partir de la couche penultimate
      features, logits = self.feature_extractor.predict_tensor(inputs)
      # la matrice des données de test A_test
      A_test = self.op.convert_to_numpy(features[0].cpu())      
      # aplatir toutes les features dans une dimension (batch, features)
      A_test = A_test.reshape(A_test.shape[0], -1)
      # transformation des données dans la nouvelle bases NMF
      W_test = self.NMF.transform(A_test)
      # Trouver les k plus proches voisins de W_test
      distances, indices = self.KNN.kneighbors(W_test)
      # prendre la distance euclidienne minimale
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



