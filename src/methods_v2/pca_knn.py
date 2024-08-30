from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors



class PCA_KNN(OODBaseDetector):
    def __init__(
        self,
        n_components=16,
        n_neighbors=50,
        distance="euclidean"


    ):
      super().__init__()
      self.n_components = n_components
      self.n_neighbors = n_neighbors
      self.distance = distance
      self.Scaler = StandardScaler()
      self.PCA = PCA(n_components=self.n_components)
      self.KNN = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.distance)
      self.U = None
      self.V = None
      self.A_train = None
      self.Labels_train = None

      

    def _fit_to_dataset(self, fit_dataset):
      # Extraire les features à partir de la couche penultimate
      print("extracting features .....................")
      training_features = self.feature_extractor.predict(fit_dataset)
      self.A_train = self.op.convert_to_numpy(training_features[0][0])
      # aplatir toutes les features dans une seul dimension (batch, features)
      self.A_train = self.A_train.reshape(self.A_train.shape[0], -1)
      self.Labels_train = self.op.convert_to_numpy(training_features[1]["labels"])
      # normalisation des données (moy : 0, var : 1)
      self.A_train = self.Scaler.fit_transform(self.A_train)
      # Appliquer PCA
      print("Applying PCA Algorithm .....................")
      self.U = self.PCA.fit_transform(self.A_train)   # La matrice des coefficients W
      self.V = self.PCA.components_ # la base H
      # Créer et ajuster le modèle kNN
      print("fitting KNN Algorithm .....................")
      self.KNN.fit(self.U)
      print("Done Fitting .....................")
  
      return

    def _score_tensor(self, inputs):
      # Extraire les features à partir de la couche penultimate des données Test
      features, logits = self.feature_extractor.predict_tensor(inputs)
      A_test = self.op.convert_to_numpy(features[0].cpu())      
      # flatten all the features into one dimension
      A_test = A_test.reshape(A_test.shape[0], -1)
      # normalisation des données test
      A_test = self.Scaler.transform(A_test)
      # la matrice des coefficients de A_test dans la base H avec taille (N_test, K)
      W_test = self.PCA.transform(A_test) 
      # Trouver les k plus proches voisins de W_test
      distances, indices = self.KNN.kneighbors(W_test)
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