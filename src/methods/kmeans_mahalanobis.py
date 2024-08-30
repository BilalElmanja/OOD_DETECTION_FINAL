
from oodeel.datasets import OODDataset
from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.spatial.distance import cdist
from sklearn.covariance import MinCovDet


class K_Means_Mahalanobis(OODBaseDetector):
    def __init__(
        self,
        n_centroids = 10
    ):
      super().__init__()

      self.CAVs = None
      self.k = n_centroids
      self.A_in = None

    def _fit_to_dataset(self, fit_dataset):
      # we calculate the activations_matrix A_train for the training dataset, in order to calculate the CAVs Matrix
      training_features = self.feature_extractor.predict(fit_dataset)
      # the activations_matrix A_train
      A_train = training_features[0][0]
      A_train = self.op.convert_to_numpy(A_train)
      self.A_in = A_train
      if len(self.A_in.shape) > 2:
         self.A_in = self.A_in[:,:, 0, 0]
      
      print("Performing K-means clustering...")
      print("shape of A_in is : ", self.A_in.shape)
      kmeans = KMeans(n_clusters=self.k, random_state=42, max_iter=200).fit(self.A_in)
      print("K-means clustering Done...")
      print("#------------------------------------------------------------")
      # get the centroids coordinates in the feature space with shape (10, 10) k*p
      self.CAVs = kmeans.cluster_centers_
      # get the labels of the centroids
      self.MCD = MinCovDet().fit(self.A_in)
      return

    def _score_tensor(self, inputs):

      features, logits = self.feature_extractor.predict_tensor(inputs)
      
      if len(features[0].shape) > 2:
         features[0] = features[0][:,:, 0, 0]
      # Calculate the Euclidean distance between each sample and the centroids
      distances = cdist(features[0].cpu(), self.CAVs, 'mahalanobis', VI=self.MCD.precision_)
      
      min_distances = distances.min(axis=1)
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



# kmeans = K_Means()