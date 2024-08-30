from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.decomposition import NMF
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import MinCovDet
from scipy.optimize import minimize
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

def reconstruction_loss(W_flat, A_test, H_base):
    """Calculer la perte de reconstruction ||A_test - W_test * H_base||_2."""
    W_test = W_flat.reshape(A_test.shape[0], -1)
    reconstruction = np.dot(W_test, H_base)
    return np.linalg.norm(A_test - reconstruction)



class NMF_Unique_Class_Mahalanobis(OODBaseDetector):
    def __init__(
        self,
        n_components=9
    ):
      super().__init__()
      self.A_train = None
      self.A_in = None
      self.labels_train = None
      self.Scaler = None
      self.n_components = n_components 
      self.NMFs = {}
      self.H_Bases = {}
      self.W_trains = {}

    def _fit_to_dataset(self, fit_dataset):
      # Extraction des caractéristiques et des étiquettes
        training_features = self.feature_extractor.predict(fit_dataset)
        A_train = self.op.convert_to_numpy(training_features[0][0])
        self.labels_train = self.op.convert_to_numpy(training_features[1]["labels"])
        if len(A_train.shape) > 2:
            A_train = A_train[:,:, 0, 0]
        # S'assurer que les données sont positives pour NMF
        self.A_in = A_train - np.min(A_train) + 1e-5
        
        # self.Scaler = StandardScaler()
        # A_train_scaled = self.Scaler.fit_transform(self.A_in)
        self.classes_ = np.unique(self.labels_train)
        
        for class_label in self.classes_:

            A_train_class = self.A_in[self.labels_train == class_label]
            nmf = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=400)
            W_train_class = nmf.fit_transform(A_train_class)
            H_Base_class = nmf.components_
            
            self.NMFs[class_label] = nmf  # Stocker l'objet NMF
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
    #   A_test_scaled = self.Scaler.transform(A_test)
      A_test = A_test - np.min(A_test) + 1e-5
      min_distances = np.inf * np.ones(A_test.shape[0])
      print("shape of min_distances is : ", min_distances.shape)
      for class_label in self.classes_:
          W_train_class = self.W_trains[class_label]
          MCD = self.MCDs[class_label]  # Get the precision matrix for the class
          H_base_class = self.H_Bases[class_label]
          nmf = self.NMFs[class_label]
          # Initialisation de W_test comme une matrice aplatie (pour l'optimisation)
        #   initial_W_test_flat = np.random.randn(A_test.shape[0] * W_train_class.shape[1])
          # Minimiser la perte de reconstruction
        #   result = minimize(reconstruction_loss, initial_W_test_flat, args=(A_test, H_base_class), method='L-BFGS-B')
          # Remodeler W_test dans sa forme originale (M, K)
        #   W_test_optimized = result.x.reshape(A_test.shape[0], W_train_class.shape[1])
          W_test = nmf.transform(A_test)
          
          # Calculate Mahalanobis distance using the parallel function
          distance_matrix = cdist(W_test, W_train_class, 'mahalanobis', VI=MCD.precision_)
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



nmf_per_class_mahalanobis = NMF_Unique_Class_Mahalanobis()

