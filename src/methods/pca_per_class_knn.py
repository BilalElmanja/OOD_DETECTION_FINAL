from oodeel.methods.base import OODBaseDetector
import numpy as np
from IPython.display import clear_output
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors




class PCA_unique_class_KNN(OODBaseDetector):
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

      return

    def _score_tensor(self, inputs):

      features, logits = self.feature_extractor.predict_tensor(inputs)

      if len(features[0].shape) > 2:
         features[0] = features[0][:,:, 0, 0]
      A_test = features[0].cpu()
      A_test = self.op.convert_to_numpy(A_test) # la matrice des données de test A_test
      A_test_scaled = self.Scaler.transform(A_test)

      # Initialiser un vecteur pour stocker le score minimum pour chaque entrée
      min_distances = np.inf * np.ones(A_test_scaled.shape[0])
      # print("shape of  min_distances : ", min_distances.shape)
      
      for class_label in self.classes_:
          # print("class label number : ", class_label)
          pca = self.PCAs[class_label]  # Obtenir PCA pour la classe
          W_test_class = pca.transform(A_test_scaled)  # Transformer les données de test
          # print("shape of  W_test_class class {}: ".format(class_label), W_test_class.shape)
          # Calculer la distance aux voisins les plus proches dans l'espace PCA de la classe
          neigh = NearestNeighbors(n_neighbors=50)
          W_train_class = self.W_trains[class_label]
          neigh.fit(W_train_class)  # Adapter aux données d'entraînement de la classe
          distances, _ = neigh.kneighbors(W_test_class)  # Obtenir les distances aux k plus proches voisins
          min_distance = np.min(distances, axis=1)  # Min des distances
          # print("the shape of distances : ", distances.shape)
          # print("shape of  min_distance class {}: ".format(class_label), min_distance.shape)
          # Mettre à jour le score minimum
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



pca_per_class = PCA_unique_class_KNN()

