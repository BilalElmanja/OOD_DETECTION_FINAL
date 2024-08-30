
from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import torch





class K_Means(OODBaseDetector):
    def __init__(
        self,
        n_centroids = 10
    ):
      super().__init__()
      self.n_centroids = n_centroids
      self.Kmeans = KMeans(n_clusters=self.n_centroids, random_state=42, max_iter=400)
      self.A_train = None
      self.Labels_train = None
      self.U = None
      self.V = None

    def _fit_to_dataset(self, fit_dataset):
      # extraire les features à partir de la couche penultimate
      print("extracting features .................................")
      training_features = self.feature_extractor.predict(fit_dataset)
      self.A_train = self.op.convert_to_numpy(training_features[0][0])
      self.Labels_train = self.op.convert_to_numpy(training_features[1]["labels"])
      # aplatir toutes les features dans une dimension (batch, features)
      self.A_train = self.A_train.reshape(self.A_train.shape[0], -1)
      print("Performing K-means clustering...")
      self.U = self.Kmeans.fit_transform(self.A_train)
      print("K-means clustering Done...")
      # get the centroids coordinates in the feature space with shape (10, 10) k*p
      self.V = self.Kmeans.cluster_centers_
      # get the labels of the centroids
      return

    def _score_tensor(self, inputs):
      # extraire les features à partir de la couche penultimate
      features, logits = self.feature_extractor.predict_tensor(inputs)
      # la matrice des données de test A_test
      A_test = self.op.convert_to_numpy(features[0].cpu())      
      # aplatir toutes les features dans une dimension (batch, features)
      A_test = A_test.reshape(A_test.shape[0], -1)
      # Calculate the Euclidean distance between each sample and the centroids
      distances = self.Kmeans.transform(A_test)
      min_distance = np.min(distances, axis=1)
      return min_distance

    def visualize_data_distributions(self, ds_in, ds_out):
        # Extract features from the penultimate layer for both in-distribution and out-of-distribution data
        def extract_features(dataset):
            features, labels = [], []
            for (x, y) in dataset:
                x = x.to("cuda:1")
                with torch.no_grad():
                    feature = self.feature_extractor.predict_tensor(x)[0][0].detach().cpu().numpy()
                    feature = feature.reshape(feature.shape[0], -1)
                features.append(feature)
                labels.append(y)
            return np.concatenate(features), np.concatenate(labels)

        features_in, labels_in = extract_features(ds_in)
        features_out, _ = extract_features(ds_out)

        # Use PCA to reduce dimensionality to 2D
        pca = PCA(n_components=10)
        features_in_2d = pca.fit_transform(features_in)
        features_out_2d = pca.transform(features_out)

        # Plot the 2D features
        plt.figure(figsize=(10, 7))

        # Plot in-distribution data with different colors for each class
        num_classes = 10
        colors = sns.color_palette("hsv", num_classes)
        for i in range(num_classes):
            plt.scatter(features_in_2d[labels_in == i, 1], features_in_2d[labels_in == i, 2], 
                        c=[colors[i]], label=f'Class {i}', alpha=0.6, edgecolors='w', s=40)

        # Plot out-of-distribution data with a unique color and star marker
        plt.scatter(features_out_2d[:, 0], features_out_2d[:, 1], c='k', marker='*', 
                    label='OOD', alpha=0.8, edgecolors='w', s=80)

        plt.legend()
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('2D Visualization of Penultimate Layer Features')
        plt.tight_layout()
        plt.savefig("./2D_features_1.png")
        plt.show()
    

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