from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.decomposition import PCA
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler



class PCA_CONCEPTS(OODBaseDetector):
    def __init__(self, 
                 n_components=10,
                 n_neighbors=20,
                 percentage_of_images=0.1,
                 distance="euclidean"

             ):
        super().__init__()
        self.n_components=n_components
        self.n_neighbors=n_neighbors
        self.percentage_of_images=percentage_of_images
        self.distance=distance
        self.Scaler = StandardScaler()
        self.PCA = PCA(n_components=self.n_components)
        self.A_train = None
        self.labels_train = None
        self.V = None
        self.U = None
        self.U_train = {}
        self.KNNs = {}

    def _fit_to_dataset(self, fit_dataset):
        # extraire les features à partir de la couche penultimate
        print("extracting features .................................")
        training_features = self.feature_extractor.predict(fit_dataset)
        self.A_train = self.op.convert_to_numpy(training_features[0][0])
        # aplatir toutes les features dans une dimension (batch, features)
        self.A_train = self.A_train.reshape(self.A_train.shape[0], -1)
        # prendre les logits 
        self.logits_train = training_features[1]["logits"]
        self.labels_train = self.op.convert_to_numpy(training_features[1]["labels"])
        print("getting correctly classified samples .................................")
        # Apply softmax to the logits across the last dimension
        probabilities = torch.softmax(self.logits_train, dim=1)
        # Extract the indices of the maximum value in each row, which are the predicted classes
        predicted_classes = self.op.convert_to_numpy(torch.argmax(probabilities, dim=1))
        self.A_train = self.A_train[predicted_classes == self.labels_train]
        self.labels_train = self.labels_train[predicted_classes == self.labels_train]
        print("calculating PCA ...................................")
        # build the nmf algorithm
        # normalisation des données (moy : 0, var : 1)
        self.A_train = self.Scaler.fit_transform(self.A_train)
        self.U = self.PCA.fit_transform(self.A_train)
        self.V = self.PCA.components_
        # sorting the U vectors to get the top 10% images that activates every concept
        print("calculating top 10 percent images ...................................")
        for concept in range(self.n_components):
            # prendre la colonne qui correspond au concept étudié
            U_concept = self.U[:, concept]
            # le nombre de 10% des images dans notre dataset
            top_percentage_len = int(len(U_concept) * self.percentage_of_images)
            # mettre en ordre décroissant les données qui activent le concept 
            sorting_indices = np.argsort(U_concept)[: : -1][:top_percentage_len]
            # stocker ces données dans un dict 
            self.U_train[concept] = self.U[sorting_indices]
            # créer et ajuster un KNN sur ces données
            self.KNNs[concept] = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.distance)
            self.KNNs[concept].fit(self.U_train[concept])
        
        return


    def _score_tensor(self, inputs):
        # extraire les features à partir de la couche penultimate
        features, logits = self.feature_extractor.predict_tensor(inputs)
        # la matrice des données de test A_test
        A_test = self.op.convert_to_numpy(features[0].cpu())      
        # aplatir toutes les features dans une dimension (batch, features)
        A_test = A_test.reshape(A_test.shape[0], -1)
        # normalisation des données test
        A_test = self.Scaler.transform(A_test)
        # distance minimale pour chaque données de test (batch,)
        min_distances = np.inf * np.ones(A_test.shape[0])
        # transformation des données test dans la nouvelle base
        U_test = self.PCA.transform(A_test)
        # pour chaque concept, on calcule la distance minimale entre test et les top 10% images qui activent ce concept
        for concept in range(self.n_components):
            distances, _ = self.KNNs[concept].kneighbors(U_test)
            # prendre la distance minimale 
            min_distance = np.mean(distances, axis=1)
            min_distances = np.minimum(min_distances, min_distance)

        return min_distances

    @property
    def requires_to_fit_dataset(self) -> bool:
        return True

    @property
    def requires_internal_features(self) -> bool:
        return True





