from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import numpy as np
import numpy as np
from xplique.attributions.global_sensitivity_analysis import (HaltonSequenceRS, JansenEstimator)


class EXPLIQUE_CONCEPTS(OODBaseDetector):
    def __init__(self, 
                 n_components=10,
                 n_designs=32,
                 n_important_concepts=10,
                 percentage_images_concept=0.5,
                 model=None
        
             ):
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = 20
        self.n_designs = n_designs
        self.n_important_concepts = n_important_concepts
        self.percentage_images_concept = percentage_images_concept
        self.model = model.to("cuda")
        self.NMF = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=1000)
        self.A_train = None
        self.labels_train = None
        self.U_train = {}
        self.concept_bank_w = None
        self.most_important_concepts = None
        self.KNNs = {}

    def _fit_to_dataset(self, fit_dataset):
        self.h = nn.Sequential(*(list(self.model.children())[-1:])) # penultimate layer to logits
        # extraire les features à partir de la couche penultimate
        print("extracting features .................................")
        print("length of dataset : ", len(fit_dataset.dataset))
        plt.imsave("image_1.png",(fit_dataset.dataset[0][0] * 0.5 + 0.5).permute(2, 0, 1))
        print("first image : ", fit_dataset.dataset[0][0].shape)
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
        print("calculating NMF ...................................")
        # build the nmf algorithm
        U_train = self.NMF.fit_transform(self.A_train)
        self.concept_bank_w = self.NMF.components_
        # sorting the U vectors to get the top 10% images that activates every concept
        print("calculating top 10 concepts ...................................")
        masks = HaltonSequenceRS()(self.n_components, nb_design = self.n_designs)
        estimator = JansenEstimator()
        importances = []

        if len(U_train.shape) == 2:
            # apply the original method of the paper
            for idx, coeff in enumerate(U_train):
                u_perturbated = coeff[None, :] * masks
                a_perturbated = u_perturbated @ self.concept_bank_w
                a_perturbated = torch.tensor(np.array(a_perturbated), device="cuda")
                y_pred = self.h(a_perturbated)
                y_pred = y_pred.cpu().detach().numpy()
                y_pred = y_pred[:, self.labels_train[idx]]

                stis = estimator(masks, y_pred, self.n_designs)
                importances.append(stis)

        importances = np.mean(importances, 0)
        print("importances before : ", importances)
        self.most_important_concepts = np.argsort(importances)[::-1]
        importances = importances[self.most_important_concepts][:self.n_important_concepts]
        print("most important concepts : ", self.most_important_concepts[:self.n_important_concepts])
        print("importances after : ", importances)
        
        print("applying KNN algorithm .............")
        for concept in self.most_important_concepts[:self.n_important_concepts]:
            # prendre la colonne qui correspond au concept étudié
            U_concept = U_train[:, concept]
            # le nombre de 10% des images dans notre dataset
            top_10_len = int(len(U_concept) * self.percentage_images_concept)
            # mettre en ordre décroissant les données qui activent le concept 
            sorting_indices = np.argsort(U_concept)[: : -1][:top_10_len]
            # stocker ces données dans un dict 
            self.U_train[concept] = U_train[sorting_indices]
            # créer et ajuster un KNN sur ces données
            self.KNNs[concept] = NearestNeighbors(n_neighbors=self.n_neighbors)
            self.KNNs[concept].fit(self.U_train[concept])
        
        return


    def _score_tensor(self, inputs):
        # extraire les features à partir de la couche penultimate
        features, logits = self.feature_extractor.predict_tensor(inputs)
        # la matrice des données de test A_test
        A_test = self.op.convert_to_numpy(features[0].cpu())      
        # aplatir toutes les features dans une dimension (batch, features)
        A_test = A_test.reshape(A_test.shape[0], -1)
        # distance minimale pour chaque données de test (batch,)
        min_distances = np.inf * np.ones(A_test.shape[0])
        # transformation des données test dans la nouvelle base
        U_test = self.NMF.transform(A_test)
        # pour chaque concept, on calcule la distance minimale entre test et les top 10% images qui activent ce concept
        for concept in self.most_important_concepts[:self.n_important_concepts]:
            distances, _ = self.KNNs[concept].kneighbors(U_test)
            # prendre la distance minimale 
            min_distance = np.min(distances, axis=1)
            min_distances = np.minimum(min_distances, min_distance)

        return min_distances

    @property
    def requires_to_fit_dataset(self) -> bool:
        return True

    @property
    def requires_internal_features(self) -> bool:
        return True

# Note: Assurez-vous que clear_output() est appelé à l'endroit approprié si nécessaire, par exemple :
# from IPython.display import clear_output
# clear_output()




