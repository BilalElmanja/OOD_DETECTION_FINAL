from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.decomposition import NMF
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import torch
import seaborn as sns
import torch.nn as nn
import numpy as np
import numpy as np
from xplique.attributions.global_sensitivity_analysis import (HaltonSequenceRS, JansenEstimator)


class CRAFT_CONCEPTS(OODBaseDetector):
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
        self.model = model.to("cuda:1")
        self.g = None
        self.h = None
        self.NMF = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=1000)
        self.A_train = None
        self.labels_train = None
        self.U_train = {}
        self.concept_bank_w = None
        self.most_important_concepts = None
        self.importances= None
        self.concepts_images = {}
        self.KNNs = {}

    def _fit_to_dataset(self, fit_dataset):
        self.g = nn.Sequential(*(list(self.model.children())[:-1])) # input to penultimate layer
        self.h = nn.Sequential(*(list(self.model.children())[-1:])) # penultimate layer to logits
        # extraire les features à partir de la couche penultimate
        print("extracting features .................................")
        print("length of dataset : ", len(fit_dataset.dataset))
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
        print("accuracy of the model : ", len(self.labels_train) / len(fit_dataset.dataset) * 100 )
        print("\n")
        print("calculating NMF ...................................")
        # build the nmf algorithm
        U_train = self.NMF.fit_transform(self.A_train)
        self.concept_bank_w = self.NMF.components_
        # sorting the U vectors to get the top 10% images that activates every concept
        print("calculating top important concepts ...................................")
        masks = HaltonSequenceRS()(self.n_components, nb_design = self.n_designs)
        print("shape of masks : ", masks.shape)
        estimator = JansenEstimator()
        importances = []

        if len(U_train.shape) == 2:
            # apply the original method of the paper
            for idx, coeff in enumerate(U_train):
                if idx in [1000, 10000, 30000, 40000, 49999]:
                    print("calculating the concept : ", idx)
                u_perturbated = coeff[None, :] * masks
                a_perturbated = u_perturbated @ self.concept_bank_w
                a_perturbated = torch.tensor(np.array(a_perturbated), device="cuda:1")
                # print(a_perturbated.size())
                y_pred = self.h(a_perturbated)
                y_pred = y_pred.cpu().detach().numpy()
                y_pred = y_pred[:, self.labels_train[idx]]

                stis = estimator(masks, y_pred, self.n_designs)
                importances.append(stis)

        importances = np.mean(importances, 0)
        print("importances before : ", importances)
        self.most_important_concepts = np.argsort(importances)[::-1]
        self.importances = importances[self.most_important_concepts][:self.n_important_concepts]
        print("most important concepts : ", self.most_important_concepts[:self.n_important_concepts])
        print("importances after : ", self.importances)
        
        print("applying KNN algorithm .............")
        for concept in self.most_important_concepts[:self.n_important_concepts]:
            # prendre la colonne qui correspond au concept étudié
            U_concept = U_train[:, concept]
            # le nombre de 10% des images dans notre dataset
            top_10_len =  int(len(U_concept) * self.percentage_images_concept)
            # mettre en ordre décroissant les données qui activent le concept 
            sorting_indices = np.argsort(U_concept)[: : -1][:top_10_len]
            self.concepts_images[concept] = [fit_dataset.dataset[int(index)] for index in sorting_indices]
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
            min_distance = np.mean(distances, axis=1)
            min_distances = np.minimum(min_distances, min_distance)

        return min_distances
    
    def visualize_concepts_images(self):
        # Créer une figure avec une grille pour les images et une colonne pour les importances
        fig, ax = plt.subplots(self.n_important_concepts, 5, figsize=(5, 7))
        fig.subplots_adjust(wspace=0.01, hspace=0.1)

        # Définir les couleurs pour les bordures
        colors = sns.color_palette("husl", self.n_important_concepts)

        # Visualiser les images des concepts
        for i, concept in enumerate(self.most_important_concepts[:self.n_important_concepts]):
            for j, image in enumerate(self.concepts_images[concept]):
                ax[i, j].imshow(image[0].permute(1, 2, 0) * 0.5 + 0.5)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                for spine in ax[i, j].spines.values():
                    spine.set_edgecolor(colors[i])
                    spine.set_linewidth(2)
            ax[i, 0].set_ylabel(f"Concept {concept}", fontsize=12, rotation=0, labelpad=50, color=colors[i])

        # Création d'une nouvelle figure pour les importances des concepts
        fig_importances = plt.figure(figsize=(5, 7))
        ax_importance = fig_importances.add_subplot(111)

        # Visualiser les importances des concepts
        sns.barplot(y=[f"Concept {c}" for c in self.most_important_concepts[:self.n_important_concepts]],
                    x=self.importances, palette=colors, ax=ax_importance)
        ax_importance.set_xlabel("Importance")
        ax_importance.set_title("Importances des Concepts")
        # ax_importance.invert_yaxis()  # Inverser l'axe y pour avoir le concept le plus important en haut

        plt.tight_layout()
        plt.show()
        return
    
    def plot_most_important_concepts(self):
        # get the importances of the concepts
        importances = self.importances
        # get the most important concepts
        most_important_concepts = self.most_important_concepts[:self.n_important_concepts]
        # plot the importances of the concepts
        plt.bar(np.arange(len(importances)), importances)
        plt.title("Importances of the concepts")
        plt.show()

    def enhance_image_resolution(self, image):
        pass
    
    def plot_image_importances(self, image, class_id):
        # get the features of the image
        features = self.feature_extractor.predict_tensor(image)
        # get the features of the image
        A_test = self.op.convert_to_numpy(features[0].cpu())
        # aplatir toutes les features dans une dimension (batch, features)
        A_test = A_test.reshape(A_test.shape[0], -1)
        # transformation des données test dans la nouvelle base
        U_test = self.NMF.transform(A_test)
        print("calculating top important concepts ...................................")
        masks = HaltonSequenceRS()(self.n_components, nb_design = self.n_designs)
        estimator = JansenEstimator()
        importances = []

        if len(U_test.shape) == 2:
            # apply the original method of the paper
            for idx, coeff in enumerate(U_test):
                u_perturbated = coeff[None, :] * masks
                a_perturbated = u_perturbated @ self.concept_bank_w
                a_perturbated = torch.tensor(np.array(a_perturbated), device="cuda:1")
                # print(a_perturbated.size())
                y_pred = self.h(a_perturbated)
                y_pred = y_pred.cpu().detach().numpy()
                y_pred = y_pred[:, class_id]
                stis = estimator(masks, y_pred, self.n_designs)
                importances.append(stis)

        importances = np.mean(importances, 0)
        print("importances before : ", importances)
        self.most_important_concepts = np.argsort(importances)[::-1]
        self.importances = importances[self.most_important_concepts][:1]
        print("most important concept : ", self.most_important_concepts[:1])
        print("importances after : ", importances)

    
        
        

    @property
    def requires_to_fit_dataset(self) -> bool:
        return True

    @property
    def requires_internal_features(self) -> bool:
        return True






