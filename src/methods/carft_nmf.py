from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.decomposition import NMF
import torch
from sklearn.neighbors import NearestNeighbors


class CRAFT_NMF(OODBaseDetector):
    def __init__(self, 
                 n_components=16

             ):
        super().__init__()
        self.n_components=n_components
        self.NMFs = {}  # dict pour les NMFs pour chaque classe
        self.H_Bases = {}
        self.W_trains = {}
        # self.Scaler = None
        self.labels_train = None

    def _fit_to_dataset(self, fit_dataset):
        # Extraction des caractéristiques et des étiquettes
        training_features = self.feature_extractor.predict(fit_dataset)
        # if len(training_features[0][0].shape) > 2:
        shape = training_features[0][0].shape
        A_train = training_features[0][0].reshape(shape[0], -1)

        A_train = self.op.convert_to_numpy(A_train)
        self.logits_train = training_features[1]["logits"]
        self.labels_train = self.op.convert_to_numpy(training_features[1]["labels"])
        # print("example of labels : ", self.labels_train[:10])
        # print("logits shape : ", self.logits_train.shape)
        # print("labels shape : ", self.labels_train.shape)
        # Apply softmax to the logits across the last dimension
        probabilities = torch.softmax(self.logits_train, dim=1)
        # Extract the indices of the maximum value in each row, which are the predicted classes
        predicted_classes = torch.argmax(probabilities, dim=1)
        # print("shape of predicted classes : ", predicted_classes.shape)
        predicted_classes = self.op.convert_to_numpy(predicted_classes)
        # print("example of predicted classes : ", predicted_classes[:10])
        # print(predicted_classes == self.labels_train)
        A_train = A_train[predicted_classes == self.labels_train]
        self.labels_train = self.labels_train[predicted_classes == self.labels_train]
        self.classes = np.unique(self.labels_train)
        self.NMF =None  # Dictionnaire pour stocker PCA par classe
        self.W_Base =None  # Dictionnaire pour stocker H_base par classe
        self.U_train = None# Dictionnaire pour stocker W_train par classe
        print("calculating NMF .....")
        # build the nmf algorithm
        nmf = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=1000)
        U_train = nmf.fit_transform(A_train)
        print("shape of U_train : ", U_train.shape)
        W_base = nmf.components_
        # save the nmf, W and U
        self.NMF = nmf
        self.W_Base = W_base
        self.U_train = {}
        # sorting the U vectors to get the top 10% images that activates every concept
        for concept in range(self.n_components):
            U_concept = U_train[:, concept]
            top_10_len = int(len(U_concept) * 0.01)
            sorting_indices = np.argsort(U_concept)[: : -1][:top_10_len]
            U_concept = U_train[sorting_indices]
            self.U_train[concept] = U_concept
            print("shape of U_concept : ", U_concept.shape)

        return
            
        # Calculate accuracy
        # correct_predictions = sum(1 for true, pred in zip(self.labels_train, predicted_classes) if true == pred)
        # accuracy = correct_predictions / len(self.labels_train)
        # print("accuracy is : ", accuracy)


    def _score_tensor(self, inputs):
        features, _ = self.feature_extractor.predict_tensor(inputs)
        # if len(features[0].shape) > 2:
        shape = features[0].shape
        A_test = features[0].reshape(shape[0], -1)
        A_test = self.op.convert_to_numpy(A_test.cpu())
        min_distances = np.inf * np.ones(A_test.shape[0])
        # print("shape of min_distances : ", min_distances.shape)
        # calculating u for every class
    
        nmf = self.NMF
        # getting the U vectors for test inputs
        W_test = nmf.transform(A_test)
        print("shape of W_test : ", W_test.shape)
        # for every class of training data, we get the vectors that activate every concept
        for concept in range(self.n_components):
            U_concept = self.U_train[concept]
            neigh = NearestNeighbors(n_neighbors=20)
            neigh.fit(U_concept)

            distances, _ = neigh.kneighbors(W_test)
            # print("shape of distances : ", distances.shape)
            # print( "class : ", class_label, "shape of distances : ", distances.shape)
            min_distance = np.min(distances, axis=1)
            # print("shape of min_distance : ", min_distance.shape)
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




