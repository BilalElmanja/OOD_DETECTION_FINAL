from oodeel.methods.base import OODBaseDetector
import numpy as np
from sklearn.decomposition import NMF
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from xplique.concepts import CraftTorch as Craft
from xplique.concepts import DisplayImportancesOrder
from xplique.concepts import CraftManagerTorch as CraftManager
from xplique.concepts import DisplayImportancesOrder



class CRAFT_PER_CLASS_NMF_V2(OODBaseDetector):
    def __init__(self, 
                 n_components=16,
                 model = None

             ):
        super().__init__()
        self.n_components=n_components
        self.model = model
        self.NMFs = {}  # dict pour les NMFs pour chaque classe
        self.H_Bases = {}
        self.W_trains = {}
        # self.Scaler = None
        self.labels_train = None

    def _fit_to_dataset(self, fit_dataset):
        print(list(self.model.children())[-2])
        g = nn.Sequential(*(list(self.model.children())[:-1])) # input to penultimate layer
        h = nn.Sequential(*(list(self.model.children())[-1:])) # penultimate layer to logits
        
        # create the dataset from the torhc loader
        inputs = []
        labels = []
        for x, y in fit_dataset:
            x = self.op.convert_to_numpy(x)
            inputs = inputs + [*x]
            y = self.op.convert_to_numpy(y)
            labels = labels + [*y]
        
        inputs = torch.tensor(np.array(inputs))
        print("shape of inputs : ", inputs.size())
        labels = torch.tensor(np.array(labels))
        print("shape of labels : ", labels.size())
        # list_of_class_of_interest = [0, 491, 497, 569, 574] # tench chainsaw church truck golfball
        list_of_class_of_interest = [0, 1]

        cm = CraftManager(input_to_latent_model = g,
                    latent_to_logit_model = h,
                    number_of_concepts = 10,
                    patch_size = 32,
                    batch_size = 64,
                    inputs = inputs,
                    labels = labels,
                    list_of_class_of_interest = list_of_class_of_interest,
                    device="cuda")
        
        cm.fit(nb_samples_per_class=50)
        cm.estimate_importance()


        

        return
            
        # Calculate accuracy
        # correct_predictions = sum(1 for true, pred in zip(self.labels_train, predicted_classes) if true == pred)
        # accuracy = correct_predictions / len(self.labels_train)
        # print("accuracy is : ", accuracy)

       

        return

    def _score_tensor(self, inputs):
        features, _ = self.feature_extractor.predict_tensor(inputs)
        # if len(features[0].shape) > 2:
        A_test = features[0].reshape(self.shape[0], -1)
        A_test = self.op.convert_to_numpy(A_test.cpu())
        
            

        
        
        return 

    @property
    def requires_to_fit_dataset(self) -> bool:
        return True

    @property
    def requires_internal_features(self) -> bool:
        return True

# Note: Assurez-vous que clear_output() est appelé à l'endroit approprié si nécessaire, par exemple :
# from IPython.display import clear_output
# clear_output()



craft = CRAFT_PER_CLASS_NMF_V2()

