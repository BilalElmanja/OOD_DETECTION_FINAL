import os
import argparse
import time
from contextlib import contextmanager
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import sys
sys.path.append("../")
from oodeel.methods import MLS, Energy, Entropy, DKNN, Gram, Mahalanobis, ODIN, VIM
from methods import   K_Means_Mahalanobis, PCA_MAHALANOBIS, NMF_MAHALANOBIS, PCA_unique_class_KNN, PCA_Unique_Class_Mahalanobis, NMF_Unique_Classes_KNN, NMF_Unique_Class_Mahalanobis, NMF_NORM, CRAFT_PER_CLASS_NMF, CRAFT_PER_CLASS_NMF_V2
from methods_v2 import K_Means, NMF_KNN, PCA_KNN, NMF_CONCEPTS, CRAFT_CONCEPTS, PCA_NORM
from data_preprocessing import get_train_dataset_cifar10, get_test_dataset_cifar10, get_train_dataset_cifar100, get_test_dataset_cifar100, get_test_dataset_places365, get_test_dataset_svhn, get_test_dataset_texture, get_test_dataset_Tiny, get_test_dataset_NINCO, get_test_dataset_OpenImage_O, get_train_dataset_inaturalist, get_test_dataset_SSB_hard
from models import load_pretrained_weights_32
from oodeel.eval.metrics import bench_metrics
from oodeel.types import List
from oodeel.eval.plots import plot_ood_scores, plot_roc_curve, plot_2D_features, plot_3D_features
from sklearn.metrics import accuracy_score



# load the model with pretrained weights
model = load_pretrained_weights_32()

# 1a- load in-distribution dataset: CIFAR-10
ds_fit = get_train_dataset_cifar10()
ds_in = get_test_dataset_cifar10()
ds_out = get_test_dataset_svhn()
# 1b- load out-of-distribution datasets
# ds_out_dict = {
#             "cifar100": get_test_dataset_cifar100(),
#             "svhn" : get_test_dataset_svhn(),
#             "places365" : get_test_dataset_places365(),
#             "texture" : get_test_dataset_texture(),
#             "Tin": get_test_dataset_Tiny(),
#         }


# 2- instantiate the method
method = K_Means(n_centroids=10)

# 3- fit the method to the dataset
method.fit(model.to("cuda:1"), fit_dataset=ds_fit, feature_layers_id=[-2])


method.visualize_data_distributions(ds_in, ds_out)



