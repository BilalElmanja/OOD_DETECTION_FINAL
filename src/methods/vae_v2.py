import torch
import torch.nn as nn
import torch.optim as optim
from oodeel.datasets import OODDataset
from oodeel.methods.base import OODBaseDetector
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from contextlib import contextmanager
import torch
import sys
sys.path.append("../")
from data_preprocessing import get_train_dataset_cifar10, get_test_dataset_cifar10, get_train_dataset_cifar100, get_test_dataset_cifar100, get_test_dataset_places365, get_test_dataset_svhn, get_test_dataset_texture, get_test_dataset_Tiny, get_test_dataset_NINCO, get_test_dataset_OpenImage_O, get_train_dataset_inaturalist, get_test_dataset_SSB_hard
from models import load_pretrained_weights_32
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import seaborn as sns
from IPython.display import clear_output
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

from oodeel.eval.metrics import bench_metrics
from oodeel.eval.plots import plot_ood_scores, plot_roc_curve, plot_2D_features


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, latent_dim)
        self.fc4 = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.fc3(x)
        log_var = self.fc4(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class VAE(nn.Module):
    def __init__(self, input_dim=512, latent_dim=64):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var
    
    def get_latent_variables(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return z
    

def extract_latent_variables(model, dataloader):
    model.eval()
    latent_vars = []
    with torch.no_grad():
        for features in dataloader: # Ensure data is on the correct device
            latent_var = model.get_latent_variables(features[0])
            latent_vars.append(latent_var)
    latent_vars = torch.cat(latent_vars, dim=0)
    return latent_vars





def train_vae(model, train_loader, epochs=100, learning_rate=2e-4):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        with tqdm(train_loader) as tepoch:
            for features in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                # data = data.to(device)  # Ensure data is on the correct device (CPU/GPU)
                optimizer.zero_grad()
                x_reconstructed, mu, log_var = model(features[0])
                recon_loss = nn.functional.binary_cross_entropy(x_reconstructed, features[0], reduction='sum')
                kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + kl_div
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                tepoch.set_postfix(loss=total_loss)

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader.dataset)}')


class VAE_OOD_Detector(OODBaseDetector):
    def __init__(self):
        super().__init__()
        self.model = VAE(input_dim=512, latent_dim=64)

    def _fit_to_dataset(self, fit_dataset):
      # we calculate the activations_matrix A_train for the training dataset, in order to calculate the CAVs Matrix
    #   print("shape : ", fit_dataset.shape)
      training_features = self.feature_extractor.predict(fit_dataset)
      # the activations_matrix A_train
      A_train = training_features[0][0]
      if len(A_train.shape) > 2:
        A_train = A_train[:,:, 0, 0]
        
      print(" shape of A_train is : ", A_train.shape)
      # Create a TensorDataset
      feature_dataset = TensorDataset(A_train)
      # Create a DataLoader
      train_loader = DataLoader(feature_dataset, batch_size=512, shuffle=True)
      # fitting the vae model
      train_vae(self.model, train_loader )
      # eval mode for vae
      self.model.eval()



    def _score_tensor(self, inputs):
        features, logits = self.feature_extractor.predict_tensor(inputs)
        if len(features[0].shape) > 2:
            features[0] = features[0][:,:, 0, 0]
        # self.model.eval()
        with torch.no_grad():
            x_reconstructed, _, _ = self.model(features[0])
            recon_loss = nn.functional.binary_cross_entropy(x_reconstructed, features[0], reduction='none')
            recon_loss = torch.mean(recon_loss, dim=1)
        return recon_loss.cpu().numpy()

    @property
    def requires_to_fit_dataset(self) -> bool:
        return True

    @property
    def requires_internal_features(self) -> bool:
        return True


ds_train = get_train_dataset_cifar10()
ds_in = get_test_dataset_cifar10()
ds_out = get_test_dataset_svhn()
model = load_pretrained_weights_32()
model.to(device)
model.eval()

for x, y in ds_train:
    x = x.to(device)
    y = y.to(device)

for x, y in ds_in:
    x = x.to(device)
    y = y.to(device)

for x, y in ds_out:
    x = x.to(device)
    y = y.to(device)


vae = VAE_OOD_Detector()
vae.fit(model, feature_layers_id=[-2], fit_dataset=ds_train)

print("scoring for OOD Data (svhn) ... ")
features_in = vae.feature_extractor.predict(ds_in)[0][0]
features_in = features_in[:,:, 0, 0]
features_out = vae.feature_extractor.predict(ds_out)[0][0]
features_out = features_out[:,:, 0, 0]
# Create a TensorDataset
feature_dataset_in = TensorDataset(features_in)
feature_dataset_out = TensorDataset(features_out)
# Create a DataLoader
data_loader_in = DataLoader(feature_dataset_in, batch_size=128, shuffle=True)
data_loader_out = DataLoader(feature_dataset_out, batch_size=128, shuffle=True)
# extract embeddings
latents_in = extract_latent_variables(vae.model, data_loader_in)
latents_out = extract_latent_variables(vae.model, data_loader_out)
# Flatten the latent variables if necessary and convert to NumPy for easier processing
latents_in = latents_in.cpu().numpy().flatten()
latents_out = latents_out.cpu().numpy().flatten()
# Visualization
plt.figure(figsize=(12, 6))
sns.histplot(latents_in, color="blue", label="ID", kde=True)
sns.histplot(latents_out, color="red", label="OOD", kde=True)
plt.title("Distribution of Latent Variables")
plt.legend()
plt.savefig("./latents_dist.png")




# # === metrics ===
# # auroc / fpr95
# metrics = bench_metrics(
#     (scores_in, scores_out),
#     metrics=["auroc", "fpr95tpr"],
# )
# print("=== Metrics ===")
# for k, v in metrics.items():
#     print(f"{k:<10} {v:.6f}")

# print("\n=== Plots ===")
# # hists / roc
# plt.figure(figsize=(13, 6))
# plt.subplot(121)
# plot_ood_scores(scores_in, scores_out, log_scale=False)
# plt.subplot(122)
# plot_roc_curve(scores_in, scores_out)
# plt.tight_layout()
# plt.savefig("./OOD_vae_cifar10_svhn_plot.png")


# # Assuming id_loader and ood_loader are your DataLoader instances for ID and OOD data
# id_latent_vars = extract_latent_variables(vae.model, train_loader)
# ood_latent_vars = extract_latent_variables(vae.model, train_loader)

# # Flatten the latent variables if necessary and convert to NumPy for easier processing
# id_latent_vars_np = id_latent_vars.cpu().numpy().flatten()
# ood_latent_vars_np = ood_latent_vars.cpu().numpy().flatten()


# # Visualization
# plt.figure(figsize=(12, 6))
# sns.histplot(id_latent_vars_np, color="blue", label="ID", kde=True)
# sns.histplot(ood_latent_vars_np, color="red", label="OOD", kde=True)
# plt.title("Distribution of Latent Variables")
# plt.legend()
# plt.show()







# print("scoring for OOD Data (places365) ... ")
# # scores_in, _ = vae.score(ds_in)
# ds_out = get_test_dataset_places365()
# for x, y in ds_out:
#     x = x.to(device)
#     y = y.to(device)

# scores_out, _ = vae.score(ds_out)
# # === metrics ===
# # auroc / fpr95
# metrics = bench_metrics(
#     (scores_in, scores_out),
#     metrics=["auroc", "fpr95tpr"],
# )
# print("=== Metrics ===")
# for k, v in metrics.items():
#     print(f"{k:<10} {v:.6f}")

# print("\n=== Plots ===")
# # hists / roc
# plt.figure(figsize=(13, 6))
# plt.subplot(121)
# plot_ood_scores(scores_in, scores_out, log_scale=False)
# plt.subplot(122)
# plot_roc_curve(scores_in, scores_out)
# plt.tight_layout()
# plt.savefig("./OOD_vae_cifar10_places365_plot.png")







# print("scoring for OOD Data (texture) ... ")
# # scores_in, _ = vae.score(ds_in)
# ds_out = get_test_dataset_texture()
# for x, y in ds_out:
#     x = x.to(device)
#     y = y.to(device)

# scores_out, _ = vae.score(ds_out)
# # === metrics ===
# # auroc / fpr95
# metrics = bench_metrics(
#     (scores_in, scores_out),
#     metrics=["auroc", "fpr95tpr"],
# )
# print("=== Metrics ===")
# for k, v in metrics.items():
#     print(f"{k:<10} {v:.6f}")

# print("\n=== Plots ===")
# # hists / roc
# plt.figure(figsize=(13, 6))
# plt.subplot(121)
# plot_ood_scores(scores_in, scores_out, log_scale=False)
# plt.subplot(122)
# plot_roc_curve(scores_in, scores_out)
# plt.tight_layout()
# plt.savefig("./OOD_vae_cifar10_texture_plot.png")


# print("scoring for OOD Data (Tiny) ... ")
# # scores_in, _ = vae.score(ds_in)
# ds_out = get_test_dataset_Tiny()
# for x, y in ds_out:
#     x = x.to(device)
#     y = y.to(device)

# scores_out, _ = vae.score(ds_out)
# # === metrics ===
# # auroc / fpr95
# metrics = bench_metrics(
#     (scores_in, scores_out),
#     metrics=["auroc", "fpr95tpr"],
# )
# print("=== Metrics ===")
# for k, v in metrics.items():
#     print(f"{k:<10} {v:.6f}")

# print("\n=== Plots ===")
# # hists / roc
# plt.figure(figsize=(13, 6))
# plt.subplot(121)
# plot_ood_scores(scores_in, scores_out, log_scale=False)
# plt.subplot(122)
# plot_roc_curve(scores_in, scores_out)
# plt.tight_layout()
# plt.savefig("./OOD_vae_cifar10_Tiny_plot.png")



# print("scoring for OOD Data (cifar100) ... ")
# # scores_in, _ = vae.score(ds_in)
# ds_out = get_test_dataset_cifar100()
# for x, y in ds_out:
#     x = x.to(device)
#     y = y.to(device)

# scores_out, _ = vae.score(ds_out)
# # === metrics ===
# # auroc / fpr95
# metrics = bench_metrics(
#     (scores_in, scores_out),
#     metrics=["auroc", "fpr95tpr"],
# )
# print("=== Metrics ===")
# for k, v in metrics.items():
#     print(f"{k:<10} {v:.6f}")

# print("\n=== Plots ===")
# # hists / roc
# plt.figure(figsize=(13, 6))
# plt.subplot(121)
# plot_ood_scores(scores_in, scores_out, log_scale=False)
# plt.subplot(122)
# plot_roc_curve(scores_in, scores_out)
# plt.tight_layout()
# plt.savefig("./OOD_vae_cifar10_cifar100_plot.png")



