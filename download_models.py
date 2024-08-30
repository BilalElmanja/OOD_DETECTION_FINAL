import gdown
import os
import zipfile

# Dictionnaire des IDs de téléchargement pour les checkpoints
download_id_dict = {
    'imagenet_res50_v1.5': '15PdDMNRfnJ7f2oxW6lI-Ge4QJJH3Z0Fy',
    'imagenet200_res18_v1.5': '1ddVmwc8zmzSjdLUO84EuV4Gz1c7vhIAs',
    'cifar100_res18_v1.5': '1s-1oNrRtmA0pGefxXJOUVRYpaoAML0C-',
    'cifar10_res18_v1.5': '1byGeYxM_PlLjT72wZsMQvP6popJeWBgt',
    'imagenet_res50_v1.5': '1tgY_PsfkazLDyI1pniDMDEehntBhFyF3',
    'imagenet_1k': '1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj',
}

output_dir_dict = {
    'imagenet200_res18_v1.5' : './models/ImageNet-200/',
    'imagenet_res50_v1.5': './models/ImageNet-1K/',
    'cifar100_res18_v1.5': './models/CIFAR-100/',
    'cifar10_res18_v1.5': './models/CIFAR-10/',
    'imagenet_1k' : './data/'

}

def download_checkpoint(checkpoint_name, save_dir):
    """ Télécharge et extrait le checkpoint spécifié. """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_id = download_id_dict[checkpoint_name]
    output_path = os.path.join(save_dir, checkpoint_name + '.zip')
    
    # Télécharger le fichier
    gdown.download(id=file_id, output=output_path, quiet=False)
    
    # Extraire le fichier zip
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    
    # Supprimer le fichier zip
    os.remove(output_path)
    print(f"{checkpoint_name} donwloaded and saved at : {save_dir}")

# Liste des checkpoints à télécharger
checkpoints_to_download = [
    # 'imagenet_1k',
    # 'imagenet_res50_v1.5',
    # 'imagenet200_res18_v1.5',
    # 'cifar100_res18_v1.5',
    # 'cifar10_res18_v1.5',
    'imagenet_res50_v1.5'
]

# Télécharger les checkpoints
for checkpoint in checkpoints_to_download:
    download_checkpoint(checkpoint, output_dir_dict[checkpoint])
