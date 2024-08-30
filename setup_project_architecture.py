import os

# Define the project structure again after reset
project_structure = {
    "OOD-Detection-Benchmarking": {
        "datasets": {
            "MNIST": [ "MNIST/", "Near-OOD/NotMNIST", "Near-OOD/FashionMNIST", "Far-OOD/Texture", "Far-OOD/CIFAR-10", "Far-OOD/TinyImageNet", "Far-OOD/Places365"],
            "CIFAR-10": ["CIFAR-10", "Near-OOD/CIFAR-100", "Near-OOD/TinyImageNet", "Far-OOD/MNIST", "Far-OOD/SVHN", "Far-OOD/Texture", "Far-OOD/Places365"],
            "CIFAR-100": ["CIFAR-100" ,"Near-OOD/CIFAR-10", "Near-OOD/TinyImageNet", "Far-OOD/MNIST", "Far-OOD/SVHN", "Far-OOD/Texture", "Far-OOD/Places365"],
            "ImageNet-200": ["ImageNet-200","Near-OOD/SSB-hard", "Near-OOD/NINCO", "Far-OOD/iNaturalist", "Far-OOD/Texture", "Far-OOD/OpenImage-O", "Covariate-Shifted ID/ImageNet-C", "Covariate-Shifted ID/ImageNet-R", "Covariate-Shifted ID/ImageNet-v2"],
            "ImageNet-1K": [ "ImageNet-1K", "Near-OOD/SSB-hard", "Near-OOD/NINCO", "Far-OOD/iNaturalist", "Far-OOD/Texture", "Far-OOD/OpenImage-O", "Covariate-Shifted ID/ImageNet-C", "Covariate-Shifted ID/ImageNet-R", "Covariate-Shifted ID/ImageNet-v2"]
        },
        "models": {},
        "src": {
            "data_preprocessing": {},
            "models": {},
            "nmf_mahalanobis": {},
            "pca_mahalanobis": {}
        },
        "results": {},
        "notebooks": {},
    }
}

# Function to create the directory structure
def create_dir_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        os.makedirs(path, exist_ok=True)
        if isinstance(content, dict):  # If the content is a dictionary, recurse
            create_dir_structure(path, content)
        elif isinstance(content, list):  # If the content is a list, create subdirectories
            for subname in content:
                subpath = os.path.join(path, subname)
                os.makedirs(subpath, exist_ok=True)

# Specify the base path for the project structure
base_path = './'

# Create the project structure
create_dir_structure(base_path, project_structure["OOD-Detection-Benchmarking"])

print("Project structure created successfully.")

