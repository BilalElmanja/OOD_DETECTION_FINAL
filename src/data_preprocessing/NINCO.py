import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

class NINCO(Dataset):
    def __init__(self, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (should contain 'train' and 'test' subdirectories).
            train (bool, optional): If True, creates dataset from 'train' folder, otherwise from 'test' folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = "../../data/NINCO"
        self.train = train
        self.transform = transform
        self.data_dir = os.path.join(self.root_dir, 'train' if self.train else '')
        self.classes = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)) and d != '.DS_Store']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.image_labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.split(".")[1] not in ["jpg", "png", "jpeg", "webp"] or img_name == ".DS_Store":
                    continue
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.image_labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return 900 #len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.image_labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

transform = transforms.Compose([
    # ResNet models expect 3-channel images, but NINCO is already in this format
    transforms.Resize((224, 224)),  # Ensuring the image size is 32x32
    # transforms.RandomHorizontalFlip(),  # A common augmentation for image data
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    # Normalize each channel of the NINCO images using mean and std
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()
 
])

def get_train_dataset_NINCO():
    # Initialize the NINCO datasets for training and testing
    train_dataset = NINCO( train=True, transform=transform)
    print("the length of the NINCO Training dataset : ", len(train_dataset))
    # Create the DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    return train_loader

def get_test_dataset_NINCO():
    # Initialize the CIFAR10 datasets for training and testing
    test_dataset = NINCO( train=False, transform=transform)
    print("the length of the NINCO Test dataset : ", len(test_dataset))
    # Create the DataLoaders for training and testing
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
    return test_loader




