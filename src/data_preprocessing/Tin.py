import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class Tiny(Dataset):
    def __init__(self, train=False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (should contain 'train' and 'test' subdirectories).
            train (bool, optional): If True, creates dataset from 'train' folder, otherwise from 'test' folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = "../../data/tin"
        self.train = train
        self.transform = transform
        self.data_dir = os.path.join(self.root_dir, 'train' if self.train else 'test')
        self.classes = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)) and d != '.DS_Store' and d != 'imglist.txt']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.image_labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_dir, cls_name + "/images")
            for img_name in os.listdir(cls_dir):
                if img_name.split(".")[1] not in ["jpg", "png", "jpeg", "webp", "JPEG"] or img_name == ".DS_Store":
                    continue
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.image_labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.image_labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

transform = transforms.Compose([
    # ResNet models expect 3-channel images, but Tiny is already in this format
    transforms.Resize((64, 64)),  # Ensuring the image size is 32x32
    transforms.RandomHorizontalFlip(),  # A common augmentation for image data
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    # Normalize each channel of the Tiny images using mean and std
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
 
])

def get_train_dataset_Tiny():
    # Initialize the inaturalist datasets for training and testing
    train_dataset = Tiny( train=True, transform=transform)
    print("the length of the Tiny Training dataset : ", len(train_dataset))
    # Create the DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    return train_loader

def get_test_dataset_Tiny():
    # Initialize the inaturalist datasets for training and testing
    test_dataset = Tiny( train=False, transform=transform)
    print("the length of the Tiny Test dataset : ", len(test_dataset))
    # Create the DataLoaders for training and testing
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)
    return test_loader

# train_loader = get_train_dataset_Tiny()



