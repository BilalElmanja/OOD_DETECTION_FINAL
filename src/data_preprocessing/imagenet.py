import os
import torch
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class ImageNet_1K(Dataset):
    def __init__(self, train=False, transform=None, class_id=2, train_active=True):
        """
        Args:
            root_dir (string): Directory with all the images (should contain 'train' and 'test' subdirectories).
            train (bool, optional): If True, creates dataset from 'train' folder, otherwise from 'test' folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = "../../data/imagenet_1k"
        self.train = train
        self.transform = transform
        self.max_samples = 300
        self.data_dir = os.path.join(self.root_dir, 'train' if self.train else 'val')
        self.classes = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)) and d != '.DS_Store' and d != 'imglist.txt']
        # print(self.classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.image_labels = []
        
        for cls_name in self.classes[:class_id]:
            cls_dir = os.path.join(self.data_dir, cls_name)
            if not self.train:
                cls_dir = self.data_dir
                self.max_samples = 75
            if train_active:
                for img_name in os.listdir(cls_dir)[:self.max_samples]:
                    if img_name.split(".")[1] not in ["jpg", "png", "jpeg", "webp", "JPEG"] or img_name == ".DS_Store":
                        continue
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.image_labels.append(self.class_to_idx[cls_name])
            else:
                for img_name in os.listdir(cls_dir)[self.max_samples:self.max_samples + 300 ]:
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
    

class ImageNet_1K_OOD(Dataset):
    def __init__(self, train=False, transform=None, train_active=True):
        """
        Args:
            root_dir (string): Directory with all the images (should contain 'train' and 'test' subdirectories).
            train (bool, optional): If True, creates dataset from 'train' folder, otherwise from 'test' folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = "../../data/imagenet_1k"
        self.train = train
        self.transform = transform
        self.max_samples = 300
        self.data_dir = os.path.join(self.root_dir, 'train' if self.train else 'val')
        self.classes = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)) and d != '.DS_Store' and d != 'imglist.txt']
        # print(self.classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.image_labels = []
        
        for cls_name in self.classes[10:13]:
            cls_dir = os.path.join(self.data_dir, cls_name)
            if not self.train:
                cls_dir = self.data_dir
                self.max_samples = 25
            if train_active:
                for img_name in os.listdir(cls_dir)[:self.max_samples]:
                    if img_name.split(".")[1] not in ["jpg", "png", "jpeg", "webp", "JPEG"] or img_name == ".DS_Store":
                        continue
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.image_labels.append(self.class_to_idx[cls_name])
            else:
                for img_name in os.listdir(cls_dir)[self.max_samples:self.max_samples + 200 ]:
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
    # ResNet models expect 3-channel images, but ImageNet_V2 is already in this format
    transforms.Resize((224, 224)),  # Ensuring the image size is 32x32
    # transforms.RandomHorizontalFlip(),  # A common augmentation for image data
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    # Normalize each channel of the ImageNet_V2 images using mean and std
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()
 
])

def get_train_dataset_ImageNet_1K(class_id=4):
    # Initialize the ImageNet_1K datasets for training and testing
    train_dataset = ImageNet_1K( train=True, transform=transform, class_id=class_id)
    print("the length of the ImageNet_1K Training dataset : ", len(train_dataset))
    # Create the DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    return train_loader

def get_test_dataset_ImageNet_1K(class_id =4, train_active=False):
    # Initialize the ImageNet_1K datasets for training and testing
    test_dataset = ImageNet_1K( train=True, transform=transform, class_id=class_id, train_active=train_active)
    print("the length of the ImageNet_1K Test dataset (ID) : ", len(test_dataset))
    # Create the DataLoaders for training and testing
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    return test_loader

def get_test_dataset_ImageNet_1K_OOD( train_active=True):
    # Initialize the ImageNet_1K datasets for training and testing
    test_dataset = ImageNet_1K_OOD( train=True, transform=transform,  train_active=train_active)
    print("the length of the ImageNet_1K Test dataset (OOD) : ", len(test_dataset))
    # Create the DataLoaders for training and testing
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    return test_loader

# train_loader = get_train_dataset_ImageNet_1K()
# test_loader = get_test_dataset_ImageNet_1K()




