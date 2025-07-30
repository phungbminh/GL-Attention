import os
import random
import zipfile

import gdown
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def denormalize(image_tensor: torch.Tensor, mean: list, std: list) -> np.ndarray:
    """
    Denormalize a tensor image and convert to numpy array for plotting.
    """
    img = image_tensor.clone()
    for channel, m, s in zip(img, mean, std):
        channel.mul_(s).add_(m)
    array = img.numpy().transpose(1, 2, 0)
    return np.clip(array, 0, 1)


class SubsetMemDataset(Dataset):
    """Subset dataset for train/val splits with parent reference"""
    def __init__(self, imgs, lbls, indices, tf, parent_dataset=None):
        self.imgs = imgs
        self.lbls = lbls
        self.indices = indices
        self.tf = tf
        self.num_classes = len(set(self.lbls))
        self.parent_dataset = parent_dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img = self.imgs[self.indices[i]]
        lbl = self.lbls[self.indices[i]]
        return self.tf(img), lbl
    
    def get_sampler(self, method='sqrt'):
        """Create WeightedRandomSampler for this subset"""
        if self.parent_dataset is None:
            raise ValueError("parent_dataset required for sampler creation")
        
        weights = self.parent_dataset.get_class_weights(self.indices, method)
        sample_weights = [weights[self.lbls[i]] for i in self.indices]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# ----------------------- Dataset Downloader -----------------------
class DatasetDownloader:
    """
    Download and extract medical imaging datasets from Google Drive.
    Currently supports HAM10000 and ISIC-2018-Task-3 datasets.
    """
    DRIVE_IDS = {
        "HAM10000": "1YgtSWc2tPP0qHIV-hf1qpJeLXAkmeV3O",
        "ISIC2018": "1G5xrbsVC-saor6LOmPJLePDIkO62YN1j",
    }

    def __init__(self, name: str, output_dir: str):
        self.name = name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.zip_path = os.path.join(output_dir, f"{name}.zip")

    def download_and_extract(self):
        drive_id = self.DRIVE_IDS.get(self.name)
        if not drive_id:
            return

        if not os.path.exists( os.path.join(self.output_dir, self.name)):
            if not os.path.exists(self.zip_path):
                url = f"https://drive.google.com/uc?id={drive_id}"
                gdown.download(url, self.zip_path, quiet=False)

            print(f"Extracting {self.name}...")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                for member in tqdm(zip_ref.namelist(), desc=f"Extracting {self.name}", unit="file"):
                    # skip macOS metadata
                    if '__MACOSX' in member:
                        continue
                    # extract actual dataset files
                    zip_ref.extract(member, self.output_dir)
        else:
            print('Dataset đã tồn tại')


# ----------------------- Medical Dataset Class -----------------------
class GeneralDataset(Dataset):
    """
    Medical imaging dataset wrapper for HAM10000 and ISIC-2018-Task-3.
    Provides stratified train/validation split for skin lesion classification.
    """

    def __init__(
            self,
            name: str,
            root: str,
    ):
        self.name = name
        self.root = root

        # Download dataset if needed
        if name in DatasetDownloader.DRIVE_IDS:
            DatasetDownloader(name, root).download_and_extract()

        # Load all images and labels to memory
        self.images, self.labels = self._load_data()
        assert self.images, f"No data for {name} in {root}"

        self.num_classes = len(set(self.labels))

    def get_splits(
            self,
            val_size: float = 0.2,
            seed: int = 42,
            image_size: int = 224
    ):
        """
        Trả về (train_dataset, val_dataset) sau stratified split và transform.
        """
        labels = np.array(self.labels)
        idx = np.arange(len(labels))
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=seed
        )
        train_idx, val_idx = next(splitter.split(idx, labels))

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # train_tf = transforms.Compose([
        #     transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomRotation(10),
        #     transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
        #     transforms.RandomPerspective(0.2, p=0.2),
        #     transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
        #     transforms.RandAugment(num_ops=2, magnitude=9),
        #     transforms.ToTensor(),
        #     normalize,
        #     transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        # ])
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAffine(translate=(0.1, 0.1), degrees=15),
            transforms.ToTensor(),
            normalize
        ])
        val_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


        train_ds = SubsetMemDataset(self.images, self.labels, train_idx, train_tf, self)
        val_ds = SubsetMemDataset(self.images, self.labels, val_idx, val_tf, self)

        return train_ds, val_ds

    def _load_data(self):
        """
        Load medical imaging dataset based on dataset name.
        Supports HAM10000 and ISIC-2018-Task-3.
        """
        if self.name == 'HAM10000':
            return self._load_ham10000()
        elif self.name == 'isic-2018-task-3':
            return self._load_isic2018()
        else:
            raise ValueError(f"Unsupported dataset: {self.name}. Only 'HAM10000' and 'isic-2018-task-3' are supported.")


    def _load_ham10000(self):
        """
        Load HAM10000 dataset for skin lesion classification.
        
        HAM10000 contains 10,015 dermatoscopic images in 7 categories:
        - akiec: Actinic keratoses and intraepithelial carcinoma
        - bcc: Basal cell carcinoma  
        - bkl: Benign keratosis-like lesions
        - df: Dermatofibroma
        - mel: Melanoma
        - nv: Melanocytic nevi
        - vasc: Pyogenic granulomas and hemorrhage
        """
        candidates = [
            os.path.join(self.root, 'HAM10000_metadata.csv'),
            os.path.join(self.root, 'HAM10000', 'HAM10000_metadata.csv'),
        ]
        meta_path = next((p for p in candidates if os.path.exists(p)), None)
        assert meta_path, f'HAM10000 metadata CSV not found in {self.root}'

        base_dir = os.path.dirname(meta_path)
        df = pd.read_csv(meta_path)
        
        # Create consistent label mapping
        class_names = sorted(df['dx'].unique())
        label_map = {dx: idx for idx, dx in enumerate(class_names)}
        self.class_names = class_names

        # Search for image folders
        dirs = [
            os.path.join(base_dir, 'HAM10000_images_part_1'),
            os.path.join(base_dir, 'HAM10000_images_part_2'),
            base_dir,
        ]

        images, labels = [], []
        missing_count = 0
        
        print(f"Loading {len(df)} HAM10000 images...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading HAM10000 images"):
            filename = f"{row['image_id']}.jpg"
            found = False
            
            for d in dirs:
                path = os.path.join(d, filename)
                if os.path.exists(path):
                    try:
                        with Image.open(path) as img:
                            images.append(img.convert('RGB'))
                        labels.append(label_map[row['dx']])
                        found = True
                        break
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
                        continue
            
            if not found:
                missing_count += 1
        
        if missing_count > 0:
            print(f"Warning: {missing_count} images not found in HAM10000 dataset")
        
        print(f"Loaded HAM10000: {len(images)} images, {len(class_names)} classes")
        print(f"Classes: {class_names}")
        
        return images, labels

    def _load_isic2018(self):
        """
        Load ISIC 2018 Task 3 dataset for skin lesion classification.
        
        ISIC 2018 Task 3 contains 10,015 dermoscopy images in 7 categories:
        - MEL: Melanoma
        - NV: Melanocytic nevus
        - BCC: Basal cell carcinoma
        - AKIEC: Actinic keratosis / Bowen's disease (intraepithelial carcinoma)
        - BKL: Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
        - DF: Dermatofibroma
        - VASC: Vascular lesion
        """
        # Find metadata CSV
        meta_path = None
        for rd, _, files in os.walk(self.root):
            if 'ISIC2018_Task3_Training_GroundTruth.csv' in files:
                meta_path = os.path.join(rd, 'ISIC2018_Task3_Training_GroundTruth.csv')
                break
        assert meta_path, f"ISIC2018 metadata CSV not found under {self.root}"
        
        df = pd.read_csv(meta_path)
        cols = df.columns[1:]  # Class columns (exclude 'image' column)
        self.class_names = list(cols)

        # Find image folder
        img_dir = None
        for rd, dirs, _ in os.walk(self.root):
            if 'ISIC2018_Task3_Training_Input' in dirs:
                img_dir = os.path.join(rd, 'ISIC2018_Task3_Training_Input')
                break
        assert img_dir, f"ISIC2018 images folder not found under {self.root}"

        # Load images and labels
        images, labels = [], []
        missing_count = 0
        
        print(f"Loading {len(df)} ISIC2018 images...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading ISIC2018 images"):
            img_path = os.path.join(img_dir, f"{row['image']}.jpg")
            
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        images.append(img.convert('RGB'))
                    
                    # Convert one-hot encoded labels to class index
                    one_hot = row[cols].values.astype(np.float32)
                    label_idx = int(np.argmax(one_hot))
                    labels.append(label_idx)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    missing_count += 1
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"Warning: {missing_count} images not found in ISIC2018 dataset")
        
        print(f"Loaded ISIC2018: {len(images)} images, {len(self.class_names)} classes")
        print(f"Classes: {self.class_names}")
        
        return images, labels

    def get_class_weights(self, subset_indices=None, method='sqrt'):
        """
        Calculate class weights for handling imbalance.
        Args:
            subset_indices: Indices for subset. If None, use all data.
            method: 'inverse', 'sqrt', or 'log' for different weighting strategies
        """
        if subset_indices is None:
            subset_indices = list(range(len(self.labels)))
        
        subset_labels = [self.labels[i] for i in subset_indices]
        counts = np.bincount(subset_labels, minlength=self.num_classes)
        counts = np.maximum(counts, 1)  # Avoid division by zero
        
        if method == 'inverse':
            weights = 1.0 / counts
        elif method == 'sqrt':
            weights = np.sqrt(1.0 / counts)
        elif method == 'log':
            weights = 1.0 / np.log(counts + 1)
        else:
            raise ValueError(f"Unknown weighting method: {method}")
            
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        return weights

def plot_random_images(
    dataset,
    num_images: int = 5,
    mean: list = [0.485, 0.456, 0.406],
    std: list  = [0.229, 0.224, 0.225]
):
    """
    Vẽ ngẫu nhiên `num_images` ảnh từ `dataset` (PyTorch Dataset trả về (img_tensor, label)).
    """
    N = len(dataset)
    num_images = min(num_images, N)
    idxs = random.sample(range(N), num_images)

    cols = min(5, num_images)
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten()

    for ax, i in zip(axes, idxs):
        img_t, lbl = dataset[i]
        img = denormalize(img_t, mean, std)
        ax.imshow(img)
        ax.set_title(f"Label: {lbl}")
        ax.axis('off')

    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_label_histogram(
    train_dataset,
    val_dataset,
    num_classes: int,
    title_train: str = 'Train Label Distribution',
    title_val: str = 'Validation Label Distribution',
    bar_width: float = 0.8
):
    """
    Vẽ histogram phân bố nhãn của train và validation datasets.
    Luôn hiển thị đủ cột cho tất cả nhãn; nếu số lớp > 15, không hiển thị tên nhãn (để tránh quá dày).
    """
    train_labels = [label for _, label in train_dataset]
    val_labels = [label for _, label in val_dataset]

    x = np.arange(num_classes)
    train_counts = np.bincount(train_labels, minlength=num_classes)
    val_counts = np.bincount(val_labels, minlength=num_classes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Train
    axes[0].bar(x, train_counts, width=bar_width, edgecolor='black')
    axes[0].set_title(title_train)
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks(x)
    if num_classes <= 15:
        axes[0].set_xticklabels(x)
    else:
        axes[0].set_xticklabels([''] * num_classes)

    # Validation
    axes[1].bar(x, val_counts, width=bar_width, edgecolor='black')
    axes[1].set_title(title_val)
    axes[1].set_xlabel('Label')
    axes[1].set_ylabel('Count')
    axes[1].set_xticks(x)
    if num_classes <= 15:
        axes[1].set_xticklabels(x)
    else:
        axes[1].set_xticklabels([''] * num_classes)

    plt.tight_layout()
    plt.show()

# ----------------------- Usage Example -----------------------
if __name__ == '__main__':
    # Example with HAM10000 dataset
    print("Testing HAM10000 dataset:")
    dataset = GeneralDataset('HAM10000', './data')
    train_dataset, val_dataset = dataset.get_splits(val_size=0.2, seed=42, image_size=224)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Class names: {getattr(dataset, 'class_names', 'Not available')}")

    # Sample data
    img, label = val_dataset[0]
    print(f"Sample image shape: {img.shape}, label: {label}")
    
    # Visualization
    print("\nPlotting sample images...")
    plot_random_images(val_dataset, num_images=9)
    
    print("\nPlotting label distribution...")
    plot_label_histogram(train_dataset, val_dataset, dataset.num_classes)
    
    print("\n" + "="*50)
    
    # Example with ISIC2018 dataset
    print("Testing ISIC2018 dataset:")
    try:
        dataset2 = GeneralDataset('isic-2018-task-3', './data')
        train_dataset2, val_dataset2 = dataset2.get_splits(val_size=0.2, seed=42, image_size=224)
        
        print(f"Train dataset size: {len(train_dataset2)}")
        print(f"Validation dataset size: {len(val_dataset2)}")
        print(f"Number of classes: {dataset2.num_classes}")
        print(f"Class names: {getattr(dataset2, 'class_names', 'Not available')}")
        
    except Exception as e:
        print(f"ISIC2018 dataset not available: {e}")
