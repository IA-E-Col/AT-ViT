import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import io
from tqdm import tqdm
from PIL import ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PlantTraitDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, subset='train'):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with 'code' and target variable columns.
            img_dir (str): Directory containing images.
            transform (callable, optional): Transforms to apply to images.
            subset (str): 'train' or 'test' to filter the dataframe.
        """
        self.dataframe = dataframe[dataframe['train_test_set'] == subset].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_variable = os.getenv("TARGET_VARIABLE", "feuille_base_aigue")

        # Get list of unique class labels and create mapping
        self.classes = sorted(self.dataframe[self.target_variable].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Validate images and filter out corrupt ones
        self._validate_images()

        print(f"Loaded {subset} dataset with {len(self.dataframe)} samples and {len(self.classes)} classes")

    def _validate_images(self):
        """Check for corrupt or problematic images and filter them out."""
        valid_indices = []
        corrupt_files = []

        print("Validating image files...")
        for idx, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe)):
            code = row['code']
            img_path = os.path.join(self.img_dir, f"{code}.jpg")
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_indices.append(idx)
            except (IOError, OSError, Image.DecompressionBombError) as e:
                corrupt_files.append((code, str(e)))

        if corrupt_files:
            print(f"Found {len(corrupt_files)} corrupt or problematic image files:")
            for code, err in corrupt_files[:10]:
                print(f"- {code}: {err}")
            if len(corrupt_files) > 10:
                print(f"  (and {len(corrupt_files) - 10} more)")

            results_dir = os.getenv("RESULTS_DIR", "results")
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, 'corrupt_images.txt'), 'w') as f:
                for code, err in corrupt_files:
                    f.write(f"{code}: {err}\n")

            self.dataframe = self.dataframe.iloc[valid_indices].reset_index(drop=True)
            print(f"Filtered dataset to {len(self.dataframe)} valid samples")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        code = self.dataframe.loc[idx, 'code']
        label = self.dataframe.loc[idx, self.target_variable]
        label_idx = self.class_to_idx[label]

        img_path = os.path.join(self.img_dir, f"{code}.jpg")
        try:
            with open(img_path, 'rb') as f:
                image_data = f.read()
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
        except (IOError, OSError) as e:
            results_dir = os.getenv("RESULTS_DIR", "results")
            with open(os.path.join(results_dir, 'corrupted_images.txt'), 'a') as f:
                f.write(f"{img_path}: {str(e)}\n")
            print(f"Warning: Failed to load image: {img_path} ({str(e)})")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            label_idx = self.class_to_idx[self.classes[0]]  # Default to first class

        if self.transform:
            image = self.transform(image)

        return image, label_idx, code