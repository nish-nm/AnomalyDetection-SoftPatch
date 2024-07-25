import os
import requests
import zipfile
from enum import Enum
import PIL
import torch
from torchvision import transforms
import numpy as np

_CLASSNAMES = [
    "Abyssinian", "American_Bulldog", "American_Pit_Bull_Terrier", "Basset_Hound",
    "Beagle", "Bengal", "Birman", "Bombay", "Boxer", "British_Shorthair", "Chihuahua",
    "Egyptian_Mau", "English_Cocker_Spaniel", "English_Setter", "German_Shorthaired",
    "Great_Pyrenees", "Havanese", "Japanese_Chin", "Keeshond", "Leonberger", "Maine_Coon",
    "Miniature_Pinscher", "Newfoundland", "Persian", "Pomeranian", "Pug", "Ragdoll",
    "Russian_Blue", "Saint_Bernard", "Samoyed", "Scottish_Terrier", "Shiba_Inu", "Siamese",
    "Sphynx", "Staffordshire_Bull_Terrier", "Wheaten_Terrier", "Yorkshire_Terrier"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class OxfordPetDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for the Oxford-IIIT Pet Dataset.
    """

    def __init__(
        self,
        source,
        classname=None,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=0.8,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the Oxford-IIIT Pet data folder.
            classname: [str or None]. Name of class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training, validation, or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. DatasetSplit.TRAIN.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.download_and_extract_dataset()

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.imagesize = (3, imagesize, imagesize)
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN

    def download_and_extract_dataset(self):
        # Download and extract images
        self._download_and_extract(DATASET_URL, os.path.join(self.source, "images.tar.gz"), self.source)
        
        # Download and extract annotations
        self._download_and_extract(ANNOTATIONS_URL, os.path.join(self.source, "annotations.tar.gz"), self.source)
        
    def _download_and_extract(self, url, dest_path, extract_to):
        if not os.path.exists(dest_path.replace(".tar.gz", "")):
            print(f"Downloading {url} to {dest_path}")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Extracting {dest_path}")
            with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            os.remove(dest_path)

    def __getitem__(self, idx):
        classname, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_img(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
            # "mask_path": mask_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, "images", classname)
            maskpath = os.path.join(self.source, "annotations", "trimaps", classname)

            # Create directories if they do not exist
            if not os.path.exists(classpath):
                os.makedirs(classpath)
            if not os.path.exists(maskpath):
                os.makedirs(maskpath)

            # Debugging: Print the paths
            print(f"Classpath: {classpath}")
            print(f"Maskpath: {maskpath}")

            image_files = sorted(os.listdir(classpath))

            if self.split == DatasetSplit.TRAIN or self.split == DatasetSplit.VAL:
                split_idx = int(len(image_files) * self.train_val_split)
                if self.split == DatasetSplit.TRAIN:
                    image_files = image_files[:split_idx]
                else:
                    image_files = image_files[split_idx:]

            imgpaths_per_class[classname] = [
                os.path.join(classpath, x) for x in image_files
            ]
            maskpaths_per_class[classname] = [
                os.path.join(maskpath, x.replace(".jpg", ".png")) for x in image_files
            ]

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for i, image_path in enumerate(imgpaths_per_class[classname]):
                mask_path = maskpaths_per_class[classname][i]
                data_tuple = [classname, image_path, mask_path]
                data_to_iterate.append(data_tuple)
        data_to_iterate = np.array(data_to_iterate)
        return imgpaths_per_class, data_to_iterate

