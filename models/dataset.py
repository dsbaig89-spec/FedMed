import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MRISegmentationDataset(Dataset):
    """
    Dataset loader for FedMed synthetic MRI data.
    """

    def __init__(self, data_dir):

        self.data_dir = data_dir

        self.samples = []

        image_files = sorted([
            file
            for file in os.listdir(data_dir)
            if file.endswith("_image.npy")
        ])

        for image_file in image_files:

            mask_file = image_file.replace(
                "_image.npy",
                "_mask.npy"
            )

            image_path = os.path.join(
                data_dir,
                image_file
            )

            mask_path = os.path.join(
                data_dir,
                mask_file
            )

            if os.path.exists(mask_path):

                self.samples.append(
                    (image_path, mask_path)
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        image = np.load(image_path)

        mask = np.load(mask_path)

        # Add channel dimension
        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        return image.float(), mask.float()


if __name__ == "__main__":

    dataset = MRISegmentationDataset(
        "datasets/hospital_a"
    )

    print("Dataset size:", len(dataset))

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape :", mask.shape)

    print("\nDataset loader test successful!")