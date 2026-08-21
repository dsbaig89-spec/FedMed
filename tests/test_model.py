import torch

from models.unet import UNet3D
from models.dataset import MRISegmentationDataset


def main():

    print("=" * 50)
    print("FedMed Model Integration Test")
    print("=" * 50)

    # Load dataset
    dataset = MRISegmentationDataset(
        "datasets/hospital_a"
    )

    image, mask = dataset[0]

    # Add batch dimension
    image = image.unsqueeze(0)
    mask = mask.unsqueeze(0)

    print("\nInput MRI :", image.shape)
    print("Target mask:", mask.shape)

    # Create model
    model = UNet3D()

    # Forward pass
    with torch.no_grad():

        prediction = model(image)

    print(
        "Prediction:",
        prediction.shape
    )

    print("\nModel integration successful!")


if __name__ == "__main__":
    main()