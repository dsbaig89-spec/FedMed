import os
import numpy as np


# ==========================================
# FedMed Synthetic MRI Dataset Generator
# ==========================================

VOLUME_SIZE = 32
SAMPLES_PER_HOSPITAL = 20

HOSPITALS = {
    "hospital_a": SAMPLES_PER_HOSPITAL,
    "hospital_b": SAMPLES_PER_HOSPITAL,
    "hospital_c": SAMPLES_PER_HOSPITAL,
}


def generate_sample(volume_size=32):
    """
    Generate one synthetic 3D MRI volume
    and its corresponding tumor mask.
    """

    # Background MRI intensity
    volume = np.random.normal(
        loc=0.05,
        scale=0.02,
        size=(volume_size, volume_size, volume_size)
    ).astype(np.float32)

    # Empty tumor mask
    mask = np.zeros(
        (volume_size, volume_size, volume_size),
        dtype=np.float32
    )

    # --------------------------------------
    # Brain region
    # --------------------------------------

    center = np.array([
        volume_size // 2,
        volume_size // 2,
        volume_size // 2
    ])

    brain_radius = volume_size * 0.35

    # --------------------------------------
    # Random tumor location
    # --------------------------------------

    tumor_center = center + np.random.randint(
        -5,
        6,
        size=3
    )

    tumor_radius = np.random.randint(3, 6)

    # --------------------------------------
    # Generate 3D volume
    # --------------------------------------

    for x in range(volume_size):

        for y in range(volume_size):

            for z in range(volume_size):

                point = np.array([x, y, z])

                # Distance from brain center
                brain_distance = np.linalg.norm(
                    point - center
                )

                if brain_distance <= brain_radius:

                    # Brain tissue intensity
                    volume[x, y, z] += np.random.normal(
                        0.20,
                        0.03
                    )

                    # Distance from tumor center
                    tumor_distance = np.linalg.norm(
                        point - tumor_center
                    )

                    if tumor_distance <= tumor_radius:

                        # Tumor mask
                        mask[x, y, z] = 1.0

                        # Tumor has higher intensity
                        volume[x, y, z] += np.random.normal(
                            0.50,
                            0.05
                        )

    # --------------------------------------
    # Normalize MRI volume
    # --------------------------------------

    volume -= volume.min()

    if volume.max() > 0:
        volume /= volume.max()

    return (
        volume.astype(np.float32),
        mask.astype(np.float32)
    )


def generate_hospital_dataset(
    hospital_name,
    sample_count
):
    """
    Generate a private synthetic dataset
    for one hospital.
    """

    output_dir = os.path.join(
        "datasets",
        hospital_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print(
        f"\nGenerating {sample_count} samples "
        f"for {hospital_name}..."
    )

    for i in range(sample_count):

        image, mask = generate_sample(
            VOLUME_SIZE
        )

        image_path = os.path.join(
            output_dir,
            f"patient_{i:03d}_image.npy"
        )

        mask_path = os.path.join(
            output_dir,
            f"patient_{i:03d}_mask.npy"
        )

        np.save(
            image_path,
            image
        )

        np.save(
            mask_path,
            mask
        )

    print(
        f"Completed {hospital_name}: "
        f"{sample_count} samples"
    )


def main():

    print("=" * 60)
    print("FedMed Synthetic MRI Dataset Generator")
    print("=" * 60)

    print("\nWARNING:")
    print("This is SYNTHETIC demo data.")
    print("It is NOT real patient medical data.")

    for hospital, sample_count in HOSPITALS.items():

        generate_hospital_dataset(
            hospital,
            sample_count
        )

    print("\n" + "=" * 60)
    print("Dataset generation completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()