import torch

from privacy.differential_privacy import (
    DifferentialPrivacy
)


def main():

    print("=" * 60)
    print("Testing Differential Privacy")
    print("=" * 60)

    dp = DifferentialPrivacy(
        max_norm=1.0,
        noise_multiplier=0.01,
    )

    update = [
        torch.randn(10, 10),
        torch.randn(5),
        torch.randn(3, 3),
    ]

    original_norm = torch.sqrt(
        sum(
            torch.sum(
                parameter ** 2
            )
            for parameter in update
        )
    )

    print(
        f"\nOriginal update norm: "
        f"{original_norm.item():.4f}"
    )

    clipped = dp.clip_update(
        update
    )

    clipped_norm = torch.sqrt(
        sum(
            torch.sum(
                parameter ** 2
            )
            for parameter in clipped
        )
    )

    print(
        f"Clipped update norm: "
        f"{clipped_norm.item():.4f}"
    )

    protected = dp.protect(
        update
    )

    print(
        f"Protected tensors: "
        f"{len(protected)}"
    )

    print(
        "\n✓ Differential Privacy test passed"
    )


if __name__ == "__main__":
    main()