import torch


class DifferentialPrivacy:
    """
    Basic Differential Privacy mechanism for
    federated model updates.

    Steps:
    1. Clip the complete update to max L2 norm.
    2. Add Gaussian noise.
    """

    def __init__(
        self,
        max_norm=1.0,
        noise_multiplier=0.01,
    ):
        self.max_norm = max_norm
        self.noise_multiplier = noise_multiplier

    def update_norm(self, update):
        """Calculate the global L2 norm across all tensors."""

        squared_norm = sum(
            torch.sum(parameter ** 2)
            for parameter in update
        )

        return torch.sqrt(squared_norm)

    def clip_update(self, update):
        """Clip the complete model update."""

        total_norm = self.update_norm(update)

        if total_norm > self.max_norm:

            scale = (
                self.max_norm
                / (total_norm + 1e-12)
            )

            update = [
                parameter * scale
                for parameter in update
            ]

        return update

    def add_noise(self, update):
        """Add Gaussian noise to each tensor."""

        protected_update = []

        noise_scale = (
            self.max_norm
            * self.noise_multiplier
        )

        for parameter in update:

            noise = torch.normal(
                mean=0.0,
                std=noise_scale,
                size=parameter.shape,
                device=parameter.device,
            )

            protected_update.append(
                parameter + noise
            )

        return protected_update

    def protect(self, update):
        """Apply clipping followed by Gaussian noise."""

        original_norm = self.update_norm(update)

        clipped = self.clip_update(update)

        clipped_norm = self.update_norm(clipped)

        protected = self.add_noise(clipped)

        noise = [
            protected_tensor - clipped_tensor
            for protected_tensor, clipped_tensor
            in zip(protected, clipped)
        ]

        noise_norm = self.update_norm(noise)

        print(
            f"DP original norm : {original_norm.item():.6f}"
        )

        print(
            f"DP clipped norm  : {clipped_norm.item():.6f}"
        )

        print(
            f"DP noise norm    : {noise_norm.item():.6f}"
        )

        return protected