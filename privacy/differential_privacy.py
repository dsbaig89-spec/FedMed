import torch


class DifferentialPrivacy:
    """
    Basic Differential Privacy mechanism for
    federated model updates.

    Steps:
    1. Clip the update norm.
    2. Add Gaussian noise.
    """

    def __init__(
        self,
        max_norm=1.0,
        noise_multiplier=0.01,
    ):
        self.max_norm = max_norm
        self.noise_multiplier = noise_multiplier

    def clip_update(self, update):
        """
        Clip a model update to the configured
        maximum L2 norm.
        """

        total_norm = torch.sqrt(
            sum(
                torch.sum(
                    parameter ** 2
                )
                for parameter in update
            )
        )

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
        """
        Add Gaussian noise to each tensor.
        """

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
        """
        Apply clipping followed by noise.
        """

        clipped = self.clip_update(
            update
        )

        protected = self.add_noise(
            clipped
        )

        return protected