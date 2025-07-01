from collections.abc import Iterator
from typing import Generic, Literal, TypeVar

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

Q = TypeVar("Q")


class DatasetGeneratedDataLoader(DataLoader[Q], Generic[Q]):
    """DataLoader that generates batches by calling the dataset's `generate_batch` method."""

    def __init__(
        self,
        dataset: Dataset[Q],
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        # assert that dataset has a generate_batch method
        assert hasattr(dataset, "generate_batch")
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __iter__(  # type: ignore
        self,
    ) -> Iterator[Q]:
        for _ in range(len(self)):
            yield self.dataset.generate_batch(self.batch_size)  # type: ignore


class BatchedDataLoader(DataLoader[Q], Generic[Q]):
    """DataLoader that unpacks the batch in __getitem__.

    This is used for datasets which generate a whole batch in one call to __getitem__.
    """

    def __init__(
        self,
        dataset: Dataset[Q],
        num_workers: int = 0,
    ):
        super().__init__(dataset, num_workers=num_workers)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:  # type: ignore
        for batch, label in super().__iter__():
            yield batch[0], label[0]


DataGenerationType = Literal[
    "exactly_one_active",
    "exactly_two_active",
    "exactly_three_active",
    "exactly_four_active",
    "exactly_five_active",
    "at_least_zero_active",
]


class SparseFeatureDataset(
    Dataset[
        tuple[
            Float[Tensor, "batch n_features"],
            Float[Tensor, "batch n_features"],
        ]
    ]
):
    def __init__(
        self,
        n_features: int,
        feature_probability: float,
        device: str,
        data_generation_type: DataGenerationType = "at_least_zero_active",
        value_range: tuple[float, float] = (0.0, 1.0),
        synced_inputs: list[list[int]] | None = None,
    ):
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device
        self.data_generation_type = data_generation_type
        self.value_range = value_range
        self.synced_inputs = synced_inputs

    def __len__(self) -> int:
        return 2**31

    def sync_inputs(
        self, batch: Float[Tensor, "batch n_features"]
    ) -> Float[Tensor, "batch n_features"]:
        assert self.synced_inputs is not None
        all_indices = [item for sublist in self.synced_inputs for item in sublist]
        assert len(all_indices) == len(set(all_indices)), "Synced inputs must be non-overlapping"
        for indices in self.synced_inputs:
            mask = torch.zeros_like(batch, dtype=torch.bool)
            # First, get the samples for which there is a non-zero value for any of the indices
            non_zero_samples = (batch[..., indices] != 0.0).any(dim=-1)
            for idx in indices:
                mask[..., idx] = non_zero_samples
            # Now generate random values in value_range and apply them to the masked elements
            max_val, min_val = self.value_range
            random_values = torch.rand(batch.shape[0], self.n_features, device=self.device)
            random_values = random_values * (max_val - min_val) + min_val
            batch = torch.where(mask, random_values, batch)
        return batch

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch n_features"]]:
        # TODO: This is a hack to keep backward compatibility. Probably best to have
        # data_generation_type: Literal["exactly_n_active", "at_least_zero_active"] and
        # data_generation_n: PositiveInt
        number_map = {
            "exactly_one_active": 1,
            "exactly_two_active": 2,
            "exactly_three_active": 3,
            "exactly_four_active": 4,
            "exactly_five_active": 5,
        }
        if self.data_generation_type in number_map:
            n = number_map[self.data_generation_type]
            batch = self._generate_n_feature_active_batch(batch_size, n=n)
        elif self.data_generation_type == "at_least_zero_active":
            batch = self._masked_batch_generator(batch_size)
            if self.synced_inputs is not None:
                batch = self.sync_inputs(batch)
        else:
            raise ValueError(f"Invalid generation type: {self.data_generation_type}")

        return batch, batch.clone().detach()

    def _generate_n_feature_active_batch(
        self, batch_size: int, n: int
    ) -> Float[Tensor, "batch n_features"]:
        """Generate a batch with exactly n features active per sample.

        Args:
            batch_size: Number of samples in the batch
            n: Number of features to activate per sample
        """
        if n > self.n_features:
            raise ValueError(
                f"Cannot activate {n} features when only {self.n_features} features exist"
            )

        batch = torch.zeros(batch_size, self.n_features, device=self.device)

        # Create indices for all features
        feature_indices = torch.arange(self.n_features, device=self.device)
        # Expand to batch size
        feature_indices = feature_indices.expand(batch_size, self.n_features)

        # For each instance in the batch, randomly permute the features
        perm = torch.rand_like(feature_indices.float()).argsort(dim=-1)
        permuted_features = feature_indices.gather(dim=-1, index=perm)

        # Take first n indices for each instance - guaranteed no duplicates
        active_features = permuted_features[..., :n]

        # Generate random values in value_range for the active features
        min_val, max_val = self.value_range
        random_values = torch.rand(batch_size, n, device=self.device)
        random_values = random_values * (max_val - min_val) + min_val

        # Place each active feature
        for i in range(n):
            batch.scatter_(
                dim=1, index=active_features[..., i : i + 1], src=random_values[..., i : i + 1]
            )

        return batch

    def _masked_batch_generator(self, batch_size: int) -> Float[Tensor, "batch_size n_features"]:
        """Generate a batch where each feature activates independently with probability
        `feature_probability`.
        """
        min_val, max_val = self.value_range
        batch = (
            torch.rand((batch_size, self.n_features), device=self.device) * (max_val - min_val)
            + min_val
        )
        mask = torch.rand_like(batch) < self.feature_probability
        return batch * mask

    def _generate_multi_feature_batch_no_zero_samples(
        self, batch_size: int, buffer_ratio: float
    ) -> Float[Tensor, "batch n_features"]:
        """Generate a batch where each feature activates independently with probability
        `feature_probability`.

        Ensures that there are no zero samples in the batch.

        Args:
            batch_size: Number of samples in the batch
            buffer_ratio: First generate `buffer_ratio * batch_size` samples and count the
                number of samples with all zeros. Then generate another `buffer_ratio *
                n_zeros` samples and fill in the zero samples. Continue until there are no zero
                samples.
        """
        buffer_size = int(batch_size * buffer_ratio)
        batch = torch.empty(0, device=self.device, dtype=torch.float32)
        n_samples_needed = batch_size
        while True:
            buffer = self._masked_batch_generator(buffer_size)
            # Get the indices of the non-zero samples in the buffer
            valid_indices = buffer.sum(dim=-1) != 0
            batch = torch.cat((batch, buffer[valid_indices][:n_samples_needed]))
            if len(batch) == batch_size:
                break
            else:
                # We don't have enough valid samples
                n_samples_needed = batch_size - len(batch)
                buffer_size = int(n_samples_needed * buffer_ratio)
        return batch


# Add to the existing file, after SparseFeatureDataset:

class SkipTrigramDataset(Dataset[Int[Tensor, "batch seq_len"]]):
    """Dataset for skip-trigram language modeling task.
    
    Generates sequences where:
    - Tokens 0-10: noise tokens (no predictive power, next token is random)
    - Tokens 11+: signal tokens that form skip trigrams
    - Each sequence contains exactly one skip trigram pattern: trigger1 _ trigger2 target
    """
    
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_trigrams: int = 10,
        device: str = "cpu",
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_trigrams = num_trigrams
        self.device = device
        
        # Split vocab: 0-10 are noise tokens, 11+ are signal tokens
        self.noise_tokens = list(range(11))  # 0-10
        self.signal_tokens = list(range(11, vocab_size))  # 11+
        
        if len(self.signal_tokens) < 2:
            raise ValueError(f"Need at least 2 signal tokens (vocab_size >= 13), got {vocab_size}")
        
        # Generate random skip trigrams: signal_token1 -> signal_token2 -> target_noise_token
        self.trigrams = []
        for _ in range(num_trigrams):
            # Pick two different signal tokens
            trigger_indices = torch.randperm(len(self.signal_tokens))[:2]
            trigger1 = self.signal_tokens[trigger_indices[0].item()]
            trigger2 = self.signal_tokens[trigger_indices[1].item()]
            
            # Pick a noise token as the target
            target_idx = torch.randint(0, len(self.noise_tokens), (1,)).item()
            target = self.noise_tokens[target_idx]
            
            self.trigrams.append((trigger1, trigger2, target))
        
        print(f"Generated {len(self.trigrams)} skip trigrams: {self.trigrams}")
    
    def __len__(self) -> int:
        return 2**31  # Infinite dataset like SparseFeatureDataset
    
    def generate_batch(self, batch_size: int) -> Int[Tensor, "batch seq_len"]:
        """Generate a batch of sequences, each containing exactly one skip trigram."""
        
        batch = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=self.device)
        
        if len(self.trigrams) > 0 and self.seq_len >= 4:
            for i in range(batch_size):
                # Pick a random trigram for this sequence
                trigram_idx = torch.randint(0, len(self.trigrams), (1,)).item()
                trigger1, trigger2, target = self.trigrams[trigram_idx]
                
                # Find a valid position for the pattern (trigger1, gap, trigger2, target)
                # Need positions j, j+2, j+3 to be valid
                max_start = self.seq_len - 4
                if max_start >= 0:
                    start_pos = torch.randint(0, max_start + 1, (1,)).item()
                    
                    # Insert the pattern: trigger1 at start_pos, trigger2 at start_pos+2, target at start_pos+3
                    batch[i, start_pos] = trigger1
                    batch[i, start_pos + 2] = trigger2  
                    batch[i, start_pos + 3] = target
                    
                    # The position at start_pos+1 stays random (the "skip")
        
        return batch
    
    def __getitem__(self, idx: int) -> Int[Tensor, "seq_len"]:
        """Generate a single sequence (for compatibility, but we mainly use generate_batch)."""
        return self.generate_batch(1)[0]