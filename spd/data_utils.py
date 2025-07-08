from collections.abc import Iterator
from typing import Generic, Literal, TypeVar

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from jaxtyping import Int


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
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        device: str = "cpu",
        n_trigrams: int = 32,
        allow_reflexive: bool = True,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device
        self.n_trigrams = n_trigrams
        self.allow_reflexive = allow_reflexive
        
        # Calculate maximum possible trigrams
        if allow_reflexive:
            max_possible_pairs = vocab_size * vocab_size
        else:
            max_possible_pairs = vocab_size * (vocab_size - 1)
            
        if n_trigrams > max_possible_pairs:
            raise ValueError(f"Cannot generate {n_trigrams} trigrams with vocab_size={vocab_size}. "
                           f"Maximum possible: {max_possible_pairs}")
        
        if seq_len < 3:
            raise ValueError(f"Sequence length must be at least 3 for skip trigrams, got {seq_len}")
        
        # Generate all possible (first, second) pairs
        all_pairs = []
        for first in range(vocab_size):
            for second in range(vocab_size):
                if allow_reflexive or first != second:
                    all_pairs.append((first, second))
        
        # Randomly sample n_trigrams pairs
        selected_indices = torch.randperm(len(all_pairs))[:n_trigrams]
        
        self.trigrams = {}  # (first_token, second_token) -> target_token
        
        for i in range(n_trigrams):
            pair_idx = selected_indices[i].item()
            first_token, second_token = all_pairs[pair_idx]
            
            # Target can be any token in vocab
            target_token = torch.randint(0, vocab_size, (1,)).item()
            
            self.trigrams[(first_token, second_token)] = target_token
        
        # All tokens can serve as noise
        self.noise_tokens = list(range(vocab_size))
        
        print(f"Generated {len(self.trigrams)} skip trigrams:")
        for (first, second), target in self.trigrams.items():
            print(f"  {first}, {second} -> {target}")
    
    def __len__(self) -> int:
        return 2**31  # Infinite dataset
    
    def generate_batch(self, batch_size: int) -> Int[Tensor, "batch seq_len"]:
        """Generate a batch of sequences with skip trigrams.
        
        Structure:
        - Positions 0 to seq_len-2: mix of trigram tokens and noise
        - Position seq_len-1: target token for the trigram present in the sequence
        """
        
        # Start with random noise tokens for all positions
        if len(self.noise_tokens) == 0:
            # If no noise tokens, use random tokens from vocab
            batch = torch.randint(
                low=0, 
                high=self.vocab_size, 
                size=(batch_size, self.seq_len), 
                device=self.device
            )
        else:
            batch = torch.randint(
                low=0, 
                high=len(self.noise_tokens), 
                size=(batch_size, self.seq_len), 
                device=self.device
            )
            # Map to actual noise token values
            for i in range(batch_size):
                for j in range(self.seq_len):
                    batch[i, j] = self.noise_tokens[batch[i, j].item()]
        
        # For each sequence, place one skip trigram
        if len(self.trigrams) > 0:
            for i in range(batch_size):
                # Randomly select a skip trigram
                trigram_keys = list(self.trigrams.keys())
                selected_key = trigram_keys[torch.randint(0, len(trigram_keys), (1,)).item()]
                first_token, second_token = selected_key
                target_token = self.trigrams[selected_key]
                
                # Place first and second tokens at random positions (excluding last position)
                available_positions = list(range(self.seq_len - 1))
                selected_positions = torch.randperm(len(available_positions))[:2]
                
                first_pos = available_positions[selected_positions[0].item()]
                second_pos = available_positions[selected_positions[1].item()]
                
                batch[i, first_pos] = first_token
                batch[i, second_pos] = second_token
                
                # Set target at the last position
                batch[i, -1] = target_token
        
        return batch
    
    def generate_batch_with_targets(self, batch_size: int) -> tuple[Int[Tensor, "batch seq_len"], Int[Tensor, "batch"]]:
        """Generate a batch of sequences along with their target tokens."""
        
        # Generate the batch
        batch = self.generate_batch(batch_size)
        
        # Extract targets (which are the last tokens)
        targets = batch[:, -1].clone()
        
        return batch, targets
    
    def __getitem__(self, idx: int) -> Int[Tensor, "seq_len"]:
        """Generate a single sequence (for compatibility, but we mainly use generate_batch)."""
        return self.generate_batch(1)[0]