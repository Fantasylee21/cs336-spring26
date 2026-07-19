import torch
import torch.nn as nn
import numpy.typing as npt
import numpy as np

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loads the dataset and returns a batch of input and target tensors.

    Args:
        dataset: A 1D numpy array of integers representing the dataset.
        batch_size: The number of sequences in a batch.
        context_length: The length of each sequence.
        device: The device to which the tensors should be moved.

     Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    start_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    input_sequences = np.array([dataset[start:start + context_length] for start in start_indices])
    target_sequences = np.array([dataset[start + 1:start + context_length + 1] for start in start_indices])
    return torch.tensor(input_sequences, dtype=torch.long).to(device), torch.tensor(target_sequences, dtype=torch.long).to(device)


get_batch = data_loading