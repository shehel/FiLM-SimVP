import os
import random
import numpy as np
import torch
from clearml import Dataset as ClearMLS
from torch.utils.data import Dataset
import pdb

from openstl.datasets.utils import create_loader


class TaxibjDataset(Dataset):
    """Taxibj <https://arxiv.org/abs/1610.00081>`_ Dataset
    
    This dataset class is modified to support two modes based on the `test` flag:
    - Training (`test=False`): Returns (data, labels, static_ch). Duplication and 
      quantile generation are handled in `train_collate_fn`.
    - Validation/Test (`test=True`): Returns (data, labels, static_ch, quantiles)
      with a fixed set of quantiles, preserving the original behavior.
    """

    def __init__(self, X, Y, use_augment=False, test=False, quantiles=[0.05, 0.2, 0.5, 0.8, 0.95], extra_quantiles=None):
        super(TaxibjDataset, self).__init__()
        self.X = (X + 1) / 2  # channel is 2
        self.Y = (Y + 1) / 2
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1
        self.test = test
        self.perm = False
        # Pre-convert static channel to a tensor for efficiency
        self.static_ch = torch.from_numpy(self.X.mean(axis=(0, 1), keepdims=True)[0]).float()

        self.quantiles = np.arange(0.05, 1, 0.1)

        self.pixel_list = [(10, 10),
                           (5, 25),
                           (18, 13),
                           (11, 16),
                           (20, 24),
                           (30, 19),
                           (24, 7),
                           (18, 5),
                           (16, 20)]

    def _augment_seq(self, seqs):
        """Augmentations as a video sequence"""
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3,))  # horizontal flip
        return seqs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.from_numpy(self.X[index, ::]).float()
        labels = torch.from_numpy(self.Y[index, ::]).float()

        if self.use_augment:
            len_data = data.shape[0]  # 4
            seqs = self._augment_seq(torch.cat([data, labels], dim=0))
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]

        # For validation and test, use the old scheme (fixed set of quantiles)
        if self.test:
            quantiles = np.arange(0.05, 1, 0.1)
            return data, labels, self.static_ch, quantiles
        # For training, return only the data. Quantiles and duplication are handled by the collate_fn.
        else:
            return data, labels, self.static_ch


def train_collate_fn(batch):
    """
    Custom collate function for training.
    For each sample in the batch, it duplicates the data 5 times and assigns
    a different quantile interval to each duplicate.
    """
    # 1. Define the quantile intervals for training.
    # These are the 5 low quantiles to generate intervals [low, 0.5, 1-low].
    train_low_quantiles = np.arange(0.05, 0.5, 0.1)  # [0.05, 0.15, 0.25, 0.35, 0.45]
    num_quantiles = len(train_low_quantiles)  # This will be 5

    # 2. Unpack the batch (it contains tuples of (data, labels, static_ch)).
    dynamic_input_list, target_list, static_list = zip(*batch)

    # 3. Stack into tensors.
    dynamic_input_batch = torch.stack(dynamic_input_list, dim=0)
    target_batch = torch.stack(target_list, dim=0)
    static_batch = torch.stack(static_list, dim=0)

    # Original batch size
    B = dynamic_input_batch.shape[0]

    # 4. Duplicate the data tensors for each quantile interval.
    # The new batch size will be B * num_quantiles.
    dynamic_input_batch = dynamic_input_batch.repeat_interleave(num_quantiles, dim=0)
    target_batch = target_batch.repeat_interleave(num_quantiles, dim=0)
    static_batch = static_batch.repeat_interleave(num_quantiles, dim=0)

    # 5. Create the corresponding quantiles tensor.
    quantiles_triplets = [torch.tensor([lq, 0.5, 1.0 - lq]) for lq in train_low_quantiles]
    quantiles_for_one_sample = torch.stack(quantiles_triplets, dim=0)  # Shape: (num_quantiles, 3)

    # Repeat for each sample in the original batch to match the duplicated data.
    quantiles_batch = quantiles_for_one_sample.repeat(B, 1)  # Shape: (B * num_quantiles, 3)

    # 6. Apply the static channel masking logic from the original function.
    ranges = [0, 0, 0.01, 0.02, 0.05, 0.1]
    ranges_l = [0.5, 0.01, 0.02, 0.05, 0.1, 0.5]
    rng = random.randint(0, 5)
    # Use float values for the mask to maintain tensor dtype.
    static_batch = torch.where((static_batch > ranges[rng]) & (static_batch < ranges_l[rng]), 1.0, 0.0)

    return dynamic_input_batch, target_batch, static_batch, quantiles_batch.float()


def test_collate_fn(batch):
    """
    Custom collate function for validation/testing.
    This function processes data from the dataset in test/val mode.
    """
    # Unpack the batch (it contains tuples of (data, labels, static_ch, quantiles)).
    dynamic_input_list, target_list, static_list, quantiles_list = zip(*batch)

    # Stack the tensor parts of the batch.
    dynamic_input_batch = torch.stack(dynamic_input_list, dim=0)
    target_batch = torch.stack(target_list, dim=0)
    static_batch = torch.stack(static_list, dim=0)
    
    # Stack the numpy array part (quantiles) and convert to a tensor.
    quantiles_batch = torch.from_numpy(np.stack(quantiles_list, axis=0)).float()

    # Make static_batch ones as per the original function's logic.
    static_batch = torch.ones_like(static_batch)

    return dynamic_input_batch, target_batch, static_batch, quantiles_batch


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=None, aft_seq_length=None, in_shape=None,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    try:
        data_root = ClearMLS.get(dataset_id="0a52221e6dec45b5a89aafbfcc1e8d9c").get_local_copy()
    except Exception as e:
        print(f"Could not download dataset from ClearML: {e}. Assuming local path.")

    dataset = np.load(os.path.join(data_root, 'dataset.npz'))
    X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset[
        'Y_train'], dataset['X_test'], dataset['Y_test']
    assert X_train.shape[1] == pre_seq_length and Y_train.shape[1] == aft_seq_length
    
    # The 'test' flag correctly distinguishes between training and validation/test datasets
    train_set = TaxibjDataset(X=X_train[:20000], Y=Y_train[:20000], use_augment=use_augment, test=False)
    val_set = TaxibjDataset(X=X_train[20000:], Y=Y_train[20000:], use_augment=use_augment, test=True)
    test_set = TaxibjDataset(X=X_test, Y=Y_test, use_augment=False, test=True)
    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher,
                                     collate_fn=train_collate_fn)  # Use the new training collate function
    dataloader_vali = create_loader(val_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher,
                                    collate_fn=test_collate_fn)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher,
                                    collate_fn=test_collate_fn)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    # Example usage:
    # Set a dummy data_root for local testing if ClearML is not available.
    if not os.path.exists('../../data/'):
        os.makedirs('../../data/')
        # Create a dummy dataset.npz for testing purposes
        print("Creating a dummy dataset for testing...")
        dummy_shape_X = (100, 4, 2, 32, 32)
        dummy_shape_Y = (100, 4, 2, 32, 32)
        dummy_X_train = np.random.rand(*dummy_shape_X) * 2 - 1
        dummy_Y_train = np.random.rand(*dummy_shape_Y) * 2 - 1
        dummy_X_test = np.random.rand(*dummy_shape_X) * 2 - 1
        dummy_Y_test = np.random.rand(*dummy_shape_Y) * 2 - 1
        np.savez_compressed(os.path.join('../../data/', 'dataset.npz'),
                            X_train=dummy_X_train, Y_train=dummy_Y_train,
                            X_test=dummy_X_test, Y_test=dummy_Y_test)

    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                  val_batch_size=4,
                  data_root='../../data/',
                  num_workers=4,
                  pre_seq_length=4, aft_seq_length=4)

    print(f"Train Dataloader Length: {len(dataloader_train)}, Test Dataloader Length: {len(dataloader_test)}")
    
    print("\n--- Training Dataloader Batch Shape ---")
    train_batch = next(iter(dataloader_train))
    # Expected shape for data: (batch_size * 5, seq_len, channels, H, W) -> (16 * 5, 4, 2, 32, 32) -> (80, 4, 2, 32, 32)
    # Expected shape for quantiles: (batch_size * 5, 3) -> (80, 3)
    print(f"Data shape: {train_batch[0].shape}")
    print(f"Labels shape: {train_batch[1].shape}")
    print(f"Static shape: {train_batch[2].shape}")
    print(f"Quantiles shape: {train_batch[3].shape}")

    print("\n--- Test Dataloader Batch Shape ---")
    test_batch = next(iter(dataloader_test))
    # Expected shape for data: (batch_size, seq_len, channels, H, W) -> (4, 4, 2, 32, 32)
    # Expected shape for quantiles: (batch_size, num_quantiles) -> (4, 10)
    print(f"Data shape: {test_batch[0].shape}")
    print(f"Labels shape: {test_batch[1].shape}")
    print(f"Static shape: {test_batch[2].shape}")
    print(f"Quantiles shape: {test_batch[3].shape}")