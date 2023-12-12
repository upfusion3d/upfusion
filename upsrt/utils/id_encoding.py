import torch
import numpy as np

def positionally_encode_ids(ids, n_freqs, start_freq):
    """
    Positionally encoded IDs. TODO: Elaborate.
    """
    # ids should be a torch tensor of shape (*, 1)
    freq_bands = 2. ** torch.arange(start_freq, start_freq+n_freqs) * np.pi
    sin_encodings = [torch.sin(ids * freq) for freq in freq_bands]
    cos_encodings = [torch.cos(ids * freq) for freq in freq_bands]

    pos_embeddings = torch.cat(sin_encodings + cos_encodings, dim=-1) # (*, 2 * n_freqs)
    return pos_embeddings

def create_patch_id_encoding(img_features_shape, num_patches, n_freqs, start_freq):
    """
    TODO: Elaborate
    """
    b, n_inp = img_features_shape[:2]
    patch_ids_list = [
        torch.full((b*n_inp, 1, 1), i/num_patches, dtype=torch.float32)
        for i in range(1, num_patches+1)
    ]
    patch_ids = torch.cat(patch_ids_list, dim=1)
    pos_encoded_patch_ids = positionally_encode_ids(patch_ids, n_freqs, start_freq)
    pos_encoded_patch_ids = pos_encoded_patch_ids.reshape(b, n_inp, num_patches, 2*n_freqs)

    return pos_encoded_patch_ids

def create_camera_id_encoding(img_features_shape, num_patches, n_freqs, start_freq):
    """
    TODO: Elaborate
    """
    b, n_inp = img_features_shape[:2]
    canonical_camera_id = torch.full((b, 1, num_patches, 1), 1/3 + 0.05, dtype=torch.float32)
    other_camera_id = torch.full((b, n_inp-1, num_patches, 1), 2/3 + 0.05, dtype=torch.float32)
    camera_ids = torch.cat((canonical_camera_id, other_camera_id), dim=1)
    pos_encoded_camera_ids = positionally_encode_ids(camera_ids, n_freqs, start_freq)

    return pos_encoded_camera_ids
