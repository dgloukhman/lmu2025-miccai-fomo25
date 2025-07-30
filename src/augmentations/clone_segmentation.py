import numpy as np
import os


class CloneMinigioma(object):
    def __init__(self, p_per_sample=1.0, **kwargs):
        self.p_per_sample = p_per_sample
        self.kwargs = kwargs

    def __call__(self, data_dict):
        if np.random.uniform() < self.p_per_sample:
            image = data_dict["image"]
            label = data_dict.get("label", None)
            if label is None:
                label = np.zeros(image.shape[1:], dtype=np.float32)
            trgt = np.concatenate((image, label), axis=0)
            augmented_np = clone_meningioma(trgt, **self.kwargs)
            data_dict["image"] = augmented_np[:-1]
            if len(data_dict["image"].shape) == 3:
                data_dict["image"] = np.expand_dims(
                    data_dict["image"], axis=0
                )  # Ensure image has batch dimension
            data_dict["label"] = augmented_np[-1]
            if len(data_dict["label"].shape) == 3:
                data_dict["label"] = np.expand_dims(
                    data_dict["label"], axis=0
                )  # Ensure label has batch dimension
        return data_dict


def get_bounding_box(arr):
    """
    Returns the bounding box of non-zero elements in a 3D array.
    Returns None if no bounding box is found.
    """
    non_zero_indices = np.argwhere(arr > 0)
    if non_zero_indices.size == 0:
        return None
    min_z, min_y, min_x = non_zero_indices.min(axis=0)
    max_z, max_y, max_x = non_zero_indices.max(axis=0)
    if min_z == max_z and min_y == max_y and min_x == max_x:
        return None
    return (min_z, min_y, min_x, max_z, max_y, max_x)


def load_random_minigioma_npy(directory):
    """
    Reads a directory of numpy files, selects one uniformly at random, and loads its contents.
    """
    npy_files = sorted([f for f in os.listdir(directory) if f.endswith(".npy")])
    if not npy_files:
        raise FileNotFoundError("No .npy files found in the directory.")
    selected_file = np.random.choice(npy_files)
    file_path = os.path.join(directory, selected_file)
    return np.load(file_path, allow_pickle=True)


def get_random_point_in_brain(trgt, additive_box_shape, threshold=0.05):
    """
    Selects a random point from a uniform distribution within the brain's bounding box.
    """
    brain_area = trgt[1].copy()
    # brain_area[brain_area < threshold] = 0

    brain_bbox = get_bounding_box(brain_area)
    if brain_bbox is None:
        print(ValueError("No valid brain area found for random point selection."))
        return None
    min_z, min_y, min_x, max_z, max_y, max_x = brain_bbox

    # Ensure the box fits inside the brain area
    z_low = min_z + additive_box_shape[0]
    z_high = max_z - additive_box_shape[0]
    y_low = min_y + additive_box_shape[1]
    y_high = max_y - additive_box_shape[1]
    x_low = min_x + additive_box_shape[2]
    x_high = max_x - additive_box_shape[2]

    if z_low >= z_high or y_low >= y_high or x_low >= x_high:
        print(ValueError("Additive box shape too large for brain area."))
        print(f"{additive_box_shape=}, {brain_bbox=}")
        return None

    random_z = np.random.randint(z_low, z_high)
    random_y = np.random.randint(y_low, y_high)
    random_x = np.random.randint(x_low, x_high)

    return (random_z, random_y, random_x)


def get_slices_by_bounding_box3d(corner, box):
    """
    Returns slice objects for extracting/pasting a box at a given corner.
    """
    min_z, min_y, min_x, max_z, max_y, max_x = box
    box_shape = (max_z - min_z, max_y - min_y, max_x - min_x)
    cz, cy, cx = corner
    z_slice = slice(cz, cz + box_shape[0])
    y_slice = slice(cy, cy + box_shape[1])
    x_slice = slice(cx, cx + box_shape[2])
    return (z_slice, y_slice, x_slice), box_shape


def random_translation_and_shift(src):
    # TODO implement with yucca
    return src


def mean_pool_numpy(arr, kernel_size):
    """
    Mean pools a numpy array with the given kernel size.
    Args:
        arr: numpy array of shape (C, D, H, W) or (D, H, W)
        kernel_size: tuple of ints (kD, kH, kW)
    Returns:
        Mean pooled numpy array.
    """
    arr_shape = arr.shape
    if len(arr_shape) == 4:
        # Only mean pool over the first two spatial dimensions (D, H)
        C, D, H, W = arr_shape
        kD, kH = kernel_size[:2]
        out_D = D // kD
        out_H = H // kH
        arr = arr[:, : out_D * kD, : out_H * kH, :]
        arr_reshaped = arr.reshape(C, out_D, kD, out_H, kH, W)
        pooled = arr_reshaped.mean(axis=(2, 4))
        return pooled
    elif len(arr_shape) == 3:
        D, H, W = arr_shape
        kD, kH = kernel_size[:2]
        out_D = D // kD
        out_H = H // kH
        arr = arr[: out_D * kD, : out_H * kH, :]
        arr_reshaped = arr.reshape(out_D, kD, out_H, kH, W)
        pooled = arr_reshaped.mean(axis=(1, 3))
        return pooled
    else:
        raise ValueError("Input array must be 3D or 4D.")


def shrink_src(src, box):
    """
    Shrinks the source array to fit within the target shape.
    """

    def get_shape_from_box(box):
        return (
            box[3] - box[0],
            box[4] - box[1],
            box[5] - box[2],
        )

    fits = False

    target_shape = get_shape_from_box(box)

    while not fits:
        patch_bbox = get_bounding_box(src[-1])
        patch_shape = get_shape_from_box(patch_bbox)
        if (
            patch_shape[0] <= target_shape[0]
            and patch_shape[1] <= target_shape[1]
            and patch_shape[2] <= target_shape[2]
        ):
            fits = True
        else:
            src = mean_pool_numpy(src, (2, 2))
    
    src[-1][src[-1] > 0] = 1  # Ensure the mask is binary

    return src
    return src


def clone_meningioma(
    trgt, patch_start_p=None, alpha=0.2, upper_threshold=0.9, lower_threshold=0.4
):
    """
    Clones a random minigioma patch into the target array at a random or specified center.
    """
    t2_original_preproccessed = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra58seq2/finetunig/data/preprocessed/Task002_FOMO2"
    trgt_brain_area = get_bounding_box(trgt[1])
    src = load_random_minigioma_npy(t2_original_preproccessed)
    src = random_translation_and_shift(src)
    src = shrink_src(src, trgt_brain_area)

    patch_bbox = get_bounding_box(src[-1])

    src_slices, additive_box_shape = get_slices_by_bounding_box3d(patch_bbox[:3], patch_bbox)

    if patch_start_p is None:
        patch_start_p = get_random_point_in_brain(trgt, additive_box_shape)
        if patch_start_p is None:
            return trgt

    trgt_slices, _ = get_slices_by_bounding_box3d(patch_start_p, patch_bbox)

    src_slices = (slice(None), *src_slices)
    trgt_slices = (slice(None), *trgt_slices)

    # Copy the box from src and paste it into trgt
    patch = src[src_slices].copy()
    target_patch = trgt[trgt_slices]

    mask = patch > upper_threshold
    target_patch[mask] = patch[mask]

    # Alpha blending for intensity values below the threshold
    blend_mask = (patch < upper_threshold) & (patch > lower_threshold)
    target_patch[blend_mask] = (1 - alpha) * target_patch[blend_mask] + alpha * patch[
        blend_mask
    ]

    trgt[trgt_slices] = target_patch

    return trgt
