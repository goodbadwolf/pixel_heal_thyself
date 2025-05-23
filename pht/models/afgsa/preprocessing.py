"""AFGSA preprocessing."""

from random import Random

import numpy as np
import pyexr
from matplotlib import pyplot as plt
from scipy import ndimage

# constants
eps = 0.00316


def preprocess_diffuse(diffuse: np.ndarray, albedo: np.ndarray) -> np.ndarray:
    """Preprocess diffuse."""
    return diffuse / (albedo + eps)


def preprocess_specular(specular: np.ndarray) -> np.ndarray:
    """Preprocess specular."""
    # assert np.sum(specular < 0) == 0, "Negative value in specular component!"
    return np.log(specular + 1)


def preprocess_depth(depth: np.ndarray) -> np.ndarray:
    """Preprocess depth."""
    depth = np.clip(depth, 0.0, np.max(depth))
    max_feature = np.max(depth)
    if max_feature != 0:
        depth /= max_feature
    return depth


def preprocess_normal(normal: np.ndarray) -> np.ndarray:
    """Preprocess normal."""
    normal = np.nan_to_num(normal)
    normal = (normal + 1.0) * 0.5
    return np.maximum(np.minimum(normal, 1.0), 0.0)


def postprocess_diffuse(diffuse: np.ndarray, albedo: np.ndarray) -> np.ndarray:
    """Postprocess diffuse."""
    return diffuse * (albedo + eps)


def postprocess_specular(specular: np.ndarray) -> np.ndarray:
    """Postprocess specular."""
    return np.exp(specular) - 1


def scale_exr_img(
    exr_data: dict[str, np.ndarray],
    scale: float = 0.5,
) -> dict[str, np.ndarray]:
    """Scale EXR image."""
    scaled_exr_data = {}
    for channel_name, channel_data in exr_data.items():
        H, W, C = channel_data.shape  # noqa: N806
        block = int(1 / scale)
        assert H % block == 0 and W % block == 0, (  # noqa: PT018
            f"Image dims ({H}×{W}) must be divisible by 2"  # noqa: RUF001
        )
        new_H, new_W = H // block, W // block  # noqa: N806
        scaled_channel_data = channel_data.reshape(new_H, block, new_W, block, C).mean(
            axis=(1, 3),
        )
        scaled_exr_data[channel_name] = scaled_channel_data

    return scaled_exr_data


def preprocess_data(
    exr_path: str,
    gt_path: str,
    scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Preprocess data."""
    data = {}

    # high spp
    exr = pyexr.open(gt_path + ".exr")
    d = exr.get_all()
    data["gt"] = d["default"]

    # low spp
    noisy_exr = pyexr.open(exr_path + ".exr")
    d = noisy_exr.get_all()
    d_names = ["normal", "depth", "albedo", "noisy"]
    names = ["normal", "depth", "albedo", "default"]
    for i in range(len(d_names)):
        d_name = d_names[i]
        name = names[i]
        data[d_name] = d[name]

    if scale != 1.0:
        data = scale_exr_img(data, scale=scale)

    # nan to 0.0, inf to finite number
    for channel_name, channel_value in data.items():
        data[channel_name] = np.nan_to_num(channel_value)

    # clip data to avoid negative values
    data["gt"] = np.clip(data["gt"], 0, np.max(data["gt"]))
    data["noisy"] = np.clip(data["noisy"], 0, np.max(data["noisy"]))

    # normalize auxiliary features to [0.0, 1.0]
    # data['normal'] = preprocess_normal(data['normal'].copy())
    data["depth"] = preprocess_depth(data["depth"].copy())

    aux_features = np.concatenate(
        (data["normal"].copy(), data["depth"].copy(), data["albedo"].copy()),
        axis=2,
    )
    data["aux"] = aux_features

    return data


def get_variance_map(
    buffer: np.ndarray,
    patch_size: int,
    relative: bool = False,
) -> np.ndarray:
    """Get variance map."""
    # compute variance
    mean = ndimage.uniform_filter(buffer, size=(patch_size, patch_size, 1))
    square_mean = ndimage.uniform_filter(buffer**2, size=(patch_size, patch_size, 1))
    variance = np.maximum(square_mean - mean**2, 0)

    # convert to relative variance if requested
    if relative:
        variance = variance / np.maximum(mean**2, 1e-4)

    # take the max variance along the three channels, gamma correct it to get a
    # less peaky map, and normalize it to the range [0,1]
    variance = variance.max(axis=2)
    variance = np.minimum(variance ** (1.0 / 2.2), 1.0)

    return variance / np.maximum(variance.max(), 1e-4)


def get_importance_map(
    buffers: list[np.ndarray],
    metrics: list[str],
    weights: list[float],
    patch_size: int,
) -> np.ndarray:
    """Get importance map."""
    if len(metrics) != len(buffers):
        metrics = [metrics[0]] * len(buffers)
    if len(weights) != len(buffers):
        weights = [weights[0]] * len(buffers)

    importance_map = None
    for buffer, metric, weight in zip(buffers, metrics, weights, strict=False):
        if metric == "variance":
            temp = get_variance_map(buffer, patch_size, relative=False)
        elif metric == "relative":
            temp = get_variance_map(buffer, patch_size, relative=True)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if importance_map is None:
            importance_map = temp * weight
        else:
            importance_map += temp * weight

    return importance_map / np.max(importance_map)


def get_square_distance(x: int, y: int, patches: np.ndarray) -> float:
    """Get square distance."""
    if len(patches) == 0:
        return np.inf
    dist = patches - [x, y]
    return np.sum(dist**2, axis=1).min()


def sample_patches_dart_throwing(
    exr_shapes: tuple[int, int],
    patch_size: int,
    num_patches: int,
    rng: Random,
    max_iter: int = 5000,
) -> np.ndarray:
    """Sample patches dart throwing."""
    full_area = float(exr_shapes[0] * exr_shapes[1])
    sample_area = full_area / num_patches

    # get corresponding dart throwing radius
    radius = np.sqrt(sample_area / np.pi)
    min_square_distance = (2 * radius) ** 2

    # perform dart throwing, progressively reducing the radius
    rate = 0.96
    patches = np.zeros((num_patches, 2), dtype=int)
    x_min, x_max = 0, exr_shapes[1] - patch_size - 1
    y_min, y_max = 0, exr_shapes[0] - patch_size - 1
    for patch_index in range(num_patches):
        done = False
        while not done:
            for _i in range(max_iter):
                x = rng.randint(x_min, x_max)
                y = rng.randint(y_min, y_max)
                square_distance = get_square_distance(x, y, patches[:patch_index, :])
                if square_distance > min_square_distance:
                    patches[patch_index, :] = [x, y]
                    done = True
                    break
            if not done:
                radius *= rate
                min_square_distance = (2 * radius) ** 2
    return patches


def plot_patches(patches: np.ndarray) -> None:
    """Plot patches."""
    plt.figure(figsize=(15, 10))
    plt.scatter([p[0] for p in patches], [p[1] for p in patches])
    plt.show()


def get_region_list(
    exr_shapes: tuple[int, int],
    step: int,
) -> list[tuple[int, int, int, int]]:
    """Get region list."""
    regions = []
    for y in range(0, exr_shapes[0], step):
        xrange = (
            range(0, exr_shapes[1], step)
            if y // step % 2 == 0
            else reversed(range(0, exr_shapes[1], step))
        )
        for x in xrange:
            regions.append((x, x + step, y, y + step))
    return regions


def split_patches(
    patches: np.ndarray,
    region: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Split patches into current and remain."""
    current = np.empty_like(patches)
    remain = np.empty_like(patches)
    current_count, remain_count = 0, 0
    for i in range(patches.shape[0]):
        x, y = patches[i, 0], patches[i, 1]
        if region[0] <= x <= region[1] and region[2] <= y <= region[3]:
            current[current_count, :] = [x, y]
            current_count += 1
        else:
            remain[remain_count, :] = [x, y]
            remain_count += 1
    return current[:current_count, :], remain[:remain_count, :]


def prune_patches(
    exr_shapes: tuple[int, int],
    patches: np.ndarray,
    patch_size: int,
    importance_map: np.ndarray,
    rng: Random,
) -> np.ndarray:
    """Prune patches."""
    pruned = np.empty_like(patches)
    remain = np.copy(patches)
    count, error = 0, 0
    for region in get_region_list(exr_shapes, 4 * patch_size):
        current, remain = split_patches(remain, region)
        for i in range(current.shape[0]):
            x, y = current[i, 0], current[i, 1]
            if importance_map[y, x] - error > rng.random():
                pruned[count, :] = [x, y]
                count += 1
                error += 1 - importance_map[y, x]
            else:
                error += 0 - importance_map[y, x]
    return pruned[:count, :]


def importance_sampling(
    data: dict[str, np.ndarray],
    patch_size: int,
    num_patches: int,
    rng: Random,
) -> np.ndarray:
    """Apply importance sampling."""
    # extract buffers
    buffers = []
    for b in ["noisy", "normal"]:
        buffers.append(data[b][:, :, :])

    # build the importance map
    metrics = ["relative", "variance"]
    weights = [1.0, 1.0]
    importance_map = get_importance_map(buffers, metrics, weights, patch_size)

    # get patches
    patches = sample_patches_dart_throwing(
        buffers[0].shape[:2],
        patch_size,
        num_patches,
        rng,
    )

    # prune patches
    pad = patch_size // 2
    pruned = np.maximum(
        0,
        prune_patches(
            buffers[0].shape[:2],
            patches + pad,
            patch_size,
            importance_map,
            rng,
        )
        - pad,
    )

    return pruned + pad


def crop(
    data: dict[str, np.ndarray],
    position: tuple[int, int],
    patch_size: int,
) -> dict[str, np.ndarray]:
    """Crop."""
    half_patch = patch_size // 2
    hx, hy = half_patch, half_patch
    px, py = position
    temp = {}
    for key, value in data.items():
        if key in ["albedo", "depth", "normal"]:
            continue
        temp[key] = value[
            (py - hy) : (py + hy + patch_size % 2),
            (px - hx) : (px + hx + patch_size % 2),
            :,
        ]
        temp["kernel_gt"] = None
    return temp


def get_cropped_patches(  # noqa: PLR0913
    exr_path: str,
    gt_path: str,
    patch_size: int,
    num_patches: int,
    rng: Random,
    scale: float = 1.0,
) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    """Get cropped patches."""
    data = preprocess_data(exr_path, gt_path, scale=scale)
    patches = importance_sampling(data, patch_size, num_patches, rng)
    cropped = [crop(data, tuple(position), patch_size) for position in patches]
    return cropped, patches
