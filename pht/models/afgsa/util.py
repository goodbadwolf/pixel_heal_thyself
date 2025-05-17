"""AFGSA utilities."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyexr

from pht.logger import logger
from pht.models.afgsa.preprocessing import postprocess_diffuse, postprocess_specular

mpl.use("Agg")


# =========================================================exr==========================================================
def _show_data(data: np.ndarray, channel: str) -> None:
    """Show data."""
    figsize = (15, 15)
    plt.figure(figsize=figsize)
    plt.title(channel)
    img_plot = plt.imshow(data, aspect="equal")
    img_plot.axes.get_xaxis().set_visible(False)
    img_plot.axes.get_yaxis().set_visible(False)
    plt.show()


def process_data(data: np.ndarray, channel: str, width: int, height: int) -> np.ndarray:
    """Process data."""
    if channel in ["default", "target", "diffuse", "albedo", "specular"]:
        data = np.clip(data, 0, 1) ** 0.45454545
    elif channel in ["normal", "normalA"]:
        # normalize
        for i in range(height):
            for j in range(width):
                data[i][j] = data[i][j] / np.linalg.norm(data[i][j])
        data = np.abs(data)
    elif channel in ["depth", "visibility", "normalVariance"] and np.max(data) != 0:
        data /= np.max(data)

    if data.shape[2] == 1:
        # reshape
        data = data.reshape(height, width)

    return data


def show_exr_info(exr_path: str) -> None:
    """Show EXR info."""
    assert exr_path, "Exr_path cannot be empty."
    assert exr_path.endswith("exr"), "Img to be shown must be in '.exr' format."
    exr = pyexr.open(exr_path)
    logger.info(f"Width: {exr.width}")
    logger.info(f"Height: {exr.height}")
    logger.info("Available channels:")
    exr.describe_channels()
    logger.info(f"Default channels: {exr.channel_map['default']}")


def show_exr_channel(exr_path: str, channel: str) -> None:
    """Show EXR channel."""
    exr = pyexr.open(exr_path)
    data = exr.get(channel)
    logger.info(f"Channel: {channel}")
    logger.info(f"Shape: {data.shape}")
    logger.info(f"Max: {np.max(data):f}    Min: {np.min(data):f}")
    data = process_data(data, channel, exr.width, exr.height)
    _show_data(data, channel)


# ========================================================img===========================================================
def tone_mapping(matrix: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Tone mapping."""
    return np.clip(matrix ** (1.0 / gamma), 0, 1)


def tensor2img(
    image_numpy: np.ndarray,
    post_spec: bool = False,
    post_diff: bool = False,
    albedo: np.ndarray | None = None,
) -> np.ndarray:
    """Tensor to image."""
    if post_diff:
        assert albedo is not None, "must provide albedo when post_diff is True"
    image_type = np.uint8

    # multiple images
    if image_numpy.ndim == 4:  # noqa: PLR2004
        temp = []
        for i in range(len(image_numpy)):
            if post_diff:
                temp.append(
                    tensor2img(
                        image_numpy[i],
                        post_spec=False,
                        post_diff=True,
                        albedo=albedo[i],
                    ),
                )
            else:
                temp.append(
                    tensor2img(
                        image_numpy[i],
                        post_spec=post_spec,
                        post_diff=False,
                    ),
                )
        return np.array(temp)
    image_numpy = np.transpose(image_numpy, (1, 2, 0))

    # postprocessing
    if post_spec:
        image_numpy = postprocess_specular(image_numpy)
    elif post_diff:
        albedo = np.transpose(albedo, (1, 2, 0))
        image_numpy = postprocess_diffuse(image_numpy, albedo)
    image_numpy = tone_mapping(image_numpy) * 255.0
    return np.clip(image_numpy, 0, 255).astype(image_type)


def save_img(
    save_path: str,
    img: np.ndarray,
    figsize: tuple[float, float],
    dpi: int,
    color: str | None = None,
) -> None:
    """Save image."""
    plt.cla()
    plt.figure(figsize=figsize, dpi=dpi)
    plt.axis("off")
    plt.imshow(img)
    fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    if color:
        plt.gca().add_patch(
            plt.Rectangle(
                xy=(0, 0),
                width=img.shape[1],
                height=img.shape[0],
                edgecolor=color,
                fill=False,
                linewidth=img.shape[1] * 1 / 92,
            ),
        )
    fig.savefig(save_path, format="png", transparent=True, pad_inches=0)


def save_img_group(
    save_path: str,
    index: int,
    noisy: np.ndarray,
    output: np.ndarray,
    y: np.ndarray,
) -> None:
    """Save image group."""
    name = os.path.join(save_path, f"{index}.png")
    # multiple images, just save the first one
    if noisy.ndim == 4:  # noqa: PLR2004
        noisy = noisy[0]
        output = output[0]
        y = y[0]
    plt.subplot(131)
    plt.axis("off")
    plt.imshow(noisy)
    plt.title("Noisy")

    plt.subplot(132)
    plt.axis("off")
    plt.imshow(output)
    plt.title("Output")

    plt.subplot(133)
    plt.axis("off")
    plt.imshow(y)
    plt.title("Reference")
    plt.savefig(name, bbox_inches="tight")


# ========================================================util==========================================================
def create_folder(path: str, still_create: bool = False) -> str:
    """
    Create a folder.

    Args:
        path: path to the folder
        still_create: still create or not when there's already a folder with the same name

    Returns:
        path to the created folder

    """
    if not os.path.exists(path):
        os.mkdir(path)
    elif still_create:
        dir_root = path[: path.rfind("\\")] if "\\" in path else "."
        count = 1
        original_dir_name = path.split("\\")[-1]
        while True:
            dir_name = original_dir_name + f"_{count}"
            path = os.path.join(dir_root, dir_name)
            if os.path.exists(path):
                count += 1
            else:
                os.mkdir(path)
                break
    return path
