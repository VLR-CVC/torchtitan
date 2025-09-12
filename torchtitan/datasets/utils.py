import torch
import torchvision


def load_image(image_loc) -> torch.Tensor:
    """
    Expect local path of image (str)

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    try:
        image = torchvision.io.decode_image(image_loc)
    except Exception as e:
        raise ValueError("failed to load local image as Tensor") from e

    return image
