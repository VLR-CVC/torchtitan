import torch
import torchvision

from torchvision.transforms.functional import pil_to_tensor

from urllib import request

def load_image_PIL(image) -> torch.Tensor:

    try:
        image_tensor = pil_to_tensor(image)
    except Exception as e:
        raise ValueError("Failed to load PIL image into Tensor") from e

    return image_tensor


def load_image(image_loc) -> torch.Tensor:
    """
    Expect local path of image (str)

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """

    print(image_loc)

    if isinstance(image_loc, str) and image_loc.startswith("http"):
        try:
            image_loc = request.urlopen(image_loc).read()
            image = torchvision.io.decode_image(
                torch.frombuffer(image_loc, dtype=torch.uint8),
                mode="RGB",
            )
        except Exception as e:
            raise ValueError("Failed to load remote image as torch.Tensor from the web") from e

    # Open the local image as a Tensor image
    else:
        try:
            image = torchvision.io.decode_image(image_loc, mode="RGB")
        except Exception as e:
            raise ValueError("Failed to load local image as torch.Tensor") from e

    return image