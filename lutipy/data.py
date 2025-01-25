import numpy as np

def cells():
    """
    Sample image of cells. Channel 1: DAPI, Channel 2: GFP.
    Dimensions: 512x512 pixels.

    TODO:
    1. Add more image details
    2. Provide image acknowledgement
    """
    try:
        from imageio.v2 import volread
    except ImportError as e:
        raise ImportError(
            "Please `pip install imageio` to load cells"
        ) from e

    image_url = "https://raw.githubusercontent.com/rkarmaka/sample-datasets/main/cells/cells_1.tif"
    image = np.asarray(volread(image_url))

    return image


def nori():
    """
    Sample image of cells. Channel 1: Protein, Channel 2: Lipid, Channel 3: Endomucin.
    Dimensions: 1024x1024 pixels.

    TODO:
    1. Add more image details
    2. Provide image acknowledgement
    """
    try:
        from imageio.v2 import volread
    except ImportError as e:
        raise ImportError(
            "Please `pip install imageio` to load nori"
        ) from e

    image_url = "https://raw.githubusercontent.com/rkarmaka/sample-datasets/main//nori/nori.tif"
    image = np.asarray(volread(image_url))

    return image
