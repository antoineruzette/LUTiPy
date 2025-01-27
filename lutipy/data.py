import numpy as np

def cells():
    """
    Sample image of cells.
    Shape: 1392x1040x2 pixels.
    Channel 1: Phalloidin, Channel 2: DAPI.

    Reference: http://cellimagelibrary.org/images/CCDB_6843
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
    Sample image of cells.
    Dimensions: 1024x1024 pixels.
    Channel 1: Protein, Channel 2: Lipid, Channel 3: Endomucin.

    Reference: https://pubmed.ncbi.nlm.nih.gov/35452314/
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
