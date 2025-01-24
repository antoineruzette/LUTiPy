import numpy as np
from lutipy.utils import ImageProcessor, ComplementaryColors, PanelCreator




class LUTiPy:
    """Main interface for processing and visualizing images using LUTs."""

    def __init__(self, rgb: tuple = (255, 0, 255), channel_names: list = None, layout: str = 'grid', scale_length: float = 0.1, pixel_size: str = "10 nm"):
        """Initialize the LUTIPy object.

        Args:
            rgb (tuple): Base RGB color for LUT generation.
            channel_names (list): Names for each channel in the image.
            layout (str): Layout for the panel ('grid' or 'horizontal').
            scale_length (float): Scale bar length as a fraction of image width.
            pixel_size (str): Physical size of a pixel (e.g., "10 nm").
        """
        self.rgb = rgb
        self.channel_names = channel_names
        self.layout = layout
        self.scale_length = scale_length
        self.pixel_size = pixel_size

    def process_image(self, image: np.ndarray) -> None:
        """Process the image and create a panel visualization.

        Args:
            image (np.ndarray): Input image to process.

        Returns:
            None
        """
        image_8bit = ImageProcessor.convert_to_8bit(image)
        image_8bit = ImageProcessor.resize_image(image_8bit, (200,200))
        num_channels = image_8bit.shape[-1]
        complementary_colors = ComplementaryColors(self.rgb).find_n_complementary_colors(num_channels)
        composite_image = ImageProcessor.apply_complementary_luts(image_8bit, complementary_colors)

        if self.channel_names is None:
            self.channel_names = [f"Channel {i + 1}" for i in range(num_channels)]

        PanelCreator.create_panel(
            image_8bit, complementary_colors, self.channel_names, composite_image,
            layout=self.layout, scale_length=self.scale_length, pixel_size=self.pixel_size
        )
