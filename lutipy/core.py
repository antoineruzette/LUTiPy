import numpy as np
from matplotlib import pyplot as plt
from lutipy.utils import ImageProcessor, ComplementaryColors, PanelCreator




class LUTiPy:
    """Main interface for processing and visualizing images using LUTs."""

    def __init__(self, rgb: tuple = (255, 0, 255), channel_names: list = None, layout: str = 'grid', scale_length: float = 0.1, pixel_size: str = "10 nm", name_position="top-left", show_box_background=True,
                 scalebar=True, scalebar_position="bottom-left"):
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
        self.name_position = name_position
        self.show_box_background = show_box_background
        self.scalebar = scalebar
        self.scalebar_position = scalebar_position

    def process_image(self, image: np.ndarray) -> None:
        """Process the image and create a panel visualization.

        Args:
            image (np.ndarray): Input image to process.

        Returns:
            None
        """

        """
        TODO:
        1. Check the scale and modify wrt image resizing, otherwise it is incorrect.
        """
        image = ImageProcessor.convert_to_channel_last(image)
        image_8bit = ImageProcessor.convert_to_8bit(image)
        aspect_ratio = image_8bit.shape[1] / image_8bit.shape[0]
        image_8bit = ImageProcessor.resize_image(image_8bit, (400,int(400*aspect_ratio)))
        num_channels = image_8bit.shape[-1]
        complementary_colors = ComplementaryColors(self.rgb).find_n_complementary_colors(num_channels)
        composite_image = ImageProcessor.apply_complementary_luts(image_8bit, complementary_colors)

        if self.channel_names is None:
            self.channel_names = [f"Channel {i + 1}" for i in range(num_channels)]

        # Create panel and store the figure
        self._figure = PanelCreator.create_panel(
            image_8bit, complementary_colors, self.channel_names,
            ImageProcessor.apply_complementary_luts(image_8bit, complementary_colors),
            layout=self.layout, scale_length=self.scale_length, pixel_size=self.pixel_size, 
            name_position = self.name_position,
            show_box_background = self.show_box_background,
            scalebar=self.scalebar, scalebar_position=self.scalebar_position
        )

    def save_figure(self, filename: str) -> None:
        """Save the generated figure to a file.

        Args:
            filename (str): Path where the figure will be saved.
        """
        if self._figure is None:
            raise ValueError("No figure has been generated yet. Please call process_image() first.")
        
        self._figure.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}.")