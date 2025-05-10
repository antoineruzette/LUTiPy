import numpy as np
from matplotlib import pyplot as plt
from lutipy.utils import ImageProcessor, ComplementaryColors, PanelCreator




class LUTiPy:
    """Main interface for processing and visualizing images using LUTs."""

    def __init__(self, rgb: tuple = (255, 0, 255), channel_names: list = None, layout: str = 'grid', scale_length: float = 0.25, pixel_size: str = None, name_position="top-left", show_box_background=True,
                 scalebar=False, scalebar_position="bottom-left", in_physical_units=False):
        """Initialize the LUTIPy object.

        Args:
            rgb (tuple): Base RGB color for LUT generation.
            channel_names (list): Names for each channel in the image.
            layout (str): Layout for the panel ('grid' or 'horizontal').
            scale_length (float): Scale bar length. If in_physical_units is False, this is a fraction of image width.
                                If in_physical_units is True, this is the physical length in the same units as pixel_size.
            pixel_size (str): Physical size of a pixel (e.g., "10 nm").
            in_physical_units (bool): If True, scale_length is interpreted as physical length. If False, it's a fraction of image width.
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
        self.in_physical_units = in_physical_units

    def _convert_pixel_size(self, image: np.ndarray, pixel_size: str) -> float:
        """Convert pixel size to a float value in meters."""
        try:
            value, unit = pixel_size.split()
            value = float(value)
            unit = unit.lower()
        except ValueError as e:
            raise ValueError(f"Invalid pixel size format: {pixel_size}") from e
        
        if not self.in_physical_units:
            # For fraction case, multiply by image width to get total width
            value = float(value) * image.shape[1]
        
        return value
        


    def process_image(self, image: np.ndarray) -> None:
        """Process the image and create a panel visualization.

        Args:
            image (np.ndarray): Input image to process.

        Returns:
            None
        """
        image = ImageProcessor.convert_to_channel_last(image)
        image_8bit = ImageProcessor.convert_to_8bit(image)
        pixel_size_value = self._convert_pixel_size(image_8bit, self.pixel_size)
        aspect_ratio = image_8bit.shape[1] / image_8bit.shape[0]
        original_width = image_8bit.shape[1]
        image_8bit = ImageProcessor.resize_image(image_8bit, (400,int(400*aspect_ratio)))
        resize_factor = image_8bit.shape[1] / original_width
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
            name_position=self.name_position,
            show_box_background=self.show_box_background,
            scalebar=self.scalebar, scalebar_position=self.scalebar_position,
            in_physical_units=self.in_physical_units,
            resize_factor=resize_factor
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