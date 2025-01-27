import colorsys
import numpy as np
import matplotlib.pyplot as plt


class ColorConverter:
    """Handles RGB ↔ HSL color conversions."""

    @staticmethod
    def rgb_to_hsl(r: int, g: int, b: int) -> tuple:
        """Convert RGB to HSL.

        Args:
            r (int): Red channel value (0-255).
            g (int): Green channel value (0-255).
            b (int): Blue channel value (0-255).

        Returns:
            tuple: HSL values as (h, l, s) where h, l, s are floats.
        """
        return colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)

    @staticmethod
    def hsl_to_rgb(h: float, l: float, s: float) -> tuple:
        """Convert HSL to RGB.

        Args:
            h (float): Hue value (0-1).
            l (float): Lightness value (0-1).
            s (float): Saturation value (0-1).

        Returns:
            tuple: RGB values as integers (0-255).
        """
        return tuple(round(i * 255) for i in colorsys.hls_to_rgb(h, l, s))

class ComplementaryColors:
    """Generates complementary colors for a given RGB color."""

    def __init__(self, rgb: tuple):
        """Initialize with an RGB color.

        Args:
            rgb (tuple): RGB color as (r, g, b) with values (0-255).
        """
        self.rgb = rgb

    def find_n_complementary_colors(self, n: int) -> list:
        """Find n complementary colors.

        Args:
            n (int): Number of complementary colors to generate.

        Returns:
            list: List of complementary RGB colors as tuples.
        """
        if n < 1:
            raise ValueError("Number of complementary colors must be at least 1.")

        r, g, b = self.rgb
        h, l, s = ColorConverter.rgb_to_hsl(r, g, b)

        offsets = [(i / n) for i in range(n)]
        complementary_colors = []
        for offset in offsets:
            h_complementary = (h + offset) % 1.0
            complementary_rgb = ColorConverter.hsl_to_rgb(h_complementary, l, s)
            complementary_colors.append(complementary_rgb)

        return complementary_colors

class ImageProcessor:
    """Performs image normalization and LUT application."""

    @staticmethod
    def convert_to_channel_last(image: np.ndarray) -> np.ndarray:
        """Convert an image to channel-last format (HWC).

        Args:
            image (np.ndarray): Input image with arbitrary channel order.

        Returns:
            np.ndarray: Image in (height, width, channels) format.
        """
        if image.ndim != 3:
            raise ValueError("Input image must be a 3D array (x, y, z or similar).")
        channel_axis = np.argmin(image.shape)
        return np.moveaxis(image, channel_axis, -1)

    @staticmethod
    def convert_to_8bit(image: np.ndarray) -> np.ndarray:
        """Convert a multi-channel image to 8-bit.

        Args:
            image (np.ndarray): Input image (e.g., 16-bit).

        Returns:
            np.ndarray: 8-bit image.
        """
        image_8bit = np.zeros_like(image, dtype=np.uint8)
        for channel in range(image.shape[-1]):
            channel_data = image[:, :, channel]
            channel_min = channel_data.min()
            channel_max = channel_data.max()
            if channel_max > channel_min:
                image_8bit[:, :, channel] = ((channel_data - channel_min) / (channel_max - channel_min) * 255).astype(np.uint8)
            else:
                image_8bit[:, :, channel] = 0
        return image_8bit
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize a single-channel image.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: normalized image.
        """
        normalized_image = np.zeros_like(image, dtype=np.uint8)
        
        normalized_image = image
        image_min = normalized_image.min()
        image_max = normalized_image.max()
        if image_max > image_min:
            normalized_image = ((normalized_image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
        
        return normalized_image
    
    @staticmethod
    def resize_image(image: np.ndarray, new_shape: tuple) -> np.ndarray:
        """Resize a multi-channel image to a new shape.

        Args:
            image (np.ndarray): Input 3D array (HWC format).
            new_shape (tuple): Desired shape (height, width).

        Returns:
            np.ndarray: Resized image with same number of channels.
        """
        zoom_factors = (new_shape[0] / image.shape[0], new_shape[1] / image.shape[1], 1)
        resized_image = np.zeros((new_shape[0], new_shape[1], image.shape[2]), dtype=image.dtype)
        for channel in range(image.shape[2]):
            resized_image[:, :, channel] = np.array(
                [[image[int(y / zoom_factors[0]), int(x / zoom_factors[1]), channel] for x in range(new_shape[1])] for y in range(new_shape[0])]
            )
        return resized_image

    @staticmethod
    def apply_complementary_luts(image: np.ndarray, luts: list) -> np.ndarray:
        """Apply complementary LUTs to an image.

        Args:
            image (np.ndarray): Input image in 8-bit format.
            luts (list): List of LUT colors (RGB tuples).

        Returns:
            np.ndarray: Composite RGB image.
        """
        height, width = image.shape[:2]
        composite_image = np.zeros((height, width, 3), dtype=np.float32)
        for channel_idx, lut_color in enumerate(luts):
            channel = image[:, :, channel_idx]
            normalized_channel = ImageProcessor.normalize_image(channel)/255.
            composite_image[:, :, 0] += normalized_channel * lut_color[0]
            composite_image[:, :, 1] += normalized_channel * lut_color[1]
            composite_image[:, :, 2] += normalized_channel * lut_color[2]
        return np.clip(composite_image, 0, 255).astype(np.uint8)

# class PanelCreator:
#     """Handles visualization of images and composite panels."""

#     @staticmethod
#     def create_panel(image: np.ndarray, luts: list, channel_names: list, composite_image: np.ndarray, scale_length: float = 0.1, pixel_size: str = "10 nm", layout: str = 'grid') -> None:
#         """
#         Create a tiled panel image where each channel is displayed as a grayscale image
#         with the channel name written in the corresponding LUT color. The last panel
#         displays the composite RGB image.

#         Args:
#             image (np.ndarray): Input image with multiple channels.
#             luts (list): List of LUT colors (RGB tuples).
#             channel_names (list): Names for each channel.
#             composite_image (np.ndarray): Composite RGB image.
#             scale_length (float): Scale bar length as a fraction of image width.
#             pixel_size (str): Physical size of a pixel (e.g., "10 nm").
#             layout (str): Layout of the panel ('grid' or 'horizontal').

#         Returns:
#             None: Displays the tiled image panel.
#         """
#         if image.ndim != 3:
#             raise ValueError("Input image must be 3D (height, width, channels).")
#         if len(luts) != image.shape[-1] or len(channel_names) != image.shape[-1]:
#             raise ValueError("Number of LUTs and channel names must match the number of channels in the image.")

#         print(image.shape)
#         num_channels = image.shape[-1]
#         # Determine grid size for tiling
#         if layout == 'grid':
#             grid_cols = int(np.ceil(np.sqrt(num_channels + 1)))
#             grid_rows = int(np.ceil((num_channels + 1) / grid_cols))
#         if layout == 'horizontal':
#             grid_cols = num_channels + 1
#             grid_rows = 1
#         if layout == 'vertical':
#             grid_cols = 1
#             grid_rows = num_channels + 1

#         # # Create a figure for the tiled panel
#         # fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(5 * grid_cols, 5 * grid_rows))
#         # axes = axes.flatten()

#         # Create a figure for the tiled panel with dynamic figsize
#         if image.shape[0] / image.shape[1] >= 1:
#             fig_width = 5 * grid_cols
#             fig_height = 5 * grid_rows * (image.shape[0] / image.shape[1])  # Adjust height based on image aspect ratio
        
#         if image.shape[0] / image.shape[1] < 1:
#             fig_width = 5 * grid_cols * (image.shape[1] / image.shape[0])  # Adjust width based on image aspect ratio
#             fig_height = 5 * grid_rows

        

#         fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height), constrained_layout=True)

#         axes = axes.flatten()

#         for i in range(num_channels):
#             # Extract the channel, resize and normalize it
#             channel = image[:, :, i]
#             normalized_channel = ImageProcessor.normalize_image(channel)

#             # Display the grayscale image
#             axes[i].imshow(normalized_channel, cmap='gray')
#             axes[i].axis('off')

#             # Add the channel name on the image
#             lut_color = np.array(luts[i]) / 255.0   # Normalize LUT to [0, 1] for matplotlib
#             axes[i].text(
#                 0.5, 0.05, channel_names[i],
#                 color=lut_color, fontsize=12, ha='center', va='center', transform=axes[i].transAxes,
#                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
#             )

#         # Display the composite image in the last panel
#         axes[num_channels].imshow(composite_image)
#         axes[num_channels].axis('off')
#         axes[num_channels].text(
#             0.5, 0.05, "Composite Image",
#             color='white', fontsize=12, ha='center', va='center', transform=axes[num_channels].transAxes,
#             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
#         )

#         # # Add scale bar to composite image with black overlay box
#         # scale_bar_length = image.shape[1]*scale_length  # Length in micrometers
#         # scale_value = scale_bar_length*(int(pixel_size.split(" ")[0]))
#         # axes[num_channels].add_patch(plt.Rectangle(
#         #     (composite_image.shape[1] - scale_bar_length - 15, 5), scale_bar_length + 10, 20,
#         #     color='black', alpha=0.7, transform=axes[num_channels].transData, clip_on=False
#         # ))
#         # axes[num_channels].plot(
#         #     [composite_image.shape[1] - scale_bar_length - 10, composite_image.shape[1] - 10],
#         #     [10, 10], color='white', lw=3, transform=axes[num_channels].transData, clip_on=False
#         # )
#         # axes[num_channels].text(
#         #     composite_image.shape[1] - scale_bar_length / 2 - 10, 20, f'{scale_value} {pixel_size.split(" ")[1]}',
#         #     color='white', fontsize=10, ha='center', va='bottom',
#         #     transform=axes[num_channels].transData
#         # )

        
#        # Add scale bar inside a black box with padding
#         black_box_padding = 0.02  # Fraction of image height for padding inside the black box
#         x_margin = image.shape[1] * 0.02  # Margin from the right edge (2% of image width)
#         y_margin = image.shape[0] * 0.05  # Margin from the bottom (5% of image height)

#         # Dimensions of the black box
#         box_width = int(image.shape[1] * scale_length) + int(image.shape[1] * black_box_padding * 2)
#         box_height = int(image.shape[0] * 0.05)  # Height of the box (5% of image height)

#         # Bottom-left corner of the black box
#         box_x = image.shape[1] - box_width - x_margin
#         box_y = y_margin

#         # # Add the black rectangle (background box)
#         # axes[num_channels].add_patch(plt.Rectangle(
#         #     (box_x, box_y),
#         #     box_width,  # Width of the box
#         #     box_height,  # Height of the box
#         #     color='black', alpha=0.7, transform=axes[num_channels].transData, clip_on=False
#         # ))

#         # Scale bar dimensions (with padding inside the black box)
#         scale_bar_length_px = int(image.shape[1] * scale_length)
#         scale_bar_x = box_x + int(image.shape[1] * black_box_padding)  # Start inside the box
#         scale_bar_y = box_y + int(box_height * 0.5) - 2  # Center vertically in the black box

#         # Draw the white scale bar line
#         axes[num_channels].plot(
#             [scale_bar_x, scale_bar_x + scale_bar_length_px],
#             [scale_bar_y, scale_bar_y],
#             color='white', lw=3, transform=axes[num_channels].transData, clip_on=False
#         )

#         # Add scale bar text below the line
#         text_y = box_y + int(image.shape[1] * black_box_padding * 0.5)  # Slightly above the bottom of the black box
#         axes[num_channels].text(
#             scale_bar_x + scale_bar_length_px / 2,  # Centered relative to the scale bar
#             text_y,
#             f'{scale_bar_length_px * int(pixel_size.split(" ")[0])} {pixel_size.split(" ")[1]}',
#             color='white', ha='center', va='bottom',
#             transform=axes[num_channels].transData
#         )



#         # Hide any unused axes
#         for j in range(num_channels + 1, len(axes)):
#             axes[j].axis('off')

#         # Adjust layout and show the panel
#         plt.tight_layout()
#         plt.show()

#         return fig


# class PanelCreator:
#     """Handles visualization of images and composite panels."""

#     @staticmethod
#     def create_panel(
#         image: np.ndarray,
#         luts: list,
#         channel_names: list,
#         composite_image: np.ndarray,
#         scale_length: float = 0.1,
#         pixel_size: float = None,  # Pixel size in physical units, required for scalebar
#         layout: str = 'grid',
#         name_position: str = 'bottom-center',
#         show_box_background: bool = True,
#         scalebar: bool = False,  # Whether to add a scalebar
#         scalebar_position: str = 'bottom-right'  # Position of the scalebar
#     ) -> plt.Figure:
#         """Create a tiled panel image with user-defined name positioning and optional scalebar.

#         Args:
#             image (np.ndarray): Input image with multiple channels.
#             luts (list): List of LUT colors (RGB tuples).
#             channel_names (list): Names for each channel.
#             composite_image (np.ndarray): Composite RGB image.
#             scale_length (float): Length of the scalebar as a fraction of the image width.
#             pixel_size (float): Physical size of a pixel (e.g., 0.1 micrometers). Required if scalebar is True.
#             layout (str): Layout of the panel ('grid', 'horizontal', 'vertical').
#             name_position (str): Position of channel names ('bottom-center', 'top-left', etc.).
#             show_box_background (bool): Whether to show a black background behind names.
#             scalebar (bool): Whether to add a scalebar.
#             scalebar_position (str): Position of the scalebar ('bottom-right', 'top-left', etc.).

#         Returns:
#             plt.Figure: The figure object for the created panel.
#         """
#         if image.ndim != 3:
#             raise ValueError("Input image must be 3D (height, width, channels).")
#         if len(luts) != image.shape[-1] or len(channel_names) != image.shape[-1]:
#             raise ValueError("Number of LUTs and channel names must match the number of channels in the image.")

#         num_channels = image.shape[-1]

#         # Determine grid size for tiling
#         if layout == 'grid':
#             grid_cols = int(np.ceil(np.sqrt(num_channels + 1)))
#             grid_rows = int(np.ceil((num_channels + 1) / grid_cols))
#         elif layout == 'horizontal':
#             grid_cols = num_channels + 1
#             grid_rows = 1
#         elif layout == 'vertical':
#             grid_cols = 1
#             grid_rows = num_channels + 1
#         else:
#             raise ValueError("Invalid layout. Choose from 'grid', 'horizontal', or 'vertical'.")

#         # Create a figure for the tiled panel with dynamic figsize
#         if image.shape[0] / image.shape[1] >= 1:
#             fig_width = 5 * grid_cols
#             fig_height = 5 * grid_rows * (image.shape[0] / image.shape[1])  # Adjust height based on image aspect ratio
        
#         if image.shape[0] / image.shape[1] < 1:
#             fig_width = 5 * grid_cols * (image.shape[1] / image.shape[0])  # Adjust width based on image aspect ratio
#             fig_height = 5 * grid_rows

#         fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height), constrained_layout=True)
#         axes = axes.flatten()

#         # Define text alignment based on position
#         position_map = {
#             'bottom-center': {'x': 0.5, 'y': 0.05, 'ha': 'center', 'va': 'bottom'},
#             'top-center': {'x': 0.5, 'y': 0.95, 'ha': 'center', 'va': 'top'},
#             'top-left': {'x': 0.05, 'y': 0.95, 'ha': 'left', 'va': 'top'},
#             'top-right': {'x': 0.95, 'y': 0.95, 'ha': 'right', 'va': 'top'},
#             'bottom-left': {'x': 0.05, 'y': 0.05, 'ha': 'left', 'va': 'bottom'},
#             'bottom-right': {'x': 0.95, 'y': 0.05, 'ha': 'right', 'va': 'bottom'}
#         }

#         if name_position not in position_map:
#             raise ValueError(f"Invalid name_position. Choose from {list(position_map.keys())}.")
#         if scalebar_position not in position_map:
#             raise ValueError(f"Invalid scalebar_position. Choose from {list(position_map.keys())}.")

#         name_text_position = position_map[name_position]
#         scalebar_text_position = position_map[scalebar_position]

#         for i in range(num_channels):
#             # Extract the channel, resize, and normalize it
#             channel = image[:, :, i]
#             normalized_channel = ImageProcessor.normalize_image(channel)

#             # Display the grayscale image
#             axes[i].imshow(normalized_channel, cmap='gray')
#             axes[i].axis('off')

#             # Add the channel name with or without a black background
#             lut_color = np.array(luts[i]) / 255.0  # Normalize LUT to [0, 1] for matplotlib
#             bbox_props = dict(facecolor='black', alpha=0.5, edgecolor='none') if show_box_background else None
#             axes[i].text(
#                 name_text_position['x'], name_text_position['y'], channel_names[i],
#                 color=lut_color, fontsize=12, ha=name_text_position['ha'], va=name_text_position['va'],
#                 transform=axes[i].transAxes,
#                 bbox=bbox_props
#             )

#         # Display the composite image in the last panel
#         axes[num_channels].imshow(composite_image)
#         axes[num_channels].axis('off')
#         axes[num_channels].text(
#             name_text_position['x'], name_text_position['y'], "Composite Image",
#             color='white', fontsize=12, ha=name_text_position['ha'], va=name_text_position['va'],
#             transform=axes[num_channels].transAxes,
#             bbox=bbox_props
#         )

#         # Add scalebar if enabled
#         if scalebar and pixel_size is not None:
#             # Calculate scale bar length in pixels and physical length
#             scale_bar_length_px = int(image.shape[1] * scale_length)  # Scale bar length in pixels
#             scale_bar_length_physical = scale_bar_length_px * float(pixel_size.split(" ")[0])  # Physical length (e.g., in µm)

#             # Get scalebar position from position_map
#             if scalebar_position not in position_map:
#                 raise ValueError(f"Invalid scalebar_position. Choose from {list(position_map.keys())}.")

#             scalebar_props = position_map[scalebar_position]
#             x_start = (
#                 0.5 - (scale_bar_length_px / (2 * image.shape[1])) if scalebar_props["ha"] == "center"
#                 else (0.1 if scalebar_props["ha"] == "left" else 0.9 - scale_bar_length_px / image.shape[1])
#             )
#             y_start = 0.1 if scalebar_props["va"] == "bottom" else 0.9

#             # Draw the scalebar line
#             x_end = x_start + scale_bar_length_px / image.shape[1]  # End of the line
#             axes[num_channels].plot(
#                 [x_start, x_end], [y_start, y_start], color='white', lw=3,
#                 transform=axes[num_channels].transAxes, clip_on=False
#             )

#             # Add the scalebar label (centered with the line)
#             text_x = (x_start + x_end) / 2  # Center of the scalebar
#             text_y = y_start - 0.02 if scalebar_props["va"] == "bottom" else y_start - 0.05  # Offset based on position
#             axes[num_channels].text(
#                 text_x, text_y, f'{scale_bar_length_physical:.2f} {pixel_size.split(" ")[1]}',
#                 color='white', ha='center', va='top' if scalebar_props["va"] == "bottom" else 'bottom',
#                 transform=axes[num_channels].transAxes, fontsize=10
#             )


#         if scalebar and pixel_size is None:
#             raise ValueError("Pixel size must be provided for scalebar.")


#         # Hide any unused axes
#         for j in range(num_channels + 1, len(axes)):
#             axes[j].axis('off')

#         return fig


class PanelCreator:
    """Handles visualization of images and composite panels."""

    @staticmethod
    def create_panel(
        image: np.ndarray,
        luts: list,
        channel_names: list,
        composite_image: np.ndarray,
        scale_length: float = 0.25,
        pixel_size: float = None,  # Pixel size in physical units, required for scalebar
        layout: str = 'grid',
        name_position: str = 'bottom-center',
        show_box_background: bool = True,
        scalebar: bool = False,  # Whether to add a scalebar
        scalebar_position: str = 'bottom-right'  # Position of the scalebar
    ) -> plt.Figure:
        """
        Create a tiled panel image with user-defined name positioning and optional scalebar.

        Args:
            image (np.ndarray): Input image with multiple channels.
            luts (list): List of LUT colors (RGB tuples).
            channel_names (list): Names for each channel.
            composite_image (np.ndarray): Composite RGB image.
            scale_length (float): Length of the scalebar as a fraction of the image width.
            pixel_size (float): Physical size of a pixel (e.g., 0.1 micrometers). Required if scalebar is True.
            layout (str): Layout of the panel ('grid', 'horizontal', 'vertical').
            name_position (str): Position of channel names ('bottom-center', 'top-left', etc.).
            show_box_background (bool): Whether to show a black background behind names.
            scalebar (bool): Whether to add a scalebar.
            scalebar_position (str): Position of the scalebar ('bottom-right', 'top-left', etc.).

        Returns:
            plt.Figure: The figure object for the created panel.
        """
        if image.ndim != 3:
            raise ValueError("Input image must be 3D (height, width, channels).")
        if len(luts) != image.shape[-1] or len(channel_names) != image.shape[-1]:
            raise ValueError("Number of LUTs and channel names must match the number of channels in the image.")

        num_channels = image.shape[-1]

        # Configure grid layout
        grid_rows, grid_cols = PanelCreator._configure_grid_layout(layout, num_channels)

        # Create figure with appropriate size
        fig, axes = PanelCreator._create_figure(image, grid_rows, grid_cols)

        # Define name and scalebar positions
        name_text_position, scalebar_text_position = PanelCreator._get_positions(name_position, scalebar_position)

        # Display each channel
        for i in range(num_channels):
            PanelCreator._display_channel(
                axes[i], image[:, :, i], luts[i], channel_names[i], name_text_position, show_box_background
            )

        # Display composite image
        PanelCreator._display_channel(
            axes[num_channels], composite_image, (255, 255, 255), "Composite Image", name_text_position, show_box_background
        )

        # Add scalebar if enabled
        if scalebar:
            if pixel_size is None:
                raise ValueError("Pixel size must be provided for scalebar.")
            PanelCreator._add_scalebar(axes[num_channels], scale_length, pixel_size, scalebar_text_position, image.shape, show_box_background)

        # Hide any unused axes
        for j in range(num_channels + 1, len(axes)):
            axes[j].axis('off')

        return fig

    @staticmethod
    def _configure_grid_layout(layout: str, num_channels: int) -> tuple:
        """Determine the grid layout based on the number of channels and layout style."""
        if layout == 'grid':
            grid_cols = int(np.ceil(np.sqrt(num_channels + 1)))
            grid_rows = int(np.ceil((num_channels + 1) / grid_cols))
        elif layout == 'horizontal':
            grid_cols = num_channels + 1
            grid_rows = 1
        elif layout == 'vertical':
            grid_cols = 1
            grid_rows = num_channels + 1
        else:
            raise ValueError("Invalid layout. Choose from 'grid', 'horizontal', or 'vertical'.")
        return grid_rows, grid_cols

    @staticmethod
    def _create_figure(image: np.ndarray, grid_rows: int, grid_cols: int) -> tuple:
        """Create a figure with dynamic aspect ratio based on the image dimensions."""
        if image.shape[0] / image.shape[1] >= 1:
            fig_width = 5 * grid_cols
            fig_height = 5 * grid_rows * (image.shape[0] / image.shape[1])
        else:
            fig_width = 5 * grid_cols * (image.shape[1] / image.shape[0])
            fig_height = 5 * grid_rows

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height), constrained_layout=True)
        axes = axes.flatten()
        return fig, axes

    @staticmethod
    def _get_positions(name_position: str, scalebar_position: str) -> tuple:
        """Get text alignment and position configurations for names and scalebar."""
        position_map = {
            'bottom-center': {'x': 0.5, 'y': 0.05, 'ha': 'center', 'va': 'bottom'},
            'top-center': {'x': 0.5, 'y': 0.95, 'ha': 'center', 'va': 'top'},
            'top-left': {'x': 0.05, 'y': 0.95, 'ha': 'left', 'va': 'top'},
            'top-right': {'x': 0.95, 'y': 0.95, 'ha': 'right', 'va': 'top'},
            'bottom-left': {'x': 0.05, 'y': 0.05, 'ha': 'left', 'va': 'bottom'},
            'bottom-right': {'x': 0.95, 'y': 0.05, 'ha': 'right', 'va': 'bottom'}
        }

        if name_position not in position_map or scalebar_position not in position_map:
            raise ValueError(f"Invalid position. Choose from {list(position_map.keys())}.")

        return position_map[name_position], position_map[scalebar_position]

    @staticmethod
    def _display_channel(axis, channel: np.ndarray, lut: tuple, name: str, text_position: dict, show_box_background: bool):
        """Display a single channel with LUT and optional text overlay."""
        normalized_channel = ImageProcessor.normalize_image(channel)
        axis.imshow(normalized_channel, cmap='gray')
        axis.axis('off')

        # Add channel name overlay
        lut_color = np.array(lut) / 255.0
        bbox_props = dict(facecolor='black', alpha=0.5, edgecolor='none') if show_box_background else None
        axis.text(
            text_position['x'], text_position['y'], name,
            color=lut_color, fontsize=12, ha=text_position['ha'], va=text_position['va'],
            transform=axis.transAxes, bbox=bbox_props
        )

    @staticmethod
    def _add_scalebar(axis, scale_length: float, pixel_size: float, text_position: dict, image_shape: tuple, show_box_background: bool):
        """Add a scalebar to the image."""
        scale_bar_length_px = int(image_shape[1] * scale_length)
        scale_bar_length_physical = scale_length * float(pixel_size.split(" ")[0])  # Physical length

        # Calculate dynamic starting positions based on text_position
        x_start = text_position['x'] - (scale_bar_length_px / (2 * image_shape[1])) if text_position['ha'] == 'center' else (
            0.05 if text_position['ha'] == 'left' else 0.95 - scale_bar_length_px / image_shape[1]
        )
        y_start = text_position['y'] - 0.05 if text_position['va'] == 'top' else text_position['y'] 

        # Draw the scalebar line
        x_end = x_start + scale_bar_length_px / image_shape[1]

        # Optionally add a black box background around the scalebar
        if show_box_background:
            box_x = x_start - 0.01
            box_y = y_start - 0.02
            box_width = (x_end - x_start) + 0.02
            box_height = 0.08
            axis.add_patch(
                plt.Rectangle(
                    (box_x, box_y), box_width, box_height,
                    color='black', alpha=0.5, transform=axis.transAxes, clip_on=False
                )
            )


        axis.plot(
            [x_start, x_end], [y_start, y_start], color='white', lw=3,
            transform=axis.transAxes, clip_on=False
        )

        # Add the scalebar label
        axis.text(
            (x_start + x_end) / 2, y_start + 0.05 if text_position['va'] == 'bottom' else y_start + 0.01,
            f'{scale_bar_length_physical:.2f} {pixel_size.split(" ")[1]}',
            color='white', ha='center', va='top' if text_position['va'] == 'bottom' else 'bottom',
            transform=axis.transAxes, fontsize=12
        )