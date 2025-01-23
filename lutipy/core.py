import colorsys
import numpy as np
import matplotlib.pyplot as plt


def rgb_to_hsl(r, g, b):
    """Convert RGB to HSL."""
    return colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)

def hsl_to_rgb(h, l, s):
    """Convert HSL to RGB."""
    return tuple(round(i * 255) for i in colorsys.hls_to_rgb(h, l, s))

def find_n_complementary_colors(rgb=(255,0,255), n=1):
    """
    Find n complementary color levels for an RGB input.
    Input: 
        rgb (tuple of R, G, B values, e.g., (255, 0, 0)).
        n (int): Number of complementary colors to generate.
    Output: 
        A list of n complementary RGB colors.
    """
    if n < 1:
        raise ValueError("Number of complementary colors must be at least 1.")

    # Convert RGB to HSL
    r, g, b = rgb
    h, l, s = rgb_to_hsl(r, g, b)

    # Calculate hue offsets based on the number of channels
    offsets = [(i / n) for i in range(0, n)]

    # Calculate complementary colors
    complementary_colors = []
    for offset in offsets:
        h_complementary = (h + offset) % 1.0  # Ensure the hue is within [0, 1]
        complementary_rgb = hsl_to_rgb(h_complementary, l, s)
        complementary_colors.append(complementary_rgb)

    return complementary_colors




def apply_complementary_luts(image, luts):
    """
    Apply complementary LUTs to an n-channel image and create a composite image.

    Args:
        image (numpy.ndarray): Input image of shape (height, width, channels).
        base_rgb (tuple): Base RGB color for generating complementary LUTs.

    Returns:
        numpy.ndarray: Composite image of shape (height, width, 3) (RGB).
    """
    # # Validate input dimensions
    # if image.ndim != 3:
    #     raise ValueError("Input image must be 3D (height, width, channels).")
    
    # num_channels = image.shape[-1]
    
    # # Generate complementary LUTs for each channel
    # complementary_colors = find_n_complementary_colors(base_rgb, num_channels)

    
    # Normalize image channel-wise and apply LUTs
    height, width = image.shape[:2]
    composite_image = np.zeros((height, width, 3), dtype=np.float32)  # Composite image (float for intermediate calculations)

    for channel_idx, lut_color in enumerate(luts):
        channel_data = image[:, :, channel_idx]
        
        # Normalize the channel to [0, 1]
        normalized_channel = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)
        
        # Map the normalized channel to the LUT color
        composite_image[:, :, 0] += normalized_channel * lut_color[0]  # Red
        composite_image[:, :, 1] += normalized_channel * lut_color[1]  # Green
        composite_image[:, :, 2] += normalized_channel * lut_color[2]  # Blue

    # Normalize the composite image to [0, 255] and convert to uint8
    composite_image = np.clip(composite_image, 0, 255).astype(np.uint8)

    return composite_image


def convert_to_channel_last(image):
    """
    Convert an image of arbitrary axis order to height, width, channel (HWC) format.

    Args:
        image (numpy.ndarray): Input image with arbitrary axis order.
                               Assumes 3D input (x, y, z or similar).

    Returns:
        numpy.ndarray: Image in HWC format (height, width, channel).
    """
    # Validate input dimensions
    if image.ndim != 3:
        raise ValueError("Input image must be a 3D array (x, y, z or similar).")
    
    # Identify the dimension with the smallest size as the likely 'channel' axis
    axes_sizes = image.shape
    channel_axis = np.argmin(axes_sizes)  # Channel usually has smallest size
    
    # Move the identified channel axis to the last position (HWC format)
    hwc_image = np.moveaxis(image, channel_axis, -1)

    return hwc_image


def convert_to_8bit(image):
    """
    Convert a 16-bit image to 8-bit by normalizing pixel values.

    Args:
        image_16bit (numpy.ndarray): Input 16-bit image (dtype=np.uint16).

    Returns:
        numpy.ndarray: 8-bit image (dtype=np.uint8).
    """

    # Initialize the output 8-bit image
    image_8bit = np.zeros_like(image, dtype=np.uint8)

    # Normalize each channel independently
    for channel in range(image.shape[-1]):
        channel_data = image[:, :, channel]
        channel_min = channel_data.min()
        channel_max = channel_data.max()
        
        # Avoid divide by zero if the channel has constant intensity
        if channel_max > channel_min:
            image_8bit[:, :, channel] = ((channel_data - channel_min) / (channel_max - channel_min) * 255).astype(np.uint8)
        else:
            # Set to zero if the channel has constant intensity
            image_8bit[:, :, channel] = 0

    return image_8bit

# Example usage
# if __name__ == "__main__":
#     # Simulate an input image (100x100 with 3 channels)
#     input_image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

#     # Simulate 3 LUTs (e.g., 256x3 RGB LUTs for each channel)
#     lut1 = np.array([[i, 0, 255-i] for i in range(256)], dtype=np.uint8)  # Blue gradient LUT
#     lut2 = np.array([[255-i, i, 0] for i in range(256)], dtype=np.uint8)  # Green gradient LUT
#     lut3 = np.array([[0, 255-i, i] for i in range(256)], dtype=np.uint8)  # Red gradient LUT

#     luts = [lut1, lut2, lut3]  # List of LUTs for channels

#     # Apply LUTs to the image
#     composite = apply_lut_to_image(input_image, luts)

#     # Save or visualize the composite image
#     from PIL import Image
#     composite_image = Image.fromarray(composite)
#     composite_image.show()


# Example
# original_color = (255, 0, 0)  # Red
# num_complementary_colors = 3  # Generate 5 complementary colors
# complementary_colors = find_n_complementary_colors(original_color, num_complementary_colors)

# print("Original Color (RGB):", original_color)
# print(f"{num_complementary_colors} Complementary Colors (RGB):")
# for idx, color in enumerate(complementary_colors, start=1):
#     print(f"  Level {idx}: {color}")



def create_panel(image, luts, channel_names, composite_image, scale_length=0.1, pixel_size="10 nm", layout='grid'):
    """
    Create a tiled panel image where each channel is displayed as a grayscale image
    with the channel name written in the corresponding LUT color. The last panel
    displays the composite RGB image.

    Args:
        image (numpy.ndarray): Input image of shape (height, width, channels).
        luts (list of tuple): List of LUT colors (RGB) for each channel.
        channel_names (list of str): List of names corresponding to each channel.
        composite_image (numpy.ndarray): Composite RGB image.

    Returns:
        None: Displays the tiled image panel.
    """
    if image.ndim != 3:
        raise ValueError("Input image must be 3D (height, width, channels).")

    if len(luts) != image.shape[-1] or len(channel_names) != image.shape[-1]:
        raise ValueError("Number of LUTs and channel names must match the number of channels in the image.")

    num_channels = image.shape[-1]

    if layout == 'grid':
        # Determine grid size for tiling
        grid_cols = int(np.ceil(np.sqrt(num_channels + 1)))
        grid_rows = int(np.ceil((num_channels + 1) / grid_cols))
    
    elif layout == 'horizontal':
        grid_cols = num_channels + 1
        grid_rows = 1

    # Create a figure for the tiled panel
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(5 * grid_cols, 5 * grid_rows))
    axes = axes.flatten()

    for i in range(num_channels):
        # Extract the channel
        channel = image[:, :, i]

        # Normalize the channel to [0, 1] for display
        normalized_channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)

        # Display the grayscale image
        axes[i].imshow(normalized_channel, cmap='gray')
        axes[i].axis('off')

        # Add the channel name on the image
        lut_color = np.array(luts[i]) / 255.0  # Normalize LUT to [0, 1] for matplotlib
        axes[i].text(
            0.5, 0.05, channel_names[i],
            color=lut_color, fontsize=12, ha='center', va='center', transform=axes[i].transAxes,
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
        )

    # Display the composite image in the last panel
    axes[num_channels].imshow(composite_image)
    axes[num_channels].axis('off')
    axes[num_channels].text(
        0.5, 0.05, "Composite Image",
        color='white', fontsize=12, ha='center', va='center', transform=axes[num_channels].transAxes,
        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
    )

    # Add scale bar to composite image with black overlay box
    scale_bar_length = image.shape[1]*scale_length  # Length in micrometers
    scale_value = scale_bar_length*(int(pixel_size.split(" ")[0]))
    axes[num_channels].add_patch(plt.Rectangle(
        (composite_image.shape[1] - scale_bar_length - 15, 5), scale_bar_length + 10, 20,
        color='black', alpha=0.7, transform=axes[num_channels].transData, clip_on=False
    ))
    axes[num_channels].plot(
        [composite_image.shape[1] - scale_bar_length - 10, composite_image.shape[1] - 10],
        [10, 10], color='white', lw=3, transform=axes[num_channels].transData, clip_on=False
    )
    axes[num_channels].text(
        composite_image.shape[1] - scale_bar_length / 2 - 10, 20, f'{scale_value} {pixel_size.split(" ")[1]}',
        color='white', fontsize=10, ha='center', va='bottom',
        transform=axes[num_channels].transData
    )


    # Hide any unused axes
    for j in range(num_channels + 1, len(axes)):
        axes[j].axis('off')
        # Add the scale bar





    # Adjust layout and show the panel
    plt.tight_layout()
    plt.show()



def image_resize(image, new_size):
    """
    Resize an image to a new size.

    Args:
        image (numpy.ndarray): Input image.
        new_size (tuple): New size (height, width).

    Returns:
        numpy.ndarray: Resized image.
    """
    resized_image = np.zeros_like(image, dtype=image.dtype)
    for i in range(image.shape[-1]):
        resized_image[:, :, i] = image[:, :, i].resize(new_size)
    return resized_image

def lutipy(image, rgb=(255,0,255), channel_names=None, layout='grid', scale_length=0.1, pixel_size="10 nm"):
    image_8bit = convert_to_8bit(image)
    num_channels = image_8bit.shape[-1]
    luts = find_n_complementary_colors(rgb=rgb, n=num_channels)
    image_lut = apply_complementary_luts(image_8bit, luts)

    if channel_names is None:
        channel_names = [f"Channel {i+1}" for i in range(num_channels)]

    create_panel(image_8bit, luts, channel_names, image_lut, layout=layout, scale_length=scale_length, pixel_size=pixel_size)