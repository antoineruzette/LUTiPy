# LUTiPy
LUTiPy is a Python package for creating aesthetically pleasing composite images from fluorescent microscopy data using complementary LUTs.

## Features
- Apply preset or custom LUTs to microscopy images.
- Create composite images with complementary colors.

## Usage
```python
import tifffile as tiff
from lutipy.core import lutipy

image = tiff.imread("example.tiff")
lutipy(image)
```

<!-- ![Example Image](assets/example_1.png) -->
<img src="assets/example_1.png" width="300" />