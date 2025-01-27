# Lookup Table in Python (LUTiPy)
LUTiPy is a Python package for creating aesthetically pleasing composite images from fluorescent microscopy data using complementary LUTs.

## Features
- Apply preset or custom LUTs to microscopy images.
- Create composite images with complementary colors.

## Usage
```python
from lutipy.core import LUTiPy
import lutipy.data as data

image = data.cells()

# Create a LUTiPy object
lutipy = LUTiPy(layout="horizontal", 
                scalebar=True, pixel_size="0.31 nm", scale_length=0.25)

lutipy.process_image(image)
```

<img src="assets/out.png"/>

To save the frame:

```python
lutipy.save_figure("out.png")
```