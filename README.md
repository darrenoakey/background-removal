![](banner.jpg)

# Background Removal Tool

Remove backgrounds from images with a single command.

## Purpose

This tool removes the background from any image and outputs a transparent PNG. It uses a high-quality neural network model optimized for accurate edge detection and fine detail preservation.

## Installation

1. Create and set up the virtual environment:

```bash
python3.12 -m venv ~/.venv/background-removal
~/.venv/background-removal/bin/pip install -r requirements.txt
```

2. (Optional) Create a convenience symlink for system-wide access:

```bash
ln -s /path/to/background-removal/run ~/bin/remove-background
```

## Usage

### Direct Command

```bash
./run remove <input_image> <output_image>
```

### With Symlink

```bash
remove-background <input_image> <output_image>
```

## Examples

Remove background from a photo:

```bash
./run remove photo.jpg photo_no_bg.png
```

Process a product image:

```bash
./run remove product.jpg transparent_product.png
```

Using the convenience wrapper:

```bash
remove-background portrait.jpg portrait_cutout.png
```

## Notes

- Output is always PNG format with transparency. If you specify a different extension, it will be replaced with `.png`.
- Output dimensions match the input image exactly.
- The model is downloaded and cached on first run (~500MB).

## Quality Gates

Run linting:

```bash
./run lint
```

Run tests:

```bash
./run test src/remove_background_test.py::test_remove_background_creates_alpha
```

Run full quality checks:

```bash
./run check
```