# Background Removal

Python tool using BiRefNet_HR model for background removal from images.

## Key Files

- `src/remove_background.py` - Main implementation
- `src/remove_background_test.py` - Tests (co-located)

## Critical: Model Input Size

BiRefNet requires **1024x1024 input** (`MODEL_INPUT_SIZE`) regardless of original image dimensions. Using the image's native size produces poor masks, especially for small images.

The mask is generated at 1024x1024, then resized back to match original dimensions. This preserves aspect ratio.

## Model Cache

Models cached at `~/.cache/background-removal/huggingface`

## Dependencies

- torch, torchvision, transformers, PIL
- kornia (required by BiRefNet)

## Running Tests

```bash
python3 -m pytest src/remove_background_test.py -v
```
