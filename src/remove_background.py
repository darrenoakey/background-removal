import os
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "background-removal" / "huggingface"
MODEL_NAME = "ZhengPeng7/BiRefNet_HR"
MODEL_INPUT_SIZE = (1024, 1024)


# ##################################################################
# ensure directory
# make sure a directory exists before writing any files
def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ##################################################################
# set default cache env
# keep model and torch caches in repo-local storage unless already set
def set_default_cache_env(cache_dir: Path) -> None:
    ensure_directory(cache_dir)
    cache_value = str(cache_dir)
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = cache_value
    if not os.environ.get("TRANSFORMERS_CACHE"):
        os.environ["TRANSFORMERS_CACHE"] = cache_value
    if not os.environ.get("TORCH_HOME"):
        os.environ["TORCH_HOME"] = cache_value
    if not os.environ.get("XDG_CACHE_HOME"):
        os.environ["XDG_CACHE_HOME"] = cache_value


# ##################################################################
# resolve device
# select the best available device for inference
def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ##################################################################
# load model
# load the background removal model and move it to the chosen device
def load_model(device: torch.device, cache_dir: Path) -> torch.nn.Module:
    model = AutoModelForImageSegmentation.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(cache_dir),
    )
    model.eval()
    return model.to(device)


# ##################################################################
# build image transform
# create the preprocessing pipeline expected by the model
def build_image_transform(image_size: tuple[int, int]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# ##################################################################
# load image
# open an image file with orientation applied
def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        loaded = image.copy()
    return ImageOps.exif_transpose(loaded)


# ##################################################################
# ensure rgb image
# normalize input to an RGB image for model preprocessing
def ensure_rgb_image(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


# ##################################################################
# predict alpha mask
# run the model and return a resized alpha mask image
def predict_alpha_mask(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    transform_image: transforms.Compose,
) -> Image.Image:
    input_tensor = transform_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        prediction = outputs[-1].sigmoid().cpu()
    alpha_tensor = prediction[0].squeeze()
    alpha_image = transforms.ToPILImage()(alpha_tensor)
    return alpha_image.resize(image.size)


# ##################################################################
# apply alpha mask
# combine the original image with the predicted alpha mask
def apply_alpha_mask(image: Image.Image, alpha_mask: Image.Image) -> Image.Image:
    image_with_alpha = image.convert("RGBA")
    image_with_alpha.putalpha(alpha_mask)
    return image_with_alpha


# ##################################################################
# ensure png path
# enforce a .png file extension for output
def ensure_png_path(path: Path) -> Path:
    if path.suffix.lower() == ".png":
        return path
    return path.with_suffix(".png")


# ##################################################################
# remove background
# run the full background removal pipeline and save the png result
def remove_background(input_path: Path, output_path: Path) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    set_default_cache_env(DEFAULT_CACHE_DIR)

    device = resolve_device()
    model = load_model(device, DEFAULT_CACHE_DIR)
    original_image = load_image(input_path)
    model_image = ensure_rgb_image(original_image)
    transform_image = build_image_transform(MODEL_INPUT_SIZE)
    alpha_mask = predict_alpha_mask(model, model_image, device, transform_image)

    output_image = apply_alpha_mask(original_image, alpha_mask)
    final_path = ensure_png_path(output_path)
    ensure_directory(final_path.parent)
    output_image.save(final_path, format="PNG")
    return final_path
