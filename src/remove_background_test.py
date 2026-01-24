from pathlib import Path

from PIL import Image, ImageDraw

from remove_background import remove_background

OUTPUT_DIR = Path("output/testing")


# ##################################################################
# create input image
# generate a simple test image with a clear foreground object
def create_input_image(path: Path) -> None:
    image = Image.new("RGB", (512, 512), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((96, 96, 416, 416), fill=(200, 30, 30))
    image.save(path, format="PNG")


# ##################################################################
# test remove background creates alpha
# verify the output is png with transparency and correct size
def test_remove_background_creates_alpha_white_background() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = OUTPUT_DIR / "input.png"
    output_path = OUTPUT_DIR / "output.png"
    create_input_image(input_path)

    result_path = remove_background(input_path, output_path)

    with Image.open(result_path) as image:
        image.load()
        assert image.mode == "RGBA"
        assert image.size == (512, 512)
        alpha_min, alpha_max = image.split()[-1].getextrema()
        assert alpha_min < alpha_max


# ##################################################################
# create colored background image
# generate a test image with a bright colored background
def create_colored_background_image(path: Path, bg_color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (128, 128), bg_color)
    draw = ImageDraw.Draw(image)
    draw.ellipse((24, 24, 104, 104), fill=(50, 50, 50))
    image.save(path, format="PNG")


# ##################################################################
# test remove background with colored background
# verify model handles bright colored backgrounds correctly
def test_remove_background_with_cyan_background() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = OUTPUT_DIR / "cyan_bg_input.png"
    output_path = OUTPUT_DIR / "cyan_bg_output.png"
    create_colored_background_image(input_path, (0, 255, 255))

    result_path = remove_background(input_path, output_path)

    with Image.open(result_path) as image:
        image.load()
        assert image.mode == "RGBA"
        assert image.size == (128, 128)
        alpha_channel = image.split()[-1]
        alpha_min, alpha_max = alpha_channel.getextrema()
        assert alpha_min < 50, f"Background should be mostly transparent, got min={alpha_min}"
        assert alpha_max > 200, f"Foreground should be mostly opaque, got max={alpha_max}"


# ##################################################################
# create wide aspect ratio image
# generate a test image with extreme aspect ratio
def create_wide_image(path: Path) -> None:
    image = Image.new("RGB", (512, 128), (0, 200, 200))
    draw = ImageDraw.Draw(image)
    draw.ellipse((192, 16, 320, 112), fill=(80, 80, 80))
    image.save(path, format="PNG")


# ##################################################################
# test aspect ratio preservation
# verify extreme aspect ratios are preserved in output
def test_remove_background_preserves_aspect_ratio() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = OUTPUT_DIR / "wide_input.png"
    output_path = OUTPUT_DIR / "wide_output.png"
    create_wide_image(input_path)

    result_path = remove_background(input_path, output_path)

    with Image.open(result_path) as image:
        image.load()
        assert image.size == (512, 128), f"Aspect ratio changed: expected (512, 128), got {image.size}"
