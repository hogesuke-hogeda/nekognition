from PIL import Image
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg


def load_expected_output_image(path: str) -> Image.Image:
    img = Image.open(path)
    # RGBを比較するために変換する
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return img


def figure_to_rgb_image(fig) -> Image.Image:
    """FigureCanvasAggのFigureをPIL.Image(RGB)に変換"""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()

    # ARGBバイト列の取得と配列化 *canvasは`tostring_rgb`を持たない
    argb = np.frombuffer(
        canvas.tostring_argb(),
        dtype=np.uint8
    ).reshape((h, w, 4))

    # ARGBバイト列からRGBバイト列を生成
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = argb[..., 1]  # R
    rgb[..., 1] = argb[..., 2]  # G
    rgb[..., 2] = argb[..., 3]  # B
    return Image.fromarray(rgb, 'RGB')


def images_are_equal(img1: Image.Image, img2: Image.Image) -> bool:
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    if arr1.shape != arr2.shape:
        return False
    # 完全一致で比較（JPEGノイズがある場合はatolを大きくする）
    return bool(np.allclose(arr1, arr2))
