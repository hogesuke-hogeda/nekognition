from matplotlib.patches import Rectangle
from mypy_boto3_rekognition.type_defs import BoundingBoxTypeDef


def calculate_left_top(box: BoundingBoxTypeDef, image_width: int, image_height: int) -> tuple[int, int]:
    "枠の始点(左上)を計算する"
    left = box["Left"] * image_width
    top = box["Top"] * image_height
    return left, top


def generate_box(
    box: BoundingBoxTypeDef,
    left: int,
    top: int,
    image_width: int,
    image_height: int,
    color: str
) -> Rectangle:
    """枠のサイズを計算し、Rectangleインスタンスを返す"""
    box_width = box["Width"] * image_width
    box_height = box["Height"] * image_height

    return Rectangle(
        (left, top),
        box_width,
        box_height,
        linewidth=2,
        edgecolor=color,
        facecolor='none'
    )
