from abc import ABC, abstractmethod

from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from mypy_boto3_rekognition.type_defs import (
    BoundingBoxTypeDef,
    InstanceTypeDef,
)


class IBoundingBoxDrawer(ABC):
    def __init__(self, default_color):
        self._default_color = default_color
        self._bounding_boxes: list[tuple[Rectangle, str]] = []

    @property
    def bounding_boxes(self) -> list[tuple[Rectangle, str]]:
        return self._bounding_boxes

    @abstractmethod
    def draw(
        self,
        instance: InstanceTypeDef,
        image_width: int,
        image_height: int,
        ax: Axes,
        label_name: str,
        label_confidence: str
    ):
        pass

    @abstractmethod
    def update_bounding_box_color(self, highlight_states: dict[str, bool], highlight_color: str):
        pass


class BoundingBoxDrawer(IBoundingBoxDrawer):
    """検出された物体の枠線を描画、ハイライトする処理"""

    def __init__(self, default_color="gray"):
        super().__init__(default_color)

    def draw(
        self,
        instance: InstanceTypeDef,
        image_width: int,
        image_height: int,
        ax: Axes,
        label_name: str,
        label_confidence: str
    ):
        if "BoundingBox" in instance:
            box = instance["BoundingBox"]
            left, top = self._calculate_box_left_top(
                box, image_width, image_height
            )
            rect = self._generate_box(
                box, left, top, image_width, image_height
            )

            # 枠線の描画と、枠線に対する説明文[(ラベル名)(信頼度)]の記載
            ax.add_patch(rect)
            ax.text(
                left,
                top - 10,
                f"{label_name}({label_confidence})",
                color=self._default_color,
                fontsize=10,
                weight='bold'
            )

            # 枠線とラベル名を紐づけて管理
            self._bounding_boxes.append((rect, label_name))

    def update_bounding_box_color(self, highlight_states: dict[str, bool], highlight_color: str = "red"):
        for rect, label_name in self._bounding_boxes:
            rect.set_edgecolor(
                highlight_color if highlight_states[label_name] else self._default_color
            )

    def _generate_box(self, box: BoundingBoxTypeDef, left: int, top: int, image_width: int, image_height: int) -> Rectangle:
        """枠のサイズを計算し、Rectangleインスタンスを返す"""
        box_width = box["Width"] * image_width
        box_height = box["Height"] * image_height

        return Rectangle(
            (left, top),
            box_width,
            box_height,
            linewidth=2,
            edgecolor=self._default_color,
            facecolor='none'
        )

    def _calculate_box_left_top(self, box: BoundingBoxTypeDef, image_width: int, image_height: int) -> tuple[int, int]:
        "枠の始点(左上)を計算する"
        left = box["Left"] * image_width
        top = box["Top"] * image_height
        return left, top
