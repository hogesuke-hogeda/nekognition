from abc import ABC, abstractmethod
from PIL.Image import Image

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mypy_boto3_rekognition.type_defs import DetectLabelsResponseTypeDef
import matplotlib.pyplot as plt

from lib.boundary_draw.utils import (
    calculate_left_top,
    generate_box,
)
from lib.rekognition.utils import extract_bounding_boxes_from_label


class IBoundaryDrawer(ABC):
    @abstractmethod
    def draw(
        self,
        target_image: Image,
        detect_labels_res: DetectLabelsResponseTypeDef,
        highlight_states: dict[str, bool],
        default_color: str,
        highlight_color: str,
    ) -> tuple[Figure, Axes]:
        pass


class BoundingBoxDrawer(IBoundaryDrawer):
    def draw(
        self,
        target_image: Image,
        detect_labels_res: DetectLabelsResponseTypeDef,
        highlight_states: dict[str, bool] = {},
        default_color: str = "gray",
        highlight_color: str = "red",
    ) -> tuple[Figure, Axes]:
        """
        rekognitionで検出された物体の枠線（矩形）を描画し、処理済みのFigureを返す処理
        highlight_states={"Cat-1": True, "Cat-2": False} -> Cat-1に対応する枠線をハイライトして描画
        """
        fig, ax = plt.subplots()
        ax.imshow(target_image)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        image_width, image_height = target_image.size

        for index, label in enumerate(detect_labels_res["Labels"]):
            label_name = f"Cat-{index+1}"
            label_confidence = f"{label['Confidence']:.2f}%" if label["Confidence"] else "信頼度情報なし"
            bounding_boxes = extract_bounding_boxes_from_label(label)

            color = highlight_color if highlight_states[label_name] else default_color

            for bounding_box in bounding_boxes:
                left, top = calculate_left_top(
                    bounding_box, image_width, image_height
                )
                rect = generate_box(
                    bounding_box, left, top, image_width, image_height, color
                )

                ax.add_patch(rect)
                ax.text(
                    left,
                    top - 10,
                    f"{label_name}({label_confidence})",
                    color=color,
                    fontsize=10,
                    weight='bold'
                )

        return fig, ax
