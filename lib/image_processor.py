from PIL import Image

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mypy_boto3_rekognition.type_defs import DetectLabelsResponseTypeDef, FaceDetailTypeDef

from lib.face_mosaic_drawer import IFaceMosaicDrawer
from lib.boundary_draw.drawer import IBoundaryDrawer


class ImageProcessor:
    def __init__(
        self,
        mosaic_drawer: IFaceMosaicDrawer,
        bounding_box_drawer: IBoundaryDrawer
    ):
        self._mosaic_drawer = mosaic_drawer
        self._bounding_box_drawer = bounding_box_drawer

    def process_image(
        self,
        image: Image.Image,
        face_details: list[FaceDetailTypeDef],
        detect_labels_res: DetectLabelsResponseTypeDef,
        highlight_states: dict[str, bool],
        mosaic_size: int = 5,
        default_color: str = "gray",
        highlight_color: str = "red"
    ) -> tuple[Figure, Axes]:
        """
        顔にモザイクをかけ、検知した物体の枠線を描画したFigure, Axesを返す
        """
        face_mosaiced_image = self._mosaic_drawer.apply_mosaic(
            image, face_details, mosaic_size
        )
        fig, ax = self._bounding_box_drawer.draw(
            face_mosaiced_image,
            detect_labels_res,
            highlight_states,
            default_color,
            highlight_color
        )
        return fig, ax
