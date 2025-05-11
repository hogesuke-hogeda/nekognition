import io
from abc import ABC, abstractmethod
from PIL import Image

import boto3
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.axes import Axes
from mypy_boto3_rekognition import RekognitionClient
from mypy_boto3_rekognition.type_defs import (
    DetectLabelsResponseTypeDef,
    LabelTypeDef
)
from streamlit.runtime.uploaded_file_manager import UploadedFile

from lib.bounding_box_drawer import IBoundingBoxDrawer


class IImageProcessor(ABC):
    def __init__(self):
        self._fig, self._ax = plt.subplots()
        self._rekognition_client: RekognitionClient = boto3.client(
            "rekognition"
        )

    @abstractmethod
    def set_uploaded_file(self, uploaded_file: UploadedFile):
        pass

    @abstractmethod
    def display_processed_image(self, bouding_box_drawer: IBoundingBoxDrawer):
        pass


class ImageProcessor(IImageProcessor):
    def set_uploaded_file(self, uploaded_file: UploadedFile):
        self._uploaded_file = uploaded_file
        self._image_bytes = self._uploaded_file.read()
        self._image_file = Image.open(io.BytesIO(self._image_bytes))

        # アップロードされた画像に枠線を描画するなどの加工をするための前処理
        self._ax.imshow(self._image_file)
        self._ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def display_processed_image(self, bouding_box_drawer: IBoundingBoxDrawer):
        self._label_cats(bouding_box_drawer)
        st.pyplot(self._fig)

    def _label_cats(self, bouding_box_drawer: IBoundingBoxDrawer):
        """猫の検出結果、信頼度を表示、検出された猫の位置の枠線をアップロードされた画像に設定する"""
        # amazon rekognitionによるラベル検出
        detect_label_response = self._detect_cat_labels(
            self._rekognition_client, self._image_bytes
        )

        # アップロードされた画像における猫の検出結果と信頼度の表示, 検出された物体の枠線を描画
        label_name_prefix = "Cat-"
        image_width, image_height = self._image_file.size
        highlight_states: dict[str, bool] = {}  # 枠線のハイライト有無をラベルごとに管理する

        for index, label in enumerate(detect_label_response["Labels"]):
            if label["Name"].lower() == "cat":
                label_name = label_name_prefix + f"{index+1}"
                label_confidence = f"{label['Confidence']:.2f}%"

                self._display_labels(
                    label_name, label_confidence, highlight_states
                )

                self._draw_bounding_boxes(
                    label, bouding_box_drawer, image_width, image_height, self._ax, label_name, label_confidence
                )

        if not bouding_box_drawer.bounding_boxes:
            st.write("アップロードされた画像において猫は検出されませんでした")

        bouding_box_drawer.update_bounding_box_color(highlight_states)

    def _detect_cat_labels(self, rekognition_client: RekognitionClient, image_bytes: bytes) -> DetectLabelsResponseTypeDef:
        """Amazon Rekognitionを使用して画像から猫のラベルを検出する"""
        return rekognition_client.detect_labels(
            Image={"Bytes": image_bytes},  # Max:5MB
            MaxLabels=10,
            MinConfidence=75,
            Features=["GENERAL_LABELS"],
            Settings={
                "GeneralLabels": {
                    "LabelInclusionFilters": ["Cat"]
                }
            },
        )

    def _display_labels(self, label_name: str, label_confidence: str, highlight_states: dict[str, bool]):
        """検出されたラベルと枠線ハイライト有無のチェックボックスを1行で表示する"""
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"- {label_name} ({label_confidence})")
        with col2:
            highlight_states[label_name] = st.checkbox(
                f"highlighten border-line", value=False, key=label_name
            )

    def _draw_bounding_boxes(
        self,
        label: LabelTypeDef,
        bounding_box_drawer: IBoundingBoxDrawer,
        image_width: int,
        image_height: int,
        ax: Axes,
        label_name: str,
        label_confidence: str
    ):
        """検出されたラベルに対応する枠線を全て描画する"""
        for instance in label.get("Instances", []):
            bounding_box_drawer.draw(instance, image_width, image_height,
                                     ax, label_name, label_confidence)
