from abc import ABC, abstractmethod
from mypy_boto3_rekognition import RekognitionClient
from mypy_boto3_rekognition.type_defs import (
    DetectLabelsResponseTypeDef,
    FaceDetailTypeDef,
)

from lib.rekognition.utils import validate_image_bytes


class IRekognitionClientWrapper(ABC):
    @abstractmethod
    def detect_faces(self, image_bytes: bytes) -> list[FaceDetailTypeDef]:
        pass

    @abstractmethod
    def detect_cats(
        self, image_bytes: bytes, max_labels: int = 10, min_confidence: int = 75
    ) -> DetectLabelsResponseTypeDef:
        pass


class RekognitionClientWrapper(IRekognitionClientWrapper):
    def __init__(self, rekognition_client: RekognitionClient):
        self._client = rekognition_client

    def detect_faces(self, image_bytes: bytes) -> list[FaceDetailTypeDef]:
        validate_image_bytes(image_bytes)

        response = self._client.detect_faces(
            Attributes=["DEFAULT"], Image={"Bytes": image_bytes}
        )
        return response["FaceDetails"]

    def detect_cats(
        self,
        image_bytes: bytes,
        max_labels: int = 10,
        min_confidence: int = 75,
    ) -> DetectLabelsResponseTypeDef:
        validate_image_bytes(image_bytes)

        return self._client.detect_labels(
            Image={"Bytes": image_bytes},
            MaxLabels=max_labels,
            MinConfidence=min_confidence,
            Features=["GENERAL_LABELS"],
            Settings={
                "GeneralLabels": {
                    "LabelInclusionFilters": ["Cat"]  # 物体へのラベル付けは猫に限定する
                }
            },
        )
