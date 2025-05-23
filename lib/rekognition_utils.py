from mypy_boto3_rekognition import RekognitionClient
from mypy_boto3_rekognition.type_defs import (
    BoundingBoxTypeDef,
    DetectLabelsResponseTypeDef,
    FaceDetailTypeDef,
    LabelTypeDef,
)


def detect_faces(rekognition_client: RekognitionClient, image_bytes: bytes) -> list[FaceDetailTypeDef]:
    response = rekognition_client.detect_faces(
        Attributes=["DEFAULT"],
        Image={"Bytes": image_bytes}
    )
    return response["FaceDetails"]


def detect_cats(
        rekognition_client: RekognitionClient,
        image_bytes: bytes,
        max_labels: int = 10,
        min_confidence: int = 75
) -> DetectLabelsResponseTypeDef:
    return rekognition_client.detect_labels(
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


def extract_bounding_boxes_from_label(label: LabelTypeDef) -> list[BoundingBoxTypeDef]:
    """
    LabelTypeDefからBoundingBoxを抽出してリストで返す
    """
    bounding_boxes: list[BoundingBoxTypeDef] = []
    for instance in label.get("Instances", []):
        if "BoundingBox" in instance:
            bounding_boxes.append(instance["BoundingBox"])
    return bounding_boxes
