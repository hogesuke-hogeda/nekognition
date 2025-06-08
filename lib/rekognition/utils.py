from typing import Optional

from mypy_boto3_rekognition.type_defs import (
    BoundingBoxTypeDef,
    DetectLabelsResponseTypeDef,
    InstanceTypeDef,
    LabelTypeDef
)


def extract_bounding_boxes_from_instance(instance: InstanceTypeDef) -> list[BoundingBoxTypeDef]:
    """
    LabelTypeDefからBoundingBoxを抽出してリストで返す
    """
    bounding_boxes: list[BoundingBoxTypeDef] = []
    if "BoundingBox" in instance:
        bounding_boxes.append(instance["BoundingBox"])
    return bounding_boxes


def validate_image_bytes(image_bytes: bytes):
    """
    rekoginitionクライアントに渡された画像バイトのサイズを検証し、不正な場合はエラーとする
    https://docs.aws.amazon.com/rekognition/latest/APIReference/API_Image.html#API_Image_Contents
    """
    min_bytes = 1
    max_bytes = 5242880
    if not (min_bytes <= len(image_bytes) <= max_bytes):
        raise ValueError(
            "image_bytes must be between 1 and 5242880 bytes."
        )


def extract_cat_label(detect_labels_res: DetectLabelsResponseTypeDef) -> Optional[LabelTypeDef]:
    """
    DetectLabelsResponseTypeDefから"Name"が"Cat"のラベルを抽出して返し、該当するラベルが無ければNone
    """
    for label in detect_labels_res.get("Labels", []):
        if label.get("Name") == "Cat":
            return label
    return None


def get_cat_instance_name_and_confidence(index: int, instance: InstanceTypeDef) -> tuple[str, str]:
    """
    Catインスタンスの連番名（Cat-1, Cat-2, ...）と信頼度文字列をタプルで返す
    インスタンスに信頼度情報がなかった場合は、"no confidence"を返す
    """
    instance_name = f"Cat-{index+1}"
    instance_confidence = f"{instance['Confidence']:.2f}%" if "Confidence" in instance else "no confidence"

    return instance_name, instance_confidence
