from mypy_boto3_rekognition.type_defs import (
    BoundingBoxTypeDef,
    LabelTypeDef,
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


def validate_image_bytes(image_bytes: bytes):
    """
    rekoginitionクライアントに渡された画像バイトのサイズを検証し、不正な場合はエラーとする
    https://docs.aws.amazon.com/rekognition/latest/APIReference/API_Image.html#API_Image_Contents
    """
    if not (1 <= len(image_bytes) <= 5242880):
        raise ValueError(
            "image_bytes must be between 1 and 5242880 bytes."
        )
