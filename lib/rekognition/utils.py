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
