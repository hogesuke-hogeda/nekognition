from PIL import Image

import matplotlib.pyplot as plt
from mypy_boto3_rekognition.type_defs import DetectLabelsResponseTypeDef, FaceDetailTypeDef

from lib.image_processor import ImageProcessor
from tests.utils.image import figure_to_rgb_image, images_are_equal, load_expected_output_image


def get_drawn_output_image(
    image: Image.Image,
    processor: ImageProcessor,
    face_details: list[FaceDetailTypeDef],
    detect_labels_res: DetectLabelsResponseTypeDef,
    highlight_states: dict[str, bool],
) -> Image.Image:
    """
    ImageProcessorを使って、顔モザイクと物体検出枠線を描画した画像をPIL.Image(RGB)で返す
    """
    fig, ax = processor.process_image(
        image, face_details, detect_labels_res, highlight_states
    )
    output_image = figure_to_rgb_image(fig)
    plt.close(fig)
    return output_image


def dummy_labels_response_two_cats():
    return {
        "Labels": [
            {
                "Name": "Cat",
                "Instances": [
                    {
                        "BoundingBox": {
                            "Left": 0.48330920934677124,
                            "Top": 0.1411377638578415,
                            "Width": 0.5120457410812378,
                            "Height": 0.8588539361953735
                        },
                        "Confidence": 94.3099365234375
                    },
                    {
                        "BoundingBox": {
                            "Left": 0.4027925133705139,
                            "Top": 0.3347506523132324,
                            "Width": 0.3240136206150055,
                            "Height": 0.6651014089584351
                        },
                        "Confidence": 88.44355010986328
                    }
                ]
            }
        ]
    }


def dummy_labels_response_one_cat():
    return {
        "Labels": [
            {
                "Name": "Cat",
                "Instances": [
                    {
                        "BoundingBox": {
                            "Left": 0.43967190384864807,
                            "Top": 0.713158369064331,
                            "Width": 0.3027733862400055,
                            "Height": 0.28244075179100037,
                        },
                        "Confidence": 94.49180603027344
                    },
                ]
            }
        ]
    }


def dummy_labels_response_no_cat():
    return {"Labels": []}


def dummy_two_faces_details():
    return [
        {
            "BoundingBox": {
                "Width": 0.1257249414920807,
                "Height": 0.25461503863334656,
                "Left": 0.2533753514289856,
                "Top": 0.139725923538208
            },
        },
        {
            "BoundingBox": {
                "Width": 0.10855169594287872,
                "Height": 0.19728143513202667,
                "Left": 0.5083261728286743,
                "Top": 0.22977197170257568
            },
        }
    ]


def dummy_face_details_no_face():
    return []


def test_image_processor_no_face_no_cat(mosaic_drawer, box_drawer):
    processor = ImageProcessor(mosaic_drawer, box_drawer)
    input_image = Image.open(
        "tests/images/image_processor/input_no_face_no_cat.jpg"
    )
    expected_image = load_expected_output_image(
        "tests/images/image_processor/expected_output_no_face_no_cat.png"
    )
    output_image = get_drawn_output_image(
        input_image,
        processor,
        dummy_face_details_no_face(),
        dummy_labels_response_no_cat(),
        {}
    )
    assert images_are_equal(output_image, expected_image)


def test_image_processor_no_face_two_cats(mosaic_drawer, box_drawer):
    processor = ImageProcessor(mosaic_drawer, box_drawer)
    input_image = Image.open(
        "tests/images/image_processor/input_no_face_two_cats.jpg"
    )
    expected_image = load_expected_output_image(
        "tests/images/image_processor/expected_output_no_face_two_cats.png"
    )
    output_image = get_drawn_output_image(
        input_image,
        processor,
        dummy_face_details_no_face(),
        dummy_labels_response_two_cats(),
        {"Cat-1": False, "Cat-2": False}
    )
    assert images_are_equal(output_image, expected_image)


def test_image_processor_two_faces_one_cat(mosaic_drawer, box_drawer):
    processor = ImageProcessor(mosaic_drawer, box_drawer)
    input_image = Image.open(
        "tests/images/image_processor/input_two_faces_one_cat.jpg"
    )
    expected_image = load_expected_output_image(
        "tests/images/image_processor/expected_output_two_faces_one_cat.png"
    )
    output_image = get_drawn_output_image(
        input_image,
        processor,
        dummy_two_faces_details(),
        dummy_labels_response_one_cat(),
        {"Cat-1": False}
    )
    assert images_are_equal(output_image, expected_image)


def test_image_processor_two_faces_one_cat_highlight(mosaic_drawer, box_drawer):
    processor = ImageProcessor(mosaic_drawer, box_drawer)
    input_image = Image.open(
        "tests/images/image_processor/input_two_faces_one_cat.jpg"
    )
    expected_image = load_expected_output_image(
        "tests/images/image_processor/expected_output_two_faces_one_cat_highlight.png"
    )
    output_image = get_drawn_output_image(
        input_image,
        processor,
        dummy_two_faces_details(),
        dummy_labels_response_one_cat(),
        {"Cat-1": True}
    )
    assert images_are_equal(output_image, expected_image)
