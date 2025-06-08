from PIL import Image

import matplotlib.pyplot as plt

from lib.boundary_draw.drawer import BoundingBoxDrawer
from tests.utils.image import figure_to_rgb_image, load_expected_output_image, images_are_equal


def get_drawn_output_image(
    drawer: BoundingBoxDrawer,
    input_image: Image.Image,
    labels_response: dict,
    highlight_states: dict
) -> Image.Image:
    """
    BoundingBoxDrawerを使って画像にラベルのバウンディングボックスを描画し、
    その結果をPIL.Image(RGB)として返す
    """
    fig, ax = drawer.draw(
        input_image, labels_response, highlight_states
    )
    output_image = figure_to_rgb_image(fig)
    plt.close(fig)
    return output_image


def dummy_labels_response_one_cat():
    return {
        "Labels": [
            {
                "Name": "Cat",
                "Instances": [
                    {
                        "BoundingBox": {
                            "Left": 0.25687479972839355,
                            "Top": 0.13289812207221985,
                            "Width": 0.5788983702659607,
                            "Height": 0.5985955595970154,
                        },
                        "Confidence": 98.4122543334961
                    },
                ]
            }
        ]
    }


def dummy_labels_response_one_cat_no_confidence():
    return {
        "Labels": [
            {
                "Name": "Cat",
                "Instances": [
                    {"BoundingBox": {
                        "Left": 0.25687479972839355,
                        "Top": 0.13289812207221985,
                        "Width": 0.5788983702659607,
                        "Height": 0.5985955595970154
                    }}
                ]
            }
        ]
    }


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


def dummy_labels_response_two_cats_no_confidence_cat1():
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
                        }
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


def dummy_labels_response_two_cats_no_confidence_cat2():
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
                            "Height": 0.8588539361953735,
                        },
                        "Confidence": 94.3099365234375
                    },
                    {
                        "BoundingBox": {
                            "Left": 0.4027925133705139,
                            "Top": 0.3347506523132324,
                            "Width": 0.3240136206150055,
                            "Height": 0.6651014089584351,
                        }
                    }
                ]
            }
        ]
    }


def dummy_labels_response_two_cats_no_confidence_cat1_2():
    return {
        "Labels": [
            {
                "Name": "Cat",
                "Instances": [
                    {"BoundingBox": {
                        "Left": 0.48330920934677124,
                        "Top": 0.1411377638578415,
                        "Width": 0.5120457410812378,
                        "Height": 0.8588539361953735,
                    }},
                    {"BoundingBox": {
                        "Left": 0.4027925133705139,
                        "Top": 0.3347506523132324,
                        "Width": 0.3240136206150055,
                        "Height": 0.6651014089584351,
                    }}
                ]
            }
        ]
    }


def dummy_labels_response_no_cat():
    return {"Labels": []}


def test_bounding_box_drawer_one_cat(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_one_cat.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_one_cat.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_one_cat(),
        {"Cat-1": False}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_one_cat_highlight(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_one_cat.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_one_cat_highlight.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_one_cat(),
        {"Cat-1": True}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_one_cat_no_confidence(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_one_cat.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_one_cat_no_confidence.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_one_cat_no_confidence(),
        {"Cat-1": False}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_one_cat_highlight_no_confidence(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_one_cat.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_one_cat_highlight_no_confidence.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_one_cat_no_confidence(),
        {"Cat-1": True}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_two_cats(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_two_cats.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_two_cats.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_two_cats(),
        {"Cat-1": False, "Cat-2": False}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_two_cats_highlight_cat1(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_two_cats.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_two_cats_highlight_cat1.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_two_cats(),
        {"Cat-1": True, "Cat-2": False}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_two_cats_highlight_cat2(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_two_cats.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_two_cats_highlight_cat2.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_two_cats(),
        {"Cat-1": False, "Cat-2": True}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_two_cats_no_confidence_cat1(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_two_cats.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_two_cats_no_confidence_cat1.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_two_cats_no_confidence_cat1(),
        {"Cat-1": False, "Cat-2": False}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_two_cats_no_confidence_cat2(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_two_cats.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_two_cats_no_confidence_cat2.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_two_cats_no_confidence_cat2(),
        {"Cat-1": False, "Cat-2": False}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_two_cats_no_confidence_cat1_2(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_two_cats.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_two_cats_no_confidence_cat1_2.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_two_cats_no_confidence_cat1_2(),
        {"Cat-1": False, "Cat-2": False}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_two_cats_highlight_cat1_no_confidence_cat1(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_two_cats.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_two_cats_highlight_cat1_no_confidence_cat1.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_two_cats_no_confidence_cat1(),
        {"Cat-1": True, "Cat-2": False}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_two_cats_highlight_cat2_no_confidence_cat1(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_two_cats.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_two_cats_highlight_cat2_no_confidence_cat1.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_two_cats_no_confidence_cat1(),
        {"Cat-1": False, "Cat-2": True}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_two_cats_highlight_cat1_2_no_confidence_cat1(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_two_cats.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_two_cats_highlight_cat1_2_no_confidence_cat1.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_two_cats_no_confidence_cat1(),
        {"Cat-1": True, "Cat-2": True}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_two_cats_highlight_cat1_no_confidence_cat2(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_two_cats.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_two_cats_highlight_cat1_no_confidence_cat2.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_two_cats_no_confidence_cat2(),
        {"Cat-1": True, "Cat-2": False}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_two_cats_highlight_cat2_no_confidence_cat2(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_two_cats.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_two_cats_highlight_cat2_no_confidence_cat2.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_two_cats_no_confidence_cat2(),
        {"Cat-1": False, "Cat-2": True}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_two_cats_highlight_cat1_2_no_confidence_cat2(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_two_cats.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_two_cats_highlight_cat1_2_no_confidence_cat2.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_two_cats_no_confidence_cat2(),
        {"Cat-1": True, "Cat-2": True}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_two_cats_highlight_cat1_no_confidence_cat1_2(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_two_cats.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_two_cats_highlight_cat1_no_confidence_cat1_2.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_two_cats_no_confidence_cat1_2(),
        {"Cat-1": True, "Cat-2": False}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_two_cats_highlight_cat2_no_confidence_cat1_2(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_two_cats.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_two_cats_highlight_cat2_no_confidence_cat1_2.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_two_cats_no_confidence_cat1_2(),
        {"Cat-1": False, "Cat-2": True}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_two_cats_highlight_cat1_2_no_confidence_cat1_2(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_two_cats.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_two_cats_highlight_cat1_2_no_confidence_cat1_2.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_two_cats_no_confidence_cat1_2(),
        {"Cat-1": True, "Cat-2": True}
    )
    assert images_are_equal(output_image, expected_output)


def test_bounding_box_drawer_no_cat(box_drawer):
    input_image = Image.open(
        "tests/images/bounding_box_draw/input_no_cat.jpg"
    )
    expected_output = load_expected_output_image(
        "tests/images/bounding_box_draw/expected_output_no_cat.png"
    )
    output_image = get_drawn_output_image(
        box_drawer,
        input_image,
        dummy_labels_response_no_cat(),
        {}
    )
    assert images_are_equal(output_image, expected_output)
