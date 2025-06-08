from PIL import Image

from tests.utils.image import images_are_equal, load_expected_output_image


def dummy_face_details():
    return [
        {
            "BoundingBox": {
                "Width": 0.11904211342334747,
                "Height": 0.23009070754051208,
                "Left": 0.5385562181472778,
                "Top": 0.11311305314302444
            },
        },
        {
            "BoundingBox": {
                "Width": 0.1115478053689003,
                "Height": 0.2153182327747345,
                "Left": 0.29374393820762634,
                "Top": 0.3033357560634613
            },
        }
    ]


def dummy_face_details_no_face():
    return []


def test_ellipse_face_mosaic_drawer_no_face(mosaic_drawer):
    input_image = Image.open(
        "tests/images/face_mosaic_draw/input_no_face.jpg"
    )
    output_image = mosaic_drawer.apply_mosaic(
        input_image, dummy_face_details_no_face(), mosaic_size=5
    )
    expected_image = load_expected_output_image(
        "tests/images/face_mosaic_draw/expected_output_no_face.png"
    )
    assert images_are_equal(output_image, expected_image)


def test_ellipse_face_mosaic_drawer_two_faces(mosaic_drawer):
    input_image = Image.open(
        "tests/images/face_mosaic_draw/input_two_faces.jpg"
    )
    output_image = mosaic_drawer.apply_mosaic(
        input_image, dummy_face_details(), mosaic_size=5
    )
    expected_image = load_expected_output_image(
        "tests/images/face_mosaic_draw/expected_output_two_faces.png"
    )
    assert images_are_equal(output_image, expected_image)
