from lib.rekognition.utils import validate_image_bytes

import pytest


def test_validate_image_bytes_accepts_min_and_max():
    # 最小バイト
    validate_image_bytes(b"a")
    # 最大バイト
    validate_image_bytes(b"a" * 5242880)


def test_validate_image_bytes_raises_on_too_small():
    with pytest.raises(ValueError):
        validate_image_bytes(b"")


def test_validate_image_bytes_raises_on_too_large():
    with pytest.raises(ValueError):
        validate_image_bytes(b"a" * 5242881)
