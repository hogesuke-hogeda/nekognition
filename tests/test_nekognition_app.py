import pytest
from types import SimpleNamespace
from unittest.mock import Mock

from app.nekognition_app import NekognitionApp
from lib.rekognition.utils import validate_image_bytes
from lib.rekognition.wrapper import IRekognitionClientWrapper


class MockRekognitionClientWrapper(IRekognitionClientWrapper):
    def detect_faces(self, image_bytes: bytes):
        validate_image_bytes(image_bytes)
        return ["face"]

    def detect_cats(
        self, image_bytes: bytes, max_labels: int = 10, min_confidence: int = 75
    ):
        validate_image_bytes(image_bytes)
        return {"Labels": ["cat"]}


@pytest.fixture
def dummy_uploaded_file() -> Mock:
    mock = Mock()
    mock.read.return_value = b"dummy_image_bytes"
    mock.name = "dummy.png"
    return mock


@pytest.fixture
def dummy_uploaded_file_min_bytes() -> Mock:
    mock = Mock()
    mock.read.return_value = b"a"
    mock.name = "dummy.png"
    return mock


@pytest.fixture
def dummy_uploaded_file_max_bytes() -> Mock:
    mock = Mock()
    mock.read.return_value = b"a" * 5242880
    mock.name = "dummy.png"
    return mock


@pytest.fixture
def dummy_uploaded_file_smaller_than_min_bytes() -> Mock:
    mock = Mock()
    mock.read.return_value = b""
    mock.name = "dummy.png"
    return mock


@pytest.fixture
def dummy_uploaded_file_larger_than_max_bytes() -> Mock:
    mock = Mock()
    mock.read.return_value = b"a" * 5242881
    mock.name = "dummy.png"
    return mock


@pytest.fixture
def mock_st() -> SimpleNamespace:
    st = SimpleNamespace()
    st.session_state = {}
    return st


@pytest.fixture
def rekognition_mocked_app() -> NekognitionApp:
    return NekognitionApp(rekognition_client=MockRekognitionClientWrapper())


def test_nekognition_app_update_session_state(
    monkeypatch, dummy_uploaded_file, mock_st, rekognition_mocked_app
):
    monkeypatch.setattr("app.nekognition_app.st", mock_st)
    rekognition_mocked_app._update_session_state_with_detection(
        dummy_uploaded_file
    )

    assert mock_st.session_state["image_bytes"] == b"dummy_image_bytes"
    assert mock_st.session_state["uploaded_filename"] == "dummy.png"
    assert mock_st.session_state["detect_face_res"] == ["face"]
    assert mock_st.session_state["detect_cats_res"] == {"Labels": ["cat"]}


def test_nekognition_app_update_session_state_min_bytes_uploaded_file(
    monkeypatch, dummy_uploaded_file_min_bytes, mock_st, rekognition_mocked_app
):
    monkeypatch.setattr("app.nekognition_app.st", mock_st)
    rekognition_mocked_app._update_session_state_with_detection(
        dummy_uploaded_file_min_bytes
    )

    assert mock_st.session_state["image_bytes"] == b"a"
    assert mock_st.session_state["uploaded_filename"] == "dummy.png"
    assert mock_st.session_state["detect_face_res"] == ["face"]
    assert mock_st.session_state["detect_cats_res"] == {"Labels": ["cat"]}


def test_nekognition_app_update_session_state_max_bytes_uploaded_file(
    monkeypatch, dummy_uploaded_file_max_bytes, mock_st, rekognition_mocked_app
):
    monkeypatch.setattr("app.nekognition_app.st", mock_st)
    rekognition_mocked_app._update_session_state_with_detection(
        dummy_uploaded_file_max_bytes
    )

    assert mock_st.session_state["image_bytes"] == b"a" * 5242880
    assert mock_st.session_state["uploaded_filename"] == "dummy.png"
    assert mock_st.session_state["detect_face_res"] == ["face"]
    assert mock_st.session_state["detect_cats_res"] == {"Labels": ["cat"]}


def test_nekognition_app_update_session_state_raises_value_error_on_smaller_than_min_bytes_uploaded_file(
    monkeypatch, dummy_uploaded_file_smaller_than_min_bytes, mock_st, rekognition_mocked_app
):
    monkeypatch.setattr("app.nekognition_app.st", mock_st)
    with pytest.raises(ValueError):
        rekognition_mocked_app._update_session_state_with_detection(
            dummy_uploaded_file_smaller_than_min_bytes
        )


def test_nekognition_app_update_session_state_raises_value_error_on_larger_than_max_bytes_uploaded_file(
    monkeypatch, dummy_uploaded_file_larger_than_max_bytes, mock_st, rekognition_mocked_app
):
    monkeypatch.setattr("app.nekognition_app.st", mock_st)
    with pytest.raises(ValueError):
        rekognition_mocked_app._update_session_state_with_detection(
            dummy_uploaded_file_larger_than_max_bytes
        )
