from unittest.mock import Mock

import cv2
import pytest

from robot_camera import camera as camera_module


def test_open_failure_releases_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise CameraError and release the camera without reading if opening fails."""
    fake_camera = Mock(spec=["open", "read", "release"])
    fake_camera.open.return_value = False

    monkeypatch.setattr(cv2, "VideoCapture", lambda: fake_camera)

    with pytest.raises(camera_module.CameraError, match="open camera"):
        camera_module.capture_jpeg("/dev/video-test")

    fake_camera.release.assert_called_once_with()
    fake_camera.read.assert_not_called()


def test_missing_frame_releases_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise CameraError and release the camera if reading returns no frame."""
    fake_camera = Mock(spec=["open", "read", "release"])
    fake_camera.open.return_value = True
    fake_camera.read.return_value = (False, None)

    monkeypatch.setattr(cv2, "VideoCapture", lambda: fake_camera)

    with pytest.raises(camera_module.CameraError, match="No image received"):
        camera_module.capture_jpeg("/dev/video-test")

    fake_camera.read.assert_called_once_with()
    fake_camera.release.assert_called_once_with()


def test_encoding_failure_releases_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise CameraError and release the camera if JPEG encoding fails."""
    fake_camera = Mock(spec=["open", "read", "release"])
    fake_frame = Mock(spec=["size"])
    fake_frame.size = 100

    fake_camera.open.return_value = True
    fake_camera.read.return_value = (True, fake_frame)

    fake_encoder = Mock(return_value=(False, None))

    monkeypatch.setattr(cv2, "VideoCapture", lambda: fake_camera)
    monkeypatch.setattr(cv2, "imencode", fake_encoder)

    with pytest.raises(camera_module.CameraError, match="JPEG encoding failed"):
        camera_module.capture_jpeg("/dev/video-test")

    fake_encoder.assert_called_once_with(".jpg", fake_frame)
    fake_camera.release.assert_called_once_with()


def test_successful_capture_returns_jpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return encoded JPEG bytes and release the camera after successful capture."""
    fake_camera = Mock(spec=["open", "read", "release"])
    fake_frame = Mock(spec=["size"])
    fake_frame.size = 100

    expected_jpeg = b"encoded-jpeg-data"
    fake_encoded = Mock(spec=["size", "tobytes"])
    fake_encoded.size = len(expected_jpeg)
    fake_encoded.tobytes.return_value = expected_jpeg

    fake_camera.open.return_value = True
    fake_camera.read.return_value = (True, fake_frame)
    fake_encoder = Mock(return_value=(True, fake_encoded))

    monkeypatch.setattr(cv2, "VideoCapture", lambda: fake_camera)
    monkeypatch.setattr(cv2, "imencode", fake_encoder)

    result = camera_module.capture_jpeg("/dev/video-test")

    assert result == expected_jpeg
    fake_encoder.assert_called_once_with(".jpg", fake_frame)
    fake_encoded.tobytes.assert_called_once_with()
    fake_camera.release.assert_called_once_with()
