import cv2


class CameraError(RuntimeError):
    """A camera operation failed"""


def capture_jpeg(device: str) -> bytes:
    """Capture one frame from a Linux camera as JPEG bytes"""
    if not device.strip():
        raise ValueError("Camera device must not be empty")

    camera = cv2.VideoCapture()

    try:
        if not camera.open(device, cv2.CAP_V4L2):
            raise CameraError(f"could not open camera: {device}")

        captured, frame = camera.read()

        if not captured or frame is None or frame.size == 0:
            raise CameraError(f"No image received from the camera: {device}")

        encoded_ok, encoded = cv2.imencode(".jpg", frame)

        if not encoded_ok or encoded.size == 0:
            raise CameraError(f"JPEG encoding failed for camera: {device}")

        return encoded.tobytes()

    except cv2.error as exc:
        raise CameraError(f"OpenCV failed while capturing from {device}") from exc

    finally:
        camera.release()
