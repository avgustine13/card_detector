import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import numpy as np


JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


@dataclass
class CameraOptions:
    camera_index: int
    backend: str
    width: int
    height: int
    fps: int
    jpeg_quality: int = 85


def has_rpicam() -> bool:
    try:
        result = subprocess.run(
            ["rpicam-hello", "--list-cameras"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


class OpenCvCapture:
    def __init__(self, options: CameraOptions) -> None:
        cap = cv2.VideoCapture(options.camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(options.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open camera index {options.camera_index} with OpenCV")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, options.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, options.height)
        cap.set(cv2.CAP_PROP_FPS, options.fps)
        self._cap = cap

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self._cap.read()
        return ok, frame if ok else None

    def release(self) -> None:
        self._cap.release()


class RpiCamCapture:
    def __init__(self, options: CameraOptions) -> None:
        self._last_error = ""
        self._buffer = bytearray()
        self._process = subprocess.Popen(
            [
                "rpicam-vid",
                "--camera",
                str(options.camera_index),
                "--nopreview",
                "--timeout",
                "0",
                "--codec",
                "mjpeg",
                "--width",
                str(options.width),
                "--height",
                str(options.height),
                "--framerate",
                str(options.fps),
                "--quality",
                str(options.jpeg_quality),
                "-o",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self._process.stdout is None or self._process.stderr is None:
            self.release()
            raise RuntimeError("failed to start rpicam-vid pipes")

        self._stdout = self._process.stdout
        self._stderr = self._process.stderr
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        try:
            for raw_line in iter(self._stderr.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line:
                    self._last_error = line
        finally:
            try:
                self._stderr.close()
            except OSError:
                pass

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        while True:
            chunk = self._stdout.read(4096)
            if not chunk:
                code = self._process.poll()
                if code is not None:
                    message = self._last_error or f"rpicam-vid exited with code {code}"
                    raise RuntimeError(message)
                return False, None

            self._buffer.extend(chunk)
            start = self._buffer.find(JPEG_SOI)
            if start < 0:
                if len(self._buffer) > 1024 * 1024:
                    del self._buffer[:-2]
                continue

            end = self._buffer.find(JPEG_EOI, start + 2)
            if end < 0:
                if start > 0:
                    del self._buffer[:start]
                continue

            end += 2
            jpeg = bytes(self._buffer[start:end])
            del self._buffer[:end]

            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            return True, frame

    def release(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()


def open_camera(options: CameraOptions) -> Tuple[Any, str]:
    backend = options.backend.strip().lower()
    if backend not in {"auto", "opencv", "rpicam"}:
        raise RuntimeError(f"unsupported backend: {options.backend}")

    if backend == "rpicam":
        return RpiCamCapture(options), "rpicam"
    if backend == "opencv":
        return OpenCvCapture(options), "opencv"

    if has_rpicam():
        try:
            return RpiCamCapture(options), "rpicam"
        except RuntimeError:
            pass
    return OpenCvCapture(options), "opencv"
