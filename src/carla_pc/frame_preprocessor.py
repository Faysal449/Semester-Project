import struct
from typing import Tuple

import cv2
import numpy as np


class FramePreprocessor:
    """
    Prepares CARLA camera frames for TCP transmission.

    Expected CARLA raw image format:
        BGRA uint8, shape = (height, width, 4)

    Output protocol:
        Header: 12 bytes
            - payload_size (uint32)
            - width        (uint32)
            - height       (uint32)
        Payload:
            - grayscale image bytes, shape = (height, width)
    """

    HEADER_STRUCT = "!III"  
    HEADER_SIZE = struct.calcsize(HEADER_STRUCT)

    def __init__(
        self,
        target_width: int = 80,
        target_height: int = 60,
        grayscale: bool = True,
        jpeg_compression: bool = False,
        jpeg_quality: int = 80,
    ) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self.grayscale = grayscale
        self.jpeg_compression = jpeg_compression
        self.jpeg_quality = jpeg_quality

    def carla_to_numpy(self, image) -> np.ndarray:
        """
        Convert a CARLA image object into a BGRA NumPy array.
        """
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        return arr

    def preprocess(self, bgra_frame: np.ndarray) -> Tuple[bytes, int, int]:
        """
        Convert BGRA -> grayscale (or BGR), resize, and encode as raw bytes or JPEG.

        Returns:
            payload_bytes, width, height
        """
        if bgra_frame.ndim != 3 or bgra_frame.shape[2] != 4:
            raise ValueError(
                f"Expected BGRA frame with shape (H, W, 4), got {bgra_frame.shape}"
            )

      
        bgr = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2BGR)

       
        resized = cv2.resize(
            bgr,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA,
        )

        if self.grayscale:
            processed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            processed = resized

        if self.jpeg_compression:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            ok, encoded = cv2.imencode(".jpg", processed, encode_params)
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            payload = encoded.tobytes()
        else:
            payload = processed.tobytes()

        height, width = processed.shape[:2]
        return payload, width, height

    def build_packet(self, bgra_frame: np.ndarray) -> bytes:
        """
        Build a full packet = header + payload.
        """
        payload, width, height = self.preprocess(bgra_frame)
        header = struct.pack(self.HEADER_STRUCT, len(payload), width, height)
        return header + payload
