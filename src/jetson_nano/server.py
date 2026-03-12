import socket
import struct
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np


HEADER_STRUCT = "!III"  
HEADER_SIZE = struct.calcsize(HEADER_STRUCT)


def recv_exact(sock: socket.socket, size: int) -> Optional[bytes]:
    """
    Receive exactly `size` bytes from the socket.
    Returns None if the connection is closed before enough bytes arrive.
    """
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def decode_frame(payload: bytes, width: int, height: int) -> np.ndarray:
    """
    Decode raw grayscale payload into a NumPy array.
    """
    expected_size = width * height
    if len(payload) != expected_size:
        raise ValueError(
            f"Payload size mismatch: got {len(payload)}, expected {expected_size}"
        )

    frame = np.frombuffer(payload, dtype=np.uint8).reshape((height, width))
    return frame


def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Put your Jetson-side image processing here.
    For now:
    - verifies frame
    - can optionally apply blur / edge detection later
    """
    return frame


def start_server(host: str = "0.0.0.0", port: int = 5000, show_preview: bool = False) -> None:
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)

    print(f"[SERVER] Listening on {host}:{port}")

    while True:
        print("[SERVER] Waiting for connection...")
        conn, addr = server_sock.accept()
        print(f"[SERVER] Connected by {addr}")

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                header = recv_exact(conn, HEADER_SIZE)
                if header is None:
                    print("[SERVER] Client disconnected while reading header.")
                    break

                payload_size, width, height = struct.unpack(HEADER_STRUCT, header)

                payload = recv_exact(conn, payload_size)
                if payload is None:
                    print("[SERVER] Client disconnected while reading payload.")
                    break

                frame = decode_frame(payload, width, height)
                processed = process_frame(frame)

                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0

                print(f"[SERVER] Frame ok: {processed.shape} | count={frame_count} | fps={fps:.2f}")

                if show_preview:
                    cv2.imshow("Jetson Grayscale Stream", processed)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[SERVER] Preview stopped by user.")
                        break

        except Exception as exc:
            print(f"[SERVER] Error: {exc}")

        finally:
            conn.close()
            print("[SERVER] Connection closed.")

            if show_preview:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 5000
    show_preview = False

    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    if len(sys.argv) > 3:
        show_preview = bool(int(sys.argv[3]))

    start_server(host=host, port=port, show_preview=show_preview)
