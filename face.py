# face.py
import os
import io
import base64
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import cv2
import numpy as np


@dataclass
class VideoConfig:
    """
    - sample_per_sec: 초당 몇 장의 프레임을 뽑을지 (1.0 = 1초마다 1장)
    - max_frames: 너무 긴 영상일 때 프레임 최대 개수 제한 (None 이면 제한 없음)
    - resize_width: 프레임 크기를 줄이고 싶으면 지정 (None이면 원본 크기)
    """
    sample_per_sec: float = 1.0
    max_frames: Optional[int] = None
    resize_width: Optional[int] = 640


class FaceVideoProcessor:
    """
    동영상 bytes를 받아서:
    - 1초 단위로 프레임을 잘라서
    - numpy 배열 또는 base64(JPEG)로 반환
    """

    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()

    # ==============================
    # 내부 유틸: 동영상 bytes → cv2.VideoCapture
    # ==============================
    def _bytes_to_videocap(self, video_bytes: bytes) -> cv2.VideoCapture:
        """
        Windows에서도 잘 동작하도록, NamedTemporaryFile을
        닫은 뒤 다시 여는 방식으로 처리
        """
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        try:
            tmp.write(video_bytes)
            tmp.flush()
            tmp_path = tmp.name
        finally:
            tmp.close()  # Windows에서 파일 잠김 방지

        cap = cv2.VideoCapture(tmp_path)

        # 나중에 cap.release() 후에 파일 삭제
        cap._tmp_path = tmp_path  # 편하게 저장
        return cap

    def _cleanup_videocap(self, cap: cv2.VideoCapture):
        """
        VideoCapture와 임시 파일 정리
        """
        tmp_path = getattr(cap, "_tmp_path", None)
        cap.release()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # ==============================
    # 내부 유틸: 프레임 리사이즈 및 JPEG 인코딩
    # ==============================
    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        if self.config.resize_width is None:
            return frame
        h, w = frame.shape[:2]
        if w <= self.config.resize_width:
            return frame
        scale = self.config.resize_width / float(w)
        new_size = (self.config.resize_width, int(h * scale))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    def _frame_to_jpeg_bytes(self, frame: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Failed to encode frame to JPEG")
        return buf.tobytes()

    # ==============================
    # 핵심: bytes → 1초 단위 프레임 리스트 (numpy)
    # ==============================
    def extract_frames_from_bytes(self, video_bytes: bytes) -> List[np.ndarray]:
        """
        1초 단위로 프레임 추출 (sample_per_sec 기준)
        return: [frame(BGR numpy), ...]
        """
        cap = self._bytes_to_videocap(video_bytes)

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0  # FPS 정보를 못 읽으면 기본값 30으로 가정

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            frames: List[np.ndarray] = []

            t = 0.0
            while t < duration:
                frame_idx = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    break

                frame = self._resize_if_needed(frame)
                frames.append(frame)

                if self.config.max_frames is not None and len(frames) >= self.config.max_frames:
                    break

                # 다음 샘플 시각 (1초마다 1장, 또는 sample_per_sec 기준)
                t += 1.0 / self.config.sample_per_sec

        finally:
            self._cleanup_videocap(cap)

        return frames

    # ==============================
    # bytes → 1초 단위 프레임(JPEG base64) 리스트
    # ==============================
    def extract_frames_base64(self, video_bytes: bytes) -> List[str]:
        """
        안드로이드로 프레임 이미지를 보내고 싶을 때 사용.
        각 프레임을 base64(JPEG)로 인코딩해서 반환.
        """
        frames = self.extract_frames_from_bytes(video_bytes)

        b64_list: List[str] = []
        for frame in frames:
            jpeg_bytes = self._frame_to_jpeg_bytes(frame)
            b64_str = base64.b64encode(jpeg_bytes).decode("ascii")
            b64_list.append(b64_str)

        return b64_list
