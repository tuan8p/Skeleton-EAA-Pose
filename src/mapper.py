"""Map 33 BlazePose landmarks -> 25 NTU joints (fixed mapping table).

Thu tu 25 khop NTU:
0 SpineBase, 1 SpineMid, 2 Neck, 3 Head,
4 ShoulderL, 5 ElbowL, 6 WristL, 7 HandL,
8 ShoulderR, 9 ElbowR, 10 WristR, 11 HandR,
12 HipL, 13 KneeL, 14 AnkleL, 15 FootL,
16 HipR, 17 KneeR, 18 AnkleR, 19 FootR,
20 SpineShoulder, 21 HandTipL, 22 ThumbL, 23 HandTipR, 24 ThumbR
"""
from __future__ import annotations

import numpy as np

# NTU joint -> BlazePose joints to average
BLAZE_TO_NTU: list[list[int]] = [
    [23, 24],        # 0  SpineBase = mid-hip
    [11, 12, 23, 24],  # 1  SpineMid
    [11, 12],        # 2  Neck
    [0],             # 3  Head = nose
    [11],            # 4  ShoulderLeft
    [13],            # 5  ElbowLeft
    [15],            # 6  WristLeft
    [17, 19, 21],    # 7  HandLeft = mean(pinky, index, thumb)
    [12],            # 8  ShoulderRight
    [14],            # 9  ElbowRight
    [16],            # 10 WristRight
    [18, 20, 22],    # 11 HandRight
    [23],            # 12 HipLeft
    [25],            # 13 KneeLeft
    [27],            # 14 AnkleLeft
    [31],            # 15 FootLeft = foot_index
    [24],            # 16 HipRight
    [26],            # 17 KneeRight
    [28],            # 18 AnkleRight
    [32],            # 19 FootRight
    [11, 12],        # 20 SpineShoulder
    [19],            # 21 HandTipLeft = index
    [21],            # 22 ThumbLeft
    [20],            # 23 HandTipRight
    [22],            # 24 ThumbRight
]

NTU_BONES: list[tuple[int, int]] = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (6, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (10, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

NUM_NTU_JOINTS = 25


class Mapper:
    """Map 33-landmark BlazePose output to 25 NTU joints. Static methods for easy testing."""

    @staticmethod
    def map_coords(landmarks: np.ndarray) -> np.ndarray:
        """landmarks (33, C) -> (25, C), C = 3 (xyz)."""
        out = np.zeros((NUM_NTU_JOINTS, landmarks.shape[1]), dtype=np.float32)
        for j, src in enumerate(BLAZE_TO_NTU):
            out[j] = landmarks[src].mean(axis=0)
        return out

    @staticmethod
    def map_visibility(visibility: np.ndarray) -> np.ndarray:
        """visibility (33,) -> (25,) by averaging component joint visibilities."""
        out = np.zeros(NUM_NTU_JOINTS, dtype=np.float32)
        for j, src in enumerate(BLAZE_TO_NTU):
            out[j] = visibility[src].mean()
        return out
