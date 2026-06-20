"""
eaa_pose.pose.mapping_25
========================
Maps RTMW3D's 133 COCO-Wholebody keypoints to the NTU-120 / PKU-MMD
25-joint skeleton format.

NTU-120 joint order (1-based in the original paper, 0-based here)
------------------------------------------------------------------
Index  Name
-----  ----
 0     SpineBase
 1     SpineMid
 2     Neck
 3     Head
 4     LShoulder
 5     LElbow
 6     LWrist
 7     LHand
 8     RShoulder
 9     RElbow
10     RWrist
11     RHand
12     LHip
13     LKnee
14     LAnkle
15     LFoot
16     RHip
17     RKnee
18     RAnkle
19     RFoot
20     SpineShoulder
21     LHandTip
22     LThumb
23     RHandTip
24     RThumb

COCO-Wholebody 133-keypoint indices (subset used here)
-------------------------------------------------------
Body (0-16, COCO-17):
  0=nose  1=l_eye  2=r_eye  3=l_ear  4=r_ear
  5=l_shoulder  6=r_shoulder  7=l_elbow  8=r_elbow
  9=l_wrist  10=r_wrist  11=l_hip  12=r_hip
  13=l_knee  14=r_knee  15=l_ankle  16=r_ankle

Foot (17-22):
  17=l_big_toe  18=l_small_toe  19=l_heel
  20=r_big_toe  21=r_small_toe  22=r_heel

Left hand (91-111):
  91=l_wrist  92-111 = l fingers

Right hand (112-132):
  112=r_wrist  113-132 = r fingers

Face (23-90) — not used for body mapping.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# COCO-Wholebody index constants (for clarity in the mapping code)
# ---------------------------------------------------------------------------

_C = {
    # Body
    "nose":        0,
    "l_eye":       1,  "r_eye":       2,
    "l_ear":       3,  "r_ear":       4,
    "l_shoulder":  5,  "r_shoulder":  6,
    "l_elbow":     7,  "r_elbow":     8,
    "l_wrist":     9,  "r_wrist":     10,
    "l_hip":       11, "r_hip":       12,
    "l_knee":      13, "r_knee":      14,
    "l_ankle":     15, "r_ankle":     16,
    # Foot
    "l_big_toe":   17, "l_small_toe": 18, "l_heel": 19,
    "r_big_toe":   20, "r_small_toe": 21, "r_heel": 22,
    # Left hand (fingertips within 91-111)
    "l_hand_root": 91,
    "l_index_tip": 95,   # approximate fingertip
    "l_thumb_tip": 92,
    # Right hand (fingertips within 112-132)
    "r_hand_root": 112,
    "r_index_tip": 116,
    "r_thumb_tip": 113,
    # Face head center approximation
    "l_face_edge": 23,   # leftmost face point
    "r_face_edge": 46,   # rightmost face point
    "chin":        8,    # Note: in 68-pt face set, 0=chin; 23+8=31 here
    # Use nose for head center as fallback
}

# Face keypoints in COCO-Wholebody start at index 23.
# The 68-point face layout: 0=chin/jaw, 8=nose-bridge, 27=nose tip...
# We approximate head center as the mean of outer face ring points.
_FACE_START = 23
_FACE_END   = 90   # inclusive


class NTU25Mapper:
    """Maps 133-keypoint COCO-Wholebody poses to NTU-120 25-joint format.

    Computed joints (SpineBase, SpineMid, SpineShoulder, Neck, Head,
    LHand/RHand, HandTip, Thumb, Foot) are derived from midpoints or
    averages of existing keypoints.  Direct body joints (shoulders,
    elbows, wrists, hips, knees, ankles) are copied directly.

    Parameters
    ----------
    conf_for_computed:
        Confidence assigned to joints computed as midpoints.  Set to the
        minimum of the constituent joint confidences scaled by this factor.
    """

    N_JOINTS = 25

    def __init__(self, conf_for_computed: float = 0.9) -> None:
        self._conf_scale = conf_for_computed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map(self, keypoints_133: np.ndarray) -> np.ndarray:
        """Map one person's 133-keypoint array to 25 NTU joints.

        Parameters
        ----------
        keypoints_133:
            Numpy array of shape ``(133, 4)`` with columns
            ``[x, y, z, confidence]``.

        Returns
        -------
        Numpy array of shape ``(25, 4)`` — same column layout.
        """
        kp = keypoints_133   # (133, 4)
        out = np.zeros((self.N_JOINTS, 4), dtype=np.float32)

        # ---------------------------------------------------------------
        # Direct mappings (single source keypoint)
        # ---------------------------------------------------------------
        direct: list[tuple[int, int]] = [
            # (NTU_index, COCO_index)
            (4,  _C["l_shoulder"]),
            (5,  _C["l_elbow"]),
            (6,  _C["l_wrist"]),
            (8,  _C["r_shoulder"]),
            (9,  _C["r_elbow"]),
            (10, _C["r_wrist"]),
            (12, _C["l_hip"]),
            (13, _C["l_knee"]),
            (14, _C["l_ankle"]),
            (16, _C["r_hip"]),
            (17, _C["r_knee"]),
            (18, _C["r_ankle"]),
        ]
        for ntu_i, coco_i in direct:
            out[ntu_i] = kp[coco_i]

        # ---------------------------------------------------------------
        # Computed joints (midpoints / averages)
        # ---------------------------------------------------------------

        # 0  SpineBase = midpoint(LHip, RHip)
        out[0] = self._midpoint(kp, _C["l_hip"], _C["r_hip"])

        # 20 SpineShoulder = midpoint(LShoulder, RShoulder)
        out[20] = self._midpoint(kp, _C["l_shoulder"], _C["r_shoulder"])

        # 1  SpineMid = midpoint(SpineBase, SpineShoulder)
        out[1] = self._midpoint_arr(out[0], out[20])

        # 2  Neck = midpoint(SpineShoulder, Head)
        #    We compute head first, then neck
        out[3] = self._head_center(kp)  # 3 = Head

        out[2] = self._midpoint_arr(out[20], out[3])  # 2 = Neck

        # 7  LHand = l_hand_root (or l_wrist as fallback)
        out[7] = self._best_of(kp, [_C["l_hand_root"], _C["l_wrist"]])

        # 11 RHand = r_hand_root (or r_wrist as fallback)
        out[11] = self._best_of(kp, [_C["r_hand_root"], _C["r_wrist"]])

        # 21 LHandTip = l_index fingertip proxy
        out[21] = self._best_of(kp, [_C["l_index_tip"], _C["l_hand_root"], _C["l_wrist"]])

        # 22 LThumb = l_thumb_tip proxy
        out[22] = self._best_of(kp, [_C["l_thumb_tip"], _C["l_hand_root"], _C["l_wrist"]])

        # 23 RHandTip = r_index fingertip proxy
        out[23] = self._best_of(kp, [_C["r_index_tip"], _C["r_hand_root"], _C["r_wrist"]])

        # 24 RThumb = r_thumb_tip proxy
        out[24] = self._best_of(kp, [_C["r_thumb_tip"], _C["r_hand_root"], _C["r_wrist"]])

        # 15 LFoot = average(l_big_toe, l_small_toe, l_heel)
        out[15] = self._average(kp, [_C["l_big_toe"], _C["l_small_toe"], _C["l_heel"]])

        # 19 RFoot = average(r_big_toe, r_small_toe, r_heel)
        out[19] = self._average(kp, [_C["r_big_toe"], _C["r_small_toe"], _C["r_heel"]])

        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _midpoint(kp: np.ndarray, i: int, j: int) -> np.ndarray:
        """Midpoint of two keypoints; confidence = min(conf_i, conf_j)."""
        result = (kp[i] + kp[j]) / 2.0
        result[3] = min(kp[i, 3], kp[j, 3])  # conservative confidence
        return result.astype(np.float32)

    @staticmethod
    def _midpoint_arr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Midpoint of two already-computed joint arrays."""
        result = (a + b) / 2.0
        result[3] = min(a[3], b[3])
        return result.astype(np.float32)

    @staticmethod
    def _best_of(kp: np.ndarray, indices: list[int]) -> np.ndarray:
        """Return the keypoint with the highest confidence from a list."""
        best = kp[indices[0]]
        for i in indices[1:]:
            if kp[i, 3] > best[3]:
                best = kp[i]
        return best.astype(np.float32)

    @staticmethod
    def _average(kp: np.ndarray, indices: list[int]) -> np.ndarray:
        """Weighted average of keypoints; confidence = mean of conf."""
        stacked = kp[indices]
        result = stacked.mean(axis=0)
        result[3] = stacked[:, 3].mean()
        return result.astype(np.float32)

    @staticmethod
    def _head_center(kp: np.ndarray) -> np.ndarray:
        """Estimate head center from face keypoints or nose fallback.

        Uses the mean of the outer jaw / face contour keypoints (indices
        23-36 in the 68-pt face layout embedded at COCO offset 23).
        Falls back to the nose keypoint if face keypoints are unreliable.
        """
        # Face contour: COCO indices 23-39 (17 points of jaw/chin line)
        face_contour = kp[23:40]  # (17, 4)
        conf = face_contour[:, 3]

        if conf.max() > 0.1:
            # Use only confident face contour points
            valid = face_contour[conf > 0.1]
            center = valid.mean(axis=0)
            center[3] = conf[conf > 0.1].mean()
            return center.astype(np.float32)

        # Fallback: use nose point
        return kp[_C["nose"]].copy().astype(np.float32)
