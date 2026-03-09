# modules/tracker.py

##################################### Imports #####################################
# Libraries
import os
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path

from boxmot import BotSort, ByteTrack
import torch

# Modules
import config
from modules.utils import log

###################################################################################

##################################################################################
#                              Tracker Blueprint
##################################################################################

class BaseTracker(ABC):
    @abstractmethod
    def update(self, frame):
        pass

    @abstractmethod
    def _format_output(self, tracks):
        pass


##################################################################################
#                                BoT-SORT Tracker
##################################################################################

class BoTSORTTracker(BaseTracker):
    def __init__(self):
        self.device = 0 if config.RUN_ON_GPU else 'cpu'
        model_path = os.path.join("assets", "models", "osnet_x0_25_msmt17.pt")

        self.tracker = BotSort(
            reid_weights=model_path, 
            device=self.device, 
            half=False,

            # --- Tweakable Parameters ---
            track_high_thresh=0.45, # Lower this slightly so it's easier to START a track
            track_low_thresh=0.1,  # Keep tracking even if detection score is 10%
            new_track_thresh=0.6,  # Be strict about NEW IDs to prevent ghosts
            track_buffer=60,       # Remember the face for 60 frames (~2 seconds) after it vanishes
            match_thresh=0.85,     # Increase Re-ID weight to favor "look" over "position"
            proximity_thresh=0.5,  # Spatial distance threshold
            appearance_thresh=0.25, # Feature distance threshold
            cmc_method='orb'       # Compensates for the turret's own movements
        )

        log("Tracker: BoT-SORT Block Initialized.", "INFO")

    def update(self, raw_detections, frame):
        if raw_detections is None or len(raw_detections) == 0:
            tracks = self.tracker.update(np.empty((0, 6)), frame)
        else:
            tracks = self.tracker.update(raw_detections, frame)
            
        return self._format_output(tracks)

    def _format_output(self, tracks):
        detections = []
        for t in tracks:
            # BoxMOT Output: [x1, y1, x2, y2, id, conf, cls, ind]
            x1, y1, x2, y2, track_id = t[:5]
            detections.append({
                "id": int(track_id),
                "face_bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": (int((x1+x2)/2), int((y1+y2)/2))
            })
        return detections

##################################################################################
#                                ByteTrack Tracker
##################################################################################

class ByteTrackTracker(BaseTracker):
    def __init__(self):
        self.device = 0 if config.RUN_ON_GPU else 'cpu'
        self.tracker = ByteTrack(
            device=self.device, 
            half=False
        )
        log("Tracker: ByteTrack Block Initialized.", "INFO")

    def update(self, raw_detections, frame):
        if raw_detections is None or len(raw_detections) == 0:
            tracks = self.tracker.update(np.empty((0, 6)), frame)
        else:
            tracks = self.tracker.update(raw_detections, frame)
            
        # We reuse the same formatting logic (could be moved to BaseTracker)
        return self._format_output(tracks)

    def _format_output(self, tracks):
        detections = []
        for t in tracks:
            # BoxMOT Output: [x1, y1, x2, y2, id, conf, cls, ind]
            x1, y1, x2, y2, track_id = t[:5]
            detections.append({
                "id": int(track_id),
                "face_bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": (int((x1+x2)/2), int((y1+y2)/2))
            })
        return detections