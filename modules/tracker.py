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
        self.tracker = BotSort(
            reid_weights=Path('osnet_x0_25_msmt17.pt'), 
            device=self.device, 
            half=False
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