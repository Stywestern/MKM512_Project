# modules/tracker.py

##################################### Imports #####################################
# Libraries
import os
import numpy as np
from abc import ABC, abstractmethod

# Modules
import config
from modules.utils import log

###################################################################################

##################################################################################
#                              Tracker Blueprint
##################################################################################

class BaseTracker(ABC):
    @abstractmethod
    def track(self, frame):
        pass

##################################################################################
#                               BoT-SORT Tracker
##################################################################################

class SentryTracker:
    def __init__(self):
        # Here we would initialize the BoT-SORT / ByteTrack engine
        # For now, let's define the interface
        self.tracks = {} 
        log("Lego Block: Independent Tracker Initialized.", "INFO")

    def update(self, raw_boxes, frame):
        """
        Input: The (N, 5) array from the Detector
        Output: List of dicts with persistent IDs: [{'id': 1, 'bbox': [...]}, ...]
        """
        # This is where the 'Magic' happens. 
        # The tracker compares raw_boxes to its memory of previous frames.
        
        active_tracks = []
        
        # --- PLACEHOLDER FOR BOT-SORT ENGINE ---
        # online_targets = self.tracker.update(raw_boxes, frame)
        # for t in online_targets:
        #     active_tracks.append({"id": t.track_id, "bbox": t.tlbr})
        
        return active_tracks