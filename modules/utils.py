# modules/detector.py

##################################### Imports #####################################
# Libraries
from datetime import datetime

###################################################################################

# Custom Logger
def log(message, level="INFO"):
    """
    Standardized logger for the project.
    Levels: INFO for standart stuff, WARNING for weird occasions, ERROR for unwanted behaviour
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {message}")