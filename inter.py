"""
Getting a visibility graph from a collection of photos.

These photos can be automatically generated from the collection of
measurement videos.

Saves results from each run of this script into its own directory.
"""

import sys
from pathlib import Path

from helpers import is_in_script_args
from inter_profile import InterProfile, Mode

PRINT = is_in_script_args("print")
TAKE_JUST_TWO = is_in_script_args("take")
SAVE_RESULT = not is_in_script_args("notsave")
AVG_OF_X = None

# Optionally getting the cut length
CUT_LENGTH = 100  # default
for script_arg in sys.argv[1:]:
    if "length=" in script_arg:
        CUT_LENGTH = int(script_arg.split("=")[1])

# Optionally getting the cut length
CUT_AMOUNT = 1  # default
for script_arg in sys.argv[1:]:
    if "cuts=" in script_arg:
        CUT_AMOUNT = int(script_arg.split("=")[1])

# Choose mode from script arguments, or default to NORMAL
if is_in_script_args("norm"):
    MODE = Mode.NORMAL
elif is_in_script_args("avg"):
    MODE = Mode.NORMAL_AVERAGE
    # Look into script arguments for the number of images to average, if there
    AVG_OF_X = 5  # default
    for script_arg in sys.argv[1:]:
        if "avg=" in script_arg:
            AVG_OF_X = int(script_arg.split("=")[1])
elif is_in_script_args("sin"):
    MODE = Mode.FIT_SINUS
else:
    MODE = Mode.NORMAL


if __name__ == "__main__":
    interfero_profile = InterProfile(
        MODE=MODE,
        PRINT=PRINT,
        SAVE_RESULT=SAVE_RESULT,
        CUT_LENGTH=CUT_LENGTH,
        CUT_AMOUNT=CUT_AMOUNT,
        AVG_OF_X=AVG_OF_X,
    )
    # interfero_profile.save_all_pictures(Path("videos"), 5)
    interfero_profile.get_visibility_graph(Path("photos"))
