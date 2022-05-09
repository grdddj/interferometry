"""
Getting a visibility graph from a collection of photos.

These photos can be automatically generated from the collection of
measurement videos.

Saves results from each run of this script into its own directory.
"""

from pathlib import Path

from helpers import get_value_from_script_args, is_in_script_args
from inter_profile import InterProfile, Mode

PRINT = is_in_script_args("print")
TAKE_JUST_TWO = is_in_script_args("take")
SAVE_RESULT = not is_in_script_args("notsave")
AVG_OF_X = None

CUT_LENGTH = int(get_value_from_script_args("length=", "100"))
CUT_AMOUNT = int(get_value_from_script_args("cuts=", "1"))

# Choose mode from script arguments, or default to NORMAL
if is_in_script_args("norm"):
    MODE = Mode.NORMAL
elif is_in_script_args("avg"):
    MODE = Mode.NORMAL_AVERAGE
    AVG_OF_X = int(get_value_from_script_args("avg=", "5"))
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
