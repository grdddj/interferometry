from pathlib import Path
from typing import List, Tuple
from matplotlib import pyplot as plt
import skimage
import cv2
import numpy as np
from scipy import optimize
import sys
from enum import Enum
from datetime import datetime


def get_new_image_dir() -> Path:
    """Creates and returns a new directory to save semiresults in, according to current date"""
    dir_to_save = Path(".") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_to_save.mkdir(parents=True, exist_ok=True)
    return dir_to_save


NEW_IMAGE_DIRECTORY = get_new_image_dir()


def is_in_script_args(arg: str) -> bool:
    """Checks if an argument is contained in the script arguments."""
    return any(arg in script_arg for script_arg in sys.argv[1:])


if is_in_script_args("print"):
    PRINT = True
else:
    PRINT = False

SAVE_RESULT = True


class Mode(Enum):
    """The different modes of the program."""

    NORMAL = 1
    NORMAL_AVERAGE = 2
    FIT_SINUS = 3


# Setting the current mode
MODE = Mode.NORMAL


# Default values
TAKE_AVERAGE_OF_X = None
FIT_BY_SINUS = False
DEFAULT_P0 = []

# Changing default values according to current mode
if MODE == Mode.NORMAL_AVERAGE:
    TAKE_AVERAGE_OF_X = 8
elif MODE == Mode.FIT_SINUS:
    FIT_BY_SINUS = True
    DEFAULT_P0 = [30, 0.05, 0, 40]

PHOTO_INDEX = 0
TOP_TO_BOTTOM_PHOTOS = [1, 2, 3, 4, 5, 6, 7, 9]
CURRENT_IMAGE = ""

ANGLE_START = (0, 64)
ANGLE_END = (200, 224)

TAKE_JUST_TWO = True
TAKE_JUST_TWO = False

# DEFAULT_P0 = [30, 0.2, 0, 40]

INDEXES_OPERATIONS = {1: {}}


def save_all_pictures(video_folder: Path, index_to_take: int = 0) -> None:
    """Saves a screenshot from each video in a folder."""
    all_files = video_folder.rglob("*.mp4")
    for file in all_files:
        save_picture(file, index_to_take)


def save_picture(video: Path, index_to_take: int = 0) -> None:
    """Saves a screenshot from a video."""
    # TODO: simplify to take just the first one
    vidcap = cv2.VideoCapture(str(video))
    success, image = vidcap.read()
    count = 0
    while success:
        if count == index_to_take:
            picture_name = f"photos/{video.stem}.jpg"
            print(f"saving {picture_name}")
            cv2.imwrite(picture_name, image)
            break
        success, image = vidcap.read()
        print(f"Read a new frame: {video.stem}")
        count += 1


def filter_items(item: float) -> bool:
    """Filters out some extreme values."""
    return 5 < item < 1000


def get_v(imax: float, imin: float) -> float:
    """Calculates the V value according to the minimum and maximum."""
    return (imax - imin) / (imax + imin)


def get_lowest_and_biggest_intensity(image, position: int) -> tuple[float, float]:
    """Calculates the lowest and biggest intensity of the image at the given position."""
    # Deciding where to take the "cut" on the image
    # (depends whether the photo is rightly horizontal or rotated)
    if PHOTO_INDEX in TOP_TO_BOTTOM_PHOTOS:
        start = (0, position)
        end = (image.shape[0] - 1, position)
    else:
        start = ANGLE_START
        end = ANGLE_END

    # Extract the real colour profile and take the integer values from it
    profile = skimage.measure.profile_line(image, start, end)
    values = [int(item[0]) for item in profile]

    # Getting rid of extremes at the beginning and the end
    del values[:25]
    del values[-25:]
    print("len(values)", len(values))
    print(values)

    if FIT_BY_SINUS:

        def test_func_sinus(x, a, b, c, d) -> float:
            """Test function for fitting - a sinus one."""
            return a * np.sin(b * x + c) + d

        # Need to convert python lists to numpy arrays
        x_data = np.array(list(range(len(values))))
        y_data = np.array(values)

        # Getting the parameters from the fit
        if (
            PHOTO_INDEX in INDEXES_OPERATIONS
            and "p0" in INDEXES_OPERATIONS[PHOTO_INDEX]
        ):
            p0 = INDEXES_OPERATIONS[PHOTO_INDEX]["p0"]
        else:
            # p0 = DEFAULT_P0
            p0 = calculate_p0(x_data, y_data)

        params, _ = optimize.curve_fit(test_func_sinus, x_data, y_data, p0=p0)
        print("params", params)

        # Constructing lowest and highest values from the fitted parameters
        lowest = params[-1] - abs(params[0])
        biggest = params[-1] + abs(params[0])

        # Saving or showing the semiresults
        if PRINT or SAVE_RESULT:
            fig, ax = plt.subplots(1, 2)

            ax[0].set_title(
                f"Mode: {MODE.name}\n"
                f"Index: {PHOTO_INDEX}\n"
                f"Image: {CURRENT_IMAGE}\n"
                f"p0: {list(round(x, 2) for x in p0)}\n"
                f"params: {list(round(x, 2) for x in params)}\n"
                f"lowest: {lowest:.2f}, biggest: {biggest:.2f}\n"
                f"visiblity: {get_v(biggest, lowest):.2f}"
            )
            ax[0].imshow(image)
            ax[0].plot([start[1], end[1]], [start[0], end[0]], "r")

            ax[1].set_title("y_data")
            ax[1].plot(y_data)

            ax[1].set_title("Our values and sinus fit")
            ax[1].plot(
                x_data,
                test_func_sinus(x_data, params[0], params[1], params[2], params[3]),
                label="Sinus fit",
            )
            ax[1].legend()

            if SAVE_RESULT:
                plt.tight_layout()
                my_dpi = 96
                fig.set_size_inches(14, 8)
                plt.savefig(NEW_IMAGE_DIRECTORY / f"{PHOTO_INDEX}.jpg", dpi=my_dpi)
                plt.close(fig)  # Closing the figure so that it is not shown
            elif PRINT:
                plt.show()
    else:
        # So that we can show it in the picture
        raw_values = values.copy()

        # Disregarding some extreme values and sorting
        values = [value for value in values if filter_items(value)]
        values.sort()

        # If we want, we can take the average of the first X values
        if TAKE_AVERAGE_OF_X is not None:
            biggest = sum(values[-TAKE_AVERAGE_OF_X:]) // TAKE_AVERAGE_OF_X
            lowest = sum(values[:TAKE_AVERAGE_OF_X]) // TAKE_AVERAGE_OF_X
        else:
            biggest = values[-1]
            lowest = values[0]

        # Optionally show the photo together with the profile
        if PRINT:
            fig, ax = plt.subplots(1, 2)
            ax[0].set_title(
                f"{MODE.name=}\n{PHOTO_INDEX=}\n{CURRENT_IMAGE}\n{lowest=}, {biggest=}\n{get_v(biggest, lowest)=}"
            )
            ax[0].imshow(image)
            ax[0].plot([start[1], end[1]], [start[0], end[0]], "r")
            ax[1].set_title("Profile / our filtered values")
            ax[1].plot(profile)
            ax[1].plot(raw_values, label="Our values")
            ax[1].legend()
            plt.show()

    return lowest, biggest


def calculate_p0(x_data, y_data) -> Tuple[float, float, float, float]:
    """Calculates the initial p0 for the fitting function according to the data."""
    ff = np.fft.fftfreq(len(x_data), (x_data[1] - x_data[0]))  # assume uniform spacing
    F_y_data = abs(np.fft.fft(y_data))
    guess_freq = abs(
        ff[np.argmax(F_y_data[1:]) + 1]
    )  # excluding the zero frequency "peak", which is related to offset
    guess_amp = 3 * np.std(y_data) * 2.0**0.5
    guess_offset = np.mean(y_data)
    guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])
    return tuple(guess)


def get_visibility_from_photo(photo: Path) -> float:
    image = skimage.io.imread(photo)

    # Exactly in half vertically
    positions = [image.shape[1] // 2]

    LOWESTS: list[float] = []
    BIGGEST: list[float] = []

    # Aggregating all the positions into its average
    for position in positions:
        low, big = get_lowest_and_biggest_intensity(image, position)
        LOWESTS.append(low)
        BIGGEST.append(big)

    avg_big = sum(BIGGEST) // len(BIGGEST)
    avg_low = sum(LOWESTS) // len(LOWESTS)
    print("BIGGEST", BIGGEST)
    print("LOWESTS", LOWESTS)

    visibility = get_v(avg_big, avg_low)
    print("visibility", visibility)

    return visibility


def get_visibility_graph(photo_folder: Path) -> None:
    global PHOTO_INDEX
    global CURRENT_IMAGE

    V_VALUES: list[float] = []
    all_photos = photo_folder.rglob("*.jpg")
    for index, photo in enumerate(all_photos, start=1):
        if TAKE_JUST_TWO:
            if index not in [12, 13]:
                continue

        PHOTO_INDEX = index
        CURRENT_IMAGE = str(photo)
        print(f"analyzing {photo}")
        v_value = get_visibility_from_photo(photo)
        V_VALUES.append(v_value)

    print("V_VALUES", V_VALUES)

    # Plot the final result together with some useful information
    plt.plot(V_VALUES)
    plt.title(
        f"Visibility\n{FIT_BY_SINUS=}, {DEFAULT_P0=}, {TAKE_AVERAGE_OF_X=}, {ANGLE_START=}, {ANGLE_END=}"
    )
    plt.ylabel("Visibility")

    plt.savefig(NEW_IMAGE_DIRECTORY / "RESULT.jpg")

    plt.show()


if __name__ == "__main__":
    # save_all_pictures(Path("videos"), 5)
    get_visibility_graph(Path("photos"))
