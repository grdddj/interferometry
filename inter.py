"""
Getting a visibility graph from a collection of photos.

These photos can be automatically generated from the collection of
measurement videos.

Saves results from each run of this script into its own directory.
"""

import math
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import skimage
from matplotlib import pyplot as plt
from scipy import optimize


class Mode(Enum):
    """The different modes of the program."""

    NORMAL = 1
    NORMAL_AVERAGE = 2
    FIT_SINUS = 3


def is_in_script_args(arg: str) -> bool:
    """Checks if an argument is contained in the script arguments."""
    return any(arg in script_arg for script_arg in sys.argv[1:])


PRINT = is_in_script_args("print")
TAKE_JUST_TWO = is_in_script_args("take")
SAVE_RESULT = not is_in_script_args("notsave")
AVG_OF_X = None

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


class InterProfile:
    def __init__(
        self,
        MODE: Mode,
        PRINT: bool,
        SAVE_RESULT: bool,
        TAKE_JUST_TWO: bool,
        AVG_OF_X: int | None,
    ) -> None:
        self.MODE = MODE
        self.PRINT = PRINT
        self.SAVE_RESULT = SAVE_RESULT
        self.TAKE_JUST_TWO = TAKE_JUST_TWO
        self.AVG_OF_X = AVG_OF_X

        self.NEW_IMAGE_DIRECTORY = self._get_new_image_dir(comment=self.MODE.name)

        self.PHOTO_INDEX = 0
        self.CURRENT_IMAGE = ""

        # Indexes of pictures that are correctly horizontal (all others are rotated)
        self.TOP_TO_BOTTOM_PHOTOS = [1, 2, 3, 4, 5, 6, 7, 9]
        # How many pixels to neglect both from top and bottom for horizontal pictures
        self.VERTICAL_OFFSET = 100

        # Coordinates of start and length of the cut of the rotated picture to get profile
        # End will be determined automatically according to the angle
        self.ANGLE_START = (25, 110)
        self.CUT_LENGTH = 50

    def get_visibility_graph(self, photo_folder: Path) -> None:
        V_VALUES: list[float] = []
        all_photos = photo_folder.rglob("*.jpg")
        for index, photo in enumerate(all_photos, start=1):
            if self.TAKE_JUST_TWO:
                if index not in [12, 13]:
                    continue

            self.PHOTO_INDEX = index
            self.CURRENT_IMAGE = str(photo)
            print(f"analyzing {photo}")
            v_value = self._get_visibility_from_photo(photo)
            V_VALUES.append(v_value)

        print("V_VALUES", V_VALUES)

        # Plot the final result together with some useful information
        plt.plot(V_VALUES)
        title_str = f"Visibility\nMode: {self.MODE.name}"
        if self.MODE == Mode.NORMAL_AVERAGE:
            title_str += f" (avg {self.AVG_OF_X})"
        plt.title(title_str)
        plt.ylabel("Visibility")
        plt.xlabel("Image index")

        plt.savefig(self.NEW_IMAGE_DIRECTORY / "RESULT.jpg")
        plt.show()

    def save_all_pictures(self, video_folder: Path, index_to_take: int = 0) -> None:
        """Saves a screenshot from each video in a folder."""
        all_files = video_folder.rglob("*.mp4")
        for file in all_files:
            self._save_picture(file, index_to_take)

    def _get_visibility_from_photo(self, photo: Path) -> float:
        image = skimage.io.imread(photo)

        # Deciding where to take the "cut" on the image
        # (depends whether the photo is rightly horizontal or rotated)
        if self.PHOTO_INDEX in self.TOP_TO_BOTTOM_PHOTOS:
            # Exactly in half vertically
            middle_vertical = image.shape[1] // 2
            start = (self.VERTICAL_OFFSET, middle_vertical)
            end = (image.shape[0] - self.VERTICAL_OFFSET, middle_vertical)
        else:
            black_start = self._get_black_start(photo)
            angle = self._get_angle(photo, black_start)
            perpendicular_angle = 90 - angle
            radian_perpendicular_angle = math.radians(perpendicular_angle)
            start = self.ANGLE_START
            end = (
                self.ANGLE_START[0]
                + self.CUT_LENGTH * math.cos(radian_perpendicular_angle),
                self.ANGLE_START[1]
                + self.CUT_LENGTH * math.sin(radian_perpendicular_angle),
            )

        low, big = self._get_lowest_and_biggest_intensity(image, start, end)

        visibility = self._get_v(low, big)

        return visibility

    def _get_angle(self, photo: Path, black_start: tuple[int, int]) -> int:
        """Get the angle of the black line so that we can cut the picture perpendicular to it."""
        image = skimage.io.imread(photo)

        start_x = black_start[0]
        start_y = black_start[1]

        degrees_thresholds = {}
        for degrees in range(91):
            radians = degrees * 2 * np.pi / 360

            # Make the length as long as possible
            length = int(start_x / math.cos(radians))
            if length + start_y > image.shape[1]:
                length = image.shape[1] - start_y

            end = (
                int(start_x - np.cos(radians) * length),
                int(start_y + np.sin(radians) * length),
            )

            profile = skimage.measure.profile_line(image, black_start, end)
            values = [int(item[0]) for item in profile]
            values.sort()

            # 90 percent of values are less that the threshold
            threshold = values[int(len(values) * 0.9)]
            degrees_thresholds[degrees] = threshold

        lowest = min(degrees_thresholds.items(), key=lambda x: x[1])

        return lowest[0]

    def _get_black_start(self, photo: Path) -> tuple[int, int]:
        """Find out a black start point of the profile. Needed to find out the angle."""
        image = skimage.io.imread(photo)

        # Taking two lines next to each other and creating an average of them
        start_x = 200
        y = 20
        start1 = (start_x, y)
        start2 = (start_x, y + 1)
        end1 = (500, y)
        end2 = (500, y + 1)

        values1 = [
            int(item[0]) for item in skimage.measure.profile_line(image, start1, end1)
        ]
        values2 = [
            int(item[0]) for item in skimage.measure.profile_line(image, start2, end2)
        ]
        values = [values1[i] + values2[i] for i in range(len(values1))]
        values = [item // 2 for item in values]

        values_copy = values.copy()
        values.sort()

        # 20 percent of values are less that the threshold
        threshold = values[int(len(values) * 0.2)]

        # Finding all the streaks of pixels that are below the threshold
        found = False
        indexes = []
        new_indexes = []
        for index, val in enumerate(values_copy):
            if val < threshold:
                found = True
                new_indexes.append(index)
            else:
                if found:
                    found = False
                    indexes.append(new_indexes)
                    new_indexes = []

        # Extracting the longest streak
        indexes.sort(key=len, reverse=True)
        indexes_to_take = indexes[0]
        index_to_take = indexes_to_take[len(indexes_to_take) // 2]

        return (start_x + index_to_take, y)

    def _get_lowest_and_biggest_intensity(
        self, image, start: tuple[int, int], end: tuple[int, int]
    ) -> tuple[float, float]:
        """Calculates the lowest and biggest intensity of the image at the given position."""
        # Extract the real colour profile and take the integer values from it
        profile = skimage.measure.profile_line(image, start, end)
        values = [int(item[0]) for item in profile]

        if MODE == Mode.FIT_SINUS:

            def test_func_sinus(x, a, b, c, d) -> float:
                """Test function for fitting - a sinus one."""
                return a * np.sin(b * x + c) + d

            # Need to convert python lists to numpy arrays
            x_data = np.array(list(range(len(values))))
            y_data = np.array(values)

            # Get the initial guess for the parameters
            p0 = self._calculate_p0(x_data, y_data)

            # Get the sinus fit parameters
            params, _ = optimize.curve_fit(test_func_sinus, x_data, y_data, p0=p0)

            # Constructing lowest and highest values from the fitted parameters
            lowest = params[-1] - abs(params[0])
            biggest = params[-1] + abs(params[0])

            # Saving or showing the semiresults
            if self.PRINT or self.SAVE_RESULT:
                fig, ax = plt.subplots(1, 2)

                ax[0].set_title(
                    f"Mode: {MODE.name}\n"
                    f"Index: {self.PHOTO_INDEX}\n"
                    f"Image: {self.CURRENT_IMAGE}\n"
                    f"p0: {list(round(x, 2) for x in p0)}\n"
                    f"params: {list(round(x, 2) for x in params)}\n"
                    f"lowest: {lowest:.2f}, biggest: {biggest:.2f}\n"
                    f"visibility: {self._get_v(lowest, biggest):.2f}"
                )
                ax[0].imshow(image)
                ax[0].plot([start[1], end[1]], [start[0], end[0]], "r", label="cut")
                ax[0].legend()

                ax[1].set_title("Profile data and sinus fit")
                ax[1].plot(y_data, label="Profile data")
                ax[1].plot(
                    x_data,
                    test_func_sinus(x_data, params[0], params[1], params[2], params[3]),
                    label="Sinus fit",
                )
                ax[1].legend()

                if self.PRINT:
                    plt.show()
                elif self.SAVE_RESULT:
                    plt.tight_layout()
                    fig.set_size_inches(14, 8)
                    plt.savefig(
                        self.NEW_IMAGE_DIRECTORY / f"{self.PHOTO_INDEX}.jpg", dpi=96
                    )
                    plt.close(fig)  # Closing the figure so that it is not shown
        else:
            # So that we can show it in the picture
            raw_values = values.copy()

            # Disregarding some extreme values and sorting
            values = [value for value in values if self._filter_items(value)]
            values.sort()

            # If we want, we can take the average of the first X values
            if self.AVG_OF_X is not None:
                biggest = sum(values[-self.AVG_OF_X :]) // self.AVG_OF_X
                lowest = sum(values[: self.AVG_OF_X]) // self.AVG_OF_X
            else:
                biggest = values[-1]
                lowest = values[0]

            # Optionally show the photo together with the profile
            if self.PRINT or self.SAVE_RESULT:
                fig, ax = plt.subplots(1, 2)

                ax[0].set_title(
                    f"Mode: {self.MODE.name}\n"
                    f"Index: {self.PHOTO_INDEX}\n"
                    f"Image: {self.CURRENT_IMAGE}\n"
                    f"lowest: {lowest:.2f}, biggest: {biggest:.2f}\n"
                    f"visibility: {self._get_v(lowest, biggest):.2f}"
                )
                ax[0].imshow(image)
                ax[0].plot([start[1], end[1]], [start[0], end[0]], "r")
                ax[0].set_ylabel("Pixels")
                ax[0].set_xlabel("Pixels")

                ax[1].set_title("Profile / our filtered values")
                ax[1].plot(raw_values, label="Profile values")
                ax[1].legend()
                ax[1].set_xlabel("Pixel index")

                if self.PRINT:
                    plt.show()
                elif self.SAVE_RESULT:
                    plt.tight_layout()
                    fig.set_size_inches(14, 8)
                    plt.savefig(
                        self.NEW_IMAGE_DIRECTORY / f"{self.PHOTO_INDEX}.jpg", dpi=96
                    )
                    plt.close(fig)  # Closing the figure so that it is not shown

        return lowest, biggest

    @staticmethod
    def _calculate_p0(x_data, y_data) -> Tuple[float, float, float, float]:
        """Calculates the initial p0 for the fitting function according to the data."""
        ff = np.fft.fftfreq(
            len(x_data), (x_data[1] - x_data[0])
        )  # assume uniform spacing
        F_y_data = abs(np.fft.fft(y_data))
        guess_freq = abs(
            ff[np.argmax(F_y_data[1:]) + 1]
        )  # excluding the zero frequency "peak", which is related to offset
        guess_amp = 3 * np.std(y_data) * 2.0**0.5
        guess_offset = np.mean(y_data)
        return (guess_amp, 2 * np.pi * guess_freq, 0.0, guess_offset)

    @staticmethod
    def _get_new_image_dir(comment: str | None) -> Path:
        """Creates and returns a new directory to save semiresults in, according to current date"""
        if comment:
            dir_to_save = Path(".") / (
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{comment}"
            )
        else:
            dir_to_save = Path(".") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_to_save.mkdir(parents=True, exist_ok=True)
        return dir_to_save

    @staticmethod
    def _save_picture(video: Path, index_to_take: int = 0) -> None:
        """Saves a screenshot from a video."""
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

    @staticmethod
    def _filter_items(item: float) -> bool:
        """Filters out some extreme values."""
        return 5 < item < 1000

    @staticmethod
    def _get_v(i_min: float, i_max: float) -> float:
        """Calculates the V value according to the minimum and maximum."""
        return (i_max - i_min) / (i_max + i_min)


if __name__ == "__main__":
    interfero_profile = InterProfile(
        MODE=MODE,
        PRINT=PRINT,
        SAVE_RESULT=SAVE_RESULT,
        TAKE_JUST_TWO=TAKE_JUST_TWO,
        AVG_OF_X=AVG_OF_X,
    )
    # interfero_profile.save_all_pictures(Path("videos"), 5)
    interfero_profile.get_visibility_graph(Path("photos"))
