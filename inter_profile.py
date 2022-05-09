"""
Main class InterProfile for getting a visibility graph.
"""

import math
import random
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import skimage
from matplotlib import pyplot as plt
from scipy import optimize

from helpers import Point
from perpendicular_angle import get_perpendicular_angle


class Mode(Enum):
    """The different modes of the program."""

    NORMAL = 1
    NORMAL_AVERAGE = 2
    FIT_SINUS = 3


class InterProfile:
    def __init__(
        self,
        MODE: Mode,
        PRINT: bool,
        SAVE_RESULT: bool,
        CUT_LENGTH: int,
        CUT_AMOUNT: int,
        AVG_OF_X: int | None,
    ) -> None:
        self.MODE = MODE
        self.PRINT = PRINT
        self.SAVE_RESULT = SAVE_RESULT
        self.CUT_LENGTH = CUT_LENGTH
        self.CUT_AMOUNT = CUT_AMOUNT
        self.AVG_OF_X = AVG_OF_X

        max_x = 600 - self.CUT_LENGTH
        max_y = 700 - self.CUT_LENGTH
        self.ALL_START_POINTS: list[Point] = []
        for _ in range(self.CUT_AMOUNT):
            start_x = random.randint(0, max_x)
            start_y = random.randint(0, max_y)
            self.ALL_START_POINTS.append((start_x, start_y))

        self.NEW_IMAGE_DIRECTORY = self._get_new_image_dir(comment=self.MODE.name)

        self.CURRENT_PHOTO_INDEX = 0
        self.CURRENT_IMAGE_NAME = ""

        # Indexes of pictures that are correctly horizontal (all others are rotated)
        self.TOP_TO_BOTTOM_PHOTOS = [1, 2, 3, 4, 5, 6, 7, 9]
        # How many pixels to neglect both from top and bottom for horizontal pictures
        self.VERTICAL_OFFSET = 100

        # Coordinates of start of the cut of the rotated picture to get profile
        # End will be determined automatically according to the angle
        self.ANGLE_START = (25, 110)

    def get_visibility_graph(self, photo_folder: Path) -> None:
        V_VALUES: list[float] = []
        all_photos = photo_folder.rglob("*.jpg")
        for index, photo in enumerate(all_photos, start=1):
            print(f"analyzing {photo}")
            self.CURRENT_PHOTO_INDEX = index
            self.CURRENT_IMAGE_NAME = str(photo)
            v_value = self._get_visibility_from_photo(photo)
            V_VALUES.append(v_value)

        # Save results to a file
        with open(self.NEW_IMAGE_DIRECTORY / "RESULT.txt", "w") as f:
            for v_value in V_VALUES:
                f.write(f"{v_value:.3f}\n")

        # Plot the final result together with some useful information
        plt.plot(V_VALUES)
        title_str = f"Visibility\nMode: {self.MODE.name}\nCut length:{self.CUT_LENGTH}\nCut amount:{self.CUT_AMOUNT}"
        if self.MODE == Mode.NORMAL_AVERAGE:
            title_str += f" (avg {self.AVG_OF_X})"
        plt.title(title_str)
        plt.ylabel("Visibility")
        plt.xlabel("Image index")

        plt.tight_layout()
        plt.savefig(self.NEW_IMAGE_DIRECTORY / "RESULT.jpg")
        plt.show()

    def save_all_pictures(self, video_folder: Path, index_to_take: int = 0) -> None:
        """Saves a screenshot from each video in a folder."""
        all_files = video_folder.rglob("*.mp4")
        for file in all_files:
            self._save_picture(file, index_to_take)

    def _get_visibility_from_photo(self, photo: Path) -> float:
        image = skimage.io.imread(photo)
        lows = []
        bigs = []

        all_cuts = self._get_all_cuts(image)
        for start, end in all_cuts:
            low, big = self._get_lowest_and_biggest_intensity(image, start, end)
            lows.append(low)
            bigs.append(big)

        big = sum(bigs) // len(bigs)
        low = sum(lows) // len(lows)

        return self._get_visibility(low, big)

    def _get_all_cuts(self, image) -> list[tuple[Point, Point]]:
        if self.CURRENT_PHOTO_INDEX in self.TOP_TO_BOTTOM_PHOTOS:
            # Exactly in half vertically
            middle_vertical = image.shape[1] // 2
            start = (self.VERTICAL_OFFSET, middle_vertical)
            end = (image.shape[0] - self.VERTICAL_OFFSET, middle_vertical)
            return [(start, end)]

        perpendicular_angle = get_perpendicular_angle(self.CURRENT_IMAGE_NAME)
        all_cuts: list[tuple[Point, Point]] = []
        for start_point in self.ALL_START_POINTS:
            end_x = start_point[0] + self.CUT_LENGTH * math.cos(
                math.radians(perpendicular_angle)
            )
            end_y = start_point[1] + self.CUT_LENGTH * math.sin(
                math.radians(perpendicular_angle)
            )
            all_cuts.append((start_point, (int(end_x), int(end_y))))

        return all_cuts

    def _get_start_and_end_coords(self, image) -> tuple[Point, Point]:
        """Returns the start and end coordinates of the cut."""
        # Deciding where to take the "cut" on the image
        # (depends whether the photo is rightly horizontal or rotated)
        if self.CURRENT_PHOTO_INDEX in self.TOP_TO_BOTTOM_PHOTOS:
            # Exactly in half vertically
            middle_vertical = image.shape[1] // 2
            start = (self.VERTICAL_OFFSET, middle_vertical)
            end = (image.shape[0] - self.VERTICAL_OFFSET, middle_vertical)
        else:
            perpendicular_angle = get_perpendicular_angle(self.CURRENT_IMAGE_NAME)
            radian_perpendicular_angle = math.radians(perpendicular_angle)
            start = self.ANGLE_START
            end = (
                self.ANGLE_START[0]
                + self.CUT_LENGTH * math.cos(radian_perpendicular_angle),
                self.ANGLE_START[1]
                + self.CUT_LENGTH * math.sin(radian_perpendicular_angle),
            )

        return start, end

    def _get_lowest_and_biggest_intensity(
        self, image, start: Point, end: Point
    ) -> tuple[float, float]:
        """Calculates the lowest and biggest intensity of the image at the given position."""
        # Extract the real colour profile and take the integer values from it
        profile = skimage.measure.profile_line(image, start, end)
        values = [int(item[0]) for item in profile]

        if self.MODE == Mode.FIT_SINUS:

            def test_func_sinus(x, a, b, c, d) -> float:
                """Test function for fitting - a sinus one."""
                return a * np.sin(b * x + c) + d

            # Need to convert python lists to numpy arrays
            x_data = np.array(list(range(len(values))))
            y_data = np.array(values)

            # Get the initial guess for the parameters
            p0 = self._calculate_p0(x_data, y_data)

            # Get the sinus fit parameters
            params, _ = optimize.curve_fit(
                test_func_sinus, x_data, y_data, p0=p0, maxfev=10000
            )

            # Constructing lowest and highest values from the fitted parameters
            lowest = params[-1] - abs(params[0])
            biggest = params[-1] + abs(params[0])
        else:
            # So that we can show it in the picture
            raw_values = values.copy()

            # Disregarding some extreme values (optional) and sorting
            values = [value for value in values if self._filter_items(value)]
            values.sort()

            # If we want, we can take the average of the first X values
            if self.AVG_OF_X is not None:
                biggest = sum(values[-self.AVG_OF_X :]) // self.AVG_OF_X
                lowest = sum(values[: self.AVG_OF_X]) // self.AVG_OF_X
            else:
                biggest = values[-1]
                lowest = values[0]

        return lowest, biggest

    def _save_result(self, plt, fig) -> None:
        """Saves the picture."""
        plt.tight_layout()
        fig.set_size_inches(14, 8)
        plt.savefig(
            self.NEW_IMAGE_DIRECTORY / f"{self.CURRENT_PHOTO_INDEX}.jpg", dpi=96
        )
        plt.close(fig)  # Closing the figure so that it is not shown

    @staticmethod
    def _calculate_p0(x_data, y_data) -> tuple[float, float, float, float]:
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
    def _get_visibility(i_min: float, i_max: float) -> float:
        """Calculates the V value according to the minimum and maximum."""
        return (i_max - i_min) / (i_max + i_min)
