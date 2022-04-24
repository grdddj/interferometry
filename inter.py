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
    for script_arg in sys.argv[1:]:
        if "avg" in script_arg:
            AVG_OF_X = int(script_arg.split("=")[1])
            break
    else:
        AVG_OF_X = 5
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

        self.NEW_IMAGE_DIRECTORY = self._get_new_image_dir()

        self.PHOTO_INDEX = 0
        self.CURRENT_IMAGE = ""
        self.TOP_TO_BOTTOM_PHOTOS = [1, 2, 3, 4, 5, 6, 7, 9]

        self.ANGLE_START = (25, 110)
        self.ANGLE_END = (250, 270)

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

        # Exactly in half vertically
        positions = [image.shape[1] // 2]

        LOWESTS: list[float] = []
        BIGGEST: list[float] = []

        # Aggregating all the positions into its average
        for position in positions:
            low, big = self._get_lowest_and_biggest_intensity(image, position)
            LOWESTS.append(low)
            BIGGEST.append(big)

        avg_big = sum(BIGGEST) // len(BIGGEST)
        avg_low = sum(LOWESTS) // len(LOWESTS)
        visibility = self._get_v(avg_big, avg_low)

        return visibility

    def _get_lowest_and_biggest_intensity(
        self, image, position: int
    ) -> tuple[float, float]:
        """Calculates the lowest and biggest intensity of the image at the given position."""
        # Deciding where to take the "cut" on the image
        # (depends whether the photo is rightly horizontal or rotated)
        if self.PHOTO_INDEX in self.TOP_TO_BOTTOM_PHOTOS:
            offset = 100
            start = (offset, position)
            end = (image.shape[0] - offset, position)
        else:
            start = self.ANGLE_START
            end = self.ANGLE_END

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

            # Getting the parameters from the fit
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
                    f"visibility: {self._get_v(biggest, lowest):.2f}"
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
                    f"visibility: {self._get_v(biggest, lowest):.2f}"
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
        guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])
        return tuple(guess)

    @staticmethod
    def _get_new_image_dir() -> Path:
        """Creates and returns a new directory to save semiresults in, according to current date"""
        dir_to_save = Path(".") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_to_save.mkdir(parents=True, exist_ok=True)
        return dir_to_save

    @staticmethod
    def _save_picture(video: Path, index_to_take: int = 0) -> None:
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

    @staticmethod
    def _filter_items(item: float) -> bool:
        """Filters out some extreme values."""
        return 5 < item < 1000

    @staticmethod
    def _get_v(imax: float, imin: float) -> float:
        """Calculates the V value according to the minimum and maximum."""
        return (imax - imin) / (imax + imin)


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
