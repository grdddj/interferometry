import math
from pathlib import Path

import numpy as np
import skimage

from helpers import file_cache


@file_cache("perpendicular_angle.json")
def get_perpendicular_angle(photo_path: str) -> int:
    black_start = _get_black_start(photo_path)
    angle = _get_angle(photo_path, black_start)
    perpendicular_angle = 90 - angle
    return perpendicular_angle


def _get_black_start(photo: str | Path) -> tuple[int, int]:
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


def _get_angle(photo: str | Path, black_start: tuple[int, int]) -> int:
    """Get the angle of the black line so that we can cut the picture perpendicular to it."""
    image = skimage.io.imread(photo)

    start_x = black_start[0]
    start_y = black_start[1]

    degrees_thresholds = {}
    for degrees in range(91):
        radians = math.radians(degrees)

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
