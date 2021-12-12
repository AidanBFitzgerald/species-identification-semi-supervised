"""
Discards videos missing a usable tag.
Discards videos with fewer than 45 frames.
Trims the length of the videos to 45 frames.
Interpolates each of the cropped frames to 24 x 24.
Outputs 3 channels:
    (1) The raw thermal values (min-max normalization)
    (2) The raw thermal values (each frame normalized independently)
    (3) The thermal values minus the background (min-max normalization)
Splits the data into training, validation, and test sets.
Encodes the labels as integers.
Saves the pre-processed data and the labels as numpy arrays.
"""

import h5py
import argparse
import numpy as np
import torch.nn as nn
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import cv2 as cv

VALIDATION_NUM = 1500
TEST_NUM = 1500


def dense_optical_flow(frame, prev_gray, mask):
    """
    Returns optical flow for given thermal video frame
    """
    frame = frame.astype("float32").transpose(1,2,0)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    return rgb, gray, mask


def get_best_index(vid):
    """
    Returns an index such that the selected 45 frames from a given video correspond to
    the 45 frames where the animal is nearest to the camera.
    """
    mass = np.zeros(vid.attrs["frames"])
    for f in range(vid.attrs["frames"]):
        mass[f] = np.sum(vid[str(f)][4])
    total_mass_over_next_45 = np.cumsum(mass) - np.hstack(
        [np.zeros(45), np.cumsum(mass[:-45])]
    )
    return f - np.argmax(total_mass_over_next_45[::-1]) - 44


def make24x24(frame):
    """
    Interpolates a given frame so its largest dimension is 24. The padding uses the minimum
    of the frame's values across each channel.
    """
    scale = (24.5 / np.array(frame.shape[1:])).min()
    frame = torch.tensor(np.expand_dims(frame, 0))
    frame = np.array(
        nn.functional.interpolate(frame, scale_factor=scale, mode="area")[0]
    )
    square = np.tile(np.min(frame, (1, 2)).reshape(3, 1, 1), (1, 24, 24))
    offset = ((np.array([24, 24]) - frame.shape[1:]) / 2).astype(np.int)
    square[
    :,
    offset[0]: offset[0] + frame.shape[1],
    offset[1]: offset[1] + frame.shape[2],
    ] = frame
    return square


def normalize(frame):
    """
    Min-max normalizes the first channel (clipping outliers).
    Min-max normalizes the second channel for each frame independently.
    Min-max normalizes the third channel (clipping outliers).
    """
    frame[0] = np.clip((frame[0] - 2500) / 1000, 0, 1)
    frame[1] = np.nan_to_num(
        (frame[1] - frame[1].min()) / (frame[1].max() - frame[1].min())
    )
    frame[2] = np.clip(frame[2] / 400, 0, 1)
    return frame

def normalize_optical(optical_flow, min, max):
    """
    Min-max normalizes optical flow given min and max values
    """
    return ((optical_flow - min)/(max - min))


def temporal_gradient(frame_current, frame_next):
    """
    Calculates difference between two consecutive frames
    Normalizes difference between 0 - 1
    """
    movement = np.subtract(frame_current, frame_next)
    # Normalize movement for each channel
    movement = (movement + 1) / 2
    return movement


def main(input_file, output_dir):
    f = h5py.File(input_file, "r")  # Read in the dataset
    d = f[list(f.keys())[0]]  # Access the thermal videos key
    clips = np.zeros(
        [10664, 45, 3, 24, 24], dtype=np.float16
    )  # np.float16 saves storage space

    clips_movement = np.zeros(
        [10664, 45, 3, 24, 24], dtype=np.float16
    )  # np.float16 saves storage space

    clips_optical = np.zeros(
        [10664, 45, 3, 24, 24], dtype=np.float16
    )  # np.float16 saves storage space

    labels_raw = []
    processed = 0
    for i in range(len(d.keys())):
        x = d[list(d.keys())[i]]
        for j in range(len(x.keys()) - 1):
            vid = x[list(x.keys())[j]]
            tag = vid.attrs["tag"]
            if tag == "bird/kiwi":
                tag = "bird"
            if vid.attrs["frames"] >= 45 and not tag in [
                "unknown",
                "part",
                "poor tracking",
                "sealion",
            ]:
                labels_raw += [tag]
                ind = get_best_index(vid)
                for f in range(45):
                    frame = np.array(vid[str(f + ind)], dtype=np.float16)[
                            :2
                            ]  # Read a single frame
                    frame = np.concatenate(
                        [np.expand_dims(frame[0], 0), frame], 0
                    )  # The desired 3 channels
                    frame = make24x24(frame)  # Interpolate the frame
                    frame = normalize(frame)  # Normalizes each channel
                    clips[processed, f] = frame

                # get gray of first frame for optical flow
                frame = clips[processed, 0].astype("float32").transpose(1,2,0)
                mask = np.zeros_like(frame)
                mask[..., 1] = 255
                prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                optical_flow, prev_gray, mask = dense_optical_flow(clips[processed,0], prev_gray, mask)
                clips_optical[processed,0] = optical_flow.transpose(2,0,1)
                # Get temporal gradient and optical flow from each frame
                for f in range(1,45):
                    current_frame = clips[processed, f]
                    prev_frame = clips[processed, f - 1]
                    optical_flow, prev_gray, mask = dense_optical_flow(current_frame, prev_gray, mask)
                    # set optical_flow to same format as other frames
                    optical_flow = optical_flow.transpose(2,0,1)
                    clips_optical[processed, f] = optical_flow
                    movement = temporal_gradient(current_frame, prev_frame)
                    clips_movement[processed, f] = movement

                # Normalize optical flow across clip
                max = np.nanmax(clips_optical[processed])
                min = np.nanmin(clips_optical[processed])

                if np.any(np.isnan(clips_optical[processed])):
                    clips_optical[processed] = np.nan_to_num(clips_optical[processed])
                for f in range(45):
                    clips_optical[processed,f] = normalize_optical(clips_optical[processed,f], min, max)

                # Get temporal gradient from first frame
                current_frame = clips[processed, 0]
                prev_frame = clips[processed, 0]
                movement = temporal_gradient(current_frame, prev_frame)
                clips_movement[processed, 0] = movement

                processed += 1
                if processed % 100 == 0:
                    print(processed, "clips processed!")

    # We encode the labels as an integer for each class
    labels = LabelEncoder().fit_transform(labels_raw)

    labels_raw = np.array(labels_raw)
    labels_raw_optical = labels_raw
    labels_raw_movement = labels_raw
    labels_optical = labels
    labels_movement = labels

    # We extract the training, test and validation sets, with a fixed random seed for reproducibility and stratification
    clips, val_vids, labels, val_labels, labels_raw, val_labels_raw = train_test_split(
        clips,
        labels,
        labels_raw,
        test_size=VALIDATION_NUM,
        random_state=123,
        stratify=labels,
    )
    (
        train_vids,
        test_vids,
        train_labels,
        test_labels,
        train_labels_raw,
        test_labels_raw,
    ) = train_test_split(
        clips, labels, labels_raw, test_size=TEST_NUM, random_state=123, stratify=labels
    )

    clips_optical, val_vids_optical, labels_optical, val_labels_optical, labels_raw_optical, val_labels_raw_optical = train_test_split(
        clips_optical,
        labels_optical,
        labels_raw_optical,
        test_size=VALIDATION_NUM,
        random_state=123,
        stratify=labels_optical,
    )

    (
    train_vids_optical,
    test_vids_optical,
    train_labels_optical,
    test_labels_optical,
    train_labels_raw_optical,
    test_labels_raw_optical,
    ) = train_test_split(
        clips_optical, labels_optical, labels_raw_optical, test_size=TEST_NUM, random_state=123, stratify=labels_optical
    )

    clips_movement, val_vids_movement, labels_movement, val_labels_movement, labels_raw_movement, val_labels_raw_movement = train_test_split(
        clips_movement,
        labels_movement,
        labels_raw_movement,
        test_size=VALIDATION_NUM,
        random_state=123,
        stratify=labels_movement,
    )

    (
        train_vids_movement,
        test_vids_movement,
        train_labels_movement,
        test_labels_movement,
        train_labels_raw_movement,
        test_labels_raw_movement,
    ) = train_test_split(
        clips_movement, labels_movement, labels_raw_movement, test_size=TEST_NUM, random_state=123,
        stratify=labels_movement
    )

    # We save all of the files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(f"{output_dir}/training", train_vids)
    np.save(f"{output_dir}/validation", val_vids)
    np.save(f"{output_dir}/test", test_vids)
    np.save(f"{output_dir}/training-labels", train_labels)
    np.save(f"{output_dir}/validation-labels", val_labels)
    np.save(f"{output_dir}/test-labels", test_labels)
    np.save(f"{output_dir}/training-labels_raw", train_labels_raw)
    np.save(f"{output_dir}/validation-labels_raw", val_labels_raw)
    np.save(f"{output_dir}/test-labels_raw", test_labels_raw)
    np.save(f"{output_dir}/training-optical-flow", train_vids_optical)
    np.save(f"{output_dir}/validation-optical-flow", val_vids_optical)
    np.save(f"{output_dir}/test-optical-flow", test_vids_optical)
    np.save(f"{output_dir}/training-temporal-gradients", train_vids_movement)
    np.save(f"{output_dir}/validation-temporal-gradients", val_vids_movement)
    np.save(f"{output_dir}/test-temporal-gradients", test_vids_movement)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    main(args.input_file, args.output_dir)
