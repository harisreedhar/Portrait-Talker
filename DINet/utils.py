import cv2
import random
import numpy as np

def compute_crop_radius(video_size, landmark_data_clip, random_scale=None):
    video_w, video_h = video_size[0], video_size[1]
    landmark_max_clip = np.max(landmark_data_clip, axis=1)
    if random_scale is None:
        random_scale = random.random() / 10 + 1.05
    else:
        random_scale = random_scale
    radius_h = (landmark_max_clip[:, 1] - landmark_data_clip[:, 29, 1]) * random_scale
    radius_w = (
        landmark_data_clip[:, 54, 0] - landmark_data_clip[:, 48, 0]
    ) * random_scale
    radius_clip = np.max(np.stack([radius_h, radius_w], 1), 1) // 2
    radius_max = np.max(radius_clip)
    radius_max = (np.int32(radius_max / 4) + 1) * 4
    radius_max_1_4 = radius_max // 4
    clip_min_h = landmark_data_clip[:, 29, 1] - radius_max
    clip_max_h = landmark_data_clip[:, 29, 1] + radius_max * 2 + radius_max_1_4
    clip_min_w = landmark_data_clip[:, 33, 0] - radius_max - radius_max_1_4
    clip_max_w = landmark_data_clip[:, 33, 0] + radius_max + radius_max_1_4
    if min(clip_min_h.tolist() + clip_min_w.tolist()) < 0:
        return False, None
    elif max(clip_max_h.tolist()) > video_h:
        return False, None
    elif max(clip_max_w.tolist()) > video_w:
        return False, None
    elif max(radius_clip) > min(radius_clip) * 1.5:
        return False, None
    else:
        return True, radius_max

def face_crop(image, landmark, crop_radius):
    landmark = landmark.astype('int32')
    crop_radius_1_4 = crop_radius // 4
    cropped = image[
        landmark[29, 1]
        - crop_radius : landmark[29, 1]
        + crop_radius * 2
        + crop_radius_1_4,
        landmark[33, 0]
        - crop_radius
        - crop_radius_1_4 : landmark[33, 0]
        + crop_radius
        + crop_radius_1_4,
        :,
    ]
    return cropped

def final_image_fill(full_image, cropped_face, landmark, crop_radius):
    landmark = landmark.astype('int32')
    crop_radius_1_4 = crop_radius // 4
    full_image[
        landmark[29, 1]
        - crop_radius : landmark[29, 1]
        + crop_radius * 2,
        landmark[33, 0]
        - crop_radius
        - crop_radius_1_4 : landmark[33, 0]
        + crop_radius
        + crop_radius_1_4,
        :,
    ] = cropped_face[: crop_radius * 3, :, :]
    return full_image