import cv2
import numpy as np


def align_crop(img, landmark, size):
    template_ffhq = np.array(
	[
		[192.98138, 239.94708],
		[318.90277, 240.19366],
		[256.63416, 314.01935],
		[201.26117, 371.41043],
		[313.08905, 371.15118]
	])
    template_ffhq *= (512 / size)
    matrix = cv2.estimateAffinePartial2D(landmark, template_ffhq, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
    warped = cv2.warpAffine(img, matrix, (size, size), borderMode=cv2.BORDER_REPLICATE)
    return warped, matrix


def get_cropped_head(img, landmark, scale=1.4, size=512):
    center = np.mean(landmark, axis=0)
    landmark = center + (landmark - center) * scale
    return align_crop(img, landmark, size)
