import cv2
import numpy as np


def umeyama(src, dst, estimate_scale):
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = np.dot(dst_demean.T, src_demean) / num
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.double)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))
    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0
    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale
    return T

templates_112 = np.array([
    [[51.6420, 50.1150], [57.6170, 49.9900], [35.7400, 69.0070], [51.1570, 89.0500], [57.0250, 89.7020]],
    [[45.0310, 50.1180], [65.5680, 50.8720], [39.6770, 68.1110], [45.1770, 86.1900], [64.2460, 86.7580]],
    [[39.7300, 51.1380], [72.2700, 51.1380], [56.0000, 68.4930], [42.4630, 87.0100], [69.5370, 87.0100]],
    [[46.8450, 50.8720], [67.3820, 50.1180], [72.7370, 68.1110], [48.1670, 86.7580], [67.2360, 86.1900]],
    [[54.7960, 49.9900], [60.7710, 50.1150], [76.6730, 69.0070], [55.3880, 89.7020], [61.2570, 89.0500]],
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
], dtype=np.float32)

def align_crop_112_from_best(img, lmk, multiplier=1):
    templates = templates_112 * multiplier
    test_lmk = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_error, best_matrix = float('inf'), []
    for i in np.arange(templates.shape[0]):
        matrix = umeyama(lmk, templates[i], True)[0:2, :]
        error = np.sum(np.sqrt(np.sum((np.dot(matrix, test_lmk.T).T - templates[i]) ** 2, axis=1)))
        if error < min_error:
            min_error, best_matrix = error, matrix

    cropped_img = cv2.warpAffine(img, best_matrix, (112, 112), borderValue=0.0)
    return cropped_img, best_matrix


arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)


def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    M = umeyama(lmk, dst, True)[0:2, :]
    return M


def norm_crop_arcface(img, landmark, image_size=112, mode='arcface', from_best=True):
    if from_best:
        return align_crop_112_from_best(img, landmark)
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    return warped, M


def get_cropped_head(img, landmark, scale=1.4, size=512):
    # it is ugly but works :D
    center = np.mean(landmark, axis=0)
    landmark = center + (landmark - center) * scale
    M = estimate_norm(landmark, 128, mode='arcface')
    M /= (128/size)
    warped = cv2.warpAffine(img, M, (size,size), borderValue=0.0)
    return warped, M