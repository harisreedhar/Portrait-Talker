import os
import cv2
import torch
import subprocess
import argparse
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from collections import OrderedDict

from utils.retinaface import RetinaFace
from utils.face_alignment import get_cropped_head

from DINet.DINet import DINet
from DINet.wav2vec import Wav2VecFeatureExtractor
from DINet.wav2vecDS import Wav2vecDS
from DINet.utils import compute_crop_radius, face_crop, final_image_fill

import dlib
import random


DLIB_68 = "./pretrained_models/shape_predictor_68_face_landmarks.dat"
DINET_PATH = "./pretrained_models/clip_training_DINet_256mouth.pth"
WAV2VECDS_PATH = "./pretrained_models/wav2vecDS.pt"
RETINAFACE_PATH = "./pretrained_models/det_10g.onnx"
MASK = cv2.imread("./mask.jpg")


def get_landmark(face_detector, shape_predictor, image):
    # bboxes, kpss = face_detector.detect(image, det_thresh=0.6)
    # bbox = np.array(bboxes).astype('int32')[0]
    # face = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_detector(image_gray)[0]
    landmark = shape_predictor(image_gray, face)
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    landmark_bxy = np.array([[np.array([lmk.x, lmk.y]) for lmk in landmark.parts()]]).astype('float32')
    return landmark_bxy, (x1, y1, x2, y2)


class Demo():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        print('[ loading models ]')

        # face detector model
        # self.face_detector = RetinaFace(model_file=RETINAFACE_PATH, provider=["CUDAExecutionProvider", "CPUExecutionProvider"], session_options=None)
        self.face_detector = dlib.get_frontal_face_detector()

        # 68 point face alignment
        self.shape_predictor = dlib.shape_predictor(DLIB_68)

        # audio to vec model
        self.feature_extractor = Wav2VecFeatureExtractor(self.device)
        self.audio_mapping = Wav2vecDS(model_path=WAV2VECDS_PATH)

        # dinet model
        self.dinet = DINet(3, 15, 29).to(self.device)
        state_dict = torch.load(DINET_PATH)["state_dict"]["net_g"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        self.dinet.load_state_dict(new_state_dict)
        self.dinet.eval()

    def run(self):

        print('[ Extracting audio features ]')

        ds_feature = self.feature_extractor.compute_audio_feature(self.args.driving_path, chunk_duration=10)
        ds_feature = self.audio_mapping.mapping(ds_feature)

        print('[ Getting source face landmarks ]')

        source_image = cv2.imread(self.args.source_path)
        height, width = source_image.shape[0:2]

        landmarks = get_landmark(self.face_detector, self.shape_predictor, source_image)[0]

        mouth_region_size = self.args.mouth_region_size
        m_2 = mouth_region_size // 2
        m_4 = mouth_region_size // 4
        m_8 = mouth_region_size // 8

        resize_w = int(mouth_region_size + m_4)
        resize_h = int(m_2 * 3 + m_8)
        crop_flag, crop_radius = compute_crop_radius((width, height), landmarks)
        cropped_face = face_crop(source_image, landmarks[0], crop_radius)
        cropped_face_shape = (cropped_face.shape[1], cropped_face.shape[0])
        cropped_face_resize = cv2.resize(cropped_face, (resize_w, resize_h)).astype('float32')[:, :, ::-1]

        cropped_face_blank = cropped_face_resize.copy()
        cropped_face_blank[m_2:m_2 + mouth_region_size, m_8:m_8 + mouth_region_size, :] = 0


        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        os.makedirs(self.args.save_dir, exist_ok=True)
        save_path = os.path.join(self.args.save_dir, 'audio_driven_result_without_audio.avi')
        fps = 25
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        reference_list = [cropped_face_resize / 255] * 5
        reference_array = np.concatenate(reference_list, 2)

        with torch.no_grad():
            total_frames = ds_feature.shape[0]

            reference_tensor = (torch.from_numpy(reference_array).permute(2, 0, 1).unsqueeze(0).float().to(self.device))
            source_face_tensor = (torch.from_numpy(cropped_face_blank / 255).float().to(self.device).permute(2, 0, 1).unsqueeze(0))

            for index in tqdm(range(0, total_frames), total=total_frames, desc="Processing"):

                deepspeech_tensor = (torch.from_numpy(ds_feature[index:index+5, :]).permute(1, 0).unsqueeze(0).float().to(self.device))

                pred = self.dinet(source_face_tensor, reference_tensor, deepspeech_tensor)
                pred_np = (pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255)[:, :, ::-1]
                pred_np = cv2.resize(pred_np, cropped_face_shape)

                final_frame = final_image_fill(source_image.copy(), pred_np, landmarks[0], crop_radius)
                out.write(final_frame.astype('uint8'))

        out.release()
        cv2.destroyAllWindows()

        print('[ Merging audio ]')

        final_path = os.path.join(self.args.save_dir, 'audio_driven_result.mp4')
        cmd = f"/home/harisree/Documents/GitHub/Swap-Mukham/assets/ffmpeg/ffmpeg -i {save_path} -i {self.args.driving_path} -strict experimental -map 0:v:0 -map 1:a:0 -y {final_path}"
        subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default='')
    parser.add_argument("--driving_path", type=str, default='')
    parser.add_argument("--save_dir", type=str, default='./test/result')
    parser.add_argument("--mouth_region_size", type=int, default=256)
    args = parser.parse_args()

    demo = Demo(args)
    demo.run()
