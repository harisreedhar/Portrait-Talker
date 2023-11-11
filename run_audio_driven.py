import os
import cv2
import glob
import dlib
import torch
import random
import warnings
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from utils.utils import is_image, is_video, create_directory

from DINet.DINet import DINet
from DINet.wav2vec import Wav2VecFeatureExtractor
from DINet.wav2vecDS import Wav2vecDS
from DINet.utils import compute_crop_radius, face_crop, final_image_fill


warnings.filterwarnings("ignore", category=UserWarning, module="torch")


DLIB_68 = "./pretrained_models/shape_predictor_68_face_landmarks.dat"
DINET_PATH = "./pretrained_models/clip_training_DINet_256mouth.pth"
WAV2VECDS_PATH = "./pretrained_models/wav2vecDS.pt"


def extract_frames_from_video(video_path, save_dir):
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if int(fps) != 25:
        print("Warning: the input video is not 25 fps, it would be better to trans it to 25 fps!")
    frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    os.makedirs(save_dir, exist_ok=True)
    ffmpeg_command = ["ffmpeg", "-i", video_path, os.path.join(save_dir, "%06d.png")]
    subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return frame_width, frame_height


def get_landmarks(video_frame_paths):
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(DLIB_68)
    progress_bar = tqdm(total=len(video_frame_paths), desc='[ Getting face landmarks ]', unit='frame', dynamic_ncols=True)
    landmarks = []
    for frame_index, image_path in enumerate(video_frame_paths):
        frame = cv2.imread(image_path)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray_frame)
        if len(faces) == 0:
            print(f"Error: No Face detected on frame {frame_index}!")
            break
        if len(faces) > 1:
            print(f"Warning: More than one face detected on frame {frame_index}. Using first detected one!")
        face = faces[0]
        landmark68 = shape_predictor(gray_frame, face)
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        landmarks.append([np.array([lmk.x, lmk.y]) for lmk in landmark68.parts()])
        progress_bar.update(1)
    progress_bar.close()
    landmarks = np.array(landmarks)
    return landmarks


class Demo():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        print('[ Loading models ]')

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

        print('[ Extracting driving audio features ]')

        ds_feature = self.feature_extractor.compute_audio_feature(self.args.driving_path, chunk_duration=10)
        ds_feature = self.audio_mapping.mapping(ds_feature)
        res_frame_length = ds_feature.shape[0]
        ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode="edge")

        if is_video(self.args.source_path):
            temp_dir = os.path.join(self.args.save_dir, "temp_sequence")
            create_directory(temp_dir, remove_existing=True)
            video_size = extract_frames_from_video(self.args.source_path, temp_dir)
            video_frame_path_list = glob.glob(os.path.join(temp_dir, "*.png"))
            video_frame_path_list.sort()
            video_landmark_data = get_landmarks(video_frame_path_list).astype(np.int32)

        elif is_image(self.args.source_path):
            image = cv2.imread(self.args.source_path)
            video_size = image.shape[1], image.shape[0]
            video_frame_path_list = [self.args.source_path] * res_frame_length
            video_landmark_data = np.array([get_landmarks([self.args.source_path])[0]] * res_frame_length).astype(np.int32)
        else:
            print(f"Unknown source {self.args.source_path}")

        video_frame_path_list_cycle = video_frame_path_list + video_frame_path_list[::-1]
        video_landmark_data_cycle = np.concatenate([video_landmark_data, np.flip(video_landmark_data, 0)], 0)
        video_frame_path_list_cycle_length = len(video_frame_path_list_cycle)

        if video_frame_path_list_cycle_length >= res_frame_length:
            res_video_frame_path_list = video_frame_path_list_cycle[:res_frame_length]
            res_video_landmark_data = video_landmark_data_cycle[:res_frame_length, :, :]
        else:
            divisor = res_frame_length // video_frame_path_list_cycle_length
            remainder = res_frame_length % video_frame_path_list_cycle_length
            res_video_frame_path_list = (video_frame_path_list_cycle * divisor + video_frame_path_list_cycle[:remainder])
            res_video_landmark_data = np.concatenate([video_landmark_data_cycle] * divisor + [video_landmark_data_cycle[:remainder, :, :]], 0,)

        res_video_frame_path_list_pad = ([video_frame_path_list_cycle[0]] * 2 + res_video_frame_path_list + [video_frame_path_list_cycle[-1]] * 2)
        res_video_landmark_data_pad = np.pad(res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode="edge")

        assert (ds_feature_padding.shape[0] == len(res_video_frame_path_list_pad) == res_video_landmark_data_pad.shape[0])
        pad_length = ds_feature_padding.shape[0]

        m_2 = self.args.mouth_region_size // 2
        m_4 = self.args.mouth_region_size // 4
        m_8 = self.args.mouth_region_size // 8

        ref_img_list = []
        resize_shape = (int(self.args.mouth_region_size + m_4), int((m_2) * 3 + m_8))
        ref_index_list = random.sample(range(5, len(res_video_frame_path_list_pad) - 2), 5)

        for ref_index in ref_index_list:
            crop_flag, crop_radius = compute_crop_radius(video_size, res_video_landmark_data_pad[ref_index - 5 : ref_index, :, :])
            if not crop_flag:
                raise ValueError("DINET cannot handle videos with large changes in facial size!!")

            ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index - 3])[:, :, ::-1]
            ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]
            ref_img_crop = face_crop(ref_img, ref_landmark, crop_radius)
            ref_img_crop = cv2.resize(ref_img_crop, resize_shape)
            ref_img_crop = ref_img_crop / 255.0
            ref_img_list.append(ref_img_crop)

        ref_video_frame = np.concatenate(ref_img_list, 2)
        ref_img_tensor = (torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        os.makedirs(self.args.save_dir, exist_ok=True)
        save_path = os.path.join(self.args.save_dir, 'audio_driven_result_without_audio.avi')
        video_writer = cv2.VideoWriter(save_path, fourcc, 25, video_size)
        progress_bar = tqdm(total=res_frame_length, desc='[ Processing frames ]', unit='frame', dynamic_ncols=True)

        for clip_end_index in range(5, pad_length, 1):
            crop_flag, crop_radius = compute_crop_radius(video_size, res_video_landmark_data_pad[clip_end_index - 5 : clip_end_index, :, :], random_scale=1.05)
            if not crop_flag:
                raise ("DINET can not handle videos with large change of facial size!!")

            frame_data = cv2.imread(res_video_frame_path_list_pad[clip_end_index - 3])[:, :, ::-1]
            frame_landmark = res_video_landmark_data_pad[clip_end_index - 3, :, :]

            crop_frame_data = face_crop(frame_data, frame_landmark, crop_radius)
            crop_frame_h, crop_frame_w = crop_frame_data.shape[0], crop_frame_data.shape[1]
            crop_frame_data = cv2.resize(crop_frame_data, resize_shape)
            crop_frame_data = crop_frame_data / 255.0
            crop_frame_data[m_2 : m_2 + self.args.mouth_region_size, m_8 : m_8 + self.args.mouth_region_size, :,] = 0

            crop_frame_tensor = (
                torch.from_numpy(crop_frame_data)
                .float()
                .to(self.device)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            deepspeech_tensor = (
                torch.from_numpy(ds_feature_padding[clip_end_index - 5 : clip_end_index, :])
                .permute(1, 0)
                .unsqueeze(0)
                .float()
                .to(self.device)
            )

            with torch.no_grad():
                pre_frame = self.dinet(crop_frame_tensor, ref_img_tensor, deepspeech_tensor)
                pre_frame = (pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255)

            pre_frame_resize = cv2.resize(pre_frame, (crop_frame_w, crop_frame_h))
            final_frame = final_image_fill(frame_data.copy(), pre_frame_resize, frame_landmark, crop_radius)

            video_writer.write(final_frame[:, :, ::-1])
            progress_bar.update(1)

        video_writer.release()
        progress_bar.close()

        print('[ Merging audio ]')

        video_with_audio_path = save_path.replace("_without_audio", "")
        if os.path.exists(video_with_audio_path):
            os.remove(video_with_audio_path)

        ffmpeg_command = f"ffmpeg -i {save_path} -i {self.args.driving_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {video_with_audio_path}"
        subprocess.call(ffmpeg_command, shell=True)

        if os.path.exists(save_path) and os.path.exists(video_with_audio_path):
            os.remove(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default='')
    parser.add_argument("--driving_path", type=str, default='')
    parser.add_argument("--save_dir", type=str, default='./test/result')
    parser.add_argument("--mouth_region_size", type=int, default=256)
    args = parser.parse_args()

    demo = Demo(args)
    demo.run()
