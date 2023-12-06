import os
import cv2
import torch
import torch.nn as nn
import torchvision
import argparse
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from LIA.generator import Generator
from utils.retinaface import RetinaFace
from utils.face_alignment import get_cropped_head


LIA_PATH = "./pretrained_models/vox512.pt"
RETINAFACE_PATH = "./pretrained_models/det_10g.onnx"
MASK = cv2.imread("./mask.jpg")


def paste_back(img, face, matrix):
    inverse_affine = cv2.invertAffineTransform(matrix)
    h, w = img.shape[0:2]
    face_h, face_w = face.shape[0:2]
    inv_restored = cv2.warpAffine(face, inverse_affine, (w, h))
    inv_restored = inv_restored.astype('float32')
    mask = MASK.copy().astype('float32') / 255
    mask = cv2.resize(mask, (face_w, face_h))
    inv_mask = cv2.warpAffine(mask, inverse_affine, (w, h))
    img = inv_mask * inv_restored + (1 - inv_mask) * img
    return img.clip(0, 255).astype('uint8')


def process_source(model, img_path, size, crop_scale=1.6):
    ori_img = cv2.imread(img_path)
    bboxes, kpss = model.detect(ori_img, det_thresh=0.6)
    aimg, mat = get_cropped_head(ori_img, kpss[0], size=size, scale=crop_scale)
    aimg = np.transpose(aimg[:,:,::-1], (2, 0, 1)) / 255
    aimg_tensor = torch.from_numpy(aimg).unsqueeze(0).float()
    aimg_tensor_norm = (aimg_tensor - 0.5) * 2.0
    return aimg_tensor_norm, ori_img, mat


def process_driving(vid_path, size=256):
    cap = cv2.VideoCapture(vid_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (size, size))
            video.append(frame)
        else:
            break
    cap.release()
    vid = torch.from_numpy(np.asarray(video)).permute(0, 3, 1, 2).unsqueeze(0)
    vid_norm = (vid / 255.0 - 0.5) * 2.0
    return vid_norm, video_fps


class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()
        self.size = 512
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print('[ loading model ]')

        detector = RetinaFace(model_file=RETINAFACE_PATH, provider=["CUDAExecutionProvider", "CPUExecutionProvider"], session_options=None)
        self.gen = Generator(self.size, 512, 20, 1).to(self.device)
        weight = torch.load(LIA_PATH, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        print('[ loading data ]')

        self.source, self.original_image, self.matrix = process_source(detector, args.source_path, self.size, crop_scale=args.crop_scale)
        self.driving, self.fps = process_driving(args.driving_path, size=self.size)

        self.source = self.source.to(self.device)
        self.driving = self.driving.to(self.device)

    def run(self):
        print('[ running ]')

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        h, w = self.original_image.shape[:2]
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, 'video_driven_result.avi')
        out = cv2.VideoWriter(save_path, fourcc, int(self.fps), (w,h))

        with torch.no_grad():
            h_start = self.driving[:, 0, :, :, :].to(self.device)
            h_start = self.gen.enc.enc_motion(h_start)
            for i in tqdm(range(self.driving.size(1))):
                img_target = self.driving[:, i, :, :, :].to(self.device)
                img_recon = self.gen(self.source, img_target, h_start)
                np_img_recon = img_recon.permute(0,2,3,1).cpu().numpy()[0]
                np_img_recon = ((np_img_recon[:,:,::-1] + 1) / 2) * 255
                pasted = paste_back(self.original_image, np_img_recon, self.matrix)
                out.write(pasted.astype('uint8'))

        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default='')
    parser.add_argument("--driving_path", type=str, default='')
    parser.add_argument("--save_dir", type=str, default='./test/result')
    parser.add_argument("--crop_scale", type=float, default=1.5)
    args = parser.parse_args()

    demo = Demo(args)
    demo.run()
