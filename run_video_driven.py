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
    assert len(kpss) != 0, "No face detected"
    aimg, mat = get_cropped_head(ori_img, kpss[0], size=size, scale=crop_scale)
    aimg = np.transpose(aimg[:,:,::-1], (2, 0, 1)) / 255
    aimg_tensor = torch.from_numpy(aimg).unsqueeze(0).float()
    aimg_tensor_norm = (aimg_tensor - 0.5) * 2.0
    return aimg_tensor_norm, ori_img, mat


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
        self.source = self.source.to(self.device)

    def run(self):
        print('[ running ]')

        cap = cv2.VideoCapture(args.driving_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        h, w = self.original_image.shape[:2]
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, 'video_driven_result.avi')
        out = cv2.VideoWriter(save_path, fourcc, int(fps), (w,h))

        with torch.no_grad():
            h_start = None
            for index in tqdm(range(total_frames), desc='[ Processing frames ]', unit='frame'):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (self.size, self.size))
                    frame_np = np.expand_dims(frame, axis = 0).astype('float32')
                    frame_np = (frame_np / 255 - 0.5) * 2.0
                    frame_np = frame_np.transpose(0, 3, 1, 2)
                    frame_torch = torch.from_numpy(frame_np).to(self.device)
                    if index == 0:
                        h_start = self.gen.enc.enc_motion(frame_torch)
                    frame_recon = self.gen(self.source, frame_torch, h_start)
                    frame_recon_np = frame_recon.permute(0, 2, 3, 1).cpu().numpy()[0]
                    frame_recon_np = ((frame_recon_np[:,:,::-1] + 1) / 2) * 255
                    pasted = paste_back(self.original_image, frame_recon_np, self.matrix)
                    out.write(pasted.astype('uint8'))
                else:
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default='')
    parser.add_argument("--driving_path", type=str, default='')
    parser.add_argument("--save_dir", type=str, default='./test/result')
    parser.add_argument("--crop_scale", type=float, default=1.25)
    args = parser.parse_args()

    demo = Demo(args)
    demo.run()
