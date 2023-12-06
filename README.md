# Portrait-Talker

Talking head animation

## Note: This project is not complete it is still WIP.

## Installation

### step 1. Clone repo

```
git clone https://github.com/harisreedhar/Portrait-Talker.git
cd Portrait-Talker
```

### step 2. Download pre-trained models

`put these models in Portrait-Talker/pretrained_models`

- [det_10g.onnx](https://huggingface.co/bluefoxcreation/insightface-retinaface-arcface-model/resolve/main/det_10g.onnx)
- [vox512.pt](https://huggingface.co/bluefoxcreation/LIA-512/resolve/main/vox512.pt?download=true)
- [shape_predictor_68_face_landmarks.dat](https://github.com/tzutalin/dlib-android/blob/master/data/shape_predictor_68_face_landmarks.dat)
- [wav2vecDS.pt](https://huggingface.co/bluefoxcreation/DINet_unofficial/resolve/main/wav2vecDS.pt)
- [clip_training_DINet_256mouth.pth](https://huggingface.co/bluefoxcreation/DINet_unofficial/resolve/main/clip_training_DINet_256mouth.pth)

### step 3. Create env & Install dependencies

```
conda create -n portrait-talker python=3.10 -y
conda activate portrait-talker
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### step 4. Test Run

```
python run_video_driven.py  --source_path ./test/source.jpg --driving_path ./test/driving.mov
```

### or

```
python run_audio_driven.py  --source_path ./test/source.jpg --driving_path ./test/driving.wav
```

## Acknowledgment

- [LIA](https://github.com/wyhsirius/LIA/)
- [DINet](https://github.com/MRzzm/DINet)
- [Insightface](https://github.com/deepinsight/insightface/tree/master/python-package/insightface)
- [Dlib](https://github.com/davisking/dlib)
