# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['sadtalker_video_lip_sync',
 'sadtalker_video_lip_sync.src',
 'sadtalker_video_lip_sync.src.audio2exp_models',
 'sadtalker_video_lip_sync.src.audio2pose_models',
 'sadtalker_video_lip_sync.src.dain_model',
 'sadtalker_video_lip_sync.src.face3d',
 'sadtalker_video_lip_sync.src.face3d.data',
 'sadtalker_video_lip_sync.src.face3d.models',
 'sadtalker_video_lip_sync.src.face3d.models.arcface_torch',
 'sadtalker_video_lip_sync.src.face3d.models.arcface_torch.backbones',
 'sadtalker_video_lip_sync.src.face3d.models.arcface_torch.configs',
 'sadtalker_video_lip_sync.src.face3d.models.arcface_torch.eval',
 'sadtalker_video_lip_sync.src.face3d.models.arcface_torch.utils',
 'sadtalker_video_lip_sync.src.face3d.options',
 'sadtalker_video_lip_sync.src.face3d.util',
 'sadtalker_video_lip_sync.src.facerender',
 'sadtalker_video_lip_sync.src.facerender.modules',
 'sadtalker_video_lip_sync.src.facerender.sync_batchnorm',
 'sadtalker_video_lip_sync.src.utils',
 'sadtalker_video_lip_sync.third_part.GFPGAN.gfpgan',
 'sadtalker_video_lip_sync.third_part.GFPGAN.gfpgan.archs',
 'sadtalker_video_lip_sync.third_part.GFPGAN.gfpgan.data',
 'sadtalker_video_lip_sync.third_part.GFPGAN.gfpgan.models',
 'sadtalker_video_lip_sync.third_part.GPEN',
 'sadtalker_video_lip_sync.third_part.GPEN.face_detect',
 'sadtalker_video_lip_sync.third_part.GPEN.face_detect.data',
 'sadtalker_video_lip_sync.third_part.GPEN.face_detect.facemodels',
 'sadtalker_video_lip_sync.third_part.GPEN.face_detect.layers',
 'sadtalker_video_lip_sync.third_part.GPEN.face_detect.layers.functions',
 'sadtalker_video_lip_sync.third_part.GPEN.face_detect.layers.modules',
 'sadtalker_video_lip_sync.third_part.GPEN.face_detect.utils',
 'sadtalker_video_lip_sync.third_part.GPEN.face_detect.utils.nms',
 'sadtalker_video_lip_sync.third_part.GPEN.face_model',
 'sadtalker_video_lip_sync.third_part.GPEN.face_model.op',
 'sadtalker_video_lip_sync.third_part.GPEN.face_morpher',
 'sadtalker_video_lip_sync.third_part.GPEN.face_morpher.facemorpher',
 'sadtalker_video_lip_sync.third_part.GPEN.face_parse']

package_data = \
{'': ['*'],
 'sadtalker_video_lip_sync': ['checkpoints/*',
                              'dian_output/*',
                              'examples/driven_audio/*',
                              'examples/driven_video/*',
                              'results/*',
                              'sync_show/*',
                              'third_part/GFPGAN/*',
                              'third_part/GFPGAN/options/*'],
 'sadtalker_video_lip_sync.src': ['config/*'],
 'sadtalker_video_lip_sync.src.face3d.models.arcface_torch': ['docs/*'],
 'sadtalker_video_lip_sync.third_part.GFPGAN.gfpgan': ['weights/*'],
 'sadtalker_video_lip_sync.third_part.GPEN.face_detect.data': ['FDDB/*'],
 'sadtalker_video_lip_sync.third_part.GPEN.face_morpher': ['scripts/*']}

install_requires = \
['basicsr @ git+https://github.com/XPixelGroup/BasicSR.git@master',
 'dlib-bin>=19.24.2.post1,<20.0.0',
 'face-alignment==1.3.5',
 'facexlib==0.2.5',
 'gfpgan>=1.3.8,<2.0.0',
 'gradio>=4.38.1,<5.0.0',
 'imageio-ffmpeg==0.4.7',
 'imageio==2.19.3',
 'kornia==0.6.8',
 'librosa==0.9.2',
 'numba>=0.60.0,<0.61.0',
 'numpy==1.23.4',
 'pydub==0.25.1',
 'pyyaml>=6.0.1,<7.0.0',
 'resampy==0.3.1',
 'torch==2.2.0',
 'torchvision==0.17.0',
 'tqdm>=4.66.4,<5.0.0',
 'yacs==0.1.8']

entry_points = \
{'console_scripts': ['sadtalker = sadtalker_video_lip_sync.inference:main']}

setup_kwargs = {
    'name': 'sadtalker-video-lip-sync',
    'version': '0.1.0',
    'description': '',
    'long_description': '# SadTalker-Video-Lip-Sync\n\n\n本项目基于SadTalkers实现视频唇形合成的Wav2lip。通过以视频文件方式进行语音驱动生成唇形，设置面部区域可配置的增强方式进行合成唇形（人脸）区域画面增强，提高生成唇形的清晰度。使用DAIN 插帧的DL算法对生成视频进行补帧，补充帧间合成唇形的动作过渡，使合成的唇形更为流畅、真实以及自然。\n\n## 1.环境准备(Environment)\n\n```python\npip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113\nconda install ffmpeg\npip install -r requirements.txt\n\n#如需使用DAIN模型进行补帧需安装paddle\n# CUDA 11.2\npython -m pip install paddlepaddle-gpu==2.3.2.post112 \\\n-f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html\n```\n\n## 2.项目结构(Repository structure)\n\n```\nSadTalker-Video-Lip-Sync\n├──checkpoints\n|   ├──BFM_Fitting\n|   ├──DAIN_weight\n|   ├──hub\n|   ├── ...\n├──dian_output\n|   ├── ...\n├──examples\n|   ├── audio\n|   ├── video\n├──results\n|   ├── ...\n├──src\n|   ├── ...\n├──sync_show\n├──third_part\n|   ├── ...\n├──...\n├──inference.py\n├──README.md\n```\n\n## 3.模型推理(Inference)\n\n```python\npython inference.py --driven_audio <audio.wav> \\\n                    --source_video <video.mp4> \\\n                    --enhancer <none,lip,face> \\  #(默认lip)\n                    --use_DAIN \\ #(使用该功能会占用较大显存和消耗较多时间)\n             \t\t--time_step 0.5 #(插帧频率，默认0.5，即25fps—>50fps;0.25,即25fps—>100fps)\n```\n\n\n\n## 4.合成效果(Results)\n\n```python\n#合成效果展示在./sync_show目录下：\n#original.mp4 原始视频\n#sync_none.mp4 无任何增强的合成效果\n#none_dain_50fps.mp4 只使用DAIN模型将25fps添帧到50fps\n#lip_dain_50fps.mp4 对唇形区域进行增强使唇形更清晰+DAIN模型将25fps添帧到50fps\n#face_dain_50fps.mp4 对全脸区域进行增强使唇形更清晰+DAIN模型将25fps添帧到50fps\n\n#下面是不同方法的生成效果的视频\n#our.mp4 本项目SadTalker-Video-Lip-Sync生成的视频\n#sadtalker.mp4 sadtalker生成的full视频\n#retalking.mp4 retalking生成的视频\n#wav2lip.mp4 wav2lip生成的视频\n```\n\nhttps://user-images.githubusercontent.com/52994134/231769817-8196ef1b-c341-41fa-9b6b-63ad0daf14ce.mp4\n\n视频拼接到一起导致帧数统一到25fps了，插帧效果看不出来区别，具体细节可以看./sync_show目录下的单个视频进行比较。\n\n**本项目和sadtalker、retalking、wav2lip唇形合成的效果比较：**\n\n|                           **our**                            |                        **sadtalker**                         |\n| :----------------------------------------------------------: | :----------------------------------------------------------: |\n| <video  src="https://user-images.githubusercontent.com/52994134/233003969-91fa9e94-a958-4e2d-b958-902cc7711b8a.mp4" type="video/mp4"> </video> | <video  src="https://user-images.githubusercontent.com/52994134/233003985-86d0f75c-d27f-4a52-ac31-2649ccd39616.mp4" type="video/mp4"> </video> |\n|                        **retalking**                         |                         **wav2lip**                          |\n| <video  src="https://user-images.githubusercontent.com/52994134/233003982-2fe1b33c-b455-4afc-ab50-f6b40070e2ca.mp4" type="video/mp4"> </video> | <video  src="https://user-images.githubusercontent.com/52994134/233003990-2f8c4b84-dc74-4dc5-9dad-a8285e728ecb.mp4" type="video/mp4"> </video> |\n\nreadme中展示视频做了resize，原始视频可以看./sync_show目录下不同类别合成的视频进行比较。\n\n## 5.预训练模型（Pretrained model）\n\n预训练的模型如下所示：\n\n```python\n├──checkpoints\n|   ├──BFM_Fitting\n|   ├──DAIN_weight\n|   ├──hub\n|   ├──auido2exp_00300-model.pth\n|   ├──auido2pose_00140-model.pth\n|   ├──epoch_20.pth\n|   ├──facevid2vid_00189-model.pth.tar\n|   ├──GFPGANv1.3.pth\n|   ├──GPEN-BFR-512.pth\n|   ├──mapping_00109-model.pth.tar\n|   ├──ParseNet-latest.pth\n|   ├──RetinaFace-R50.pth\n|   ├──shape_predictor_68_face_landmarks.dat\n|   ├──wav2lip.pth\n```\n\n预训练的模型checkpoints下载路径:\n\n百度网盘：https://pan.baidu.com/s/15-zjk64SGQnRT9qIduTe2A  提取码：klfv\n\n谷歌网盘：https://drive.google.com/file/d/1lW4mf5YNtS4MAD7ZkAauDDWp2N3_Qzs7/view?usp=sharing\n\n夸克网盘：https://pan.quark.cn/s/2a1042b1d046  提取码：zMBP\n\n```python\n#下载压缩包后解压到项目路径（谷歌网盘和夸克网盘下载的需要执行）\ncd SadTalker-Video-Lip-Sync\ntar -zxvf checkpoints.tar.gz\n```\n\n## 参考(Reference)\n\n- SadTalker:https://github.com/Winfredy/SadTalker\n-  VideoReTalking：https://github.com/vinthony/video-retalking\n- DAIN :https://arxiv.org/abs/1904.00830\n- PaddleGAN:https://github.com/PaddlePaddle/PaddleGAN\n',
    'author': 'Your Name',
    'author_email': 'you@example.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.10,<4.0',
}


setup(**setup_kwargs)

