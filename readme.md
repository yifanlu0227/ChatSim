# ChatSim

## Installation
clone this repo.

```bash
git clone https://github.com/yifanlu0227/ChatSim.git --recursive
```

### Step 2: Setup environment
```
conda create -n chatsim python=3.8
conda activate chatsim

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
imageio_download_bin freeimage
```

### Step 1: Install McNeRF 
The installation follows [F2-NeRF](https://github.com/totoro97/f2-nerf).

#### Step 1-1 Install dependencies

For Debian based Linux distributions:
```
sudo apt install zlib1g-dev
```

For Arch based Linux distributions:
```
sudo pacman -S zlib
```

#### Step 1-2 Download pre-compiled LibTorch
Taking `torch-1.13.1+cu117` for example.
```shell
cd chatsim/background/mcnerf
cd External

wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip
unzip ./libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip
rm ./libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip
```

#### Step 1-3 Compile
The lowest g++ version is 7.5.0. You may need to add the Libtorch's lib to LIBRARY_PATH and LD_LIBRARY_PATH
```shell
cd ..
cmake . -B build
cmake --build build --target main --config RelWithDebInfo -j
```

### Step 2: Install Inpainting tools

#### Step 2-1 Setup Video Inpainting

```
cd chatsim/background/inpainting/Inpaint-Anything
python -m pip install -e segment_anything
```
Download the model checkpoints provided in [Segment Anything](./segment_anything/README.md) and [STTN](./sttn/README.md) (e.g., [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and [sttn.pth](https://drive.google.com/file/d/1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv/view)), and put them into `./pretrained_models`. Further, download [OSTrack](https://github.com/botaoye/OSTrack) pretrained model from [here](https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy) (e.g., [vitb_384_mae_ce_32x4_ep300.pth](https://drive.google.com/drive/folders/1XJ70dYB6muatZ1LPQGEhyvouX-sU_wnu)) and put it into `./pytracking/pretrain`. For simplicity, you can also go [here](https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing), directly download [pretrained_models](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing), put the directory into `./` and get `./pretrained_models`. Additionally, download [pretrain](https://drive.google.com/drive/folders/1SERTIfS7JYyOOmXWujAva4CDQf-W7fjv?usp=sharing), put the directory into `./pytracking` and get `./pytracking/pretrain`.

#### Step 2-2 Setup Image Inpainting
```bash
cd chatsim/background/inpainting/latent-diffusion
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .

# download pretrained autoencoder and ldm
bash scripts/download_first_stages.sh
bash scripts/download_models.sh
```

### Step 3: Install McLight
If you want to train the skydome model, install this submodule and follow the readme `chatsim/foreground/mclight/skydome_lighting/readme.md`. You can use our provided HDRI and skip this step.
```
cd chatsim/foreground/mclight/skydome_lighting

python setup.py develop
```

### Step 4: Install Blender Software and our Blender Utils
We tested with [Blender 3.5.1](https://download.blender.org/release/Blender3.5/blender-3.5.1-linux-x64.tar.xz). Note that Blender 3+ requires Ubuntu version >= 20.04.

#### Step 4-1 Install Blender software
```
cd chatsim/foreground/Blender
wget https://download.blender.org/release/Blender3.5/blender-3.5.1-linux-x64.tar.xz
tar -xvf blender-3.5.1-linux-x64.tar.xz
rm blender-3.5.1-linux-x64.tar.xz
```

#### Step 4-2 Install blender utils for Blender's python
locate the internal Python of Blender, for example `blender-3.5.1-linux-x64/3.5/python/bin/python3.10`

```
export blender_py=$PWD/blender-3.5.1-linux-x64/3.5/python/bin/python3.10

cd utils

# install dependency (use the -i https://pypi.tuna.tsinghua.edu.cn/simple if you are in the Chinese mainland)
$blender_py -m pip install -r requirements.txt 
$blender_py -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

$blender_py setup.py develop
```

