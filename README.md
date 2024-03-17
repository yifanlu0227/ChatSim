# ChatSim
Editable Scene Simulation for Autonomous Driving via LLM-Agent Collaboration

[Arxiv](https://arxiv.org/abs/2402.05746) | [Project Page](https://yifanlu0227.github.io/ChatSim/) | [Video](https://youtu.be/5xWz5YBsE5M)

![teaser](./img/teaser.jpg)

## Requirement
- Ubuntu version >= 20.04 (for using Blender 3.+)
- Metashape software (not necessary, we provide recalibrated poses)
- OpenAI API Key

## Installation
First clone this repo recursively.

```bash
git clone https://github.com/yifanlu0227/ChatSim.git --recursive
```

### Step 1: Setup environment
```
conda create -n chatsim python=3.8
conda activate chatsim

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
imageio_download_bin freeimage
```

### Step 2: Install McNeRF 
The installation is the same as [F2-NeRF](https://github.com/totoro97/f2-nerf). Please go through the following steps.

```bash
cd chatsim/background/mcnerf/

# mcnerf use the same data directory. 
ln -s ../../../data .
```

#### Step 2.1: Install dependencies

For Debian based Linux distributions:
```
sudo apt install zlib1g-dev
```

For Arch based Linux distributions:
```
sudo pacman -S zlib
```

#### Step 2.2: Download pre-compiled LibTorch
Taking `torch-1.13.1+cu117` for example.
```shell
cd chatsim/background/mcnerf
cd External

wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip
unzip ./libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip
rm ./libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip
```

#### Step 2.3: Compile
The lowest g++ version is 7.5.0. 
```shell
cd ..
cmake . -B build
cmake --build build --target main --config RelWithDebInfo -j
```

### Step 3: Install Inpainting tools

#### Step 3.1: Setup Video Inpainting
```bash
cd ../inpainting/Inpaint-Anything/
python -m pip install -e segment_anything
```
Go [here](https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing) to download [pretrained_models](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing), put the directory into `./` and get `./pretrained_models`. Additionally, download [pretrain](https://drive.google.com/drive/folders/1SERTIfS7JYyOOmXWujAva4CDQf-W7fjv?usp=sharing), put the directory into `./pytracking` as `./pytracking/pretrain`.

#### Step 3.2: Setup Image Inpainting
```bash
cd ../latent-diffusion
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .

# download pretrained ldm
wget -O models/ldm/inpainting_big/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
```

### Step 4: Install Blender Software and our Blender Utils
We tested with [Blender 3.5.1](https://download.blender.org/release/Blender3.5/blender-3.5.1-linux-x64.tar.xz). Note that Blender 3+ requires Ubuntu version >= 20.04.

#### Step 4.1: Install Blender software
```bash
cd ../../Blender
wget https://download.blender.org/release/Blender3.5/blender-3.5.1-linux-x64.tar.xz
tar -xvf blender-3.5.1-linux-x64.tar.xz
rm blender-3.5.1-linux-x64.tar.xz
```

#### Step 4.2: Install blender utils for Blender's python
locate the internal Python of Blender, for example `blender-3.5.1-linux-x64/3.5/python/bin/python3.10`

```bash
export blender_py=$PWD/blender-3.5.1-linux-x64/3.5/python/bin/python3.10

cd utils

# install dependency (use the -i https://pypi.tuna.tsinghua.edu.cn/simple if you are in the Chinese mainland)
$blender_py -m pip install -r requirements.txt 
$blender_py -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

$blender_py setup.py develop
```

### Step 5: Install McLight (optional)
If you want to train the skydome model, follow the READMD in `chatsim/foreground/mclight/skydome_lighting/readme.md`. You can download our provided skydome HDRI in the next section and start the simulation.


## Usage

### Data Preparation

#### Download and extract waymo data
```bash
mkdir data
mkdir data/waymo_tfrecords
mkdir data/waymo_tfrecords/1.4.2
```
Download the [waymo perception dataset v1.4.2](https://waymo.com/open/download/) to the `data/waymo_tfrecords/1.4.2`. In the google cloud console, a correct folder path is `waymo_open_dataset_v_1_4_2/individual_files/training`. Static scene are listed here

<details>
<summary><span style="font-weight: bold;">Static waymo scenes</span></summary>

- segment-10061305430875486848_1080_000_1100_000_with_camera_labels
- segment-10247954040621004675_2180_000_2200_000_with_camera_labels
- segment-10275144660749673822_5755_561_5775_561_with_camera_labels
- segment-10676267326664322837_311_180_331_180_with_camera_labels
- segment-11379226583756500423_6230_810_6250_810_with_camera_labels
- segment-1172406780360799916_1660_000_1680_000_with_camera_labels
- segment-12879640240483815315_5852_605_5872_605_with_camera_labels
- segment-13085453465864374565_2040_000_2060_000_with_camera_labels
- segment-13142190313715360621_3888_090_3908_090_with_camera_labels
- segment-13196796799137805454_3036_940_3056_940_with_camera_labels
- segment-13238419657658219864_4630_850_4650_850_with_camera_labels
- segment-13469905891836363794_4429_660_4449_660_with_camera_labels
- segment-14004546003548947884_2331_861_2351_861_with_camera_labels
- segment-14333744981238305769_5658_260_5678_260_with_camera_labels
- segment-14348136031422182645_3360_000_3380_000_with_camera_labels
- segment-14424804287031718399_1281_030_1301_030_with_camera_labels
- segment-14869732972903148657_2420_000_2440_000_with_camera_labels
- segment-15221704733958986648_1400_000_1420_000_with_camera_labels
- segment-15270638100874320175_2720_000_2740_000_with_camera_labels
- segment-15349503153813328111_2160_000_2180_000_with_camera_labels
- segment-15365821471737026848_1160_000_1180_000_with_camera_labels
- segment-15868625208244306149_4340_000_4360_000_with_camera_labels
- segment-16345319168590318167_1420_000_1440_000_with_camera_labels
- segment-16470190748368943792_4369_490_4389_490_with_camera_labels
- segment-16608525782988721413_100_000_120_000_with_camera_labels
- segment-16646360389507147817_3320_000_3340_000_with_camera_labels
- segment-17761959194352517553_5448_420_5468_420_with_camera_labels
- segment-3425716115468765803_977_756_997_756_with_camera_labels
- segment-3988957004231180266_5566_500_5586_500_with_camera_labels
- segment-4058410353286511411_3980_000_4000_000_with_camera_labels
- segment-8811210064692949185_3066_770_3086_770_with_camera_labels
- segment-9385013624094020582_2547_650_2567_650_with_camera_labels

</details>

After downloading tfrecords, you should see folder structure like 

```
data
|-- ...
|-- ...
`-- waymo_tfrecords
    `-- 1.4.2
        |-- segment-10247954040621004675_2180_000_2200_000_with_camera_labels.tfrecord
        |-- segment-11379226583756500423_6230_810_6250_810_with_camera_labels.tfrecord
        |-- ...
        `-- segment-1172406780360799916_1660_000_1680_000_with_camera_labels.tfrecord
```
We extract the images, camera poses, LiDAR file, etc. out of the tfrecord files with the `data_utils/process_waymo_script.py`. 

```bash
cd data_utils
python process_waymo_script.py --waymo_data_dir=../data/waymo_tfrecords/1.4.2 --nerf_data_dir=../data/waymo_multi_view
```
This will generate the data folder `data/waymo_multi_view`. 

#### Recalibrate waymo data (or just download our recalibrated files)
Use Metashape to calibrate images in the `data/waymo_multi_view/{SCENE_NAME}/images` folder and convert them back to the waymo world coordinate. Please follow the tutorial in `data_utils/README.md`. And the final camera extrinsics and intrinsics are stored as `cam_meta.npy` and `poses_bounds.npy`.

Or you can download our recalibration files [here](https://huggingface.co/datasets/yifanlu/waymo_recalibrated_poses/tree/main) and run the final step in the tutorial. After converting the recalibrated camera extrinsics and intrinsics back to waymo's coordinate, you should see 

```bash
data
`-- waymo_multi_view
    |-- ...
    `-- segment-1172406780360799916_1660_000_1680_000_with_camera_labels
        |-- 3d_boxes.npy                # 3d bounding boxes of the first frame
        |-- images                      # a clip of waymo images used in chatsim (typically 40 frames)
        |-- images_all                  # full waymo images (typically 198 frames)
        |-- map.pkl                     # map data of this scene
        |-- point_cloud                 # point cloud file of the first frame
        |-- camera.xml                  # relibration file from Metashape
        |-- cams_meta.npy               # Camera ext&int calibrated by metashape and transformed to waymo coordinate system.
        |-- poses_bounds.npy            # Camera ext&int calibrated by metashape and transformed to waymo coordinate system (for mcnerf training)
        |-- poses_bounds_metashape.npy  # Camera ext&int calibrated by metashape
        |-- poses_bounds_waymo.npy      # Camera ext&int from original waymo dataset
        |-- shutters                    # normalized exposure time (mean=0 std=1)
        |-- tracking_info.pkl           # tracking data
        `-- vehi2veh0.npy               # transformation matrix from i-th frame to the first frame.
```

### Step 6: Setup Trajectory Tracking Module(optional)
If you want to get smoother and more realistic trajectories, you can install the trajectory module and change the parameter motion_agent-motion_tracking to True in .yaml file. For installation(both code and pretrained model), you can run the following commands in terminal
```bash
pip install frozendict gym==0.26.2 stable-baselines3[extra] protobuf==3.20.1

cd chatsim/foreground
git clone --recursive git@github.com:MARMOTatZJU/drl-based-trajectory-tracking.git -b v1.0.0

cd drl-based-trajectory-tracking
source setup-minimum.sh
```
Then switch to the original directory to run main.py. And if the parameter motion_agent-motion_tracking is set as True, each trajectory will be tracked by this module to make it smoother and more realistic.

### Train the model

#### Download pretrain for quick start-up
You need to train the McNeRF model for each scene as well as the McLight's skydome estimation network. To get started quickly, you can download our skydome estimation and some Blender 3D assets.

- [Skydome HDRI](https://huggingface.co/datasets/yifanlu/Skydome_HDRI/tree/main). Download and put them in `data/waymo_skydome`
- [Blender Assets](https://huggingface.co/datasets/yifanlu/Blender_3D_assets/tree/main). Download and put them in `data/blender_assets`. Our 3D models are collected from the Internet. We tried our best to contact the author of the model and ensure that copyright issues are properly dealt with (our open source projects are not for profit). If you are the author of a model and our behavior infringes your copyright, please contact us immediately and we will delete the model.


### Train McNeRF
```
cd chatsim/background/mcnerf
```
Make sure you have the `data` folder linking to `../../../data`, and train your model with 

```
python scripts/run.py --config-name=wanjinyou_big \
dataset_name=waymo_multi_view case_name=${CASE_NAME} \
exp_name=${EXP_NAME} dataset.shutter_coefficient=0.15 mode=train_hdr_shutter +work_dir=$(pwd) 
```
where `${CASE_NAME}` are those like `segment-11379226583756500423_6230_810_6250_810_with_camera_labels` and `${EXP_NAME}` can be anything like `exp_coeff_0.15`. `dataset.shutter_coefficient = 0.15` or `dataset.shutter_coefficient = 0.3` work well.

You can simply run scripts like `bash train-1137.sh` for training and `bash render_novel_view-1137.sh` for testing. 


#### Start simulation

```bash
export OPENAI_API_KEY=<your api key>
```

Now you can start the simulation with
``` bash
python main.py -y ${CONFIG YAML} \
               -p ${PROMPT} \
               -s ${SIMULATION NAME}
```

- `${CONFIG YAML}` specifies the scene information, and yamls are stored in `config` folder. e.g. `config/waymo-1137.yaml`.

- `${PROMPT}` is your input prompt, which should be wrapped in quotation marks. e.g. `add a straight driving car in the scene`.

- `${SIMULATION NAME}` determines the name of the folder when saving results. default `demo`.

You can try
``` bash
python main.py -y config/waymo-1137.yaml -p 'add a straight driving car in the scene' -s demo
```

The rendered results are saved in `results/1137_demo_%Y_%m_%d_%H_%M_%S`. Intermediate files are saved in `results/cache/1137_demo_%Y_%m_%d_%H_%M_%S` for debug and visualization if `save_cache` are enabled in `config/waymo-1137.yaml`.

#### Config file explanation
`config/waymo-1137.yaml` contains the detailed explanation for each entry. We will give some extra explanation. Suppose the yaml is read into `config_dict`:

- `config_dict['scene']['is_wide_angle']` determines the rendering view. If set to `True`, we will expand waymo's intrinsics (width -> 3 x width) to render wide-angle images. Also note that `is_wide_angle = True` comes with `rendering_mode = 'render_wide_angle_hdr_shutter'`; `is_wide_angle = False` comes with `rendering_mode = 'render_hdr_shutter'`

- `config_dict['scene']['frames']` the frame number for rendering.

- `config_dict['agents']['background_rendering_agent']['nerf_quiet_render']` will determine whether to print the output of mcnerf to the terminal. Set to `False` for debug use.

- `config_dict['agents']['foreground_rendering_agent']['use_surrounding_lighting']` defines whether we use the surrounding lighting. Currently `use_surrounding_lighting = True` only takes effect when merely one vehicle is added, because HDRI is a global illumination in Blender. It is difficult to set a separate HDRI for each car. `use_surrounding_lighting = True` can also lead to slow rendering, since it will call nerf `#frame` times. We set it to `False` in each default yaml. 

- `config_dict['agents']['foreground_rendering_agent']['skydome_hdri_idx']` is the filename (w.o. extension) we choose from `data/waymo_skydome/${SCENE_NAME}/`. It is the skydome HDRI estimation from the first frame(`'000'`) by default, but you can manually select a better estimation from another frame. To view the HDRI, we recommend the [VERIV](https://github.com/mcrescas/veriv) for vscode and [tev](https://github.com/Tom94/tev) for desktop environment.



### Train McLight's Skydome estimation network
Go to `chatsim/foreground/mclight/skydome_lighting` and follow `chatsim/foreground/mclight/skydome_lighting/readme.md` for the training.

## Todo
- [x] arxiv paper release
- [x] code and model release
- [x] motion tracking module [drl-based-trajectory-tracking ](https://github.com/MARMOTatZJU/drl-based-trajectory-tracking) (to smooth trajectory) 
- [ ] multi-round wrapper code

## Citation
```
@InProceedings{wei2024editable,
      title={Editable Scene Simulation for Autonomous Driving via Collaborative LLM-Agents}, 
      author={Yuxi Wei and Zi Wang and Yifan Lu and Chenxin Xu and Changxing Liu and Hao Zhao and Siheng Chen and Yanfeng Wang},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month={June},
      year={2024},
}
```
