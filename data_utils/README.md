We suggest create a new virtual environment to process waymo data by 
```
pip install waymo-open-dataset-tf-2-4-0==1.4.1
pip install opencv-python
pip install open3d
pip install imageio
```

### Step 1: Run 
```
python process_waymo_script.py --waymo_data_dir=../data/waymo_tfrecords/1.4.2 --nerf_data_dir=../data/waymo_multi_view
```

### Step 2: Using Metashape to calibrate images and get `camera.xml`

#### Step 2.1: Use `Workflow->Add Folder` to upload the images, and then choose `Single Cameras` as follows:
<img src="./instruction_metashape/single_camera.jpg" width="400" />

#### Step 2.2: Use `Workflow->Align Photos` to calibrate the images with the following configuration:
<img src="./instruction_metashape/align.jpg" width="400" />

#### Step 2.3: Use `File->Export->Export Cameras` to export the parameters of cameras.

### Step 3: Run 
```
python convert_metashape_to_waymo_script.py --datadir=../data/waymo_multi_view
```
