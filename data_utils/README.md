# Multi camera alignment

### Step 1: Run 
```
python process_waymo_script.py --waymo_data_dir=../data/waymo_tfrecords/1.4.2 --nerf_data_dir=../data/waymo_multi_view
```

### Step 2: Using Metashape to calibrate images and get `camera.xml`

#### Step 2.1: Use `Workflow->Add Folder` to upload the images from `data/waymo_multi_view/{SCENE_NAME}/images`, and then choose `Single Cameras` as follows:
<img src="./instruction_metashape/single_camera.jpg" width="400" />

#### Step 2.2: Use `Workflow->Align Photos` to calibrate the images with the following configuration:
<img src="./instruction_metashape/align.jpg" width="400" />

#### Step 2.3: Use `File->Export->Export Cameras` to export the parameters of cameras. Put them in folder `data/waymo_multi_view/{SCENE_NAME}`.


### Step 3: Convert to waymo coordinate
```
python convert_metashape_to_waymo_script.py --datadir=../data/waymo_multi_view
```
