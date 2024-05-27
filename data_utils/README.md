# Multi camera alignment

We first extract data (images, LiDAR, camera poses, bboxes, map data) from tfrecord files. We set the first vehicle's coordinate as a new world's coordinate.

Then we use existing SfM software to reclibrate the images and transform the sfm poses back to waymo's coordinate (real-world scale). **COLMAP** and **Metashape** are supported. Since **COLMAP** is open-source and easy to install, we suggest users to use COLMAP in Step 2.

### Step 1: Waymo Data Extraction 
```
python process_waymo_script.py --waymo_data_dir=../data/waymo_tfrecords/1.4.2 --nerf_data_dir=../data/waymo_multi_view
```

### Step 2: Using COLMAP or Metashape to calibrate images


<details>
<summary>COLMAP usage</summary>

#### Step 2.2: Run COLMAP sparse reconstruction:
```shell
bash run_colmap.sh ../data/waymo_multi_view/${SCENE_NAME}
```

This will generate `colmap/sparse_undistorted` folder in your `../data/waymo_multi_view/${SCENE_NAME}`
</details>

<details>
<summary>Metashape usage</summary>

Using Metashape to calibrate images and get `camera.xml`. You need a metashape software with GUI.

First, use `Workflow->Add Folder` to upload the images from `data/waymo_multi_view/${SCENE_NAME}/images`, and then choose `Single Cameras` as follows:
<img src="./instruction_metashape/single_camera.jpg" width="400" />

Second, use `Workflow->Align Photos` to calibrate the images with the following configuration:
<img src="./instruction_metashape/align.jpg" width="400" />

Finally, use `File->Export->Export Cameras` to export the parameters of cameras. Put them in folder `data/waymo_multi_view/${SCENE_NAME}`.
</details>

### Step 3: Convert to waymo coordinate (real-world scale)

If you use COLMAP in Step 2, use the following command to convert to waymo's coordinate. But since we have include this in `run_colmap.sh`, you don't need to run it again. 
```bash
python convert_to_waymo.py -d ../data/waymo_multi_view/${SCENE_NAME} -c colmap
```

If you use Metashape in Step 2, change `--calibration_tool=metashape`. You need to run it mannually.

```bash
python convert_to_waymo.py -d ../data/waymo_multi_view/${SCENE_NAME} -c metashape
```

