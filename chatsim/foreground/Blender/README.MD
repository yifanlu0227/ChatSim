Put blender software here and install dependency for them. 
You probably have already done these if you follow the project's main README.

We tested with [Blender-3.5.1](https://download.blender.org/release/Blender3.5/blender-3.5.1-linux-x64.tar.xz)

## Installation

### install Blender software
```
cd chatsim/foreground/Blender
wget https://download.blender.org/release/Blender3.5/blender-3.5.1-linux-x64.tar.xz
tar -xvf blender-3.5.1-linux-x64.tar.xz
rm blender-3.5.1-linux-x64.tar.xz
```
You will have `chatsim/foreground/Blender/blender-3.5.1-linux-x64` in this case.

### install utils for Blender's python
locate the internal Python of Blender, for example `chatsim/foreground/Blender/blender-3.5.1-linux-x64/3.5/python/bin/python3.10`

```
export blender_py=$PWD/blender-3.5.1-linux-x64/3.5/python/bin/python3.10

cd utils

# install dependency
$blender_py -m pip install -r requirements.txt 

# or you can use the -i https://pypi.tuna.tsinghua.edu.cn/simple for speedup if you are in Chinese mainland
$blender_py -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

$blender_py setup.py develop
```

Then dependencies are installed into Blender's python (not your system python or conda python).
