# Installing Pet

This document covers how to install Pet, its dependencies (including Pytorch), and the COCO dataset.

- For general information about Pet, please see [`README.md`](README.md).

**Requirements:**

- NVIDIA GPU, Linux, Python3.6
- Pytorch-1.6, various standard Python packages and the COCO API; Instructions for installing these dependencies are found below

**Notes:**

- PytorchEveryThing has been tested extensively with CUDA >= 10.1.243 and cuDNN 7.5.1.


## Python3.6

To install Python3.6 and soft link to `python`.

1. Add source:

```
sudo apt-get install python-software-properties

sudo apt-get install software-properties-common

sudo add-apt-repository ppa:fkrull/deadsnakes

sudo apt-get update
```

2. Install python3.6:

```
sudo apt-get install python3.6

sudo apt-get install python3.6-dev

sudo apt-get install python3.6-tk
```

3. Soft link to `python` and check (you can use `python2` for `python2.7`):

```
sudo rm -r /usr/bin/python

sudo ln -s /usr/bin/python3.6 /usr/bin/python

python3 -V
```

4. Install `pip3` and upgrade (if you want to user pip to install packages for python2, please use `pip2`):

```
sudo apt-get install python3-pip

sudo pip3 install --upgrade pip
```

   **Note:** If some packages cannot be installed, you can go into `/usr/local/lib/python3.6/dist-packages/` or `/usr/lib/python3/dist-packages/` or `/usr/lib/python3.6/` to delete the old version by `sudo`.


5. Wrapper `opencv` to `python`:

```
sudo pip3 install opencv-python
```

For install opencv3, please refer to [`Compile_opencv3.md`](https://github.com/soeaver/environment_config/blob/master/Compile_opencv3.md)


## Pytorch and torchvision

Install Pytorch with CUDA support.

1. Install Pytorch-1.6.0:

```
sudo pip3 install torch==1.6.0
```

2. Install torchvision:

```
sudo pip3 install torchvision==0.7.0
```

## Pet

1. Clone the Pet repository:

```
git clone https://github.com/BUPT-PRIV/Pet-dev.git
```

2. Install the requirements.txt:

```
cd Pet-dev/pet
sudo pip3 install -r requirements.txt
```

3. Set up `pet`:

```
cd Pet-dev/pet
sh make.sh
```

   **Note:** If you get `CompileError: command 'x86_64-linux-gnu-gcc'`, please:

```
export CXXFLAGS="-std=c++11"

export CFLAGS="-std=c99"

sh ./make.sh
```

   **Note:** If you get `name 'unicode' is not defined` when use cocoapi, please:

```
sudo vim /usr/local/lib/python3.6/dist-packages/pycocotools/coco.py

change 'if type(resFile) == str or type(resFile) == unicode:' to
'if type(resFile) == str:' in line num 308
```

   **Note:** If you get `RuntimeError: Expected a 'N2at13CUDAGeneratorE' but found 'PN2at9GeneratorE'`, please:

```
vim {python_path}/site-packages/torch/utils/data/distributed.py

change
indices = torch.randperm(len(self.dataset), generator=g).tolist()
to
cpu = torch.device('cpu')
indices = torch.randperm(len(self.dataset), device=cpu, generator=g).tolist()
```
