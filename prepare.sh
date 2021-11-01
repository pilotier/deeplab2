sudo apt install protobuf-compiler -y

git clone https://github.com/tensorflow/models.git
git clone https://github.com/cocodataset/cocoapi.git
cd ./cocoapi/PythonAPI
make
cd ../..

cd /usr/lib/python3/dist-packages/tensorflow/include/third_party
mkdir gpus/cuda
sudo ln -s /usr/include /usr/lib/python3/dist-packages/tensorflow/include/third_party/gpus/cuda

pip install numpy==1.19.5
pip install cython



# bash ./prepare_data/merge_datasets.sh /mnt/test2fs/dataset/cityscapes_vps  /mnt/test2fs/dataset/cityscapes/leftImg8bit_sequence/val/  /mnt/test2fs/dataset/cityscapes/gtFine/val/
# bash ./prepare_data/create_panoptic_labels.sh /mnt/test2fs/dataset/cityscapes_vps
# pip install git+https://github.com/cocodataset/panopticapi.git