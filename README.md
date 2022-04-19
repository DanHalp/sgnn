# SG-NN

This forked repo follows SG-NN. This work presents a self-supervised approach that converts partial and noisy RGB-D scans into high-quality 3D scene reconstructions by inferring unobserved scene geometry. For more details please see our paper.
- [SG-NN: Sparse Generative Neural Networks for Self-Supervised Scene Completion of RGB-D Scans](https://arxiv.org/pdf/1912.00036.pdf).

[<img src="sgnn.jpg">](https://arxiv.org/abs/1912.00036)

In this work, we adapt this state of the art model to suit outdoor scenery, collected by LiDAR pointclouds from the nuScenes comprehensive dataset.

## Code
### Installation:  
Training is implemented with [PyTorch](https://pytorch.org/). This code was developed with:
- Python 3.7
- PyTorch 1.11.0
- CUDA 11.6
-  [SparseConvNet](https://github.com/facebookresearch/SparseConvNet). For installation, use `python setup.py develop`  
-  For visualization, please install the marching cubes by `python setup.py install` in `marching_cubes`.



### Training:  
* See `python train.py --help` for all train options. 
* Example command: `python train.py --gpu 0 --data_path ./data/completion_blocks --train_file_list ../filelists/train_list.txt --val_file_list ../filelists/val_list.txt --save_epoch 1 --save logs/mp --max_epoch 4`
* Trained model: [sgnn.pth](http://kaldir.vc.in.tum.de/adai/SGNN/sgnn.pth) (7.5M)

### Testing
* See `python test_scene.py --help` for all test options. 
* Example command: `python test_scene.py --gpu 0 --input_data_path ./data/mp_sdf_vox_2cm_input --target_data_path ./data/mp_sdf_vox_2cm_target --test_file_list ../filelists/mp-rooms_val-scenes.txt --model_path sgnn.pth --output ./output  --max_to_vis 20`


### Data (Original)- ScanNet:
* Scene data: 
  - [mp_sdf_vox_2cm_input.zip](http://kaldir.vc.in.tum.de/adai/SGNN/mp_sdf_vox_2cm_input.zip) (44G)
  - [mp_sdf_vox_2cm_target.zip](http://kaldir.vc.in.tum.de/adai/SGNN/mp_sdf_vox_2cm_target.zip) (58G)
* Train data:
  - [completion_blocks.zip](http://kaldir.vc.in.tum.de/adai/SGNN/completion_blocks.zip) (88G)
* [GenerateScans](datagen/GenerateScans) depends on the [mLib](https://github.com/niessner/mLib) library.


### Tasks:

* .sfds: What is the format? How was it collected? 
* Distance fields vs. Occupancy fields: Do we just expect the code to perform the same?
    - The original code has "occ" lost, meaning it might already deal with occupancy fields.
* Do we need to voxelize the input .pcd point clouds from nuScenes?
    - We need to that with hierarchy, for different resolutions.
* In the .sfds file we hold the affine transformation matrix, which is then assigned to the variable "world2grid". Is it connected to the RGB-D? Is the TSDF created from 2D images? The RGB-D are two dimensional images, with 4 channels, where the 4th is the depth value . If so, I assume it is not relevant to point clouds...


* Modify the losses, to remove the l1 loss and stay only with BCE.
    - Check again what the occ loss is, debug the compute_losses()
* Test with differenct truncations:
  - 3
  - 1
  - 2e-2


### Notes:
* The model takes only the sdf as an input.
* The original work creates its input by sampling a subset of the frames, to create an intermediate incomplete scene. In the new works. Should do the same for the nuScenes part.
* World2grid is the affine transformation matrix of the RGB-D - useless for the pointclouds!
* "Known" - TODO

