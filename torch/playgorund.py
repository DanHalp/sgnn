from copy import deepcopy
from glob import glob
import shutil
import time
from importlib_metadata import files
from nuscenes.nuscenes import NuScenes
import torch
import numpy as np
import os
import open3d
import open3d as o3d
from data_util import visualize_sparse_locs_as_points, make_scale_transform
import argparse
from scipy.spatial import distance_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="/home/halperin/sgnn/torch/data/datasets/nuscenes_samples", help="Where the samples are")
parser.add_argument("--output", default="/home/halperin/sgnn/torch/#####################################HERE", help="Some bulshit")
parser.add_argument("--num", default=0, type=int, help="How many scans should we take for the complete scans")
parser.add_argument("--train_txt", default="/home/halperin/sgnn/torch/data/datasets/nuscenes_train.txt")
parser.add_argument("--scene_name", default="", help="What scene do we want to work with?")
parser.add_argument('-l','--list', nargs='+', default=[], type=int, help='<Required> Set flag')
args = parser.parse_args()



def load_point_clouds(voxel_size=0.02):
    with open(args.train_txt) as f:
        files = f.read().splitlines()
    
    voxels = []
    for file in files:
        scheisse = os.path.join(args.data_path, file)
        p = open3d.io.read_point_cloud(scheisse)
        p = p.voxel_down_sample(voxel_size=voxel_size)
        p.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        voxels.append(p)
    
    return voxels

def pairwise_registration(source, target, max_correspondence_distance_coarse=1, max_correspondence_distance_fine=0.5):
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    
    addition = int(n_pcds // 10)
    
    for source_id in range(n_pcds):
        # for target_id in range(source_id + 1,n_pcds):
        for target_id in range(source_id + 1, min(source_id + 2 + addition, n_pcds)):
            transformation_icp, information_icp = pairwise_registration(pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

def merge_point_clouds():
    
    voxels = load_point_clouds()
    voxel_size = 0.02
    max_correspondence_distance_coarse = voxel_size * 2
    max_correspondence_distance_fine = voxel_size * 1
    s = time.time()
    pose_graph = full_registration(deepcopy(voxels),
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)
    print(time.time() - s)
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=voxel_size,
        reference_node=0)
    
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    pcds = deepcopy(voxels)
    pcd_combined = o3d.geometry.PointCloud() + pcds[0]
    visualize_sparse_locs_as_points(np.asarray(pcds[0].voxel_down_sample(voxel_size).points), os.path.join(args.output,f"{ 1}.ply"), make_scale_transform(1), pcd=True)
    pcd_not_transformed = o3d.geometry.PointCloud()
    # x = np.asarray(pcds[0].points)
    # mat = distance_matrix(x, np.array([[0,0,0]]))
    # x = x[mat.flatten() < 15]

    for point_id in range(1, len(pcds)):
        pcd_not_transformed += pcds[point_id]
    
        visualize_sparse_locs_as_points(np.asarray(pcds[point_id].voxel_down_sample(voxel_size).points), os.path.join(args.output,f"{point_id + 1}.ply"), make_scale_transform(1), pcd=True)    
        # y = np.asarray(pcds[point_id].points)
        # mat = distance_matrix(y, np.array([[0,0,0]]))
        # y = y[mat.flatten() < 15]
        # mat = distance_matrix(y, x)
        # indices = np.min(mat, axis=1) 
        # new_pcd = o3d.geometry.PointCloud()
        # new_pcd.points = o3d.utility.Vector3dVector(y[indices <= 0.3]) 
        # visualize_sparse_locs_as_points(np.asarray(new_pcd.voxel_down_sample(voxel_size).points), os.path.join(args.output,f"hara.ply"), make_scale_transform(1), pcd=True)    

        pcd_combined += pcds[point_id]
        
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    pcd_combined_down = pcd_combined_down.voxel_down_sample(voxel_size=voxel_size)
    points = np.asarray(pcd_combined_down.points)
    
    print(f"len of points {len(points)}")
    visualize_sparse_locs_as_points(points, os.path.join(args.output, "transformed.ply"), make_scale_transform(1), pcd=True)
    
    pcd_combined_down_2 = pcd_not_transformed.voxel_down_sample(voxel_size=voxel_size)
    points = np.asarray(pcd_combined_down_2.points)
    visualize_sparse_locs_as_points(points, os.path.join(args.output,"transformed2.ply"), make_scale_transform(1), pcd=True)
    
    if args.list:
        pcds = deepcopy(voxels)
        pcd_combined = o3d.geometry.PointCloud()
        for point_id in args.list:
            point_id -= 1
            pcds[point_id].transform(pose_graph.nodes[point_id].pose)
            print(pose_graph.nodes[point_id].pose)
            pcd_combined += pcds[point_id]
        pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
        points = np.asarray(pcd_combined_down.points)
        visualize_sparse_locs_as_points(points, os.path.join(args.output,"partial.ply"), make_scale_transform(1), pcd=True)
        
        


def create_train_file_one_scene(scene_name):
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)
    
    ret_dict = dict()
    if args.num > 0 and scene_name:
        
        files = sorted(glob(os.path.join(args.data_path, scene_name) + "*"))[:args.num] 
        with open(args.train_txt, "w") as f:
            for i, file in enumerate(files):
                file = os.path.basename(file)
                if i + 1== len(files):
                    print(file, file=f, end="")
                else:
                    print(file, file=f)
    
    return ret_dict

if __name__ == "__main__":
    create_train_file_one_scene(args.scene_name)
    merge_point_clouds()
    
