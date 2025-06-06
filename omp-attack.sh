# results and dataset will be saved in root_dir
root_dir='mnt/dataset/nuScenes/results'

# target scene instance file path, the content of the file is "{scene_name}-{instance_token}".
# E.g.
# scene_inst_tokens.txt
# - scene-0103-3620feb00d744241a94855f3a8913a78
# - scene-0002-4c663ba19176469ca67944f0f0c6c825
target_scene_instance_file=${root_dir}/attack_pred/scene_inst_tokens.txt

# main code directory
code_dir='mnt/projects/omp-attack'
# nuScenes dataset directory
nusc_datadir='mnt/dataset/nuScenes/trainval'
# nuScenes version
nusc_version='v1.0-trainval'


### transfer target scene nuScenes data to kitti format ###
# todo: need config.yaml. You can modify this file in omp-attack/config.yaml.
cd ${code_dir}/PIXOR_nuscs/srcs
conda activate pixor_nuscs

python -W ignore export_kitti.py \
        --root_dir ${root_dir} \
        --target_scene_instance_file ${target_scene_instance_file} \
        --nusc_datadir ${nusc_datadir} \
        --nusc_version ${nusc_version}
### transfer target scene nuScenes data to kitti format ###


### sampling perturbations ###
cd ${code_dir}/PIXOR_nuscs/srcs
conda activate pixor_nuscs

specific_scenes='scene-0103'
specific_instance='3620feb00d744241a94855f3a8913a78'
begin_frame=14

python -W ignore attack_sampling.py \
        --data_dir ${root_dir} \
        --scene_name ${specific_scenes} \
        --begin_frame ${begin_frame}
### sampling perturbations ###


### multi-frame attack ###
cd ${code_dir}/Trajectron-plus-plus/experiments/nuScenes
conda activate trajectron++

# double vehicle environment preprocess
python -W ignore process_data_nuscs.py \
        --data_dir ${nusc_datadir} \
        --version ${nusc_version} \
        --output_dir ${root_dir}/attack_pred \
        --target_scene_instance_file ${target_scene_instance_file} \
        --scene_name ${specific_scenes}

# multi-frame attack
python -W ignore attack_multi_frame.py \
        --data_dir ${root_dir} --scene_name ${specific_scenes}
### multi-frame attack  ###


### location optimization  ###
cd ${code_dir}/PIXOR_nuscs/srcs
conda activate pixor_nuscs

# modify the para
scene_name=('scene-0103')
target_per_dir=('mnt/dataset/nuScenes/results/scene-0103-3620feb00d744241a94855f3a8913a78')
begin_frame=14

python -W ignore attack_pso.py \
        --data_dir ${root_dir} \
        --scene_name ${scene_name} \
        --target_per_dir ${target_per_dir} \
        --nusc_datadir ${nusc_datadir} \
        --nusc_version ${nusc_version} \
        --begin_frame ${begin_frame}
### location optimization  ###


### detection the result after place the objects at adv loc ###
cd ${code_dir}/PIXOR_nuscs/srcs
conda activate pixor_nuscs

python -W ignore infer_det_result.py \
        --data_dir ${root_dir} \
        --scene_name ${specific_scenes} \
        --nusc_datadir ${nusc_datadir} \
        --nusc_version ${nusc_version} \
        --begin_frame ${begin_frame}
### detection the result after place the objects at adv loc ###


### pred infer ###
cd ${code_dir}/Trajectron-plus-plus/experiments/nuScenes
conda activate trajectron++

# infer
python -W ignore eval_nuscs.py \
        --data_dir ${root_dir} --scene_name ${specific_scenes} --attack_type 'multiframe_attack'
### pred infer ###
