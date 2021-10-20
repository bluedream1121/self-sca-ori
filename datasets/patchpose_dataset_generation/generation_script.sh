# PatchPoseA generation
python data_generation.py --dataset patchPoseA --dataset_path ./SPair-71k --num_patches_per_image 3  --output_dir ../patchPose
python filter.py --dataset patchPoseA  --dataset_path ../patchPose
python dataset_split.py --dataset patchPoseA  --dataset_path ../patchPose

# PatchPoseB generation
python data_generation.py --dataset patchPoseB --dataset_path ./SPair-71k --num_patches_per_image 3  --output_dir ../patchPose
python filter.py --dataset patchPoseB  --dataset_path ../patchPose
python dataset_split.py --dataset patchPoseB --dataset_path ../patchPose

