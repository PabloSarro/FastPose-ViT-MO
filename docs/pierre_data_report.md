# Data issues in `BOP_Datasets` (draft for Pierre)

Draft written 2026-11-03 (M8 bullet 1). All items re-checked on 2026-11-03 from the RCP `dev` job.
To be sent by Pablo; nothing was changed in the datasets folder (read only for us).

---

Hi Pierre,

While setting up BOP training on your `BOP_Datasets` folder
(`/mnt/cvlab/scratch/cvlab/home/pvancey/datasets/BOP_Datasets`), I found a few issues. None of them
blocks me (I train YCB-V on PBR only for now), but I thought you would like to know:

1. **`hb/` is empty.** The HB zips (`hb_base.zip`, `hb_models.zip`, `hb_train_pbr.zip`,
   `hb_test_*`, `hb_val_*`) were downloaded into `xyzibd/` instead.
2. **`xyzibd/` has zips only, and some are missing:** `xyzibd_train_pbr.z01`, `xyzibd_models.zip`
   and `xyzibd_test_all.zip` are not there, so it cannot be extracted.
3. **YCB-V `train_real` is incomplete:** `ycbv_train_real.z01` is missing, so only part of the
   archive was extracted (`ycbv_train_real/train_real/` has 3 scenes, 000005, 000010, 000012, with
   1, 0 and 0 images; the full split has 80 scenes). The other YCB-V splits look complete.
4. **A stray empty folder literally named `*`** at the top level of `BOP_Datasets/`.
5. **IPD test GT in Jupyter checkpoints:** `ipd/test/000000/.ipynb_checkpoints/` and
   `ipd/test/000004/.ipynb_checkpoints/` contain `scene_gt_cam1-checkpoint.json` (and
   `scene_gt_info_cam1-checkpoint.json`), i.e. test ground truth that BOP keeps hidden. Probably
   worth deleting so nobody uses it by accident. (There are also `.ipynb_checkpoints/` folders in
   `ipd/train_pbr/000000/mask_cam1/` and `ipd/train_pbr/000026/`, harmless.)

Thanks a lot for the datasets!

Pablo
