´´´python infer_pair_replica.py  data/replica_v2/office_0/imap/00/rgb/rgb_0.png  data/repli
ca_v2/office_0/imap/00/rgb/rgb_50.png --traj-file  data/replica_v2/office_0
/imap/00/traj_w_cgl.txt --outdir outputs/pair_0_50 --model-dir /cluster/51/
dengnick/models/splatt3r --save-npz
´´´
´´´ python merge_gaussians.py outputs/pair_*/gaussians.npz --out outputs/merged/merged.npz --ply outputs/merged/merged.ply --radius 0.02´´´
