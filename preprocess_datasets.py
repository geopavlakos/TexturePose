#!/usr/bin/python
"""
Preprocess datasets and generate npz files to be used for training testing.
It is recommended to first read datasets/preprocess/README.md
"""
import argparse
import config as cfg
from datasets.preprocess import h36m_valid_extract,\
                                pw3d_extract, \
                                lsp_dataset_extract

parser = argparse.ArgumentParser()
parser.add_argument('--eval_files', default=False, action='store_true', help='Extract files needed for evaluation')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # define path to store extra files
    out_path = cfg.DATASET_NPZ_PATH

    if args.eval_files:
        # Human3.6M preprocessing (two protocols)
        h36m_extract(cfg.H36M_ROOT, out_path, protocol=1, extract_img=True)
        h36m_extract(cfg.H36M_ROOT, out_path, protocol=2, extract_img=False)
        
        # 3DPW dataset preprocessing (test set)
        pw3d_extract(cfg.PW3D_ROOT, out_path)

        # LSP dataset preprocessing (test set)
        lsp_dataset_extract(cfg.LSP_ROOT, out_path)
