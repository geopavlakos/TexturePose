"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

H36M_ROOT = '/NAS/data/h36m'
LSP_ROOT = '/NAS/data/lsp/lsp_dataset'
PW3D_ROOT = '/NAS/data/3DPW'
UPI_S1H_ROOT = '/NAS/data/up/upi-s1h'

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'

# Path to test/train npz files
DATASET_FILES = {  'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
                   'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2.npz'),
                   'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
                   '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                   }

DATASET_FOLDERS = {'h36m-p1': H36M_ROOT,
                   'h36m-p2': H36M_ROOT,
                   'lsp': LSP_ROOT,
                   '3dpw': PW3D_ROOT,
                   'upi-s1h': UPI_S1H_ROOT,
                   }

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
JOINT_REGRESSOR_COCOPLUS = 'data/cocoplus_regressor.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
SMPL_MEAN_PARAMS = 'data/neutral_smpl_mean_params.h5'
SMPL_MODEL_DIR = 'data/smpl'
