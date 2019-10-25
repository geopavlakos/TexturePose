import os
import torch
from string import upper
import smplx
from smplx.utils import Struct
from smplx.lbs import vertices2joints
from smplx.body_models import ModelOutput
import numpy as np
import config
try:
    import cPickle as pickle
except ImportError:
    import pickle

class SMPL(smplx.SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        smpl_dir = args[0]
        if 'gender' not in kwargs:
            kwargs['gender'] = 'neutral'
        smpl_file = os.path.join(smpl_dir, 'SMPL_%s.pkl' % upper(kwargs['gender']))
        with open(smpl_file, 'rb') as f:
            data_struct = Struct(**pickle.load(f))
        kwargs['data_struct'] = data_struct
        kwargs['create_transl'] = False
        super(SMPL, self).__init__(*args, **kwargs)
        J_regressor_cocoplus = np.load(config.JOINT_REGRESSOR_COCOPLUS)
        self.register_buffer('J_regressor_cocoplus', torch.tensor(J_regressor_cocoplus, dtype=torch.float32))

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        joints = vertices2joints(self.J_regressor_cocoplus, smpl_output.vertices)[:, :14]
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output
