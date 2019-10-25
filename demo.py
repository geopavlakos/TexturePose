#!/usr/bin/python
"""
Demo code

To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

In summary, we provide 3 different ways to use our demo code and models:
1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/im1010_bbox.json```.

Example with OpenPose detection .json
```
python demo.py --checkpoint=data/texturepose_checkpoint.pth --img=examples/im1025.png --openpose=examples/im1025_openpose.json
```
Example with predefined Bounding Box
```
python demo.py --checkpoint=data/texturepose_checkpoint.pth --img=examples/im1025.png --bbox=examples/im1025_bbox.json
```
Example with cropped and centered image
```
python demo.py --checkpoint=data/texturepose_checkpoint.pth --img=examples/im1025.png
```

Running the previous command will save the results in ```examples/im1010_{shape,shape_side}.png```. The file ```im1010_shape.png``` shows the overlayed reconstruction of human shape. We also render a side view, saved in ```im1010_shape_side.png```.
"""

from __future__ import division
from __future__ import print_function

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import smplx

from models.encoder import smplresnet50
from models.model_utils import rot6d_to_rotmat, batch_rodrigues
from models.smpl import SMPL
from utils.imutils import crop
from utils.renderer import SMPLRenderer, visualize_mesh
import h5py
import config

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, required=True, help='Path to pretrained checkpoint')
parser.add_argument('--img', type=str, required=True, help='Path to input image')
parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')

def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale

def process_image(img_file, bbox_file, openpose_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    if bbox_file is None and openpose_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        if bbox_file is not None:
            center, scale = bbox_from_json(bbox_file)
        elif openpose_file is not None:
            center, scale = bbox_from_openpose(openpose_file)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    return img

if __name__ == '__main__':
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model
    model = smplresnet50()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_enc'], strict=False)

    model.to(device)
    model.eval()
    f = h5py.File(config.SMPL_MEAN_PARAMS, 'r')
    init_grot = np.array([np.pi, 0., 0.])
    init_pose = np.hstack([init_grot, f['pose'][3:]])
    init_grot = torch.tensor(init_grot.astype('float32'))
    init_pose = torch.tensor(init_pose.astype('float32'))
    init_shape = torch.tensor(f['shape'][:].astype('float32')).to(device).view(1, 10)
    init_cam = torch.tensor([0.9, 0., 0.]).to(device).view(1, 3)
    init_rotmat = batch_rodrigues(init_pose.unsqueeze(0).contiguous())
    init_rot6d = init_rotmat.view(-1,3,3)[:,:,:2].contiguous().view(1,-1).to(device)

    smpl_gen = SMPL(config.SMPL_MODEL_DIR).to(device)

    # Setup renderer for visualization
    renderer = SMPLRenderer(img_size=224, face_path='data/smpl_faces.npy')

    # Preprocess input image and generate predictions
    img = process_image(args.img, args.bbox, args.openpose, input_res=224)
    with torch.no_grad():
        _, _, _, \
        _, _, _, \
        pred_rot6d3, pred_shape3, pred_cam3 = \
        model(img.unsqueeze(0).to(device), init_rot6d, init_shape, init_cam)
        pred_rotmat3 = rot6d_to_rotmat(pred_rot6d3).unsqueeze(0)
        pred_verts = smpl_gen(global_orient=pred_rotmat3[:, [0]], body_pose=pred_rotmat3[:, 1:], betas=pred_shape3, pose2rot=False).vertices
        pred_cam3 = pred_cam3.cpu().numpy()

        
    verts = pred_verts[0].cpu().numpy()
    img = img.permute(1,2,0).cpu().numpy()

    rend_img = visualize_mesh(img, 224, verts, pred_cam3[0], renderer)
    
    # Render side views
    aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
    center = verts.mean(axis=0)
    rot_verts = np.dot((verts - center), aroundy) + center
    
    # Render non-parametric shape
    rend_img_side = visualize_mesh(np.ones_like(img), 224, rot_verts, pred_cam3[0], renderer)
    
    outfile = args.img.split('.')[0] if args.outfile is None else args.outfile

    # Save reconstructions
    cv2.imwrite(outfile + '_shape.png', rend_img[:,:,::-1])
    cv2.imwrite(outfile + '_shape_side.png', rend_img_side[:,:,::-1])
