import torch
import torch.nn as nn
import os
import os.path as osp
from wilor.configs import get_config
from wilor.models import WiLoR as WiLoRModel
from wilor.utils.renderer import cam_crop_to_full
from pytorch3d.transforms import matrix_to_axis_angle
from utils.mano import mano
from config import cfg


def load_wilor_from_repo(checkpoint_path, cfg_path):
    print('Loading ', checkpoint_path)
    model_cfg = get_config(cfg_path, update_cachedir=True)

    if ('vit' in model_cfg.MODEL.BACKBONE.TYPE) and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()

    if ('DATA_DIR' in model_cfg.MANO):
        model_cfg.defrost()
        mano_model_path = osp.join(cfg.human_model_path, 'mano')
        model_cfg.MANO.DATA_DIR = mano_model_path
        model_cfg.MANO.MODEL_PATH = mano_model_path
        model_cfg.MANO.MEAN_PARAMS = osp.join(cfg.wilor_root_path, 'mano_data', 'mano_mean_params.npz')
        model_cfg.freeze()

    model = WiLoRModel.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg


class WiLoR(nn.Module):
    def __init__(self):
        super(WiLoR, self).__init__()
        cwd = os.getcwd()
        try:
            os.chdir(cfg.wilor_root_path)
            self.model, _ = load_wilor_from_repo(checkpoint_path='./pretrained_models/wilor_final.ckpt', cfg_path='./pretrained_models/model_config.yaml')
        finally:
            os.chdir(cwd)
        self.rgb_mean = (0.485, 0.456, 0.406)
        self.rgb_std = (0.229, 0.224, 0.225)
    
    def forward(self, rhand_img, lhand_img, rhand_bbox, lhand_bbox):
        batch_size = rhand_img.shape[0]
        
        # combine right and left hand images after flipping the left hand images
        img = torch.cat((rhand_img, torch.flip(lhand_img,[3]))) # right hand + flipped left hnad images
        bbox = torch.cat((rhand_bbox, lhand_bbox))
        is_lhand = torch.cat((torch.zeros((batch_size)), torch.ones((batch_size)))).float().cuda() == 1
        
        # forward to WiLoR
        img = img - torch.FloatTensor(self.rgb_mean).cuda().view(1,3,1,1)
        img = img / torch.FloatTensor(self.rgb_std).cuda().view(1,3,1,1)
        img = img.to(dtype=torch.float16) # use half-precision input for fast inference
        out = self.model({'img': img})
        
        # get camera translation (restore flipped left hands)
        pred_cam = out['pred_cam']
        pred_cam[is_lhand,1] = -pred_cam[is_lhand,1]
        box_center = (bbox[:,:2] + bbox[:,2:])/2.
        box_size = bbox[:,2] - bbox[:,0]
        img_size = torch.FloatTensor([cfg.input_body_shape[1],cfg.input_body_shape[0]]).view(1,2).cuda().repeat(batch_size*2,1)
        scaled_focal_length = 5000 / cfg.input_hand_shape[0] * max(cfg.input_body_shape)
        transl = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
        rhand_transl, lhand_transl = transl[is_lhand==0].float(), transl[is_lhand==1].float()
        
        # get MANO vertices (restore flipped left hands)
        vert_cam = out['pred_vertices'] 
        vert_cam[is_lhand,:,0] = -vert_cam[is_lhand,:,0]
        vert_cam = vert_cam + transl.view(-1,1,3)
        rhand_vert_cam, lhand_vert_cam = vert_cam[is_lhand==0].float(), vert_cam[is_lhand==1].float()
        rhand_kpt_cam = torch.bmm(torch.from_numpy(mano.kpt['regressor']).cuda()[None,:,:].repeat(batch_size,1,1), rhand_vert_cam).float()
        lhand_kpt_cam = torch.bmm(torch.from_numpy(mano.kpt['regressor']).cuda()[None,:,:].repeat(batch_size,1,1), lhand_vert_cam).float()
       
        # get MANO parameters (restore flipped left hands)
        mano_param = out['pred_mano_params']
        root_pose = matrix_to_axis_angle(mano_param['global_orient'])
        hand_pose = matrix_to_axis_angle(mano_param['hand_pose'])
        shape_param = mano_param['betas']
        root_pose[is_lhand,:,1:3] = -root_pose[is_lhand,:,1:3]
        hand_pose[is_lhand,:,1:3] = -hand_pose[is_lhand,:,1:3]
        rhand_root_pose, lhand_root_pose = root_pose[is_lhand==0].view(batch_size,3).float(), root_pose[is_lhand==1].view(batch_size,3).float()
        rhand_pose, lhand_pose = hand_pose[is_lhand==0].view(batch_size,-1).float(), hand_pose[is_lhand==1].view(batch_size,-1).float()
        rhand_shape_param, lhand_shape_param = shape_param[is_lhand==0].float(), shape_param[is_lhand==1].float()
        rhand_pose -= mano.layer['right'].pose_mean.float().cuda().view(1,-1)[:,3:]
        lhand_pose -= mano.layer['left'].pose_mean.float().cuda().view(1,-1)[:,3:]

        # get image feature (restore flipped left hands)
        img_feat = out['img_feat'].float()
        rhand_img_feat = img_feat[is_lhand==0]
        lhand_img_feat = torch.flip(img_feat[is_lhand==1], [3])

        return rhand_vert_cam, rhand_kpt_cam, rhand_root_pose, rhand_pose, rhand_shape_param, rhand_transl, rhand_img_feat, lhand_vert_cam, lhand_kpt_cam, lhand_root_pose, lhand_pose, lhand_shape_param, lhand_transl, lhand_img_feat
