from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from models.model_att_mil import Att_MIL
from models.resnet_custom import resnet50_baseline
import h5py
import shutil

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
	file = h5py.File(output_path, mode)
	for key, val in asset_dict.items():
		data_shape = val.shape
		if key not in file:
			data_type = val.dtype
			chunk_shape = (1, ) + data_shape[1:]
			maxshape = (None, ) + data_shape[1:]
			dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
			dset[:] = val
			if attr_dict is not None:
				if key in attr_dict.keys():
					for attr_key, attr_val in attr_dict[key].items():
						dset.attrs[attr_key] = attr_val
		else:
			dset = file[key]
			dset.resize(len(dset) + data_shape[0], axis=0)
			dset[-data_shape[0]:] = val
	file.close()
	return output_path


def initiate_model(ckpt_path):
    print('Init Model')
    model_dict = {"dropout": True, 'n_classes': 2}

    #if args.model_size is not None and args.model_type == 'amil':
    model_dict.update({"size_arg": 'small'})
    model = Att_MIL(**model_dict)

    Print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''): ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    model.relocate()
    model.eval()
    return model

def infer_single_slide(model, features, label, reverse_label_dict, k=1):
    features = features.to(device)
    with torch.no_grad():
        if isinstance(model, (Att_MIL)):
            model_results_dict = model(features)
            logits, Y_prob, Y_hat, A, _ = model(features)
            Y_hat = Y_hat.item()
            A = A.view(-1, 1).cpu().numpy()

        else:
            raise NotImplementedError

        print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label,
                                                    ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))

        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])

    return ids, preds_str, probs, A

def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100
    return scores

def screen_coords(scores, coords, top_left, bot_right):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords

def top_k(scores, k, invert=False):
    if invert:
        top_k_ids=scores.argsort()[:k]
    else:
        top_k_ids=scores.argsort()[::-1][:k]
    return top_k_ids

def sample_rois(scores, coords, k=5,top_left=None, bot_right=None, invert=False):

    if len(scores.shape) == 2:
        scores = scores.flatten()

    scores = to_percentiles(scores)
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)

    if invert:
        sampled_ids = top_k(scores, k, invert=True)
    else:
        sampled_ids = top_k(scores, k, invert=False)

    coords = coords[sampled_ids]
    scores = scores[sampled_ids]

    asset = {'sampled_coords': coords, 'sampled_scores': scores}
    return asset

def compu_k(slide_id):
    Ref = pd.read_csv('csv/WSI_282.csv')
    info = Ref[Ref['filename']==slide_id]
    cnt = int(info.patch_num)

    k_sample = 512

    return k_sample

parser = argparse.ArgumentParser(description='Select the top tiles of high attention for tumor vs normal network training')
parser.add_argument('--ckpt_path', type=str, default=None,
                    help='checkpoint directory')
parser.add_argument('--data_root', type=str, default=None,
                    help='Directory saves patches')
parser.add_argument('--sample_dir', type=str, default=None,
                    help='directory saves selected sample patches')
parser.add_argument('--h5_dir', type=str, default=None,
                    help='directory saves h5 files')
parser.add_argument('--pt_dir', type=str, default=None,
                    help='directory saves pt files')
parser.add_argument('--block_dir', type=str, default=None,
                    help='directory saves block files')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--csv', type=str, default=None,
                    help='csv file saving patient information')

args = parser.parse_args()



if __name__ == '__main__':

    print('\ninitializing model from checkpoint')
    ckpt_path = args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))

    model = initiate_model(ckpt_path)

    feature_extractor = resnet50_baseline(pretrained=True)
    feature_extractor.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Done!')

    label_dict={'Non-hypermutated': 0, 'Hypermutated': 1}
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))}

    if torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to('cuda:0')
    else:
        feature_extractor = feature_extractor.to(device)

    sample_dir = args.sample_dir
    if not os.path.exists(sample_dir):
        os.makedirs((sample_dir))

    patch_root = args.data_root

    features_dir = args.pt_dir
    h5_dir = args.h5_dir

    process_df = pd.read_csv(args.csv)

    for i in range(len(process_df)):
        slide_id = process_df.loc[i, 'slide_id']
        print('\nprocessing: ', slide_id)
        try:
            label = process_df.loc[i, 'label']
        except KeyError:
            label = 'Unspecified'


        features_path = os.path.join(args.pt_dir, slide_id + '.pt')
        h5_path = os.path.join(args.h5_dir, slide_id + '.h5')

        # load features
        features = torch.load(features_path)
        Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, k=2)
        predict_label = Y_hats[0]
        del features

        block_map_save_path = os.path.join(args.block_dir, '{}_blockmap.h5'.format(slide_id))

        if not os.path.isfile(block_map_save_path):
            file = h5py.File(h5_path, "r")
            coords = file['coords'][:]
            file.close()
            asset_dict = {'attention_scores': A, 'coords': coords}
            block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')

        file = h5py.File(block_map_save_path, 'r')
        dset = file['attention_scores']
        coord_dset = file['coords']
        scores = dset[:]
        coords = coord_dset[:]
        file.close()

        sample_save_dir = os.path.join(sample_dir, slide_id)
        os.makedirs(sample_save_dir, exist_ok=True)
        print('sampling {}'.format(slide_id))
        #k_sample = int(compu_k(slide_id))
        k_sample = 10
        topk_sample_results = sample_rois(scores, coords, k_sample, invert=False)
        lastk_sample_results = sample_rois(scores, coords, k_sample, invert=True)

        topk_summary = []
        lastk_summary = []

        for idx, (s_coord, s_score) in enumerate(
                zip(topk_sample_results['sampled_coords'], topk_sample_results['sampled_scores'])):
            print('coord: {} score: {:.3f}'.format(s_coord, s_score))
            patch_path = os.path.join(patch_root,slide_id)
            patch_name = os.path.join(patch_path,slide_id+'_({},{}).jpg'.format(s_coord[0], s_coord[1]))
            topk_summary.append([slide_id,label,predict_label,s_coord[0], s_coord[1],s_score])
            dest = os.path.join(sample_save_dir,"topk")
            if not os.path.exists(dest):
                os.makedirs((dest))
            if os.path.exists(patch_name):
                shutil.copy2(patch_name,dest)

        for idx, (s_coord, s_score) in enumerate(
                zip(lastk_sample_results['sampled_coords'], lastk_sample_results['sampled_scores'])):
            print('coord: {} score: {:.3f}'.format(s_coord, s_score))
            patch_path = os.path.join(patch_root,slide_id)
            patch_name = os.path.join(patch_path,slide_id+'_({},{}).jpg'.format(s_coord[0], s_coord[1]))
            lastk_summary.append([slide_id,label,predict_label,s_coord[0], s_coord[1],s_score])
            dest = os.path.join(sample_save_dir, "lastk")
            if not os.path.exists(dest):
                os.makedirs((dest))
            if os.path.exists(patch_name):
                shutil.copy2(patch_name, dest)


        topk_summary_df = pd.DataFrame(topk_summary, columns=[ "slide_id", "label", "predict_label", "X_coord", "Y_coord", "att score"])
        topk_summary_path = os.path.join(sample_save_dir,slide_id +'_summary_top.csv')
        topk_summary_df.to_csv(topk_summary_path, index=False)

        lastk_summary_df = pd.DataFrame(lastk_summary,columns=["slide_id", "label", "predict_label", "X_coord", "Y_coord", "att score"])
        lastk_summary_path = os.path.join(sample_save_dir, slide_id + '_summary_last.csv')
        lastk_summary_df.to_csv(lastk_summary_path, index=False)