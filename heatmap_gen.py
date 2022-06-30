#!/usr/bin/env python3
import torch
import numpy as np
import os
from PIL import Image
import h5py
import matplotlib.pyplot as plt
import shutil
import argparse

import utils.utils as utils
from models.model_att_mil import Att_MIL
from Topk_tiles import screen_coords,sample_rois

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


def initiate_model(ckpt_path):
    print('Init Model')
    model_dict = {"dropout": True, 'n_classes': 2}

    # if args.model_size is not None and args.model_type == 'amil':
    model_dict.update({"size_arg": 'small'})
    model = Att_MIL(**model_dict)

    utils.Print_network(model)

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


def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
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

def top_k(scores, k, invert=False):
    if invert:
        top_k_ids = scores.argsort()[:k]
    else:
        top_k_ids = scores.argsort()[::-1][:k]
    return top_k_ids

def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average') / len(scores) * 100
    return scores


parser = argparse.ArgumentParser(description='Generate heatmaps')
parser.add_argument('--ckpt_path', type=str, help='checkpoint directory')
parser.add_argument('--slide_name', type=str, help='Slide using to generate heatmaps')
parser.add_argument('--label', type=str, default="Hypermutated",help='Slide label')
parser.add_argument('--root_dir', type=str, default=None, help='directory saving patches')
parser.add_argument('--h5_path', type=str, default=None,help='directory saving h5 files')
parser.add_argument('--save_dir', type=str, default="heatmaps",help='directory saves block files')
parser.add_argument('--seed', type=int, default=1,help='random seed for reproducible experiment (default: 1)')

args = parser.parse_args()


if __name__ == '__main__':

    model = initiate_model(args.ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Done!')

    label_dict = {'Hypermutated': 0, 'Non-hypermutated': 1}
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))}


    slide_name = args.slide_name
    label = args.label
    h5_path = os.path.join(args.h5_path,slide_name+'.h5')
    if not os.path.exists(args.save_dir):
        os.makedirs((args.save_dir))

    top_left = None
    bot_right = None

    file = h5py.File(h5_path, "r")
    features = torch.tensor(file['features'][:])
    coords = file['coords'][:]
    file.close()
    Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, k=2)
    del features

    asset_dict = {'attention_scores': A, 'coords': coords}
    block_map_save_path = os.path.join(args.save_dir, slide_name, '{}_blockmap.h5'.format(slide_name))
    if not os.path.exists(os.path.join(args.save_dir, slide_name)):
        os.makedirs(os.path.join(args.save_dir, slide_name))

    block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')

    file = h5py.File(block_map_save_path, 'r')
    dset = file['attention_scores']
    coord_dset = file['coords']
    scores = dset[:]
    coords = coord_dset[:]
    file.close()


    scores = to_percentiles(scores)
    scores /= 100

    threshold = 0.0
    patch_size = 1010

    region_size = (coords[:, 0].max(), coords[:, 1].max())

    overlay = np.full(np.flip(region_size), 0).astype(float)
    counter = np.full(np.flip(region_size), 0).astype(np.uint16)

    for idx in range(len(coords)):
        score = scores[idx]
        coord = coords[idx]
        # accumulate attention
        overlay[coord[1]:coord[1] + patch_size, coord[0]:coord[0] + patch_size] += score
        # accumulate counter
        counter[coord[1]:coord[1] + patch_size, coord[0]:coord[0] + patch_size] += 1

    img = np.array(Image.new(size=region_size, mode="RGB", color=(255, 255, 255)))

    cmap = 'coolwarm'
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    for idx in range(len(coords)):
        score = scores[idx]
        coord = coords[idx]

        # attention block
        raw_block = overlay[coord[1]:coord[1] + patch_size, coord[0]:coord[0] + patch_size]

        # image block (either blank canvas or orig image)
        img_block = img[coord[1]:coord[1] + patch_size, coord[0]:coord[0] + patch_size].copy()

        # color block (cmap applied to attention block)
        color_block = (cmap(raw_block) * 255)[:, :, :3].astype(np.uint8)

        img_block = color_block

        # rewrite image block
        img[coord[1]:coord[1] + patch_size, coord[0]:coord[0] + patch_size] = img_block.copy()

    del overlay
    img = Image.fromarray(img)
    w, h = img.size
    custom_downsample = 4
    img2 = img.resize((int(w / custom_downsample), int(h / custom_downsample)))
    savepath = os.path.join(args.save_dir, slide_name, '{}_blockmap.jpeg'.format(slide_name))
    img2.save(savepath)


    k_sample = 10
    topk_sample_results = sample_rois(scores, coords, k_sample, invert=False)
    lastk_sample_results = sample_rois(scores, coords, k_sample, invert=True)


    for idx, (s_coord, s_score) in enumerate(
            zip(topk_sample_results['sampled_coords'], topk_sample_results['sampled_scores'])):
        print('coord: {} score: {:.3f}'.format(s_coord, s_score))
        patch_path = os.path.join(args.root_dir,slide_name)
        patch_name = os.path.join(patch_path, slide_name + '_({},{}).jpg'.format(s_coord[0], s_coord[1]))
        dest = os.path.join(args.save_dir, "topk")
        if not os.path.exists(dest):
            os.makedirs((dest))
        if os.path.exists(patch_name):
            shutil.copy2(patch_name, dest)

    for idx, (s_coord, s_score) in enumerate(
            zip(lastk_sample_results['sampled_coords'], lastk_sample_results['sampled_scores'])):
        print('coord: {} score: {:.3f}'.format(s_coord, s_score))
        patch_path = os.path.join(args.root_dir,slide_name)
        patch_name = os.path.join(patch_path, slide_name + '_({},{}).jpg'.format(s_coord[0], s_coord[1]))
        dest = os.path.join(args.save_dir, "lastk")
        if not os.path.exists(dest):
            os.makedirs((dest))
        if os.path.exists(patch_name):
            shutil.copy2(patch_name, dest)