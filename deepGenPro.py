#!/usr/bin/env python
# coding: utf-8
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
import cv2
from matplotlib import pyplot as plt
from tensorflow.python.platform import gfile
from training import misc
import os
import warnings
warnings.filterwarnings('ignore')

import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import l2
from keras.preprocessing.image import *

from PIL import Image
from PIL import ImageFilter
import argparse
from tqdm import tqdm

from preprocess import get_and_save_residual_mask, get_arm_and_clothing_mask
from DGP import projection, semantic_search, pattern_search
from visualize import save_img, save_video

def main(args):

    # set base info
    cloth_dir = args.cloth_dir
    model_dir = args.model_dir
    cloth_sleeve = args.cloth_sleeve
    output_dir = args.output_dir
    network_pkl = args.network_pkl

    # get residual mask between model and clothing
    model_parse = Image.open(model_dir+f'model_parse.png')
    cloth_parse = Image.open(cloth_dir+f'cloth_parse.png')
    get_and_save_residual_mask(cloth_parse, cloth_sleeve, model_parse, cloth_dir)
    parse_res =cv2.resize(cv2.imread(cloth_dir+f'residual_parse.png',cv2.IMREAD_GRAYSCALE), (512,512), interpolation = cv2.INTER_CUBIC)
    parse_residual_np = (1-parse_res//255)[np.newaxis,np.newaxis,]

    # get mask arm and mask inner
    mask_arm, mask_inner = get_arm_and_clothing_mask(cloth_dir, model_dir)
    
    # Load Encoder and StyleGAN2 model
    tflib.init_tf()
    args.E, args.D, args.Gs = misc.load_pkl(network_pkl) # Encoder, Discriminator, Generator
    

    # setting variables
    args.rough_align_placeholder = tf.placeholder(shape=[1,3, 512,512],dtype=tf.float32) # Rough alignment
    args.latent_code_placeholder = args.E.get_output_for(args.rough_align_placeholder, is_training=False) # Latent code
    args.generate_img_placeholder = args.Gs.components.synthesis.get_output_for(args.latent_code_placeholder,randomize_noise=False) # Encoding result
    
    args.noise_vars = [var for name, var in args.Gs.components.synthesis.vars.items() if name.startswith('noise')]
    args.noise_np = tflib.run(args.noise_vars)

    args.wp = tf.get_variable(shape=[1,16,512],dtype=tf.float32,name='wp0')
    args.x_rec = args.Gs.components.synthesis.get_output_for(args.wp, randomize_noise=False)
    args.x_rec_255 = (tf.transpose(args.x_rec, [0, 2, 3, 1]) + 1) / 2 * 255

    args.x = tf.placeholder(tf.float32, shape=[1, 3, 512, 512], name='rough_alignment')
    args.x_255 = (tf.transpose(args.x, [0, 2, 3, 1]) + 1) / 2 * 255
    
    sess = tf.get_default_session()

    # Projection
    rough_alignment, encoder_result, outputs_projection, projection_list = projection(args, sess, cloth_dir, parse_residual_np, args.projection_steps)
    outputs_projection_np = np.uint8(255*(outputs_projection[1][0].transpose([1,2,0])/2+0.5).clip(0,1))
    encoder_result_np = np.uint8(255*(encoder_result[0].transpose([1,2,0])/2+0.5).clip(0,1))
    projection_list_np = [np.uint8(255*(x[1][0].transpose([1,2,0])/2+0.5).clip(0,1)) for x in projection_list]
    
    # Semantic search
    outputs_semantic_search, semantic_search_list = semantic_search(args, sess, cloth_dir, model_dir, outputs_projection, mask_inner, mask_arm, args.semantic_steps)
    outputs_semantic_search_np = np.uint8(255*(outputs_semantic_search[1][0].transpose([1,2,0])/2+0.5).clip(0,1))
    semantic_search_list_np = [np.uint8(255*(x[1][0].transpose([1,2,0])/2+0.5).clip(0,1)) for x in semantic_search_list]

    # Pattern search
    outputs_pattern_search, pattern_search_list = pattern_search(args, sess, cloth_dir, model_dir, outputs_projection, mask_inner, mask_arm, args.pattern_steps)
    outputs_pattern_search_np = np.uint8(255*(outputs_pattern_search[1][0].transpose([1,2,0])/2+0.5).clip(0,1))
    pattern_search_list_np = [np.uint8(255*(x[1][0].transpose([1,2,0])/2+0.5).clip(0,1)) for x in pattern_search_list]
    
    # Save results
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # save imgs
    save_img(rough_alignment, output_dir + "rough_alignment.png")
    save_img(encoder_result_np, output_dir + "outputs_encoder.png")
    save_img(outputs_projection_np, output_dir + "outputs_projection.png")
    save_img(outputs_semantic_search_np, output_dir + "outputs_semantic_search.png")
    save_img(outputs_pattern_search_np, output_dir + "outputs_pattern_search.png")

    # save videos
    save_video(output_dir, 'projection.mp4', rough_alignment, encoder_result_np, projection_list_np)
    save_video(output_dir, 'semantic_search.mp4', rough_alignment, outputs_projection_np, semantic_search_list_np)
    save_video(output_dir, 'pattern_search.mp4', rough_alignment, outputs_semantic_search_np, pattern_search_list_np)
    save_video(output_dir, 'DGP.mp4', rough_alignment, encoder_result_np, projection_list_np + semantic_search_list_np + pattern_search_list_np)

    # Image.fromarray(rough_alignment).save(output_dir + "rough_alignment.png")
    # Image.fromarray(np.uint8(255*(outputs_projection[1][0].transpose([1,2,0])/2+0.5).clip(0,1))).save(output_dir + "outputs_projection.png")
    # Image.fromarray(np.uint8(255*(outputs_semantic_search[1][0].transpose([1,2,0])/2+0.5).clip(0,1))).save(output_dir + "outputs_semantic_search.png")
    # Image.fromarray(np.uint8(255*(outputs_pattern_search[1][0].transpose([1,2,0])/2+0.5).clip(0,1))).save(output_dir + "outputs_pattern_search.png")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VirtualTryOn Demo')

    # data dir
    parser.add_argument('--cloth_dir', type=str, default='./data/model_1/cloth_1/',
                        dest='cloth_dir', help='Custom path to img dir')

    parser.add_argument('--model_dir', type=str, default='./data/model_1/model_info/',
                        dest='model_dir', help='Custom path to model dir')

    parser.add_argument('--cloth_sleeve', type=str, default='short',
                        dest='cloth_sleeve', help='Custom cloth sleeve length, short or long')

    parser.add_argument('--output_dir', type=str, default='./output/',
                        dest='output_dir', help='Custom path to output dir')

    # network dir
    parser.add_argument('--network_pkl', type=str, default='./pretrained_models/encoder-stylegan2-upper-body-512.pkl',
                    dest='network_pkl', help='encoder and stylegan network')
    
    parser.add_argument('--percepture_model', type=str, default='./pretrained_models/vgg.h5',
                dest='percepture_model', help='percepture model')

    parser.add_argument('--attr_model', type=str, default='./pretrained_models/attributes_model_resnet101.h5',
            dest='attr_model', help='attributes model')

    # DGP parameters
    parser.add_argument('--projection_steps', type=int, default=500, dest='projection_steps', help='projection steps')
    parser.add_argument('--semantic_steps', type=int, default=700, dest='semantic_steps', help='semantic search steps')
    parser.add_argument('--pattern_steps', type=int, default=700, dest='pattern_steps', help='pattern search steps')

    args = parser.parse_args()
    main(args)