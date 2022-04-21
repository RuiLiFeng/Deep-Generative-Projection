import tensorflow as tf
import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import l2
from keras.preprocessing.image import *
from perceptual_model import PerceptualModel
import dnnlib.tflib as tflib
import cv2
from matplotlib import pyplot as plt
import numpy as np 
from tqdm import tqdm

def projection(args, sess, cloth_dir, parse_residual_np, step = 300):

    # rough alignment input
    rough_alignment_path = cloth_dir + f'coarse_align.png'
    rough_alignment = cv2.cvtColor(cv2.resize(cv2.imread(rough_alignment_path), (512,512), interpolation = cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)

    # set optimization params
    set_projection_loss(args)
    optimizer_w = tf.train.AdamOptimizer(learning_rate=0.01)
    args.w_train_op = optimizer_w.minimize(args.projection_loss, var_list=[args.wp])

    # get encoder result
    tflib.init_uninitialized_vars()

    args.x_np = (np.transpose(rough_alignment[np.newaxis,],[0,3,1,2]) / 127.5 - 1).astype('float32')
    latent_code = sess.run(args.latent_code_placeholder,{args.rough_align_placeholder:args.x_np})

    setter_w = tf.assign(args.wp,latent_code)
    setter_n = [tf.assign(var, npvar) for var, npvar in zip(args.noise_vars, args.noise_np)]

    _ = sess.run([setter_w] + setter_n, {args.x: args.x_np})
    encoder_result = sess.run(args.generate_img_placeholder,{args.rough_align_placeholder:args.x_np})

    # plt.imshow(0.5 + 0.5 * encoder_result[0].transpose([1,2,0]))
    # plt.show()
    
    
    # Projection(2/2)
    projection_list = [ ]
    for step in tqdm(range(1, step + 1), leave=False):
        sess.run(args.w_train_op, {args.x: args.x_np, args.parse_residual:parse_residual_np})
        outputs_projection = sess.run([args.wp, args.x_rec])
        projection_list.append(outputs_projection.copy())

    # plt.imshow(outputs_projection[1][0].transpose([1,2,0])/2+0.5)
    # plt.show()
    
    return rough_alignment, encoder_result, outputs_projection, projection_list
    

def semantic_search(args, sess, cloth_dir, model_dir, outputs_projection, mask_inner, mask_arm, step = 300):
    set_semantic_and_pattern_loss(args, cloth_dir, model_dir, outputs_projection, mask_inner, mask_arm)
    # Define optimizations for semantic search
    optimizer_mask_w = tf.train.AdamOptimizer(learning_rate=0.01)
    mask_train_op_w = optimizer_mask_w.minimize(args.loss_mask, var_list=[args.wp])
    tflib.init_uninitialized_vars()
    setter_w = tf.assign(args.wp, outputs_projection[0])
    _ = sess.run([setter_w], {args.x: args.x_np})

    # Semantic search
    semantic_search_list = [ ]
    for step in tqdm(range(1, step + 1), leave=False):
        sess.run(mask_train_op_w, {args.x: args.x_np})
        outputs_semantic_search = sess.run([args.wp, args.x_rec])
        semantic_search_list.append(outputs_semantic_search.copy())

    # plt.imshow(outputs_semantic_search[1][0].transpose([1,2,0])/2+0.5)
    # plt.show()

    return outputs_semantic_search, semantic_search_list


def pattern_search(args, sess, cloth_dir, model_dir, outputs_projection, mask_inner, mask_arm, step = 300):
    set_semantic_and_pattern_loss(args, cloth_dir, model_dir, outputs_projection, mask_inner, mask_arm)
    # Define optimizations for pattern search
    optimizer_mask_n = tf.train.AdamOptimizer(learning_rate=0.01)
    mask_train_op_n = optimizer_mask_n.minimize(args.loss_mask, var_list=args.noise_vars)
    tflib.init_uninitialized_vars()
    setter_n = [tf.assign(var, npvar) for var, npvar in zip(args.noise_vars, args.noise_np)]
    _ = sess.run(setter_n, {args.x: args.x_np})

    # Pattern search
    pattern_search_list = [ ]
    for step in tqdm(range(1, step + 1), leave=False):
        sess.run(mask_train_op_n, {args.x: args.x_np})
        outputs_pattern_search = sess.run([args.wp, args.x_rec])
        pattern_search_list.append(outputs_pattern_search.copy())

    # plt.imshow(outputs_pattern_search[1][0].transpose([1,2,0])/2+0.5)
    # plt.show()

    return outputs_pattern_search, pattern_search_list


def set_projection_loss(args):
    # Resize for clothing attribute model
    width = 331
    x_rec_width = tf.image.resize(args.x_rec_255[0], [width, width])
    x_rec_width = tf.expand_dims(x_rec_width, 0)

    x_width = tf.image.resize(args.x_255[0], [width, width])
    x_width = tf.expand_dims(x_width, 0)
    
    args.parse_residual = tf.placeholder(tf.float32, shape=[1, 1, 512, 512], name='parse_residual')
    parse_residual_nhwc = tf.transpose(args.parse_residual, [0, 2, 3, 1])
    
    # Attribute loss 
    base_model = keras.applications.resnet.ResNet101(weights=args.attr_model, input_shape=(width, width, 3), include_top=False, pooling='avg')
    input_tensor = Input((width, width, 3))
    xx = input_tensor
    xx = Lambda(nasnet.preprocess_input)(xx)
    xx = base_model(xx)
    attr_model = Model(input_tensor, xx)

    parse_residual_nhwc_attr = tf.image.resize(parse_residual_nhwc, [width, width])
    x_width_feat = attr_model(x_width * parse_residual_nhwc_attr)
    x_rec_width_feat = attr_model(x_rec_width * parse_residual_nhwc_attr)
    loss_attr = tf.reduce_sum(tf.square(x_width_feat - x_rec_width_feat))

    # Feature loss (VGG features)
    perceptual_model = PerceptualModel(args.percepture_model, 512, False)
    x_feat = perceptual_model(args.x_255 * parse_residual_nhwc)
    x_rec_feat = perceptual_model(args.x_rec_255 * parse_residual_nhwc)
    loss_feat = tf.reduce_mean(tf.square(x_feat - x_rec_feat), axis=[1])

    # Adv loss
    adv_score = args.D.get_output_for(args.x_rec, None)
    loss_adv = tf.reduce_mean(tf.nn.softplus(-adv_score), axis=1)

    # Pixel loss
    loss_pix = tf.reduce_mean(tf.square((args.x - args.x_rec)*args.parse_residual), axis=[1, 2, 3])

    args.projection_loss = (loss_pix + 5e-5 * loss_feat + 1 * loss_adv + loss_attr * 0.00005)
    

def set_semantic_and_pattern_loss(args, cloth_dir, model_dir, outputs_projection, mask_inner, mask_arm):
    
    # Define semantic loss and pattern loss
    loss_inter = tf.reduce_mean(tf.square(args.x - args.x_rec) * mask_inner, axis=[1, 2, 3])
    loss_arm = tf.reduce_mean(tf.square(outputs_projection[1] - args.x_rec) * mask_arm, axis=[1, 2, 3])
    loss_outer = tf.reduce_mean(tf.square(outputs_projection[1] - args.x_rec) * (1-mask_arm)*(1-mask_inner), axis=[1, 2, 3])
    adv_score = args.D.get_output_for(args.x_rec, None)
    loss_adv = tf.reduce_mean(tf.nn.softplus(-adv_score), axis=1)
    
    args.loss_mask = loss_inter + loss_adv + 0.9 * loss_outer + loss_arm
    