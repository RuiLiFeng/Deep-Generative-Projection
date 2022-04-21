from PIL import ImageChops
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from PIL import ImageFilter

# Get residual parsing result
def get_and_save_residual_mask(cloth_parse, cloth_sleeve, model_parse, cloth_dir):
    if cloth_sleeve == 'short':
        model_body = model_parse.point(lambda i: i==64, mode='1')
    else:
        model_body = model_parse.point(lambda i: i!=0, mode='1')
    cloth_less = cloth_parse.point(lambda i: i==0 or i==23, mode='1')

    model_body = model_body.filter(ImageFilter.MaxFilter(7))
    cloth_less = cloth_less.filter(ImageFilter.MaxFilter(7))
    residual_mask = ImageChops.logical_and(model_body, cloth_less)

    fig,ax = plt.subplots(1,3, figsize=(15,15))
    ax[0].imshow(model_body)
    ax[1].imshow(cloth_less)
    ax[2].imshow(residual_mask)
    plt.show()

    residual_mask.save(cloth_dir+f'residual_parse.png')

    return


def get_arm_and_clothing_mask(cloth_dir, model_dir):
    parse_body =cv2.resize(cv2.imread(model_dir+'model_parse.png',cv2.IMREAD_GRAYSCALE), (512,512), interpolation = cv2.INTER_CUBIC)
    parse_body = ((parse_body==32) + (parse_body==16)).astype('uint8')
    kernel_body = np.ones((3,3),np.uint8) 
    kernel_body_d = np.ones((11,11),np.uint8) 
    parse_body_er = cv2.erode(parse_body,kernel_body,iterations = 1,  borderType=cv2.BORDER_CONSTANT, borderValue=0)
    parse_body_er = cv2.dilate(parse_body_er,kernel_body_d,iterations = 1,  borderType=cv2.BORDER_CONSTANT, borderValue=0)


    cloth_parse =cv2.resize(cv2.imread(cloth_dir + f'cloth_parse.png',cv2.IMREAD_GRAYSCALE), (512,512), interpolation = cv2.INTER_CUBIC)
    parse_er0 = ((cloth_parse==32)|(cloth_parse==64)|(cloth_parse==128)).astype('uint8')
    kernel = np.ones((5,5),np.uint8)
    parse_er0 = cv2.dilate(parse_er0,kernel,iterations = 1,  borderType=cv2.BORDER_CONSTANT, borderValue=0)
    kernel = np.ones((15,15),np.uint8)
    parse_er0 = cv2.erode(parse_er0,kernel,iterations = 1,  borderType=cv2.BORDER_CONSTANT, borderValue=0)

    mask_arm = parse_body_er & (cloth_parse==32) # Arm covering the clothing
    mask_arm[:256] = 0
    kernel = np.ones((5,5),np.uint8)
    mask_arm = cv2.dilate(mask_arm,kernel,iterations = 1,  borderType=cv2.BORDER_CONSTANT, borderValue=0)
    plt.imshow(mask_arm)
    plt.show()

    mask_inner = parse_er0 & (1-mask_arm) # Clothing covering the body except for the arms
    plt.imshow(mask_inner)
    plt.show()

    mask_arm = (mask_arm.astype('float32')[np.newaxis,np.newaxis])
    mask_inner = (mask_inner.astype('float32')[np.newaxis,np.newaxis])
    
    return mask_arm, mask_inner
