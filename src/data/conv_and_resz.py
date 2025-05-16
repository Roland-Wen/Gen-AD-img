'''
0: road
1: car
2: traffic light
3: person
4: crahsed car
255: others
'''
CS_TO_ACC = {
    0:0,   # road
    1:0,   # sidewalk → treat as road
    2:255, # building ignore
    3:255, # wall ignore
    4:255, # fence ignore
    5:255, # pole ignore
    6:2,   # traffic light
    7:2,   # traffic sign → traffic light
    8:255, # vegetation ignore
    9:255, # terrain ignore
    10:255,# sky ignore
    11:3,  # person
    12:255,# rider ignore
    13:1,  # car     → car (undamaged)
    14:1,  # truck   → car
    15:1,  # bus     → car
    16:1,  # train   → car
    17:1,  # motorcycle → car
    18:1   # bicycle → car
}
DAMAGED_ID = 4  # keep free for accidents later
CS_ROOT = 'data/raw/leftImg8bit_trainvaltest'

import cv2, os, glob, numpy as np
from tqdm import tqdm
src_imgs = glob.glob(f"{CS_ROOT}/leftImg8bit/train/*/*_leftImg8bit.png")
for im_path in tqdm(src_imgs):
    mask_path = im_path.replace("_leftImg8bit.png","_gtFine_labelIds.png").replace("leftImg8bit","gtFine")
    
    assert os.path.exists(mask_path), f"Error: mask_path {mask_path} DNE\nim_path={im_path}"
        
    im  = cv2.imread(im_path)
    msk = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    msk = np.vectorize(lambda x: CS_TO_ACC.get(int(x), 255))(msk) # remap, ignore(255) if DNE
    im  = cv2.resize(im, (512,512), interpolation=cv2.INTER_CUBIC)
    msk = cv2.resize(msk,(512,512), interpolation=cv2.INTER_NEAREST)
    stem = os.path.basename(im_path).replace("_leftImg8bit.png",".png")
    cv2.imwrite(f"data/processed/controlnet_train/images/{stem}", im)
    cv2.imwrite(f"data/processed/controlnet_train/conditioning_image/{stem}", msk)
