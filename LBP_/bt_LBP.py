import os
import cv2
import glob
from tqdm import tqdm
import numpy as np 
from matplotlib import pyplot as plt 
      
def get_pixel(img, center, x, y): 
      
    new_value = 0
      
    try: 
        if img[x][y] >= center: 
            new_value = 1
              
    except: 
        pass
      
    return new_value 
   
def lbp_pixel(img, x, y): 
   
    center = img[x][y] 
   
    val_ar = [] 
      
    # top_left 
    val_ar.append(get_pixel(img, center, x-1, y-1)) 
      
    # top 
    val_ar.append(get_pixel(img, center, x-1, y)) 
      
    # top_right 
    val_ar.append(get_pixel(img, center, x-1, y + 1)) 
      
    # right 
    val_ar.append(get_pixel(img, center, x, y + 1)) 
      
    # bottom_right 
    val_ar.append(get_pixel(img, center, x + 1, y + 1)) 
      
    # bottom 
    val_ar.append(get_pixel(img, center, x + 1, y)) 
      
    # bottom_left 
    val_ar.append(get_pixel(img, center, x + 1, y-1)) 
      
    # left 
    val_ar.append(get_pixel(img, center, x, y-1)) 
       
    power_val = [1, 2, 4, 8, 16, 32, 64, 128] 
    val = 0
      
    for i in range(len(val_ar)): 
        val += val_ar[i] * power_val[i] 
          
    return val

def _for(height, width):
    for i in range(0, height): 
            for j in range(0, width): 
                img_lbp[i, j] = lbp_pixel(img_gray, i, j) 


    # cv2.imshow('img', img_bgr)
    # print(img_bgr.shape)
    # cv2.waitKey(0)

def _crop(img_bgr,):
    # cv2.imshow('img',img_bgr)
    # img_bgr = cv2.resize(img_bgr, (255,255))
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image_copy = np.copy(img_bgr)
    # gray_image = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(img_bgr, 1.1, 4)
    face_crop = []
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)
        # xd 
        face_crop.append(img_bgr[y:y+h, x:x+w])

    for img_bgr in face_crop:
        # img_bgr = face
        # cv2.imwrite('face.jpg',img_bgr)
        # cv2.imshow('face',face)
        # cv2.waitKey(0)
        # image = Image.fromarray(img_bgr, "RGB")
        # print(type(image))
        return img_bgr

inputFolder = 'Image/humman2'
outputFoler = 'img_lbp/humman2/'
i = 0
for img in glob.glob(inputFolder + "/*.jpg"):
    i += 1
    if i % 10000 != 0:
        img_bgr = cv2.imread(img) 
        img_bgr = _crop(img_bgr)
        # img_bgr = cv2.resize(img_bgr, (255,255))
        # cv2.imshow('face',img_bgr)
        # print(type(img_bgr))
        # print(img_bgr.shape)
        height, width, _= img_bgr.shape
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_lbp = np.zeros((height, width), np.uint8) 
        _for(height,width)
        # cv2.imshow('img', img_lbp)
        cv2.imwrite(outputFoler + str(i) + '.jpg', img_lbp)
        print('ok', +i)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows

# count = 0
# # src_fname, ext = os.path.splitext('img_lbp')
# for path, dirs, files in os.walk('img'):
#     for f in tqdm(glob.iglob(os.path.join('img', '*.jpg'))):
#         count += 1
#         print(count)
#         if count % 10000 != 0:
#             img_bgr = cv2.imread(f, 1) 
#             img_bgr = cv2.resize(img_bgr, (255,255))
#             height, width, _ = img_bgr.shape 
#             img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) 
#             img_lbp = np.zeros((height, width), np.uint8) 
#             _for(height,width)
#             # with open('img_lbp/' + str(count + 1) + '.jpg', 'wb') as f:
#                 # pickle.dump()

#             cv2.imwrite('img_lbp/' + str(count // 10000) + '.jpg', img_lbp)
            # img_lbp.save(os.path.join(outpath, os.path.basename(src_fname)+'.jpg'))

# plt.imshow(img_lbp, cmap ="gray")
# cv2.imwrite('./img_lbp/people_3.png', img_lbp)
# plt.show() 