from retinaface.detector import RetinafaceDetector
from utils import align_face
from skimage.io import imread, imsave
from tqdm import tqdm
import torch
from natsort import natsorted
import glob


def extract(pth):
    img = imread(pth)
    detector = RetinafaceDetector(net='mnet', type = 'cuda' if torch.cuda.is_available() else 'cpu') 
    bboxes, landmarks = detector.detect_faces(img)
    if len(landmarks) > 0:
        for i in range(len(landmarks)):
            landms = landmarks[i]
            out_img = align_face(img, [landms])
            # plt.imshow(out_img)
            # plt.show()
            imsave(pth.replace('frames', 'faces'), out_img, check_contrast=False)

def faceExt(folder):
    print('\n', '------extracting faces------')
    imgs = glob.glob(folder + '/*.jpg')
    imgs = natsorted(imgs)
    for img in tqdm(imgs):
        extract(img)
