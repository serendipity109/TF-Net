import cv2
import os
from tqdm import tqdm


def frameExt(inp, length=100):
    print('\n', '------extracting frames------')
    source = inp.split('/')[-1].split('.')[0]
    out = f'./frames/{source}'
    def capture(inp, out):
        cap = cv2.VideoCapture(inp)
        frameId = 0
        for i in tqdm(range(length)):
            frameId = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if not ret:
                break
            filename = os.path.join(out, inp.split('/')[-1].split('.mp4')[0] + '_' + str(int(frameId + 1)) + ".jpg")
            cv2.imwrite(filename, frame) 
    capture(inp, out)
