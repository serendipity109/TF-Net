import warnings
warnings.filterwarnings('ignore')
from imageio import imread
import torch
from torchvision import transforms
import glob
from natsort import natsorted
from model import model, load_partial_pth


def tf(img_lst):
    transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])
    return torch.cat([torch.unsqueeze(transform(imread(img)), 0) for img in img_lst])

def detect(folder, type='frame'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_lst = glob.glob(folder + '/*.jpg')
    img_lst = natsorted(img_lst)
    img = tf(img_lst)

    print('------preparing model------')
    pth = 'video.pth' if type=='video' else 'frame.pth'
    md = model(template=True, dct = True, frame = type=='frame')
    md.load_state_dict(torch.load(pth))
    md = md.to(device); md.eval()
    SEED = 0
    torch.manual_seed(SEED)
    img = img.to(device)
    print('------start detecting------')
    with torch.no_grad():
        p, pmask = md(img)
    if type == 'frame':
        import matplotlib.pyplot as plt
        for i, img in enumerate(img_lst):
            prob = p[i].item()
            count = str(i)
            plt.figure()
            plt.imshow(imread(img))
            plt.text(3, 8, prob, color='cyan')
            plt.show()
            plt.axis('off')
            plt.savefig(f'output/{count}.jpg', bbox_inches='tight', pad_inches = 0)
        return "finished!"
    else:
        print('Probability:', p.item()) # Show the probability of the entire video being manipulated
        return p.item()