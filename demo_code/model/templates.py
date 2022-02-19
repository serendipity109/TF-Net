from imageio import imread
import torch
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_templates():
  templates_list = []
  for i in range(10):
    img = imread('./MCT/template{:d}.png'.format(i))
    templates_list.append(transforms.functional.to_tensor(img)[0:1,0:19,0:19])
  templates = torch.stack(templates_list)
  templates = templates.squeeze(1)
  templates = templates.to(device)
  return templates

