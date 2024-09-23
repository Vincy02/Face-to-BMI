import argparse
from model import BiSeNet
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import time
import tqdm
import warnings

warnings.filterwarnings("ignore")

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='output/img.jpg'):
	im = np.array(im)
	vis_im = im.copy().astype(np.uint8)
	vis_parsing_anno = parsing_anno.copy().astype(np.uint8)

	canvas = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1]), dtype=np.uint8)
	index = np.where(vis_parsing_anno == 1)
	canvas[index[0], index[1]] = 255
	name = save_path
	cv2.imwrite(name, canvas)

def evaluate(respth='res/test_res', dspth='data', cp='model_final_diss.pth'):

	if not os.path.exists(respth):
		os.makedirs(respth)

	n_classes = 19
	net = BiSeNet(n_classes=n_classes)
	save_pth = osp.join('res/cp', cp)

	if CUDA_SUPPORT:
		net.cuda()
		net.load_state_dict(torch.load(save_pth))
	else:
		device = torch.device('cpu')
		net.load_state_dict(torch.load(save_pth, map_location=device))

	net.eval()

	to_tensor = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])
	with torch.no_grad():
		with tqdm.tqdm(total=len(os.listdir(dspth)), desc="Analisi immagini") as pbar:
			for image_path in os.listdir(dspth):
				img = Image.open(osp.join(dspth, image_path))
				image = img.resize((512, 512), Image.BILINEAR)
				img = to_tensor(image)
				img = torch.unsqueeze(img, 0)
				if CUDA_SUPPORT:
					img = img.cuda()
				out = net(img)[0]
				parsing = out.squeeze(0).cpu().numpy().argmax(0)
				vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))
				pbar.update(1)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Face Parcing')
	parser.add_argument("--image", default="../data/new_img", help="directory where the image file exists")
	parser.add_argument("--model", default="79999_iter.pth", help="faceparcing model")
	parser.add_argument("--output", default="../data/parsed_img", help="where the img will be saved")
	args = parser.parse_args()    

	CUDA_SUPPORT = torch.cuda.is_available()
	evaluate(respth=args.output, dspth= args.image, cp=args.model)