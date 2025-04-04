import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np

from modelAE_modi import IM_AE


import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]")
parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int, help="Iteration to train. Either epoch or iteration need to be zero [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00005, type=float, help="Learning rate for adam [0.00005]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--dataset", action="store", dest="dataset", default="dataset", help="The name of dataset") #default:all_vox256_img
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="./data/dataset", help="Root directory of dataset [data]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/", help="Directory name to save the image samples [samples]")
parser.add_argument("--sample_vox_size", action="store", dest="sample_vox_size", default=64, type=int, help="Voxel resolution for coarse-to-fine training [64]")
parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
parser.add_argument("--start", action="store", dest="start", default=0, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--end", action="store", dest="end", default=16, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--ae", action="store_true", dest="ae", default=False, help="True for ae [False]")
parser.add_argument("--svr", action="store_true", dest="svr", default=False, help="True for svr [False]")
parser.add_argument("--getz", action="store_true", dest="getz", default=False, help="True for getting latent codes [False]")
parser.add_argument("--get_mesh", action="store_true", dest="get_mesh", default=False, help="True for getting latent codes [False]")
parser.add_argument("--test_z", action="store_true", dest="test_z", default=False, help="True for getting latent codes [False]")
parser.add_argument("--novelty", action="store_true", dest="novelty", default=False, help="True for getting latent codes [False]")


FLAGS = parser.parse_args()



if not os.path.exists(FLAGS.sample_dir):
	os.makedirs(FLAGS.sample_dir)

if FLAGS.ae:
	im_ae = IM_AE(FLAGS)

	if FLAGS.train:
		im_ae.train(FLAGS)
	elif FLAGS.getz:
		im_ae.get_z(FLAGS)
	elif FLAGS.get_mesh:
		im_ae.test_z(FLAGS)
	elif FLAGS.novelty:
		im_ae.novelty_eval(FLAGS)
	else:
		im_ae.test_mesh(FLAGS)
		# im_ae.test_mesh_point(FLAGS)

else:
	print("Please specify an operation: ae or svr?")

if FLAGS.test_z:
	im_ae = IM_AE(FLAGS)
	im_ae.test_z(FLAGS)
