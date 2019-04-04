import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=int)
parser.add_argument("-i", "--index", type=int)
args = parser.parse_args()

gpu = args.gpu
index = args.index

os.system('CUDA_VISIBLE_DEVICES=%d python grid_search.py -i %d'%(gpu, index))