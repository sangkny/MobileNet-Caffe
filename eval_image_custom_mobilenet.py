from __future__ import print_function
import argparse
import numpy as np
import caffe


def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluate pretrained mobilenet models')
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)
    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)
    parser.add_argument('--image', dest='image',
                        help='path to color image', type=str)

    args = parser.parse_args()
    return args, parser


global args, parser
args, parser = parse_args()


def eval():

    nh, nw = 64, 64
    #img_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
    img_mean = np.array([0, 0, 0], dtype=np.float32)

    caffe.set_mode_cpu()
    net = caffe.Net(args.proto, args.model, caffe.TEST)

    im = caffe.io.load_image(args.image)
    h, w, _ = im.shape
    input_str = [] # # 'blob1' or 'data'
    if 'data' in net.blobs:
        input_str = 'data'
    elif 'blob1' in net.blobs:
        input_str = 'blob1'
    else:
        print('txt input definition error \n')
        return

    if h < w:
        off = int((w - h) / 2)
        im = im[:, off:off + h]
    else:
        off = int((h - w) / 2)
        im = im[off:off + h, :]
    im = caffe.io.resize_image(im, [nh, nw])

    transformer = caffe.io.Transformer({input_str: net.blobs[input_str].data.shape})
    transformer.set_transpose(input_str, (2, 0, 1))  # row to col
    transformer.set_channel_swap(input_str, (2, 1, 0))  # RGB to BGR
   # Please uncomment the following 2 lines according to your network
   # transformer.set_raw_scale(input_str, 255)  # [0,1] to [0,255]
   # transformer.set_mean(input_str, img_mean)
   # transformer.set_input_scale(input_str, 0.017)

    net.blobs[input_str].reshape(1, 3, nh, nw)
    net.blobs[input_str].data[...] = transformer.preprocess(input_str, im)
    out = net.forward()
    if 'data' == input_str:
        prob = out['prob']
    else:
        prob = out['fc_blob1'] # customized definition for our output

    prob = np.squeeze(prob)
    idx = np.argsort(-prob)

    if 'data' == input_str:
        label_names = np.loadtxt('synset.txt', str, delimiter='\t')
        for i in range(5):
            label = idx[i]
            print('%.2f - %s' % (prob[label], label_names[label]))
        return
    elif 'blob1' == input_str:
        print(prob)
        maxidx = np.array(prob).argmax()
        print('{}'.format('non object' if maxidx == 0 else 'object'))
        return
    else:
        return


if __name__ == '__main__':
    eval()
