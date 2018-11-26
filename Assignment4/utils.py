import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt
import math
import pickle

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    # preprocess data for the first time.
    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)


    # load the preprocessed the data if the data has been processed before.
    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
    # seperate the whole data into different batches.
    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        # reshape the original data into the length self.num_batches * self.batch_size * self.seq_length for convenience.
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)

        #ydata is the xdata with one position shift.
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

def getNext_batch(input , data_y , batch_num, batch_size):
    return input[(batch_num)*batch_size : (batch_num  + 1)*batch_size] \
        , data_y[(batch_num)*batch_size : (batch_num + 1)*batch_size]

def load_face(data_dir=None):
    if data_dir == None:
        dataset_name = 'face_dataset'
        data_dir = os.path.join('./data',dataset_name)
    with open(os.path.join(data_dir,'image_python3.pkl'),'rb') as f:
        im = pickle.load(f)
    with open(os.path.join(data_dir,'label_sub_python3.pkl'),'rb') as f:
        label = pickle.load(f)
    im = im.astype(np.float)
    label = label.astype(np.float)
    return im / 255., label

def load_mnist(data_dir='./data/mnist'):
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd , dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28 , 28 ,  1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28 , 28 , 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    #convert label to one-hot

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, int(y[i])] = 1.0

    return X / 255. , y_vec


def get_image(image_path , is_grayscale = False):
    return np.array(inverse_transform(imread(image_path, is_grayscale)))


def save_images(images , size , image_path):
    return imsave(inverse_transform(images) , size , image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images , size , path):
    return scipy.misc.imsave(path , merge(images , size))

def merge(images , size):
    h , w = images.shape[1] , images.shape[2]
    img = np.zeros((h*size[0] , w*size[1] , 3))
    for idx in range(images.shape[0]):
        i = idx // size[1]
        j = idx % size[1]
        img[i*h:i*h+h, j*w:j*w+w, :] = images[idx]
    return img

def inverse_transform(image):
    return (image + 1.)/2.

def read_image_list(category):
    filenames = []
    print("list file")
    list = os.listdir(category)

    for file in list:
        filenames.append(category + "/" + file)

    print("list file ending!")

    return filenames

##from caffe
def vis_square(visu_path , data , type):
    """Take an array of shape (n, height, width) or (n, height, width , 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))

    padding = (((0, n ** 2 - data.shape[0]) ,
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data , padding, mode='constant' , constant_values=1)  # pad with ones (white)

    # tilethe filters into an im age
    data = data.reshape((n , n) + data.shape[1:]).transpose((0 , 2 , 1 , 3) + tuple(range(4 , data.ndim + 1)))

    data = data.reshape((n * data.shape[1] , n * data.shape[3]) + data.shape[4:])

    plt.imshow(data[:,:,0])
    plt.axis('off')

    if type:
        plt.savefig('./{}/weights.png'.format(visu_path) , format='png')
    else:
        plt.savefig('./{}/activation.png'.format(visu_path) , format='png')


def sample_label_pre():
    num = 64
    label_vector = np.zeros((num , 10), dtype=np.float)
    for i in range(0 , num):
        label_vector[i , i/8] = 1.0
    return label_vector

def sample_label(num):
    label_vector = np.zeros((num,10), dtype=np.float)
    for i in range(0, num):
        label_vector[i, i % 10] = 1.0
    return label_vector

def sample_label_face(num):
    label_vector = np.zeros((num,3))
    for i in range(64):
        if i % 2 == 0:
            label_vector[i,0]=2
        else:
            label_vector[i,0]=-2
        if (i/2) % 2 == 0:
            label_vector[i,1]=2
        else:
            label_vector[i,1]=-2
        if (i/4) % 2 == 0:
            label_vector[i,2]=2
        else:
            label_vector[i,2]=-2
    return label_vector
