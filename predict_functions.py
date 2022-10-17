import argparse
from torchvision import models
import torch
from torch import nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. If the user fails to provide
    some or all of the arguments, then the default values are used for the
    missing arguments.
    This function returns these arguments as an ArgumentParser object.
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """

    parser = argparse.ArgumentParser(description='Retrieving inputs from user')


    parser.add_argument('image_path', type = str, default = './predict.jpg',
                help = 'image path to the image to predict (default: ./predict.jpg)')
    parser.add_argument('checkpoint_file', type = str, default = 'vgg16_model_checkpoint.pth',
                    help = 'Checkpoint file (default: vgg16_model_checkpoint.pth)')
    parser.add_argument('--topk', type = int, default = 5,
                    help = 'Top k most likely categories (default: 5)')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                    help = 'Categories to name file (default: cat_to_name.json)')
    parser.add_argument('--arch', type = str, default = 'VGG16',
                    help = 'CNN Model Architecture: vgg16, alexnet or densenet161 (default: VGG16)')

    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Torch tensor
    '''

    img = Image.open(image)


    img_width, img_height = img.size
    ratio = img_width / img_height
    if img_width < img_height:
        img = img.resize((256, int(256 / ratio)))
    else:
        img = img.resize((int(256 * ratio) , 256))


    center_x, center_y = img.size
    left = max(0, int(center_x - 224)/2)
    upper = max(0, int(center_y - 224)/2)

    img = img.crop((left, upper, left + 224, upper + 224))


    img = np.array(img) / 255


    mean_n = np.array([0.485, 0.456, 0.406])
    std_n = np.array([0.229, 0.224, 0.225])
    img_n = ((img - mean_n) / std_n)


    img_n = img_n.transpose(2,0,1)


    img_tch = torch.from_numpy(np.array([img_n])).float()

    return img_tch


def load_checkpoint_file(filepath, model_arch):
    ''' This function loads the checkpoint_file, loar a pre-trained CNN
     according to the input CNN architecture, creates a customized classifier,
     replace it in the pre-trained CNN model and finally loads the checkpoint
     in the model.
     It returns the rebuilt model
    '''

    checkpoint = torch.load(filepath)


    pretrained_model = checkpoint['pretrained_model']
    input_size = checkpoint['input_size']
    layer1_size = checkpoint['layer1_size']
    layer2_size = checkpoint['layer2_size']
    output_size = checkpoint['output_size']
    dropout = checkpoint['dropout']


    if model_arch == 'vgg16':
        model = models.vgg16(pretrained=True)

    elif model_arch == 'alexnet':
        model = models.alexnet(pretrained=True)

    elif model_arch == 'densenet161':
        model = models.densenet161(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
        print('Invalid model name input in --arch. Loaded VGG16 model instead')
        model_name = 'vgg16'

    print('Loaded {} pretrained model'.format(model_arch))

    # Freeze parameters to not backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(input_size, layer1_size),
                               nn.ReLU(),
                               nn.Dropout(p = dropout),
                               nn.Linear(layer1_size, layer2_size),
                               nn.ReLU(),
                               nn.Dropout(p = dropout),
                               nn.Linear(layer2_size, output_size),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier


    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def predict(image, model, topk):
    ''' Predict the likely probabilities and category of an image using a
    trained deep learning model.
    It returns the probability and category prediction
    '''

    model.eval()


    with torch.no_grad():
        output = model.forward(image)


    ps = torch.exp(output)


    ps_topk = torch.topk(ps, topk)[0].tolist()[0]
    index_topk = torch.topk(ps, topk)[1].tolist()[0]


    indices = []
    for i in range(len(model.class_to_idx)):
        indices.append(list(model.class_to_idx.items())[i][0])
    cat_topk = [indices[index] for index in index_topk]

    return ps_topk, cat_topk

def plotting(image_path, ps_topk, labels):
    ''' This function plots the image to predict and then a horizontal bar chart
    with the top k probabilites output by the prediction algorithm.
    '''
    plt.figure(figsize = [10, 8])


    image = Image.open(image_path)
    ax1 = plt.subplot(2, 1, 1)
    ax1.axis('off')
    ax1.imshow(image)


    labels.reverse()
    ps_topk.reverse()
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_title('Prediction')
    ax2.barh(labels, ps_topk);

    plt.show(block = True)

    return