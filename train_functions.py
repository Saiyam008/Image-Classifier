import argparse
from torchvision import models
from torch import nn
import torch

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

    
    parser.add_argument('data_directory', type = str, default = './',
                    help = 'path to the data directory (default: ./)')
    parser.add_argument('--save_dir', type = str, default = './',
                    help = 'path to the folder to save checkpoint file (default: ./)')
    parser.add_argument('--arch', type = str, default = 'VGG16',
                    help = 'CNN Model Architecture: vgg16, alexnet or densenet161 (default: VGG16)')
    parser.add_argument('--learning_rate', type = float, default = 0.002,
                    help = 'Learning rate (default: 0.002)')
    parser.add_argument('--epochs', type = int, default = 1,
                    help = 'Epochs (default: 1)')
    parser.add_argument('--dropout', type = float, default = 0.1,
                    help = 'Dropout (default: 0.1)')

    return parser.parse_args()

def load_pretrained_model(model_arch):
    ''' This function load the CNN pretrained model accordint to the choosen
    architecture chosen by the user with  --arch argument when lauching
    the code.
    In case the user fails to select a valid architecture, the function loads
    the VGG-16 model.
    It returns the model itself.
    '''
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

    return model

def load_classifier(model_arch, dpout):
    ''' This function creates and returns a classifier matching with the required
    parameters of the CNN architecture choosen by the user with the --arch
    argument when lauching the code.
    '''
    if model_arch == 'vgg16':
        classifier = nn.Sequential(nn.Linear(25088, 4096),
                                   nn.ReLU(),
                                   nn.Dropout(p = dpout),
                                   nn.Linear(4096, 512),
                                   nn.ReLU(),
                                   nn.Dropout(p = dpout),
                                   nn.Linear(512, 102),
                                   nn.LogSoftmax(dim=1))

    elif model_arch == 'alexnet':
        classifier = nn.Sequential(nn.Linear(9216, 1024),
                                   nn.ReLU(),
                                   nn.Dropout(p = dpout),
                                   nn.Linear(1024, 512),
                                   nn.ReLU(),
                                   nn.Dropout(p = dpout),
                                   nn.Linear(512, 102),
                                   nn.LogSoftmax(dim=1))

    elif model_arch == 'densenet161':
        classifier = nn.Sequential(nn.Linear(2208, 512),
                                   nn.ReLU(),
                                   nn.Dropout(p = dpout),
                                   nn.Linear(512, 256),
                                   nn.ReLU(),
                                   nn.Dropout(p = dpout),
                                   nn.Linear(256, 102),
                                   nn.LogSoftmax(dim=1))
    else:
        classifier = nn.Sequential(nn.Linear(25088, 4096),
                                   nn.ReLU(),
                                   nn.Dropout(p = dpout),
                                   nn.Linear(4096, 512),
                                   nn.ReLU(),
                                   nn.Dropout(p = dpout),
                                   nn.Linear(512, 102),
                                   nn.LogSoftmax(dim=1))
        model_arch = "VGG16"

    print('Loaded new classifier for {} pretrained model'.format(model_arch))

    return classifier

def save_checkpoint(model_arch, dropout, model_class_to_idx, model_state_dict):
    ''' This function save the checkpoint and stores the paramenter matching
    with the CNN architecture choosen by the user with the --arch argument when
    lauching the code.
    '''
    if model_arch == 'vgg16':
        checkpoint = {'input_size': 25088,
                      'layer1_size': 4096,
                      'layer2_size': 512,
                      'output_size': 102}
    elif model_arch == 'alexnet':
        checkpoint = {'input_size': 9216,
                      'layer1_size': 1024,
                      'layer2_size': 512,
                      'output_size': 102}
    elif model_arch == 'densenet161':
        checkpoint = {'input_size': 2208,
                      'layer1_size': 512,
                      'layer2_size': 256,
                      'output_size': 102}
    else:
        checkpoint = {'input_size': 25088,
                      'layer1_size': 4096,
                      'layer2_size': 512,
                      'output_size': 102}

    checkpoint['pretrained_model'] = model_arch
    checkpoint['dropout'] = dropout
    checkpoint['class_to_idx'] = model_class_to_idx
    checkpoint['state_dict'] = model_state_dict

    torch.save(checkpoint, '{}_model_checkpoint.pth'.format(model_arch))

    print('Checkpoint file saved as: {}_model_checkpoint.pth'.format(model_arch))

    return