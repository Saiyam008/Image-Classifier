from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from time import time
import matplotlib.pyplot as plt
import pandas as pd


from train_functions import get_input_args
from train_functions import load_pretrained_model
from train_functions import load_classifier
from train_functions import save_checkpoint

def main():
    in_arg = get_input_args()


    data_dir = in_arg.data_directory
    training_dir = data_dir + '/train'
    validation_dir = data_dir + '/valid'


    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(p=0.33),
                                              transforms.RandomVerticalFlip(p=0.33),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    training_dataset = datasets.ImageFolder(training_dir, transform = training_transforms)
    validation_dataset = datasets.ImageFolder(validation_dir, transform = validation_transforms)


    batch_size = 50
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size, shuffle = True)

    model = load_pretrained_model(in_arg.arch.lower())
    print('state_keys 1', model.state_dict().keys())

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = load_classifier(in_arg.arch.lower(), in_arg.dropout)
    print('state_keys 2', model.state_dict().keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    lr = in_arg.learning_rate
    epochs = in_arg.epochs

    criterion = nn.NLLLoss()


    optimizer = optim.Adam(model.classifier.parameters(), lr)


    model.to(device);


    start = time()

    training_losses, validation_losses, accuracy_log = [], [], []


    model.train()

    print('\nSTART TRAINING')


    for e in range(epochs):
        running_loss = 0
        step = 0
        start_epoch = time()

        for images, labels in training_loader:
            step += 1

            images, labels = images.to(device), labels.to(device)


            optimizer.zero_grad()


            log_ps = model(images)


            loss = criterion(log_ps, labels)


            loss.backward()
            optimizer.step()

            running_loss += loss.item()


            if step % 30 == 0:
                print('------------------------------------------------------------------')
                print('Epoch: {}/{}.. '.format(e+1, epochs),
                      'Step: {}/{}'.format(step, len(training_loader)),
                      'Training Loss: {:.3f}.. '.format(running_loss/len(training_loader)))
                print(f'Total time per step: {((time() - start_epoch) / step):.3f} seconds')
                print(f'Total time: {(time() - start):.3f} seconds')

        else:

            validation_loss = 0
            accuracy = 0


            model.eval()


            with torch.no_grad():
                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)


                    log_ps = model(images)


                    validation_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


                model.train()

                training_losses.append(running_loss / len(training_loader))
                validation_losses.append(validation_loss.item() / len(validation_loader))
                accuracy_log.append(accuracy / len(validation_loader) * 100)


                print('===================================================================================')
                print('Epoch: {}/{}.. '.format(e+1, epochs),
                      'Training Loss: {:.3f}.. '.format(running_loss / len(training_loader)),
                      'Validation Loss: {:.3f}.. '.format(validation_loss.item() / len(validation_loader)),
                      'Validation Accuracy: {:.1f}%'.format(accuracy / len(validation_loader) * 100))

                start_epoch = time()


    print(f'Total time per epoch: {((time() - start) / epochs):.3f} seconds')
    print(f'Total training time: {(time() - start):.3f} seconds\n')

    model.to('cpu')
    save_checkpoint(in_arg.arch.lower(), in_arg.dropout, training_dataset.class_to_idx,
                    model.state_dict())


    epoch_labels = ['epoch_{}'.format(x) for x in range(1,epochs+1)]
    accuracy_log = ('{:.1f}%'.format(x) for x in accuracy_log)

    summary_dict = {'Training loss': training_losses,
               'Validation loss': validation_loss.item(),
               'Validation Accuracy': accuracy_log}
    summary = pd.DataFrame(summary_dict, index=epoch_labels)
    print(summary)

    print('\nEND')


if __name__ == "__main__":
    main()