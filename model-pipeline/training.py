import math
import pickle
import time

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as Data

from models.MediumPoseNet import *
from dataloader import *

VERBOSE = True


def visualiser(y_test, y_pred, epoch, name):
    keypoint_columns = [
        'LAnkle',
        'LElbow',
        'LHip',
        'LKnee',
        'LShoulder',
        # ['keypoint_0_LSmallToe_x', 'keypoint_0_LSmallToe_y'],
        'LWrist',
        # ['keypoint_0_MidHip_x', 'keypoint_0_MidHip_y'],
        # ['keypoint_0_Neck_x', 'keypoint_0_Neck_y'],
        'Nose',
        'RAnkle',
        # ['keypoint_0_RBigToe_x','keypoint_0_RBigToe_y'],
        # ['keypoint_0_REar_x','keypoint_0_REar_y'],
        'RElbow',
        # ['keypoint_0_REye_x','keypoint_0_REye_y'],
        # ['keypoint_0_RHeel_x','keypoint_0_RHeel_y'],
        'RHip',
        'RKnee',
        'RShoulder',
        # ['keypoint_0_RSmallToe_x', 'keypoint_0_RSmallToe_y'],
        'RWrist'
    ]

    num_keypoints = y_test.shape[1]

    y_pred = np.array(y_pred.cpu())
    y_test = np.array(y_test.cpu())
    y_pred = y_pred[0]
    y_test = y_test[0]

    # for i in np.random.randint(y_test.shape[0], size=(2,)):

    fig = plt.figure(figsize=(20, 3))
    count = 1

    label = []
    diff = []

    for j, keypoint in enumerate(keypoint_columns):
        ax = fig.add_subplot(3, num_keypoints, count)
        ax.set_title(keypoint)

        img = y_test[j, :, :]
        img = np.transpose(img)
        label.append(img)
        ax.imshow(img)
        ax.axis("auto")
        if j == 0:
            ax.set_ylabel("true")

        count += 1

    for j, keypoint in enumerate(keypoint_columns):
        ax = fig.add_subplot(3, num_keypoints, count)

        img = y_pred[j, :, :]
        img = np.transpose(img)
        diff.append(img - label[j])

        ax.imshow(img)
        ax.axis("auto")
        if j == 0:
            ax.set_ylabel("prediciton")

        count += 1

    for j, keypoint in enumerate(keypoint_columns):
        ax = fig.add_subplot(3, num_keypoints, count)
        ax.imshow(diff[j])
        ax.axis("auto")
        if j == 0:
            ax.set_ylabel("difference")

        count += 1

    plt.savefig('./output/epoch' + str(epoch) + '_' + name + '.png')


def get_mean_data(ds):
    params_t = {'batch_size': 1000,
                'shuffle': True
                }
    training_gen = Data.DataLoader(ds, **params_t)
    mean = 0.0
    std = 0.0
    for img in training_gen:
        img = img[0]
        batch_sample = img.size(0)
        img = img.view(batch_sample, img.size(1), -1)
        mean += img.mean(2).sum(0)
        std += img.std(2).sum(0)
    mean /= len(training_gen.dataset)
    std /= len(training_gen.dataset)
    return mean, std


# @profile
def train(model, device, train_loader, optimizer, epoch):
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    model.train()
    crit = nn.MSELoss()
    losses = list()
    itemized_loss = list()
    count = 0
    for batch_idx, (x, label, keypoints) in enumerate(train_loader):
        x, label = x.to(device), label.to(device)
        if math.isnan(label.mean()):
            count += 1
            continue
        optimizer.zero_grad()
        output = model(x)
        loss = crit(output, keypoints)

        loss.backward()
        optimizer.step()

        if epoch == 1 and batch_idx == 100:
            path = '/home/roboy/Projects/RoboyMedium/src/git/dondata/data/'
            outfile_label = path + 'label'
            outfile_output = path + 'output'
            np_label = label.cpu().numpy()
            np_output = output.cpu().detach().numpy()
            print(type(np_output))
            np.save(outfile_label, np_label)
            np.save(outfile_output, np_output)

        if batch_idx % 50 == 0:
            min_max_mean = list()
            for i in range(output.size(0)):
                w = crit(output[i, :, :, :], label[i, :, :, :])
                min_max_mean.append(w.item())
            print('Max Loss: {}, Min Loss: {}'.format(max(min_max_mean), min(min_max_mean)))
            print('Variance {}'.format(np.var(min_max_mean)))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            itemized_loss.append(loss.item())
            losses.append(loss)

            print(count)

    return losses, itemized_loss


def val(model, device, test_loader, epoch, recording_time):
    print('Starting Validation')
    device = torch.device(device)
    model.to(device)
    model.eval()
    torch.save(model.state_dict(), './models/model' + recording_time + '_epoch_' + str(epoch) + '.pt')
    test_loss = 0
    with torch.no_grad():
        for indx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            if epoch == 10 and indx % 100 == 0:
                visualiser(y_test=label, y_pred=output, epoch=epoch, name=str(indx))

            test_loss += F.mse_loss(output, label)  # sum up batch loss

    if epoch % 1 == 0:
        visualiser(y_test=label, y_pred=output, epoch=epoch, name='')
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss


def run(recording_time):
    start = time.time()

    seed = 256
    epochs = 50
    lr = 1e-4  # [1e-3, 1e-2, 0.1, 1e-4, 1e-5, 1e-6]
    torch.manual_seed(seed)
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    params_t = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 1,
    }

    # Init of training data loader
    train_dataset = HDF5Dataset_RPN(file_path='/home/roboy/Projects/RoboyMedium/data/train_hdf5_130100_proposal/',
                                    transform=None)
    training_gen = Data.DataLoader(train_dataset, **params_t)
    if VERBOSE:
        print('trainingset Len:')
        print(train_dataset.__len__())

    params_val = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 1
    }
    val_dataset = HDF5Dataset_RPN(file_path='/home/roboy/Projects/RoboyMedium/data/test_hdf5_130100_proposal/',
                                  transform=None)
    validation_gen = Data.DataLoader(val_dataset, **params_val)
    if VERBOSE:
        print('valset len: ')
        print(val_dataset.__len__())

    # loading nn structure#

    model = RF_Pose_RPN(20, 13)
    model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    # Set Optimizer Parameters
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)

    # Initialize Output
    all_losses = []
    all_val_lasses = []
    itemized = list()

    for epoch in range(1, epochs + 1):
        losses, item = train(model, device, training_gen, optimizer, epoch)
        all_losses.extend(losses)
        itemized.append(item)
        val_losses = val(model, device, validation_gen, epoch, recording_time)
        all_val_lasses.append(val_losses)
        scheduler.step()

    ende = time.time()
    duration = ende - start
    if VERBOSE:
        print(ende - start)

    return model, all_losses, all_val_lasses, itemized, duration


if __name__ == '__main__':
    MODELPATH = '/home/roboy/Projects/RoboyMedium/data/model.pt'
    recording_time = time.strftime("%Y_%m_%d_%H_%M")
    out = run(recording_time)
    model = out[0]

    torch.save(model.state_dict(), 'model_' + recording_time + '.pt')
    pickle.dump(out, open("trainig_run" + recording_time + '.pickle', "wb"))
