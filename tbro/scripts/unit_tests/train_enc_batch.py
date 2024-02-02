import os
from random import shuffle
import shutil
import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from scripts.models.kramer_original import KramerOriginal

from scripts.utils.dataset import DeepRODataset
from scripts.utils.params import Parameters

import math

def validate(test_loader, full_errors=False, batch_size=16):
  test_loss = []
  orientation_err = None
  position_err = None

  for i, data in enumerate(test_loader):
    seq_len, radar_data, positions, orientations = data[0], data[1], data[2], data[3]
    
    positions = torch.stack(positions, dim=1)
    orientations = torch.stack(orientations, dim=1)

    positions = positions.squeeze()
    orientations = orientations.squeeze()

    positions = positions.to(device)
    orientations = orientations.to(device)

    for j in range(batch_size):
      for k in range(seq_len[j]):
        radar_data[k] = radar_data[k].to(device)
        if torch.any(torch.isnan(radar_data[k])):
          print('Image (batch,elem in batch,elem in seq)'+str(i,j,k)+' contains NaN')

    with torch.no_grad():
        y_hat = model(radar_data)
    

    position_loss = loss_func(y_hat[:,3:], positions)
    orientation_loss = loss_func(y_hat[:,:3], orientations)
    loss = 1 * orientation_loss + position_loss

    test_loss.append(loss.item())

    position_err_batch = (y_hat[:,3:] - positions).abs().view(-1,3)
    
    orientation_err_batch = (y_hat[:,:3] - orientations).abs().view(-1,3)

    if orientation_err is None:
      orientation_err = orientation_err_batch
    else:
      orientation_err = torch.cat((orientation_err, orientation_err_batch), 0)
    
    if position_err is None:
      position_err = position_err_batch
    else:
      position_err = torch.cat((position_err, position_err_batch), 0)

  or_err_mean = orientation_err.mean(dim=0)
  or_err_max = orientation_err.max(dim=0).values
  or_err_min = orientation_err.min(dim=0).values
  or_err_median = orientation_err.median(dim=0).values
  or_err_std_dev = orientation_err.std(dim=0)

  pos_err_mean = position_err.mean(dim=0)
  pos_err_max = position_err.max(dim=0).values
  pos_err_min = position_err.min(dim=0).values
  pos_err_median = position_err.median(dim=0).values
  pos_err_std_dev = position_err.std(dim=0)

  print('\nTesting loss - %.2E' % (sum(test_loss) / len(test_loss)))

  print('\tMean orientation error (rpy, rad) - %f, %f, %f' % 
     (or_err_mean[0],
     or_err_mean[1],
     or_err_mean[2]))

  if full_errors:
    print('\tMedian orientation error (rpy, rad) - %f, %f, %f' % 
       (or_err_median[0],
       or_err_median[1],
       or_err_median[2]))

    print('\tMax orientation error (rpy, rad) - %f, %f, %f' %
      (or_err_max[0],
      or_err_max[1],
      or_err_max[2]))

    print('\tMin orientation error (rpy, rad) - %f, %f, %f' %
      (or_err_min[0],
      or_err_min[1],
      or_err_min[2]))

    print('\tOrientation error std dev (rpy, rad) - %f, %f, %f\n' %
      (or_err_std_dev[0],
      or_err_std_dev[1],
      or_err_std_dev[2]))

  print('\tMean position error (xyz, m) - %f, %f, %f' % 
     (pos_err_mean[0],
     pos_err_mean[1],
     pos_err_mean[2]))

  if full_errors: 
    print('\tMedian position error (xyz, m) - %f, %f, %f' % 
       (pos_err_median[0],
       pos_err_median[1],
       pos_err_median[2]))

    print('\tMax position error (xyz, m) - %f, %f, %f' %
      (pos_err_max[0],
      pos_err_max[1],
      pos_err_max[2]))

    print('\tMin position error (xyz, m) - %f, %f, %f' %
      (pos_err_min[0],
      pos_err_min[1],
      pos_err_min[2]))

    print('\tPosition error std dev (xyz, m) - %f, %f, %f' %
      (pos_err_std_dev[0],
      pos_err_std_dev[1],
      pos_err_std_dev[2]))

  print('\n')

  return test_loss

if __name__ == '__main__':
    args = Parameters()
    
    # Load dataset and create torch dataloader
    print("Root dir: " + args.directory)
    print("list: " + args.train_files)
    
    train_dataset = DeepRODataset(root_dir = args.directory, list_files = args.train_files, verbose_debug = False, sequence_length = args.max_length)
    print('Loaded ' + str(len(train_dataset)) + ' training sequences.')
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last=True)

    test_dataset = DeepRODataset(root_dir = args.directory, list_files = args.val_files, verbose_debug = False, sequence_length = args.max_length)
    print('Loaded ' + str(len(test_dataset)) + ' testing sequences.')
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last=True)

    # Set torch device
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        print('CUDA used.')
    else:
        device = torch.device('cpu')
        print('CPU used.')

    # Load Model
    model = KramerOriginal()
    model = model.to(device)

    # Create an optimizer (TODO: Test optimizers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    loss_func = torch.nn.MSELoss()

    # Loads model if it exists
    if args.load_filename != '' and os.path.exists(args.directory+args.load_filename):
        print('Loading model from file.')
        print('Model name: ', args.load_filename)
        model.load_state_dict(torch.load(args.directory+args.load_filename, map_location=device))

    # Setup epoch based training
    epoch_train_losses = []
    epoch_test_losses = []
    
    num_epochs = args.epochs

    writer = SummaryWriter()

    if not args.eval_only:
        print('---Training Model---')
        for epoch in range(num_epochs):
            start = time.time()
            train_losses = []
            model.train()
            for i, data in enumerate(train_loader):
                seq_len, radar_data, positions, orientations = data[0], data[1], data[2], data[3]

                positions = torch.stack(positions, dim=1)
                orientations = torch.stack(orientations, dim=1)

                positions = positions.squeeze()
                orientations = orientations.squeeze()

                positions = positions.to(device)
                orientations = orientations.to(device)

                for j in range(args.batch_size):
                  for k in range(seq_len[j]):
                    radar_data[k] = radar_data[k].to(device)
                    if torch.any(torch.isnan(radar_data[k])):
                      print('Image (batch,elem in batch,elem in seq)'+str(i,j,k)+' contains NaN')
                
                optimizer.zero_grad()

                y_hat = model(radar_data)

                position_loss = loss_func(y_hat[:,3:], positions)
                orientation_loss = loss_func(y_hat[:,:3], orientations)

                loss = 1 * orientation_loss + position_loss

                # writer.add_scalar("Training Loss",loss,i)
                writer.add_scalar("Training Loss",loss,i+epoch*len(train_dataset))
                
                train_losses.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()

                if i % 500 == 0:
                        # # print(gt_data[0])
                        # positions = gt_data[0][:,:3]
                        # orientations = gt_data[0][:,3:]
                        # # print(positions)
                        # # print(orientations)
                        # position_err = (y_hat[:,3:] - positions).abs().mean(dim=0)
                        # #position_err = (y_hat - positions).abs().mean(dim=0)
                        # orientation_err = (y_hat[:,:3] - orientations).abs().mean(dim=0)
                        
                        # print('[%d: %d/%d] Training loss - %.2E \n\t mean rotation error (rpy, rad) - %f, %f, %f \n\t mean translation error (xyz, m) - %f, %f, %f' %
                        # (epoch, i, 
                        # len(train_loader), 
                        # loss.item(), 
                        # orientation_err[0],
                        # orientation_err[1],
                        # orientation_err[2], 
                        # position_err[0],
                        # position_err[1],
                        # position_err[2]))
                        print('[%d: %d/%d] Training loss - %.2E'%(epoch, i, len(train_loader), loss.item()))
                        print('Training took {:.1f} sec'.format(time.time()-start))

            print('Epoch',epoch)
            print('Epoch training took {:.1f} sec'.format(time.time()-start))
            if (epoch < num_epochs - 1):
                model = model.eval()
                test_losses = validate(test_loader,full_errors = False, batch_size=args.batch_size)
                epoch_test_losses.append(sum(test_losses) / float(len(test_losses)))
                writer.add_scalar("Validation Loss", sum(test_losses) / float(len(test_losses)),epoch)

            scheduler.step()
            epoch_train_losses.append(sum(train_losses) / float(len(train_losses)))
            if args.save_filename != '':
                torch.save(model.state_dict(), args.save_filename)

    writer.flush()

    print('Running final evaluation.')
    model = model.eval()
    test_loss = validate(test_loader, True, args.batch_size)

    epoch_test_losses.append(sum(test_loss) / float(len(test_loss)))

    epoch_test_losses = torch.FloatTensor(epoch_test_losses)
    epoch_train_losses = torch.FloatTensor(epoch_train_losses)

    torch.save(epoch_test_losses, 'lr_'+str(args.learning_rate)+'test_losses.pt')
    torch.save(epoch_train_losses, 'lr_'+str(args.learning_rate)+'train_losses.pt')
    writer.close()

    '''
    plt.plot(epoch_test_losses, c='b', label='test loss')
    plt.plot(epoch_train_losses, c='r', label='train loss')
    plt.legend()
    plt.show()
    '''










