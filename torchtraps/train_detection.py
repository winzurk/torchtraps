#!/usr/bin/env python
# coding: utf-8
# Author: Zac Winzurk
# Project: Torch Traps

import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import copy
from barbar import Bar

from coco_camera_traps_loader import load_train_val
from magic import configure_report


def train_model(model, args, run):

    # configure report file
    experiment, f = configure_report(args, run)

    # load dataset
    dataset = load_train_val(args)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # Model, Loss, Optimizer
    model.classifier = nn.Linear(1024, dataset['num_classes'])
    model = torch.nn.DataParallel(model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # Training Loop
    model = training_loop(args, model, criterion, optimizer, dataset, f, device, experiment)

    return model


def training_loop(args, model, criterion, optimizer, dataset, f, device, experiment):

    start = time.time()
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch} began')
        running_loss = 0.0
        running_corrects = 0
        # training phase
        for idx, data in enumerate(Bar(dataset['train_dataloader'])):
            inputs = Variable(data.get('image')).to(device)
            target = Variable(data.get('target')).to(device)
            # forward pass
            output = model(inputs)
            _, preds = torch.max(output, 1)
            loss = criterion(output, target)
            loss = loss / args.accumulation_steps           # Normalize accumulated loss (averaged)
            loss = loss.mean()
            # backward pass
            loss.backward()                                 # Backward pass (mean of parallel loss)
            if (idx+1) % args.accumulation_steps == 0:      # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                model.zero_grad()                           # Reset gradient tensors

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == target.data)
        # log training stats
        train_epoch_loss = running_loss / len(dataset['train_data'])
        train_epoch_acc = running_corrects.double() / len(dataset['train_data'])
        print('Epoch [{}/{}], training loss:{:.4f}'.format(epoch+1, args.num_epochs, train_epoch_loss))
        print('Epoch [{}/{}], training accuracy:{:.4f}'.format(epoch+1, args.num_epochs, train_epoch_acc))
        # validation phase
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for idx, data in enumerate(Bar(dataset['val_dataloader'])):
                inputs = Variable(data.get('image')).to(device)
                target = Variable(data.get('target')).to(device)
                output = model(inputs)
                _, preds = torch.max(output, 1)
                loss = criterion(output, target).mean()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == target.data)
        # log validation stats
        valid_epoch_loss = running_loss / len(dataset['val_data'])
        valid_epoch_acc = running_corrects.double() / len(dataset['val_data'])
        print('Epoch [{}/{}], validation loss:{:.4f}'.format(epoch+1, args.num_epochs, valid_epoch_loss))
        print('Epoch [{}/{}], validation accuracy:{:.4f}'.format(epoch+1, args.num_epochs, valid_epoch_acc))
        # append to experiment report
        print(f'{epoch+1}\t{train_epoch_loss}\t{train_epoch_acc}\t{valid_epoch_loss}\t{valid_epoch_acc}',
              file=open(f, "a"))
        # save best weights
        if valid_epoch_acc > best_acc:
            best_acc = valid_epoch_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f'models/{args.dataset}/{experiment}.pth')

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=open(f, "a"))
    print('Best val Acc: {:4f}'.format(best_acc), file=open(f, "a"))

    # load best weights
    model.load_state_dict(f'models/{args.dataset}/{experiment}.pth')

    return model
