import os
import argparse

import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from loss import YoloLoss
from yolo_v2 import Yolo
import shutil



def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    learning_rate_schedule = {"0": 1e-5, "5": 1e-4,
                              "80": 1e-5, "110": 1e-6}

    training_params = {"batch_size" : opt.batch_size,
                       "shuffle" : True,
                       "drop_last" : True,
                       "collate_fn" : custom_collate_fn}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": custom_collate_fn}

    training_set = COCODataset(opt.data_path, opt.year, opt.train_set, opt.image_size)
    training_generator = DataLoader(training_set, **training_params)

    test_set = COCODataset(opt.data_path, opt.year, opt.test_set, opt.image_size, is_training=False)
    test_generator = DataLoader(test_set, **test_params)

    if torch.cuda.is_available():
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path)
        else:
            model = Yolo(training_set.num_classes)
            model.load_state_dict(torch.load(opt.pre_trained_model_path))
    else:
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage)
        else:
            model = Yolo(training_set.num_classes)
            model.load_state_dict(torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage))

    nn.init.normal_(list(model.modules())[-1].weight, 0, 0.01)
    criterion = YoloLoss(training_set.num_classes, model.anchors, opt.reduction)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=opt.momentum, weight_decay=opt.decay)
    best_loss = 1e10
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)

    for epoch in range(opt.num_epoches):
        if str(epoch) in learning_rate_schedule.keys():
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_schedule[str(epoch)]
        for iter, batch in enumerate(training_generator):
            image, label = batch
            if torch.cuda.is_available():
                image = Variable(image.cuda(), requires_grad=True)
            else:
                image = Variable(image, requires_grad=True)
            optimizer.zero_grad()
            logits = model(image)
            loss, loss_coord, loss_conf, loss_cls = criterion(logits, label)
            loss.backward()
            optimizer.step()
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss:{:.2f} (Coord:{:.2f} Conf:{:.2f} Cls:{:.2f})".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss,
                loss_coord,
                loss_conf,
                loss_cls))

        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            loss_coord_ls = []
            loss_conf_ls = []
            loss_cls_ls = []
            for te_iter, te_batch in enumerate(test_generator):
                te_image, te_label = te_batch
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_image = te_image.cuda()
                with torch.no_grad():
                    te_logits = model(te_image)
                    batch_loss, batch_loss_coord, batch_loss_conf, batch_loss_cls = criterion(te_logits, te_label)
                loss_ls.append(batch_loss * num_sample)
                loss_coord_ls.append(batch_loss_coord * num_sample)
                loss_conf_ls.append(batch_loss_conf * num_sample)
                loss_cls_ls.append(batch_loss_cls * num_sample)
            te_loss = sum(loss_ls) / test_set.__len__()
            te_coord_loss = sum(loss_coord_ls) / test_set.__len__()
            te_conf_loss = sum(loss_conf_ls) / test_set.__len__()
            te_cls_loss = sum(loss_cls_ls) / test_set.__len__()
            print("Epoch: {}/{}, Lr: {}, Loss:{:.2f} (Coord:{:.2f} Conf:{:.2f} Cls:{:.2f})".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss,
                te_coord_loss,
                te_conf_loss,
                te_cls_loss))

            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                # torch.save(model, opt.saved_path + os.sep + "trained_yolo_coco")
                torch.save(model.state_dict(), opt.saved_path + os.sep + "only_params_trained_yolo_coco")
                torch.save(model, opt.saved_path + os.sep + "whole_model_trained_yolo_coco")

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break


if __name__ == "__main__":
    opt = get_args()
    train(opt)