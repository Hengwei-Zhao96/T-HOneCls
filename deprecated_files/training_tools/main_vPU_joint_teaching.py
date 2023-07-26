import argparse
import logging
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from HOC.models.freeocnet import FreeOCNet
from HOC.loss_functions.pf_bce_loss import KLLoss, L2Loss
from HOC.loss_functions.vPU_loss import TaylorVarPULoss
from deprecated_files.utils import basic_logging, classmap_2_RGBmap, ScalarRecorder, get_cfg_dataloader, all_metric
from deprecated_files.utils.update_ema_model import update_ema_variables


def Argparse():
    parser = argparse.ArgumentParser(
        description='OneSeg+')
    parser.add_argument('-c', '--cls', type=int, default=1, help='Detected class')
    parser.add_argument('-d', '--dataset', type=str, default='HanChuan', help='Dataset')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='GPU_ID')
    parser.add_argument('-t', '--training_mode', type=str, default="vpu_joint_teaching", help='Risk Estimation')
    parser.add_argument('-m', '--asyvar_loss_m', type=int, default=1, help='m in AsyVarPULoss')
    parser.add_argument('-k', '--asyvar_loss_k', type=int, default=30, help='k in AsyVarPULoss')
    parser.add_argument('-a', '--alpha', type=float, default=1, help='alpha')
    parser.add_argument('-s', '--smooth_factor', type=float, default=0.99, help='smooth factor')
    parser.add_argument('-b', '--beta', type=float, default=0.1, help='beta')

    return parser.parse_args()


def fcn_evaluate_fn(model, test_dataloader, out_fig_config, cls, path, device):
    # start = time.time()

    model.eval()
    f1 = 0
    # start = time.time()
    with torch.no_grad():
        for (im, positive_test_mask, negative_test_mask) in test_dataloader:
            im = im.to(device)
            positive_test_mask = positive_test_mask.squeeze()
            negative_test_mask = negative_test_mask.squeeze()
            pred_pro = torch.sigmoid(model(im)).squeeze().cpu()
            # end = time.time()
            # print("Time:%f" % (end - start))
            pred_class = torch.where(pred_pro > 0.5, 1, 0)

            cls_fig = classmap_2_RGBmap(
                pred_class[:out_fig_config['image_size'][0], :out_fig_config['image_size'][1]].numpy(),
                palette=out_fig_config['palette'], cls=cls)
            cls_fig.save(path[0])

            np.save(path[1], pred_pro[:out_fig_config['image_size'][0], :out_fig_config['image_size'][1]].numpy())

            mask = (positive_test_mask + negative_test_mask).bool()

            label = positive_test_mask
            target = torch.masked_select(label.view(-1), mask.view(-1)).numpy()
            pred_class = torch.masked_select(pred_class.view(-1).cpu(), mask.view(-1)).numpy()
            pred_pro = torch.masked_select(pred_pro.view(-1).cpu(), mask.view(-1)).numpy()

            auc, fpr, tpr, threshold, pre, rec, f1 = all_metric(pred_pro, pred_class, target)
    # end = time.time()
    # print("Time:%f" % (end - start))

    return auc, fpr, tpr, threshold, pre, rec, f1


def fusion_fcn_evaluate_fn(model, test_dataloader, out_fig_config, cls, path, device):
    # start = time.time()

    for m in model:
        m.eval()
    f1 = 0
    # start = time.time()
    with torch.no_grad():
        for (im, positive_test_mask, negative_test_mask) in test_dataloader:
            im = im.to(device)
            positive_test_mask = positive_test_mask.squeeze()
            pred_pro = torch.unsqueeze(torch.zeros_like(negative_test_mask), dim=0).to(device)
            # print(positive_train_mask.sum())
            negative_test_mask = negative_test_mask.squeeze()
            for m in model:
                pred_pro += torch.sigmoid(m(im))
            pred_pro = (pred_pro / len(model)).squeeze().cpu()
            # end = time.time()
            # print("Time:%f" % (end - start))
            pred_class = torch.where(pred_pro > 0.5, 1, 0)

            cls_fig = classmap_2_RGBmap(
                pred_class[:out_fig_config['image_size'][0], :out_fig_config['image_size'][1]].numpy(),
                palette=out_fig_config['palette'], cls=cls)
            cls_fig.save(path[0])

            np.save(path[1], pred_pro[:out_fig_config['image_size'][0], :out_fig_config['image_size'][1]].numpy())

            mask = (positive_test_mask + negative_test_mask).bool()

            label = positive_test_mask
            target = torch.masked_select(label.view(-1), mask.view(-1)).numpy()
            pred_class = torch.masked_select(pred_class.view(-1).cpu(), mask.view(-1)).numpy()
            pred_pro = torch.masked_select(pred_pro.view(-1).cpu(), mask.view(-1)).numpy()

            auc, fpr, tpr, threshold, pre, rec, f1 = all_metric(pred_pro, pred_class, target)
    # end = time.time()
    # print("Time:%f" % (end - start))

    return auc, fpr, tpr, threshold, pre, rec, f1


if __name__ == '__main__':
    args = Argparse()
    # set_random_seed(2333)  # Fixed random seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config, DataLoader = get_cfg_dataloader(dataset=args.dataset)

    # loss_function = AsyVarPULoss(asy=1, m=args.asyvar_loss_m, k=args.asyvar_loss_k)
    loss_function = TaylorVarPULoss(order=1, m=args.asyvar_loss_m, k=args.asyvar_loss_k)
    alphaloss = L2Loss(equal_weight=True)
    betaloss = KLLoss(equal_weight=True)
    config['data']['train']['params']['cls'] = args.cls
    config['data']['test']['params']['cls'] = args.cls

    # Config log file
    extra_name = 'lr_0.0001_gradclip_taylor'

    folder_name = os.path.join(args.dataset,
                               'positive-samples_' + str(config['data']['train']['params'][
                                                             'num_positive_train_samples']) + '_sub-minibatch_' + str(
                                   config['data']['train']['params']['sub_minibatch']) + '_ratio_' + str(
                                   config['data']['train']['params']['ratio']),
                               args.training_mode,
                               extra_name)

    save_path = basic_logging(
        os.path.join('log', 'AsyvarPuLoss', folder_name, str(config['data']['train']['params']['cls'])))
    print("The save path is:", save_path)
    model1_save_path = os.path.join(save_path, 'model1')
    model1_smooth_save_path = os.path.join(save_path, 'model1_smooth')
    model2_save_path = os.path.join(save_path, 'model2')
    model2_smooth_save_path = os.path.join(save_path, 'model2_smooth')
    model_fusion_save_path = os.path.join(save_path, 'model_fusion')
    for path in [model1_save_path, model1_smooth_save_path, model2_save_path, model2_smooth_save_path,
                 model_fusion_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    dataloader = DataLoader(config=config['data']['train']['params'])
    test_dataloader = DataLoader(config=config['data']['test']['params'])

    model1 = FreeOCNet(config['model']['params']).to(DEVICE)
    model1_smooth = FreeOCNet(config['model']['params']).to(DEVICE)
    model2 = FreeOCNet(config['model']['params']).to(DEVICE)
    model2_smooth = FreeOCNet(config['model']['params']).to(DEVICE)

    for param in model1_smooth.parameters():
        param.detach_()
    for param in model2_smooth.parameters():
        param.detach_()

    if config['optimizer']['type'] == 'SGD':
        optimizer = torch.optim.SGD(params=[{'params': model1.parameters()}, {'params': model2.parameters()}],
                                    momentum=config['optimizer']['params']['momentum'],
                                    weight_decay=config['optimizer']['params']['weight_decay'],
                                    lr=config['learning_rate']['params']['base_lr'])
    elif config['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(params=[{'params': model1.parameters()}, {'params': model2.parameters()}],
                                     weight_decay=config['optimizer']['params']['weight_decay'],
                                     lr=config['learning_rate']['params']['base_lr'])
    else:
        NotImplemented

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                       gamma=config['learning_rate']['params']['power'])

    model1_f1_recorder = ScalarRecorder()
    model1_smooth_f1_recorder = ScalarRecorder()
    model2_f1_recorder = ScalarRecorder()
    model2_smooth_f1_recorder = ScalarRecorder()
    model_fusion_f1_recoder = ScalarRecorder()
    loss_recorder = ScalarRecorder()
    p_loss_recorder1_1 = ScalarRecorder()
    u_loss_recorder1_1 = ScalarRecorder()
    loss1_1_min = 99999
    loss1_2_min = 99999

    bar = tqdm(list(range(config['learning_rate']['params']['max_iters'])), ncols=180)
    for i in bar:
        training_loss = 0.0
        training_loss1_1 = 0.0
        training_loss1_2 = 0.0
        training_p_loss1_1 = 0.0
        training_u_loss1_1 = 0.0
        training_p_loss1_2 = 0.0
        training_u_loss1_2 = 0.0
        num = 0
        model1.train()
        model2.train()
        for (data, positive_train_mask, unlabeled_train_mask) in dataloader:
            data = data.to(DEVICE)
            positive_train_mask = positive_train_mask.to(DEVICE)
            unlabeled_train_mask = unlabeled_train_mask.to(DEVICE)

            target1 = model1(data)
            target2 = model2(data)

            with torch.no_grad():
                model1_smooth_target = model1_smooth(data)
                model2_smooth_target = model2_smooth(data)

            loss1_1, estimated_p_loss1_1, estimated_u_loss1_1 = loss_function(target1, positive_train_mask,
                                                                              unlabeled_train_mask, epoch=i)
            loss1_2, estimated_p_loss1_2, estimated_u_loss1_2 = loss_function(target2, positive_train_mask,
                                                                              unlabeled_train_mask, epoch=i)
            loss2_1 = alphaloss(target1, model1_smooth_target, positive_train_mask, unlabeled_train_mask)
            loss2_2 = alphaloss(target2, model2_smooth_target, positive_train_mask, unlabeled_train_mask)

            loss3_1 = betaloss(target1, model2_smooth_target, positive_train_mask, unlabeled_train_mask)
            loss3_2 = betaloss(target2, model1_smooth_target, positive_train_mask, unlabeled_train_mask)

            loss = (loss1_1 + loss1_2) + args.alpha * (loss2_1 + loss2_2) + args.beta * (loss3_1 + loss3_2)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model1.parameters(), 10.0)
            nn.utils.clip_grad_norm_(model2.parameters(), 10.0)

            optimizer.step()

            training_loss += loss.item()
            training_loss1_1 += loss1_1.item()
            training_loss1_2 += loss1_2.item()
            training_p_loss1_1 += estimated_p_loss1_1.item()
            training_u_loss1_1 += estimated_u_loss1_1.item()
            training_p_loss1_2 += estimated_p_loss1_2.item()
            training_u_loss1_2 += estimated_u_loss1_2.item()
            num += 1
        scheduler.step()
        update_ema_variables(model=model1, ema_model=model1_smooth, alpha=args.smooth_factor, global_step=i)
        update_ema_variables(model=model2, ema_model=model2_smooth, alpha=args.smooth_factor, global_step=i)
        # if training_loss1_1/num < loss1_1_min:
        #     update_ema_variables(model=model1, ema_model=model1_smooth, alpha=args.smooth_factor, global_step=i)
        #     loss1_1_min = training_loss1_1/num
        # if training_loss1_2 / num < loss1_2_min:
        #     update_ema_variables(model=model2, ema_model=model2_smooth, alpha=args.smooth_factor, global_step=i)
        #     loss1_2_min = training_loss1_2 / num
        loss_recorder.update_gradient(training_loss / num)
        p_loss_recorder1_1.update_gradient(training_p_loss1_1 / num)
        u_loss_recorder1_1.update_gradient(training_u_loss1_1 / num, )

        model1_path_1 = os.path.join(model1_save_path, str(i + 1) + '.png')
        model1_path_2 = os.path.join(model1_save_path, 'probaility.npy')
        model1_smooth_path_1 = os.path.join(model1_smooth_save_path, str(i + 1) + '.png')
        model1_smooth_path_2 = os.path.join(model1_smooth_save_path, 'probaility.npy')
        model2_path_1 = os.path.join(model2_save_path, str(i + 1) + '.png')
        model2_path_2 = os.path.join(model2_save_path, 'probaility.npy')
        model2_smooth_path_1 = os.path.join(model2_smooth_save_path, str(i + 1) + '.png')
        model2_smooth_path_2 = os.path.join(model2_smooth_save_path, 'probaility.npy')
        model_fusion_save_path_1 = os.path.join(model_fusion_save_path, str(i + 1) + '.png')
        model_fusion_save_path_2 = os.path.join(model_fusion_save_path, 'probaility.npy')

        model1_auc, model1_fpr, model1_tpr, model1_threshold, model1_pre, model1_rec, model1_f1 = fcn_evaluate_fn(
            model1,
            test_dataloader=test_dataloader,
            out_fig_config=config['out_config']['params'],
            cls=config['data']['train']['params']['cls'],
            device=DEVICE,
            path=(model1_path_1, model1_path_2))
        model1_smooth_auc, model1_smooth_fpr, model1_smooth_tpr, model1_smooth_threshold, model1_smooth_pre, model1_smooth_rec, model1_smooth_f1 = fcn_evaluate_fn(
            model1_smooth,
            test_dataloader=test_dataloader,
            out_fig_config=config['out_config']['params'],
            cls=config['data']['train']['params']['cls'],
            device=DEVICE,
            path=(model1_smooth_path_1, model1_smooth_path_2))
        # model2_auc, model2_fpr, model2_tpr, model2_threshold, model2_pre, model2_rec, model2_f1 = fcn_evaluate_fn(
        #    model2,
        #    test_dataloader=test_dataloader,
        #    out_fig_config=config['out_config']['params'],
        #    cls=config['data']['train']['params']['cls'],
        #    device=DEVICE,
        #    path=(model2_path_1, model2_path_2))
        # model2_smooth_auc, model2_smooth_fpr, model2_smooth_tpr, model2_smooth_threshold, model2_smooth_pre, model2_smooth_rec, model2_smooth_f1 = fcn_evaluate_fn(
        #    model2_smooth,
        #    test_dataloader=test_dataloader,
        #    out_fig_config=config['out_config']['params'],
        #    cls=config['data']['train']['params']['cls'],
        #    device=DEVICE,
        #    path=(model2_smooth_path_1, model2_smooth_path_2))

        # model1_auc, model1_fpr, model1_tpr, model1_threshold, model1_pre, model1_rec, model1_f1 = 0, 0, 0, 0, 0, 0, 0
        # model1_smooth_auc, model1_smooth_fpr, model1_smooth_tpr, model1_smooth_threshold, model1_smooth_pre, model1_smooth_rec, model1_smooth_f1 = 0, 0, 0, 0, 0, 0, 0
        model2_auc, model2_fpr, model2_tpr, model2_threshold, model2_pre, model2_rec, model2_f1 = 0, 0, 0, 0, 0, 0, 0
        model2_smooth_auc, model2_smooth_fpr, model2_smooth_tpr, model2_smooth_threshold, model2_smooth_pre, model2_smooth_rec, model2_smooth_f1 = 0, 0, 0, 0, 0, 0, 0

        # model_fusion_smooth_auc, model_fusion_smooth_fpr, model_fusion_smooth_tpr, model_fusion_smooth_threshold, model_fusion_smooth_pre, model_fusion_smooth_rec, model_fusion_smooth_f1 = fusion_fcn_evaluate_fn(
        #     [model1_smooth, model2_smooth],
        #     test_dataloader=test_dataloader,
        #     out_fig_config=config['out_config']['params'],
        #     cls=config['data']['train']['params']['cls'],
        #     device=DEVICE,
        #     path=(model_fusion_save_path_1, model_fusion_save_path_2))

        model_fusion_smooth_auc, model_fusion_smooth_fpr, model_fusion_smooth_tpr, model_fusion_smooth_threshold, model_fusion_smooth_pre, model_fusion_smooth_rec, model_fusion_smooth_f1 = 0, 0, 0, 0, 0, 0, 0

        model1_f1_recorder.update_gradient(model1_f1)
        model1_smooth_f1_recorder.update_gradient(model1_smooth_f1)
        model2_f1_recorder.update_gradient(model2_f1)
        model2_smooth_f1_recorder.update_gradient(model2_smooth_f1)
        model_fusion_f1_recoder.update_gradient(model_fusion_smooth_f1)

        bar.set_description(
            'loss: %.4f,p_loss: %.4f,u_loss: %.4f, Precision:%.6f,Recall:%6f,'
            'F1: %.6f, sF1: %.6f, F2: %.6f, sF2: %.6f, fF: %.6f' % (training_loss / num,
                                                                    training_p_loss1_1 / num,
                                                                    training_u_loss1_1 / num,
                                                                    model1_pre,
                                                                    model1_rec,
                                                                    model1_f1,
                                                                    model1_smooth_f1,
                                                                    model2_f1,
                                                                    model2_smooth_f1,
                                                                    model_fusion_smooth_f1))
        logging.info(
            "{} epoch, Training loss {:.4f}, Training loss_1 {:.4f}, Training p_loss_1 {:.4f}, Training u_loss_1 {:.4f},"
            "Training loss_2 {:.4f}, Training p_loss_2 {:.4f}, Training u_loss_2 {:.4f},"
            " Precision {:.6f}, Recall {:.6f}, F1 {:.6f}, sF1 {:.6f}, F2 {:.6f}, sF2 {:.6f}, fF {:.6f}".format(
                i + 1,
                training_loss / num,
                training_loss1_1 / num,
                training_p_loss1_1 / num,
                training_u_loss1_1 / num,
                training_loss1_2 / num,
                training_p_loss1_2 / num,
                training_u_loss1_2 / num,
                model1_pre,
                model1_rec,
                model1_f1,
                model1_smooth_f1,
                model2_f1,
                model2_smooth_f1,
                model_fusion_smooth_f1))

        if i == (len(bar) - 1):
            # print('\nModel saved!')
            torch.save(model1.state_dict(), os.path.join(model1_save_path, 'checkpoint.pth'))
            torch.save(model1_smooth.state_dict(), os.path.join(model1_smooth_save_path, 'checkpoint.pth'))
            torch.save(model2.state_dict(), os.path.join(model2_save_path, 'checkpoint.pth'))
            torch.save(model2_smooth.state_dict(), os.path.join(model2_smooth_save_path, 'checkpoint.pth'))

    loss_recorder.save_scalar_npy('loss_npy', save_path)
    loss_recorder.save_lineplot_fig('Loss', 'loss', save_path)
    p_loss_recorder1_1.save_scalar_npy('p_loss_npy', model1_save_path)
    p_loss_recorder1_1.save_lineplot_fig('Estimated Positive Loss', 'p_loss', model1_save_path)
    u_loss_recorder1_1.save_scalar_npy('u_loss_npy', model1_save_path)
    u_loss_recorder1_1.save_lineplot_fig('Estimated Unlabeled Loss', 'n_loss', model1_save_path)
    model1_f1_recorder.save_scalar_npy('f1_npy', model1_save_path)
    model1_f1_recorder.save_lineplot_fig('F1-score', 'f1-score', model1_save_path)
    model1_smooth_f1_recorder.save_scalar_npy('f1_npy', model1_smooth_save_path)
    model1_smooth_f1_recorder.save_lineplot_fig('F1-score', 'f1-score', model1_smooth_save_path)
    model2_f1_recorder.save_scalar_npy('f1_npy', model2_save_path)
    model2_f1_recorder.save_lineplot_fig('F1-score', 'f1-score', model2_save_path)
    model2_smooth_f1_recorder.save_scalar_npy('f1_npy', model2_smooth_save_path)
    model2_smooth_f1_recorder.save_lineplot_fig('F1-score', 'f1-score', model2_smooth_save_path)
    model_fusion_f1_recoder.save_scalar_npy('f1_npy', model_fusion_save_path)
    model_fusion_f1_recoder.save_lineplot_fig('F1-score', 'f1-score', model_fusion_save_path)
