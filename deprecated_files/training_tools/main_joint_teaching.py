import argparse
import copy
import logging
import os
from tqdm import tqdm
import torch
import numpy as np
from HOC.models.freeocnet import FreeOCNet
from HOC.loss_functions.pf_bce_loss import BCELoss
from HOC.loss_functions.vPU_loss import AsyVarPULoss
from deprecated_files.utils import basic_logging, classmap_2_RGBmap, ScalarRecorder, get_cfg_dataloader, all_metric

from HOC.apis.toolkit_noisy_label_learning import get_small_loss_unlabeled_samples


def Argparse():
    parser = argparse.ArgumentParser(
        description='OneSeg+')
    parser.add_argument('-c', '--cls', type=int, default=6, help='Detected class')
    parser.add_argument('-d', '--dataset', type=str, default='HanChuan', help='Dataset')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='GPU_ID')
    parser.add_argument('-r', '--risk', type=str, default="Test", help='Risk Estimation')
    parser.add_argument('-m', '--model', type=str, default='FreeOCNet', help='Model')

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


if __name__ == '__main__':
    args = Argparse()
    # set_random_seed(2333)  # Fixed random seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config, DataLoader = get_cfg_dataloader(dataset=args.dataset)

    config['data']['train']['params']['cls'] = args.cls
    config['data']['test']['params']['cls'] = args.cls

    # Config log file
    extra_name = 'asyvar_co'

    folder_name = os.path.join(args.dataset,
                               'positive-samples_' + str(config['data']['train']['params'][
                                                             'num_positive_train_samples']) + '_sub-minibatch_' + str(
                                   config['data']['train']['params']['sub_minibatch']) + '_ratio_' + str(
                                   config['data']['train']['params']['ratio']),
                               'warm_up_epoch_' + str(config['risk_estimation']['warm_up_epoch']) + '_loss_' +
                               config['risk_estimation']['loss'],
                               args.model,
                               extra_name
                               )

    save_path = basic_logging(
        os.path.join('log', args.risk, folder_name, str(config['data']['train']['params']['cls'])))
    print("The save path is:", save_path)

    asyvar_save_path = os.path.join(save_path, 'asyvar')
    asyvar_recorder_save_path = os.path.join(save_path, 'asyvar_recoder')
    bce1_save_path = os.path.join(save_path, 'bce1')
    bce2_save_path = os.path.join(save_path, 'bce2')
    for path in [asyvar_save_path, asyvar_recorder_save_path, bce1_save_path, bce2_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    dataloader = DataLoader(config=config['data']['train']['params'])
    test_dataloader = DataLoader(config=config['data']['test']['params'])

    if args.model == 'FreeOCNet':
        model = FreeOCNet(config['model']['params']).to(DEVICE)

    recorder_model = FreeOCNet(config['model']['params']).to(DEVICE)
    bce1_model = FreeOCNet(config['model']['params']).to(DEVICE)
    # bce2_model = FreeOCNet(config['model']['params']).to(DEVICE)

    if config['optimizer']['type'] == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    momentum=config['optimizer']['params']['momentum'],
                                    weight_decay=config['optimizer']['params']['weight_decay'],
                                    lr=config['learning_rate']['params']['base_lr'])
    elif config['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     weight_decay=config['optimizer']['params']['weight_decay'],
                                     lr=config['learning_rate']['params']['base_lr'])
    else:
        NotImplemented

    bce1_optimizer = torch.optim.SGD(params=bce1_model.parameters(),
                                     momentum=config['optimizer']['params']['momentum'],
                                     weight_decay=config['optimizer']['params']['weight_decay'],
                                     lr=config['learning_rate']['params']['base_lr'])
    # bce2_optimizer = torch.optim.SGD(params=bce2_model.parameters(),
    #                                  momentum=config['optimizer']['params']['momentum'],
    #                                  weight_decay=config['optimizer']['params']['weight_decay'],
    #                                  lr=config['learning_rate']['params']['base_lr'])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                       gamma=config['learning_rate']['params']['power'])

    asyvar_loss = AsyVarPULoss(asy=1)

    bce1_loss = BCELoss(equal_weight=True)
    # bce2_loss = BCELoss(equal_weight=False)
    # joint_loss =

    # Gradient KDE
    # grad_recorder = ScalarRecorder()
    asyvar_f1_recorder = ScalarRecorder()
    asyvar_recoder_f1_recorder = ScalarRecorder()
    asyvar_loss_recorder = ScalarRecorder()
    asyvar_p_loss_recorder = ScalarRecorder()
    asyvar_u_loss_recorder = ScalarRecorder()

    bce1_f1_recorder = ScalarRecorder()
    bce1_loss_recorder = ScalarRecorder()
    bce1_p_loss_recorder = ScalarRecorder()
    bce1_u_loss_recorder = ScalarRecorder()
    final_loss = 999999

    first_warm_up_epoch = 750
    second_warm_epoch = 3

    bar = tqdm(list(range(config['learning_rate']['params']['max_iters'])), ncols=180)
    for i in bar:
        asyvar_training_loss = 0.0
        asyvar_training_p_loss = 0.0
        asyvar_training_u_loss = 0.0

        bce1_training_loss = 0.0
        bce1_training_p_loss = 0.0
        bce1_training_u_loss = 0.0

        num = 0
        ################################################################################################################
        # for (data, positive_train_mask, unlabeled_train_mask) in dataloader:
        #     data = data.to(DEVICE)
        #     positive_train_mask = positive_train_mask.to(DEVICE)
        #     unlabeled_train_mask = unlabeled_train_mask.to(DEVICE)
        #
        #     if pse_bce_label is not None:
        #         unlabeled_train_mask = (unlabeled_train_mask * (
        #                 torch.ones_like(unlabeled_train_mask) - torch.squeeze(pse_bce_label, dim=0))).detach()
        #
        #     model.train()
        #     target = model(data)
        #     asyvar_loss_value, asyvar_estimated_p_loss, asyvar_estimated_n_loss = asyvar_loss(target,
        #                                                                                       positive_train_mask,
        #                                                                                       unlabeled_train_mask)
        #     optimizer.zero_grad()
        #     asyvar_loss_value.backward()
        #     optimizer.step()
        #
        #     asyvar_training_loss += asyvar_loss_value.item()
        #     asyvar_training_p_loss += asyvar_estimated_p_loss.item()
        #     asyvar_training_u_loss += asyvar_estimated_n_loss.item()
        #
        #     num += 1
        #
        # scheduler.step()
        # if asyvar_training_loss / num < final_loss:
        #     recorder_model = copy.deepcopy(model)
        #     final_loss = asyvar_training_loss / num
        #
        # pse_asyvar_label = None
        # if first_warm_up_epoch < i:
        #     bce1_matrix_loss = None
        #     for (data, positive_train_mask, unlabeled_train_mask) in dataloader:
        #         data = data.to(DEVICE)
        #         positive_train_mask = positive_train_mask.to(DEVICE)
        #         unlabeled_train_mask = unlabeled_train_mask.to(DEVICE)
        #
        #         if pse_asyvar_label is None:
        #             recorder_model.eval()
        #             with torch.no_grad():
        #                 pse_asyvar_label = torch.where(recorder_model(data) > 0, 1, 0)
        #
        #         pse_negative_label = (unlabeled_train_mask * (
        #                 torch.ones_like(unlabeled_train_mask) - torch.squeeze(pse_asyvar_label, dim=0))).detach()
        #
        #         bce1_model.eval()
        #         # with torch.no_grad():
        #         bce1_eval_target = bce1_model(data)
        #         _, _, _, bce1_matrix_loss = bce1_loss(
        #             bce1_eval_target,
        #             positive_train_mask,
        #             unlabeled_train_mask,
        #             reduction=True)
        #
        #         ratio = pse_asyvar_label.sum() / (unlabeled_train_mask.shape[1] * unlabeled_train_mask.shape[2])
        #         pse_bce_label = get_small_loss_unlabeled_samples(unlabeled_train_mask=unlabeled_train_mask,
        #                                                          pse_negative_label=pse_negative_label,
        #                                                          loss_mask=torch.squeeze(bce1_matrix_loss, dim=0),
        #                                                          ratio=torch.min(torch.tensor(0.5).to(DEVICE),ratio))
        #
        #         bce1_model.train()
        #         bce1_target = bce1_model(data)
        #         bce1_loss_value, bce1_estimated_p_loss, bce1_estimated_n_loss = bce1_loss(
        #             bce1_eval_target,
        #             positive_train_mask,
        #             pse_bce_label)
        #
        #         bce1_optimizer.zero_grad()
        #         bce1_loss_value.backward()
        #         bce1_optimizer.step()
        #
        #         bce1_training_loss += bce1_loss_value.item()
        #         bce1_training_p_loss += bce1_estimated_p_loss.item()
        #         bce1_training_u_loss += bce1_estimated_n_loss.item()
        #
        #         # pse_bce_label = torch.where(bce1_target > 0, 1, 0).detach()
        #
        #     # if True:
        #     #     pse_bce_label = get_small_loss_unlabeled_samples(unlabeled_train_mask=unlabeled_train_mask,
        #     #                                                      loss_mask=torch.squeeze(bce1_matrix_loss, dim=0),
        #     #                                                      ratio=torch.min(torch.tensor(0.5).to(DEVICE),pse_asyvar_label.sum()/(unlabeled_train_mask.shape[1]*unlabeled_train_mask.shape[2])))
        ################################################################################################################
        if first_warm_up_epoch > i:
            for (data, positive_train_mask, unlabeled_train_mask) in dataloader:
                data = data.to(DEVICE)
                positive_train_mask = positive_train_mask.to(DEVICE)
                unlabeled_train_mask = unlabeled_train_mask.to(DEVICE)

                model.train()
                target = model(data)
                asyvar_loss_value, asyvar_estimated_p_loss, asyvar_estimated_n_loss = asyvar_loss(target,
                                                                                                  positive_train_mask,
                                                                                                  unlabeled_train_mask)

                bce1_model.train()
                bce1_target = bce1_model(data)
                bce1_loss_value, bce1_estimated_p_loss, bce1_estimated_n_loss = bce1_loss(
                    bce1_target,
                    positive_train_mask,
                    unlabeled_train_mask)

                optimizer.zero_grad()
                asyvar_loss_value.backward()
                optimizer.step()

                bce1_optimizer.zero_grad()
                bce1_loss_value.backward()
                bce1_optimizer.step()

                asyvar_training_loss += asyvar_loss_value.item()
                asyvar_training_p_loss += asyvar_estimated_p_loss.item()
                asyvar_training_u_loss += asyvar_estimated_n_loss.item()

                bce1_training_loss += bce1_loss_value.item()
                bce1_training_p_loss += bce1_estimated_p_loss.item()
                bce1_training_u_loss += bce1_estimated_n_loss.item()

                num += 1

                scheduler.step()
                if asyvar_training_loss / num < final_loss:
                    recorder_model = copy.deepcopy(model)
                    final_loss = asyvar_training_loss / num
        else:
            for (data, positive_train_mask, unlabeled_train_mask) in dataloader:
                data = data.to(DEVICE)
                positive_train_mask = positive_train_mask.to(DEVICE)
                unlabeled_train_mask = unlabeled_train_mask.to(DEVICE)

                recorder_model.eval()
                # model.eval()
                bce1_model.eval()
                pse_asyvar_label = torch.where(recorder_model(data) > 0, 1, 0).detach()
                # pse_asyvar_label = torch.where(model(data) > 0, 1, 0).detach()
                bce1_eval_target = bce1_model(data)

                ratio = pse_asyvar_label.sum() / (unlabeled_train_mask.shape[1] * unlabeled_train_mask.shape[2])
                _, _, _, bce1_matrix_loss = bce1_loss(bce1_eval_target, positive_train_mask, unlabeled_train_mask,
                                                      reduction=True)

                pse_bce_label = get_small_loss_unlabeled_samples(unlabeled_train_mask=unlabeled_train_mask,
                                                                 pse_negative_label=unlabeled_train_mask,
                                                                 loss_mask=torch.squeeze(bce1_matrix_loss, dim=0),
                                                                 ratio=torch.min(torch.tensor(0.5).to(DEVICE), ratio))

                asyvar_label = pse_bce_label.detach()
                # bce1_label = (unlabeled_train_mask * (
                #         torch.ones_like(unlabeled_train_mask) - torch.squeeze(pse_asyvar_label, dim=0))).detach()

                model.train()
                bce1_model.train()

                target = model(data)
                asyvar_loss_value, asyvar_estimated_p_loss, asyvar_estimated_n_loss = asyvar_loss(target,
                                                                                                  positive_train_mask,
                                                                                                  unlabeled_train_mask)
                optimizer.zero_grad()
                asyvar_loss_value.backward()
                optimizer.step()

                asyvar_training_loss += asyvar_loss_value.item()
                asyvar_training_p_loss += asyvar_estimated_p_loss.item()
                asyvar_training_u_loss += asyvar_estimated_n_loss.item()

                bce1_target = bce1_model(data)
                bce1_loss_value, bce1_estimated_p_loss, bce1_estimated_n_loss = bce1_loss(
                    bce1_eval_target,
                    positive_train_mask,
                    asyvar_label)

                bce1_optimizer.zero_grad()
                bce1_loss_value.backward()
                bce1_optimizer.step()

                bce1_training_loss += bce1_loss_value.item()
                bce1_training_p_loss += bce1_estimated_p_loss.item()
                bce1_training_u_loss += bce1_estimated_n_loss.item()

                num += 1

                scheduler.step()
                if asyvar_training_loss / num < final_loss and i < first_warm_up_epoch:
                    recorder_model = copy.deepcopy(model)
                    final_loss = asyvar_training_loss / num

        asyvar_loss_recorder.update_gradient(asyvar_training_loss / num)
        asyvar_p_loss_recorder.update_gradient(asyvar_training_p_loss / num)
        asyvar_u_loss_recorder.update_gradient(asyvar_training_u_loss / num)

        bce1_loss_recorder.update_gradient(bce1_training_loss / num)
        bce1_p_loss_recorder.update_gradient(bce1_training_p_loss / num)
        bce1_u_loss_recorder.update_gradient(bce1_training_u_loss / num)

        asyvar_path_1 = os.path.join(asyvar_save_path, str(i + 1) + '.png')
        asyvar_path_2 = os.path.join(asyvar_save_path, 'probaility.npy')

        asyvar_recoder_path_1 = os.path.join(asyvar_recorder_save_path, str(i + 1) + '.png')
        asyvar_recoder_path_2 = os.path.join(asyvar_recorder_save_path, 'probaility.npy')

        bce1_path_1 = os.path.join(bce1_save_path, str(i + 1) + '.png')
        bce1_path_2 = os.path.join(bce1_save_path, 'probaility.npy')

        auc, fpr, tpr, threshold, pre, rec, f1 = fcn_evaluate_fn(model,
                                                                 test_dataloader=test_dataloader,
                                                                 out_fig_config=config['out_config']['params'],
                                                                 cls=config['data']['train']['params']['cls'],
                                                                 device=DEVICE,
                                                                 path=(asyvar_path_1, asyvar_path_2))

        recoder_auc, recoder_fpr, recoder_tpr, recoder_threshold, recoder_pre, recoder_rec, recoder_f1 = fcn_evaluate_fn(
            recorder_model,
            test_dataloader=test_dataloader,
            out_fig_config=config['out_config']['params'],
            cls=config['data']['train']['params']['cls'],
            device=DEVICE,
            path=(asyvar_recoder_path_1, asyvar_recoder_path_2))
        if first_warm_up_epoch < i:
            bce1_auc, bce1_fpr, bce1_tpr, bce1_threshold, \
            bce1_pre, bce1_rec, bce1_f1 = fcn_evaluate_fn(bce1_model,
                                                          test_dataloader=test_dataloader,
                                                          out_fig_config=
                                                          config['out_config']['params'],
                                                          cls=config['data']['train']['params']['cls'],
                                                          device=DEVICE,
                                                          path=(bce1_path_1, bce1_path_2))
        else:
            bce1_auc, bce1_fpr, bce1_tpr, bce1_threshold, bce1_pre, bce1_rec, bce1_f1 = 0, 0, 0, 0, 0, 0, 0

        asyvar_f1_recorder.update_gradient(f1)
        asyvar_recoder_f1_recorder.update_gradient(recoder_f1)
        bce1_f1_recorder.update_gradient(bce1_f1)

        asyvar_auc_roc = {}
        asyvar_auc_roc['fpr'] = fpr
        asyvar_auc_roc['tpr'] = tpr
        asyvar_auc_roc['threshold'] = threshold
        asyvar_auc_roc['auc'] = auc

        bce1_auc_roc = {}
        bce1_auc_roc['fpr'] = bce1_fpr
        bce1_auc_roc['tpr'] = bce1_tpr
        bce1_auc_roc['threshold'] = bce1_threshold
        bce1_auc_roc['auc'] = bce1_auc

        np.save(os.path.join(asyvar_save_path, 'auc_roc.npy'), asyvar_auc_roc)
        np.save(os.path.join(bce1_save_path, 'auc_roc.npy'), bce1_auc_roc)

        bar.set_description(
            'loss: %.4f,p_loss: %.4f,u_loss: %.4f,AUC:%.6f, Precision:%.6f,Recall:%6f,'
            'F1: %.6f, recoder-F1: %.6f, bce-F1: %.6f' % (asyvar_training_loss / num,
                                                          asyvar_training_p_loss / num,
                                                          asyvar_training_u_loss / num,
                                                          auc,
                                                          pre,
                                                          rec,
                                                          f1,
                                                          recoder_f1,
                                                          bce1_f1))
        logging.info(
            "{} epoch, Training loss {:.4f}, Training p_loss {:.4f}, Training u_loss {:.4f}, AUC {:.6f},"
            " Precision {:.6f}, Recall {:.6f}, F1 {:.6f}, recoder-F1 {:.6f}, bce-F1 {:.6f}".format(
                i + 1,
                asyvar_training_loss / num,
                asyvar_training_p_loss / num,
                asyvar_training_u_loss / num,
                auc,
                pre,
                rec,
                f1,
                recoder_f1,
                bce1_f1))

    asyvar_loss_recorder.save_scalar_npy('loss_npy', asyvar_save_path)
    asyvar_loss_recorder.save_lineplot_fig('Loss', 'loss', asyvar_save_path)
    asyvar_p_loss_recorder.save_scalar_npy('p_loss_npy', asyvar_save_path)
    asyvar_p_loss_recorder.save_lineplot_fig('Estimated Positive Loss', 'p_loss', asyvar_save_path)
    asyvar_u_loss_recorder.save_scalar_npy('u_loss_npy', asyvar_save_path)
    asyvar_u_loss_recorder.save_lineplot_fig('Estimated Unlabeled Loss', 'n_loss', asyvar_save_path)
    # grad_recorder.save_kde_fig(save_path)
    # grad_recorder.save_scalar_npy('gradient_npy', save_path)
    # grad_recorder.save_lineplot_fig('Gradient', 'gradient', save_path)
    asyvar_f1_recorder.save_scalar_npy('f1_npy', asyvar_save_path)
    asyvar_f1_recorder.save_lineplot_fig('F1-score', 'f1-score', asyvar_save_path)

    bce1_loss_recorder.save_scalar_npy('loss_npy', bce1_save_path)
    bce1_loss_recorder.save_lineplot_fig('Loss', 'loss', bce1_save_path)
    bce1_p_loss_recorder.save_scalar_npy('p_loss_npy', bce1_save_path)
    bce1_p_loss_recorder.save_lineplot_fig('Estimated Positive Loss', 'p_loss', bce1_save_path)
    bce1_u_loss_recorder.save_scalar_npy('u_loss_npy', bce1_save_path)
    bce1_u_loss_recorder.save_lineplot_fig('Estimated Unlabeled Loss', 'n_loss', bce1_save_path)
    # grad_recorder.save_kde_fig(save_path)
    # grad_recorder.save_scalar_npy('gradient_npy', save_path)
    # grad_recorder.save_lineplot_fig('Gradient', 'gradient', save_path)
    bce1_f1_recorder.save_scalar_npy('f1_npy', bce1_save_path)
    bce1_f1_recorder.save_lineplot_fig('F1-score', 'f1-score', bce1_save_path)
