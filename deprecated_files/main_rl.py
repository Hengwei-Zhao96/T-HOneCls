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

from HOC.apis.toolkit_reinforcement_learning import Agent
from HOC.apis.toolkit_reinforcement_learning import warming_up_training, agent_training, env_training


def Argparse():
    parser = argparse.ArgumentParser(
        description='OneSeg+')
    parser.add_argument('-c', '--cls', type=int, default=4, help='Detected class')
    parser.add_argument('-d', '--dataset', type=str, default='HongHu', help='Dataset')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='GPU_ID')
    parser.add_argument('-r', '--risk', type=str, default="AsyVarPULoss", help='Risk Estimation')
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
    extra_name = 'rl_asyvar1_env_pse'

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

    env_save_path = os.path.join(save_path, 'env')
    agent_save_path = os.path.join(save_path, 'agent')
    for path in [env_save_path, agent_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    dataloader = DataLoader(config=config['data']['train']['params'])
    test_dataloader = DataLoader(config=config['data']['test']['params'])

    if args.model == 'FreeOCNet':
        model = FreeOCNet(config['model']['params']).to(DEVICE)

    recorder_model = FreeOCNet(config['model']['params']).to(DEVICE)

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

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                       gamma=config['learning_rate']['params']['power'])

    # env_warming_up_trainaing_loss_1_bce = BCELoss(equal_weight=True)
    env_warming_up_trainaing_loss_1_asyvar = AsyVarPULoss(asy=1)
    env_warming_up_trainaing_loss_2_bce = BCELoss(equal_weight=True)
    # env_warming_up_trainaing_loss_2_asyvar = AsyVarPULoss(asy=1)
    # env_mix_loss = MixLoss(loss_1=env_warming_up_trainaing_loss_1_asyvar, loss_2=env_warming_up_trainaing_loss_1_bce)

    agent_model = FreeOCNet(config['model']['params']).to(DEVICE)
    agent = Agent(agent_model=agent_model)
    agent_warming_up_trainaing_loss_1_bce = BCELoss(equal_weight=True)
    agent_warming_up_trainaing_loss_1_asyvar = AsyVarPULoss(asy=1)
    agent_warming_up_trainaing_loss_2_bce = BCELoss(equal_weight=True)
    agent_optimizer = torch.optim.SGD(params=agent_model.parameters(),
                                      momentum=config['optimizer']['params']['momentum'],
                                      weight_decay=config['optimizer']['params']['weight_decay'],
                                      lr=config['learning_rate']['params']['base_lr'])

    # Gradient KDE
    # grad_recorder = ScalarRecorder()
    env_f1_recorder = ScalarRecorder()
    env_loss_recorder = ScalarRecorder()
    env_p_loss_recorder = ScalarRecorder()
    env_u_loss_recorder = ScalarRecorder()

    agent_f1_recorder = ScalarRecorder()
    agent_loss_recorder = ScalarRecorder()
    agent_p_loss_recorder = ScalarRecorder()
    agent_u_loss_recorder = ScalarRecorder()
    final_loss = 999999

    first_warm_up_epoch = 1
    second_warm_epoch = 3

    bar = tqdm(list(range(config['learning_rate']['params']['max_iters'])), ncols=180)
    for i in bar:
        env_training_loss = 0.0
        env_training_p_loss = 0.0
        env_training_u_loss = 0.0

        agent_training_loss = 0.0
        agent_training_p_loss = 0.0
        agent_training_u_loss = 0.0

        num = 0

        for (data, positive_train_mask, unlabeled_train_mask) in dataloader:
            data = data.to(DEVICE)
            positive_train_mask = positive_train_mask.to(DEVICE)
            unlabeled_train_mask = unlabeled_train_mask.to(DEVICE)

            model.train()
            agent.train()

            if i <= first_warm_up_epoch:
                # env_mix_loss.get_epoch(epoch=i, total_epoch=first_warm_up_epoch)

                target = model(data)

                env_loss_value, env_estimated_p_loss, env_estimated_n_loss = warming_up_training(optimizer=optimizer,
                                                                                                 loss_fun=env_warming_up_trainaing_loss_1_asyvar,
                                                                                                 target=target,
                                                                                                 pse_label=None,
                                                                                                 positive_train_mask=positive_train_mask,
                                                                                                 unlabeled_train_mask=unlabeled_train_mask)

                agent_loss_value, agent_estimated_p_loss, agent_estimated_n_loss = torch.tensor(0), torch.tensor(
                    0), torch.tensor(0)

                # agent_output = agent(data)
                # agent_loss_value, agent_estimated_p_loss, agent_estimated_n_loss = warming_up_training(
                #     optimizer=agent_optimizer,
                #     loss_fun=agent_warming_up_trainaing_loss_1_bce,
                #     target=agent_output,
                #     pse_label=None,
                #     positive_train_mask=positive_train_mask,
                #     unlabeled_train_mask=unlabeled_train_mask)

                if i == first_warm_up_epoch:
                    recorder_model.eval()
                    with torch.no_grad():
                        logits = recorder_model(data)
                    pseudo_label = torch.where(logits > 0, 1, 0)

            elif first_warm_up_epoch < i <= second_warm_epoch:
                target = model(data)
                env_loss_value, env_estimated_p_loss, env_estimated_n_loss = warming_up_training(optimizer=optimizer,
                                                                                                 loss_fun=env_warming_up_trainaing_loss_2_bce,
                                                                                                 target=target,
                                                                                                 pse_label=pseudo_label,
                                                                                                 positive_train_mask=positive_train_mask,
                                                                                                 unlabeled_train_mask=unlabeled_train_mask)

                agent.act(data)
                model.eval()
                with torch.no_grad():
                    target_after = model(data)
                agent_loss_value, agent_estimated_p_loss, agent_estimated_n_loss = agent_training(
                    agent_optimizer=agent_optimizer,
                    agent=agent,
                    env_logits=target_after,
                    positive_train_mask=positive_train_mask,
                    unlabeled_train_mask=unlabeled_train_mask)
                agent.onpolicy_reset()

            else:
                target = model(data)
                agent.act(data)
                env_loss_value, env_estimated_p_loss, env_estimated_n_loss = env_training(env_optimizer=optimizer,
                                                                                          target=target,
                                                                                          pse_label=agent.action,
                                                                                          # pse_label=agent.probability,
                                                                                          positive_train_mask=positive_train_mask,
                                                                                          unlabeled_train_mask=unlabeled_train_mask)
                model.eval()
                with torch.no_grad():
                    target_after = model(data)
                agent_loss_value, agent_estimated_p_loss, agent_estimated_n_loss = agent_training(
                    agent_optimizer=agent_optimizer,
                    agent=agent,
                    env_logits=target_after,
                    positive_train_mask=positive_train_mask,
                    unlabeled_train_mask=unlabeled_train_mask)
                agent.onpolicy_reset()

            env_training_loss += env_loss_value.item()
            env_training_p_loss += env_estimated_p_loss.item()
            env_training_u_loss += env_estimated_n_loss.item()

            agent_training_loss += agent_loss_value.item()
            agent_training_p_loss += agent_estimated_p_loss.item()
            agent_training_u_loss += agent_estimated_n_loss.item()

            num += 1

            if i < first_warm_up_epoch and env_training_loss / num < final_loss:
                recorder_model = copy.deepcopy(model)
                final_loss = env_training_loss / num

        scheduler.step()

        env_loss_recorder.update_gradient(env_training_loss / num)
        env_p_loss_recorder.update_gradient(env_training_p_loss / num)
        env_u_loss_recorder.update_gradient(env_training_u_loss / num)

        agent_loss_recorder.update_gradient(agent_training_loss / num)
        agent_p_loss_recorder.update_gradient(agent_training_p_loss / num)
        agent_u_loss_recorder.update_gradient(agent_training_u_loss / num)

        env_path_1 = os.path.join(env_save_path, str(i + 1) + '.png')
        env_path_2 = os.path.join(env_save_path, 'probaility.npy')

        agent_path_1 = os.path.join(agent_save_path, str(i + 1) + '.png')
        agent_path_2 = os.path.join(agent_save_path, 'probaility.npy')

        auc, fpr, tpr, threshold, pre, rec, f1 = fcn_evaluate_fn(model,
                                                                 test_dataloader=test_dataloader,
                                                                 out_fig_config=config['out_config']['params'],
                                                                 cls=config['data']['train']['params']['cls'],
                                                                 device=DEVICE,
                                                                 path=(env_path_1, env_path_2))

        if i <= first_warm_up_epoch:
            a_auc, a_fpr, a_tpr, a_threshold, a_pre, a_rec, a_f1 = fcn_evaluate_fn(recorder_model,
                                                                                   test_dataloader=test_dataloader,
                                                                                   out_fig_config=config['out_config'][
                                                                                       'params'],
                                                                                   cls=config['data']['train']['params'][
                                                                                       'cls'],
                                                                                   device=DEVICE,
                                                                                   path=(agent_path_1, agent_path_2))
        else:
            a_auc, a_fpr, a_tpr, a_threshold, a_pre, a_rec, a_f1 = fcn_evaluate_fn(agent,
                                                                                   test_dataloader=test_dataloader,
                                                                                   out_fig_config=config['out_config'][
                                                                                       'params'],
                                                                                   cls=
                                                                                   config['data']['train']['params'][
                                                                                       'cls'],
                                                                                   device=DEVICE,
                                                                                   path=(agent_path_1, agent_path_2))

        env_f1_recorder.update_gradient(f1)
        agent_f1_recorder.update_gradient(a_f1)

        env_auc_roc = {}
        env_auc_roc['fpr'] = fpr
        env_auc_roc['tpr'] = tpr
        env_auc_roc['threshold'] = threshold
        env_auc_roc['auc'] = auc

        agent_auc_roc = {}
        agent_auc_roc['fpr'] = a_fpr
        agent_auc_roc['tpr'] = a_tpr
        agent_auc_roc['threshold'] = a_threshold
        agent_auc_roc['auc'] = a_auc

        np.save(os.path.join(env_save_path, 'auc_roc.npy'), env_auc_roc)
        np.save(os.path.join(agent_save_path, 'auc_roc.npy'), agent_auc_roc)

        bar.set_description(
            'loss: %.4f,p_loss: %.4f,u_loss: %.4f,AUC:%.6f, Precision:%.6f,Recall:%6f,'
            'F1: %.6f, a-F1: %.6f' % (env_training_loss / num,
                                      env_training_p_loss / num,
                                      env_training_u_loss / num,
                                      auc,
                                      pre,
                                      rec,
                                      f1, a_f1))
        logging.info(
            "{} epoch, Training loss {:.4f}, Training p_loss {:.4f}, Training u_loss {:.4f}, AUC {:.6f},"
            " Precision {:.6f}, Recall {:.6f}, F1 {:.6f}, a-F1 {:.6f}".format(
                i + 1,
                env_training_loss / num,
                env_training_p_loss / num,
                env_training_u_loss / num,
                auc,
                pre,
                rec,
                f1, a_f1))

    env_loss_recorder.save_scalar_npy('loss_npy', env_save_path)
    env_loss_recorder.save_lineplot_fig('Loss', 'loss', env_save_path)
    env_p_loss_recorder.save_scalar_npy('p_loss_npy', env_save_path)
    env_p_loss_recorder.save_lineplot_fig('Estimated Positive Loss', 'p_loss', env_save_path)
    env_u_loss_recorder.save_scalar_npy('u_loss_npy', env_save_path)
    env_u_loss_recorder.save_lineplot_fig('Estimated Unlabeled Loss', 'n_loss', env_save_path)
    # grad_recorder.save_kde_fig(save_path)
    # grad_recorder.save_scalar_npy('gradient_npy', save_path)
    # grad_recorder.save_lineplot_fig('Gradient', 'gradient', save_path)
    env_f1_recorder.save_scalar_npy('f1_npy', env_save_path)
    env_f1_recorder.save_lineplot_fig('F1-score', 'f1-score', env_save_path)

    agent_loss_recorder.save_scalar_npy('loss_npy', agent_save_path)
    agent_loss_recorder.save_lineplot_fig('Loss', 'loss', agent_save_path)
    agent_p_loss_recorder.save_scalar_npy('p_loss_npy', agent_save_path)
    agent_p_loss_recorder.save_lineplot_fig('Estimated Positive Loss', 'p_loss', agent_save_path)
    agent_u_loss_recorder.save_scalar_npy('u_loss_npy', agent_save_path)
    agent_u_loss_recorder.save_lineplot_fig('Estimated Unlabeled Loss', 'n_loss', agent_save_path)
    # grad_recorder.save_kde_fig(save_path)
    # grad_recorder.save_scalar_npy('gradient_npy', save_path)
    # grad_recorder.save_lineplot_fig('Gradient', 'gradient', save_path)
    agent_f1_recorder.save_scalar_npy('f1_npy', agent_save_path)
    agent_f1_recorder.save_lineplot_fig('F1-score', 'f1-score', agent_save_path)
