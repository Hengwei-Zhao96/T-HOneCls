import argparse
import logging
from cleanlab.classification import LearningWithNoisyLabels
from HOC.utils.classmap_2_RGBmap import classmap_2_RGBmap as classmap2rgbmap
import torch
import numpy as np
from models import FreeNetEncoder
from deprecated_files.PR import PR
import os
from deprecated_files.utils import basic_logging, set_random_seed, get_patch_dataset, all_metric


def Argparse():
    parser = argparse.ArgumentParser(
        description='FreeNet_Encoder')
    parser.add_argument('-c', '--cls', type=int, default=18, help='Detected class')
    parser.add_argument('-d', '--dataset', type=str, default="HongHu", help='Dataset')
    parser.add_argument('-g', '--gpu_id', type=str, default="0", help='GPU_ID')
    parser.add_argument('-s', '--patch_size', type=int, default=11, help='Patch Size')
    parser.add_argument('-m', '--mode', type=str, default="pbl", help='Mode')

    return parser.parse_args()


def predict(image, positive_test_indicator, negative_test_indicator, patch_size, cls, model, device, out_fig_config,
            path):
    # start = time.time()
    c, h, w = image.shape[0], positive_test_indicator.shape[0], positive_test_indicator.shape[1]
    classmap = np.zeros((h, w))
    promap = np.zeros((h, w))

    for i in range(h):
        data = np.zeros((w, c, patch_size, patch_size))
        for j in range(w):
            data[j, :, :, :] = image[:, i:i + patch_size, j:j + patch_size]

        model.clf.define_test_data(data)
        result = model.predict(np.asarray(list(range(data.shape[0]))))

        promap[i] = model.predict_proba()[:, 1]
        classmap[i] = result

    classmap = classmap[:out_fig_config['image_size'][0], :out_fig_config['image_size'][1]]
    promap = promap[:out_fig_config['image_size'][0], :out_fig_config['image_size'][1]]

    cls_fig = classmap2rgbmap(classmap.astype(np.int64), out_fig_config['palette'], cls)
    cls_fig.save(path)

    positive_test_indicator = positive_test_indicator[:out_fig_config['image_size'][0],
                              :out_fig_config['image_size'][1]]
    negative_test_indicator = negative_test_indicator[:out_fig_config['image_size'][0],
                              :out_fig_config['image_size'][1]]

    mask = (torch.from_numpy(positive_test_indicator) + torch.from_numpy(negative_test_indicator)).bool()
    target = torch.masked_select(torch.from_numpy(positive_test_indicator).reshape(-1), mask.reshape(-1)).numpy()
    pred_class = torch.masked_select(torch.from_numpy(classmap).reshape(-1).cpu(), mask.reshape(-1)).numpy()
    pred_pro = torch.masked_select(torch.from_numpy(promap).reshape(-1).cpu(), mask.reshape(-1)).numpy()

    auc, fpr, tpr, threshold, pre, rec, f1 = all_metric(pred_pro, pred_class, target)

    # end = time.time()
    # print("Time:%f" % (end - start))
    return auc, fpr, tpr, threshold, pre, rec, f1


if __name__ == "__main__":
    args = Argparse()
    SEED = 2333
    set_random_seed(SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    patch_size = args.patch_size
    mode = args.mode

    config, image_data, positive_train_data, unlabeled_train_data, positive_test_indicator, negative_test_indicator = \
        get_patch_dataset(patch_size=patch_size, dataset=args.dataset, cls_index=args.cls)

    train_data = np.concatenate((positive_train_data, unlabeled_train_data))
    label = np.concatenate((np.ones(positive_train_data.shape[0]), np.zeros(unlabeled_train_data.shape[0])))
    inds = np.random.permutation(train_data.shape[0])
    train_data = train_data[inds]
    label = label[inds]

    folder_name = args.dataset + '_positive-samples_' + str(
        config['data']['train']['params']['num_positive_train_samples']) + '_sub-minibatch_' + str(
        config['data']['train']['params']['sub_minibatch']) + '_ratio_' + str(
        config['data']['train']['params']['ratio'])
    save_path = basic_logging(
        os.path.join('log', 'deep_pu_calibration', mode.upper(), 'patch_size_' + str(patch_size), folder_name,
                     str(config['data']['train']['params']['cls'])))
    print("The save path is:", save_path)

    model = FreeNetEncoder(in_channel=image_data.shape[0], out_channels=2, patch_size=patch_size)
    optimizer = torch.optim.SGD(params=model.parameters(),
                                momentum=config['optimizer']['params']['momentum'],
                                weight_decay=config['optimizer']['params']['weight_decay'],
                                lr=config['learning_rate']['params']['base_lr'])
    pu_estimator = PR(estimator=model, epoch=1, optimizer=optimizer)
    lnl = LearningWithNoisyLabels(clf=pu_estimator, pulearning=1)
    lnl.clf.define_train_data(X=train_data, y=label)
    lnl.fit(X=np.asarray(list(range(train_data.shape[0]))), s=label)

    path = os.path.join(save_path, mode + '.png')
    auc, fpr, tpr, threshold, pre, rec, f1 = predict(image=image_data,
                                                     positive_test_indicator=positive_test_indicator,
                                                     negative_test_indicator=negative_test_indicator,
                                                     patch_size=patch_size,
                                                     cls=config['data']['train']['params']['cls'],
                                                     model=lnl,
                                                     device=DEVICE,
                                                     out_fig_config=config['out_config']['params'],
                                                     path=path)
    auc_roc = {}
    auc_roc['fpr'] = fpr
    auc_roc['tpr'] = tpr
    auc_roc['threshold'] = threshold
    auc_roc['auc'] = auc

    np.save(os.path.join(save_path, 'auc_roc.npy'), auc_roc)

    logging.info("AUC {:.6f}, Precision {:.6f}, Recall {:.6f}, F1 {:.6f}".format(auc, pre, rec, f1))
