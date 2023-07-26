import argparse
from sklearn import svm
from HOC.utils.classmap_2_RGBmap import classmap_2_RGBmap as classmap2rgbmap
import torch
import numpy as np
import logging
import os
from deprecated_files.utils import basic_logging, get_patch_dataset, set_random_seed, all_metric
from cleanlab.classification import LearningWithNoisyLabels


def Argparse():
    parser = argparse.ArgumentParser(description='..')
    parser.add_argument('-c', '--cls', type=int, default=4, help='Detected class')
    parser.add_argument('-d', '--dataset', type=str, default="HongHu", help='Dataset')

    return parser.parse_args()


def predict(image, positive_test_indicator, negative_test_indicator, cls, model, out_fig_config, path):
    c, h, w = image.shape[0], positive_test_indicator.shape[0], positive_test_indicator.shape[1]
    image = image.transpose((1, 2, 0))
    classmap = np.zeros((h, w))
    promap = np.zeros((h, w))
    for i in range(h):
        data = image[i, :, :]
        classmap[i] = model.predict(data)
        promap[i] = model.predict_proba(data)[:, 1]

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

    return auc, fpr, tpr, threshold, pre, rec, f1


if __name__ == "__main__":
    args = Argparse()
    SEED = 2333
    set_random_seed(SEED)

    config, image_data, positive_train_data, unlabeled_train_data, positive_test_indicator, negative_test_indicator = \
        get_patch_dataset(patch_size=1, dataset=args.dataset, cls_index=args.cls)

    folder_name = args.dataset + '_positive-samples_' + str(
        config['data']['train']['params']['num_positive_train_samples']) + '_sub-minibatch_' + str(
        config['data']['train']['params']['sub_minibatch']) + '_ratio_' + str(
        config['data']['train']['params']['ratio'])
    save_path = basic_logging(
        os.path.join('log', "pr".upper(), folder_name,
                     str(config['data']['train']['params']['cls'])))
    print("The save path is:", save_path)

    train_data = np.concatenate((positive_train_data, unlabeled_train_data))
    label = np.concatenate((np.ones(positive_train_data.shape[0], dtype=np.int64),
                            np.zeros(unlabeled_train_data.shape[0], dtype=np.int64)))
    inds = np.random.permutation(train_data.shape[0])
    train_data = train_data[inds]
    label = label[inds]

    estimator = svm.SVC(kernel='rbf', probability=True)
    pr = LearningWithNoisyLabels(clf=estimator, n_jobs=1,pulearning=1)
    pr.fit(X=train_data, s=label)

    path = os.path.join(save_path, 'pr' + '.png')
    auc, fpr, tpr, threshold, pre, rec, f1 = predict(image_data, positive_test_indicator, negative_test_indicator,
                                                     cls=config['data']['train']['params']['cls'],
                                                     model=pr, out_fig_config=config['out_config']['params'],
                                                     path=path)
    auc_roc = {}
    auc_roc['fpr'] = fpr
    auc_roc['tpr'] = tpr
    auc_roc['threshold'] = threshold
    auc_roc['auc'] = auc

    np.save(os.path.join(save_path, 'auc_roc.npy'), auc_roc)

    logging.info("AUC {:.6f}, Precision {:.6f}, Recall {:.6f}, F1 {:.6f}".format(auc, pre, rec, f1))
