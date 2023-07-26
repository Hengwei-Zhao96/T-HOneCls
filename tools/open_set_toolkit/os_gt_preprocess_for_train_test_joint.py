import copy
import numpy as np

if __name__ == "__main__":
    label_path = "/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/indian_gt17.npy"
    save_path_for_open = "/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/new_indian_gt17.npy"
    save_path_for_close = "/home/zhw2021/code/HOneCls/Data/HSI/Indian_Pines/new_mcc_indian_gt17.npy"
    unlabeled_class = [1, 4, 6, 7, 9, 13, 15, 16, 17]  # for indian pines
    # unlabeled_class = [10, 11, 12, 13, 14, 15]  # for paviaU
    # unlabeled_class = [17]  # for salinas

    label = np.load(label_path)
    unlabeled_class.sort()
    new_label = copy.deepcopy(label).astype(np.float64)
    new_total_class = np.max(label) - len(unlabeled_class) + 1
    print("Raw label")
    print(np.unique(label))
    print("The number of background class of training label:%f" % len(np.where(label == 0)[0]))
    print("**************************************************")
    for c in unlabeled_class:
        new_label[label == c] = -1
        print("The label of new label-mask unknown class")
        print(np.unique(new_label))
    print("**************************************************")
    sum = 1
    for c in unlabeled_class:
        new_label[(label > c) & (new_label != -1)] = label[(label > c) & (new_label != -1)] - sum
        print("The label of new test label-change class index")
        print(np.unique(new_label))
        sum += 1
    print("**************************************************")
    new_label[new_label == -1] = new_total_class
    new_label = new_label.astype(np.int8)
    print("The label of new label")
    print(np.unique(new_label))
    print("The number of background class of new test label:%f" % len(np.where(new_label == 0)[0]))
    print("**************************************************")
    np.save(save_path_for_open, new_label)
    new_label_for_close = copy.deepcopy(new_label)
    new_label_for_close[new_label_for_close == new_total_class] = 0
    new_label_for_close = new_label_for_close.astype(np.int8)
    np.save(save_path_for_close, new_label_for_close)
