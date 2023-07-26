import torch


def get_kflod_mask(mask: torch.Tensor, k_flod=5, seed=2333):
    mask = mask.squeeze()

    train_mask = torch.zeros(mask.shape,device=mask.device)
    val_mask = torch.zeros(mask.shape)

    id_x, id_y = torch.where(mask == 1)

    g = torch.Generator()
    g.manual_seed(seed)
    shuffle_id = torch.randperm(id_x.size(0), generator=g)
    id_x = id_x[shuffle_id]
    id_y = id_y[shuffle_id]

    val_num = id_x.shape[0] // k_flod

    val_mask[id_x[:val_num], id_y[:val_num]] = 1
    train_mask[id_x[val_num:], id_y[val_num:]] = 1

    train_mask = train_mask.unsqueeze(0)
    val_mask = val_mask.unsqueeze(0)

    return train_mask, val_mask

# if __name__ == '__main__':
#     a = torch.Tensor([[0, 1, 1, 2], [2, 2, 2, 1]])
#     get_kflod_mask(a)
