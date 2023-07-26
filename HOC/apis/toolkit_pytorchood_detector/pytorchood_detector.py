from pytorch_ood.detector import MCD, MaxSoftmax, Mahalanobis, EnergyBased, Entropy, MaxLogit, ODIN, KLMatching


def detector_init(detector_name, model):
    if detector_name == 'MCD':
        return MCD(model)
    elif detector_name == 'MaxSoftmax':
        return MaxSoftmax(model)
    elif detector_name == 'Mahalanobis':
        return Mahalanobis(model)
    elif detector_name == 'EnergyBased':
        return EnergyBased(model)
    elif detector_name == 'Entropy':
        return Entropy(model)
    elif detector_name == 'MaxLogit':
        return MaxLogit(model)
    elif detector_name == 'ODIN':
        return ODIN(model)
    elif detector_name == 'KLMatching':
        return KLMatching(model)