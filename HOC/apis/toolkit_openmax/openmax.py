import torch
from pytorch_ood.detector import OpenMax


class HOCOpenMax(OpenMax):
    def __init__(self,
                 model: torch.nn.Module,
                 tailsize: int = 25,
                 alpha: int = 10,
                 euclid_weight: float = 1.0, ):
        super(HOCOpenMax, self).__init__(model=model, tailsize=tailsize, alpha=alpha, euclid_weight=euclid_weight)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input, will be passed through the model to obtain logits
        """
        with torch.no_grad():
            z = self.model(x).cpu().numpy()

        return torch.tensor(self._openmax.predict(z))
