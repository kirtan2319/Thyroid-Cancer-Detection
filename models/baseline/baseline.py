from models.utils import standard_train
from models.basenet import BaseNet
from importlib import import_module


class baseline(BaseNet):
    def __init__(self, opt, wandb):
        super(baseline, self).__init__(opt, wandb)
        self.set_network(opt)
        self.set_optimizer(opt)

    def set_network(self, opt):
        """Define the network"""
        
        if self.is_3d:
            mod = import_module("models.basemodels_3d")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, pretrained=self.pretrained).to(self.device)
        elif self.is_tabular:
            mod = import_module("models.basemodels_mlp")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, in_features=self.in_features, hidden_features=1024).to(self.device)
        else:
            mod = import_module("models.basemodels")
            cusModel = getattr(mod, self.backbone)
            self.network = cusModel(n_classes=self.output_dim, pretrained=self.pretrained).to(self.device)
            
    def _train(self, loader):
        """Train the model for one epoch"""
        print("train part")
        self.network.train()

        # print("Input loader received:", len(loader))
        # for i, (inputs, labels) in enumerate(loader):
        #     print(f"Batch {i}: Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")


        print("train2")
        auc, train_loss = standard_train(self.opt, self.network, self.optimizer, loader, self._criterion, self.wandb)
        print("train3")
        print(f'Training epoch {self.epoch}: AUC: {auc}')
        print(f'Training epoch {self.epoch}: loss: {train_loss}')
        
        self.epoch += 1
