# models/binary_classifier.py
import torch
import torch.nn as nn
import pytorch_lightning as pl

class BinaryClassifier(pl.LightningModule):
    """
    PyTorch Lightning实现的二分类模型
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = self._build_layers()
        
    def _build_layers(self):
        layers = []
        prev_dim = self.config['architecture']['input_dim']
        
        for i, dim in enumerate(self.config['architecture']['hidden_layers']):
            # 添加线性层
            layers.append(nn.Linear(prev_dim, dim))
            
            # 添加批归一化
            if self.config['architecture']['use_batch_norm']:
                layers.append(nn.BatchNorm1d(dim))
            
            # 添加激活函数
            layers.append(nn.ReLU())
            
            # 添加dropout
            if i < len(self.config['architecture']['dropout_rates']):
                layers.append(nn.Dropout(self.config['architecture']['dropout_rates'][i]))
            
            prev_dim = dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.float().view(-1, 1))
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.float().view(-1, 1))
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config['training']['learning_rate']
        )
