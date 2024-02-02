import os

from scripts.models.kramer_original import KramerOriginal

from scripts.utils.dataset import DeepRODataModule
from scripts.utils.params import Parameters

import pytorch_lightning as pl

if __name__== '__main__':
    args = Parameters()
    data_module = DeepRODataModule(args)
    model = KramerOriginal(args)
    trainer = pl.Trainer(logger=True,accelerator='gpu',devices=1,max_epochs=args.epochs)
    trainer.fit(model,data_module)