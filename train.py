import json

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from dataset import WaveformDataModule
from model import CleanUNet

if __name__ == '__main__':
    with open('config/config.json') as f:
        data = f.read()
    config = json.loads(data)
    config_common = config['common']
    config_train = config['train']

    if config_train['checkpoint'] is None:
        model = CleanUNet(learning_rate=config_train['optimizer']['learning_rate'])
    else:
        print('Loading checkpoint...')
        model = CleanUNet.load_from_checkpoint(config_train['checkpoint'])

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    callbacks = [checkpoint_callback]
    if config_train['early_stopping']['enabled']:
        callbacks.append(EarlyStopping(monitor=config_train['early_stopping']['monitor'],
                                       patience=config_train['early_stopping']['patience'],
                                       ))

    if config_train['accelerator'] == 'gpu':
        trainer = Trainer(max_epochs=config_train['epochs'],
                          accelerator=config_train['accelerator'],
                          gpus=-1,
                          callbacks=callbacks
                          )
    else:
        trainer = Trainer(max_epochs=config_train['epochs'],
                          accelerator=config_train['accelerator'],
                          callbacks=callbacks
                          )

    trainer.fit(model, datamodule=WaveformDataModule())