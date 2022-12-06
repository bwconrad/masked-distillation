import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data import DataModule
from src.model import MaskDistillModel
from src.pl_utils import MyLightningArgumentParser, init_logger

model_class = MaskDistillModel
dm_class = DataModule

# Parse arguments
parser = MyLightningArgumentParser()
parser.add_lightning_class_args(pl.Trainer, None)  # type:ignore
parser.add_lightning_class_args(dm_class, "data", skip=["patch_size"])
parser.add_lightning_class_args(model_class, "model")
parser.link_arguments("data.size", "model.img_size")
args = parser.parse_args()

# Setup trainer
logger = init_logger(args)
checkpoint_callback = ModelCheckpoint(
    filename="best-{epoch}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_last=True,
)
model = model_class(**args["model"])
dm = dm_class(patch_size=model.student.patch_size, **args["data"])  # type:ignore

trainer = pl.Trainer.from_argparse_args(
    args, logger=logger, callbacks=[checkpoint_callback]
)

# Train
trainer.tune(model, dm)
trainer.fit(model, dm)
