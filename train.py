from architecture import ChestXrayCNNRes
from dataset import get_train_dataset, get_test_dataset, get_dataloader
import torch
from logger import init_logger
from trainingmanager import TrainingManager
import os

os.system(f"caffeinate -is -w {os.getpid()} &")


EXPERIMENT_DIRECTORY = "runs/v2-res"


device = "mps" if torch.backends.mps.is_available() else "cpu"

dataloader = get_dataloader(get_train_dataset())

testloader = get_dataloader(get_test_dataset())


net = ChestXrayCNNRes()
net.to(device)

trainer = TrainingManager(
    net=net,
    dir=EXPERIMENT_DIRECTORY,
    dataloader=dataloader,
    device=device,
    trainstep_checkin_interval=10,
    epochs=50,
    val_dataloader=testloader,
)


for imgs, label in dataloader:
    init_logger(
        net, imgs.to(device), dir=os.path.join(EXPERIMENT_DIRECTORY, "tensorboard")
    )
    break

trainer.train()
