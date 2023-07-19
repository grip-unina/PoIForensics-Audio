import torch
from torchvision.models import resnet50
import os

def change_input(model, num_inputs):
    data = model.conv1.weight.data
    old_num_inputs = int(data.shape[1])
    if num_inputs > old_num_inputs:
        times = num_inputs // old_num_inputs
        if (times * old_num_inputs) < num_inputs:
            times = times + 1
        data = data.repeat(1, times, 1, 1) / times
    elif num_inputs == old_num_inputs:
        return model

    data = data[:, :num_inputs, :, :]
    #print(model.conv1.weight.data.shape, '->', data.shape)
    model.conv1.weight.data = data

    return model

def Batch2Group(bn, num_groups):
    gn = torch.nn.GroupNorm(
        num_groups=num_groups,
        num_channels=bn.num_features,
        eps=bn.eps,
        affine=bn.affine).to(bn.weight.device)

    if gn.affine:
        # dem = torch.sqrt(bn.running_var+bn.eps)
        # gn.weight.data =  bn.weight / dem
        # gn.bias.data = bn.bias - bn.running_mean * gn.weight.data
        gn.weight.data = bn.weight.data
        gn.bias.data = bn.bias.data

    return gn

def convertAllBatch2Group(model, count = 0):
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.BatchNorm2d):
            setattr(model, name, Batch2Group(layer, 4))
            count = count + 1
        else:
            count = count + convertAllBatch2Group(layer)[1]
    return model, count


def get_model(num_classes=256, checkpoint="./checkpoint/model_no_augmentation.th", device="cuda:0"):
    model = resnet50(weights='IMAGENET1K_V2')
    model = change_input(model, 1)
    model.channel_first = True
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model, _ = convertAllBatch2Group(model)
    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise ValueError(f"The specified checkpoint file {checkpoint} does not exist")
        model.load_state_dict(torch.load(checkpoint, map_location=device)['net_time'])
        print("Checkpoint Loaded!")
    else:
        print("No resuming!")

    if device:
        model.to(device)


    return model.eval()
