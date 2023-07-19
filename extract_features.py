import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from model import get_model
from dataset import TestDataset, ToSpecTorch

def file_feats_generator(dataloader, batch_size):
    datas = list()
    files = list()
    for l, d in dataloader:
        files.extend(l)
        datas.extend(d)
        if len(datas)>=batch_size:
            yield files, datas
            datas = list()
            files = list()
    if len(datas)>0:
        yield files, datas

def extract_feats(data, network, to_spec, device):
    norm = 0

    with torch.no_grad():
        feats = to_spec(data.to(device))
        del data

        feats = feats.unsqueeze(1)

        feats = network(feats)

        if len(feats.shape) > 2:
            feats = feats[:, :, feats.shape[2] // 2]  # B x C

        if norm > 0:
            feats = torch.nn.functional.normalize(feats, p=opt['norm'], dim=1, eps=1e-12)
        elif norm < 0:
            feats = feats / (feats.norm(-opt['norm'], dim=1, keepdim=True) + 1.0)

        feats = np.float16(feats.cpu().numpy())

    return feats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-csv', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--weights', type=str, default="./checkpoint/model_with_augmentation.th")
    parser.add_argument("--features-folder", type=str, default="./features")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seconds", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--manual-seed", type=int, default=42)

    opt = parser.parse_args()

    if (opt.gpu >= 0):
        torch.cuda.manual_seed_all(opt.manual_seed)
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(opt.gpu)

    device = f'cuda:{opt.gpu}' if (opt.gpu >= 0) else 'cpu'
    print(f"Device is {device}")

    if opt.dataset_csv == None:
        raise ValueError("The flag --dataset-csv has to be se to a csv file as specified in the ReadMe instructions")
    if opt.dataset_name == None:
        opt.dataset_name = os.path.basename(opt.dataset_csv)


    ## Create folder structure that we are going to use
    os.makedirs(os.path.join(opt.features_folder, opt.dataset_name), exist_ok=True)

    ## define function that we are going to use
    def get_output_filename(filename):
        return os.path.join(opt.features_folder, opt.dataset_name, f"{filename}.npy")

    ## lets open the csv file
    df = pd.read_csv(opt.dataset_csv)

    ## create a list of files we need to extract features from
    filelist = [(row['videoname'], row['filepath']) for _, row in df.iterrows()]
    ## let's keep only the files that we haven't extracted and saved yet
    filelist = [_ for _ in filelist if not os.path.isfile(get_output_filename(_[0]))]

    ## Define the spectrogram extractor
    spectrogram = ToSpecTorch().to(device)

    ## Load the model
    model = get_model(checkpoint=opt.weights, device=device)

    ## Define the dataset
    dataset = TestDataset(filelist, spectrogram.get_len(opt.seconds*100), spectrogram.get_stride(opt.seconds*100))

    ## Define a dataloader so we can open multiple files at the same time
    dataloader = DataLoader(dataset, batch_size=None, num_workers=opt.num_workers, shuffle=False)

    ## Define variables to keep track of which file are we working on and a list to save features related to that file
    current_file = ''
    current_file_feats = list()

    with tqdm(total=len(dataloader), desc="Extracting features") as pbar:
        for files, datas in file_feats_generator(dataloader, opt.batch_size): #while len(datas) > 0:
            feats = torch.stack(datas, 0).to(device) #datas[:opt.batch_size]
            #files_batch = files[:opt.batch_size]

            feats = extract_feats(feats, model, spectrogram, device)

            for filename, features in zip(files, feats):
                if filename != current_file: ## if we are going onto a new file (or the first file)
                    if len(current_file_feats) > 0: ## and there are features stored
                        output_filename = get_output_filename(current_file)
                        np.save(output_filename, np.stack(current_file_feats, 0)) ## we save them to disc
                    current_file = filename ## change current filename we are working on
                    current_file_feats = list() ## and empty the cache of features
                current_file_feats.append(features) ## append features to an empty list or a list containing only features of the same file

            output_filename = get_output_filename(current_file)
            np.save(output_filename, np.stack(current_file_feats, 0))  ## we save last file to disk
            pbar.update(min(opt.batch_size,len(files)))

        pbar.close()

    print("Extraction done!")
    ## Terminal cleanup in case tqdm break the terminal
    os.system("stty sane")