import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import random
import pandas as pd
from multiprocessing import get_context
import traceback

if __name__ == '__main__':
    ctx = get_context("fork")

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-csv', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument("--features-folder", type=str, default="./features")
    parser.add_argument("--scores-folder", type=str, default="./scores")
    parser.add_argument("--strategy", type=str, default="ms")
    parser.add_argument("--num-threads", type=int, default=16)

    opt = parser.parse_args()

    assert opt.strategy in ['cb', 'ms'], "Strategy not supported, please select one between Centroid-Based (cb) or " \
                                         "Multi-Similarity (ms)."


    def get_features_filename(filename):
        return os.path.join(opt.features_folder, opt.dataset_name, f"{filename}.npy")


    ## Load the csv and split it into two csv, test and reference
    df = pd.read_csv(opt.dataset_csv, index_col='videoname')
    df.index = df.index.astype(str)
    df_ref = df[df['in_ref'] != 0][['poi', 'context', 'label']]
    df_tst = df[df['in_tst'] != 0][['poi', 'context']]
    df_tst['value'] = np.nan
    assert np.all(df_ref['label'] == 0)
    del df

    ## A list of all the unique identities
    list_poi = df_ref['poi'].unique()
    ## for each identity
    for poi in list_poi:
        # we get the test and reference list of files for the given poi
        df_ref_poi = df_ref[df_ref['poi'] == poi]
        df_tst_poi = df_tst[df_tst['poi'] == poi]
        ## If one of the two lists are empty, we skip this poi (no file to tests or no reference available)
        if (len(df_ref_poi) == 0) or (len(df_tst_poi) == 0):
            continue

        ## We split features given their context. When testing, we only take files from a different context to avoid polatization
        list_dict = {k: list() for k in df_ref_poi['context'].unique()}

        for videoname, row in tqdm(df_ref_poi.iterrows(), total=len(df_ref_poi), desc='Loading ' + poi):
            features_file = get_features_filename(videoname)
            context = row['context']
            ## Check the POI is correct
            assert row['poi'] == poi

            try:
                list_dict[context].append(np.load(features_file))
            except:
                print('ERROR with ref file: ', poi, context, features_file)

        ## Concatenate the list into an array
        for k in list_dict:
            list_dict[k] = np.concatenate(list_dict[k], 0)
        print('DONE Loading ' + poi, flush=True)


        def parralel_distance(input):
            videoname, row = input
            context = row['context']
            ## Double check the poi is correct
            assert row['poi'] == poi

            ## Default value is NaN
            value = np.nan

            ## load feature file if exists
            features_file = get_features_filename(videoname)
            if os.path.isfile(features_file):
                try:
                    features = np.load(features_file)
                    ## Get all the features with different contexts from the test audio
                    references = np.concatenate([list_dict[k] for k in list_dict if k != context], 0)

                    if opt.strategy == 'ms':
                        value = np.min(np.sum(np.square(features[:1, :] - references), -1), -1)
                    elif opt.strategy == 'cb':
                        references = np.mean(references, 0)
                        value = np.sum(np.square(features[0, :] - references), -1)

                except:
                    traceback.print_exc()
            else:
                print(f'feature file {features_file} does not exist!')
            return videoname, value


        with ctx.Pool(opt.num_threads) as pool:
            out = pool.imap_unordered(parralel_distance, df_tst_poi.iterrows())
            for video, value in tqdm(out, total=len(df_tst_poi), desc='Testing ' + poi):
                df_tst.loc[video, 'value'] = value

        print('DONE Testing ' + poi, flush=True)

    ## Save result to file
    df_tst.to_csv(os.path.join(opt.scores_folder, f"{opt.dataset_name}_{opt.strategy}.csv"))
