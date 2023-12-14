
# PoIForensics-Audio

[![Paper](https://img.shields.io/badge/-Paper-B31B1B.svg?style=for-the-badge)](https://doi.org/10.1109/WIFS55849.2022.9975428)
[![GRIP](https://img.shields.io/badge/-GRIP-0888ef.svg?style=for-the-badge)](https://www.grip.unina.it)

Code repository for the paper [Deepfake audio detection by speaker verification](https://doi.org/10.1109/WIFS55849.2022.9975428)

This repo containes the code for the PoIForensics method.

## Requirements
This repo has been tested with python 3.11. 

To install the necessaries libraries just run ```pip install -r requirements.txt```.

## Extract Features

The whole pipeline consists of two steps. Feature extraction is the first one. To extract the features you will need a csv file formatted in a certain way and the weights of the model. In the [CSVs](./CSVs/) folder you will find the csv files we used in the paper that you can use to replicate our experiments but if you want to use them yourself, you will need to fix the filepath value for each row. Weights are stored in the [checkpoints](./checkpoints/) folder.

As an example, to start extracting features on the InTheWild dataset using the model trained without augmentation, run:

```python extract_features.py --dataset-csv CSVs/InTheWild.csv --dataset-name InTheWild --weights checkpoints/model_no_augmentation.th```

This will create the folder ```./features``` and inside of it another folder called as the value of the parameter ```--dataset-name``` ( ```InTheWild``` in this case). Refer to the [extract_features.py](extract_features.py) file for the complete parameters declaration to customize the settings.

If you want to test on your own data, you will need to create a csv file in a similar format. Take extra care that the these columns are needed and should not be omitted:
1. **videoname** is the file name without the extension.
2. **filepath** is the absolute path to the file containing audio.
3. **poi** is a string declaring which identity this audio relates to.
4. **context** refers to the setting the audio was taken from. A real audio and a fake audio made from such video should have the same context. This way we avoid the bias of testing a fake video by using the real it was made from.
5. **label** 0 or 1, depending if the audio is real or fake.
6. **in_tst** 0 or 1, depending if the audio is in the test set or not.
7. **in_ref** 0 or 1, depending if the audio has to be used as a reference or not.

## Compute Distances

The second step is to compute the distances of the audios under test given the audios in the reference set. To compute these distances from the previous example using the Multi-Similarity strategy you can simply run:

```python compute_distances.py --dataset-csv CSVs/InTheWild.csv --dataset-name InTheWild --strategy ms```

A single csv file will be saved in the ```./scores/``` folder where for each audio file there will be a value of the distance from the reference set. Refer to the [compute_distances.py](compute_distances.py) file for the complete parameters declaration.

## Visualize results

You can use the [Visualize.ipynb](Visualize.ipynb) notebook to compare all the different experiments saved in the scores folder

## Acknowledgment

We gratefully acknowledge the support of this research by the Defense Advanced Research Projects Agency (DARPA) under agreement number FA8750-20-2-1004. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright notation thereon. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of DARPA or the U.S. Government. This work is also co-funded by the European Union under the Horizon Europe vera.ai project, Grant Agreement number 101070093, and is supported by the PREMIER project, funded by the Italian Ministry of Education, University, and Research within the PRIN 2017 program.

## Paper citation

If you use our code, please remember to cite us using the data below:

```
@inproceedings{pianese2022deepfake,
  title={Deepfake audio detection by speaker verification},
  author={Pianese, Alessandro and Cozzolino, Davide and Poggi, Giovanni and Verdoliva, Luisa},
  booktitle={2022 IEEE International Workshop on Information Forensics and Security (WIFS)},
  pages={1--6},
  year={2022},
  organization={IEEE}
}
```

