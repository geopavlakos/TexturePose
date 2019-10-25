## Data preparation
Besides the demo code, we also provide training and evaluation code for our approach. To use this functionality, you need to download the relevant datasets.
The datasets that our code supports are:
1. [Human3.6M](http://vision.imar.ro/human3.6m/description.php)
2. [MPII video](http://human-pose.mpi-inf.mpg.de)
3. [MoSh data](https://github.com/akanazawa/hmr/blob/master/doc/train.md#mosh-data)
4. [VLOG-people & InstaVariety](https://github.com/akanazawa/human_dynamics)
5. [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)
6. [LSP](http://sam.johnson.io/research/lsp.html)
7. [UPi-S1h](http://files.is.tuebingen.mpg.de/classner/up/)

More specifically:
1. **Human3.6M**: Unfortunately, due to license limitations, we are not allowed to redistribute the MoShed data that we used for training. We only provide code to evaluate our approach on this benchmark. To download the relevant data, please visit the [website of the dataset](http://vision.imar.ro/human3.6m/description.php) and download the Videos, BBoxes MAT (under Segments) and 3D Positions Mono (under Poses) for Subjects S9 and S11. After downloading and uncompress the data, store them in the folder ```${Human3.6M root}```. The sructure of the data should look like this:
```
${Human3.6M root}
|-- S9
    |-- Videos
    |-- Segments
    |-- Bboxes
|-- S11
    |-- Videos
    |-- Segments
    |-- Bboxes
```
You also need to edit the file ```config.py``` to reflect the path ```${Human3.6M root}``` you used to store the data. 

2. **MPII Video**: We use the data from the video sequences from the MPII human pose dataset. You need to download the video sequences [here](http://human-pose.mpi-inf.mpg.de/#download). After you decompress the .tar.gz files, make a folder ```mpii_human_pose_v1_sequences``` with the subfolders for all the sequences. You will also need to download the [image-video mapping file](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_sequences_keyframes.mat) in the same folder. Finally, add the root path of the dataset in the ```config.py``` file.

3. **MoSh**: You need to download the [MoSh data](https://github.com/akanazawa/hmr/blob/master/doc/train.md#mosh-data) for the CMU dataset, provided by the authors of HMR. In case you use this data, please respect the license and cite the original [MoSh work](http://mosh.is.tue.mpg.de). After you unzip the .tar file, please edit ```config.py``` to include the path for this dataset.

4. **VLOG-people & InstaVariety**: These datasets can be used as an additional source of training data. We suggest that you download the preprocessed tfrecords for [VLOG-people](https://github.com/akanazawa/human_dynamics/blob/master/doc/vlog_people.md#pre-processed-tfrecords) and [InstaVariety](https://github.com/akanazawa/human_dynamics/blob/master/doc/insta_variety.md#pre-processed-tfrecords), provided by the authors of HMMR. After you extract the tfrecords from the compressed .tar.gz files, please complete the relevant root paths of the datasets in the ```config.py``` file.

5. **3DPW**: We use this dataset only for evaluation. You need to download the data from the [dataset website](https://virtualhumans.mpi-inf.mpg.de/3DPW/). After you unzip the dataset files, please complete the root path of the dataset in the file ```config.py```.

6. **LSP**: We use the LSP test set for evaluation. You need to download the low resolution version of [LSP dataset](http://sam.johnson.io/research/lsp_dataset.zip). After you unzip the dataset files, please complete the relevant root path of the dataset in the file ```config.py```.

7. **UPi-S1h**: We only need the annotations provided with this dataset to enable silhouette evaluation on the LSP dataset. You need to download the [UPi-S1h zip](http://files.is.tuebingen.mpg.de/classner/up/datasets/upi-s1h.zip). After you unzip, please edit ```config.py``` to include the path for this dataset.

### Generate dataset files
After preparing the data, we continue with the preprocessing to produce the data/annotations for each dataset in the expected format. With the exception of Human3.6M, we already provide these files and you can get them by running the ```fetch_data.sh``` script. If you want to generate the files yourself, you need to run the file ```preprocess_datasets.py``` from the main folder of this repo that will do all this automatically. Keep in mind that this assumes you have already run OpenPose on the images of all datasets used for training and you have provided the respective folders with the OpenPose ```.json``` files in the ```config.py``` folder. Depending on whether you want to do evaluation or/and training, we provide two modes:

If you want to generate the files such that you can evaluate our pretrained models, you need to run:
```
python preprocess_datasets.py --eval_files
```
If you want to generate the files such that you can train using the supported datasets, you need to run:
```
python preprocess_datasets.py --train_files
```
