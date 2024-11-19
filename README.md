# ScalerGAN : Speech Time-Scale Modification With GANs
# Nitzan and Gal First Runs
Eyal Cohen (cohen.eyal@campus.technion.ac.il)\
Felix Kreuk (felixkreuk@gmail.com)\
Joseph Keshet (jkeshet@technion.ac.il)

ScalerGAN is a software package for Time-Scale Modification (AKA speed-up or slow-down) of a given recording using a novel unsupervised learning algorithm.


The model was present in the paper [Speech Time-Scale Modification With GANs](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9747953) and the paper was published in [IEEE Signal Processing Letters](https://signalprocessingsociety.org/publications-resources/ieee-signal-processing-letters).

Audio examples can be found [here](https://eyalcohen308.github.io/ScalerGAN/).

If you use this code, please cite the following paper:
```
@article{cohen2022scalergan,
  author={Cohen, Eyal and Kreuk, Felix and Keshet, Joseph},
  journal={IEEE Signal Processing Letters},
  title={Speech Time-Scale Modification With GANs},
  year={2022},
  volume={29},
  pages={1067-1071},
  doi={10.1109/LSP.2022.3164361}}
  publisher={IEEE}
}
```

## Pre-requisites:
1. Python 3.8.5+
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](requirements.txt).
4. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/).
5. Create an input.txt file with the path to the audio files in the dataset.\
One can use the following command to create the input.txt file from the dataset:
```
$ ls <PATH/TO/LJ_dataset/DIR> | xargs realpath > ./data/input.txt
```
Or modify the [data/input.txt](data/input.txt) to point to the dataset files.
## Training:
### Quick Training (using default parameters):
```
$ python ScalerGAN/train.py
```
See [configs.py](ScalerGAN/configs/configs.py) for more options.

### Single GPU Training:
```
$ python ScalerGAN/train.py --input_file <PATH/TO/INPUT.txt> --output_dir <PATH/TO/OUTPUT/DIR> --device cuda
```

### Distributed training (multiple GPUs) using [torch.distributed.launch](https://pytorch.org/docs/stable/distributed.html):
```
$ python -m torch.distributed.launch --nproc_per_node=N train.py --device cuda --name multi_gpu --distributed --distributed_backend='nccl'
```
Modify N value in the nproc_per_node argument to match the number of GPUs available.

### Resume training from checkpoint:
```
$ python ScalerGAN/train.py --input_file <PATH/TO/INPUT.txt>  --device cuda --resume <PATH/TO/CHECKPOINT>
```
### Train with manual min and max time-scale factors:
```
$ python ScalerGAN/train.py --input_file <PATH/TO/INPUT.txt>  --device cuda --max_scale 0.5 --min_scale 2.0
```

## Inference:

One can modify the inference file [data/inference.txt](data/inference.txt) to the desired audio files or use '--inference_file' to specify infrenece txt file. The artifacts will be saved in the output directory.
### Quick Inference with mel-spectrogram output:
```
$ python ScalerGAN/inference.py --device cuda
```
### Quick Inference with mel-spectrogram and audio output:
```
$ python ScalerGAN/inference.py --device cuda --infer_hifi
```
The flag '--infer_scales' can be used to specify the scales for inference. If not specified, the model will infer the default scales [0.5, 0.7, 0.9, 1.1, 1.3, 1.5].
### Inference with specific checkpoint:

```
$ python ScalerGAN/inference.py --checkpoint_path <PATH/TO/CHECKPOINT> --device cuda
```

## Fine Tuning:
For fine tuning the ScalerGAN model on a different dataset, one can use the following command:
```
$ python ScalerGAN/train.py --input_file <PATH/TO/INPUT.txt>  --device cuda --fine_tune --checkpoint_path <PATH/TO/CHECKPOINT>
```

For fine tuning the HiFi-GAN Vocoder do the following:
1. Run ScalerGAN inference on the desired dataset.
```
$ python ScalerGAN/inference.py --input_file <PATH/TO/INPUT.txt>  --device cuda
```
* The cropped audio and the generated Mel spectrograms will be saved in the output directory, copy them to desired location by the HiFi-GAN Vocoder.
2. Follow the instructions in the [HiFi-GAN](https://github.com/jik876/hifi-gan) repository for fine tuning the model.
3. Use the generated HiFi-GAN checkpoint and config for inference with ScalerGAN, using the flag '--hifi_config' and '--hifi_checkpoint':
```
$ python ScalerGAN/inference.py --checkpoint_path <PATH/TO/SCALER_GAN/CHECKPOINT> --device cuda --infer_hifi --hifi_config <PATH/TO/HIFI_CONFIG> --hifi_checkpoint <PATH/TO/HIFI_CHECKPOINT>
```


## Please Note
* Prediction/ Training is significantly faster if you run on GPU.
* The model is trained on 22.05kHz audio files. If you run on a different sampling rate, the model will not work as expected.
* The model crops the audio files to be divisible by '--must_divide' to fit the model architecture. The default value is 8.
* The model is trained on 80 Mel bins. If you want to change it, you must change it in both the training and inference scripts.


## Acknowledgements
- This work was realized as part of the [Speech Processing and Learning Lab](https://keshet.net.technion.ac.il/) at the [Technion](https://www.technion.ac.il/en/).
- We referred to [HiFi-GAN](https://github.com/jik876/hifi-gan) to implement this.
