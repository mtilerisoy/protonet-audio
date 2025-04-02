# protonet-audio
ProtoNet Application for *Speech Commands Dataset* using *Audio Spectrogram Transformer* as the backbone model.

## Environment Setup
Run the following code snippet to create the conda environment using the *environment.yml*. Don't forget to change the **prefix** section at the bottom of the file: ```conda env create -f environment.yml```

## Codebase Structure

**dataset :** The folder that contians the *SpeechCommand* daataset (ignored by default and willbe automatically created).

**models :** The cache directory used to download the model through the Huggingface *transformers* library.

**data_explore.ipynb :** The jupyter notebook used to analyse the dataset.

**train.ipynb :** The notebook that implements the ProtoNet training using *Audio Spectrogram Transformer* as the backbone.

**environment.yml :** The yml file containing the conda environment information.

## Reproduction

The train.ipynb notebook is the main notebook. It implements and end-to-end training of the Audio Spectrogram Model using a Prototypical Learning approach.

One can adjust the hyperparameters under the configuration cell (cell number 2) to test different settings. Since the local development environment is limited on resources the number of episodes and number of classes included was limited during training.

The notebook is currently running on *cpu* because some operations aren't implemented for the *mps backend* which is the local development backend (macOS). The backward operation throws te following error:

```NotImplementedError: The operator 'aten::_cdist_backward' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://github.com/pytorch/pytorch/issues/141287 and mention use-case, that resulted in missing op as well as commit hash Unknown. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.```

## Results

After the training, the model is evaluated on the test split of the Speech Commands dataset. The trained model has an accuracy of **68.67%** while the initial model had **30.72%**, both evaluated on 5-way, 5-shot, and 15-query setting.