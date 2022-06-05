# NeuralM7: Physics-Informed Learning of Aerosol Microphysics

## How to run the code

Clone the repository and install the requirements
```sh
$ git clone https://github.com/paulaharder/aerosol-microphysics-emulation.git
$ conda env create -f requirements.yml
$ conda activate aerosol-emulation
```

#### Data download

The data is available at https://zenodo.org/record/6583397, to get it directly on your server use wget:
```sh
$ mkdir data
$ wget https://zenodo.org/record/6583397/files/aerosol_emulation_data.zip
```

then unzip
```sh
$ unzip -o aerosol_emulation_data.zip -d data/
$ rm aerosol_emulation_data.zip
```

#### Run training

Run the baseline model (no mass conservation and positivity enforcement) with

```sh
$ python main.py --model_id standard_test
```

To run the completion model (guarenteed mass conservation) run

```sh
$ python main.py --model completion --model_id completion_test
```

There is a bug for this case, will be fixed soon: To run the correction model (guarenteed positive predictions) run

```sh
$ python main.py --model correction --model_id correction_test
```

To run the model with mass loss term run

```sh
$ python main.py --loss mse_mass --model_id mass_loss_test
```

To run the model with positivity loss term run

```sh
$ python main.py --loss mse_positivity --model positivity --model_id positivity_loss_test
```

To train on logarithmically transformed variables you first need to run the classification network

```sh
$ python main.py --model classification --scale log --model_id class_test
```

then

```sh
$ python main.py --scale log --model_id log_test
```

#### Run inference

Run inference for the baseline model with

```sh
$ python main.py --mode eval --model_id standard_test
```




