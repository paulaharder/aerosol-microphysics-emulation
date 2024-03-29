# NeuralM7: Physics-Informed Learning of Aerosol Microphysics

If you using our code, consider citing our preprint:
```sh
@misc{harder2022,
  url = {https://arxiv.org/abs/2207.11786}, 
  author = {Harder, Paula and Watson-Parris, Duncan and Stier, Philip and Strassel, Dominik and Gauger, Nicolas R. and Keuper, Janis},
  title = {Physics-Informed Learning of Aerosol Microphysics},
  year = {2022},
}
```

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

To run the correction model (guarenteed positive predictions) run

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

If you are using the old dataset (https://zenodo.org/record/5837936) use --old_data True

#### Run inference

Run inference for the baseline model with

```sh
$ python main.py --mode eval --model_id standard_test
```




