# Physics-Informed Learning of Aerosol Microphysics

## How to run the code

Clone the repository and install the requirements
```sh
$ git clone https://github.com/paulaharder/aerosol-microphysics-emulation.git
$ conda env create -f requirements.yml
$ conda activate aerosol-emulation
```

#### Data download

The data is available at https://zenodo.org/record/5837936, to get it directly on your server use wget:
```sh
$ mkdir data
$ wget https://zenodo.org/record/5837936/files/aerosol_emulation_data.zip
```

then unzip
```sh
$ unzip -o aerosol_emulation_data.zip -d data/
$ rm aerosol_emulation_data.zip
```

#### Run training

```sh
$ main.py
```
