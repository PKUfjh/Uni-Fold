# How to run the inference code
All the inference scripts are in `inference_scripts` directory. All scripts are run on GPU by default, if you want to run the scripts on cpu, you should set the flag by `--device "cpu"`.
### denoise mode
* `run_ufconf_denoise.py`: Runs the full backward process.

The running parameters (`num_replicas`, `protein ID`...) are defined in the JSON file specified by `-t` argument.  The model (`.pt` checkpoint) used for inference is specified by `-c`. 
In default the script will be run with the input fasta file (will download MSA if needed), the fasta file path is specified by the argument `-i` or `--input_pdbs`. The fasta file name is specified by the `fasta` param in the JSON input. Other than the fasta file as input, the script can also accept `pdb` and `cif` files as input. If you provide the input structure in `.pdb` format, the argument `--from_pdb` should be set, and the input pdb file name should be specified in the `pdb` param in the JSON input. If you provide the input structure in `.cif` format, the argument `--from_cif` should be set.

To run the script,
if provide the `.fasta` file as input:
```bash
python run_ufconf_denoise.py -t example_ufconf/1ake_from_fasta.json -i input_fastas/ -c checkpoint.pt -o ./ufconf_out
```
If provide the `.pdb` file as input:
```bash
python run_ufconf_denoise.py -t example_ufconf/1ake_from_pdb.json -i input_pdbs/ -c checkpoint.pt -o ./ufconf_out --from_pdb
```
If provide the `.cif` file as input:
```bash
python run_ufconf_denoise.py -t example_ufconf/1ake_from_pdb.json -i input_pdbs/ -c checkpoint.pt -o ./ufconf_out --from_cif
```

If you are in machines without web access, execute the msa generation code first
```bash
python extract_msa_fastas.py -t example_ufconf/1ake_from_fasta.json -i input_fastas/ -o ./ufconf_out
```

### langevin mode
* `run_ufconf_langevin.py`:  Runs langevin dynamics from a predefined diffusion timestep in the JSON file (MD-like). This is to test the *correlated sampling* of the model besides diffusion dynamics, which is *uncorrelated sampling*.  

To run the script:
```bash
python run_ufconf_langevin.py -t example_ufconf/1ake_from_pdb.json -i input_pdbs/ -c checkpoint.pt -o ./ufconf_out
```

### interpolation mode
* `run_ufconf_interpolate.py`: Runs the interpolation inference between 2 pdbs (from A -> B) where the order is defined in the JSON input file.

To run the script:
```bash
python run_ufconf_interpolate.py -t example_ufconf/1ake_4ake_inter.json -i input_pdbs/ -c checkpoint.pt -o ./ufconf_out
```

## guidance mode
```bash
python run_ufconf_denoise.py -t example_ufconf/O00148.json -i ./input_fastas -c checkpoint.pt -o ./ufconf_out --use_guidance
 ```

# Installation

## Create new conda enviroment
```bash
conda create -n ufconf python=3.9
```

## Based on your cuda version, install Pytorch (Using china image if necessary)
```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Clone the Uni-core repository and install
```bash
pip install -r requirements.txt
python setup.py install
```

## Clone Uni-Fold repository and install
```bash
pip install .
```

## Install other packages for result postprocessing
```bash
pip install MDAnalysis matplotlib
pip install biopython==1.81
conda install mdtraj -c conda-forge
```

# Input preparation
## task configuration
The json file of task configuration is a dictionary of dictionaries.

The outer dictionary is of items indexed by (unique) task ids.

Available keyword arguments are below.

```python
{
    "residue_idx": "1-140/142-281/283-427/429-573",
    # global index of residues that one wants to sample in ufconf, start from 0, using "/" to separate regions, index is across all chains
    "fasta": "1AEL-12_A", (optional)
    # fast file name of the system.
    # the fast file must exist at path `<input_pdbs>/<fasta>.fa`.
    "pdb": "1AEL-12_A",
    # pdb name of one conformation of a protein.
    # a pdb file must exist at path `<input_pdbs>/<pdb>.pdb`.
    "other_pdb": "1URE-8_A", (optional)
    # pdb name of the other conformation of a protein.
    "initial_t": 1.,
    # float in (0, 1], specifies the start of the diffusion process. 
    # with t=1 the pure prior is used; 
    # with t<1, noise is injected (the lower t, the lower noise ratio).
    "num_replica": 10,
    # repeating times of the job (different seeds).
    "inf_steps": 50,
    # number of reverse-diffusion steps.
    # notably, [`initial_t`, 0.] is uniformly splitted into `int_steps`+1 grids.
    "num_steps":10, (optional)
    # number of langevin steps if using langevin sampling mode,
    # number of interpolation steps if using interpolation mode.
    "save_trajectory": True,
    # whether to save the trajectory. Setting `False` is very slightly faster.
}
```

