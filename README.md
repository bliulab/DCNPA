# DCNPA

<div align="center">

</div>

Peptide-protein interactions (PepPIs), driven by specific residue-level contacts, are fundamental to cellular regulation and serve as pivotal targets for therapeutic development. Despite the substantial progress of deep learning in PepPI prediction, current methods encounter two critical bottlenecks: poor generalization to unseen targets due to reliance on static patterns, and the oversight of the complex macromolecular environment involving multimers. Here we present **Dynamic Context Networks for peptide-protein Pragmatic Analysis (DCNPA)**, a framework that models peptide-protein multimeric interaction from a dynamic contextual perspective. DCNPA integrates evolutionary priors and environmental constraints through the target-adaptive and multimer-aware dynamic context sub-networks, respectively. By fusing these dynamic contexts with intrinsic molecular features, DCNPA achieves robust residue-level interaction prediction under multi-granularity supervision. Applied to diverse benchmarks, DCNPA demonstrates strong generalization to novel targets and precise identification of binding interfaces, while revealing a multimer-mediated regulatory mechanism involving a transition from entropy-driven to enthalpy-dominated recognition. In downstream peptide drug virtual screening, DCNPA predictions show high concordance with wet-lab experimental results, highlighting its practical potential.

DCNPA is available as a web-based demonstration (http://bliulab.net/DCNPA), whereas **the full target-adaptive dynamic context subnetwork requires local deployment**. Users may select the optimal deployment mode based on their specific needs and computational resources.

![Model](/imgs/Model.png)

**Fig. 1: The framework of DCNPA.** Multimodal features of peptides, proteins, and environmental contexts are integrated via dynamic context networks to enable pragmatic analysis. a, Local representations of peptides, proteins, and environmental molecules are independently encoded via TextCNN and GraphSAGE at the residue level, then integrated by a multiscale transformer for information fusion. b, The target-adaptive dynamic context sub-network utilizes DeepBLAST to retrieve homologous sequences, which are structurally aligned via a residue-level alignment module to dynamically extract consensus template features as evolutionary priors. c, The multimer-aware dynamic context sub-network employs dual-branch attention mechanisms to model and encode multimeric constraints exerted by environmental molecules. d, Finally, a multi-view interaction modeling and fusion strategy integrates intrinsic molecular features with dynamic evolutionary and environmental contexts to predict the resultant peptide-protein interaction map.


# 1 Installation

## 1.1 Create conda environment

```
conda create -n dcnpa python=3.11
conda activate dcnpa
```

## 1.2 Requirements
We recommend installing the environment using the provided `environment.yaml` file to ensure compatibility:
```
conda env update -f environment.yaml --prune
```


If this approach fails or Conda is not available, you can manually install the main dependencies as listed below:
```
python  3.11
biopython 1.85
huggingface-hub 0.29.2
numpy 2.1.2
transformers 4.49.0
tokenizers 0.21.0
torch 2.6.0+cu118
torchaudio 2.6.0+cu118
torchvision 0.21.0+cu118
torch-geometric 2.6.1
```

> **Note** If you have an available GPU, the accelerated DCNPA can be used to predict peptide-protein pair-specific binding residues. Change the URL below to reflect your version of the cuda toolkit (cu118 for cuda=11.6 and cuda 11.8, cu121 for cuda 12.1). However, do not provide a number greater than your installed cuda toolkit version!
> 
> ```
> pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
> ```
>
> For more information on other cuda versions, see the [pytorch installation documentation](https://pytorch.org/).

## 1.3 Tools
Feature extraction tools and databases on which DCNPA relies: 
```
SCRATCH-1D 1.2
IUPred2A \
ncbi-blast 2.13.0
ProtT5 \
trRosetta \
DeepBLAST \
```

Databases and model:
```
nrdb90 [ncbi-blast Database](http://bliulab.net/DCNPA/static/download/nrdb90.tar.gz)
uniclust30_2018_08 [HHsuite sequence Database](http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08)
model_res2net_202108 [Pre-trained network models of trRosetta](https://yanglab.qd.sdu.edu.cn/trRosetta/download/)
DeepBLAST pretrained model [deepblast-v3.cpt](https://figshare.com/s/e414d6a52fd471d86d69)
```

**The default paths to all tools and databases are shown in `conf.py`. You can change the paths to the tools and databases as needed by configuring `conf.py`.**

`SCRATCH-1D`, `IUPred2A`, `ncbi-blast`, `ProtT5`, and `trRosetta` are recommended to be configured as the system envirenment path. Your can follow these steps to install them:

### 1.3.1 How to install SCRATCH-1D
Download (For linux, about 6.3GB. More information, please see **https://download.igb.uci.edu/**)
```
wget https://download.igb.uci.edu/SCRATCH-1D_1.2.tar.gz
tar -xvzf SCRATCH-1D_1.2.tar.gz
```

Install
```
cd SCRATCH-1D_1.2
perl install.pl
```

> **Note:** The 32 bit linux version of blast is provided by default in the 'pkg' sub-folder of the package but can, probably should, and in some cases has to be replaced by the 64 bit or Mac OS version of the blast software for improved performances and compatibility on such systems.


Finally, test the installation of SCRATCH-1D
```
cd <INSTALL_DIR>/doc
../bin/run_SCRATCH-1D_predictors.sh test.fasta test.out 4
```

> **Note:** If your computer has less than 4 cores, replace 4 by 1 in the command line above.

**In addition, users need to move the `SCRATCH.sh` script in `/tools/SCRATCH-1D_1.2/` to the `bin/` directory under the SCRATCH-1D_1.2 installation path for DCNPA to call.**

### 1.3.2 How to install IUPred2A
For download and installation of IUPred2A, please refer to **https://iupred2a.elte.hu/download_new**. It should be noted that this automation service is **only applicable to academic users.** For business users, please contact the original authors for authorization. 

After obtaining the IUPred2A software package, decompress it.
```
tar -xvzf iupred2a.tar.gz
```

Finally, test the installation of IUPred2A
```
cd <INSTALL_DIR>
python3 iupred2a.py P53_HUMAN.seq long
```

**In addition, users need to move the `iupred2a.sh` script in `/tools/iupred2a/` to the IUPred2A installation path for DCNPA to call. Please also make sure to modify the path to the Python script (`iupred2a.py`) within the `iupred2a.sh` file to reflect the absolute path of your IUPred2A installation. This is necessary for the script to function correctly.**

### 1.3.3 How to install ncbi-blast
Download (For x64-linux, about 220M. More information, please see **https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html**)
```
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.13.0/ncbi-blast-2.13.0+-x64-linux.tar.gz
tar -xvzf ncbi-blast-2.13.0+-x64-linux.tar.gz
```
Add the path to system envirenment in `~/.bashrc`.
```
export BLAST_HOME={your_path}/ncbi-blast-2.13.0+
export PATH=$PATH:$BLAST_HOME/bin
```

Finally, reload the system envirenment and check the ncbi-blast command:
```
source ~/.bashrc
psiblast -h
```

> **Note:** The purpose of DCNPA with the help of ncbi-blast is to extract the position-specific scoring matrix (PSSM). It should be noted that for sequences that cannot be effectively aligned, the PSSM is further extracted by blosum62 (which can be found in [`blosum62.txt`](http://bliulab.net/DCNPA/static/download/blosum62.txt)).


### 1.3.4 How to install ProtT5 (ProtT5-XL-UniRef50)
Download and install (More information, please see **https://github.com/agemagician/ProtTrans** or **https://zenodo.org/record/4644188**, about 5.3GB)

```
wget https://zenodo.org/records/4644188/files/prot_t5_xl_uniref50.zip?download=1
unzip prot_t5_xl_uniref50.zip
```


### 1.3.5 How to install trRosetta
Download `trRosettaX.tar.bz2` from **https://yanglab.qd.sdu.edu.cn/trRosetta/download/** (about 94MB)

```
tar -xvjf trRosettaX.tar.bz2
cd trRosettaX/
```

Then, refer to the README file to install and configure it. **In addition, users need to move the `generate_msa.sh` and `predict.sh` scripts in `/tools/trRosettaX/` to the trRosettaX installation path for DCNPA to call.**
Please also ensure the following modifications are made for proper execution:

**1) Conda Environment:**
Both `generate_msa.sh` and `predict.sh` need to run under the Conda environment required by trRosettaX.
This can be achieved by modifying the `export PATH=...` line in each script to point to the bin directory of the corresponding Conda environment.

**2) Database Path Configuration:**

In `generate_msa.sh`, set the `-hhdb` parameter to the absolute path of the uniclust30_2018_08 database.

In `predict.sh`, set the `-mdir` parameter to the absolute path of the model_res2net_202108 model directory.

These changes are necessary to ensure DCNPA can correctly invoke trRosetta for structure prediction.


## 1.4 Install DCNPA
To install from the development branch run
```
git clone git@github.com:bliulab/DCNPA.git
cd DCNPA/
```

**Finally, configure the defalut path of the above tool and the database in `conf.py`. You can change the path of the tool and database by configuring `conf.py` as needed.**


# 2 Usage
It takes 2 steps to predict peptide-protein binary interaction and peptide-protein-specific binding residues:

(1) Replace the default peptide sequence in the `example/Peptide_Seq.fasta` file with your peptide sequence (FASTA format). Similarly, replace the default protein sequence in the `example/Protein_Seq.fasta` file with your protein sequence (FASTA format). If you don't want to do this, you can also test your own peptide-protein pairs by modifying the paths to the files passed in by the `run_predictor.py` script (the parameter is `-uip`, respectively).

(2) Then, run `run_predictor.py` to make prediction, including **pairwise non-covalent interaction** prediction and **non-covalent bond type** identification. It should be noted that `run_predictor.py` automatically calls the scripts `FeatureExtract.py`, and `PSSMExtract.py` to generate the multi-source isomerization features of peptides and proteins.
```
conda activate dcnpa
python run_predictor.py -uip example
```

If you want to retrain based on your private dataset, find the original DCNPA model in `model.py`. The DCNPA source code we wrote is based on the Pytorch implementation and can be easily imported by instantiating it.

# 3 Problem feedback
If you have questions on how to use DCNPA, feel free to raise questions in the [discussions section](https://github.com/bliulab/DCNPA/discussions). If you identify any potential bugs, feel free to raise them in the [issuetracker](https://github.com/bliulab/DCNPA/issues).

In addition, if you have any further questions about DCNPA, please feel free to contact us [**stchen@bliulab.net**]

# 4 Citation

If you find our work useful, please cite us at
```
@article{
  title={Dynamic context networks integrating evolutionary priors and environmental constraints for deciphering peptide-protein multimeric interaction mechanisms},
  author={Shutao Chen, Ke Yan, Tianqi Hu, Hongjun Yu, and Bin Liu},
  journal={submitted},
  year={2026},
  publisher={}
}

```
