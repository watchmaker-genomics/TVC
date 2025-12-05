# TAPS+ Variant Caller (TVC)

**TVC** is a germline variant caller purpose-built for **TAPS+** data. It is **CpG-aware**, **read-oriented**, and designed to factor out subtle methylation-induced patterns in order to avoid overcalling noise.

TVC performs variant calling while accounting for **CpG context** — both in the reference genome and in the observed read data. The algorithm models general noise using a **binomial distribution**, then estimates the most likely genotype (homozygous reference, heterozygous, or homozygous alternate) given observed allele counts and error rate parameters. This is inspired by the algorithm for determining genotypes in CpG that is used in [Rastair](https://bitbucket.org/bsblabludwig/rastair/src/master/) which was developed Benjamin Schuster-Böckler's lab at the University of Oxford, Ludwig Institute for Cancer Research.

---

## Installation

You can obtain prebuilt binaries or container images from our official distribution channels:

## Option 1: Download Binaries
Precompiled binaries for macOS, Linux are available under the [**Releases**](https://github.com/watchmaker-genomics/TVC/releases) section of this repository.

## Option 2: Pull from Amazon ECR Public

The tool is also available as an amd64-only container image on Amazon ECR.

## Option 3: Build from Dockerfile

You can build your own docker container by navigating to the top level directory with the docker file and running `docker build -t tvc ./`

### Pull the latest image
docker pull public.ecr.aws/e5r9o8m6/watchmakergenomics/tvc:latest

### Run directly
docker run -it --rm public.ecr.aws/e5r9o8m6/watchmakergenomics/tvc:latest -h

> Note: the `-it` allows for the progress bar to become visible when run. 


---

## Assumptions and Input Requirements

- TVC assumes diploid genotypes.  
- TVC assumes that the orientation of either R1 or R2 is in the same orientation of the original molecule. This means adapters that do not preserve orientation such as AB adapters are not supported.
- Reference and BAM files must be indexed (`.fai` and `.bai` respectively) and bams must contain NM tag entries, which come by default with aligners such as BWA-MEM and Bowtie2.

## Usage

```
tvc [OPTIONS] <INPUT_REF> <INPUT_BAM> <OUTPUT_VCF>
```

### Arguments

| Argument | Description |
|-----------|-------------|
| `<INPUT_REF>` | Reference FASTA file |
| `<INPUT_BAM>` | Aligned and indexed BAM file |
| `<OUTPUT_VCF>` | Output VCF path |

### Options

| Option | Description | Default |
|---------|-------------|----------|
| `-b, --min-bq <MIN_BQ>` | Minimum base quality for considering a read position | `20` |
| `-m, --min-mapq <MIN_MAPQ>` | Minimum mapping quality | `1` |
| `-d, --min-depth <MIN_DEPTH>` | Minimum depth required to make a genotype call | `2` |
| `-e, --end-of-read-cutoff <END_OF_READ_CUTOFF>` | Ignore bases within this distance from the end of the read for SNP calling | `5` |
| `-i, --indel-end-of-read-cutoff <INDEL_END_OF_READ_CUTOFF>` | Ignore bases within this distance from read ends when calling indels | `20` |
| `-x, --max-mismatches <MAX_MISMATCHES>` | Maximum allowed mismatches in an alignment for inclusion | `10` |
| `-a, --min-ao <MIN_AO>` | Minimum alternate observations required for a variant call | `2` |
| `-t, --num-threads <NUM_THREADS>` | Number of threads to use | `4` |
| `-c, --chunk-size <CHUNK_SIZE>` | Number of reference bases processed per thread batch | `1,000,000` |
| `-p, --error-rate <ERROR_RATE>` | Estimated per-base sequencing error rate used for binomial modeling | `0.005` |
| `-r, --stranded_read <READ_NUMBER>` | The read that is in the same orientation of the molecule (r1/r2) | `r1`|
| `-l, --log-level <LOG_LEVEL>`| The desired level of logging can be [error, warn, info, debug, trace]| `info`|
| `-h, --help` | Print help message | — |

---

## Algorithm Summary

1. **Context Awareness:**  
   Each site is evaluated with respect to its **CpG context**, allowing the model to distinguish true C→T mutations from transitions due to TAPS+ modification or background noise.


2. **Noise Modeling:**  
   Observed alternate allele counts are tested against a **binomial distribution** parameterized by the estimated error rate default value of error is 0.005 and using an alpha of 0.05.  
   This step filters out unlikely allelic configurations arising purely from noise.

3. **Genotype Likelihood Estimation:**  
   For positions passing initial filters, TVC computes the likelihoods for:  
   - Homozygous reference (0/0)  
   - Heterozygous (0/1)  
   - Homozygous alternate (1/1)  
   and assigns the most probable genotype per site.


---

## Example

```
tvc \
    -t 8 \
    reference.fa \
    sample.bam \
    output.vcf
```

## Benchmarking TVC and other TAPS+ materials 

Please see how we calculate F1 scores and it's current performance on a demo data set in [this notebook](analysis/TVC_benchmarking.ipynb).

We have other methylation based scripts in our [taps foundry](https://github.com/watchmaker-genomics/taps-foundry)


## Development

### Development requirements

All pull requests to this repository are gated by the unit test passing, a format check with cargo fmt, and finally linting with cargo clippy.

### Git LFS Requirement

This project uses [Git LFS](https://git-lfs.github.com/) (Large File Storage) for handling large files specifically for testing assets.  

Before starting development, ensure Git LFS is installed on your system.

#### Install Git LFS

- **macOS (Homebrew):**
    ```bash
    brew install git-lfs
    ```
- **Linux (Debian/Ubuntu):**
    ```bash
    sudo apt install git-lfs
    ```

#### Initialize Git LFS
After installation, run:
```bash
git lfs install
```

#### Pulling LFS Files
When cloning the repository, make sure to fetch LFS files as well:
```bash
git clone git@github.com:watchmaker-genomics/TVC.git
cd TVC
git lfs pull
```

For existing repositories:
```bash
git pull
git lfs pull
```

> **Note:** Development cannot proceed without Git LFS installed, as some large files required for the project are managed through LFS.

# License and Use
This project is licensed under the LGPL 3.0 License.

- Attribution is required: please retain the copyright notice and license in any redistributed versions or substantial portions of this software.
- All software is provided “AS IS”, without warranties or conditions of any kind.

## Important Notes

- **Research Use Only**: This code is provided for research and development purposes. It is not validated for diagnostic or clinical use, and Watchmaker Genomics makes no representations about regulatory compliance.
- **No Warranty**: Use of this code is at your own risk. Watchmaker Genomics shall not be liable for any claims or damages arising from its use.
- **Contributions**: If you contribute code, you agree that your contributions will be licensed under the same LGPL 3.0 License.

By making a contribution to this project, I certify that:

(a) The contribution was created by me and I have the right to submit it under the LGPL 3.0 License; or 
(b) The contribution is based upon previous work that is covered by the LGPL 3.0 License and I have the right to submit it under the same license; or 
(c) The contribution was provided to me by someone who certified (a) or (b).

I understand and agree that this project and the contribution are public and that a record of the contribution (including my name and email) is maintained indefinitely.
