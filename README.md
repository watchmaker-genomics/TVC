# Taps Variant Caller (TVC)

**TVC** is a germline variant caller purpose-built for **TAPS (TET-assisted pyridine borane sequencing)** data.  
It is **CpG-aware**, **read-oriented**, and designed to factor out subtle methylation-induced patterns in order to avoid overcalling noise.

---

## Overview

TVC performs variant calling while accounting for **CpG context** — both in the reference genome and in the observed read data.  
The algorithm models general noise using a **binomial distribution**, then estimates the most likely genotype (homozygous reference, heterozygous, or homozygous alternate) given observed allele counts and error rate parameters. This is inspired by the algorithm used in [Rastair](https://bitbucket.org/bsblabludwig/rastair/src/master/) .

---

## Assumptions and Input requirements

- TVC assumes diploid genotypes.  
- TVC assumes that the orientation of either R1 or R2 is in the same orientation of the original molecule. This means adapters that do not preserve orientation such as AB adapters are not supported.
- Reference and BAM files must be indexed (`.fai` and `.bai` respectively) and bams must contain NM tag entries.

## Usage

```
taps_variant_caller [OPTIONS] <INPUT_REF> <INPUT_BAM> <OUTPUT_VCF>
```

### Arguments

| Argument | Description |
|-----------|-------------|
| `<INPUT_REF>` | Reference FASTA file |
| `<INPUT_BAM>` | Aligned BAM file (paired-end) |
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
| `-r, --stranded_read <READ_NUMBER>` | The read that is in the same orientation of the molecule (R1/R2) | `R1`|
| `-h, --help` | Print help message | — |

---

## Algorithm Summary

1. **Context Awareness:**  
   Each site is evaluated with respect to its **CpG context**, allowing the model to distinguish true C→T mutations from transitions due to TAPs modification or background noise.

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
taps_variant_caller \
    -t 8 \
    reference.fa \
    sample.bam \
    output.vcf

## Development

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
- **Windows:**
    Download and run the installer from [https://git-lfs.com](https://git-lfs.com).

#### Initialize Git LFS
After installation, run:
```bash
git lfs install
```

#### Pulling LFS Files
When cloning the repository, make sure to fetch LFS files as well:
```bash
git clone <repo_url>
cd <repo_name>
git lfs pull
```

For existing repositories:
```bash
git pull
git lfs pull
```

> **Note:** Development cannot proceed without Git LFS installed, as some large files required for the project are managed through LFS.

