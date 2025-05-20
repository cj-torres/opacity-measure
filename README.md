# Orthographic Opacity Measurement

This repository contains code and data for measuring orthographic opacity across different languages and writing systems. Orthographic opacity refers to the complexity and unpredictability of the relationship between spelling (orthography) and pronunciation (phonology) in a language.

## Project Overview

This project implements a computational approach to quantify orthographic opacity using mutual algorithmic information. The method analyzes how predictable the mapping is between:
- Orthography to phonology (spelling to pronunciation)
- Phonology to orthography (pronunciation to spelling)

The higher the mutual algorithmic information, the more opaque (less transparent) the writing system.

## Data Sources

The project uses several data sources:
- **Leipzig Corpora**: A collection of word frequencies for many languages
- **WikiPron**: Pronunciation data for words in multiple languages

## Repository Structure

- `opacity-measurement.py`: Main Python script implementing the neural network models and opacity measurement algorithms
- `plot-creation.R`: R script for data visualization and statistical analysis
- `compression-by-time.R`: R script for analyzing compression over time
- `sig_table.R`: R script for generating significance tables
- `leipzig-corpora/`: Directory containing language corpora
- CSV files: Results from CNN models for different languages

## Requirements

### Python Dependencies
- PyTorch
- pandas
- tqdm
- numpy

### R Dependencies
- tidyverse

## Usage

### Measuring Opacity

To reproduce paper results run

```bash
python opacity-measurement.py -f granular_phon_cnn_small_1layer_ker3_sanity.csv -n 40 -type phon -typ 5000 -tok 25000 -set leipzig -sz 32 -mod cnn -o True -e 25 -tar 300K -ly 1 -ker 3
python opacity-measurement.py -f granular_orth_cnn_small_1layer_ker3_sanity.csv -n 40 -type orth -typ 5000 -tok 25000 -set leipzig -sz 32 -mod cnn -o True -e 25 -tar 300K -ly 1 -ker 3
```

### Analyzing Results

After generating measurements, use the R scripts to analyze and visualize the results:

```bash
Rscript plot-creation.R
```

