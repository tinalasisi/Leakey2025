# Primate Hair Trait Evolution - Leakey 2025

This repository contains data and analyses for studying the phylogenetic evolution of hair traits in primates, with a focus on sexual dimorphism and ontogenetic trajectories of hair color development.

## Project Overview

This project investigates how hair color traits evolve across primate species, examining:
- Sexual dimorphism in hair color
- Natal coat characteristics
- Ontogenetic trajectories (developmental changes in hair color)
- Phylogenetic patterns of trait evolution

## Repository Structure

- `analysis/` - R Markdown files containing phylogenetic comparative analyses
- `code/` - Command-line scripts and shared R code
- `data/` - Primate hair trait datasets and phylogenetic trees
  - Contains OpenApePose image dataset for computer vision analysis (large files ignored in git)
- `docs/` - Rendered website and documentation
- `output/` - Analysis results and session data

## Methods

The project uses phylogenetic comparative methods including:
- Mk models for discrete trait evolution
- Ancestral state reconstruction
- Transition rate analysis between ontogenetic states

## Dependencies

- R packages: ape, phytools, tidyverse, ggtree, workflowr
- For computer vision components: OpenMMLab, MMdetect, MMpose
