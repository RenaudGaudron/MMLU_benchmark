# MMLU Benchmark Evaluation Tool

This repository provides a streamlined tool for evaluating open-source and proprietary Large Language Models (LLMs) on the Massive Multitask Language Understanding (MMLU) benchmark.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Results structure](#results-structure)
- [License](#license)
- [Acknowledgements](#acknowledgments)

## Introduction

Evaluating Large Language Models (LLMs) on various benchmarks is crucial for understanding their capabilities and limitations. The MMLU benchmark, with its diverse range of subjects, is a widely recognised standard for assessing knowledge and reasoning. This project aims to provide an easy-to-use script to run MMLU evaluations, log performance metrics, and save results in a structured format.

## Features

This project provides an easy-to-use and standardised framework for evaluating both open-source and proprietary Large Language Models (LLMs) on the Massive Multitask Language Understanding (MMLU) dataset. It aims to streamline the benchmarking process, allowing users to efficiently assess model performance (accuracy) and inference efficiency (speed and VRAM utilisation) across various MMLU subjects. By offering a clear and reproducible methodology, this project facilitates comparative analysis of LLMs to better understand their capabilities and resource requirements for diverse applications.

Currently, the following types of models are supported: 

- Hugging Face `transformer` open-source models
- Bedrock models

Quantisation and batching are supported for the Hugging Face `transformer` open-source models.

## Results structure

Evaluation results will be saved in the `results/` folder. JSON files will store the accuracy, number of examples, execution time, used VRAM, and available VRAM for each evaluated MMLU subject.

The evaluation results will be stored in a JSON file with the following structure (not actual data):

```json
{
    "abstract_algebra": {
        "accuracy": 0.8235294117647058,
        "number_examples": 17,
        "execution_time": 364.404,
        "used_VRAM": 7934,
        "total_VRAM": 8188
    },
    "high_school_physics": {
        "accuracy": 0.625,
        "number_examples": 16,
        "execution_time": 840.058,
        "used_VRAM": 2224,
        "total_VRAM": 8188
    }
}
```

## License
- MIT License. 
- Copyright holder: Renaud Gaudron

## Acknowledgements
This project leverages the incredible work of various communities. We extend our sincere gratitude to:

* **Hugging Face:** For their invaluable `transformers` and `datasets` libraries, which are central to the LLM loading and data management within this benchmarking suite.
* **Amazon Web Services (AWS):** For their Bedrock service and the `boto3` library.
* **PyTorch:** For providing the foundational deep learning framework, essential for the operation of the Hugging Face models.
* **MMLU Dataset Creators:** For developing and making available the Massive Multitask Language Understanding (MMLU) dataset, which forms the core of these LLM evaluations. The dataset can be found at: https://huggingface.co/datasets/cais/mmlu
* **The Python Community:** For the rich ecosystem of libraries.