# MMLU Benchmark Evaluation Tool

This repository provides a streamlined tool for evaluating Large Language Models (LLMs) on the Massive Multitask Language Understanding (MMLU) benchmark.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Results structure](#results-structure)
- [License](#license)
- [Acknowledgements](#acknowledgments)

## Introduction

Evaluating Large Language Models (LLMs) on various benchmarks is crucial for understanding their capabilities and limitations. The MMLU benchmark, with its diverse range of subjects, is a widely recognized standard for assessing knowledge and reasoning. This project aims to provide an easy-to-use script to run MMLU evaluations, log performance metrics, and save results in a structured format.

## Features

This project provides an easy-to-use and standardized framework for evaluating open-source Large Language Models (LLMs) on the Massive Multitask Language Understanding (MMLU) dataset. It aims to streamline the benchmarking process, allowing users to efficiently assess model performance (accuracy) and inference efficiency (speed and VRAM utilization) across various MMLU subjects. By offering a clear and reproducible methodology, this project facilitates comparative analysis of LLMs, including the impact of techniques like quantization, to better understand their capabilities and resource requirements for diverse applications.

## Results structure

Evaluation results will be saved in the results/results.json file. This JSON file will store the accuracy, number of examples, execution time, used VRAM, and available VRAM for each evaluated MMLU subject.

The evaluation results will be stored in a JSON file with the following structure:

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
MIT License. Copyright holder: Renaud Gaudron

## Acknowledgements
- Hugging Face Transformers for providing easy access to models and tokenizers.
- Hugging Face Datasets for the MMLU dataset.
- Qwen Team for releasing the Qwen3 1.7B model.