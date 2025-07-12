"""Collection of utility functions for benchmarking LLMs with the MMLU dataset.

This module offers a collection of utility functions for benchmarking Large Language Models (LLMs)
using the MMLU (Massive Multitask Language Understanding) dataset. It streamlines common tasks such
as safely loading JSON data, managing Hugging Face models and datasets, setting up Bedrock models,
and providing helpers for displaying and selecting MMLU subjects and examples.

Functions:
    - load_config: Loads configuration settings from a YAML file.
    - load_json_safely: Safely loads JSON data from a specified file path.
    - setup_model: Loads a pre-trained Hugging Face model and tokeniser or sets up a Bedrock model.
    - load_and_prepare_dataset: Loads and optionally samples from a Hugging Face dataset.
    - list_subjects: Extracts and optionally displays unique subjects from a dataset.
    - display_results: Displays previously evaluated subjects and their performance metrics.
    - random_subject: Selects a random, unevaluated subject from a list.
    - print_random_examples: Prints random examples from a filtered dataset for inspection.
    - get_nvidia_smi_output: Fetches the used and total VRAM using nvidia-smi.
"""

import hashlib
import json
import random as rd
import subprocess
from pathlib import Path

import boto3
import torch
import yaml
from datasets import Dataset, disable_progress_bar, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_config(config_path: str = "config.yaml") -> tuple:
    """Load configuration from a YAML file into a dict.

    Also, prints the relevant parameters on screen.

    Args:
        config_path (str): The path to the YAML configuration file. Defaults to "config.yaml".

    Returns:
        dict: A dictionary containing the model information.

    """
    # Check if the config file exists directly in the current working directory
    with Path(config_path).open() as f:
        # Load the YAML configuration file
        config_dict = yaml.safe_load(f)

        if config_dict["model_transformers"]["turned_on"]:
            print(f"Model Name: {config_dict['model_transformers']['name']}")
            print(f"Max New Tokens: {config_dict['model_transformers']['max_new_tokens']}")
            print(f"quantisation: {config_dict['model_transformers']['quantisation']}")
            print(f"Batch status: {config_dict['model_transformers']['batch_status']}")
            print(f"Batch size: {config_dict['model_transformers']['batch_size']}")
        elif config_dict["model_bedrock"]["turned_on"]:
            print(f"Bedrock profile name: {config_dict['model_bedrock']['profile_name']}")
            print(f"Region for profile: {config_dict['model_bedrock']['region_for_profile']}")
            print(f"Model name: {config_dict['model_bedrock']['name']}")

        print(f"Dataset Name: {config_dict['dataset']['name']}")
        print(f"Number Examples: {config_dict['dataset']['number_examples']}")
        print(f"Number Subjects: {config_dict['dataset']['number_subjects']}")

        print(f"Results Path: {config_dict['paths']['results']}")
        print(f"Log Path: {config_dict['paths']['logs']}")
        print(f"Context Path: {config_dict['paths']['context']}")

        return config_dict


def load_json_safely(config_dict: dict) -> tuple[str, dict]:
    """Load data from a JSON file, handling cases where the file is empty, malformed, or missing.

    Handles two cases: Hugging Face transformer models and Bedrock models.

    Args:
        config_dict (dict): The dictionary containing the configuration settings.

    Returns:
        tuple[str, dict]: A tuple containing:
                          - The full path of the file that was attempted to be loaded.
                          - The data loaded from the JSON file as a dictionary.
                            Returns an empty dictionary `{}` if the file is empty,
                            not valid JSON, or not found.

    """
    result_path = config_dict["paths"]["results"]

    try:
        # Check if the file exists
        if not Path(result_path).exists():
            print(
                f"Warning: The file '{result_path}' was not found. Returning an empty dictionary.",
            )
            return result_path, {}

        # Check if the file is empty
        if Path(result_path).stat().st_size == 0:
            print(f"Warning: The file '{result_path}' is empty. Returning an empty dictionary.")
            return result_path, {}

        with Path(result_path).open(encoding="utf-8") as f:
            return result_path, json.load(f)
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from the file '{result_path}'. "
            "It might be malformed or not contain valid JSON. Returning an empty dictionary.",
        )
        return result_path, {}


def setup_model(config_dict: dict) -> dict:
    """Set up the model, either a Huggingface transformer model or Bedrock model.

    Huggingface transformer model: Load a pre-trained model and tokeniser from the library.
    Also loads the BitsAndBytesConfig for quantisation. Handles batching.

    Bedrock model: initialises boto3 session.

    Args:
        config_dict (dict): The dictionary containing the configuration settings.

    Returns:
        dict: A dictionary containing the loaded model and tokeniser.

    """
    model_dict = {}

    if config_dict["model_transformers"]["turned_on"]:
        print(
            f"Loading model '{config_dict['model_transformers']['name']}' with quantisation: "
            f"{config_dict['model_transformers']['quantisation']} "
            f"and batching: {config_dict['model_transformers']['batch_status']}",
        )

        quantisation_config = None

        if config_dict["model_transformers"]["quantisation"] == "4bit":
            quantisation_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif config_dict["model_transformers"]["quantisation"] == "8bit":
            quantisation_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load the model and tokeniser
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config_dict["model_transformers"]["name"],
                quantization_config=quantisation_config,
                torch_dtype="auto",  # "auto" lets transformers decide based on config/model
                device_map="auto",
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print(
                "Make sure you have `accelerate` and `bitsandbytes` installed with quantisation.",
            )
            raise

        tokeniser = AutoTokenizer.from_pretrained(config_dict["model_transformers"]["name"])

        # Configure tokeniser for batching if enabled
        if config_dict["model_transformers"]["batch_status"]:
            if tokeniser.pad_token is None:
                tokeniser.pad_token = tokeniser.eos_token
            tokeniser.padding_side = "left"

        # Store the model and tokeniser
        model_dict["model_transformers"] = model
        model_dict["tokeniser"] = tokeniser

    elif config_dict["model_bedrock"]["turned_on"]:
        # Initialise a Boto3 Session with the specified profile
        session = boto3.Session(
            profile_name=config_dict["model_bedrock"]["profile_name"],
            region_name=config_dict["model_bedrock"]["region_for_profile"],
        )

        # Initialise the Bedrock Runtime client from the session
        bedrock_runtime = session.client(service_name="bedrock-runtime")
        model_dict["bedrock_runtime"] = bedrock_runtime

    return model_dict


def load_and_prepare_dataset(config_dict: dict) -> Dataset:
    """Load a dataset from the Hugging Face datasets library and prepares it for use.

    If not all examples are loaded, a seed insures that the same subset is selected every time.

    Args:
        config_dict (dict): The dictionary containing the configuration settings.

    Returns:
        Dataset: A prepared dataset object.

    """
    print(f"Loading dataset '{config_dict['dataset']['name']}'... \n")

    # Load the dataset
    if config_dict["dataset"]["number_examples"] == "all":
        dataset = load_dataset(config_dict["dataset"]["name"], "all")
        dataset = dataset["test"][:]
        dataset = Dataset.from_dict(dataset)
    else:
        dataset = load_dataset(config_dict["dataset"]["name"], "all")
        dataset = dataset.shuffle(seed=5155)["test"][
            : int(config_dict["dataset"]["number_examples"])
        ]
        dataset = Dataset.from_dict(dataset)

    print(f"Number of questions loaded: {len(dataset['question'])}")

    return dataset


def list_subjects(dataset: Dataset) -> list:
    """List all unique subjects in a dataset.

    Args:
        dataset (Dataset): The huggingface dataset from which to extract subjects.

    Returns:
        list: A list of unique subjects in the dataset.

    """
    unique_subject = list(set(dataset["subject"]))

    print("Unique subjects in the dataset: \n")
    for subject in unique_subject:
        print(f"\u2022 {subject} \n")

    return unique_subject


def display_results(results: dict) -> list:
    """Display the topics already evaluated and their results.

    Args:
        results (dict): A dictionary containing the evaluation results for each topic.

    Returns:
        list: A list of subjects that have been evaluated.

    """
    subjects_evaluated = list(results.keys()) if results else []

    print(f"Subjects already evaluated: {subjects_evaluated} \n")
    for topic in results:
        print(f"Topic: {topic}")
        print(f"\t \u2022 Number of examples: {results[topic]['number_examples']}")
        print(f"\t \u2022 Accuracy: {results[topic]['accuracy']}")

    return subjects_evaluated


def random_subject(unique_subject: list, subjects_evaluated: list, config_dict: dict) -> list[str]:
    """Select random subjects from the list of unique subjects not yet evaluated.

    The number of random subjects is provided by the configuration file.

    Args:
        unique_subject (list): A list of unique subjects in the dataset.
        subjects_evaluated (list): A list of subjects that have already been evaluated.
        config_dict (dict): The dictionary containing the configuration settings.

    Returns:
        list[str]: A list of randomly selected subjects that have not been evaluated yet.
                   If n_sub is greater than the number of available subjects, all remaining
                   subjects will be returned.

    """
    remaining_subjects = list(set(unique_subject) - set(subjects_evaluated))

    if not remaining_subjects:
        print("No subjects remaining to be evaluated.")
        return []

    # Ensure that the number of subjects does not exceed the number of available subjects
    num_to_select = min(config_dict["dataset"]["number_subjects"], len(remaining_subjects))

    # Use random.sample to select unique random subjects
    subjects_of_interest = rd.sample(remaining_subjects, num_to_select)

    if num_to_select == 1:
        print(f"The random subject of interest is: {subjects_of_interest[0]}")
    else:
        print(f"The random subjects of interest are: {subjects_of_interest}")

    return subjects_of_interest


def print_random_examples(dataset: Dataset, subject_of_interest: list, n_ex: int = 1) -> None:
    """Print n_ex random examples from the dataset for the given subject.

    Args:
        dataset (Dataset): The dataset to sample from.
        subject_of_interest (list): The subject to filter examples by.
        n_ex (int): The number of random examples to print. Defaults to 1.

    """
    print(f"Random examples from the subjects of interest: {subject_of_interest} \n")
    print("--" * 50 + "\n")

    # Disable the progress bar for all subsequent dataset operations
    disable_progress_bar()

    questions = dataset.filter(lambda x: x["subject"] in subject_of_interest).shuffle()

    number_examples = min(n_ex, len(questions))

    questions = questions.select(range(number_examples))

    for ind, _ in enumerate(questions):
        print(f"The subject is {questions['subject'][ind]} \n")
        print(f"The question is: {questions['question'][ind]} \n")

        print("The options are:\n")
        for choice in questions["choices"][ind]:
            print(f"\t \u2022 {choice} \n")

        sol_index = questions["answer"][ind]
        print(f"The answer is: {questions['choices'][ind][sol_index]} \n")
        print("--" * 50 + "\n")


def get_nvidia_smi_output() -> tuple[float, float]:
    """Execute nvidia-smi command and returns its JSON output.

    Returns:
        tuple[float, float]: A tuple containing the used and total VRAM in MB.

    """
    # Query for memory usage in JSON format
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd).decode("utf-8").strip()  # noqa: S603
    # Parse the output
    used_memory_mb, total_memory_mb = map(int, output.split(","))
    return used_memory_mb, total_memory_mb


def get_unique_id(input_string: str) -> str:
    """Generate a unique SHA-256 hash for a given string.

    Args:
        input_string (str): The input string for which to generate a unique ID.

    Returns:
        str: A hexadecimal string representing the SHA-256 hash of the input string.

    """
    # Encode the string to bytes, as hash functions operate on bytes
    bytes_string = input_string.encode("utf-8")

    # Create a SHA-256 hash object
    hasher = hashlib.sha256()

    # Update the hash object with the bytes string
    hasher.update(bytes_string)

    return hasher.hexdigest()
