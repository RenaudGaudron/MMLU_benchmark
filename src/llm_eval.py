"""Provides the core logic for evaluating Language Models (LLMs).

This is done on specific subjects from the MMLU dataset, particularly tailored for
multiple-choice question answering. It handles the prompting of the LLM,
parsing of its responses, and logging of the evaluation process and results.

The module is designed to work with Hugging Face Transformer models or Bedrock models,
 and it incorporates logging to track the evaluation progress and outcomes for each subject.

It supports quantisation and batching for transformer models, which can both be turned on or off.

Functions:
    - llm_eval: The primary entry point for evaluation, determining whether to
            use batching or single-item evaluation based on configuration and
            whether to use a transformer model or a Bedrock model.
    - single_llm_eval: The main evaluation function that orchestrates the
      prompting, response generation, prediction extraction, and accuracy
      calculation for a given subject when batching is off.
    - batch_llm_eval: The main evaluation function that orchestrates the
      prompting, response generation, prediction extraction, and accuracy
      calculation for a given subject when batching is on.
    - llm_eval_init: initialises evaluation parameters and log file paths.
    - log_initialisation: Writes initial setup messages to the log.
    - log_subject_initialisation: Logs the start of evaluation for a new
                                  subject.
    - log_processing_item: Logs the progress of processing individual
                           items in the dataset.
    - generate_and_log_llm_prompt: Constructs and logs the prompt sent
                                   to the LLM.
    - batch_generate_llm_response: Generates a response from the LLM
                                   based on provided messages.
    - log_response: Logs the LLM's raw response and the correct answer.
    - extract_and_log_prediction: Parses the LLM's response to extract
                                  a prediction and logs it.
    - cal_and_store_subject_acc: Calculates, logs, and stores the accuracy
                                 for a subject.
    - batch_init: initialises the batching process for a subset of the dataset.
    - batch_log_processing_item: Logs the progress of processing a batch of items.
    - batch_generate_llm_prompt: Generates a formatted prompt for the LLM for
                                 a single item in a batch.
    - single_iteration: Iterates through a subject's dataset without batching.
                                 Function for Hugging Face transformer models only.
    - single_iteration_bedrock: Iterates through a subject's dataset without batching.
                                Function for Bedrock models only.
    - batch_iteration: Iterates through a subject's dataset using batching.
    - single_generate_llm_prompt: initialises the processing and generates
                                  a prompt for a single item.
    - bedrock_generate_llm_prompt: initialises the processing and generates
                                   a prompt for a single item - Bedrock models.
    - single_generate_llm_response: Generates a text response from the LLM
                                    for a single set of messages.
    - format_message_transformers: Formats the message list for the LLM.
"""

# Importing necessary libraries
import datetime
import json
import re
import time
from pathlib import Path
from typing import TextIO

import evaluate
import torch
from datasets import Dataset
from tqdm import tqdm

from src.utils import get_nvidia_smi_output


def llm_eval(
    dataset: Dataset,
    result_path: str,
    model_dict: dict,
    subject_of_interest: list,
    results_dict: dict,
    config_dict: dict,
) -> None:
    """Call the appropriate function depending on the configuration file.

    Two options are currently available:
    - Huggingface model without batching or Bedrock models.
    - Huggingface models with batching.

    Args:
        dataset (Dataset): The Huggingface dataset containing the questions and answers.
        result_path (str): The path of the JSON file containing the result for this quantisation.
        model_dict (dict): The dictionary containing the model settings.
        subject_of_interest (list): A list containing the subjects to be evaluated.
        results_dict (dict): A dictionary to store the evaluation results for each subject.
        config_dict (dict): The dictionary containing the configuration settings.

    """
    if (
        (config_dict["model_transformers"]["turned_on"])
        & (config_dict["model_transformers"]["batch_status"])
    ):
        print("Huggingface model. Batch status is True. Calling batch_llm_eval...")
        batch_llm_eval(
            dataset,
            result_path,
            model_dict,
            subject_of_interest,
            results_dict,
            config_dict,
        )
    else:
        print("Batch status is False. Calling single_llm_eval...")
        single_llm_eval(
            dataset,
            result_path,
            model_dict,
            subject_of_interest,
            results_dict,
            config_dict,
        )


def batch_llm_eval(
    dataset: Dataset,
    result_path: str,
    model_dict: dict,
    subject_of_interest: list,
    results_dict: dict,
    config_dict: dict,
) -> None:
    """Evaluate a Language Model (LLM) on a specific subject from a dataset using batching.

    Args:
        dataset (Dataset): The Huggingface dataset containing the questions and answers.
        result_path (str): The path of the JSON file containing the result for this quantisation.
        model_dict (dict): The dictionary containing the model and tokeniser.
        subject_of_interest (list): A list containing the subjects to be evaluated.
        results_dict (dict): A dictionary to store the evaluation results for each subject.
        config_dict (dict): The dictionary containing the configuration settings.

    """
    log_name = llm_eval_init(config_dict["paths"]["logs"])

    with Path(log_name).open("w") as f_log:
        log_initialisation(f_log, config_dict)

        for subject in subject_of_interest:
            start_time = log_subject_initialisation(f_log, subject)

            # Filtering the dataset to the subject of interest
            subject_dataset = dataset.filter(lambda x, s=subject: x["subject"] == s)

            # Initialise the subject dictionary
            subject_dict = {
                "subject": subject,
                "dataset": subject_dataset,
                "question_id": [],
                "references": [],
                "predictions": [],
            }

            # Iterate through a subject's dataset in batches
            batch_iteration(f_log, model_dict, config_dict, subject_dict)

            # Calculate and store the accuracy for the subject
            cal_and_store_subject_acc(
                f_log,
                result_path,
                start_time,
                subject_dict,
                results_dict,
            )


def single_llm_eval(
    dataset: Dataset,
    result_path: str,
    model_dict: dict,
    subject_of_interest: list,
    results_dict: dict,
    config_dict: dict,
) -> None:
    """Evaluate a Language Model (LLM) on a specific subject from a dataset without batching.

    Can be either a transformer model or a Bedrock model.

    Args:
        dataset (Dataset): The Huggingface dataset containing the questions and answers.
        result_path (str): The path of the JSON file containing the result for this quantisation.
        model_dict (dict): The dictionary containing the model settings.
        subject_of_interest (list): A list containing the subjects to be evaluated.
        results_dict (dict): A dictionary to store the evaluation results for each subject.
        config_dict (dict): The dictionary containing the configuration settings.

    """
    log_name = llm_eval_init(config_dict["paths"]["logs"])

    with Path(log_name).open("w") as f_log:
        log_initialisation(f_log, config_dict)

        for subject in subject_of_interest:
            start_time = log_subject_initialisation(f_log, subject)

            # Filtering the dataset to the subject of interest
            subject_dataset = dataset.filter(lambda x, s=subject: x["subject"] == s)

            # Initialise the subject dictionary
            subject_dict = {
                "subject": subject,
                "dataset": subject_dataset,
                "question_id": [],
                "references": [],
                "predictions": [],
            }

            # Iterate through a subject's dataset without batching
            if config_dict["model_transformers"]["turned_on"]:
                single_iteration(
                    f_log,
                    model_dict,
                    config_dict,
                    subject_dict,
                )
            elif config_dict["model_bedrock"]["turned_on"]:
                single_iteration_bedrock(
                    f_log,
                    model_dict,
                    config_dict,
                    subject_dict,
                )

            # Calculate and store the accuracy for the subject
            cal_and_store_subject_acc(
                f_log,
                result_path,
                start_time,
                subject_dict,
                results_dict,
            )


def llm_eval_init(log_path: str) -> str:
    """Create the log file path based on the current datetime.

    This function generates a unique log file name using the current timestamp,
    ensuring that each evaluation run has its own log.

    Args:
        log_path (str): The directory path where the log file will be saved.

    Returns:
        str: The full path for the log file, named with a datetime stamp.

    """
    current_time = datetime.datetime.now(tz=datetime.timezone.utc)
    datetime_str = current_time.strftime("%Y%m%d_%H%M%S")

    return log_path + f"evaluation_log_{datetime_str}.txt"


def log_initialisation(f_log: TextIO, config_dict: dict) -> None:
    """Write the first few initialisation lines in the log file.

    Handles both transformer and Bedrock models.

    Args:
        f_log (TextIO): The log file object to write the initialisation message to.
        config_dict (dict): Contains the configuration settings, to be written in the log.

    """
    f_log.write("--" * 50 + "\n")
    f_log.write("Initialisation\n")
    f_log.write("--" * 50 + "\n")

    if config_dict["model_transformers"]["turned_on"]:
        f_log.write(f"Model Name: {config_dict['model_transformers']['name']}" + "\n")
        f_log.write(f"Max New Tokens: {config_dict['model_transformers']['max_new_tokens']}" + "\n")
        f_log.write(f"quantisation: {config_dict['model_transformers']['quantisation']}" + "\n")
        f_log.write(f"Batch status: {config_dict['model_transformers']['batch_status']}" + "\n")
        f_log.write(f"Batch size: {config_dict['model_transformers']['batch_size']}" + "\n")
        f_log.write(f"Dataset Name: {config_dict['dataset']['name']}" + "\n")
        f_log.write(f"Number Examples: {config_dict['dataset']['number_examples']}" + "\n")
        f_log.write(f"Number Subjects: {config_dict['dataset']['number_subjects']}" + "\n")
    elif config_dict["model_bedrock"]["turned_on"]:
        f_log.write(f"Bedrock profile name: {config_dict['model_bedrock']['profile_name']}" + "\n")
        f_log.write(f"Region: {config_dict['model_bedrock']['region_for_profile']}" + "\n")
        f_log.write(f"Model name: {config_dict['model_bedrock']['name']}" + "\n")

    f_log.write("--" * 50 + "\n")


def log_subject_initialisation(f_log: TextIO, subject: str) -> float:
    """Log the initialisation message for a specific subject being evaluated.

    Also, initialises the timer.

    Args:
        f_log (TextIO): The log file object to write to.
        subject (str): The subject to be evaluated.

    Returns:
        time: The start time of the timer.

    """
    start_time = time.perf_counter()

    f_log.write(f"Processing subject: {subject}\n")
    f_log.write("--" * 50 + "\n")

    return start_time


def batch_init(
    filtered_dataset: Dataset,
    s_index: int,
    batch_size: int,
) -> tuple[Dataset, list, list, list]:
    """Initilise the batching process.

    Args:
        filtered_dataset (Dataset): The subset of the dataset
                                             containing items relevant to the current subject.
        s_index (int): The starting index for the current batch.
        batch_size (int): The number of items to include in each batch.

    """
    batch_items = filtered_dataset[s_index : s_index + batch_size]

    batch_prompts_for_llm = []
    batch_correct_answers = []
    batch_log_prompts = []

    return batch_items, batch_prompts_for_llm, batch_correct_answers, batch_log_prompts


def log_processing_item(
    f_log: TextIO,
    subject_dict: dict,
    item: int,
    message: str,
) -> None:
    """Log the progress of processing an individual item within a subject's dataset.

    This function writes a formatted message to the log file indicating the current
    item number being processed out of the total items for the given subject.

    Args:
        f_log (TextIO): The log file object to write to.
        subject_dict (dict): A dictionary containing the subject name and dataset.
        item (int): The 0-based index of the current item being processed within the
                    `filtered_dataset`.
        message (str): Message sent to the LLM for item number item + 1.

    """
    f_log.write(
        f"Subject {subject_dict['subject']} - Processing item number:"
        f"{item + 1} out of {len(subject_dict['dataset'])} \n",
    )
    f_log.write("--" * 50 + "\n")

    f_log.write(
        f"Message sent to the LLM for item {item + 1}: {message}\n",
    )
    f_log.write("--" * 50 + "\n")


def batch_generate_llm_response(
    batch_messages: list[list[dict]],
    model_dict: dict,
    max_new_tokens: int = 38_912,
) -> list[str]:
    """Generate text responses from a Language Model (LLM) for a batch of messages.

    This function prepares the input messages for a batch using the tokeniser's
    chat template, tokenizes them with padding, passes them to the model for
    text generation, and then decodes the generated token IDs back into
    human-readable strings.

    Args:
        batch_messages (List[List[dict]]): A list of message lists, where each inner list
                                            represents the chat history for one item in the batch.
        model_dict (dict): The dictionary containing the model and tokeniser.
        max_new_tokens (int): The maximum number of new tokens to generate for each response.

    Returns:
        List[str]: A list of generated text responses from the LLM, one for each item in the batch.

    """
    # Load the tokeniser and model
    tokeniser = model_dict["tokeniser"]
    model = model_dict["model_transformers"]

    # Apply chat template to each set of messages in the batch
    batch_texts = [
        tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in batch_messages
    ]

    # Tokenize the entire batch with padding
    model_inputs = tokeniser(batch_texts, return_tensors="pt", padding=True, truncation=True).to(
        model.device,
    )

    # Generate responses for the batch
    with torch.no_grad():  # Ensure no gradients are calculated during inference
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )

    # Decode the generated IDs. Need to slice correctly for each original input length.
    # Note: When padding_side="left", the input_ids are on the right,
    # so the generated tokens are appended to the *end* of the padded sequence.
    # We still slice based on the original input_ids length.
    decoded_responses = []
    for i in range(len(batch_messages)):
        input_length = model_inputs.input_ids[i].shape[0]
        # In batch decode, generated_ids is (batch_size,
        # sequence_length_including_input_and_generation)
        # We need to slice each sequence from the original
        # input length to get only the generated part.
        output_ids_for_item = generated_ids[i, input_length:]
        decoded_responses.append(tokeniser.decode(output_ids_for_item, skip_special_tokens=True))

    return decoded_responses


def log_response(f_log: TextIO, response: str, correct: str, subject_dict: dict) -> None:
    """Log the LLM's generated answer and the corresponding correct answer to the log file.

    This function writes both the LLM's output and the ground truth answer
    into the provided log file.

    Args:
        f_log (TextIO): The log file object to write the information to.
        response (str): The LLM's generated response.
        correct (str): The correct answer associated with the question.
        subject_dict (dict): A dictionary to store the question ID, references,
                            and predictions for the current subject.

    """
    f_log.write(f"Response made by LLM: {response} \n")
    f_log.write("--" * 50 + "\n")

    f_log.write(f"Correct answer: {correct} \n")
    f_log.write("--" * 50 + "\n")

    # Storing the correct answer in the reference list
    subject_dict["references"].append(correct)


def extract_and_log_prediction(f_log: TextIO, response: str, subject_dict: dict) -> None:
    """Extract the predicted answer from an LLM's response.

    This is done using regex, then logs the extracted prediction,
    and appends it to a list of predictions.

    It attempts to parse an integer option from the response. If parsing fails
    or no match is found, a default value of 99 is used for the prediction.

    Then, logs the current lists of ground truth references and model predictions.

    Args:
        f_log (TextIO): The log file object to write the extracted prediction to.
        response (str): The raw text response generated by the LLM.
        subject_dict (dict): A dictionary to store the question ID, references,
                            and predictions for the current subject.

    """
    # Extracting the model's answer using Regex
    corrected_match = re.search(
        r"answer is: Option (.*?)\s*\. Done",
        response,
    )  # Added \. to match literal dot
    predicted_answer = (
        corrected_match.group(1).strip() if corrected_match else 99
    )  # Default to 99 if no match found
    f_log.write(f"Predicted answer: {predicted_answer} \n")
    f_log.write("--" * 50 + "\n")

    # Storing the predicted index
    try:
        subject_dict["predictions"].append(int(predicted_answer))
    except ValueError:
        subject_dict["predictions"].append(99)

    # Printing the references and predictions for debugging
    f_log.write(f"References: {subject_dict['references']} \n")
    f_log.write(f"Predictions: {subject_dict['predictions']} \n")
    f_log.write("--" * 50 + "\n")


def cal_and_store_subject_acc(
    f_log: TextIO,
    result_path: str,
    start_time: time,
    subject_dict: dict,
    results_dict: dict,
) -> None:
    """Calculate the accuracy for a given subject.

    Also logs the result, and stores the subject's overall accuracy and number of examples
    in the results dictionary.

    Args:
        f_log (TextIO): The log file object.
        result_path (str): The location of the JSON file containing the results.
        start_time (time): The start time of the timer, used to calculate the execution time.
        subject_dict (dict): A dictionary containing the question ID, references, and predictions.
        results_dict (dict): The dictionary where the overall evaluation results
                             for each subject are stored.

    """
    # Calculates the execution time for that subject
    execution_time = round(time.perf_counter() - start_time, 3)

    # Accuracy metric for evaluation
    accuracy_metric = evaluate.load("accuracy")

    # Calculating accuracy for the subject
    f_log.write(f"Calculating accuracy for the subject: {subject_dict['subject']} \n")
    acc_metric = accuracy_metric.compute(
        predictions=subject_dict["predictions"],
        references=subject_dict["references"],
    )["accuracy"]
    f_log.write(f"The accuracy for subject {subject_dict['subject']} is {acc_metric} \n\n")
    f_log.write("--" * 50 + "\n")

    # Fetching the used and total VRAM
    used_vram, total_vram = get_nvidia_smi_output()

    # Adding the results to the results dictionary
    results_dict[subject_dict["subject"]] = {
        "accuracy": acc_metric,
        "number_examples": len(subject_dict["dataset"]),
        "execution_time": execution_time,
        "used_VRAM": used_vram,
        "total_VRAM": total_vram,
        "detailed_results": {key: subject_dict[key] for key in ["references", "predictions"]},
    }

    # Save the results to the result JSON file
    with Path(result_path).open("w") as f:
        json.dump(results_dict, f, indent=4)


def batch_log_processing_item(
    f_log: TextIO,
    subject_dict: dict,
    start_idx: int,
    batch_size: int,
) -> None:
    """Log the progress of processing a batch of items.

    Args:
        f_log (TextIO): The log file object to write to.
        subject_dict (dict): The dictionary containing the subject name and dataset.
        start_idx (int): The starting index of the current batch.
        batch_size (int): The size of the current batch.

    """
    end_idx = min(start_idx + batch_size, len(subject_dict["dataset"]))

    f_log.write(
        f"Subject {subject_dict['subject']} - Processing items from {start_idx + 1} "
        f"to {end_idx} out of {len(subject_dict['dataset'])}\n",
    )
    f_log.write("--" * 50 + "\n")


def batch_generate_llm_prompt(
    batch_items: dict,
    batch_item_idx: int,
    batch_prompts: list,
    batch_log_prompts: list,
    batch_truth: list,
    config_dict: dict,
) -> tuple[list[dict], str]:
    """Generate a formatted prompt for the LLM for a single item within a batch.

    Returns the message list and the raw string prompt for logging.

    Args:
        batch_items (Dict): A dictionary representing the current batch of items,
                            e.g., {'question': [q1, q2], 'choices': [[c1,c2..],[c1,c2..]], ...}.
        batch_item_idx (int): The index of the current item within the batch.
        batch_prompts (list): A list to which the prompt sent to the LLM will be appended.
        batch_log_prompts (list): A list to which the raw string prompt for logging
                                  will be appended.
        batch_truth (list): A list to which the ground truth will be appended.
        config_dict (dict): The dictionary containing the configuration settings.

    Returns:
        tuple[list[dict], str]: A tuple containing:
            - list(dict): A list of dictionaries representing the messages to be sent to the LLM.
            - str: The raw string of the prompt generated for logging.

    """
    agg_text = "Question. "
    agg_text += batch_items["question"][batch_item_idx]
    agg_text += ". Choose between the following options. "

    for i, choice in enumerate(batch_items["choices"][batch_item_idx]):
        agg_text += f"Option {i}: {choice}"
        if i < len(batch_items["choices"][batch_item_idx]) - 1:
            agg_text += ", "

    messages = format_message_transformers(agg_text, config_dict)

    batch_prompts.append(messages)
    batch_log_prompts.append(agg_text)
    batch_truth.append(batch_items["answer"][batch_item_idx])

    return batch_prompts, batch_log_prompts, batch_truth


def single_iteration(
    f_log: TextIO,
    model_dict: dict,
    config_dict: dict,
    subject_dict: dict,
) -> None:
    """Iterate through a subject's dataset without batching.

    Works only for Hugging Face transformer models.

    This function processes each item in a given subject's dataset individually.
    For each item, it constructs a prompt, generates a response from the LLM,
    and then extracts a prediction, compares it against the ground truth,
    and logs the results.

    Args:
        f_log (TextIO): The file object for the evaluation log, where processing
                        details and results will be written.
        model_dict (dict): The dictionary containing the model and tokeniser.
        config_dict (dict): The dict containing the configuration settings.
        subject_dict (dict): A dictionary to store the question ID, references,
                              and predictions for the current subject.

    """
    max_new_tokens = config_dict["model_transformers"]["max_new_tokens"]

    for i, item in tqdm(
        enumerate(subject_dict["dataset"]),
        total=len(subject_dict["dataset"]),
        desc=f"Processing {subject_dict['subject']}: ",
    ):
        # Initialise processing for a single item
        messages, log_prompt, correct_answer = single_generate_llm_prompt(item, config_dict)

        log_processing_item(f_log, subject_dict, i, log_prompt)

        # Generate response for the single item
        response = single_generate_llm_response(messages, model_dict, max_new_tokens)

        # Log response and extract prediction
        log_response(f_log, response, correct_answer, subject_dict)
        extract_and_log_prediction(f_log, response, subject_dict)


def single_iteration_bedrock(
    f_log: TextIO,
    model_dict: dict,
    config_dict: dict,
    subject_dict: dict,
) -> None:
    """Iterate through a subject's dataset using a Bedrock model.

    This function processes each item in a given subject's dataset individually.
    For each item, it constructs a prompt, generates a response from the LLM,
    and then extracts a prediction, compares it against the ground truth,
    and logs the results.

    Args:
        f_log (TextIO): The file object for the evaluation log, where processing
                        details and results will be written.
        model_dict (dict): The dictionary containing the model settings.
        config_dict (dict): The dict containing the configuration settings.
        subject_dict (dict): A dictionary to store the question ID, references,
                              and predictions for the current subject.

    """
    for i, item in tqdm(
        enumerate(subject_dict["dataset"]),
        total=len(subject_dict["dataset"]),
        desc=f"Processing {subject_dict['subject']}: ",
    ):
        # Initialise processing for a single item
        messages, log_prompt, correct_answer = bedrock_generate_llm_prompt(item, config_dict)

        log_processing_item(f_log, subject_dict, i, log_prompt)

        # Generate response for the single item

        model_response = model_dict["bedrock_runtime"].converse(
            modelId=config_dict["model_bedrock"]["name"],
            messages=messages,
        )

        response = model_response["output"]["message"]["content"][0]["text"]

        # Log response and extract prediction
        log_response(f_log, response, correct_answer, subject_dict)
        extract_and_log_prediction(f_log, response, subject_dict)


def batch_iteration(
    f_log: TextIO,
    model_dict: dict,
    config_dict: dict,
    subject_dict: dict,
) -> None:
    """Iterate through a subject's dataset in batches.

    Hugging Face transformer models only.

    This function orchestrates the batch-wise processing of questions for a given
    subject. For each batch, it constructs prompts, generates responses from the LLM,
    and then processes each individual response to extract predictions, compare
    them against ground truth, and log the results.

    Args:
        f_log (TextIO): The file object for the evaluation log, where processing
                        details and results will be written.
        model_dict (dict): The dictionary containing the model and tokeniser.
        config_dict (dict): The dict containing the configuration settings.
        subject_dict (dict): A dictionary to store the question IDs, references,
                              and predictions for the current subject.

    """
    batch_size = config_dict["model_transformers"]["batch_size"]
    max_new_tokens = config_dict["model_transformers"]["max_new_tokens"]

    # Iterate through the dataset in batches
    for i in tqdm(
        range(0, len(subject_dict["dataset"]), batch_size),
        desc=f"Processing {subject_dict['subject']}: ",
    ):
        batch_items, batch_prompts, batch_truth, batch_log_prompts = batch_init(
            subject_dict["dataset"],
            i,
            batch_size,
        )

        # Iterate over items in the current batch
        for batch_item_idx in range(len(batch_items["question"])):
            # Generate prompts for each item in the batch
            batch_prompts, batch_log_prompts, batch_truth = batch_generate_llm_prompt(
                batch_items,
                batch_item_idx,
                batch_prompts,
                batch_log_prompts,
                batch_truth,
                config_dict,
            )

        batch_log_processing_item(f_log, subject_dict, i, batch_size)

        # Generate responses for the entire batch
        batch_responses = batch_generate_llm_response(
            batch_prompts,
            model_dict,
            max_new_tokens,
        )

        # Process each response and correct answer in the batch
        for j, response in enumerate(batch_responses):
            correct = batch_truth[j]

            log_processing_item(f_log, subject_dict, i + j, batch_log_prompts[j])

            log_response(f_log, response, correct, subject_dict)
            extract_and_log_prediction(f_log, response, subject_dict)


def single_generate_llm_prompt(
    item: dict,
    config_dict: dict,
) -> tuple[list[dict], str, str]:
    """Initialise the processing for a single item.

    Hugging Face transformer models only.

    Args:
        item (dict): A dictionary representing the current item from the dataset.
        config_dict (dict): The dictionary containing the configuration settings.

    Returns:
        tuple[list[dict], str, str]: A tuple containing:
            - list[dict]: The list of messages formatted for the LLM.
            - str: The raw string of the prompt generated for logging.
            - str: The correct answer for the current item.

    """
    agg_text = "Question. "
    agg_text += item["question"]
    agg_text += ". Choose between the following options. "

    for idx, choice in enumerate(item["choices"]):
        agg_text += f"Option {idx}: {choice}"
        if idx < len(item["choices"]) - 1:
            agg_text += ", "

    messages = format_message_transformers(agg_text, config_dict)

    correct_answer = item["answer"]
    return messages, agg_text, correct_answer


def bedrock_generate_llm_prompt(
    item: dict,
    config_dict: dict,
) -> tuple[list[dict], str, str]:
    """Initialise the processing for a single item. Bedrock models only.

    Args:
        item (dict): A dictionary representing the current item from the dataset.
        config_dict (dict): The dictionary containing the configuration settings.

    Returns:
        tuple[list[dict], str, str]: A tuple containing:
            - list[dict]: The list of messages formatted for the LLM.
            - str: The raw string of the prompt generated for logging.
            - str: The correct answer for the current item.

    """
    agg_text = load_context(config_dict["paths"]["context"])

    # Now the question
    agg_text += " Question. "
    agg_text += item["question"]
    agg_text += ". Choose between the following options. "

    for idx, choice in enumerate(item["choices"]):
        agg_text += f"Option {idx}: {choice}"
        if idx < len(item["choices"]) - 1:
            agg_text += ", "

    messages = [
        {
            "role": "user",
            "content": [{"text": agg_text}],
        },
    ]
    correct_answer = item["answer"]
    return messages, agg_text, correct_answer


def single_generate_llm_response(
    messages: list[dict],
    model_dict: dict,
    max_new_tokens: int = 38_912,
) -> str:
    """Generate a text response from a Language Model (LLM) for a single set of messages.

    Hugging Face transformer models only.

    This function prepares a single input message using the tokeniser's
    chat template, tokenises it, passes it to the model for text generation,
    and then decodes the generated token IDs back into a human-readable string.

    Args:
        messages (list[dict]): A list of dictionaries representing the chat history
                               for a single conversational turn.
        model_dict (dict): The dictionary containing the model and tokeniser.
        max_new_tokens (int): The maximum number of new tokens to generate for the response.

    Returns:
        str: The generated text response from the LLM.

    """
    # Load the tokeniser and model
    tokeniser = model_dict["tokeniser"]
    model = model_dict["model_transformers"]

    # Apply chat template to the single set of messages
    text = tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenise the single text. No padding needed for a single input.
    model_inputs = tokeniser(text, return_tensors="pt", truncation=True).to(model.device)

    # Generate response for the single input
    with torch.no_grad():  # Ensure no gradients are calculated during inference
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )

    # Decode the generated IDs.
    # The generated_ids tensor will be of shape (1, sequence_length_including_input_and_generation)
    # We slice from the original input_ids length to get only the generated part.
    input_length = model_inputs.input_ids[0].shape[0]
    output_ids_for_item = generated_ids[0, input_length:]

    return tokeniser.decode(output_ids_for_item, skip_special_tokens=True)


def format_message_transformers(agg_text: str, config_dict: dict) -> list:
    """Format the message list for the LLM.

    Hugging Face transformer models only.

    Args:
        agg_text (str): The prepared question and answer for the LLM.
        config_dict (dict): The dictionary containing the configuration settings.

    Returns:
        list: A list containing the formatted message to be fed to the LLM.

    """
    system_content = load_context(config_dict["paths"]["context"])

    return [
        {
            "role": "system",
            "content": system_content,
        },
        {"role": "user", "content": agg_text},
    ]


def load_context(context_path: str) -> str:
    """Load the context from the context.txt file.

    This function reads the content of the context.txt file located in the
    same directory as this script and returns it as a string.

    Args:
        context_path (str): The path to the context.txt file.

    Returns:
        str: The content of the context.txt file.

    """
    with Path(context_path).open("r") as f:
        return f.read().replace("\n", " ")
