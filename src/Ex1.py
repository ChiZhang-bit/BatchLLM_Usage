import os
import random
import re

import numpy as np
import pandas as pd
import torch

from typing import Any, Dict, List
from datasets import load_dataset
from InstructorEmbedding import INSTRUCTOR
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from rich import print
from rich.progress import track
from sentence_transformers import util
from pathlib import Path

# The maximum number of tokens in the input and output
MAX_INPUT_TOKENS = 6192 * 4
MAX_OUTPUT_TOKENS = 2048

os.environ["OPENAI_API_KEY"] = "sk-BHBJElSOXC6F7ZZwF76f4718C0D9417188De33Bd2967753c"


def _fetch_in_context_examples(
        train, test, k: int = 3, retrieve_similar: bool = True
):
    if retrieve_similar:
        # Step1: Cluster for embed the train and test
        embedder = INSTRUCTOR("../instructor-model")
        embedding_instructions = "Represent the Medicine dialogue for clustering:"
        test_dialogues = embedder.encode(
            [
                [embedding_instructions, f"dataset: {dataset} dialogue: {dialogue}"]
                for dataset, dialogue in zip(test["dataset"], test["dialogue"])
            ],
            batch_size=8,
            show_progress_bar=True,
        )
        train_dialogues = embedder.encode(
            [
                [embedding_instructions, f"dataset: {dataset} dialogue: {dialogue}"]
                for dataset, dialogue in zip(train["dataset"], train["dialogue"])
            ],
            batch_size=8,
            show_progress_bar=True,
        )

    # Get top-k most similar examples in the train set for each example in the test set
    top_k_indices = []
    for i, test_dialogue in enumerate(test_dialogues):
        train_indices = list(range(len(train["dialogue"])))
        if retrieve_similar:
            scores = util.cos_sim(np.expand_dims(test_dialogue, 0), train_dialogues[train_indices])
            scores = torch.squeeze(scores)
            top_k_indices_ds = torch.topk(scores, k=min(k, len(scores))).indices.flatten().tolist()
            top_k_indices.append([train_indices[idx] for idx in top_k_indices_ds])
        else:
            # random:
            top_k_indices.append(random.sample(range(len(train_indices)), k=min(k, len(train_indices))))

    return top_k_indices


def _format_in_context_example(
        train: Dict[str, Any],
        idx: int,
        include_dialogue: bool = False
) -> str:
    example = ""
    if include_dialogue:
        example += f'\nEXAMPLE DIALOGUE:\n{train["dialogue"][idx].strip()}'
    example += f'\nEXAMPLE NOTE:\n{train["note"][idx].strip()}'
    return example


def main(
        train_fp: str,
        test_fp: str,
        output_dir: str,
        model_name: str = "gpt-3.5-turbo-1106",
        temperature: float = 0.05,
        k: int = 2,
        retrieve_similar: bool = True,
        dialogue_note: bool = False,
) -> None:
    """
    train_fp: path of train dataset
    test_fp: path of test dataset
    output_dir: path of output result
    model_name: LLM Model Name
    temperature: the temperature of LLM,
    k: the num of in-context examples
    retrieve_similar: according to the similarity to choose the in-context example
    dialogue_note: if True, choose dialogue_note as in-context example; Else, choose only_note as in-context example

    Generates predictions using Langchain for the given task and run on the given test set.
    """
    # Step1: load the dataset:
    train = load_dataset(
        path="csv",
        data_files={
            "train": train_fp,
        },
    )["train"]

    test = load_dataset(
        path="csv",
        data_files={
            "test": test_fp,
        }
    )["test"]

    top_k_indices = _fetch_in_context_examples(
        train=train,
        test=test,
        k=1,
        retrieve_similar=retrieve_similar
    )

    # Step2: setup the LLM
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        max_tokens=MAX_OUTPUT_TOKENS,
        openai_api_key="sk-BHBJElSOXC6F7ZZwF76f4718C0D9417188De33Bd2967753c",
        openai_api_base="https://cd.aiskt.com/v1"
    )

    # Step3: setup the chain
    prompt = PromptTemplate(
        input_variables=["examples", "dialogue"],
        template="""Write a clinical note reflecting this doctor-patient dialogue. Use the example notes below to decide the structure of the clinical note. Do not make up information.
    {examples}

    DIALOGUE: {dialogue}
    CLINICAL NOTE:
            """,
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    # Step4: Run the chain to generate predictions
    predictions = []
    for i, dialogue in track(
            enumerate(test["dialogue"]),
            description="Generating predictions... ",
            total=len(test["dialogue"])
    ):
        # collect k in-context examples as we can fit in the max input tokens
        examples = ""
        for top_k_idx in top_k_indices[i]:
            example = _format_in_context_example(train, idx=top_k_idx, include_dialogue=dialogue_note)
            examples += example
            prediction = chain.invoke(
                input={
                    "dialogue": dialogue,
                    "examples": examples
                }
            )
            predictions.append(prediction['text'])

    # Step5: Save prediction results
    ct_output = {
        "TestID": test["encounter_id"],
        "SystemOutput": predictions
    }
    output_dir = Path(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ct_fn = "Ex1_result.csv"
    ct_fp = os.path.join(output_dir, ct_fn)
    pd.DataFrame.from_dict(ct_output).to_csv(ct_fp, index=False)

    print(f"[green]Predictions saved to {output_dir}[/green]")

    return predictions


if __name__ == '__main__':

    # Ex1:
    main(
        train_fp="../data/train.csv",
        test_fp="../data/test.csv",
        output_dir="../output"
    )
