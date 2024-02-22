import csv
import pandas as pd
import json

from pathlib import Path
from datasets import load_dataset


def load_data(train_fp, test_fp, valid_fp):
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
        },
    )["test"]

    valid = load_dataset(
        path="csv",
        data_files={
            "valid": valid_fp,
        },
    )["valid"]

    return train, test, valid


def trans_format(dataset, output_fp):
    notes = []
    dataset_list = []
    for i, note in enumerate(dataset["note"]):
        note_sentences = note.split('\n')
        note_sentences = [x.strip() for x in note_sentences if note.strip() != ""]

        section_dict = {}
        title = ""

        for j in range(len(note_sentences)):
            """
            主要包含几个部分：
            CHIEF COMPLAINT ...
            """
            if note_sentences[j].isupper():
                title = note_sentences[j]
                section_dict[title] = ""
            else:
                if title is not None:
                    section_dict[title] += note_sentences[j]
                    section_dict[title] += '\n'

        item = {
            "dataset": dataset["dataset"][i],
            "encounter_id": dataset["encounter_id"][i],
            "dialogue": dataset["dialogue"][i],
            "note": section_dict
        }

        dataset_list.append(item)
        notes.append(section_dict)

    with open(output_fp, "w", encoding="utf-8") as f:
        json.dump(dataset_list, f, indent=2)
        print("ok")
    # ct_output = {
    #     "dataset": dataset["dataset"],
    #     "encounter_id": dataset["encounter_id"],
    #     "dialogue": dataset["dialogue"],
    #     "note": notes
    # }
    # ct_fn = "process.csv"
    # pd.DataFrame.from_dict(ct_output).to_csv(ct_fn, index=False)


if __name__ == '__main__':
    train, test, valid = load_data(
        train_fp="train.csv",
        test_fp="test.csv",
        valid_fp="valid.csv"
    )
    # print(train)
    # print(test)
    # print(valid)

    trans_format(train, "train_sec.json")
    trans_format(test, "test_sec.json")
    trans_format(valid, "valid_sec.json")
