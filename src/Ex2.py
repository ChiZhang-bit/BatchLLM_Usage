import tiktoken
import os
import re
import csv

from tqdm import trange
from openai import OpenAI
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# The maximum number of tokens in the input and output
MAX_INPUT_TOKENS = 6192*4
MAX_OUTPUT_TOKENS = 2048

os.environ["OPENAI_API_KEY"] = "sk-BHBJElSOXC6F7ZZwF76f4718C0D9417188De33Bd2967753c"


def load_data(file_path):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            encounter_id = row['encounter_id']
            dialogue = row['dialogue']
            note = row['note']
            data.append({
                'encounter_id': encounter_id,
                'dialogue': dialogue,
                'note': note
            })
    return data


def main(train_data, test_data):
    # Setup the LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-1106",
        temperature=0.2,
        max_tokens=MAX_OUTPUT_TOKENS,
        openai_api_key="sk-BHBJElSOXC6F7ZZwF76f4718C0D9417188De33Bd2967753c",
        openai_api_base="https://cd.aiskt.com/v1"
    )

    # Setup the chain
    prompt = PromptTemplate(
        input_variables=["examples", "dialogue1", "dialogue2"],
        template=
        """Write clinical note reflecting doctor-patient dialogue. 
        I provide you with two dialogues(1,2) and you should generate two clinical notes. 
        DIALOGUES1: [{dialogue1}]
        DIALOGUES2: [{dialogue2}]
        EXAMPLES: [{examples}]
        Write clinical note reflecting doctor-patient dialogue. Use the example notes above to decide the structure of the clinical note. Do not make up information.
        I want you to give your output in a markdown table where the first column is the id of dialogues and the second is the clinical note for each dialogue.
        """
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    # Run the chain to generate predictions
    predictions = []

    for i in trange(20):

        dialogue1 = test_data[i]['dialogue']
        dialogue2 = test_data[i+1]['dialogue']

        # Load the example:
        examples = ""
        examples += train_data[0]['note']

        prompt_length = llm.get_num_tokens(prompt.format(
            dialogue1=dialogue1,
            dialogue2=dialogue2,
            examples=examples
        ))
        # new_example = train_data[1]['note']
        # if (prompt_length + llm.get_num_tokens(new_example)) < MAX_INPUT_TOKENS:
        #     examples += new_example

        prediction = chain.invoke(
            input={
                "dialogue1": dialogue1,
                "dialogue2": dialogue2,
                "examples": examples
            }
        )

        # GPT sometimes makes the mistake of placing Imaging by itself, which won't be picked up by the official
        # evaluation script. So we prepend "RESULTS" to it here, which will be picked up.
        # prediction = re.sub(
        #     r"(\n|^)(?!.*\\nRESULTS.*)[Ii][Mm][Aa][Gg][Ii][Nn][Gg](:)?", r"\1RESULTS\nImaging\2", prediction
        # )

        predictions.append(prediction['text'])

    with open("result2.txt", "w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(str(prediction) + "\n")

    return


def test():
    # Setup the LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-1106",
        temperature=0.2,
        max_tokens=MAX_OUTPUT_TOKENS,
        openai_api_key="sk-BHBJElSOXC6F7ZZwF76f4718C0D9417188De33Bd2967753c",
        openai_api_base="https://cd.aiskt.com/v1"
    )

    # Setup the chain
    prompt = PromptTemplate(
        input_variables=["member"],
        template="test {member}"
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    prediction = chain.invoke(
        input={
            "member": "you"
        }
    )
    print(prediction)


if __name__ == '__main__':

    train_data = load_data("../data/train.csv")
    test_data = load_data("../data/test.csv")
    valid_data = load_data("../data/valid.csv")

    main(train_data, test_data)