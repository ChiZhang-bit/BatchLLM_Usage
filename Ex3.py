import os
import json

from tqdm import trange
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


# The maximum number of tokens in the input and output
MAX_INPUT_TOKENS = 6192 * 4
MAX_OUTPUT_TOKENS = 2048

os.environ["OPENAI_API_KEY"] = "sk-BHBJElSOXC6F7ZZwF76f4718C0D9417188De33Bd2967753c"

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        reader = json.load(jsonfile)
        for row in reader:
            encounter_id = row['encounter_id']
            dialogue = row['dialogue']
            if 'HISTORY OF PRESENT ILLNESS' in row['note']:
                history_of_present_illness = row['note']['HISTORY OF PRESENT ILLNESS']
            elif 'HPI' in row['note']:
                history_of_present_illness = row['note']['HPI']
            data.append({
                'encounter_id': encounter_id,
                'dialogue': dialogue,
                'history_of_present_illness': history_of_present_illness
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
        input_variables=["examples", "dialogue1", "dialogue2", "dialogue3", "dialogue4"],
        template=
        """Write the history_of_present_illness reflecting doctor-patient dialogue. 
        I provide you with four dialogues(1,2,3,4) and you should generate four history_of_present_illness. 
        DIALOGUES1: [{dialogue1}]
        DIALOGUES2: [{dialogue2}]
        DIALOGUES3: [{dialogue3}]
        DIALOGUES4: [{dialogue4}]
        EXAMPLES: [{examples}]
        Write history_of_present_illness reflecting doctor-patient dialogue. Use the example history_of_present_illness above to decide the structure of the history_of_present_illness. Do not make up information. 
        I want you to give your output in a markdown table where the first column is the id of dialogues and the second is the history_of_present_illness for each dialogue.
        """
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    # Run the chain to generate predictions
    predictions = []

    for i in trange(3):
        i *= 4
        dialogue1 = test_data[i]['dialogue']
        dialogue2 = test_data[i+1]['dialogue']
        dialogue3 = test_data[i+2]['dialogue']
        dialogue4 = test_data[i+3]['dialogue']
        # Load the example:
        examples = ""
        examples += train_data[0]['history_of_present_illness']

        """
        prompt_length = llm.get_num_tokens(prompt.format(
            dialogue1=dialogue1,
            dialogue2=dialogue2,
            examples=examples
        ))
        new_example = train_data[1]['history_of_present_illness']
        if (prompt_length + llm.get_num_tokens(new_example)) < MAX_INPUT_TOKENS:
            examples += new_example
        """

        prediction = chain.invoke(
            input={
                "dialogue1": dialogue1,
                "dialogue2": dialogue2,
                "dialogue3": dialogue3,
                "dialogue4": dialogue4,
                "examples": examples
            }
        )

        predictions.append(prediction['text'])


    with open("result3.txt", "w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(str(prediction) + "\n")

    return


if __name__ == '__main__':
    train_data = load_data("../data/section/train_sec.json")
    test_data = load_data("../data/section/test_sec.json")
    main(train_data, test_data)
