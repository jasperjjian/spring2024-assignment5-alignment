from cs336_alignment import utils
import csv
import os
import pandas as pd
from vllm import LLM 
from tqdm import tqdm

def process_gsm_file(file_path):
    data = utils.read_jsonl(file_path)
    for example in data:
        example['answer'] = example['answer'].split("#### ")[1]
    return data

def evaluate_gsm(gsm_data, model):
    # make question prompt
    questions = []
    for example in gsm_data:
        question = example['question']
        question_prompt = f"""{question}
        Answer:
        """
        questions.append(question_prompt)
    # get the model output
    model_output = utils.generate_from_llama(questions, "/data/Meta-Llama-3-8B", model=model)
    # parse answers from output
    parsed_outputs = [utils.parse_gsm8k_response(output) for output in model_output]

    # get the number of correct answers
    verified_outputs = [parsed_outputs[i] == gsm_data[i]['answer'] for i in range(len(parsed_outputs))]    

    return sum(verified_outputs), model_output, parsed_outputs

if __name__ == "__main__":

    gsm_test_path = "/home/c-jjian/assignments/spring2024-assignment5-alignment/data/gsm8k/test.jsonl"
    
    # make a dataframe to store results with headers for the columns: topic and accuracy
    compiled_results = pd.DataFrame(columns=['dataset', 'accuracy'])

    #total_questions = 0
    #total_correct = 0

    llm = LLM(model="/data/Meta-Llama-3-8B")    
    # iterate through the topics
    # get the file path for the topic
   
    # get the data from the file
    gsm_data = process_gsm_file(gsm_test_path)
    # get the number of questions

    total_questions = len(gsm_data)

    # get the number of correct answers
    correct, outputs, parsed_outputs = evaluate_gsm(gsm_data, model=llm)
    total_correct = correct

    # serialize the results
    save_path = os.path.join("/home/c-jjian/assignments/spring2024-assignment5-alignment/results/gsm8k/" + "results.json")
    serialized = utils.serialize_question_outputs([example["question"] for example in gsm_data], outputs, parsed_outputs, [example["answer"] for example in gsm_data], save_path)

    # get the accuracy
    accuracy = total_correct / total_questions
    # store the results in the dataframe
    new_results = pd.DataFrame({'dataset': "gsm8k_zero_shot", 'accuracy': accuracy}, index=[0])
    compiled_results = pd.concat([compiled_results, new_results], ignore_index=True)
    
    print(compiled_results)
    compiled_results.to_csv("/home/c-jjian/assignments/spring2024-assignment5-alignment/results/gsm8k/overall_results.csv", index=False)

