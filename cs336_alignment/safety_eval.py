from cs336_alignment import utils
import csv
import os
import pandas as pd
from vllm import LLM 
from tqdm import tqdm
import json

def process_safety_file(file_path):
    # read from a csv to a dictionary where the first row is are the keys
    data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def generate_safety_responses(safety_data, model):
    # make question prompt
    questions = []
    for example in safety_data:
        question = example['prompts_final']
        question_prompt = f"""{question}"""
        questions.append(question_prompt)
    # get the model output
    model_output = utils.generate_from_llama(questions, "/data/Meta-Llama-3-8B", model=model)
    # parse answers from output
    parsed_outputs = [utils.parse_alpaca_response(output) for output in model_output]

    return model_output, parsed_outputs



if __name__ == "__main__":

    llm = LLM(model="/data/Meta-Llama-3-8B")
    safety_test_file = "/home/c-jjian/assignments/spring2024-assignment5-alignment/data/simple_safety_tests/simple_safety_tests.csv"
    safety_file = process_safety_file(safety_test_file)
    _, parsed_outputs = generate_safety_responses(safety_file, model=llm)

    # add the results to the data
    final_output = []
    for i in range(len(safety_file)):
        outdict = {}
        outdict['output'] = parsed_outputs[i]
        outdict['prompts_final'] = safety_file[i]['prompts_final']
        outdict['harm_area'] = safety_file[i]['harm_area']
        outdict['generator'] = "llama-3-8b-base"
        final_output.append(outdict)
    
    # write the results to json

    json.dump(final_output, open("/home/c-jjian/assignments/spring2024-assignment5-alignment/results/simple_safety_tests/llama_3_8b_results.json", "w"))
    utils.write_jsonl("/home/c-jjian/assignments/spring2024-assignment5-alignment/results/simple_safety_tests/llama_3_8b_results.jsonl", final_output)


