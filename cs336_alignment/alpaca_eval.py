from cs336_alignment import utils
import csv
import os
import pandas as pd
from vllm import LLM 
from tqdm import tqdm
import json

def process_alpaca_file(file_path):
    data = utils.read_jsonl(file_path)
    return data

def generate_alpaca_responses(alpaca_data, model):
    # make question prompt
    questions = []
    for example in alpaca_data:
        question = example['instruction']
        question_prompt = f"""{question}"""
        questions.append(question_prompt)
    # get the model output
    model_output = utils.generate_from_llama(questions, "/data/Meta-Llama-3-8B", model=model)
    # parse answers from output
    parsed_outputs = [utils.parse_alpaca_response(output) for output in model_output]

    return model_output, parsed_outputs



if __name__ == "__main__":

    llm = LLM(model="/data/Meta-Llama-3-8B")
    alpaca_test_path = "/home/c-jjian/assignments/spring2024-assignment5-alignment/data/alpaca_eval/alpaca_eval.jsonl"
    alpaca_data = process_alpaca_file(alpaca_test_path)
    _, parsed_outputs = generate_alpaca_responses(alpaca_data, model=llm)

    # add the results to the data
    for i in range(len(alpaca_data)):
        alpaca_data[i]['output'] = parsed_outputs[i]
        alpaca_data[i]['generator'] = "llama-3-8b-base"
    
    # write the results to json

    json.dump(alpaca_data, open("/home/c-jjian/assignments/spring2024-assignment5-alignment/results/alpaca_eval/llama_3_8b_results.formatted.json", "w"))

    utils.write_jsonl("/home/c-jjian/assignments/spring2024-assignment5-alignment/results/alpaca_eval/llama_3_8b_results.jsonl", alpaca_data)


