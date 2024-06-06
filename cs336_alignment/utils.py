from vllm import LLM, SamplingParams
import string
import json

def generate_from_llama(prompts : list[str], model_path : str, max_tokens : int = 1024, model=None):
    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, stop=["```\n", "#Query:", '\n```']
    )
    concatenated_prompts = []
    for prompt in prompts:
        concatenate = f"""
        # Instruction
        Below is a list of conversations between a human and an AI assistant (you).
        Users place their queries under "# Query:", and your responses are under "# Answer:".
        You are a helpful, respectful, and honest assistant.
        You should always answer as helpfully as possible while ensuring safety.
        Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
        Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
        Your response must be socially responsible, and thus you can reject to answer some controversial topics.

        # Query:
        ```{prompt}```

        # Answer:
        ```
        """
        concatenated_prompts.append(concatenate)
    # Create an LLM.
    if model is None:
        llm = LLM(model=model_path)
    else:
        llm = model
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(concatenated_prompts, sampling_params)
    # Print the outputs.
    
    """for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")"""

    return [output.outputs[0].text for output in outputs]

def parse_mmlu_results(output, mmlu_example):
    #translator = str.maketrans('', '', string.punctuation)
    generated_texts = output.split(". ")
    #generated_texts = [sentence.translate(translator).strip() for sentence in output_sentences if sentence != ""]
    generated_texts = [generated_text.replace("The correct answer is ", "").strip() for generated_text in generated_texts if generated_text != ""]
    
    # get whether it is one of A, B, C, or D, or none of the above
    
    if "A" in generated_texts or "A." in generated_texts:
        return "A"
    elif "B" in generated_texts or "B." in generated_texts:
        return "B"
    elif "C" in generated_texts or "C." in generated_texts:
        return "C"
    elif "D" in generated_texts or "D." in generated_texts:
        return "D"
    if mmlu_example["options"][0].replace(".", "") in generated_texts:
        return "A"
    elif mmlu_example["options"][1].replace(".", "") in generated_texts:
        return "B"
    elif mmlu_example["options"][2].replace(".", "") in generated_texts:
        return "C"
    elif mmlu_example["options"][3].replace(".", "") in generated_texts:
        return "D"
    else:
        return None
    
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def write_jsonl(file_path, json_list):
    with open(file_path, 'w') as file:
        for json_obj in json_list:
            file.write(json.dumps(json_obj) + '\n')
    
def parse_gsm8k_response(output):
    # get the last number in the output
    translator = str.maketrans('', '', string.punctuation)
    output_cleaned = output.translate(translator).strip()
    numbers = [int(s) for s in output_cleaned.split() if s.isdigit()]
    if len(numbers) > 0:
        return str(numbers[-1])
    else:
        return None
    
def parse_alpaca_response(output):
    return output.strip()

def serialize_question_outputs(questions, outputs, parsed, answers, save_path):
    serialized = []
    for i in range(len(questions)):
        serialized.append({'question' : questions[i], 'output' : outputs[i], 'parsed_output' : parsed[i], 'answer' : answers[i]})
    
    with open(save_path, 'w') as f:
        json.dump(serialized, f)
    
    return serialized