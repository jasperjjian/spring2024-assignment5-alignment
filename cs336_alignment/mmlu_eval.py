from cs336_alignment import utils
import csv
import os
import pandas as pd
from vllm import LLM 
from tqdm import tqdm

def process_mmlu_file(file_path, topic):
    data = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            # Extract the question, options, and answer
            question = row[0]
            options = row[1:5]
            answer = row[5]
            # Store them in a dictionary
            data.append({
                'subject' : topic,
                'question': question,
                'options': options,
                'answer': answer
            })
    return data

def evaluate_mmlu(mmlu_data, topic, model):
    topic = topic.split("_test")[0]
    # make question prompt
    questions = []
    for example in mmlu_data:
        question = example['question']
        options = example['options']
        question_prompt = f"""Answer the following multiple choice question about {topic}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).
        Question: {question}
        A. {options[0]}
        B. {options[1]}
        C. {options[2]}
        D. {options[3]}
        Answer:
        """
        questions.append(question_prompt)
    # get the model output
    model_output = utils.generate_from_llama(questions, "/data/Meta-Llama-3-8B", model=model)
    # parse answers from output
    parsed_outputs = [utils.parse_mmlu_results(output, mmlu_data[i]) for i, output in enumerate(model_output)]

    # get the number of correct answers
    verified_outputs = [parsed_outputs[i] == mmlu_data[i]['answer'] for i in range(len(parsed_outputs))]    

    return sum(verified_outputs), model_output, parsed_outputs



if __name__ == "__main__":

    # get the folder containing the MMLU files
    mmlu_dir = "/home/c-jjian/assignments/spring2024-assignment5-alignment/data/mmlu/test"
    # get the topics from the directory files without their csv extension alphabetically
    mmlu_topics = [os.path.splitext(file)[0] for file in os.listdir(mmlu_dir)]
    mmlu_topics.sort()
    
    # make a dataframe to store results with headers for the columns: topic and accuracy
    compiled_results = pd.DataFrame(columns=['topic', 'accuracy'])

    total_questions = 0
    total_correct = 0

    llm = LLM(model="/data/Meta-Llama-3-8B")    
    # iterate through the topics
    for topic in tqdm(mmlu_topics):
        # get the file path for the topic
        file_path = os.path.join(mmlu_dir, topic + ".csv")
        # get the data from the file
        mmlu_data = process_mmlu_file(file_path, topic)
        # get the number of questions
        total_questions += len(mmlu_data)
        # get the number of correct answers
        correct, outputs, parsed_outputs = evaluate_mmlu(mmlu_data, topic, model=llm)
        total_correct += correct

        # serialize the results
        save_path = os.path.join("/home/c-jjian/assignments/spring2024-assignment5-alignment/results/mmlu", topic + "_results.json")
        serialized = utils.serialize_question_outputs([example["question"] for example in mmlu_data], outputs, parsed_outputs, [example["answer"] for example in mmlu_data], save_path)

        # get the accuracy
        accuracy = correct / len(mmlu_data)
        # store the results in the dataframe
        new_results = pd.DataFrame({'topic': topic, 'accuracy': accuracy}, index=[0])
        compiled_results = pd.concat([compiled_results, new_results], ignore_index=True)
    
    # get the overall accuracy
    overall_accuracy = total_correct / total_questions
    # store the overall accuracy in the dataframe
    overall_results = pd.DataFrame({'topic': 'overall', 'accuracy': overall_accuracy}, index=[0])
    compiled_results = pd.concat([compiled_results, overall_results], ignore_index=True)
    # save the results to a csv file
    print(compiled_results)
    compiled_results.to_csv("/home/c-jjian/assignments/spring2024-assignment5-alignment/results/mmlu/overall_results.csv", index=False)

