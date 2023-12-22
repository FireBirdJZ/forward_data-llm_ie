import ast
import requests
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import openai
import json
from typing import List, Any
import re
from pathlib import Path
import os
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

# Choose any model available at https://health.petals.dev
#model_name = "petals-team/StableBeluga2"  # This one is fine-tuned Llama 2 (70B)
#model_name = "bigscience/bloom-560m"  

# Connect to a distributed network hosting model layers
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoDistributedModelForCausalLM.from_pretrained(model_name)

def localLLM(prompt, max_new_tokens_amount=25, max_length=4096):
    inputs = tokenizer(f"{prompt}", return_tensors="pt", max_length=max_length, truncation=True)["input_ids"]
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens_amount)
    response = tokenizer.decode(outputs[0])
    cleaned_response = response.replace("</s>", "")
    return cleaned_response  

config_file_path = '/home/jzale2/forward_data_llm_ie/config.json'

with open(config_file_path, "r") as config_file:
    config = json.load(config_file)
    openai.api_key = config["api_key"]

def chatGpt(prompt):
   # Use the GPT-3 model to generate a response
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        #model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract the message generated by the chat model
    return chat_completion['choices'][0]['message']['content']



class FacultyDataHarvester:
    def __init__(self):
        self.raw_html = ""
        self.raw_text = ""
        self.prof_names_set = set()

    def fetch_html_from_url(self, url):
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            def handle_response(response):
                if response.url == url and response.status == 200:
                    self.raw_html = response.text()

            page.on("response", handle_response)
            page.goto(url)
            browser.close()
        
        return self.raw_html

    def save_html_to_file(self, url, folder_path='saved_faculty_html_files'):
        # Define folder path and file name
        file_name = url.split('//')[-1].split('/')[0] + '.html'  # Example to derive file name from URL
        full_path = os.path.join(folder_path, file_name)

        # Check if the folder exists, create if not
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Write HTML content to the file
        with open(full_path, 'w', encoding='utf-8') as file:
            file.write(self.raw_html)

        print(f"HTML content saved to {full_path}")

    def load_html_from_file(self, url):
        # Define folder path and file name
        folder_path = 'saved_faculty_html_files'
        file_name = url.split('//')[-1].split('/')[0] + '.html'  # Same convention as before
        full_path = os.path.join(folder_path, file_name)

        # Check if the file exists
        if os.path.exists(full_path):
            # Read HTML content from the file
            with open(full_path, 'r', encoding='utf-8') as file:
                self.raw_html = file.read()
            print(f"HTML content loaded from {full_path}")
            return self.raw_html
        else:
            raise FileNotFoundError(f"No saved HTML file found for URL: {url}")


    def extract_text(self):
        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(self.raw_html, 'html.parser')
        # Extract raw text from the parsed HTML
        self.raw_text = soup.get_text(separator=' ', strip=True)
        return self.raw_text

    # write_mode returns the json output in either a .json file or .txt
    # The llm can sometimes mess up generating the .json so its currently safer
    # To make the output in a .txt file
    def find_names_in_region(self, region: List[List[Any]], folder_path="example_compare_models", file_name="compare_models_output", write_mode='txt'):
        
        def classify_region_for_prof_names(region: List[List[Any]]) -> bool:
            print(f"region: {region}")
            #prompt = f"If This Given List has atleast one Professor Name inside of it, output exactly yes or no- {region}"
            # chain of thought prompt
            prompt = (
                "If the given list has at least one Professor's name, output 'Yes' or 'No'. " 
                "To determine this, examine each item in the list and identify if it mentions a Professor. " 
                "A Professor's name is likely to be followed by academic titles or positions. " 
                "Apply this criterion to each item in the list. For example, " 
                "the list provided contains names followed by titles like 'Research Professor', 'Professor Emeritus', " 
                "and specific Professorial chairs, which indicate these are Professors' names. Therefore, the conclusion is 'Yes', " 
                f"the list contains at least one Professor's name.- {region}"
            )
            print(f"prompt: {prompt}")
            answer: str = chatGpt(prompt)
            print(f"answer: {answer}")
            extracted_answer = answer.split(':')[1].strip().lower() if ':' in answer else answer.strip().lower()

            return extracted_answer == "yes"

        if classify_region_for_prof_names(region) is True:
            print("Succesfull Classify")
            counter = 0
            combined_record = []
            for record in region:
                combined_record.append(record)
                counter += 1
                if counter % 3 == 0 or record == region[-1]:
                    
                    # To Improve accuracy of less than 3 records duplicate 1 record
                    #if record == region[-1]:
                    #   combined_record.append(record) 

                    # Flatten the list of lists into a single list of strings
                    flat_list = [item for sublist in combined_record for item in sublist if item is not None]
                    # Now join the strings
                    combined_records_str = " ".join(flat_list)

                    # Updated prompt to request JSON formatted data
                    # prompt = (
                    #     "Output in JSON format the names, faculty positions, and research interests of professors "
                    #     "from the following text. Format the output as a list of dictionaries, each containing 'name', "
                    #     "'position', and 'research_interests' fields. If a field is not available, put 'null' in its place.\n\n"
                    #     f"Text: {combined_records_str}"
                    # )

                    # cot prompt V1
                    # prompt = """
                    #     Understanding the Task: The task involves creating JSON objects for each professor using text data. Each object should 
                    #     include 'name', 'position', and 'research_interests'.

                    #     Analyzing the Text: The text contains information about professors. Each professor's details are in a list format, where 
                    #     the first element is the name, the second the position, and the third (if present) is research interests.

                    #     Handling Missing Information: In cases where information about a professor's research interests is missing, the value 
                    #     should be replaced with 'null'.

                    #     Creating Individual JSON Objects:
                    #     - For each professor, create a dictionary with the keys 'name', 'position', and 'research_interests'.
                    #     - If any of these elements are missing, use 'null' as the value for that key.

                    #     Building the Final JSON Output:
                    #     - Combine these individual dictionaries into a single list.
                    #     - This list represents the entire JSON output, without wrapping it inside another object like "professors".

                    #     Examples:
                    #     - For 'Brian P. Bailey', the JSON object will be: 
                    #     {
                    #         "name": "Brian P. Bailey",
                    #         "position": "Professor",
                    #         "research_interests": null
                    #     }
                    #     - For 'Michael Bailey', the JSON object will be:
                    #     {
                    #         "name": "Michael Bailey",
                    #         "position": "Adjunct Professor",
                    #         "research_interests": null
                    #     }
                    #     - And so on for other professors in the list.

                    #     Final Output: The final output should be a JSON array consisting of these JSON objects. Ensure there is no additional 
                    #     nesting under a "professors" key to avoid errors in data processing.

                    #     Note on JSON Structure: It is crucial to maintain correct JSON syntax and structure, ensuring that no additional or 
                    #     unwarranted layers (like a "professors" key) are added to the final JSON output.
                    #     """ + f"-Text: {combined_records_str}"

                     # COT PromptV2
                    # prompt = f"Text to Process: {combined_records_str}\n"+"""
                    # Understanding the Task:
                    # The task is to create JSON objects for each professor using the provided text data. Each JSON object should include keys for 'name', 'position', and 'research_interests'.

                    # Analyzing the Text:
                    # The provided text data contains details about professors in list format. The first element is the professor's name, the second element (if present) is their position, and the third element (if present) is their research interests.

                    # Handling Missing Information:
                    # If any of these elements are missing (position or research interests), the value should be set to 'null'.

                    # Watch Out For:
                    # pronouns such as he, they, or her. Don't add these anywhere to the JSON object.

                    # Step-by-Step Process with Examples:

                    # Step 1: Extracting Names

                    # Extract the first element from each list for the name.
                    # Example:
                    # Given ['Devin H. Bailey', None, 'Associate Professor'], the 'name' key will be "Devin H. Bailey".
                    # Step 2: Extracting Positions

                    # Extract the second element for the position, using 'null' if it is missing.
                    # Example:
                    # Given ['Devin H. Bailey', None, 'Associate Professor'], the 'position' key will be 'Associate Professor'.
                    # Step 3: Extracting Research Interests

                    # Extract the third element for research interests, using 'null' if it is missing.
                    # Example:
                    # Given ['Devin H. Bailey', None, 'Associate Professor'], the 'research_interests' key will be 'null' since the research interests are missing.
                    # Creating Individual JSON Objects:

                    # For each professor, create a JSON object (dictionary in Python) with the keys 'name', 'position', and 'research_interests'.
                    # Ensure to replace missing elements with 'null'.
                    # Building the Final JSON Output:

                    # Combine these individual JSON objects into a single JSON array.
                    # The final output should be a clean JSON array without any additional nesting.
                    # Final Example:

                    # For 'Devin H. Bailey', the JSON object will be:
                    # {
                    #     "name": "Devin H. Bailey",
                    #     "position": Associate Professor,
                    #     "research_interests": "null"
                    # }
                    # Continue this process for Rest of Professors, following the same steps. Don't write the Final Example JSON Object.
                    # """


                    # COT PromptV2.2
                    prompt = f"Text to Process: {combined_records_str}\n"+"""
                    Understanding the Task:
                    The task is to create JSON objects for each professor using the provided text data. Each JSON object should include keys for 'name', 'position', and 'research_interests'.

                    Analyzing the Text:
                    The provided text data contains details about professors in list format. The first element is the professor's name, the second element (if present) is their position, and the third element (if present) is their research interests.

                    Note: Research interests are often specific and technical, like 'Computer Architecture', 'Multicore Processors & Cloud Computing', or 'Artificial Intelligence + Machine Learning'. Ensure to capture these accurately.

                    Handling Missing Information:
                    If any of these elements are missing (position or research interests), the value should be set to 'null'.

                    Watch Out For:
                    pronouns such as he, they, or her. Don't add these anywhere to the JSON object.

                    Step-by-Step Process with Examples:

                    Step 1: Extracting Names

                    Extract the first element from each list for the name.
                    Example:
                    Given ['Devin H. Bailey', None, 'Associate Professor'], the 'name' key will be "Devin H. Bailey".
                    Given ['Jeff Hamming', 'Armen Wolak (1965) Professor', 'jham@csail.mit.edu', '(617) 103-9122', 'Office: 12-345', 'Algorithms', 'AI for Healthcare', 'Game Theory'] the 'name' key will be Jeff Hamming. Don't get confused by professor positions that look like names such as Armen Wolak. 
                    Step 2: Extracting Positions

                    Extract the second element for the position, using 'null' if it is missing.
                    Example:
                    Given ['Devin H. Bailey', None, 'Associate Professor'], the 'position' key will be 'Associate Professor'.
                    Given ['Jeff Hamming', 'Armen Wolak (1965) Professor', 'jham@csail.mit.edu', '(617) 103-9122', 'Office: 12-345', 'Algorithms', 'AI for Healthcare', 'Game Theory'] the 'position' key will be 'Armen Wolak (1965) Professor'.
                    Step 3: Extracting Research Interests

                    Extract the third element for research interests, using 'null' if it is missing. Ensure to capture all available information, resorting to 'null' only when certain information is genuinely missing.
                    Example:
                    Given ['Devin H. Bailey', None, 'Associate Professor'], the 'research_interests' key will be 'null' since the research interests are missing.
                    Given ['Jeff Hamming', 'Armen Wolak (1965) Professor', 'jham@csail.mit.edu', '(617) 103-9122', 'Office: 12-345', 'Algorithms', 'AI for Healthcare', 'Game Theory'] the 'research_interests' key will be 'Algorithms, AI for Healthcare, Game Theory'. 

                    Creating Individual JSON Objects:

                    For each professor, create a JSON object (dictionary in Python) with the keys 'name', 'position', and 'research_interests'.
                    Ensure to replace missing elements with 'null'.
                    Building the Final JSON Output:

                    Combine these individual JSON objects into a single JSON array.
                    The final output should be a clean JSON array without any additional nesting.
                    Final Example:

                    For 'Devin H. Bailey', the JSON object will be:
                    {
                        "name": "Devin H. Bailey",
                        "position": Associate Professor,
                        "research_interests": "null"
                    }
                    For 'Jeff Hamming', the JSON object will be:
                    {
                        "name": "Jeff Hamming",
                        "position": "Armen Wolak (1965) Professor",
                        "research_interests": "Algorithms, AI for Healthcare, Game Theory"
                    } 
                    Continue this process for Rest of Professors, following the same steps. Don't write the Final Example JSON Object.
                    """
                    answer = chatGpt(prompt)
                    print(f"answer: {answer}")

                    def append_to_json_file(file_path, data):
                        # Check if the file exists
                        if Path(file_path).exists():
                            # Read the existing content
                            with open(file_path, 'r+', encoding='utf-8') as file:
                                try:
                                    file_data = json.load(file)
                                    if isinstance(file_data, list):  # Check if the file contains a list
                                        # Append the new data
                                        file_data.append(data)
                                        # Write back to the file
                                        file.seek(0)
                                        json.dump(file_data, file, indent=4)
                                        file.truncate()  # Remove any remaining parts of old data
                                except json.JSONDecodeError:
                                    print("File is not valid JSON. Appending not possible.")
                        else:
                            # Create a new file with the data in a list
                            with open(file_path, 'w', encoding='utf-8') as file:
                                json.dump([data], file, indent=4)

                    def append_to_txt_file(file_path, data):
                        with open(file_path, 'a', encoding='utf-8') as file:
                            file.write(str(data) + '\n\n')

                    # folder_path = 'cot_comparing_models'
                    # file_name = 'gpt3.5_illini.json'
                    # full_path = os.path.join(folder_path, file_name)

                    # if answer.strip().lower() == 'null':
                    #     print("No relevant information found")
                    #     continue
                    # else:
                    #     try:
                    #         prof_info_list = json.loads(answer)

                    #         # Check if the folder exists, create if not
                    #         if not os.path.exists(folder_path):
                    #             os.makedirs(folder_path)

                    #         # Append data to the JSON file within the folder
                    #         append_to_json_file(full_path, prof_info_list)

                    #         for prof_info in prof_info_list:
                    #             try:
                    #                 name = prof_info.get('name')
                    #                 if name:
                    #                     formatted_name = name.replace('"', '').replace("'", '')
                    #                     self.prof_names_set.add(formatted_name)
                    #             except AttributeError:
                    #                 print("Something is wrong with the data format.")
                    #                 continue  # Continue to the next iteration of the loop
                    #     except json.JSONDecodeError:
                    #         print("Received invalid JSON data")
                    def process_data(answer, folder_path, file_name, write_mode='txt'):
                        full_path = os.path.join(folder_path, file_name)
                        if answer.strip().lower() == 'null':
                            print("No relevant information found")
                        else:
                            try:
                                prof_info_list = json.loads(answer) if write_mode == 'json' else answer
                                if not os.path.exists(folder_path):
                                    os.makedirs(folder_path)

                                if write_mode == 'json':
                                    append_to_json_file(full_path, prof_info_list)
                                    
                                if write_mode == 'txt':
                                    append_to_txt_file(full_path, prof_info_list)
                            except json.JSONDecodeError:
                                print("Received invalid JSON data")

                    process_data(answer, folder_path, f"{file_name}.{write_mode}", write_mode)            
                    combined_record = []
                
        else: print("Failed Classify")
        print(f"Professor Name Set: {self.prof_names_set}")