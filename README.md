# forward_data-llm_ie In Progress

## Overview

This module can be split into two main parts:
    
1. The first module extracts all of the professors names, positions and research interests from a faculty cs webpage by Classifying similar html subtrees into regions containing records for each professor with pydepta; then extracting the professor's information inside a records with a large language model. 

2. The second module compares two large language models(gpt3.5turbo and Beluga which is a finetuned Llama2) by running a set of web information extraction prompts to test the capability of each model with direct comparison of each other.

A Break Down of the structure of the repo's file structure:
```
.
├── LICENSE
├── README.md
├── requirements.txt
└── src
    ├── experiments
    │   ├── generated_code_directory
    │   │   ├── generated_code.py
    │   │   └── output.txt
    │   ├── langchain_testing.ipynb
    │   ├── natbot_extract_text_json
    │   ├── natbot_generated_program_directory
    │   │   ├── natbot_generated_output.txt
    │   │   └── natbot_generated_program.py
    │   ├── natbot_prompt_program.py
    │   ├── natbot_testing.py
    │   ├── sematic_regular_expressions_testing.py
    │   └── testing_markuplm.py
    ├── extracted_info.json
    ├── main.py
    ├── segmentation
    │   └── pydepta
    │       ├── LICENSE
    │       ├── Makefile.buildbot
    │       ├── pydepta
    │       │   ├── comparing_models
    │       │   │   ├── gpt3.5turbo_illini.json
    │       │   │   └── gpt3.5turbo_mit.json
    │       │   ├── comparing_models.py
    │       │   ├── cot_comparing_models
    │       │   │   ├── gpt3.5turbo_illini2.json
    │       │   │   └── gpt3.5turbo_illini.json
    │       │   ├── depta.py
    │       │   ├── extract_prof_names.py
    │       │   ├── htmls.py
    │       │   ├── illini1_professors.json
    │       │   ├── illini2_professors.json
    │       │   ├── illlini3_professors.json
    │       │   ├── __init__.py
    │       │   ├── llm_benchmark_suite
    │       │   │   ├── benchmark_output.txt
    │       │   │   ├── benchmark_prompts_file.txt
    │       │   │   └── text_analysis
    │       │   │       ├── output_analysis.txt
    │       │   │       └── prompt_analysis.txt
    │       │   ├── LLMBenchmarkSuite.py
    │       │   ├── mdr.py
    │       │   ├── __pycache__
    │       │   │   ├── comparing_models.cpython-311.pyc
    │       │   │   ├── extract_prof_names.cpython-311.pyc
    │       │   │   ├── htmls.cpython-311.pyc
    │       │   │   ├── mdr.cpython-311.pyc
    │       │   │   ├── trees.cpython-311.pyc
    │       │   │   └── trees_cython.cpython-311.pyc
    │       │   ├── saved_faculty_html_files
    │       │   │   ├── csd.cmu.edu.html
    │       │   │   ├── cs.illinois.edu.html
    │       │   │   └── www.eecs.mit.edu.html
    │       │   ├── tests
    │       │   │   ├── __init__.py
    │       │   │   ├── resources
    │       │   │   │   ├── 1.html
    │       │   │   │   ├── 1.txt
    │       │   │   │   ├── 2.html
    │       │   │   │   ├── 2.txt
    │       │   │   │   ├── 3.html
    │       │   │   │   ├── 3.txt
    │       │   │   │   ├── 4.html
    │       │   │   │   ├── 4.txt
    │       │   │   │   ├── 5.html
    │       │   │   │   ├── 5.txt
    │       │   │   │   ├── 6.html
    │       │   │   │   └── 7.html
    │       │   │   └── test_depta.py
    │       │   ├── trees_cython.c
    │       │   ├── trees_cython.py
    │       │   ├── trees_cython.pyx
    │       │   └── trees.py
    │       ├── README.rst
    │       ├── requirements.txt
    │       ├── runtests.sh
    │       ├── setup.py
    │       ├── snapshot1.png
    │       └── test.py
    ├── shopify_extracted_info.json
    ├── vector_db.py
    └── web_extractor.py   
```



## Setup

module's dependencies: 
Python Version 3.11.5
Pip Version 23.3

The Rest of the python dependencies can be installed with:
```
pip install -r requirements.txt 
```

You will also need a openai api-key inorder to use gpt3.5turbo which can be stored inside a config.json file.

Some of the important files/components of the repo:
*`src/segmentation/pydepta/pydepta/`: Runs pydepta to classify regions in HTML and extracts professor info with LLM.
*`src/segmentation/pydepta/pydepta/comparing_models.py`: Class where depta.py calls to classify regions and extract professor info with LLM.
*'src/segmentation/pydepta/pydepta/LLMBenchmarkSuite.py': File which runs gpt3.5turbo and Beluga on different standarized prompts for IE in inorder to test the capability of each model.



## Functional Design (Usage)
Describe all functions / classes that will be available to users of your module. This section should be oriented towards users who want to _apply_ your module! This means that you should **not** include internal functions that won't be useful to the user in this section. You can think of this section as a documentation for the functions of your package. Be sure to also include a short description of what task each function is responsible for if it is not apparent. You only need to provide the outline of what your function will input and output. You do not need to write the pseudo code of the body of the functions. 

* Takes as input a list of strings, each representing a document and outputs confidence scores for each possible class / field in a dictionary
```python
    def classify_docs(docs: list[str]):
        ... 
        return [
            { 'cs': cs_score, 'math': math_score, ..., 'chemistry': chemistry_score },
            ...
        ]
```

* Outputs the weights as a numpy array of shape `(num_classes, num_features)` of the trained neural network 
```python
    def get_nn_weights():
        ...
        return W
```



## Demo video

Include a link to your demo video, which you upload to our shared Google Drive folder (see the instructions for code submission).



## Algorithmic Design 
This section should contain a detailed description of all different components and models that you will be using to achieve your task as well as a diagram. Here is a very basic example of what you should include:

We generate vector representations for each document using BERT, we then train a simple, single-layer fully connected neural network using the documents and labels from the training set.

First, we select a set of labeled text documents `d_1, d_2, …, d_N` from the arxiv dataset available on Kaggle. The documents are randomly partitioned into two sets for training and testing. We use the BERT language model's output as the input to the neural network. Only the weights of the neural network are modified during training. 

After training, we run the trained model to classify the test documents into one of the classes in C. Below is a picture of the architecture of the module. The diagram below was constructed using draw.io 

![design architecture](https://github.com/Forward-UIUC-2021F/guidelines/blob/main/template_diagrams/sample-design.png)

My Algo:

First, We take the input of a faculty webpage url or it's HTML and run it through pydepta to generate regions that classifies similar structure subtrees in the HTML.

Next, we classify these regions with a large language model asking if a region contains Professor names. If the region does contain Professor names we go to the next step in the algorithm, otherwise we skip this region and go onto the next one to classify.

After, We check 3 records at a time inside a region. a record is a list which usually consists of single Professor with information related to the Professor that pydepta classifies in it's subtree such as the Professor's email and Position depending on the webpage. We put these 3 records inside a large language model's and asking the model to output in JSON format the Professor's name, position and research interests. If it doesn't find this information it should fill it with null.(Exact prompts used are in comparing_models.py).

Lastly, we take the returned answer from the language model from the current 3 records and insert it inside a file to store. Then repeat the same steps onto the next 3 records inside that Region.  



## Issues and Future Work

* Localllm Beluga doesn't work with the same exact program in comparing_models.py that was built for gpt3.5turbo.
* In comparing_models.py the large language model will not always output correct JSON output which will cause the program to skip inserting into the file to prevent the program from crashing. In the future I will most likely switch this file from a .json to a .txt file when inserting, or come up with a way to parse the llm's answer and manually build the Json as the large language model is inconsistent of building the correct JSON object even if it's answer could be correct of extraction the correct information.
* For very long webpages it's possible that pydepta can crash.
* Pydepta pip package does not currently work as the project seems to be abandoned as well as it's dependencies. So I modified the files to run inside of pydepta and used only what I needed for my research.



## Change log

*12/19/2023: Changed COT prompt for extracting Professor information comparing_models.py to the COT prompt found in benchmarking llm models with a more exact step process. 




## References 
include links related to datasets and papers describing any of the methodologies models you used. E.g. 

* Dataset: https://www.kaggle.com/Cornell-University/arxiv 
* BERT paper: Jacob Devlin, Ming-Wei Chang, Kenton Lee, & Kristina Toutanova. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
