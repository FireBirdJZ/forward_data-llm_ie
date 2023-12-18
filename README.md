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

It is very important to also include an overall breakdown of your repo's file structure. Let people know what is in each directory and where to look if they need something specific. This will also let users know how your repo needs to structured so that your module can work properly

```
firstname-lastname-repo-name/
    - requirements.txt
    - data/ 
        -- eval_articles.csv
        -- train_articles.csv
        -- Keywords_Springer.csv
    - trained_models/
        -- best.model
    - src/
        -- create_train_data/
            --- query_google.py 
            --- extract_fInclude a brief summary of your module here. For example: this module is responsible for classifying pieces of text using a neural network on top of BERT. 

Note: if this is a second or latter iteration of a module, you may reuse the old iteration's README as a starting point (you should still update it). 

## Setup

List the steps needed to install your module's dependencies: 

1. Include what version of Python (e.g. 3.8.12) and what version of pip (e.g. 21.3.1) you used when running your module. If you do not specify these, other users may run into several problems when trying to install dependencies!

2. Include a requirements.txt containing all of the python dependencies needed at your project's root (see this [link](https://stackoverflow.com/questions/31684375/automatically-create-requirements-txt) for instructions on how to create a requirements.txt). If you used a python virtual environment, use `pip freeze -l > requirements.txt` to generate your requirements.txt file. Make sure to include the below line somewhere in this section to tell users how to use your requirements.txt. 
```
pip install -r requirements.txt 
```

3. Additionally, list any other setup required to run your module such as installing MySQL or downloading data files that you module relies on. 

4. Include instructions on how to run any tests you have written to verify your module is working properly. 

It is very important to also include an overall breakdown of your repo's file structure. Let people know what is in each directory and where to look if they need something specific. This will also let users know how your repo needs to structured so that your module can work properly

```
firstname-lastname-repo-name/
    - requirements.txt
    - data/ 
        -- eval_articles.csv
        -- train_articles.csv
        -- Keywords_Springer.csv
    - trained_models/
        -- best.model
    - src/
        -- create_train_data/
            --- query_google.py 
            --- extract_from_url.py
        -- train.py
        -- classify_articles/
            --- main.py
            --- utils.py
   - tests/
       -- data_preprocess_test.py
       -- eval_pretrained_model_test.py
```

Include text description of all the important files / componenets in your repo. 
* `src/create_train_data/`: fetches and pre-processes articles
* `src/train.py`: trains model from pre-processed data
* `src/classify_articles/`: runs trained model on input data
* `data/eval_artcles.csv`: articles to be classified (each row should include an 'id', and 'title')

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





## Issues and Future Work

In this section, please list all know issues, limitations, and possible areas for future improvement. For example:

* High false negative rate for document classier. 
* Over 10 min run time for one page text.
* Replace linear text search with a more efficient text indexing library (such as whoosh)
* Include an extra label of "no class" if all confidence scores low. 


## Change log

Use this section to list the _major_ changes made to the module if this is not the first iteration of the module. Include an entry for each semester and name of person working on the module. For example 

Fall 2021 (Student 1)
* Week of 04/11/2022: added two new functions responsible for ...
* Week of 03/14/2022: fixed bug and added support for ...

Spring 2021 (Student 2)
...

Fall 2020 (Student 3)
...


## References 
include links related to datasets and papers describing any of the methodologies models you used. E.g. 

* Dataset: https://www.kaggle.com/Cornell-University/arxiv 
* BERT paper: Jacob Devlin, Ming-Wei Chang, Kenton Lee, & Kristina Toutanova. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
