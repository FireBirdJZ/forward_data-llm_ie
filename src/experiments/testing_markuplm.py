import requests
from bs4 import BeautifulSoup

# Currently not getting good results with this model, rewrite in function later.
MODEL_STR = "microsoft/markuplm-base-finetuned-websrc"

from transformers import MarkupLMProcessor, MarkupLMForQuestionAnswering

processor = MarkupLMProcessor.from_pretrained(MODEL_STR)
model = MarkupLMForQuestionAnswering.from_pretrained(MODEL_STR)

headers = {
'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36',
}

sample_url = "https://news.ycombinator.com/item?id=35550567"

page = requests.get(sample_url, headers=headers)

soup = BeautifulSoup(page.content, "html.parser")

body = soup.find('body')

html_string = str(body)

question = "What is the title of the webpage?"

encoding = processor(html_string, questions=question, return_tensors="pt", truncation="only_second", 
                     stride=100, max_length=512, return_overflowing_tokens=True, padding=True)

del encoding['overflow_to_sample_mapping']
encoding['token_type_ids'] = encoding['token_type_ids'].fill_(0)

#dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'xpath_tags_seq', 'xpath_subs_seq'])

for k,v in encoding.items():
    print(k,v.shape)

import torch
import torch.nn.functional as F
import numpy as np

with torch.no_grad():
    outputs = model(**encoding)

print(outputs.start_logits.shape)
print(outputs.end_logits.shape)

torch.Size([15, 512])
torch.Size([15, 512])

start_probs = F.softmax(outputs.start_logits, dim=1).numpy()
end_probs = F.softmax(outputs.end_logits, dim=1).numpy()

i = 0

start_index = np.argmax(start_probs[i])
end_index = np.argmax(end_probs[i])
confidence = max(start_probs[i]) * max(end_probs[i])

print(f"Span Start Index: {start_index}")
print(f"Span End Index: {end_index}")
print(f"Span Confidence: {confidence:.4f}")

predict_answer_tokens = encoding.input_ids[0, start_index : end_index + 1]
answer = processor.decode(predict_answer_tokens, skip_special_tokens=True)
print(f"Answer: {answer}")


# We also calculate the index where the question ends, for filtering answers
question_index = encoding[0].tokens.index('</s>')

# Compute number of segments to iterate over
n_segments = encoding['input_ids'].shape[0]

# Maximum number of characters allowed in answers
max_answer_len = 50

# Minimum confidence
min_confidence = 0.9

import torch.nn.functional as F
import numpy as np

answers = []

for i in range(n_segments):
    
    start_index = np.argmax(start_probs[i])
    end_index = np.argmax(end_probs[i])
    confidence = max(start_probs[i]) * max(end_probs[i])

    if end_index > start_index and end_index - start_index <= max_answer_len and start_index > question_index and end_index > question_index and confidence > min_confidence:

        predict_answer_tokens = encoding.input_ids[0, start_index : end_index + 1]
        answer = processor.decode(predict_answer_tokens, skip_special_tokens=True)
        
        answers.append({"answer": answer, "confidence": confidence})

#print(answers[0])