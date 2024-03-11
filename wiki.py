import pandas as pd
import json
import requests
import tagme
import re
import wikipedia
from tqdm import tqdm
import pickle

input_file = sys.argv[1]
output_file = sys.argv[2]

informative_train_df = pd.read_csv(input_file, sep = '\t')
print(informative_train_df)


def preprocess(text):
    return re.sub(r"[:,\n\r;@#!]|http[s]?://\S+|\.\s*|[^\x00-\x7F]+", '', text)



informative_train_df['tweet_text'] =  informative_train_df['tweet_text'].apply(lambda x: preprocess(x) )


informative_train_df['final_text'] = informative_train_df['tweet_text']+ informative_train_df['event_name']

#informative_train_df['tweet_text'][2]

def final_wiki_text(text):
    #IP_ADDRESS = "https://rel-entity-linker.d4science.org/"
    entities= []
    wiki_entities=[]
    final_wiki_tags=[]
    tagme.GCUBE_TOKEN = '<add tagme token>'
    annotations = tagme.annotate(text)

    # Print annotations with a score higher than 0.1
    if (annotations != None):
        for ann in annotations.get_annotations(0.15):
            entities.append(ann.mention.split(',')[0])
            wiki_entities.append(ann.entity_title)
            
    #print(wiki_entities)
            
           
    for entity in wiki_entities:
        try:
            page = wikipedia.page(entity)
            text+= page.summary
        except:
            text+= ''
    
    annotations = tagme.annotate(text)
     # Print annotations with a score higher than 0.1
    if (annotations != None):
        for ann in annotations.get_annotations(0.15):
            entities.append(ann.mention.split(',')[0])
            final_wiki_tags.append(ann.entity_title)
    return text,  final_wiki_tags
        

final_text=[]
final_tags=[]
final_ls=[]
for i,row in tqdm(informative_train_df.iterrows(), total= len(informative_train_df)):
    text, final_wiki_tags = final_wiki_text(row['final_text'])
    final_text.append(text)
    final_tags.append(final_wiki_tags)

final_ls.append([final_text,final_tags])



with open(output_file, 'wb') as f:
    pickle.dump(final_ls, f)





