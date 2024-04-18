# -*- coding: utf-8 -*-

import re
from transformers import BartForConditionalGeneration, BertTokenizer


def cleanfuc(text):
    """clean text"""
    pat = re.compile(r'\[[^()]*\]', re.S)  
    text = re.sub(pat, '', text)
    return text


if __name__ == '__main__':
    '''
        QG
    '''

    model_path = './output/gec/1'
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    n = 20
    while True:
        sent = input('>>')
        ids = tokenizer.encode(sent, return_tensors='pt')
        
        outputs = model.generate(eos_token_id=tokenizer.sep_token_id,
                                decoder_start_token_id=tokenizer.cls_token_id,
                                num_beams=n,
                                num_return_sequences=n,
                                input_ids=ids, 
                                use_cache=True,
                                max_length=50,
                                no_repeat_ngram_size=2,
                                early_stopping=True)
        
        # outputs = model.generate(eos_token_id=tokenizer.sep_token_id,
        #                         decoder_start_token_id=tokenizer.cls_token_id,
        #                         do_sample=True,
        #                         top_k=40, 
        #                         top_p=0.9, 
        #                         temperature=0.9,
        #                         num_return_sequences=10,
        #                         input_ids=ids, 
        #                         use_cache=True,
        #                         max_length=50)

        for output in outputs:
            print(cleanfuc(''.join(tokenizer.decode(output, skip_special_tokens=True)).replace(' ', '')))