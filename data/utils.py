import re
import json
import os
import torch.distributed as dist
import code_utils


def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename, code_utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))

    dist.barrier()

    if code_utils.is_main_process():
        # combine results from all processes
        result = []

        for rank in range(code_utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)

    return final_result_file


class Dict2Object:
    def __init__(self, input_dict: dict):
        my_dict = input_dict.copy()
        for k, v in input_dict.items():
            if isinstance(v, dict):
                my_dict[k] = Dict2Object(v)

        self._my_dict = my_dict

    def __getattr__(self, name):
        try:
            return self._my_dict[name]
        except KeyError:
            raise AttributeError(f"'MyObject' object has no attribute '{name}'")

    def __getitem__(self, item):
        return getattr(self, item)

    def __setstate__(self, state):
        my_dict = state['_my_dict']
        for k, v in state.items():
            if isinstance(v, dict):
                my_dict[k] = Dict2Object(v)
        self._my_dict = my_dict
        return my_dict

    def __str__(self):
        return str(self._my_dict)

    def to_dict(self):
        return self._my_dict.copy()