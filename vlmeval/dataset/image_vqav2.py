from .image_base import ImageBaseDataset
import json
import pandas as pd

class VQAV2Dataset(ImageBaseDataset):

    TYPE = 'VQAV2'

    DATASET_URL = {
        'VQAV2_VAL': None,
    }

    DATASET_MD5 = {
        'VQAV2_VAL': None,
    }

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'VQAV2'
        with open('/mnt/msranlp/wenwan/kt/data/task_data/vqav2/v2_OpenEnded_mscoco_val2014_questions.jsonl', 'r') as f:
            data = [json.loads(line) for line in f][:5000]
        for i, line in enumerate(data):
            line['index'] = i
            line['question'] = line['conversations'][0]['value']
            del line['conversations'], line['video'], line['video_path'], line['image']
        # transformer into pandas dataframe
        self.data = pd.DataFrame(data)
    
    def build_prompt(self, line):
        msgs = []
        msgs.append(dict(type='image', value=line['image_path']))
        msgs.append(dict(type='text', value=line['question']))
        return msgs

    # It returns a dictionary of scores
    @classmethod
    def evaluate(self, eval_file, **kwargs):
        pass
