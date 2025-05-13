import json


def format_sample(sample):
    return {
        "text": sample['text'],
        "spo_list": sample['spo_list']
    }


def format_reflexion_sample(sample):
    return {
        "text": sample['text'],
        "golden": sample['golden'],
        "pred": sample['pred'],
        "reflexion": sample['reflexion']
    }


def format_sample_str(sample):
    return json.dumps(format_sample(sample), ensure_ascii=False)


def format_reflexion_sample_str(sample):
    return json.dumps(format_reflexion_sample(sample), ensure_ascii=False)
