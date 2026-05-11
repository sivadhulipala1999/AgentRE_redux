import json
import random
import numpy as np
import pandas as pd
import torch
from config.configurator import configs
from trainer.logger import Logger
from trainer.metrics_v2 import EvaluatorRE
from datasets import Dataset
from .utils_trainer import DisabledSummaryWriter, log_exceptions
from data_utils.data_handler_re import DataHandlerRE
from models.base_model import BaseModel as Model

from modules.tools.adaptive_chunking import adaptive_chunk


def init_seed(seed=0):
    if 'seed' in configs['model']:
        seed = configs['model']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


class TrainerSCAgent(object):
    """ 
    属性: 
        logger: Logger, 同时输出到文件和控制台
    """
    data_handler: DataHandlerRE
    logger: Logger
    evaluator: EvaluatorRE = EvaluatorRE()

    def __init__(self, data_handler: DataHandlerRE, logger: Logger):
        self.data_handler = data_handler
        self.logger = logger
        if 'train_num_samples' in configs['data']:
            self.train_num_samples = configs['data']['train_num_samples']
        else:
            self.train_num_samples = 0
        self.num_samples = configs['data']['num_samples']

    @log_exceptions
    def predict(self, model: Model):
        # Reset evaluator and input trace so training metrics don't bleed in
        self.evaluator = EvaluatorRE()
        model.llm_inputs = []

        ds_full: Dataset = self.data_handler.ds_test
        start_idx = self.train_num_samples
        end_idx = min(self.num_samples, len(ds_full))
        ds_predict = ds_full.select(range(start_idx, end_idx))

        self.logger.info(f"Start predicting with {len(ds_predict)} samples (indices {start_idx}..{end_idx-1}).")
        
        proces_res = []
        chunking = configs['model']['chunking']
        for idx, sample in enumerate(ds_predict):
            text = sample['text']

            if chunking: 
                chunks = adaptive_chunk(text)
            
                all_spo_pred = []
                all_history = []
                all_final_output = ""
                last_error_code = 0

                for chunk_idx, chunk in enumerate(chunks):
                    sample_chunk = sample.copy()
                    sample_chunk['text'] = chunk
                    res = model.process_sample(sample_chunk, idx)
                    all_spo_pred.extend(res.get('spo_list_pred', []))
                    all_history.extend(res.get('history', []))
                    all_final_output += res.get('final_output', "") + "\n"
                    last_error_code = res.get('errorCode', 0)
                    self.logger.info(f"idx={idx}, chunk_idx={chunk_idx} - has been processed.")

                combined_res = {
                    'spo_list_pred': all_spo_pred,
                    'errorCode': last_error_code,
                    'history': all_history,
                    'final_output': all_final_output
                }

                self.logger.info(f"idx={idx}, golden={sample['spo_list']}, combined_res={combined_res}")
            
                proces_res.append(combined_res)
            else: 
                res = model.process_sample(sample, idx)
                proces_res.append(res)
                combined_res = res
                # self.logger.info(f"idx={idx}, golden={sample['spo_list']}, res={res}")
                
            golden = sample['spo_list']
            pred = json.dumps(combined_res['spo_list_pred'], ensure_ascii=False)
            self.evaluator.add(golden, pred)
            metric_dict = self.evaluator.get_metric_dict()
            self.logger.info(f"idx={idx}, metric_dict={metric_dict}")
            
        if configs['data']['input_trace']:
            try:
                code_type = configs['llm']['code_version']
            except KeyError:
                code_type = ""
            self.logger.info(
                f"Saving LLM input trace to input_trace_{configs['data']['name']}/llm_input_trace_{configs['model']['name']}_{configs['llm']['model_name']}_{configs['data']['name']}_{code_type}.json...")
            with open(f"input_trace_{configs['data']['name']}/llm_input_trace_{configs['model']['name']}_{configs['llm']['model_name']}_{configs['data']['name']}_{code_type}.json", "w", encoding="utf-8") as f:
                json.dump(model.llm_inputs, f,
                          ensure_ascii=False, indent=4)

        df_pred = pd.concat(
            [pd.DataFrame(proces_res), ds_predict.to_pandas()], axis=1)
        ds_pred = Dataset.from_pandas(df_pred)
        self.logger.info(f"Finish predicting with {len(ds_pred)} samples.")
        self.evaluator.dump_audit_report(
            self.data_handler.data_meta.ofn_report)

        self.data_handler.ds_pred = ds_pred
        self.data_handler.save_results()
        return ds_pred


    
    @log_exceptions
    def evaluate(self, model):
        """ 
        [v2] New evaluation function
        Reads the result of `predict` and then performs an 
        evaluation (`predict` needs to be run first).
        """
        evaluator = EvaluatorRE()       # 新建一个评估器

        self.logger.info("Loading results...")
        self.data_handler.load_results()


        start_idx = self.train_num_samples
        end_idx = min(self.num_samples, len(self.data_handler.ds_test))
        ds_predict = self.data_handler.ds_test.select(range(start_idx, end_idx))

        self.logger.info("Evaluating...")
        spo_list_pred = self.data_handler.ds_pred['spo_list_pred']
        spo_list_golden = ds_predict['spo_list']
        assert len(spo_list_pred) == len(spo_list_golden)
        for pred, golden in zip(spo_list_pred, spo_list_golden):
            evaluator.add(golden, json.dumps(pred, ensure_ascii=False))

        ret_info = evaluator.get_metric_dict()
        self.logger.info(f"Evaluation result: {ret_info}")

        evaluator.dump_audit_report(self.data_handler.data_meta.ofn_report)
        return ret_info