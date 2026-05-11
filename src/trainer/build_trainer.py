from config.configurator import configs
import importlib

def build_trainer(data_handler, logger):
    trainer_name = configs['train']['trainer'] if ('trainer' in configs['train'] and configs['train']['trainer']) else 'Trainer'
    
    module_path = 'trainer.' + trainer_name
    expected_class_name = trainer_name.replace('_', '')

    trainers = importlib.import_module(module_path)
    for attr in dir(trainers):
        if attr.lower() == expected_class_name.lower():
            return getattr(trainers, attr)(data_handler, logger)
        elif hasattr(trainers, 'Trainer'):
            return getattr(trainers, 'Trainer')(data_handler, logger)
    else:
        raise NotImplementedError('Trainer Class {} is not defined in {}'.format(trainer_name, 'trainer.trainer'))
