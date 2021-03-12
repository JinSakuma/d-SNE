import numpy as np
import torch


def set_random_seeds(seed):
    """Set random seeds to ensure reproducibility."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    
def log_dict(results, output_dict, phase, domain=None):
    if phase == 'train':
        loss, Lcls, Ldsne, acc = results
        output_dict['train_acc'].append(acc)
        output_dict['train_loss'].append(loss)
        output_dict['train_Lcls'].append(Lcls)
        output_dict['train_Ldsne'].append(Ldsne)
    else:
        loss, acc = results
        output_dict['val_acc_{}'.format(domain)].append(acc)
        output_dict['val_loss_{}'.format(domain)].append(loss)
#         output_dict['val_Lcls_{}'.format(domain)].append(Lcls)
#         output_dict['val_Ldsne_{}'.format(domain)].append(Ldsne)
        
    return output_dict
        