import torch as th
import numpy as np
from functools import partial
from inspect import getmembers, isfunction, isclass, signature
import re 

def get_generic_bends(transform, transform_name, duration=30, layer=0, start = 0, end = 30):
    bends = {'layer': layer,  'RGB':True}
    pattern = re.compile('modulation')
    sign_string = str(signature(transform.__init__))
    if re.search(pattern, sign_string ):

        intro_tl = th.zeros(start)
        end_tl = th.ones(duration-end)
        if transform_name == 'Translate':
            active_tl = np.concatenate([np.linspace(0, 1, end-start)])
            
            x_tl = np.concatenate([intro_tl, active_tl, end_tl])
            y_tl = np.zeros(duration)
            modulate = (th.tensor([x_tl, y_tl]).float().T).unsqueeze(1)
        elif transform_name == 'Rotate': 
            active_tl = np.concatenate([np.linspace(0, 360, (end-start)//2)] * 2)
            x_tl = np.concatenate([intro_tl, active_tl, end_tl])
            modulate = th.tensor([x_tl]).float().squeeze()
            
        elif transform_name == 'Roll':
            active_tl = np.concatenate([np.linspace(0, 1, (end-start)//3)]*3)
            x_tl = np.concatenate([intro_tl, active_tl, end_tl])
            modulate = th.tensor([x_tl]).float().squeeze()
        
        modulated_func = lambda batch: partial(transform)(batch)
        bends.update({'modulation':modulate, 'transform':modulated_func })
    else:
        bend = th.nn.Sequential(transform()) 
        bends.update({'transform':bend})
    
    return [bends]
