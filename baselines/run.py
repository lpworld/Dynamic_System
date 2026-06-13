import os, sys, argparse, importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config
from baselines import common

# name -> (module, class, extra model kwargs)
REGISTRY = {
    'mf':              ('baselines.factorization', 'MF', {}),
    'bsa_mf':          ('baselines.factorization', 'BSAMF', {}),
    'prop_mf':         ('baselines.factorization', 'PropMF', {}),
    'deepfm':          ('baselines.factorization', 'DeepFM', {}),
    'din':             ('baselines.factorization', 'DIN', {}),

    'transrec':        ('baselines.sequential', 'TransRec', {}),
    'sasrec':          ('baselines.sequential', 'SASRec', {}),
    'bert4rec':        ('baselines.sequential', 'BERT4Rec', {}),
    'mamba4rec':       ('baselines.sequential', 'Mamba4Rec', {}),

    'linucb':          ('baselines.bandit_rl', 'LinUCB', {}),
    'cofiba':          ('baselines.bandit_rl', 'COFIBA', {}),
    'reinforce':       ('baselines.bandit_rl', 'REINFORCE', {}),
    'drn':             ('baselines.bandit_rl', 'DRN', {}),
    'rlur':            ('baselines.bandit_rl', 'RLUR', {}),

    'prefrec':         ('baselines.exploration', 'PrefRec', {}),
    'deep_exploration':('baselines.exploration', 'DeepExploration', {}),
    'hac':             ('baselines.exploration', 'HAC', {}),
    'resact':          ('baselines.exploration', 'ResAct', {}),

    'dreamrec':        ('baselines.generative', 'DreamRec', {}),
    'diffrec':         ('baselines.generative', 'DiffRec', {}),
    'tallrec':         ('baselines.generative', 'TALLRec', {}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(REGISTRY))
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    module_name, class_name, kwargs = REGISTRY[args.model]
    model_cls = getattr(importlib.import_module(module_name), class_name)
    cfg = load_config(args.config)
    common.run(model_cls, cfg, **kwargs)


if __name__ == '__main__':
    main()
