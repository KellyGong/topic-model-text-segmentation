import logging
from Dataset_loader import load_wikisection, load_wiki727k
from pathlib import Path
from Solver import BERT_Solver, NVI_BERT_Solver
import json


def load_dataset(args):
    if args.dataset == 'wikisection':
        return load_wikisection(args)
    elif args.dataset == 'wiki727k':
        return load_wiki727k(args)
    else:
        raise FileNotFoundError


def init_logger(args):
    logger_path = Path(args.save_path)
    if not logger_path.exists():
        logger_path.mkdir()

    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG, filename=logger_path / 'new.log', filemode='a')


def save(args):
    with Path(Path(args.save_path) / 'args.json').open('w', encoding='UTF-8') as f:
        json.dump(args.__dict__, f, indent=4)


def load(args):
    if Path(Path(args.save_path) / 'args.json').exists() and args.load_json:
        with Path(Path(args.save_path) / 'args.json').open('r', encoding='UTF-8') as f:
            args.__dict__ = json.load(f)


def train(args):
    train_set, valid_set, test_set = load_dataset(args)
    init_logger(args)

    # load args from file
    if args.load_json:
        load(args)

    save(args)

    if args.model == 'NVI-BERT':
        solver = NVI_BERT_Solver(args, train_set, valid_set, test_set)

    else:
        solver = BERT_Solver(args, train_set, valid_set, test_set)

    solver.train_and_evaluate()

