import argparse
from train import train

parser = argparse.ArgumentParser(description='train text segmentation parser')

file_parser = parser.add_argument_group(title='file', description='select dataset file')
file_parser.add_argument('--dataset', choices=['wikisection', 'wiki727k'], default='wikisection')
file_parser.add_argument('--train', type=str, required=True, help='path contain train files')
file_parser.add_argument('--valid', type=str, default=None, help='path contain validation files')
file_parser.add_argument('--test', type=str, default=None, help='path contain test files')
file_parser.add_argument('--rebuild_vocab', help='reprocess dataset to produce vocab file', action='store_true')
file_parser.add_argument('--load_json', help='load args from json file', action='store_true')
file_parser.add_argument('--language', choices=['EN', 'DE'], default='DE', help='dataset language')


model_parser = parser.add_argument_group(title='model', description='model parameter')
model_parser.add_argument('--model', choices=['LSTM-LSTM', 'BERT-LSTM', 'NVI-BERT', 'BERT-TRANSFORMER', 'BERT-S-LSTM'], required=True)
model_parser.add_argument('--lstm_hidden_size', type=int, default=128)
model_parser.add_argument('--lstm_layer', type=int, default=2)
model_parser.add_argument('--topic_size', type=int, default=50)


nvi_parser = parser.add_argument_group(title='nvi model', description='nvi model parameter')
nvi_parser.add_argument('--en1-units', type=int, default=100)
nvi_parser.add_argument('--en2-units', type=int, default=100)
nvi_parser.add_argument('--num-topic', type=int, default=50)
nvi_parser.add_argument('--init-mult', type=float, default=1.0)
nvi_parser.add_argument('--variance', type=float, default=0.995)


train_parser = parser.add_argument_group(title='train', description='train hyper parameter')
train_parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
train_parser.add_argument('--enforced_teach', type=int, default=10, help='number of epochs to use label segmentation')
train_parser.add_argument('--batch_size', type=int, default=8, help='number of documents in a batch')
train_parser.add_argument('--seed', type=int, default=2020, help='use seed to reproduce result')
train_parser.add_argument('--optimizer', choices=['adam', 'adagrad'], default='adam')
train_parser.add_argument('--device', type=str, default='cpu')
train_parser.add_argument('--lr_scheduler', choices=['None', 'Plateau'], default='Plateau')
train_parser.add_argument('--factor', type=float, default=0.8)
train_parser.add_argument('--min_lr', type=float, default=0.0000001)
train_parser.add_argument('--epoch', type=int, default=100)


save_parser = parser.add_argument_group(title='save', description='save model')
save_parser.add_argument('--save_path', type=str, required=True, help='path to save model or load model')


args = parser.parse_args(['--train', 'data/wikisection_dataset_ref/de_city/de_city_train',
                          '--valid', 'data/wikisection_dataset_ref/de_city/de_city_validation',
                          '--test', 'data/wikisection_dataset_ref/de_city/de_city_test',
                          '--model', 'BERT-S-LSTM',
                          # '--rebuild_vocab',
                          # '--load_json',
                          '--device', 'cuda:0',
                          '--lstm_hidden_size', '128',
                          '--dataset', 'wikisection',
                          '--save_path', 'result/12-18-Bert-LSTM-wikisection-de_city_0'])

# args = parser.parse_args(['--train', 'data/wiki_727/train',
#                           '--valid', 'data/wiki_727/dev',
#                           '--test', 'data/wiki_727/test',
#                           '--model', 'BERT-LSTM',
#                           '--rebuild_vocab',
#                           '--dataset', 'wiki727k',
#                           '--save_path', '12-11-Bert-LSTM-wiki727-0'])


train(args)
