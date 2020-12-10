import argparse

parser = argparse.ArgumentParser(description='train text segmentation parser')

file_parser = parser.add_argument_group(title='file', description='select dataset file')
file_parser.add_argument('--dataset', choices=['wikisection', 'wiki727k'], default='wikisection')
file_parser.add_argument('--input', type=str, required=True, help='path contain train files')
file_parser.add_argument('--valid', type=str, default=None, help='path contain validation files')
file_parser.add_argument('--test', type=str, default=None, help='path contain test files')


model_parser = parser.add_argument_group(title='model', description='model parameter')
model_parser.add_argument('--pretrain', help='use pretrain model to produce sentence embedding', action='store_true')
model_parser.add_argument('--model', choices=['LSTM-LSTM', 'BERT-LSTM', 'NVI-BERT'], required=True)
model_parser.add_argument('--lstm_hidden_size', type=int, default=128)
model_parser.add_argument('--topic_size', type=int, default=50)


train_parser = parser.add_argument_group(title='train', description='train hyper parameter')
train_parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
train_parser.add_argument('--enforced_teach', type=int, default=10, help='number of epochs to use label segmentation')
train_parser.add_argument('--batch_size', type=int, default=8, help='number of documents in a batch')

args = parser.parse_args(['--input', 'train_file', '--valid', 'valid_file', '--model', 'BERT-LSTM'])


print(args)
