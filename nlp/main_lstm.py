import argparse
import copy
import json
import math
import numpy as np
import os
import random
import time
import torch
import torch.distributed as dist
from model.splitcross import SplitCrossEntropyLoss
from model.awdlstm import RNNModel
from sgd_clip import SGDClipGrad


###############################################################################
# Make batch
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, local_batch_size, world_size):
    global_batch_size = local_batch_size * world_size
    # Work out how cleanly we can divide the dataset into batch_size parts.
    nbatch = data.size(0) // global_batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * global_batch_size)
    # Evenly divide the data across the batch_size batches.
    data_batched = data.view(global_batch_size, -1)
    data_splits = [data_batched[i*local_batch_size: (i+1)*local_batch_size].t().contiguous().cuda() for i in range(world_size)]
    return data_splits

def get_batch(source, i, bptt, seq_len=None):
    seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

###############################################################################
# Training code
###############################################################################

def evaluate(model, criterion, data_source, bptt, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.0
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, bptt):
        data, targets = get_batch(data_source, i, bptt)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    avg_loss = total_loss.item() / len(data_source)
    return avg_loss

# Model Averaging
def average_model(world_size, model, group):
    for param in model.parameters():
        dist.reduce(param.data, dst=0, op=dist.ReduceOp.SUM, group=group)
        param.data /= world_size
        dist.broadcast(param.data, src=0, group=group)

# Gradient Averaging
def average_grad(world_size, model, group):
    for param in model.parameters():
        dist.reduce(param.grad.data, dst=0, op=dist.ReduceOp.SUM, group=group)
        param.grad.data /= world_size
        dist.broadcast(param.grad.data, src=0, group=group)

def comp_grad_l2_norm(model) -> float:
    grad_l2_norm_sq = torch.tensor(0.0)
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_l2_norm_sq += torch.sum(param.grad.data * param.grad.data)
    grad_l2_norm = torch.sqrt(grad_l2_norm_sq).item()
    return grad_l2_norm

def train(args, model, criterion, optimizer, train_data_splits, val_data, test_data, eval_batch_size, test_batch_size):
    results = {
        'train_losses': [],
        'train_ppl': [],
        'val_losses': [],
        'val_ppl': [],
        'test_losses': [],
        'test_ppl': [],
        'epoch_elasped_times': [],
        'epoch_clip_operations': [] 
    }

    begin_avg = False
    ax = {}
    avg_cnt = 0
    best_val_loss = []

    world_size = args.world_size
    group = dist.new_group(range(world_size))
    model_clone = copy.deepcopy(model)
    criterion_clone = copy.deepcopy(criterion)
    global_average_grad_l2_norm = 0.0

    t_total = 0

    for epoch in range(0, args.epochs):
        train_data = train_data_splits[(args.rank + epoch) % args.world_size]
        # Turn on training mode which enables dropout.
        hidden = model.init_hidden(args.batch_size)
        epoch_start_time = time.time()
        clip_operations = []
        running_loss_cur_epoch = 0
        total_ite_cur_epoch = 0
        model.train()
        i = 0
        while i < train_data.size(0) - 1 - 1:
            # print(f'I {args.communication_interval} Rank {args.rank} -- Epoch {epoch} Iteration {total_ite_cur_epoch}')
            if 0 == t_total % args.communication_interval:
                # if args.gpu_id == 0:
                #     print(f'Average model at Epoch {epoch} Iteration{total_ite_cur_epoch}')
                with torch.no_grad():
                    average_model(world_size, model, group)
                    average_model(world_size, criterion, group)

            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            # There's a very small chance that it could select a very long sequence length resulting in OOM
            # seq_len = min(seq_len, args.bptt + 10)

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
            if 'clipping_param' in optimizer.param_groups[0]:
                clipping_param2 = optimizer.param_groups[0]['clipping_param']
                optimizer.param_groups[0]['clipping_param'] = clipping_param2 * seq_len / args.bptt
            model.train()
            data, targets = get_batch(train_data, i, args.bptt, seq_len=seq_len)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)
            # hidden = nn.Parameter(hidden)

            optimizer.zero_grad()
            output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
            raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

            loss = raw_loss
            # Activation Regularization
            if args.alpha: loss = loss + sum(
                args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if 0 == t_total % args.communication_interval:
                if args.baseline:
                    average_grad(world_size, model, group)
                    average_grad(world_size, criterion, group)
                elif args.clipping_option != 'local':
                    with torch.no_grad():
                        for param, param_clone in zip(model.parameters(), model_clone.parameters()):
                            if param.grad is None:
                                param_clone.grad = None
                            else:
                                param_clone.grad = param.grad.clone()
                        average_grad(world_size, model_clone, group)
                        model_global_average_grad_l2_norm = comp_grad_l2_norm(model_clone)
                        for param, param_clone in zip(criterion.parameters(), criterion_clone.parameters()):
                            if param.grad is None:
                                param_clone.grad = None
                            else:
                                param_clone.grad = param.grad.clone()
                        average_grad(world_size, criterion_clone, group)
                        criterion_global_average_grad_l2_norm = comp_grad_l2_norm(criterion_clone)
                        global_average_grad_l2_norm = math.sqrt(
                            model_global_average_grad_l2_norm*model_global_average_grad_l2_norm + criterion_global_average_grad_l2_norm*criterion_global_average_grad_l2_norm)

            _, clip_operation = optimizer.step(global_average_grad_l2_norm)
            clip_operations.append(clip_operation)

            running_loss_cur_epoch += raw_loss.data.item()
            optimizer.param_groups[0]['lr'] = lr2
            if 'clipping_param' in optimizer.param_groups[0]:
                optimizer.param_groups[0]['clipping_param'] = clipping_param2

            total_ite_cur_epoch += 1
            i += seq_len
            t_total += 1

        # Evaluation
        epoch_elapsed_time = time.time() - epoch_start_time
        results['train_losses'].append(running_loss_cur_epoch / total_ite_cur_epoch)
        results['train_ppl'].append(math.exp(running_loss_cur_epoch / total_ite_cur_epoch))
        results['epoch_elasped_times'].append(epoch_elapsed_time)
        results['epoch_clip_operations'].append(clip_operations)

        average_model(world_size, model, group)
        average_model(world_size, criterion, group)
        if begin_avg:
            avg_cnt += 1
            for prm in model.parameters():
                if avg_cnt == 1:
                    ax[prm] = prm.data.clone()
                else:
                    ax[prm].add_(prm.data.sub(ax[prm]).mul(1 / avg_cnt))
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                if len(ax) > 0:
                    prm.data.copy_(ax[prm])

            val_loss2 = evaluate(model, criterion, val_data, args.bptt, eval_batch_size)
            val_ppl2 = math.exp(val_loss2)
            results['val_losses'].append(val_loss2)
            results['val_ppl'].append(val_ppl2)
            test_loss2 = evaluate(model, criterion, test_data, args.bptt, test_batch_size)
            test_ppl2 = math.exp(test_loss2)
            results['test_losses'].append(test_loss2)
            results['test_ppl'].append(test_ppl2)
            print(f'| epoch {epoch:3d} | '
                    f'time: {epoch_elapsed_time:5.2f}s | '
                    f'valid loss {val_loss2:7.4f} | valid ppl {val_ppl2:9.3f} | valid bpc {val_loss2 / math.log(2):8.3f} | '
                    f'test loss {test_loss2:7.4f} | test ppl {test_ppl2:9.3f} | test bpc {test_loss2 / math.log(2):8.3f} |')

            for prm in model.parameters():
                prm.data.copy_(tmp[prm])

        else:
            val_loss = evaluate(model, criterion, val_data, args.bptt, eval_batch_size)
            val_ppl = math.exp(val_loss)
            results['val_losses'].append(val_loss)
            results['val_ppl'].append(val_ppl)
            test_loss = evaluate(model, criterion, test_data, args.bptt, test_batch_size)
            test_ppl = math.exp(test_loss)
            results['test_losses'].append(test_loss)
            results['test_ppl'].append(test_ppl)
            print(f'| epoch {epoch:3d} | '
                    f'time: {epoch_elapsed_time:5.2f}s | '
                    f'valid loss {val_loss:7.4f} | valid ppl {val_ppl:9.3f} | valid bpc {val_loss / math.log(2):8.3f} | '
                    f'test loss {test_loss:7.4f} | test ppl {test_ppl:9.3f} | test bpc {test_loss / math.log(2):8.3f} |')

            if not begin_avg and len(best_val_loss) > args.nonmono \
                and val_loss > min(best_val_loss[:-args.nonmono]):
                print('Starting averaging')
                begin_avg = True

            best_val_loss.append(val_loss)

    return results

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch NLP Distributed Training')
    parser.add_argument('--dataroot', type=str, default='data/ptbdataset/',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='PennTreebank',
                        help='Which dataset to run on (default: PennTreebank).')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--eta0', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='momentum')
    parser.add_argument('--nesterov', action='store_true',
                        help='Use nesterov momentum (default: False).')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit of each node')
    parser.add_argument('--batch-size', type=int, default=80, metavar='N',
                        help='batch size of the local node')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.3,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.65,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.5,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--weight-decay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--optim-method', type=str, default='SGDClipGrad',
                        help='optimizer to use (default: SGDClipGrad)')

    parser.add_argument('--clipping-param', type=float, default=1.0,
                        help='Weight decay used in optimizer (default: 1.0).')
    parser.add_argument('--clipping-option', type=str, default='local',
                        choices=['max', 'local', 'global_average'],
                        help='How to clip (default: local).')
    parser.add_argument('--baseline', action='store_true',
                        help='Do baseline local SGD (True) or not (False) (default: False).')  

    parser.add_argument('--world-size', type=int, default=8,
                        help='Number of nodes in training (default: 8).')
    parser.add_argument('--rank', type=int, default=0,
                        help='Which node is this (default: 0).')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='Which GPU is used in this node (default: 0).')
    parser.add_argument('--init-method', type=str, default='file://',
                        help='URL specifying how to initialize the process group (default: file//).')  
    parser.add_argument('--communication-interval', type=int, default=8,
                        help='Number of iterations to average the model across all nodes (default: 8).')

    parser.add_argument('--reproducible', action='store_true',
                        help='Ensure reproducibility (default: False).')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='Random seed (default: 0).')

    parser.add_argument('--log-folder', type=str, default='../logs',
                        help='Where to store results.')

    return parser.parse_args()

def main():
    args = arg_parser()
    args.tied = True

    dist.init_process_group(backend='nccl',
                            init_method=args.init_method,
                            world_size=args.world_size,
                            rank=args.rank)

    torch.cuda.set_device(args.gpu_id)
    print(f"| Rank {args.rank} | Requested GPU {args.gpu_id} "
          f'| Assigned GPU {torch.cuda.current_device()} |')

    # Set the random seed manually for reproducibility.
    if args.reproducible:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

    # Load data
    fn = f'corpus.{args.dataset}.data'
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        from dataset.nlp_data import Corpus
        print('Producing dataset...')
        corpus = Corpus(args.dataroot)
        torch.save(corpus, fn)

    eval_batch_size = 10
    test_batch_size = 2
    train_data_splits = batchify(corpus.train, args.batch_size, args.world_size)
    val_data = batchify(corpus.valid, eval_batch_size, world_size=1)[0]
    test_data = batchify(corpus.test, test_batch_size, world_size=1)[0]

    # Build the model
    ntokens = len(corpus.dictionary)
    model = RNNModel(args.model,
                     ntokens,
                     args.emsize,
                     args.nhid,
                     args.nlayers,
                     args.dropout,
                     args.dropouth,
                     args.dropouti,
                     args.dropoute,
                     args.wdrop,
                     args.tied,)
    init_model_path = f"{args.dataset}_{args.model}_init_model.pt"
    if os.path.isfile(init_model_path):
        model.load_state_dict(torch.load(init_model_path))
    else:
        torch.save(model.state_dict(), init_model_path)
    model.cuda()

    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
    init_criterion_path = f"{args.dataset}_{args.model}_init_criterion.pt"
    if os.path.isfile(init_criterion_path):
        criterion.load_state_dict(torch.load(init_criterion_path))
    else:
        torch.save(criterion.state_dict(), init_criterion_path)
    criterion.cuda()

    params = list(model.parameters()) + list(criterion.parameters())

    # Train
    optimizer = SGDClipGrad(params, lr=args.eta0, momentum=args.momentum,
                            weight_decay=args.weight_decay, nesterov=args.nesterov,
                            clipping_param=args.clipping_param, clipping_option=args.clipping_option)
    train_results = train(args, model, criterion, optimizer, train_data_splits, val_data, test_data, eval_batch_size, test_batch_size)

    # Logging results.
    print('Writing the results.')
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    log_name = (f'{args.dataset}_{args.model}_SGDClipGrad_'
                + ('Eta0_%g_' % (args.eta0))
                + ('Momentum_%g_' % (args.momentum))
                + ('WD_%g_' % (args.weight_decay))
                + ('Clipping_%s_%g_' % (args.clipping_option, args.clipping_param))
                + ('baseline_' if args.baseline else '')
                + ('Epoch_%d_Batchsize_%d_' % (args.epochs, args.batch_size))
                + ('Comm_I_%d_' % args.communication_interval)
                + (f'Rank_{args.rank}_GPU_{args.gpu_id}'))
    with open(f"{args.log_folder}/{log_name}.json", 'w') as f:
        json.dump(train_results, f)

    print('Finished.')

if __name__ == '__main__':
    main()
