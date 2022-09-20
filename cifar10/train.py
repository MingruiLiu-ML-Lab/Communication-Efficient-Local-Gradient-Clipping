"""
Train a model on the training set.
"""
import copy
import time
import torch
import torch.distributed as dist
from evaluate import evaluate
from sgd_clip import SGDClipGrad


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
    grad_l2_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_l2_norm_sq += torch.sum(param.grad.data * param.grad.data)
    grad_l2_norm = torch.sqrt(grad_l2_norm_sq).item()
    return grad_l2_norm

def train(args, train_loader, test_loader, net, criterion):
    """
    Args:
        args: parsed command line arguments.
        train_loader: an iterator over the training set.
        test_loader: an iterator over the test set.
        net: the neural network model employed.
        criterion: the loss function.

    Outputs:
        All training losses, training accuracies, test losses, and test
        accuracies on each evaluation during training.
    """
    optimizer = SGDClipGrad(params=net.parameters(), lr=args.eta0, momentum=args.momentum,
                            weight_decay=args.weight_decay, nesterov=args.nesterov,
                            clipping_param=args.clipping_param, clipping_option=args.clipping_option)

    num_iterations_per_epoch = int(len(train_loader) / args.world_size)
    world_size = args.world_size
    group = dist.new_group(range(world_size))
    net_clone = copy.deepcopy(net)
    global_average_grad_l2_norm = 0.0

    all_train_losses = []
    all_train_accuracies = []
    all_test_losses = []
    all_test_accuracies = []
    epoch_elasped_times = []
    epoch_clip_operations = []
    t_total = 0
    epoch_start = time.time()
    clip_operations = []
    for _ in range(1, args.train_epochs + 1):
        net.train()

        for data in train_loader:
            # print(f'Rank {args.rank} -- {t_total}')
            if 0 == t_total % args.communication_interval:
                with torch.no_grad():
                    average_model(world_size, net, group)

            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if 0 == t_total % args.communication_interval:
                if args.baseline:
                    average_grad(world_size, net, group)
                elif args.clipping_option != 'local':
                    with torch.no_grad():
                        for param, param_clone in zip(net.parameters(), net_clone.parameters()):
                            if param.grad is None:
                                param_clone.grad = None
                            else:
                                param_clone.grad = param.grad.clone()
                        average_grad(world_size, net_clone, group)
                        global_average_grad_l2_norm = comp_grad_l2_norm(net_clone)

            _, clip_operation = optimizer.step(global_average_grad_l2_norm)
            clip_operations.append(clip_operation)

            # Evaluate the model on training and validation dataset.
            if t_total % num_iterations_per_epoch == 0:
                elapsed_time = time.time() - epoch_start
                epoch_elasped_times.append(elapsed_time)

                epoch_clip_operations.append(clip_operations)
                clip_operations = []

                net_clone = copy.deepcopy(net)
                average_model(world_size, net_clone, group)

                train_loss, train_accuracy = evaluate(train_loader, net_clone, criterion)
                all_train_losses.append(train_loss)
                all_train_accuracies.append(train_accuracy)

                test_loss, test_accuracy = evaluate(test_loader, net_clone, criterion)
                all_test_losses.append(test_loss)
                all_test_accuracies.append(test_accuracy)

                epoch_idx_total = t_total // num_iterations_per_epoch

                print(f'| Rank {args.rank} '
                      f'| GPU {args.gpu_id} '
                      f'| Epoch {epoch_idx_total} '
                      f'| training time {elapsed_time} seconds '
                      f'| train loss {train_loss:.4f} '
                      f'| train accuracy {train_accuracy:.4f} '
                      f'| test loss {test_loss:.4f} '
                      f'| test accuracy {test_accuracy:.4f} |')

                if str(epoch_idx_total) in args.step_decay_milestones:
                    print(f'Decay step size and clip param at Epoch {epoch_idx_total}.')
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= args.step_decay_factor
                        if 'clipping_param' in param_group:
                            param_group['clipping_param'] *= args.step_decay_factor

                epoch_start = time.time()
                net.train()

            t_total += 1

    return {'train_losses': all_train_losses,
            'train_accuracies': all_train_accuracies,
            'test_losses': all_test_losses,
            'test_accuracies': all_test_accuracies,
            'epoch_elasped_times': epoch_elasped_times,
            'epoch_clip_operations': epoch_clip_operations}
