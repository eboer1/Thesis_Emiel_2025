import time
import torch
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# add distillation loss function for incremental learning
def distillation_loss(old_logits, new_logits, T=2.0):
    # Compute soft targets from old logits
    old_probs = torch.nn.functional.softmax(old_logits / T, dim=1)
    new_log_probs = torch.nn.functional.log_softmax(new_logits / T, dim=1)
    # KL divergence as distillation loss
    return torch.nn.functional.kl_div(new_log_probs, old_probs, reduction='batchmean') * (T * T)



def save_confusion_matrix(conf_matrix, epoch, class_names, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for Epoch {epoch}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{output_dir}/confusion_matrix_epoch_{epoch}.png')
    plt.close()

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, subtractive_margin=True, use_ortho_loss=False,
                   epoch=None, experiment_dir=None, class_names=None,
                   old_model=None, old_num_classes=None):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''

    # Dynamically retrieve class names from the dataset
    class_names = dataloader.dataset.classes
    
    is_train = optimizer is not None
    
    kd_loss_sum = 0.0
    kd_loss_count = 0
    
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_l2 = 0
    total_ortho_loss = 0
    max_offset = 0
    
    # Initialize metrics for all cases
    precision, recall, f1_score, conf_matrix = [], [], [], []
    
    weighted_prec, weighted_rec, weighted_f1 = None, None, None

    if use_l1_mask:
        l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
        l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
    else:
        l1 = model.module.last_layer.weight.norm(p=1)
    
    # To store all predictions and true labels
    all_predictions = []
    all_targets = []

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            if subtractive_margin:
                output, additional_returns = model(input, is_train=is_train, 
                                                    prototypes_of_wrong_class=prototypes_of_wrong_class)
            else:
                output, additional_returns = model(input, is_train=is_train, prototypes_of_wrong_class=None)

            max_activations = additional_returns[0]
            marginless_logits = additional_returns[1]
            conv_features = additional_returns[2]
            
            with torch.no_grad():
                prototype_shape = model.module.prototype_shape
                epsilon_val = model.module.epsilon_val
                n_eps_channels = model.module.n_eps_channels

                x = conv_features
                epsilon_channel_x = torch.ones(x.shape[0], n_eps_channels, x.shape[2], x.shape[3]) * epsilon_val
                epsilon_channel_x = epsilon_channel_x.cuda()
                x = torch.cat((x, epsilon_channel_x), -3)
                input_vector_length = model.module.input_vector_length
                normalizing_factor = (prototype_shape[-2] * prototype_shape[-1])**0.5
                
                input_length = torch.sqrt(torch.sum(torch.square(x), dim=-3))
                input_length = input_length.view(input_length.size()[0], 1, input_length.size()[1], input_length.size()[2]) 
                input_normalized = input_vector_length * x / input_length
                input_normalized = input_normalized / normalizing_factor
                offsets = model.module.conv_offset(input_normalized)

            epsilon_val = model.module.epsilon_val
            n_eps_channels = model.module.n_eps_channels
            epsilon_channel_x = torch.ones(conv_features.shape[0], n_eps_channels, conv_features.shape[2], conv_features.shape[3]) * epsilon_val
            epsilon_channel_x = epsilon_channel_x.cuda()
            conv_features = torch.cat((conv_features, epsilon_channel_x), -3)
            
            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                # calculate cluster cost
                correct_class_prototype_activations, _ = torch.max(max_activations * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(correct_class_prototype_activations)

                # calculate separation cost
                incorrect_class_prototype_activations, _ = \
                    torch.max(max_activations * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(incorrect_class_prototype_activations)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(max_activations * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                offset_l2 = offsets.norm()

            else:
                max_activations, _ = torch.max(max_activations, dim=1)
                cluster_cost = torch.mean(max_activations)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(marginless_logits.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            
            # Collect predictions and true labels for validation or testing
            if not is_train:
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
            
            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_l2 += offset_l2
            total_avg_separation_cost += avg_separation_cost.item()
            batch_max = torch.max(torch.abs(offsets))
            max_offset = torch.max(torch.Tensor([batch_max, max_offset]))

            '''
            Compute keypoint-wise orthogonality loss, i.e. encourage each piece
            of a prototype to be orthogonal to the others.
            '''
            orthogonalities = model.module.get_prototype_orthogonalities()
            orthogonality_loss = torch.norm(orthogonalities)
            total_ortho_loss += orthogonality_loss.item()

        # add distillation logic in the training 
        if is_train:
            # ADDED DISTILLATION LOGIC
            if is_train and old_model is not None and old_num_classes is not None:
                with torch.no_grad():
                    old_output, _ = old_model(input)
                kd_loss = distillation_loss(old_output[:, :old_num_classes], output[:, :old_num_classes], T=2.0)
                alpha = 0.5
                
                kd_loss_sum += kd_loss.item()
                kd_loss_count += 1
                
            else:
                kd_loss = 0.0
                alpha = 1.0
            
            if class_specific:
                if coefs is not None:
                    base_loss = (coefs['crs_ent'] * cross_entropy
                               + coefs['clst'] * cluster_cost
                               + coefs['sep'] * separation_cost
                               + coefs['l1'] * l1
                               + coefs['offset_bias_l2'] * offset_l2)
                    if use_ortho_loss:
                        base_loss += coefs['orthogonality_loss'] * orthogonality_loss
                else:
                    base_loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    base_loss = (coefs['crs_ent'] * cross_entropy
                               + coefs['clst'] * cluster_cost
                               + coefs['l1'] * l1)
                else:
                    base_loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1

            loss = alpha * base_loss + (1 - alpha) * kd_loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            del loss, base_loss, kd_loss

        del input, predicted, target, output, max_activations, offsets, conv_features
        del prototypes_of_correct_class, prototypes_of_wrong_class
        del epsilon_channel_x, input_normalized, input_vector_length, x, additional_returns
        del marginless_logits, cross_entropy, cluster_cost, separation_cost, orthogonalities, orthogonality_loss
        if class_specific:
            del correct_class_prototype_activations, incorrect_class_prototype_activations, avg_separation_cost, input_length

    if not is_train:  # Compute metrics only during test/validation
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_predictions, average=None, labels=np.arange(len(class_names)))
        conf_matrix = confusion_matrix(all_targets, all_predictions, labels=np.arange(len(class_names)))
        
        # Compute the weighted-average
        weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
            all_targets, 
            all_predictions, 
            average='weighted', 
            labels=np.arange(len(class_names))
        )

    end = time.time()
    log('\ttime: \t{0}'.format(end -  start))
    if use_ortho_loss:
        log('\tUsing ortho loss')
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\torthogonality loss:\t{0}'.format(total_ortho_loss / n_batches))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    log('\tavg l2: \t\t{0}'.format(total_l2 / n_batches))
    if coefs is not None:
        log('\tavg l2 with weight: \t\t{0}'.format(coefs['offset_bias_l2'] * total_l2 / n_batches))
        log('\torthogonality loss with weight:\t{0}'.format(coefs['orthogonality_loss'] * total_ortho_loss / n_batches))
    log('\tmax offset: \t{0}'.format(max_offset))
    
    # Dynamically format and log precision, recall, F1 score per class
    if not is_train:
        for idx, class_name in enumerate(class_names):
            log(f'\tClass {idx} ({class_name}): Precision: {precision[idx]:.4f}, Recall: {recall[idx]:.4f}, F1 Score: {f1_score[idx]:.4f}')
    
    if not is_train:    
        log(f'\tWeighted Avg: Precision={weighted_prec:.4f},'
            f' Recall={weighted_rec:.4f}, F1={weighted_f1:.4f}')
    
    # Log kd_loss per epoch if we are in training mode and kd_loss_count > 0
    if is_train and kd_loss_count > 0:
        avg_kd_loss = kd_loss_sum / kd_loss_count
        log(f'\tavg kd_loss: {avg_kd_loss:.4f}')
    
    if not is_train:
        if epoch is not None and experiment_dir is not None:
            save_confusion_matrix(conf_matrix, epoch, class_names, experiment_dir)



    return {
        'loss': total_cross_entropy / n_batches,
        'accuracy': n_correct / n_examples,
        'cluster': total_cluster_cost / n_batches,
        'separation': total_separation_cost / n_batches,
        'avg_separation': total_avg_separation_cost / n_batches,
        'orthogonality_loss': total_ortho_loss / n_batches,
        'precision': precision,  # Include for external logging
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': conf_matrix,
        'kd_loss_sum': kd_loss_sum,
        'kd_loss_count': kd_loss_count,
        'weighted_precision': weighted_prec,
        'weighted_recall': weighted_rec,
        'weighted_f1': weighted_f1
    }


def train(model, dataloader, optimizer, class_specific=True, coefs=None, log=print, 
          subtractive_margin=True, use_ortho_loss=False, epoch=None, experiment_dir=None,
          old_model=None, old_num_classes=None):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    metrics = _train_or_test(model, dataloader, optimizer, class_specific, use_l1_mask=True, coefs=coefs, log=log,
                             subtractive_margin=subtractive_margin, use_ortho_loss=use_ortho_loss, 
                             epoch=epoch, experiment_dir=experiment_dir, old_model=old_model, old_num_classes=old_num_classes)
    return metrics


def test(model, dataloader, class_specific=False, log=print, subtractive_margin=True,
         epoch=None, experiment_dir=None, class_names=None,
         old_model=None, old_num_classes=None):
    log('\ttest')
    model.eval()
    metrics = _train_or_test(model=model, dataloader=dataloader, optimizer=None,
        class_specific=class_specific, log=log, subtractive_margin=subtractive_margin,
        epoch=epoch, experiment_dir=experiment_dir, class_names=class_names, old_model=old_model, old_num_classes=old_num_classes)
    
    # Save confusion matrix visualization
    if epoch is not None and experiment_dir is not None:
        save_confusion_matrix(
            metrics['confusion_matrix'], epoch, class_names, experiment_dir
        )

    return metrics

def last_only(model, log=print, last_layer_fixed=True):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.conv_offset.parameters():
        p.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = not last_layer_fixed
    
    log('\tlast layer')


def warm_only(model, log=print, last_layer_fixed=True):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.conv_offset.parameters():
        p.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = not last_layer_fixed
    
    log('\twarm')

def warm_pre_offset(model, log=print, last_layer_fixed=True):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.conv_offset.parameters():
        p.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = not last_layer_fixed
    
    log('\twarm pre offset')

def joint(model, log=print, last_layer_fixed=False):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.conv_offset.parameters():
        p.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = not last_layer_fixed
    
    log('\tjoint')
