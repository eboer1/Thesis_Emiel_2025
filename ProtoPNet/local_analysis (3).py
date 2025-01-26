##### MODEL AND DATA LOADING
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import re
import os
import copy

from helpers import makedir, find_high_activation_crop
import model
import push
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-testdir', nargs=1, type=str, 
                    help='Path to your test dataset directory (with subfolders per class)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

load_model_dir = args.modeldir[0]
load_model_name = args.model[0]
test_dir = args.testdir[0]

# Prepare the save directory for local analysis
model_base_architecture = load_model_dir.split('/')[2]
experiment_run = '/'.join(load_model_dir.split('/')[3:])

save_analysis_path = os.path.join(
    '/home2/s4991125/ProtoPNet-master/local_analysis',
    model_base_architecture,
    experiment_run,
    load_model_name,
    'batch_local_analysis'
)
makedir(save_analysis_path)

log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str) if epoch_number_str else 0

log(f"Loading model from {load_model_path}...")
ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

img_size = ppnet_multi.module.img_size
prototype_shape = ppnet.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
class_specific = True

# SANITY CHECK
load_img_dir = os.path.join(load_model_dir, 'img')
prototype_info = np.load(
    os.path.join(load_img_dir, f'epoch-{epoch_number_str}', f'bb{epoch_number_str}.npy')
)
prototype_img_identity = prototype_info[:, -1]

log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
log('Their class identities are: ' + str(prototype_img_identity))

prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu().numpy()
if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
    log('All prototypes connect most strongly to their respective classes.')
else:
    log('WARNING: Not all prototypes connect most strongly to their respective classes.')

##### HELPER FUNCTIONS
def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    import copy
    undo_preprocessed_img = undo_preprocess_input_function(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1, 2, 0])
    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img

def save_prototype(fname, epoch, index):
    try:
        p_img = plt.imread(os.path.join(load_img_dir, f'epoch-{epoch}', f'prototype-img{index}.png'))
        plt.imsave(fname, p_img)
    except:
        print(f"Problem loading prototype-img{index}.png")

def save_prototype_self_activation(fname, epoch, index):
    try:
        p_img = plt.imread(os.path.join(load_img_dir, f'epoch-{epoch}',
                                        f'prototype-img-original_with_self_act{index}.png'))
        plt.imsave(fname, p_img)
    except:
        print(f"Problem loading prototype-img-original_with_self_act{index}.png")

def save_prototype_original_img_with_bbox(fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    try:
        p_img_bgr = cv2.imread(os.path.join(load_img_dir, f'epoch-{epoch}',
                                            f'prototype-img-original{index}.png'))
        cv2.rectangle(p_img_bgr, 
                      (bbox_width_start, bbox_height_start),
                      (bbox_width_end-1, bbox_height_end-1),
                      color, thickness=2)
        p_img_rgb = p_img_bgr[..., ::-1]
        p_img_rgb = np.float32(p_img_rgb) / 255
        plt.imsave(fname, p_img_rgb)
    except:
        print(f"Problem loading prototype-img-original{index}.png")

def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8,
                  (bbox_width_start, bbox_height_start),
                  (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imsave(fname, img_rgb_float)

# Build a DataLoader for the entire test directory
normalize = transforms.Normalize(mean=mean, std=std)
test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

log(f"Found {len(test_dataset)} test images total.")
log(f"Classes in test dataset: {test_dataset.classes}")

all_preds = []
all_targets = []

# We'll track how many "001.good" images we've done local analysis on:
good_count = 0
MAX_GOOD_ANALYSIS = 5  # only do local analysis for up to 5 images of "001.good"

ppnet_multi.eval()

with torch.no_grad():
    for batch_idx, (images_test, labels_test) in enumerate(test_loader):
        images_test = images_test.cuda()
        labels_test = labels_test.cuda()

        # Forward pass
        logits, min_distances = ppnet_multi(images_test)
        conv_output, distances = ppnet.push_forward(images_test)
        prototype_activations = ppnet.distance_2_similarity(min_distances)
        prototype_activation_patterns = ppnet.distance_2_similarity(distances)

        if ppnet.prototype_activation_function == 'linear':
            prototype_activations = prototype_activations + max_dist
            prototype_activation_patterns = prototype_activation_patterns + max_dist

        _, pred_class = torch.max(logits, dim=1)
        predicted_cls = pred_class.item()
        actual_cls = labels_test.item()

        all_preds.append(predicted_cls)
        all_targets.append(actual_cls)

        # Some logging
        file_path, _ = test_dataset.samples[batch_idx]
        filename = os.path.basename(file_path)
        class_name = test_dataset.classes[actual_cls]  # e.g. "001.good" or "002.defect"
        log(f"\n=== Image {batch_idx}: {filename} ===")
        log(f"  Predicted class index = {predicted_cls}, Actual class index = {actual_cls}")
        log(f"  Predicted class name  = {test_dataset.classes[predicted_cls]}")
        log(f"  Actual class name     = {class_name}")

        # Decide if we do local analysis for this image
        do_local_analysis = False
        if class_name == '001.good':
            if good_count < MAX_GOOD_ANALYSIS:
                do_local_analysis = True
                good_count += 1
        else:
            do_local_analysis = True

        if do_local_analysis:
            image_save_path = os.path.join(save_analysis_path, f"img_{batch_idx}_{filename}")
            makedir(image_save_path)

            # Save original image
            original_img = save_preprocessed_img(
                os.path.join(image_save_path, 'original_img.png'),
                images_test,
                index=0
            )

            # ============== MOST ACTIVATED PROTOTYPES (top 10) ============= #
            most_act_path = os.path.join(image_save_path, 'most_activated_prototypes')
            makedir(most_act_path)

            log('Most activated 10 prototypes of this image:')
            array_act, sorted_indices_act = torch.sort(prototype_activations[0])
            for i in range(1, 11):  # top 10
                proto_idx = sorted_indices_act[-i].item()
                log(f"   Prototype index: {proto_idx}, activation: {array_act[-i].item()}")
                
                # Save prototypes
                save_prototype(os.path.join(most_act_path, f'top-{i}_activated_prototype.png'),
                               start_epoch_number, proto_idx)
                save_prototype_original_img_with_bbox(
                    os.path.join(most_act_path, f'top-{i}_activated_prototype_in_original_pimg.png'),
                    epoch=start_epoch_number,
                    index=proto_idx,
                    bbox_height_start=prototype_info[proto_idx][1],
                    bbox_height_end=prototype_info[proto_idx][2],
                    bbox_width_start=prototype_info[proto_idx][3],
                    bbox_width_end=prototype_info[proto_idx][4]
                )
                save_prototype_self_activation(os.path.join(most_act_path, f'top-{i}_activated_prototype_self_act.png'),
                                               start_epoch_number, proto_idx)

                log(f'prototype class identity: {prototype_img_identity[proto_idx]}')
                if prototype_max_connection[proto_idx] != prototype_img_identity[proto_idx]:
                    log(f'prototype connection identity: {prototype_max_connection[proto_idx]}')
                log(f'activation value (similarity score): {array_act[-i].item()}')
                log(f'last layer connection with predicted class: {ppnet.last_layer.weight[predicted_cls][proto_idx].item()}')

                # Activation pattern
                activation_pattern = prototype_activation_patterns[0][proto_idx].detach().cpu().numpy()
                upsampled_activation_pattern = cv2.resize(activation_pattern,
                                                          dsize=(img_size, img_size),
                                                          interpolation=cv2.INTER_CUBIC)
                high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
                high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                              high_act_patch_indices[2]:high_act_patch_indices[3], :]
                plt.imsave(os.path.join(most_act_path,
                                        f'most_highly_activated_patch_by_top-{i}_prototype.png'),
                           high_act_patch)

                imsave_with_bbox(
                    os.path.join(most_act_path,
                                 f'most_highly_activated_patch_in_original_img_by_top-{i}_prototype.png'),
                    img_rgb=original_img,
                    bbox_height_start=high_act_patch_indices[0],
                    bbox_height_end=high_act_patch_indices[1],
                    bbox_width_start=high_act_patch_indices[2],
                    bbox_width_end=high_act_patch_indices[3]
                )

                # overlay
                rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
                rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
                heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[..., ::-1]
                overlayed_img = 0.5 * original_img + 0.3 * heatmap
                plt.imsave(
                    os.path.join(most_act_path,
                                 f'prototype_activation_map_by_top-{i}_prototype.png'),
                    overlayed_img
                )

            # ============== PROTOTYPES FROM TOP-k CLASSES ============== #
            k = 5
            log(f'Prototypes from top-{k} classes:')
            k = min(k, logits[0].size(0))
            topk_logits, topk_classes = torch.topk(logits[0], k=k)

            for class_idx, c in enumerate(topk_classes.detach().cpu().numpy()):
                topk_dir = os.path.join(image_save_path, f'top-{class_idx+1}_class_prototypes')
                makedir(topk_dir)

                log(f'top {class_idx+1} predicted class: {c}')
                log(f'logit of the class: {topk_logits[class_idx].item()}')

                class_prototype_indices = np.nonzero(
                    ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
                class_prototype_activations = prototype_activations[0][class_prototype_indices]
                _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

                prototype_cnt = 1
                for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
                    prototype_index = class_prototype_indices[j]
                    save_prototype(os.path.join(topk_dir,
                                                f'top-{prototype_cnt}_activated_prototype.png'),
                                   start_epoch_number, prototype_index)
                    save_prototype_original_img_with_bbox(
                        os.path.join(topk_dir,
                                     f'top-{prototype_cnt}_activated_prototype_in_original_pimg.png'),
                        epoch=start_epoch_number,
                        index=prototype_index,
                        bbox_height_start=prototype_info[prototype_index][1],
                        bbox_height_end=prototype_info[prototype_index][2],
                        bbox_width_start=prototype_info[prototype_index][3],
                        bbox_width_end=prototype_info[prototype_index][4]
                    )
                    save_prototype_self_activation(
                        os.path.join(topk_dir,
                                     f'top-{prototype_cnt}_activated_prototype_self_act.png'),
                        start_epoch_number, prototype_index
                    )

                    log(f'prototype index: {prototype_index}')
                    log(f'prototype class identity: {prototype_img_identity[prototype_index]}')
                    if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
                        log(f'prototype connection identity: {prototype_max_connection[prototype_index]}')
                    log(f'activation value (similarity score): {prototype_activations[0][prototype_index].item()}')
                    log(f'last layer connection: {ppnet.last_layer.weight[c][prototype_index].item()}')

                    activation_pattern = prototype_activation_patterns[0][prototype_index].detach().cpu().numpy()
                    upsampled_activation_pattern = cv2.resize(activation_pattern,
                                                              dsize=(img_size, img_size),
                                                              interpolation=cv2.INTER_CUBIC)
                    high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
                    high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                                  high_act_patch_indices[2]:high_act_patch_indices[3], :]
                    plt.imsave(os.path.join(topk_dir,
                                            f'most_highly_activated_patch_by_top-{prototype_cnt}_prototype.png'),
                               high_act_patch)
                    imsave_with_bbox(
                        os.path.join(topk_dir,
                                     f'most_highly_activated_patch_in_original_img_by_top-{prototype_cnt}_prototype.png'),
                        img_rgb=original_img,
                        bbox_height_start=high_act_patch_indices[0],
                        bbox_height_end=high_act_patch_indices[1],
                        bbox_width_start=high_act_patch_indices[2],
                        bbox_width_end=high_act_patch_indices[3]
                    )

                    # overlay
                    rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
                    rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_img = 0.5 * original_img + 0.3 * heatmap
                    plt.imsave(os.path.join(topk_dir,
                                            f'prototype_activation_map_by_top-{prototype_cnt}_prototype.png'),
                               overlayed_img)
                    prototype_cnt += 1

                log('***************************************************************')

log("\n=== Finished local analysis for relevant test images ===")

# Build confusion matrix, weighted precision/recall/F1, and accuracy
all_preds_np = np.array(all_preds)
all_targets_np = np.array(all_targets)

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

cm = confusion_matrix(all_targets_np, all_preds_np, labels=np.arange(len(test_dataset.classes)))
log(f"Confusion Matrix:\n{cm}")

from sklearn.metrics import ConfusionMatrixDisplay

# Display and save the confusion matrix
cmdisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)

# Adjust the figure size and font sizes
fig, ax = plt.subplots(figsize=(10, 10))  # Increase figure size for better readability
cmdisp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix ProtoPNet 3 Classes", fontsize=14)  # Increase title font size
plt.xlabel("Predicted Labels", fontsize=12)  # Add x-axis label and increase font size
plt.ylabel("True Labels", fontsize=12)  # Add y-axis label and increase font size
plt.xticks(fontsize=10)  # Increase font size of x-axis tick labels
plt.yticks(fontsize=10)  # Increase font size of y-axis tick labels
plt.tight_layout()  # Ensure layout fits well

# Save the confusion matrix plot
plt.savefig(os.path.join(save_analysis_path, "test_confusion_matrix.png"))
plt.close()

log("Confusion matrix saved.")

acc = accuracy_score(all_targets_np, all_preds_np)
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
    all_targets_np, all_preds_np, average='weighted', labels=np.arange(len(test_dataset.classes))
)

log(f"\nAccuracy on the test dataset: {acc*100:.2f}%")
log(f"Weighted Precision: {prec_w:.4f}, Weighted Recall: {rec_w:.4f}, Weighted F1: {f1_w:.4f}")

logclose()
