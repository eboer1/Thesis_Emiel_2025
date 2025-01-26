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

# Additional imports for confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import re

import os
import copy

from helpers import makedir, find_high_activation_crop
import train_and_test as tnt

from log import create_logger
from preprocess import mean, std, undo_preprocess_input_function
from push import get_deformation_info

import argparse

# -------------- PART 1: Argument Parsing -------------- #
parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
# NEW: Instead of single image path, we accept a full test directory
parser.add_argument('-testdir', nargs=1, type=str, 
                    help='Path to your test dataset directory (with subfolders per class)')

args = parser.parse_args()

prototype_layer_stride = 1

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

load_model_dir = args.modeldir[0]
load_model_name = args.model[0]
test_dir = args.testdir[0]      # e.g. "/home2/s4991125/datasets/screw_balanced/test/"

# -------------- PART 2: Model Loading (same as your original) -------------- #
# Example as in your original local_analysis:
load_model_path = os.path.join(load_model_dir, load_model_name)
print(f"Loading model from {load_model_path}...")

ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

# We'll store architecture & experiment run for logging / output paths
model_base_architecture = load_model_dir.split('/')[2]
experiment_run = '/'.join(load_model_dir.split('/')[3:])
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str) if epoch_number_str is not None else 0
if start_epoch_number == 0:
    start_epoch_number = 30
    epoch_number_str = '30'

prototype_shape = ppnet.prototype_shape
class_specific = True
img_size = ppnet_multi.module.img_size

# -------------- PART 3: Create a Single Logger for All Images -------------- #
from log import create_logger
from helpers import makedir

save_analysis_path = os.path.join(
    './saved_visualizations',
    model_base_architecture,
    experiment_run,
    load_model_name,
    'batch_local_analysis'  # a subfolder for the entire dataset
)
makedir(save_analysis_path)
log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))


# -------------- PART 4: Build a DataLoader for *All* Test Images -------------- #
from preprocess import mean, std, undo_preprocess_input_function
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
    batch_size=1,      # to keep logic simpler for local analysis
    shuffle=False
)

log(f"Found {len(test_dataset)} test images total.")
log(f"Classes in test dataset: {test_dataset.classes}")  # e.g. ['001.good', '002.manipulated_front', '003.scratch_head']


# -------------- PART 5: Prepare Structures for Overall Metrics -------------- #
all_preds = []
all_targets = []

# (Optional) Access the model's class identity info, e.g. for your prototype logic
prototype_info_path = os.path.join(load_model_dir, 'img', f'epoch-{epoch_number_str}', f'bb{epoch_number_str}.npy')
prototype_info = np.load(prototype_info_path)
prototype_img_identity = prototype_info[:, -1]

# Confirm max connections, same as your original local_analysis (optional)
prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0).cpu().numpy()

##### SANITY CHECK
# confirm prototype class identity
load_img_dir = os.path.join(load_model_dir, 'img')

prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
prototype_img_identity = prototype_info[:, -1]

log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
log('Their class identities are: ' + str(prototype_img_identity))

# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu().numpy()
if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
    log('All prototypes connect most strongly to their respective classes.')
else:
    log('WARNING: Not all prototypes connect most strongly to their respective classes.')
        
##### HELPER FUNCTIONS FOR PLOTTING
def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    
    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img

def save_prototype(fname, epoch, index):
    try:
        p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
        plt.imsave(fname, p_img)
    except:
        print("Problem loading ", os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))

def save_prototype_box(fname, epoch, index):
    try:
        p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-with_box'+str(index)+'.png'))
        plt.imsave(fname, p_img)
    except:
        print("Problem loading ", os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-with_box'+str(index)+'.png'))

def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    try:
        img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
        cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                    color, thickness=2)
        img_rgb_uint8 = img_bgr_uint8[...,::-1]
        img_rgb_float = np.float32(img_rgb_uint8) / 255
        plt.imsave(fname, img_rgb_float)
    except:
        print("Problem loading: imsave with bbox")

def save_deform_info(model, offsets, input, activations, 
                    save_dir,
                    prototype_img_filename_prefix,
                    proto_index):
    prototype_shape = model.prototype_shape
    if not hasattr(model, "prototype_dilation"):
        dilation = model.prototype_dillation
    else:
        dilation = model.prototype_dilation
    original_img_size = input.shape[0]

    colors = [(230/255, 25/255, 75/255), (60/255, 180/255, 75/255), (255/255, 225/255, 25/255),
                                (0, 130/255, 200/255), (245/255, 130/255, 48/255), (70/255, 240/255, 240/255),
                                (240/255, 50/255, 230/255), (170/255, 110/255, 40/255), (0,0,0)]
    argmax_proto_act_j = \
                list(np.unravel_index(np.argmax(activations, axis=None),
                                      activations.shape))
    fmap_height_start_index = argmax_proto_act_j[0] * prototype_layer_stride
    fmap_width_start_index = argmax_proto_act_j[1] * prototype_layer_stride

    original_img_j_with_boxes = input.copy()

    num_def_groups = 1#model.num_prototypes // model.num_classes
    def_grp_index = proto_index % num_def_groups
    def_grp_offset = def_grp_index * 2 * prototype_shape[-2] * prototype_shape[-1]

    for i in range(prototype_shape[-2]):
        for k in range(prototype_shape[-1]):
            # offsets go in order height offset, width offset
            h_index = def_grp_offset + 2 * (k + prototype_shape[-2]*i)
            w_index = h_index + 1
            h_offset = offsets[0, h_index, fmap_height_start_index, fmap_width_start_index]
            w_offset = offsets[0, w_index, fmap_height_start_index, fmap_width_start_index]

            # Subtract prototype_shape // 2 because fmap start indices give the center location, and we start with top left
            def_latent_space_row = fmap_height_start_index + h_offset + (i - prototype_shape[-2] // 2) * dilation[0]
            def_latent_space_col = fmap_width_start_index + w_offset + (k - prototype_shape[-1] // 2)* dilation[1]

            def_image_space_row_start = int(def_latent_space_row * original_img_size / activations.shape[-2])
            def_image_space_row_end = int((1 + def_latent_space_row) * original_img_size / activations.shape[-2])
            def_image_space_col_start = int(def_latent_space_col * original_img_size / activations.shape[-1])
            def_image_space_col_end = int((1 + def_latent_space_col) * original_img_size / activations.shape[-1])
 
            img_with_just_this_box = input.copy()
            cv2.rectangle(img_with_just_this_box,(def_image_space_col_start, def_image_space_row_start),
                                                    (def_image_space_col_end, def_image_space_row_end),
                                                    colors[i*prototype_shape[-1] + k],
                                                    1)
            plt.imsave(os.path.join(save_dir,
                            prototype_img_filename_prefix + str(proto_index) + '_patch_' + str(i*prototype_shape[-1] + k) + '-with_box.png'),
                img_with_just_this_box,
                vmin=0.0,
                vmax=1.0)

            cv2.rectangle(original_img_j_with_boxes,(def_image_space_col_start, def_image_space_row_start),
                                                    (def_image_space_col_end, def_image_space_row_end),
                                                    colors[i*prototype_shape[-1] + k],
                                                    1)
            
            if not (def_image_space_col_start < 0 
                or def_image_space_row_start < 0
                or def_image_space_col_end >= input.shape[0]
                or def_image_space_row_end >= input.shape[1]):
                plt.imsave(os.path.join(save_dir,
                                prototype_img_filename_prefix + str(proto_index) + '_patch_' + str(i*prototype_shape[-1] + k) + '.png'),
                    input[def_image_space_row_start:def_image_space_row_end, def_image_space_col_start:def_image_space_col_end, :],
                    vmin=0.0,
                    vmax=1.0)
            
    plt.imsave(os.path.join(save_dir,
                            prototype_img_filename_prefix + str(proto_index) + '-with_box.png'),
                original_img_j_with_boxes,
                vmin=0.0,
                vmax=1.0)

### (B) INSTEAD: Create a DataLoader for ALL test images ###
test_dir = args.testdir[0]  # path to your test dataset with subfolders
img_size = ppnet_multi.module.img_size

normalize = transforms.Normalize(mean=mean, std=std)
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False
)

log(f"Loaded test dataset from {test_dir}")
log(f"Number of test images: {len(test_dataset)}")
log(f"Classes found: {test_dataset.classes}")

### If you rely on prototype info:
load_img_dir = os.path.join(load_model_dir, 'img')
prototype_info = np.load(os.path.join(load_img_dir, 'epoch-' + epoch_number_str, 'bb' + epoch_number_str + '.npy'))
prototype_img_identity = prototype_info[:, -1]
prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0).cpu().numpy()

# Prepare to store predictions for confusion matrix
all_preds = []
all_targets = []

### (C) INSERT the new loop snippet here ###
ppnet_multi.eval()
with torch.no_grad():
    for batch_idx, (images_test, labels_test) in enumerate(test_loader):
        images_test = images_test.cuda()
        labels_test = labels_test.cuda()

        # Step A: Forward pass
        logits, additional_returns = ppnet_multi(images_test)
        prototype_activations = additional_returns[3]  # same indexing as original
        conv_output, prototype_activation_patterns = ppnet.push_forward(images_test)
        offsets, _ = get_deformation_info(conv_output, ppnet_multi)
        offsets = offsets.detach()

        # Step B: Determine predicted class vs. actual
        _, pred = torch.max(logits, dim=1)
        predicted_cls = pred.item()
        actual_cls = labels_test.item()

        # Save them for confusion matrix
        all_preds.append(predicted_cls)
        all_targets.append(actual_cls)

        # Log info
        file_path, _ = test_dataset.samples[batch_idx]
        filename = os.path.basename(file_path)
        log(f"\n=== Image {batch_idx}: {filename} ===")
        log(f"  Predicted class index = {predicted_cls}, Actual class index = {actual_cls}")
        log(f"  Predicted class name  = {test_dataset.classes[predicted_cls]}")
        log(f"  Actual class name     = {test_dataset.classes[actual_cls]}")

        # Optional: make subfolder to store local analysis for this image
        image_save_path = os.path.join(save_analysis_path, f"img_{batch_idx}_{filename}")
        makedir(image_save_path)
        
        original_img = save_preprocessed_img(
            os.path.join(image_save_path, 'original_img.png'),
            images_test,
            index=0  # since batch_size=1, there's only one image per batch
        )
        ##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
        most_act_path = os.path.join(image_save_path, 'most_activated_prototypes')
        makedir(most_act_path)
        
        log('Most activated 10 prototypes of this image:')
        array_act, sorted_indices_act = torch.sort(prototype_activations[0])
        log(f"Top activated prototypes for {filename}:")
        for i in range(1, 11): 
            proto_idx = sorted_indices_act[-i].item()
            log(f"    Prototype index: {proto_idx}, activation: {array_act[-i].item()}")
            save_prototype(os.path.join(most_act_path, 'most_activated_prototypes',
                                        'top-%d_activated_prototype.png' % i),
                           start_epoch_number, sorted_indices_act[-i].item())
            save_prototype_box(os.path.join(most_act_path, 'most_activated_prototypes',
                                        'top-%d_activated_prototype_with_box.png' % i),
                           start_epoch_number, sorted_indices_act[-i].item())
            log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
            log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
            if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
                log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
            log('activation value (similarity score): {0}'.format(array_act[-i]))
            log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
            
            activation_pattern = prototype_activation_patterns[0][sorted_indices_act[-i].item()].detach().cpu().numpy()
            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                      interpolation=cv2.INTER_CUBIC)
            
        
            save_deform_info(model=ppnet, offsets=offsets, 
                                input=original_img, activations=activation_pattern,
                                save_dir=most_act_path,
                                prototype_img_filename_prefix='top-%d_activated_prototype_' % i,
                                proto_index=sorted_indices_act[-i].item())
        
            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
            high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                          high_act_patch_indices[2]:high_act_patch_indices[3], :]
            log('most highly activated patch of the chosen image by this prototype:')
            plt.imsave(os.path.join(most_act_path,
                                    'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                       high_act_patch)
            log('most highly activated patch by this prototype shown in the original image:')
            imsave_with_bbox(fname=os.path.join(most_act_path,
                                    'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                             img_rgb=original_img,
                             bbox_height_start=high_act_patch_indices[0],
                             bbox_height_end=high_act_patch_indices[1],
                             bbox_width_start=high_act_patch_indices[2],
                             bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
            
            # show the image overlayed with prototype activation map
            rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
            rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]
            overlayed_img = 0.5 * original_img + 0.3 * heatmap
            log('prototype activation map of the chosen image:')
            #plt.axis('off')
            plt.imsave(os.path.join(most_act_path,
                                    'prototype_activation_map_by_top-%d_prototype.png' % i),
                       overlayed_img)
            log('--------------------------------------------------------------')
        
        ##### PROTOTYPES FROM TOP-k CLASSES
        k = 2
        log('Prototypes from top-%d classes:' % k)
        topk_logits, topk_classes = torch.topk(logits[0], k=k)
        for i,c in enumerate(topk_classes.detach().cpu().numpy()):
            topk_dir = os.path.join(image_save_path, f'top-{i+1}_class_prototypes')
            makedir(topk_dir)

        
            log('top %d predicted class: %d' % (i+1, c))
            log('logit of the class: %f' % topk_logits[i])
            class_prototype_indices = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
            class_prototype_activations = prototype_activations[0][class_prototype_indices]
            _, sorted_indices_cls_act = torch.sort(class_prototype_activations)
        
            prototype_cnt = 1
            for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
                prototype_index = class_prototype_indices[j]
        
                save_prototype_box(os.path.join(topk_dir, 'top-%d_class_prototypes' % (i+1),
                                            'top-%d_activated_prototype_with_box.png' % prototype_cnt),
                            start_epoch_number, prototype_index)
                log('prototype index: {0}'.format(prototype_index))
                log('prototype class identity: {0}'.format(prototype_img_identity[prototype_index]))
                if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
                    log('prototype connection identity: {0}'.format(prototype_max_connection[prototype_index]))
                log('activation value (similarity score): {0}'.format(prototype_activations[0][prototype_index]))
                log('last layer connection: {0}'.format(ppnet.last_layer.weight[c][prototype_index]))
                
                activation_pattern = prototype_activation_patterns[0][prototype_index].detach().cpu().numpy()
                upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                          interpolation=cv2.INTER_CUBIC)
        
                save_deform_info(model=ppnet, offsets=offsets, 
                                input=original_img, activations=activation_pattern,
                                save_dir=topk_dir,
                                prototype_img_filename_prefix='top-%d_activated_prototype_' % prototype_cnt,
                                proto_index=prototype_index)
                
                # show the most highly activated patch of the image by this prototype
                high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
                high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                              high_act_patch_indices[2]:high_act_patch_indices[3], :]
                log('most highly activated patch of the chosen image by this prototype:')
                plt.imsave(os.path.join(topk_dir,
                                        'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                           high_act_patch)
                log('most highly activated patch by this prototype shown in the original image:')
                imsave_with_bbox(fname=os.path.join(topk_dir,
                                                    'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                                 img_rgb=original_img,
                                 bbox_height_start=high_act_patch_indices[0],
                                 bbox_height_end=high_act_patch_indices[1],
                                 bbox_width_start=high_act_patch_indices[2],
                                 bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
                
                # show the image overlayed with prototype activation map
                rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
                rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
                heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[...,::-1]
                overlayed_img = 0.5 * original_img + 0.3 * heatmap
                log('prototype activation map of the chosen image:')
                plt.imsave(os.path.join(topk_dir,
                                        'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
                           overlayed_img)
                log('--------------------------------------------------------------')
                prototype_cnt += 1
            log('***************************************************************')

log("\n=== Finished local analysis for all test images ===")

all_preds_np = np.array(all_preds)
all_targets_np = np.array(all_targets)

cm = confusion_matrix(all_targets_np, all_preds_np)
class_names = test_dataset.classes
log("Confusion Matrix:\n" + str(cm))

cmdisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# Adjust the figure size and font sizes
fig, ax = plt.subplots(figsize=(10, 10))  # Increase figure size for better readability
cmdisp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix Baseline model (3 classes)", fontsize=14)  # Increase title font size
plt.xlabel("Predicted Labels", fontsize=12)  # Add x-axis label and increase font size
plt.ylabel("True Labels", fontsize=12)  # Add y-axis label and increase font size
plt.xticks(fontsize=10)  # Increase font size of x-axis tick labels
plt.yticks(fontsize=10)  # Increase font size of y-axis tick labels
plt.tight_layout()  # Ensure layout fits well
plt.savefig(os.path.join(save_analysis_path, "test_confusion_matrix.png"))
plt.close()

log("Confusion matrix saved.")


# ---- New lines for summary metrics ---- #
accuracy = (all_preds_np == all_targets_np).mean() * 100
log(f"\nAccuracy on the test dataset: {accuracy:.2f}%")

from sklearn.metrics import precision_recall_fscore_support
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
    all_targets_np, all_preds_np, average='weighted'
)
log(f"Weighted Precision: {prec_w:.4f}, Weighted Recall: {rec_w:.4f}, Weighted F1: {f1_w:.4f}")

logclose()