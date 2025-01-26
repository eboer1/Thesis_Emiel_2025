base_architecture = 'vgg19'
img_size = 224
prototype_shape = (20, 128, 1, 1)
num_classes = 2
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '001_S2_UNSA'

data_path = '/home2/s4991125/datasets/screw_split_undersampling/'
train_dir = data_path + 'train/'   # Adjust according to your dataset folder structure
test_dir = data_path + 'val/'     # Adjust accordingly
train_push_dir = data_path + 'train/'  # Usually the same as train_dir
train_batch_size = 6
test_batch_size = 8
train_push_batch_size = 6

joint_optimizer_lrs = {'features': 0.001,
                       'add_on_layers': 0.003,
                       'prototype_vectors': 0.003}
joint_lr_step_size = 3

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.1,
    'sep': 0.01,
    'l1': 1e-2,
}

num_train_epochs = 16
num_warm_epochs = 3

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 5 == 0]
