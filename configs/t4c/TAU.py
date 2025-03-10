method = 'TAU'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'tau'
hid_S = 32
hid_T = 128
N_T = 4
N_S = 4
alpha = 1000
# training
lr = 1e-3
batch_size = 16  # bs = 4 x 4GPUs
drop_path = 0.1
sched = 'onecycle'