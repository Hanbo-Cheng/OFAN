import time
import os
import re
import numpy as np 
import random
import pickle as pkl
import torch
from torch import optim, nn
from utils import *
from encoder_decoder_optical import Encoder_Decoder
from torch.cuda.amp import autocast as autocast
from load_encoder import * 
from logger import Logger
import sys
# whether use multi-GPUs
multi_gpu_flag = False
# whether init params
init_param_flag = True

# load configurations
# paths
scaler = torch.cuda.amp.GradScaler()
result_path = './wap_optical-14-share-ht/'

if not os.path.exists(result_path):
    os.mkdir(result_path)

sys.stdout = Logger(result_path + '/wap_share_ht.txt', sys.stdout)
sys.stderr = Logger(result_path + '/wap_share_ht.txt', sys.stderr)

datapath = './CROHME/'
online_path = './CROHME/'
dictionaries = ['./dictionary.txt']
datasets = [datapath + 'train_images.pkl',online_path + 'online_flow_train_feature.pkl', datapath + 'train_labels.txt']
valid_datasets = [datapath + '14_test_images.pkl',online_path + 'online_flow_test_feature_14.pkl', datapath + '14_test_labels.txt']
valid_output = [result_path + 'valid_decode_result.txt']
valid_result = [result_path + 'valid.wer']
saveto = result_path + 'WAP_params.pkl'

# training settings
if multi_gpu_flag:
    batch_Imagesize = 500000
    maxImagesize = 500000
    valid_batch_Imagesize = 500000
    batch_size = 24
    valid_batch_size = 24
else:
    batch_Imagesize = 800000
    maxImagesize = 320000
    valid_batch_Imagesize = 320000
    batch_size = 16
    valid_batch_size = 8
maxlen = 200
max_epochs = 5000
lrate = 1.0
my_eps = 1e-6
decay_c = 1e-4
clip_c = 100.

# early stop
estop = False
halfLrFlag = 0
bad_counter = 0
patience = 15
finish_after = 10000000

# model architecture
params = {}
params['n'] = 256
params['m'] = 256
params['dim_attention'] = 512
params['D'] = 684
params['K'] = 111
params['growthRate'] = 24
params['reduction'] = 0.5
params['bottleneck'] = True
params['use_dropout'] = True
params['input_channels'] = 1
params['online_input_channels'] = 3

# load dictionary
worddicts = load_dict(dictionaries[0])
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
    worddicts_r[vv] = kk



# load data
# train, train_uid_list = dataIterator(datasets[0], datasets[1], worddicts, batch_size=batch_size,
#                                      batch_Imagesize=batch_Imagesize, maxlen=maxlen, maxImagesize=maxImagesize)

# train, train_uid_list = dataIterator_online(datasets[0], datasets[1], datasets[2], worddicts, batch_size=batch_size,
#                                      batch_Imagesize=batch_Imagesize, maxlen=maxlen, maxImagesize=maxImagesize)

train_dataIterator = BatchBucket_online(600, 2100, 200, 800000*3, 30,  #batch size 
                    datasets[0], datasets[1], datasets[2], 
                    dictionaries[0])
train, train_uid_list = train_dataIterator.get_batches()

# valid, valid_uid_list = dataIterator(valid_datasets[0], valid_datasets[1], worddicts, batch_size=valid_batch_size,
#                                      batch_Imagesize=valid_batch_Imagesize, maxlen=maxlen, maxImagesize=maxImagesize)
valid, valid_uid_list = dataIterator_online(valid_datasets[0], valid_datasets[1], valid_datasets[2], worddicts, batch_size=valid_batch_size,
                                     batch_Imagesize=valid_batch_Imagesize, maxlen=maxlen, maxImagesize=maxImagesize)

# display
uidx = 0  # count batch
loss_s = 0.  # count loss
ud_s = 0  # time for training an epoch
validFreq = -1
saveFreq = -1
sampleFreq = -1
dispFreq = 100
if validFreq == -1:
    validFreq = len(train) * 2
if saveFreq == -1:
    saveFreq = len(train)
if sampleFreq == -1:
    sampleFreq = len(train)

# initialize model
WAP_model = Encoder_Decoder(params)
if init_param_flag:
    WAP_model.apply(weight_init)
if multi_gpu_flag:
    WAP_model = nn.DataParallel(WAP_model, device_ids=[0, 1])
WAP_model.cuda()

load_encoder(WAP_model, './pretrain_models/ABM_params164_lr0.0625_50.21795989537925.pkl', './pretrain_models/wap_52.38578680203045.pkl')


# print model's parameters
model_params = WAP_model.named_parameters()
# for k, v in model_params:
#     print(k)

# loss function
criterion = torch.nn.CrossEntropyLoss(reduce=False)
# optimizer
optimizer = optim.Adadelta(WAP_model.parameters(), lr=lrate, eps=my_eps, weight_decay=decay_c)

print('Optimization')

# statistics
history_errs = []

for eidx in range(max_epochs):
    n_samples = 0
    ud_epoch = time.time()
    random.shuffle(train)
    for x, on_x, y in train:
        with autocast():
            WAP_model.train()
            ud_start = time.time()
            n_samples += len(x)
            uidx += 1
            x, x_mask, y, y_mask = prepare_data_bidecoder_online(params, x, on_x, y)

            x = torch.from_numpy(x).cuda()
            x_mask = torch.from_numpy(x_mask).cuda()
            y = torch.from_numpy(y).cuda()
            y_mask = torch.from_numpy(y_mask).cuda()

            # permute for multi-GPU training
            y = y.permute(1, 0)
            y_mask = y_mask.permute(1, 0)

            # forward
            scores, [alphas_off, alphas_on] = WAP_model(params, x, x_mask, y, y_mask)
            
            # recover from permute
            alphas_off = alphas_off.permute(1, 0, 2, 3)
            alphas_on = alphas_on.permute(1, 0, 2, 3)
            scores = scores.permute(1, 0, 2)
            scores = scores.contiguous()
            scores = scores.view(-1, scores.shape[2])
            y = y.permute(1, 0)
            y_mask = y_mask.permute(1, 0)
            y = y.contiguous()
            
            loss = criterion(scores, y.view(-1))
            loss = loss.view(y.shape[0], y.shape[1])
            loss = (loss * y_mask).sum(0) / y_mask.sum(0)
            loss = loss.mean()
            loss_s += loss.item()

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # loss.backward()
        if clip_c > 0.:
            torch.nn.utils.clip_grad_norm_(WAP_model.parameters(), clip_c)
        
        scaler.step(optimizer)
        scaler.update()

        ud = time.time() - ud_start
        ud_s += ud

        # display
        if np.mod(uidx, dispFreq) == 0:
            ud_s /= 60.
            loss_s /= dispFreq
            print('Epoch ', eidx, 'Update ', uidx, 'Cost ', loss_s, 'UD ', ud_s, 'lrate ', lrate, 'eps', my_eps,
                  'bad_counter', bad_counter)
            ud_s = 0
            loss_s = 0.

        # validation
        valid_stop = False
        if np.mod(uidx, sampleFreq) == 0:
            WAP_model.eval()
            with torch.no_grad():
                fpp_sample = open(valid_output[0], 'w')
                valid_count_idx = 0
                for x, on_x, y in valid:
                    for xx,xx_on,yy in zip(x, on_x, y):
                        # xx_pad = xx.astype(np.float32) / 255.
                        # xx_pad = torch.from_numpy(xx_pad[None, :, :, :]).cuda()  # (1,1,H,W)
                        x, x_mask, y, y_mask = prepare_data_bidecoder_online(params, [xx], [xx_on], [yy])
                        xx_pad = torch.from_numpy(x).cuda() 
                        sample, score = gen_sample_on(WAP_model, xx_pad, params, multi_gpu_flag, k=10, maxlen=200)
                        if len(score) == 0:
                            print('valid decode error happens')
                            valid_stop = True
                            break
                        score = score / np.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                        # write decoding results
                        fpp_sample.write(valid_uid_list[valid_count_idx])
                        valid_count_idx = valid_count_idx + 1
                        # symbols (without <eos>)
                        for vv in ss:
                            if vv == 0:  # <eos>
                                break
                            fpp_sample.write(' ' + worddicts_r[vv])
                        fpp_sample.write('\n')
                    if valid_stop:
                        break
            fpp_sample.close()
            print('valid set decode done')
            ud_epoch = (time.time() - ud_epoch) / 60.
            print('epoch cost time ... ', ud_epoch)

        # calculate wer and expRate
        if np.mod(uidx, validFreq) == 0 and valid_stop == False:
            os.system('python compute-wer.py ' + valid_output[0] + ' ' + valid_datasets[
                2] + ' ' + valid_result[0])

            fpp = open(valid_result[0])
            stuff = fpp.readlines()
            fpp.close()
            m = re.search('WER (.*)\n', stuff[0])
            valid_err = 100. * float(m.group(1))
            m = re.search('ExpRate (.*)\n', stuff[1])
            valid_sacc = 100. * float(m.group(1))
            history_errs.append(valid_err)

            # the first time validation or better model
            if uidx // validFreq == 0 or valid_err <= np.array(history_errs).min():
                bad_counter = 0
                print('Saving model params ... ')
                if multi_gpu_flag:
                    torch.save(WAP_model.module.state_dict(), saveto)
                else:
                    torch.save(WAP_model.state_dict(), saveto)
            if valid_sacc>52:
                if multi_gpu_flag:
                    torch.save(WAP_model.module.state_dict(), result_path + "wap_" + str(valid_sacc) + ".pkl")
                else:
                    torch.save(WAP_model.state_dict(), result_path + "wap_" + str(valid_sacc) + ".pkl")
            # worse model
            if uidx / validFreq != 0 and valid_err > np.array(history_errs).min():
                bad_counter += 1
                if bad_counter > patience:
                    if halfLrFlag == 2:
                        print('Early Stop!')
                        estop = True
                        break
                    else:
                        print('Lr decay and retrain!')
                        bad_counter = 0
                        lrate = lrate / 10.
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lrate
                        halfLrFlag += 1
            print('Valid WER: %.2f%%, ExpRate: %.2f%%' % (valid_err, valid_sacc))

        # finish after these many updates
        if uidx >= finish_after:
            print('Finishing after %d iterations!' % uidx)
            estop = True
            break

    print('Seen %d samples' % n_samples)

    # early stop
    if estop:
        break
