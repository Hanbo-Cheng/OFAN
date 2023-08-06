import numpy as np
import copy
import sys
import pickle as pkl
import random
import torch
from torch import nn
import os
import cv2
from numpy import mean
import numpy
import torch.nn.functional as F
from typing import List
from torch import FloatTensor, LongTensor
from einops import rearrange, repeat
from collections import OrderedDict

# load data
def dataIterator(feature_file, label_file, dictionary, batch_size, batch_Imagesize, maxlen, maxImagesize):
    # offline-train.pkl
    fp = open(feature_file, 'rb')
    features = pkl.load(fp)
    fp.close()

    # train_caption.txt
    fp2 = open(label_file, 'r')
    labels = fp2.readlines()
    fp2.close()

    targets = {}
    # map word to int with dictionary
    for l in labels:
        tmp = l.strip().split()
        uid = tmp[0]
        w_list = []
        for w in tmp[1:]:
            if dictionary.__contains__(w):
                w_list.append(dictionary[w])
            else:
                print('a word not in the dictionary !! sentence ', uid, 'word ', w)
                sys.exit()
        targets[uid] = w_list

    imageSize = {}
    for uid, fea in features.items():
        imageSize[uid] = fea.shape[0] * fea.shape[1]
    # sorted by sentence length, return a list with each triple element
    imageSize = sorted(imageSize.items(), key=lambda d: d[1])

    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    uidList = []
    biggest_image_size = 0

    i = 0
    for uid, size in imageSize:
        if size > biggest_image_size:
            biggest_image_size = size
        fea = features[uid]
        lab = targets[uid]
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print('sentence', uid, 'length bigger than', maxlen, 'ignore')
        elif size > maxImagesize:
            print('image', uid, 'size bigger than', maxImagesize, 'ignore')
        else:
            uidList.append(uid)
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                feature_batch = []
                label_batch = []
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    # last batch
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print('total ', len(feature_total), 'batch data loaded')
    return list(zip(feature_total, label_total)), uidList

def prepare_data_bidecoder_online(options, images_x, online_x, seqs_y):
    """
    """


    heights_x = [s.shape[0] for s in images_x]
    widths_x = [s.shape[1] for s in images_x]
    lengths_y = [len(s) for s in seqs_y]
    n_samples = len(heights_x)
    max_height_x = np.max(heights_x)
    max_width_x = np.max(widths_x)
    maxlen_y = np.max(lengths_y) + 1


    #L2R  y_in: <sos> y1, y2, ..., yn
    #L2R  y_out: y1, y2, ..., yn, <eos>
    x = np.zeros((n_samples, options['input_channels'] + options['online_input_channels'] + 1, max_height_x, max_width_x)).astype(np.float32)
    y_in  = np.zeros((maxlen_y, n_samples)).astype(np.int64)  # <sos> must be 0 in the dict
    # y_out = np.ones((maxlen_y, n_samples)).astype(np.int64)  # <eos> must be 1 in the dict

    x_mask = np.zeros((n_samples, max_height_x, max_width_x)).astype(np.float32)
    y_mask = np.zeros((maxlen_y, n_samples)).astype(np.float32)


    for idx, [s_x, s_online_x, s_y] in enumerate(zip(images_x, online_x, seqs_y)):
        x[idx, 0, :heights_x[idx], :widths_x[idx]] = s_x / 255.
        s_online_x =s_online_x.astype(np.float32)
        # s_online_x[:,:,0:-1] = s_online_x[:,:,0:-1]/(1024.)
        s_online_length = np.sqrt(np.square(s_online_x[:,:,0]) + np.square(s_online_x[:,:,1]))
        s_online_x[:,:,:-1] = s_online_x[:,:,:-1]/(s_online_length[:,:,None] + 1e-6)
        x[idx, 1:3, :heights_x[idx], :widths_x[idx]] = s_online_x[:,:,:-1].transpose(2,0,1) 
        x[idx, -1, :heights_x[idx], :widths_x[idx]] = s_online_x[:,:,-1]
        x[idx, -2, :heights_x[idx], :widths_x[idx]] = (s_online_x[:,:,-1]>0).astype(np.float32) # TODO: 待检查
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        y_in[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx] + 1, idx] = 1.


    #R2L: y_in:  <eos> yn, yn-1, ..., y3, y2, y1
    #R2L: y_out: yn, yn-1, ..., y2, y1, <sos>



    return x, x_mask, y_in, y_mask

class BatchBucket():
    def __init__(self, max_h, max_w, max_l, max_img_size, max_batch_size, 
                 feature_file, label_file, dictionary, mode='train',
                 use_all=True):
        self._max_img_size = max_img_size
        self._max_batch_size = max_batch_size
        self._fea_file = feature_file
        self._label_file = label_file
        self._dictionary_file = dictionary
        self._use_all = use_all
        self._dict_load()
        self._data_load()
        self.keys = self._calc_keys(max_h, max_w, max_l)
        self._make_plan()
        self._reset()
        self._mode = mode

    def _dict_load(self):
        fp = open(self._dictionary_file)
        stuff = fp.readlines()
        fp.close()
        self._lexicon = {}
        for l in stuff:
            w = l.strip().split()
            self._lexicon[w[0]] = int(w[1])

    def _data_load(self):

        fp_feature=open(self._fea_file, 'rb')
        self._features=pkl.load(fp_feature)
        fp_feature.close()

        fp_label = open(self._label_file, 'r')
        labels = fp_label.readlines()
        fp_label.close()


  
        self._targets = {}
        # map word to int with dictionary
        for l in labels:
            tmp = l.strip().split()
            uid = tmp[0]
            w_list = []
            for w in tmp[1:]:
                
                if self._lexicon.__contains__(w):
                    w_list.append(self._lexicon[w])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', w)
                    
                    
                
            self._targets[uid] = w_list

        # (uid, h, w, tgt_len)
        self._data_parser = [(uid, fea.shape[0], fea.shape[1], len(self._targets[uid])) for uid, fea in
                             self._features.items()]


    def _calc_keys(self, max_h, max_w, max_l):
        mh = mw = ml = 0
        for _, h, w, l in self._data_parser:
            if h > mh:
                mh = h
            if w > mw:
                mw = w
            if l > ml:
                ml = l
        max_h = min(max_h, mh)
        max_w = min(max_w, mw)
        max_l = min(max_l, ml)
        #print('Max:', max_h, max_w, max_l)
        keys = []
        init_h = 100 if 100 < max_h else max_h
        init_w = 100 if 100 < max_w else max_w
        init_l = max_l
        h_step = 50
        w_step = 100
        l_step = 20
        h = init_h
        #print(max_h, max_w, max_l)
        while h <= max_h:
            w = init_w
            while w <= max_w:
                l = init_l
                while l <= max_l: 
                    keys.append([h, w, l, h * w * l, 0])
                    #print(keys[-1])
                    if l < max_l and l + l_step > max_l:
                        l = max_l
                        #print(l)
                    else:
                        l += l_step
                if w < max_w and w + max(int((w*0.3 // 10) * 10), w_step) > max_w:
                    w = max_w
                else:
                    w = w + max(int((w*0.3 // 10) * 10), w_step)
            if h < max_h and h + max(int((h*0.5 // 10) * 10), h_step) > max_h:
                h = max_h
            else:
                h = h + max(int((h*0.5 // 10) * 10), h_step)
        keys = sorted(keys, key=lambda area: area[3])
        for _, h, w, l in self._data_parser:
            for i in range(len(keys)):
                hh, ww, ll, _, _ = keys[i]
                if h <= hh and w <= ww and l <= ll:
                    keys[i][-1] += 1
                    break
        new_keys = []
        n_samples = len(self._data_parser)
        th = n_samples * 0.01
        if self._use_all:
            th = 1
        num = 0
        for key in keys:
            hh, ww, ll, _, n = key
            num += n
            if num >= th:
                new_keys.append((hh, ww, ll))
                num = 0
        return new_keys

    def _make_plan(self):
        self._bucket_keys = []
        for h, w, l in self.keys:
            batch_size = int(self._max_img_size / (h * w))
            if batch_size > self._max_batch_size:
                batch_size = self._max_batch_size
            if batch_size == 0:
                batch_size = 1
            self._bucket_keys.append((batch_size, h, w, l))
        self._data_buckets = [[] for key in self._bucket_keys]
        unuse_num = 0
        for item in self._data_parser:
            flag = 0
            for key, bucket in zip(self._bucket_keys, self._data_buckets):
                _, h, w, l = key
                if item[1] <= h and item[2] <= w and item[3] <= l:
                    bucket.append(item)
                    flag = 1
                    break
            if flag == 0:
                #print(item, h, w, l)
                unuse_num += 1
        print('The number of unused samples: ', unuse_num)
        all_sample_num = 0
        for key, bucket in zip(self._bucket_keys, self._data_buckets):
            sample_num = len(bucket)
            all_sample_num += sample_num
            print('bucket {}, sample number={}'.format(key, len(bucket)))
        print('All samples number={}, raw samples number={}'.format(all_sample_num, len(self._data_parser)))

    def _reset(self):
        # shuffle data in each bucket
        for bucket in self._data_buckets:
            random.shuffle(bucket)
        self._batches = []
        for id, (key, bucket) in enumerate(zip(self._bucket_keys, self._data_buckets)):
            batch_size, _, _, _ = key
            bucket_len = len(bucket)
            batch_num = (bucket_len + batch_size - 1) // batch_size
            for i in range(batch_num):
                start = i * batch_size
                end = start + batch_size if start + batch_size < bucket_len else bucket_len
                if start != end:  # remove empty batch
                    self._batches.append(bucket[start:end])

    def get_batches(self):
        batches = []
        uid_batches = []
        for batch_info in self._batches:
            fea_batch = []
            label_batch = []

            for uid, _, _, _ in batch_info:
                feature = self._features[uid]
                label = self._targets[uid]

                fea_batch.append(feature)
                label_batch.append(label)

                uid_batches.append(uid)

            batches.append((fea_batch, label_batch))
        print("Number of Bucket", len(self._data_buckets),
              "Number of Batches", len(batches),
              "Number of Samples", len(uid_batches))
        return batches, uid_batches

class BatchBucket_online():
    def __init__(self, max_h, max_w, max_l, max_img_size, max_batch_size, 
                 feature_file, feature_online_file, label_file, dictionary, mode='train',
                 use_all=True):
        self._max_img_size = max_img_size
        self._max_batch_size = max_batch_size
        self._fea_file = feature_file
        self._fea_online_file = feature_online_file
        self._label_file = label_file
        self._dictionary_file = dictionary
        self._use_all = use_all
        self._dict_load()
        self._data_load()
        self.keys = self._calc_keys(max_h, max_w, max_l)
        self._make_plan()
        self._reset()
        self._mode = mode

    def _dict_load(self):
        fp = open(self._dictionary_file)
        stuff = fp.readlines()
        fp.close()
        self._lexicon = {}
        for l in stuff:
            w = l.strip().split()
            self._lexicon[w[0]] = int(w[1])

    def _data_load(self):

        fp_feature=open(self._fea_file, 'rb')
        self._features=pkl.load(fp_feature)
        fp_feature.close()

        fp_online_feature=open(self._fea_online_file, 'rb')
        self._online_features=pkl.load(fp_online_feature)
        fp_feature.close()

        fp_label = open(self._label_file, 'r')
        labels = fp_label.readlines()
        fp_label.close()


  
        self._targets = {}
        # map word to int with dictionary
        for l in labels:
            tmp = l.strip().split()
            uid = tmp[0]
            w_list = []
            for w in tmp[1:]:
                
                if self._lexicon.__contains__(w):
                    w_list.append(self._lexicon[w])
                else:
                    print ('a symbol not in the dictionary !! formula',uid ,'symbol', w)
                    
                    
                
            self._targets[uid] = w_list

        # (uid, h, w, tgt_len)
        self._data_parser = [(uid, fea.shape[0], fea.shape[1], len(self._targets[uid])) for uid, fea in
                             self._features.items()]


    def _calc_keys(self, max_h, max_w, max_l):
        mh = mw = ml = 0
        for _, h, w, l in self._data_parser:
            if h > mh:
                mh = h
            if w > mw:
                mw = w
            if l > ml:
                ml = l
        max_h = min(max_h, mh)
        max_w = min(max_w, mw)
        max_l = min(max_l, ml)
        #print('Max:', max_h, max_w, max_l)
        keys = []
        init_h = 100 if 100 < max_h else max_h
        init_w = 100 if 100 < max_w else max_w
        init_l = max_l
        h_step = 50
        w_step = 100
        l_step = 20
        h = init_h
        #print(max_h, max_w, max_l)
        while h <= max_h:
            w = init_w
            while w <= max_w:
                l = init_l
                while l <= max_l: 
                    keys.append([h, w, l, h * w * l, 0])
                    #print(keys[-1])
                    if l < max_l and l + l_step > max_l:
                        l = max_l
                        #print(l)
                    else:
                        l += l_step
                if w < max_w and w + max(int((w*0.3 // 10) * 10), w_step) > max_w:
                    w = max_w
                else:
                    w = w + max(int((w*0.3 // 10) * 10), w_step)
            if h < max_h and h + max(int((h*0.5 // 10) * 10), h_step) > max_h:
                h = max_h
            else:
                h = h + max(int((h*0.5 // 10) * 10), h_step)
        keys = sorted(keys, key=lambda area: area[3])
        for _, h, w, l in self._data_parser:
            for i in range(len(keys)):
                hh, ww, ll, _, _ = keys[i]
                if h <= hh and w <= ww and l <= ll:
                    keys[i][-1] += 1
                    break
        new_keys = []
        n_samples = len(self._data_parser)
        th = n_samples * 0.01
        if self._use_all:
            th = 1
        num = 0
        for key in keys:
            hh, ww, ll, _, n = key
            num += n
            if num >= th:
                new_keys.append((hh, ww, ll))
                num = 0
        return new_keys

    def _make_plan(self):
        self._bucket_keys = []
        for h, w, l in self.keys:
            batch_size = int(self._max_img_size / (h * w))
            if batch_size > self._max_batch_size:
                batch_size = self._max_batch_size
            if batch_size == 0:
                batch_size = 1
            self._bucket_keys.append((batch_size, h, w, l))
        self._data_buckets = [[] for key in self._bucket_keys]
        unuse_num = 0
        for item in self._data_parser:
            flag = 0
            for key, bucket in zip(self._bucket_keys, self._data_buckets):
                _, h, w, l = key
                if item[1] <= h and item[2] <= w and item[3] <= l:
                    bucket.append(item)
                    flag = 1
                    break
            if flag == 0:
                #print(item, h, w, l)
                unuse_num += 1
        print('The number of unused samples: ', unuse_num)
        all_sample_num = 0
        for key, bucket in zip(self._bucket_keys, self._data_buckets):
            sample_num = len(bucket)
            all_sample_num += sample_num
            print('bucket {}, sample number={}'.format(key, len(bucket)))
        print('All samples number={}, raw samples number={}'.format(all_sample_num, len(self._data_parser)))

    def _reset(self):
        # shuffle data in each bucket
        for bucket in self._data_buckets:
            random.shuffle(bucket)
        self._batches = []
        for id, (key, bucket) in enumerate(zip(self._bucket_keys, self._data_buckets)):
            batch_size, _, _, _ = key
            bucket_len = len(bucket)
            batch_num = (bucket_len + batch_size - 1) // batch_size
            for i in range(batch_num):
                start = i * batch_size
                end = start + batch_size if start + batch_size < bucket_len else bucket_len
                if start != end:  # remove empty batch
                    self._batches.append(bucket[start:end])

    def get_batches(self):
        batches = []
        uid_batches = []
        for batch_info in self._batches:
            fea_batch = []
            on_fea_batch = []
            label_batch = []

            for uid, _, _, _ in batch_info:
                feature = self._features[uid]
                label = self._targets[uid]
                on_feature = self._online_features[uid]

                fea_batch.append(feature)
                label_batch.append(label)
                on_fea_batch.append(on_feature)

                uid_batches.append(uid)

            batches.append((fea_batch, on_fea_batch, label_batch))
        print("Number of Bucket", len(self._data_buckets),
              "Number of Batches", len(batches),
              "Number of Samples", len(uid_batches))
        return batches, uid_batches
    
def dataIterator_online(feature_file, on_feature_file, label_file, dictionary, batch_size, batch_Imagesize, maxlen, maxImagesize):
    # offline-train.pkl
    fp = open(feature_file, 'rb')
    features = pkl.load(fp)
    fp.close()

    fp = open(on_feature_file, 'rb')
    on_features = pkl.load(fp)
    fp.close()

    # train_caption.txt
    fp2 = open(label_file, 'r')
    labels = fp2.readlines()
    fp2.close()

    targets = {}
    for l in labels:
        tmp = l.strip().split()
        uid = tmp[0]
        w_list = []
        for w in tmp[1:]:
            if dictionary.__contains__(w):
                w_list.append(dictionary[w])
            else:
                print('a word not in the dictionary !! sentence ', uid, 'word ', w)
                sys.exit()
        targets[uid] = w_list

    imageSize = {}
    for uid, fea in features.items():
        imageSize[uid] = fea.shape[0] * fea.shape[1]
    # sorted by sentence length, return a list with each triple element
    imageSize = sorted(imageSize.items(), key=lambda d: d[1])

    feature_batch = []
    on_features_batch = []
    label_batch = []
    feature_total = []
    onfeature_total = []
    label_total = []
    uidList = []
    biggest_image_size = 0

    i = 0
    for uid, size in imageSize:
        if size > biggest_image_size:
            biggest_image_size = size
        fea = features[uid]
        on_fea = on_features[uid]
        lab = targets[uid]
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print('sentence', uid, 'length bigger than', maxlen, 'ignore')
        elif size > maxImagesize:
            print('image', uid, 'size bigger than', maxImagesize, 'ignore')
        else:
            uidList.append(uid)
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                feature_total.append(feature_batch)
                onfeature_total.append(on_features_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                feature_batch = []
                on_features_batch = []
                label_batch = []
                feature_batch.append(fea)
                on_features_batch.append(on_fea)
                label_batch.append(lab)
                i += 1
            else:
                feature_batch.append(fea)
                on_features_batch.append(on_fea)
                label_batch.append(lab)
                i += 1

    # last batch
    feature_total.append(feature_batch)
    onfeature_total.append(on_features_batch)
    label_total.append(label_batch)
    print('total ', len(feature_total), 'batch data loaded')
    return list(zip(feature_total, onfeature_total, label_total)), uidList

# load dictionary
def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for l in stuff:
        w = l.strip().split()
        lexicon[w[0]] = int(w[1])
    print('total words/phones', len(lexicon))
    return lexicon


# create batch
def prepare_data(options, images_x, seqs_y):
    heights_x = [s.shape[0] for s in images_x]
    widths_x = [s.shape[1] for s in images_x]
    lengths_y = [len(s) for s in seqs_y]
    n_samples = len(heights_x)
    max_height_x = np.max(heights_x)
    max_width_x = np.max(widths_x)
    maxlen_y = np.max(lengths_y) + 1
    x = np.zeros((n_samples, options['input_channels'], max_height_x, max_width_x)).astype(np.float32)
    y = np.zeros((maxlen_y, n_samples)).astype(np.int64)  # <eos> must be 0 in the dict
    x_mask = np.zeros((n_samples, max_height_x, max_width_x)).astype(np.float32)
    y_mask = np.zeros((maxlen_y, n_samples)).astype(np.float32)
    for idx, [s_x, s_y] in enumerate(zip(images_x, seqs_y)):
        x[idx, :, :heights_x[idx], :widths_x[idx]] = s_x / 255.
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx] + 1, idx] = 1.
    return x, x_mask, y, y_mask


# beam search
def gen_sample(model, x, params, gpu_flag, k=1, maxlen=30):
    sample = []
    sample_score = []
    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype(np.float32)

    if gpu_flag:
        next_state, ctx0 = model.module.f_init(x)
    else:
        next_state, ctx0 = model.f_init(x)
    next_w = -1 * np.ones((1,)).astype(np.int64)
    next_w = torch.from_numpy(next_w).cuda()
    next_alpha_past = torch.zeros(1, ctx0.shape[2], ctx0.shape[3]).cuda()
    ctx0 = ctx0.cpu().numpy()

    for ii in range(maxlen):
        ctx = np.tile(ctx0, [live_k, 1, 1, 1])
        ctx = torch.from_numpy(ctx).cuda()
        if gpu_flag:
            next_p, next_state, next_alpha_past = model.module.f_next(params, next_w, None, ctx, None, next_state,
                                                                      next_alpha_past, True)
        else:
            next_p, next_state, next_alpha_past = model.f_next(params, next_w, None, ctx, None, next_state,
                                                               next_alpha_past, True)
        next_p = next_p.cpu().numpy()
        next_state = next_state.cpu().numpy()
        next_alpha_past = next_alpha_past.cpu().numpy()

        cand_scores = hyp_scores[:, None] - np.log(next_p)
        cand_flat = cand_scores.flatten()

        ranks_flat = cand_flat.argsort()[:(k - dead_k)]
        voc_size = next_p.shape[1]
        trans_indices = ranks_flat // voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        new_hyp_samples = []
        new_hyp_scores = np.zeros(k - dead_k).astype(np.float32)
        new_hyp_states = []
        new_hyp_alpha_past = []
        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti] + [wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_states.append(copy.copy(next_state[ti]))
            new_hyp_alpha_past.append(copy.copy(next_alpha_past[ti]))

        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = []
        hyp_alpha_past = []
        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states.append(new_hyp_states[idx])
                hyp_alpha_past.append(new_hyp_alpha_past[idx])
        hyp_scores = np.array(hyp_scores)
        live_k = new_live_k

        # whether finish beam search
        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_w = np.array([w[-1] for w in hyp_samples])
        next_state = np.array(hyp_states)
        next_alpha_past = np.array(hyp_alpha_past)
        next_w = torch.from_numpy(next_w).cuda()
        next_state = torch.from_numpy(next_state).cuda()
        next_alpha_past = torch.from_numpy(next_alpha_past).cuda()
    return sample, sample_score

def gen_sample_on(model, x, params, gpu_flag, k=1, maxlen=30):
    sample = []
    sample_score = []
    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype(np.float32)

    if gpu_flag:
        next_state, ctx0_off, ctx0_on = model.module.f_init(x)
    else:
        next_state, ctx0_off, ctx0_on = model.f_init(x)
    next_w = -1 * np.ones((1,)).astype(np.int64)
    next_w = torch.from_numpy(next_w).cuda()
    next_alpha_past_off = torch.zeros(1, ctx0_off.shape[2], ctx0_off.shape[3]).cuda()
    next_alpha_past_on = torch.zeros(1, ctx0_on.shape[2], ctx0_on.shape[3]).cuda()
    ctx0_off = ctx0_off.cpu().numpy()
    ctx0_on = ctx0_on.cpu().numpy()

    for ii in range(maxlen):
        ctx_off = np.tile(ctx0_off, [live_k, 1, 1, 1])
        ctx_on = np.tile(ctx0_on, [live_k, 1, 1, 1])
        ctx_off = torch.from_numpy(ctx_off).cuda()
        ctx_on = torch.from_numpy(ctx_on).cuda()
        if gpu_flag:
            next_p, next_state, next_alpha_past_off, next_alpha_past_on = model.module.f_next(params, next_w, None, ctx_off, ctx_on, None, next_state,
                                                                      next_alpha_past_off, next_alpha_past_on, True)
        else:
            next_p, next_state, next_alpha_past_off, next_alpha_past_on = model.f_next(params, next_w, None, ctx_off, ctx_on, None, next_state,
                                                               next_alpha_past_off, next_alpha_past_on, True)
        next_p = next_p.cpu().numpy()
        next_state = next_state.cpu().numpy()
        next_alpha_past_off = next_alpha_past_off.cpu().numpy()
        next_alpha_past_on = next_alpha_past_on.cpu().numpy()

        cand_scores = hyp_scores[:, None] - np.log(next_p)
        cand_flat = cand_scores.flatten()

        ranks_flat = cand_flat.argsort()[:(k - dead_k)]
        voc_size = next_p.shape[1]
        trans_indices = ranks_flat // voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        new_hyp_samples = []
        new_hyp_scores = np.zeros(k - dead_k).astype(np.float32)
        new_hyp_states = []
        new_hyp_alpha_past_off = []
        new_hyp_alpha_past_on = []
        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti] + [wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_states.append(copy.copy(next_state[ti]))
            new_hyp_alpha_past_off.append(copy.copy(next_alpha_past_off[ti]))
            new_hyp_alpha_past_on.append(copy.copy(next_alpha_past_on[ti]))

        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = []
        hyp_alpha_past_off = []
        hyp_alpha_past_on = []
        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states.append(new_hyp_states[idx])
                hyp_alpha_past_off.append(new_hyp_alpha_past_off[idx])
                hyp_alpha_past_on.append(new_hyp_alpha_past_on[idx])
        hyp_scores = np.array(hyp_scores)
        live_k = new_live_k

        # whether finish beam search
        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_w = np.array([w[-1] for w in hyp_samples])
        next_state = np.array(hyp_states)
        next_alpha_past_off = np.array(hyp_alpha_past_off)
        next_alpha_past_on = np.array(hyp_alpha_past_on)
        next_w = torch.from_numpy(next_w).cuda()
        next_state = torch.from_numpy(next_state).cuda()
        next_alpha_past_off = torch.from_numpy(next_alpha_past_off).cuda()
        next_alpha_past_on = torch.from_numpy(next_alpha_past_on).cuda()
    return sample, sample_score

# init model params
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass

    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass
