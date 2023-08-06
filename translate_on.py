import argparse
import numpy as np
import os
import re
import torch
from utils import dataIterator, load_dict, gen_sample, dataIterator_online, prepare_data_bidecoder_online, gen_sample_on
from encoder_decoder_optical import Encoder_Decoder


def main(model_path, dictionary_target, fea, fea_on, latex, saveto, output, beam_k=5):
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

    # load model
    model = Encoder_Decoder(params)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.cuda()

    # load dictionary
    worddicts = load_dict(dictionary_target)
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

    # load data
    # test, test_uid_list = dataIterator(fea, latex, worddicts, batch_size=8, batch_Imagesize=500000, maxlen=20000,
    #                                    maxImagesize=500000)
    
    test, test_uid_list = dataIterator_online(fea, fea_on, latex, worddicts, batch_size=8,
                                     batch_Imagesize=500000, maxlen=20000, maxImagesize=500000)

    # testing
    model.eval()
    with torch.no_grad():
        fpp_sample = open(saveto, 'w')
        test_count_idx = 0
        print('Decoding ... ')
        for x, on_x, y in test:
            for xx,xx_on,yy in zip(x, on_x, y):
                # print('%d : %s' % (test_count_idx + 1, test_uid_list[test_count_idx]))
                x, x_mask, y, y_mask = prepare_data_bidecoder_online(params, [xx], [xx_on], [yy])
                xx_pad = torch.from_numpy(x).cuda() 
                sample, score = gen_sample_on(model, xx_pad, params, False, k=10, maxlen=200)
                score = score / np.array([len(s) for s in sample])
                ss = sample[score.argmin()]
                # write decoding results
                fpp_sample.write(test_uid_list[test_count_idx])
                test_count_idx = test_count_idx + 1
                # symbols (without <eos>)
                for vv in ss:
                    if vv == 0:  # <eos>
                        break
                    fpp_sample.write(' ' + worddicts_r[vv])
                fpp_sample.write('\n')
    fpp_sample.close()
    print('test set decode done')
    os.system('python compute-wer.py ' + saveto + ' ' + latex + ' ' + output)
    fpp = open(output)
    stuff = fpp.readlines()
    fpp.close()
    m = re.search('WER (.*)\n', stuff[0])
    test_per = 100. * float(m.group(1))
    m = re.search('ExpRate (.*)\n', stuff[1])
    test_sacc = 100. * float(m.group(1))
    print('Valid WER: %.2f%%, ExpRate: %.2f%%' % (test_per, test_sacc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('model_path', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('fea', type=str)
    parser.add_argument('fea_on', type=str)
    parser.add_argument('latex', type=str)
    parser.add_argument('saveto', type=str)
    parser.add_argument('output', type=str)

    args = parser.parse_args()

    main(args.model_path, args.dictionary_target, args.fea, args.fea_on, args.latex, args.saveto, args.output, beam_k=args.k)
