"""
Simple Translate CLI
"""
from contextlib import ExitStack

import mxnet as mx
import sys
import argparse
import logging
import time
import sockeye.utils
from sockeye.utils import acquire_gpu, get_num_gpus
import sockeye.data_io
import sockeye.arguments as arguments
import sockeye.inference
import re
import numpy as np


def main():
    
    params = argparse.ArgumentParser(description='Translate from STDIN to STDOUT')
    params.add_argument('--model-prefixes', '-m', required=False, nargs='+',
                        help='model prefix(es). Use multiple for ensemble decoding. ' +
                             'Model prefix determines config, best epoch params and vocab files.')
    params.add_argument('--epochs', '-e', required=False, default=None, type=int, nargs='+',
                        help='If not given, chooses best epochs/checkpoints for each model. If specified, must have the same ' +
                             'length as --model-prefix and be integer')

    params.add_argument('--max-input-len', '-n', type=int, default=None,
                        help='Maximum sentence length. Default: value from trained model(s).')
    params.add_argument('--output-type', default='translation', choices=["translation", "align_plot", "align_text"],
                        help='Either print the translation or visualize the alignment. Default: translation')
    params.add_argument('--align-plot-prefix', default="align",
                        help='Prefix used when plotting the alignment.')
    params.add_argument('--log-level', default=logging.INFO, type=int,
                        choices=[logging.INFO, logging.WARN, logging.DEBUG])
    params.add_argument('--beam-size', '-b', type=int, default=1, help='beam size. If == 1, greedy decode')
    params.add_argument('--ensemble-mode', type=str, default='linear', choices=['linear', 'log_linear'],
                        help='Ensemble mode: linear or log-linear interpolation of model predictions. Default: linear')
    params.add_argument('--softmax-temperature', type=float, default=None, required=False,
                        help='Controls peakiness of model predictions. Values < 1.0 produce peaked predictions, ' +
                             'values > 1.0 produce smoothed distributions.')
    params = arguments.add_device_args(params)
    args = params.parse_args()

    args.model_prefixes = ['model/']
    assert args.beam_size > 0, "Beam size must be 1 or greater."
    if args.epochs is not None:
        assert len(args.epochs) == len(args.model_prefixes), "must provide epochs for each model"

    #sockeye.utils.setup_logging(args.log_level)
    logging.basicConfig(filename='test.log', level=logging.INFO)
    
    logging.info("Command: %s", " ".join(sys.argv))
    logging.info("Arguments: %s", args)

    with ExitStack() as exit_stack:
        if args.use_cpu:
            context = mx.cpu()
        else:
            num_gpus = get_num_gpus()
            assert num_gpus > 0, "No GPUs found, consider running on the CPU with --use-cpu " \
                                 "(note: check depends on nvidia-smi and this could also mean that the nvidia-smi " \
                                 "binary isn't on the path)."
            assert len(args.device_ids) == 1, "cannot run on multiple devices for now"
            gpu_id = args.device_ids[0]
            if gpu_id < 0:
                # get an automatic gpu id:
                gpu_id = exit_stack.enter_context(acquire_gpu())
            context = mx.gpu(gpu_id)

        translator = sockeye.inference.Translator(context,
                                               args.ensemble_mode,
                                               *sockeye.inference.load_models(context,
                                                                           args.max_input_len,
                                                                           args.beam_size,
                                                                           args.model_prefixes,
                                                                           args.epochs,
                                                                           args.softmax_temperature))
        ############ CHANGE HERE ########################
        sample_file = open('train_question_token.txt','r')
        encoder_file = open('train_question_encoder.txt', "w")
        #################################################
        
        for i, line in enumerate(sample_file,1):
            trans_input = translator.make_input(i,line)
            source, source_length, bucket_key = translator._get_inference_input(trans_input.tokens)
            encoded_source, _ , _ , _, _  = translator.models[0].run_encoder(source, source_length, bucket_key)
            last_slice_source = mx.ndarray.mean(encoded_source, axis=1, keepdims=True)
            last_slice_source = last_slice_source.reshape((-1,))
            encoder_file.write(" ".join(map(str, last_slice_source.asnumpy()))+"\n")

        encoder_file.close()

if __name__ == '__main__':
    main()
