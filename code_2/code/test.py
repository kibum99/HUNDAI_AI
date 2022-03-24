import string
import argparse
import easydict
import time
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opt):
    ret1 = []
    ret2 = []
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)
    st = time.time()
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)
                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)
            ret1 += image_path_list
            ret2 += preds_str
    return [ret1,ret2,time.time()-st]

def demoPy(image_folder, saved_model, imgH, imgW, character, rgb=True, PAD=True):
    
    opt = easydict.EasyDict({
        "image_folder" : image_folder,
        "workers" : 0,
        "batch_size" : 32,
        "saved_model" : saved_model,
        "batch_max_length" : 25,
        "imgH" : imgH,
        "imgW" : imgW,
        "rgb" : rgb,
        "character" : character,
        "PAD" : PAD,
        "Transformation" : "TPS",
        "FeatureExtraction" : "ResNet",
        "SequenceModeling" : "BiLSTM",
        "Prediction" : "CTC",
        "num_fiducial" : 20,
        "input_channel" : 1,
        "output_channel" : 512,
        "hidden_size" : 256,
        "num_gpu" : 0
    })

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    return demo(opt)
