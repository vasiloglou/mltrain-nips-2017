import os
import argparse
import json

def download_model(path):
    os.system('wget https://filebox.ece.vt.edu/~jiasenlu/codeRelease/visDial.pytorch/model/HCIAE-D-MLE.pth -P %s' %(path))
    os.system('wget https://filebox.ece.vt.edu/~jiasenlu/codeRelease/visDial.pytorch/model/HCIAE-G-MLE.pth -P %s' %(path))
    os.system('wget https://filebox.ece.vt.edu/~jiasenlu/codeRelease/visDial.pytorch/model/HCIAE-G-DIS.pth -P %s' %(path))

def download_feat(path):
    os.system('wget https://filebox.ece.vt.edu/~jiasenlu/codeRelease/visDial.pytorch/data/vdl_img_vgg.h5 -P %s' %(path))
    os.system('wget https://filebox.ece.vt.edu/~jiasenlu/codeRelease/visDial.pytorch/data/visdial_data.h5 -P %s' %(path))
    os.system('wget https://filebox.ece.vt.edu/~jiasenlu/codeRelease/visDial.pytorch/data/visdial_params.json -P %s' %(path))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--path', default='', 
    							type=str, help='target path to save the model and feature')
  
    args = parser.parse_args()
    params = vars(args)
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)

    model_path = os.path.join(params['path'], 'save')
    download_model(model_path)

    # data_path = os.path.join(params['path'], 'data')
    # download_feat(data_path)

    print('Download finished ...')