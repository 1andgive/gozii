# file address scripts
import platform
import os
import sys 

addr_dataroot=os.path.join('codes', 'tools', 'data')

sys.path.append('/mnt/server5_hard1/seungjun/XAI/BAN_XAI')
# BAN address
if(platform.system() == 'Linux'):
    addr_BAN = '/mnt/server5_hard1/seungjun/VQA/BAN'
    addr_test_imgs='/mnt/server5_hard1/seungjun/Data_Share/Datas/VQA_COCO/Images/test2015'#실제파일경로.cnn할때씀
    addr_train_imgs='/mnt/server5_hard1/seungjun/Data_Share/Datas/VQA_COCO/Images/train2014'
    addr_val_imgs='/mnt/server5_hard1/seungjun/Data_Share/Datas/VQA_COCO/Images/val2014'
    addr_hdf5path='/mnt/server5_hard1/seungjun/Data_Share/Datas/VQA_COCO/BottomUpPreTrain/hdf5' 
    # object의 후보군을 뽑아내고 post processing에대한... object detection을 한 결과를 따로 저장 (이게더빠름) train, val, test 에대한 fastRCN의 결과가 저장돼있음. 학습하고 테스트할때 hdf5path에 있는 걸 쓰거나 실제 영상을 쓰거나~~ 둘 중 하나만. demo할때는 둘다.
    addr_coco_cap_train_path='/mnt/server5_hard1/seungjun/Data_Share/Datas/VQA_COCO/annotations/captions_train2014.json' 
    #cap붙어있는건 json:
    addr_coco_cap_val_path = '/mnt/server5_hard1/seungjun/Data_Share/Datas/VQA_COCO/annotations/captions_val2014.json'
    addr_coco_cap_test_path = '/mnt/server5_hard1/seungjun/Data_Share/Datas/VQA_COCO/annotations/image_info_test2015.json'
    addr_coco_cap_test_dev_path = '/mnt/server5_hard1/seungjun/Data_Share/Datas/VQA_COCO/annotations/image_info_test-dev2015.json'
    addr_coco_cap_test2014_path = '/mnt/server5_hard1/seungjun/Data_Share/Datas/VQA_COCO/annotations/image_info_test2014.json'
    addr_cider_score_path = '/mnt/server5_hard1/seungjun/XAI/coco_caption_jiasen3' #신경쓰지마
    addr_vqae_val_path = '/mnt/server5_hard1/seungjun/Data_Share/Datas/VQA-E/VQA-E_val_set.json' #vqae ; XAI하는거니까 신경ㄴㄴ
    addr_vqae_train_path = '/mnt/server5_hard1/seungjun/Data_Share/Datas/VQA-E/VQA-E_train_set.json' #XAI
    # https://github.com/peteanderson80/bottom-up-attention 두가지 dataset.
    
    addr_hdf5fix_path = '/mnt/server5_hard1/seungjun/Data_Share/Datas/VQA_COCO/BottomUpPreTrain/hdf5' #adaptive dataset 에 대한 경로. 10 to 100
    addr_pklfix_path = '/mnt/server5_hard1/seungjun/Data_Share/Datas/VQA_COCO/BottomUpPreTrain/pkl'# 36 features per image. 모든 영상에 대해서 36개
    
elif(platform.system() == 'Windows'):
    addr_BAN = 'D:\\VQA\\BAN'
    addr_test_imgs='D:\\Data_Share\\Datas\\VQA_COCO\\Images\\test2015'
    addr_train_imgs='D:\\Data_Share\\Datas\\VQA_COCO\\Images\\train2014'
    addr_val_imgs='D:\\Data_Share\\Datas\\VQA_COCO\\Images\\val2014'
    addr_hdf5path='../../Data_Share/Datas/VQA_COCO/BottomUpPreTrain/hdf5'
    addr_coco_cap_train_path='D:\\Data_Share\\Datas\\VQA_COCO\\annotations\\captions_train2014.json'
    addr_coco_cap_val_path = 'D:\\Data_Share\\Datas\\VQA_COCO\\annotations\\captions_val2014.json'
    addr_coco_cap_test_path = 'D:\\Data_Share\\Datas\\VQA_COCO\\annotations\\image_info_test2015.json'
    addr_coco_cap_test_dev_path = 'D:\\Data_Share\\Datas\\VQA_COCO\\annotations\\image_info_test-dev2015.json'
    addr_coco_cap_test2014_path = 'D:\\Data_Share\\Datas\\VQA_COCO\\annotations\\image_info_test2014.json'
    addr_cider_score_path='D:\\XAI\\coco_caption_jiasen3'
    addr_vqae_val_path='D:\\Data_Share\\Datas\\VQA-E\\VQA-E_val_set.json'
    addr_vqae_train_path='D:\\Data_Share\\Datas\\VQA-E\\VQA-E_train_set.json'

    addr_hdf5fix_path = 'D:\\Data_Share\\Datas\\VQA_COCO\\BottomUpPreTrain\\hdf5'
    addr_pklfix_path = 'D:\\Data_Share\\Datas\\VQA_COCO\\BottomUpPreTrain\\pkl'