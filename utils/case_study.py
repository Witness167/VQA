import json
import shutil
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

"""
'question_id' : quesId,
				'answers' : gts[quesId]['answers'],
				'answer_pred' : resAns,
				'accuracy' : avgGTAcc,
				'image_id' : int(gts[quesId]['image_id']),
				'answer_type' : ansType,
				'quesType' : quesType,
				'question' : self.vqa.qqa[quesId]['question']
"""

"""
用于找到可视化分析中使用的例子
"""

def rcnn_show () :
    return

def find_accuracy09() :
    """
    1. accuracy > 0.9
    2. 输出的json包含fcd和sa ga字段
    3. fcd字段sa和ga差距大
    4. questype是huw many的
    """
    """
    case_study_out+fcd+attenmap.json
    atten_map_SA_layers {3,32, 20,20}  {0.2.5}
    atten_map_GA_layers {3,32,20,14}   {0.2.5}
    fcd_x_SA_layers {6,32, 100}
    fcd_x_GA_layers {6,32,100}
    case_study_lang_att {32,100,1}
    case_study_img_att {32,14,1}
    question_id {}
    """
    """
    results/result_test/case_study_FE6A-mul0+lin+sof-val_epoch13.json
    
    visual_0.9.json : accuracy > 0.9
    {"question_id": 262242000, 
    "answers": [{"answer": "floor and ball", "answer_confidence": "yes", "answer_id": 1}, 
                {"answer": "ground and tennis ball", "answer_confidence": "yes", "answer_id": 2}, 
                {"answer": "ground", "answer_confidence": "yes", "answer_id": 3}, 
                {"answer": "ground", "answer_confidence": "yes", "answer_id": 4}, 
                {"answer": "court and ball", "answer_confidence": "yes", "answer_id": 5},
                 {"answer": "ball", "answer_confidence": "yes", "answer_id": 6}, 
                 {"answer": "ball", "answer_confidence": "yes", "answer_id": 7}, 
                 {"answer": "ball", "answer_confidence": "yes", "answer_id": 8}, 
                 {"answer": "ground ball", "answer_confidence": "yes", "answer_id": 9},
                  {"answer": "ball", "answer_confidence": "yes", "answer_id": 10}], 
    "answer_pred": "ball", 
    "accuracy": 1.0, 
    "image_id": 262242,
    "answer_type": "other",
    "quesType": "what is the", 
    "question": "What is the women's racket touching?"}, 
    """
    """
    {
    "question_id": 262242000,
    atten_map_SA_layers {3,32, 20,20}  {0.2.5}
    atten_map_GA_layers {3,32,20,14}   {0.2.5}
    fcd_x_SA_layers {6,32, 100}
    fcd_x_GA_layers {6,32,100}
    case_study_lang_att {32,100,1}
    case_study_img_att {32,14,1}
    "answer_pred": "ball", 
    "accuracy": 1.0, 
    "image_id": 262242,
    "answer_type": "other",
    "quesType": "what is the", 
    "question": "What is the women's racket touching?"
    }
    """
    all = json.load(open('visual_0.9_howmany.json', 'r'))
    ques_id_list = []
    for select_each in all:
        ques_id_list.append(select_each['question_id'])
    print("load select questionid done........")

    case_study_fcd = json.load(open('case_study_out+fcd+attenmap.json', 'r'))
    case_size = len(case_study_fcd)
    print('case_size is :{}'.format(case_size))
    case_all = []
    show_list = [0, 2, 5]
    for i in range(case_size) :
        case_fcd = case_study_fcd[i]
        if case_fcd['question_id'] in ques_id_list:
            case_all.append(case_fcd)

    case_study_compare_file = \
        'visual_0.9_howmany_attmap' + \
        '.json'
    json.dump(case_all, open(case_study_compare_file, 'w'))
    print('Save the case study to file: {}'.format(case_study_compare_file))
    print(len(case_all))
    print("find all fcd obviously")

def find_124() :
    """
    1. accuracy > 0.9
    2. 输出的json包含fcd和sa ga字段
    3. fcd字段sa和ga差距大
    4. questype是huw many的
    """

    all = json.load(open('visual_0.9_howmany.json', 'r'))
    print("load base done........")

    case_study_fcd = json.load(open('visual_0.9_howmany_attmap.json', 'r'))
    print("load attemp done........")
    case_size = len(case_study_fcd)
    print('case_size is :{}'.format(case_size))
    case_all = []
    for i in range(case_size) :
        case_base = all[i]
        case_fcd = case_study_fcd[i]
        assert case_fcd['question_id'] == case_base['question_id']
        case_each = case_fcd
        case_each['image_id'] = case_base['image_id']
        case_each['question'] = case_base['question']
        case_each['answer_pred'] = case_base['answer_pred']
        case_all.append(case_each)

    case_study_compare_file = \
        'visual_0.9_howmany_ttemp' + \
        '.json'
    json.dump(case_all, open(case_study_compare_file, 'w'))
    print('Save the case study to file: {}'.format(case_study_compare_file))
    print(len(case_all))
    print("124")

def find_1234() :
    """
    1. accuracy > 0.9
    2. 输出的json包含fcd和sa ga字段
    3. fcd字段sa和ga差距大
    4. questype是huw many的
    """

    case_study_fcd = json.load(open('visual_0.9_howmany_ttemp.json', 'r'))
    print("load attemp done........")
    case_size = len(case_study_fcd)
    print('case_size is :{}'.format(case_size))
    case_all = []
    rate = 0.67
    print("rate:" + str(rate))
    for i in range(case_size) :
        case_base = case_study_fcd[i]
        append_tage = False
        for i in range(6):
            for j in range(len(case_base['fcd_x_GA_layers'][i])):
                if case_base['fcd_x_GA_layers'][i][j] > rate:
                    append_tage = True
        if append_tage == True:
            case_all.append(case_base)

    case_study_compare_file = \
        'visual_0.9_howmany_fcdobvious' + \
        '.json'
    json.dump(case_all, open(case_study_compare_file, 'w'))
    print('Save the case study to file: {}'.format(case_study_compare_file))
    print(len(case_all))
    print("1234")

def main () :
    #case_study_umf = json.load(open('results/result_test/case_study_UMF-ED-6-55-val_epoch13.json', 'r'))
    case_study_fcd = json.load(open('results/result_test/case_study_FE6A-mul0+lin+sof-val_epoch13.json', 'r'))
    '''
    {"question_id": 262161011, 
    "answers": [{"answer": "white and green", "answer_confidence": "yes", "answer_id": 1}, 
    "answer_pred": "green", 
    "accuracy": 0.3, 
    "image_id": 262161, 
    "answer_type": "other", 
    "quesType": "what color is the", 
    "question": "What color is the bike?"}
    '''
    case_size = len(case_study_fcd)
    case_compare = []
    for i in range(case_size):
        #case_umf = case_study_umf[i]
        case_fcd = case_study_fcd[i]
        #assert case_umf['question_id'] == case_fcd['question_id']
        # if case_umf['accuracy'] <= 0.3 and case_fcd['accuracy'] >=0.9:
        #     case_compare_each = {
        #         'image_id' : case_umf['image_id'],
        #         'question' : case_umf['question'],
        #         'question_id' : case_umf['question_id'],
        #         'answers' : case_umf['answers'],
        #         'answer_pred_umf' : case_umf['answer_pred'],
        #         'answer_pred_fcd' : case_fcd['answer_pred']
        #     }
        #case_compare.append(case_compare_each)
        case_compare_each = {
            'image_id' : case_fcd['image_id'],
            'question' : case_fcd['question'],
            'question_id' : case_fcd['question_id'],
            'answers' : case_fcd['answers'],
            'answer_pred_fcd' : case_fcd['answer_pred'],
            'accuracy' : case_fcd['accuracy'],
            'quesType' : case_fcd['quesType'],

        }
        case_compare.append(case_compare_each)
            # case study
    case_study_compare_file = \
        'case_study_compare' + \
        '.json'
    json.dump(case_compare, open(case_study_compare_file, 'w'))
    print('Save the case study to file: {}'.format(case_study_compare_file))
    print(len(case_compare))

def case_study_select_fcdweight () :

    selected = json.load(open('case_study_compare.json', 'r'))
    ques_id_list = []
    for select_each in selected:
        ques_id_list.append(select_each['question_id'])
    print("load select questionid done........")

    fcd_weight_all = json.load(open('case_study_out+fcd.json', 'r'))
    all_size = len(fcd_weight_all)

    case_study_out_fcd_selected = []

    for fcd_each in fcd_weight_all:
        if fcd_each['question_id'] in ques_id_list:
            case_study_out_fcd_selected_each = {
                'question_id' : fcd_each['question_id'],
                'fcd_x_SA_layers' : fcd_each['fcd_x_SA_layers'],
                'fcd_x_GA_layers' : fcd_each['fcd_x_GA_layers'],
                'case_study_lang_att' : fcd_each['case_study_lang_att'],
                'case_study_img_att' : fcd_each['case_study_img_att']
            }
            case_study_out_fcd_selected.append(case_study_out_fcd_selected_each)
    case_study_out_fcd_selected_file = \
        'case_study_out+fcd_selected' + \
        '.json'
    json.dump(case_study_out_fcd_selected, open(case_study_out_fcd_selected_file, 'w'))
    print('Save the case study to file: {}'.format(case_study_out_fcd_selected_file))
    print(len(case_study_out_fcd_selected))

def case_study_select_fcdweight_obviously () :
    fcd_weight_selected = json.load(open('case_study_out+fcd_selected.json', 'r'))
    all_size = len(fcd_weight_selected)
    print("load done.....")

    case_study_out_fcd_selected_obviously = []

    for fcd_each in fcd_weight_selected:
        append_tage = False
        for i in range(6):
            for j in range(len(fcd_each['fcd_x_GA_layers'][i])):
                if fcd_each['fcd_x_GA_layers'][i][j] / 512 > 0.69:
                    append_tage = True
        if append_tage == True:
            case_study_out_fcd_selected_obviously.append(fcd_each)

    case_study_out_fcd_selected_obviously_file = \
        'case_study_out+fcd_selected_obviously_069' + \
        '.json'
    json.dump(case_study_out_fcd_selected_obviously, open(case_study_out_fcd_selected_obviously_file, 'w'))
    print('Save the case study to file: {}'.format(case_study_out_fcd_selected_obviously_file))
    print(len(case_study_out_fcd_selected_obviously))
def case_study_compare_obviously () :

    selected = json.load(open('case_study_out+fcd_selected_obviously_069.json', 'r'))
    ques_id_list = []
    for select_each in selected:
        ques_id_list.append(select_each['question_id'])
    print("load select questionid done........")

    fcd_weight_all = json.load(open('case_study_compare.json', 'r'))

    case_study_out_fcd_selected = []

    for fcd_each in fcd_weight_all:
        if fcd_each['question_id'] in ques_id_list:
            case_study_out_fcd_selected.append(fcd_each)
    case_study_out_fcd_selected_file = \
        'case_study_compare_obviously' + \
        '.json'
    json.dump(case_study_out_fcd_selected, open(case_study_out_fcd_selected_file, 'w'))
    print('Save the case study to file: {}'.format(case_study_out_fcd_selected_file))
    print(len(case_study_out_fcd_selected))

def remove_image () :
    delete_img_feat_path_list = glob.glob('/GPUFS/hit_qliao_1/lxx/mcan-vqa-master/datasets/case_study_image/' + '*.jpg')
    print(len(delete_img_feat_path_list))
    delete_img_feat_path_list_name = []
    for path in delete_img_feat_path_list:
        delete_img_feat_path_list_name.append(str(path.split('/')[-1].split('_')[-1].split('.')[0]))

    img_feat_path_list = glob.glob('/GPUFS/hit_qliao_1/lxx/mcan-vqa-master/datasets/raw/val2014/' + '*.jpg')
    print(len(img_feat_path_list))
    for path in img_feat_path_list:
        if str(path.split('/')[-1].split('_')[-1].split('.')[0]) in delete_img_feat_path_list_name:
            print(path)

def copy_image () :
    img_feat_path_list = glob.glob('/GPUFS/hit_qliao_1/lxx/mcan-vqa-master/datasets/case_study_image/' + '*.jpg')
    for path in img_feat_path_list:
        print(path)
        dst = os.path.join('/GPUFS/hit_qliao_1/lxx/mcan-vqa-master/datasets/raw/val2014', 'COCO_val2014_' + str(path.split('/')[-1].split('_')[-1].split('.')[0] + '.jpg'))
        shutil.copyfile(path, dst)

def draw_zhe_xain () :

    #data
    all = [
        [64.8, 65.8, 66.18, 67.07], #umf
        [65, 65.8, 66.27, 67.08], #self
        [65.1, 65.8, 66.22, 67.13], #cross
        [65.1, 66.1, 66.74, 67.21]  #fcd
    ]
    other = [
        [56.5, 57.1, 57.41, 58.49],
        [56.6, 57.3, 57.38, 58.52],
        [56.7, 57.3, 57.39, 58.45],
        [56.8, 57.7, 58.31, 58.52]
    ]
    yes_no = [
        [82.3, 83.5, 83.81, 84.8],
        [82.4, 83.3, 84.11, 84.8],
        [82.6, 83.3, 84.05, 84.8],
        [82.5, 83.6, 84.24, 85]
    ]
    number = [
        [46, 47.4, 48.63, 48.65],
        [47, 47.8, 48.58, 48.48],
        [46.9, 48, 48.36, 49.05],
        [46.6, 47.7, 48.28, 49.09]
    ]
    x = [1, 2, 4, 6]

    plt.figure()

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,8))
    #0
    ax[0].plot(x, all[0])
    ax[0].plot(x, all[1])
    ax[0].plot(x, all[2])
    ax[0].plot(x, all[3])
    #1
    ax[1].plot(x, other[0])
    ax[1].plot(x, other[1])
    ax[1].plot(x, other[2])
    ax[1].plot(x, other[3])
    #2
    ax[2].plot(x, yes_no[0])
    ax[2].plot(x, yes_no[1])
    ax[2].plot(x, yes_no[2])
    ax[2].plot(x, yes_no[3])
    #3
    ax[3].plot(x, number[0])
    ax[3].plot(x, number[1])
    ax[3].plot(x, number[2])
    ax[3].plot(x, number[3])


    plt.savefig('./case_study/test.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    find_1234()