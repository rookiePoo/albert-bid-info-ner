from albert_infer import BERT_CRF
from albert_infer import BertNerDataProcessor
import os
import itertools
from collections import defaultdict
import glob
import re

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'


label_pickle_file = 'albert_base_ner_checkpoints/label2id.pkl'
pb_path = 'albert_base_ner_checkpoints/model.ckpt-2649.pb'
vocab_file = 'albert_base_ner_checkpoints/vocab.txt'
max_seq_length = 128
bndp = BertNerDataProcessor(max_seq_length, vocab_file)
bcrf = BERT_CRF(label_pickle_file)
bcrf.load_model(pb_path)

def split_words(words, window_size=120, slide_steps=60):
    word_slice = []
    curi = 0
    word_num = len(words)
    while curi < word_num:
        word_slice.append(" ".join(words[curi:curi+window_size]))
        curi+=slide_steps
    return word_slice

def combine_labels(slice_labels, window_size=120, slide_steps = 60):
    image_label = slice_labels[0]
    # print("==================image_label====================")
    # print(image_label)
    for slice_label in slice_labels[1:]:
        image_label.extend(slice_label[slide_steps:])
        # print("==================image_label====================")
        # print(image_label)
    return image_label


def delete_extra_space(str_):
    """
    删除多余空格和.
    """
    str_ = str_.strip()
    return re.sub('\s+', ' ', str_) 

def get_contents_from_dir(txt_dir):
    txt_paths = glob.glob(os.path.join(txt_dir, '*.txt'))
    contents = []
    for tp in txt_paths:
        lines = open(tp, 'r',encoding='utf8').readlines()
        content = ""
        for l in lines[1:]:
            if len(l.strip())>0:
                content += l
        content = delete_extra_space(content)
        content = content.replace("\n", "|")
        content = content.replace(" ", "_")
        content = content.replace("\t", "_")
        contents.append(content)   
    return txt_paths, contents
        
def extractEntity(wordseq, labelseq, entity_types):
    type_entities_dict = defaultdict(list)
    words_tags = list(zip(wordseq, labelseq))
    for key, group in itertools.groupby(words_tags, key=lambda m: m[1][2:]):
        #print(key, list(group),  key in entity_types)
        if key in entity_types:
            words = [[]]
            tags = [[]]
            for word, tag in list(group):
                #print(word, tag)
                if tag.startswith("B"):
                    if len(words[-1]) == 0:
                        words[-1].append(word)
                        tags[-1].append(tag)
                    else:
                        words.append([word])
                        tags.append([tag])
                elif tag.startswith("I"):
                    words[-1].append(word)
                    tags[-1].append(tag)
                    
#             print(words)
#             print(tags)
            for word_list, tag_list in zip(words, tags):
                entity_ner = ''.join(tag_list)
                entity = ''.join(word_list)
                if entity_ner.startswith('I'):
                    print("I head ocr: ", entity, entity_ner)
                    continue
                type_entities_dict[key].append(entity)
    return type_entities_dict

if __name__ == "__main__":
    
    
    entity_types = ['PRICE', 'SDATE', 'BIDDER', 'TENDER']
    txt_dir = './test_txt'
    txt_paths, contents = get_contents_from_dir(txt_dir)
    
    for path, txt in zip(txt_paths, contents):
        print("============================================")
        print(path)
        print(txt)
        print(list(txt))
        #txt = "中标结果公告：|工程编号:_2021J00000084|建设单位名称:_北京市轨道交通建设管理有限公司|工程名称:_北京市轨道交通3号线一期工程东坝中街站总配外电源工程施工|建设地点:_北京市|中标人:_北京京宇电力安装工程有限公司|中标价(元):_127723567.67|公示开始时间:_20210531135321"
        word_slices = split_words(list(txt))
        tokens, input_ids, input_masks, segment_ids, labels = bndp.process_lines(word_slices)
        pred_labels = bcrf.predict(input_ids, input_masks, segment_ids, tokens)

        image_label = combine_labels(pred_labels, slide_steps = 60)
        #print(image_label)             
        type_entities_dict = extractEntity(list(txt), image_label, entity_types)
#         for  word, label in zip(list(txt), image_label):
#             print(word, label)
        for key, value in type_entities_dict.items():
            print(key, value)