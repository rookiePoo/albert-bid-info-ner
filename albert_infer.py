import pickle
import modeling
import tokenization
import tensorflow as tf
from tensorflow.python.platform import gfile


class InputExample(object):
      def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label

class BertNerDataProcessor:
    def __init__(self, max_seq_len, vocab_file, do_lower_case=False):
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.max_seq_len = max_seq_len

    def process_lines(self, lines):
        examples = self._create_example(lines)
        tokens, input_ids, input_masks, segment_ids, labels = [], [], [], [], []
        for (ex_idx, example) in enumerate(examples):
            #print(ex_idx ,example)
            if ex_idx % 100 == 0:
                print('converting sample %d of %d' % ( ex_idx, len(examples)))
            token, input_id, input_mask, segment_id, label = self._convert_single_example(example, self.max_seq_len, self.tokenizer)
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels.append(label)
            tokens.append(token)
        return tokens, input_ids, input_masks, segment_ids,labels


    def _create_example(self, lines, set_type='pred'):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line)
            #labels = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=texts))
        return examples

    def _convert_single_example(self, example, max_seq_len, tokenizer):
        textlist = example.text.split(' ')
        tokens = []

        for i, word in enumerate(textlist):
            #print(i, word)
            token = tokenizer.tokenize(word)
            tokens.extend(token)

        # only Account for [CLS] with "- 1".
        if len(tokens) >= max_seq_len - 1:
            tokens = tokens[0:(max_seq_len - 2)]

        ntokens = []
        segment_ids = []

        ntokens.append("[CLS]")
        segment_ids.append(0)
        #label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            #label_ids.append(label_map[labels[i]])

        ntokens.append("[SEP]")
        segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        # use zero to padding and you should
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            ntokens.append("**NULL**")
        label_seq = [0] * max_seq_len
        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

#         print("*** Example ***")
#         print("guid: %s" % (example.guid))
#         print("tokens: %s" % " ".join(
#             [tokenization.printable_text(x) for x in tokens]))
#         print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#         print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#         print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
#         print("label_ids: %s" % " ".join([str(x) for x in label_seq]))

        return ntokens, input_ids, input_mask, segment_ids, label_seq


class BERT_CRF():
    def __init__(self, label_pickle_file):
        with open(label_pickle_file, 'rb') as rf:
            label2id = pickle.load(rf)
            self.id2label = {value: key for key, value in label2id.items()}
            print(self.id2label)

    def load_model(self, pbmodel_path):
        self.sess = tf.Session()
        with gfile.FastGFile(pbmodel_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图

        # 需要有一个初始化的过程
        self.sess.run(tf.global_variables_initializer())

        # 输入
        self.input_ids = self.sess.graph.get_tensor_by_name('input_ids:0')
        self.input_mask = self.sess.graph.get_tensor_by_name('input_mask:0')
        self.input_segids = self.sess.graph.get_tensor_by_name('segment_ids:0')

        self.pred_op = self.sess.graph.get_tensor_by_name('loss/ArgMax:0')

    def predict(self, input_ids, input_masks, segment_ids, tokens):

        with self.sess.graph.as_default():
            with self.sess.as_default():
                feed_dict = {self.input_ids: input_ids,
                             self.input_mask: input_masks,
                             self.input_segids: segment_ids}

                result = self.sess.run([self.pred_op], feed_dict=feed_dict)
                # probabilities = tf.nn.softmax(log_probs, axis=-1)
                # predict = tf.argmax(probabilities, axis=-1)

                label_pred = []
                for prediction in result[0]:
                    #print(prediction)
                    #output_line = " ".join(self.id2label[id] for id in prediction if id != 0) + "\n"
                    label_pred.append([self.id2label[labelid] for labelid in prediction if labelid != 0][1:-1])
                    #print(output_line)
        return label_pred
if __name__ == "__main__":
    vocab_file = 'albert_config/vocab.txt'
    max_seq_length = 128

    bndp = BertNerDataProcessor(max_seq_length, vocab_file)
    abert = BERT_CRF()
    abert.load_model('albert_base_ner_checkpoints/model.ckpt-3.pb')

    tokens, input_ids, input_masks, segment_ids, labels = bndp.process_lines(["美 国 的 华 莱 士", "中 国", "钱 学 森 在 清 华 大 学"])

    print(input_ids)
    print(input_masks)
    print(segment_ids)
    res = abert.predict(input_ids, input_masks, segment_ids, tokens)

