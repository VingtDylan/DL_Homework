import numpy as np
import torch
from torch.autograd import Variable
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from lib.utils import *
from lib.parser import args

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def evaluate(data, model):
    model.eval()
    t_score1 = 0
    t_score2 = 0
    t_score3 = 0
    t_score4 = 0
    t = len(data.test_en)
    with open("result.txt", 'w', encoding='utf-8') as f:
        with torch.no_grad():
            for i in range(len(data.test_en)):
                reference_en = [data.en_index_dict[w] for w in data.test_en_id[i]]
                en_sent = " ".join(reference_en)
                reference_cn = [data.cn_index_dict[w] for w in data.test_cn_id[i]]
                cn_sent = " ".join(reference_cn)
                src = torch.from_numpy(np.array(data.test_en_id[i])).long().to(args.device)
                src = src.unsqueeze(0)
                src_mask = (src != 0).unsqueeze(-2)
                out = greedy_decode(model, src, src_mask, max_len = args.max_length, start_symbol = data.cn_word_dict["BOS"])
                candidate = []
                for j in range(1, out.size(1)):
                    sym = data.cn_index_dict[out[0, j].item()]
                    if sym != 'EOS':
                        candidate.append(sym)
                    else:
                        break
                # print("\n" + en_sent)
                # print("".join(cn_sent))
                # print("MyTranslation: %s" % " ".join(candidate))
                f.write("第{}条句子".format(i + 1))
                f.write("\n")
                f.write("待翻译：" + en_sent)
                f.write("\n")
                f.write("参考翻译：" + cn_sent)
                f.write("\n")
                f.write("候选翻译：" + (" ".join(candidate)))
                f.write("\n")
                # reference_cn = ["BOS", "Going", "to", "play", "basketball", "this", "afternoon" , "?", "EOS"]
                # candidate = ["Going", "to", "play", "basketball", "in", "the", "afternoon" , "?",]
                smoothie = SmoothingFunction().method1
                score1 = round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (1., 0., 0., 0.), smoothing_function=smoothie),4)
                score2 = round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (1./2, 1./2, 0., 0.), smoothing_function=smoothie),4)
                score3 = round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (1./3, 1./3, 1./3, 0.), smoothing_function=smoothie),4)
                score4 = round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (1./4, 1./4, 1./4, 1./4), smoothing_function=smoothie),4)
                # score1 = round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (1., 0., 0., 0.), smoothing_function=smoothie),4)
                # score2 = round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (0., 1., 0., 0.), smoothing_function=smoothie),4)
                # score3 = round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (0., 0., 1., 0.), smoothing_function=smoothie),4)
                # score4 = round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (0., 0., 0., 1.), smoothing_function=smoothie),4)
                # print([reference_cn[1:-1]])
                # print(candidate)
                # print(score1)
                # print(score2)
                # print(score3)
                # print(score4)
                # print(round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (1., 0., 0., 0.), smoothing_function=smoothie),4))
                # print(round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (1./2, 1./2, 0., 0.), smoothing_function=smoothie),4))
                # print(round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (1./3, 1./3, 1./3, 0.), smoothing_function=smoothie),4))
                # print(round(sentence_bleu([reference_cn[1:-1]], candidate, weights = (1./4, 1./4, 1./4, 1./4), smoothing_function=smoothie),4))
                f.write(" bleu1 score : " + str(score1))
                f.write(" bleu2 score : " + str(score2))
                f.write(" bleu3 score : " + str(score3))
                f.write(" bleu4 score : " + str(score4))
                f.write("\n")
                f.write("\n")
                t_score1 += score1
                t_score2 += score2
                t_score3 += score3
                t_score4 += score4
                if (i + 1) % 100 == 0:
                    print("测试样本: {},\t bleu1: {}\t bleu2: {}\t bleu3: {}\t bleu4: {}".format(i + 1, score1, score2, score3, score4))
                    print("\t {} sentences has been translated!".format(i + 1))
    print("Bleu1: {:.3f}".format(t_score1 / t))
    print("Bleu2: {:.3f}".format(t_score2 / t))
    print("Bleu3: {:.3f}".format(t_score3 / t))
    print("Bleu4: {:.3f}".format(t_score4 / t)) 