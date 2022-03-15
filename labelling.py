import i2v
from PIL import Image
import os, random
import pandas as pd
from tqdm import tqdm

facelist = os.listdir('./faces/')
illust2vec = i2v.make_i2v_with_chainer("illust2vec_tag_ver200.caffemodel", "tag_list.json")
tagdic = pd.read_csv('tagmap.csv')
tagdic = dict(zip(tagdic.tag,tagdic.id))
tagiddic = {i:[[]] for i in range(len(tagdic.keys()))}


def label():
    li_all = []
    for i in tqdm(range(len(facelist))):
        img = Image.open('./faces/' + facelist[i])
        res = illust2vec.estimate_plausible_tags([img], threshold=0.5)
        li_tem = []
        for j in res[0]['general']:
            li_tem.append(tagdic[j[0]])
            tagiddic[tagdic[j[0]]][0].append(facelist[i])
        for j in res[0]['character']:
            li_tem.append(tagdic[j[0]])
            tagiddic[tagdic[j[0]]][0].append(facelist[i])
        li_all.append([facelist[i],li_tem])
        # print(i)
        # if i == 9999:
        #     break

    tagfilename = pd.DataFrame.from_dict(tagiddic,orient='index',columns=['filename'])
    filenametag = pd.DataFrame(li_all, columns=['filename','tagid'])

    tagfilename.to_csv('tag_filename.csv',index=True)
    filenametag.to_csv('filename_tag.csv',index=False)

    return

label()