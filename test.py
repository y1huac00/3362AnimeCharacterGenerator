import i2v
from PIL import Image
import os, random
from collections import Counter
import time

def main():
    starttime = time.time()
    illust2vec = i2v.make_i2v_with_chainer(
        "illust2vec_tag_ver200.caffemodel", "tag_list.json")

    # In the case of caffe, please use i2v.make_i2v_with_caffe instead:
    # illust2vec = i2v.make_i2v_with_caffe(
    #     "illust2vec_tag.prototxt", "illust2vec_tag_ver200.caffemodel",
    #     "tag_list.json")

    gen = []
    char = []

    for i in range(100):
        img = Image.open('faces/' + random.choice(os.listdir("./faces/")))
        res = illust2vec.estimate_plausible_tags([img], threshold=0.5)
        for j in res[0]['general']:
            gen.append(j[0])
        for j in res[0]['character']:
            char.append(j[0])
        print(i)

    print(Counter(gen))
    print(Counter(char))
    elapsedtime = time.time() - starttime
    print(elapsedtime)

if __name__ == '__main__':
    main()