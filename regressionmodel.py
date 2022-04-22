import pickle
import numpy as np
import collections
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import sys
sys.path.append("stylegan2")

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, ElasticNet
import glob
import gc

def find_direction_binary(dlatents, targets):
    #clf = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, C=0.001).fit(dlatents, targets)
    clf = LogisticRegression(max_iter=1000).fit(dlatents, targets)
    return np.repeat(clf.coef_.reshape(1,512), 10, axis = 0)

def find_direction_continuous(dlatents, targets):
    clf = Lasso(alpha=0.01, fit_intercept=False, max_iter=10000).fit(dlatents, targets)
    if np.abs(np.sum(clf.coef_)) == 0.0:
        clf = Lasso(alpha=0.001, fit_intercept=False, max_iter=10000).fit(dlatents, targets)
    return  np.repeat(clf.coef_.reshape(1,512), 10, axis = 0)

def find_distrib_continuous(dlatents, targets):
    tag_mean = np.sum(dlatents * targets.reshape(-1, 1), axis = 0) / np.sum(targets)
    return  np.repeat(tag_mean.reshape(1,512), 10, axis = 0)

def prepare():
    image_tags = []
    latent_list = []
    dlatent_list = []
    tags_exist = set(map(lambda x: x.split("_")[-1], glob.glob("image_tags_*")))
    j = 0
    for i in ["2000",'4000','6000','8000','10000','12000','14000','16000','18000','20000','22000','24000','26000','28000','30000']:
        # if i == '65_dis.pkl' or i == '65.pkl' or i == '':
        #     print(i)
        #     continue
        # if j == 15:
        #     print('break')
        #     break
        with open("image_tags_" + i + '.pkl', 'rb') as f:
            image_tags_tmp = pickle.load(f)

        with open("dlatents_for_tagging_" + i + '.pkl', 'rb') as f:
            latent_list_tmp, dlatent_list_tmp, _ = pickle.load(f)

        image_tags.extend(image_tags_tmp)
        latent_list.extend(latent_list_tmp)
        dlatent_list.extend(list(np.array(dlatent_list_tmp).reshape(-1, 10, 512)[:, 0, :]))
        j += 1
        # break

    gc.collect()
    dlatents_for_regression = np.array(dlatent_list).reshape(-1, 512)
    return image_tags, latent_list, dlatent_list, dlatents_for_regression

def prepare2(image_tags):
    all_tags = collections.defaultdict(int)
    for tags in image_tags:
        for tag in tags:
            all_tags[tag] += 1
    tags_by_popularity = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)
    eye_tags = list(filter(lambda x: x[0].endswith("_eyes"), tags_by_popularity))
    hair_tags = list(filter(lambda x: x[0].endswith("_hair"), tags_by_popularity))
    eye_hair_tags = list(filter(lambda x: x[0].endswith("_hair") or x[0].endswith("_eyes") or x[0] in ['open_mouth', 'close_mouth'], tags_by_popularity))
    tag_binary_feats = {}
    tag_continuous_feats = {}
    for tag, _ in tags_by_popularity:
        this_tag_feats = []
        this_tag_feats_cont = []
        for tag_list_for_dl in image_tags:
            this_dl_tag_value = 0.0
            this_dl_tag_value_cont = 0.0
            if tag in tag_list_for_dl:
                this_dl_tag_value = 1.0
                this_dl_tag_value_cont = tag_list_for_dl[tag]
            this_tag_feats.append(this_dl_tag_value)
            this_tag_feats_cont.append(this_dl_tag_value_cont)
        tag_binary_feats[tag] = np.array(this_tag_feats)
        tag_continuous_feats[tag] = np.array(this_tag_feats_cont)
        popular_tags = list(filter(lambda x: x[1] > 0, tags_by_popularity))
        good_tags = popular_tags
    return tags_by_popularity, eye_tags, hair_tags, tag_binary_feats, tag_continuous_feats, good_tags, eye_hair_tags

def cal_direction(good_tags,dlatents_for_regression, tag_binary_feats, tag_continuous_feats, method, samplesize):
    tag_directions = {}
    for i, (tag, _) in enumerate(good_tags):
        print("Estimating direction for", tag, "(", i, ")")
        if method == 'logistic':
        # Variant A: Binary labels, logistic regression
            tag_directions[tag] = find_direction_binary(dlatents_for_regression, tag_binary_feats[tag])
        elif method == 'lasso':
        # Variant B: Continuous labels (confidence from deepdanbooru), Lasso regression
            tag_directions[tag] = find_direction_continuous(dlatents_for_regression, tag_continuous_feats[tag])
        elif method == 'mean':
        # Variant C: means and move to mean
            tag_directions[tag] = find_distrib_continuous(dlatents_for_regression, tag_continuous_feats[tag])
    with open(f"tagged_dlatents/tag_dirs_cont-{method}-{samplesize}.pkl", 'wb') as f:
        pickle.dump(tag_directions, f)

def tags_use():
    tags = ['blush', 'closed_mouth', 'opened_mouth', 'smile', 'orange_eyes', 'red_eyes', 'grey_hair', 'brown_hair',
            'black_hair', 'yellow_hair', 'aqua_eyes', 'pink_hair', 'unhappy', 'orange_hair', 'yellow_eyes',
            'silver_hair', 'red_hair', 'grey_eyes', 'purple_hair', 'brown_eyes', 'blue_hair', 'purple_eyes',
            'blue_eyes', 'pink_eyes', 'green_hair', 'green_eyes', 'open_mouth', 'close_mouth','blush', 'smile']

    for i in ['logistic', 'lasso', 'mean']:
        for j in ['100','2000','8000','30000','80000']:
            if i == 'logistic' and j == '80000':
                continue
            tag_directions = None
            with open(f"tagged_dlatents/tag_dirs_cont-{i}-{j}.pkl", 'rb') as f:
                tag_directions = pickle.load(f)
            tags_use = list(tag_directions.keys())
            tags_use = list(filter(lambda x: x in tags, tags_use))
            with open(f"tagged_dlatents/tags_use-{i}-{j}.pkl", 'wb') as f:
                pickle.dump(tags_use, f)


def heatmap(l):
    heat = {}
    for i in l:
        for j in i:
            if j not in list(heat.keys()):
                heat[j] = {}
            for otherkeys in i:
                # if otherkeys == j:
                #     continue
                if otherkeys not in list(heat[j].keys()):
                    heat[j][otherkeys] = 1
                else:
                    heat[j][otherkeys] += 1

    print(heat)
    df = pd.DataFrame(heat)
    df = df.fillna(0)
    df = df.sort_index(axis=0)
    df = df.sort_index(axis=1)
    print(df)
    figure(figsize=(12, 10))
    sns.heatmap(df, cmap="GnBu")
    plt.show()

if __name__ == '__main__':
    samplesize = 30000
    method = ['logistic', 'lasso', 'mean'][0]
    image_tags, latent_list, dlatent_list, dlatents_for_regression = prepare()
    tags_by_popularity, eye_tags, hair_tags, tag_binary_feats, tag_continuous_feats, good_tags, eye_hair_tags = prepare2(image_tags)
    print(eye_hair_tags)
    # print(image_tags)
    # heatmap(image_tags)
    # cal_direction(eye_hair_tags, dlatents_for_regression, tag_binary_feats, tag_continuous_feats, method, samplesize)

    # tags_use()

    # for i in ['logistic', 'mean']:
    #     with open(f"tagged_dlatents/tag_dirs_cont-{i}-76.pkl", 'rb') as f:
    #         tag_directions = pickle.load(f)
    #     tags_use = list(tag_directions.keys())
    #     # tags_use = list(filter(lambda x: x in tags, tags_use))
    #     with open(f"tagged_dlatents/tags_use-{i}-76.pkl", 'wb') as f:
    #         pickle.dump(tags_use, f)
    # tags = ['blush', 'closed_mouth', 'opened_mouth', 'smile', 'orange_eyes', 'red_eyes', 'grey_hair', 'brown_hair',
    #         'black_hair', 'yellow_hair', 'aqua_eyes', 'pink_hair', 'unhappy', 'orange_hair', 'yellow_eyes',
    #         'silver_hair', 'red_hair', 'grey_eyes', 'purple_hair', 'brown_eyes', 'blue_hair', 'purple_eyes',
    #         'blue_eyes', 'pink_eyes', 'green_hair', 'green_eyes', 'open_mouth', 'close_mouth', 'blush', 'smile']
    #
    # with open(f"tagged_dlatents/tag_dirs_cont-lasso-2000.pkl", 'rb') as f:
    #     tag_directions = pickle.load(f)
    # tags_use = list(tag_directions.keys())
    # tags_use = list(filter(lambda x: x in tags, tags_use))
    # with open(f"tagged_dlatents/tags_use-lasso-2000.pkl", 'wb') as f:
    #     pickle.dump(tags_use, f)

