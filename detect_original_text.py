from improve_order import improve_order
import pandas as pd
import numpy as np
from itertools import product
from scipy import sparse
import cv2
import re
from collections import Counter
import nltk
class detect_original_text(improve_order):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.right_left = self.calc_right_left()
        self.TH = self.calc_space_th()

    def calc_right_left(self):
        sparse_im = sparse.csr_matrix(self.img_copy).tocoo()
        df = pd.DataFrame(np.array([sparse_im.col, sparse_im.row, sparse_im.data]).T)
        df.columns = ['x', 'y', 'val']
        left_most = df.groupby(['y', 'val'])['x'].min().reset_index().rename(columns={'x': 'x_left', 'val': 'val_left'})
        right_most = df.groupby(['y', 'val'])['x'].max().reset_index().rename(
            columns={'x': 'x_right', 'val': 'val_right'})
        right_left = left_most.merge(right_most, on=['y'], how='outer')
        print(right_left.shape)
        right_left['space'] = right_left['x_left'] - right_left['x_right']
        right_left = right_left[right_left['space'] > 0]
        print(right_left.shape)
        right_left['min_space'] = right_left.groupby(['val_right', 'val_left'])['space'].transform(min)
        right_left = right_left[right_left['space'] == right_left['min_space']]
        right_left = right_left.drop_duplicates(subset=['val_right', 'val_left', 'space'])
        print(right_left.shape)
        right_left['adj'] = right_left.apply(lambda x: (x['val_right'], x['val_left']), axis=1)
        ind2colors = dict(zip(self.letters_df['index'], self.letters_df['letter_colors']))
        colors2colors = list(
            zip(self.letters_df['index'].map(ind2colors),
                self.letters_df['closest_right_with_problematic'].map(ind2colors)))
        adjes_right = np.vstack([list(product(x[0], x[1])) for x in colors2colors if type(x[1]) is list])
        adjes_right = [tuple(x) for x in adjes_right]
        colors2colors = list(
            zip(self.letters_df['closest_left'].map(ind2colors), self.letters_df['index'].map(ind2colors)))
        adjes_left = np.vstack([list(product(x[0], x[1])) for x in colors2colors if type(x[0]) is list])
        adjes_left = [tuple(x) for x in adjes_left]
        adjes = adjes_right + adjes_left
        adjes = list(set(adjes))
        right_left = right_left[right_left['adj'].isin(adjes)]
        color2ind = np.vstack(
            self.letters_df[['letter_colors', 'index']].apply(lambda x: list(product(x[0], [x[1]])), axis=1))
        d = dict(zip(color2ind[:, 0], color2ind[:, 1]))
        right_left['right_ind'] = right_left['val_left'].map(d)
        right_left['left_ind'] = right_left['val_right'].map(d)
        right_left['min_letter_space'] = right_left.groupby(['right_ind', 'left_ind'])['space'].transform('min')
        right_left = right_left[right_left['space'] == right_left['min_letter_space']]
        right_left = right_left.drop_duplicates(subset=['val_right', 'val_left', 'space'])
        return right_left

    def calc_spaces(self):
        ind2space = dict(zip(self.right_left['right_ind'], self.right_left['space']))
        self.letters_df['space'] = self.letters_df['index'].map(ind2space)

    def calc_space_th(self):
        TH, _ = cv2.threshold(self.letters_df['space'].values.astype('uint8'), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return TH

    def calc_letter2write(self):
        def add_underscore_and_spaces_before_letter(s):
            if pd.isna(s['closest_right_with_problematic']):
                return s['letters']
            if pd.isna(self.letters_df.loc[self.letters_df['index'] == s['closest_right_with_problematic'], \
                                      'group_right'].values[0]):
                curr_color = s['letter_colors']
                right_color = self.letters_df.loc[self.letters_df['index'] == s['closest_right_with_problematic'], \
                                             'letter_colors'].values[0]
                space = self.right_left.loc[self.right_left['adj'].isin(list(product(curr_color, right_color))), 'space'].min()
                if space <= self.TH:
                    res = '_' + s['letters']
                elif space > self.TH:
                    res = '_ ' + s['letters']
                print('yes')
            else:
                res = s['letters']
            if s['space'] > self.TH:
                res = res + ' '
            elif pd.isna(s['closest_left']):
                res = res + ' '
            return res
        self.letters_df['letter2write'] = self.letters_df.apply(add_underscore_and_spaces_before_letter, axis=1)

    def detect_real_text(self):
        # finds the text in Torah that matches to the current image
        # step 1: read Torah text
        with open('text/torah', 'r', encoding="utf8") as f:
            t = f.readlines()
        t = [a for a in t if 'פרק' not in a]
        t = [a for a in t if len(a) >= 30 or len(a) <= 1]
        t = '\n'.join(t)
        t = t.replace(',', ' ')

        t = t.replace('{פ}', ' ')
        t = t.replace('{ס}', ' ')
        t = t.replace('{ר}', ' ')
        t = t.replace('{ש}', ' ')
        t = t.replace(':', ' : ')
        print(t.count("׃"))
        t = re.sub(r'[\s]+[,א-ת]+[\s]+', ' ', t)
        t = re.sub(r'[\s]+[,א-ת]+[\s]+', ' ', t)
        t = re.sub(r'[\s]+[,א-ת]+[\s]+', ' ', t)
        t = re.sub(r'[\s]+[,א-ת]+[\s]+', ' ', t)
        t = re.sub(r'[\s]+', ' ', t)
        t = re.sub(r'[\s]+', ' ', t)
        t = re.sub(r'X', ' ', t)
        t = re.sub(r'[\s]+', ' ', t)
        t = t.strip()
        t = ''.join(re.findall(r'[א-ת\s־]+', t))
        t = t.replace("־", " ")
        t = re.sub(r'[\s]+', ' ', t)
        tt = t.split()
        # find the closest word sequence in the torah to the detected text
        detected_text = self.letters_df.groupby('group_right')['letter2write'].sum().str.cat(sep=' ')
        text_counts = Counter(detected_text.split(' '))
        n_words = len([t for t in detected_text.split(' ') if len(t) > 1])
        c = Counter(tt[:n_words])
        max_val = sum((c & text_counts).values())
        max_i = 0
        for i in range(n_words, len(tt)):
            c[tt[i]] += 1
            if c[tt[i - n_words]] == 1:
                del c[tt[i - n_words]]
            else:
                c[tt[i - n_words]] -= 1
            if sum((c & text_counts).values()) > max_val:
                max_val = sum((c & text_counts).values())
                max_i = i - n_words
        security = n_words // 10
        t_margin = ' '.join(tt[max_i - security:max_i + n_words + security])
        text_split = detected_text.split()
        N = 10
        all_dists = []
        t_margin_split = t_margin.split()
        for t in text_split[:N]:
            curr_dist = [nltk.edit_distance(t, x) for x in t_margin_split[:len(t_margin_split) // 2]]
            min_dist = min(curr_dist)
            curr_inds = [i for i, j in enumerate(curr_dist) if j == min_dist]
            all_dists.append(curr_inds)
        max_agreeing = 0
        df = pd.DataFrame(all_dists).ffill(axis=1)
        start_pos = 0
        flag = False
        for i, a in enumerate(all_dists):
            if flag:
                break
            for aa in a:
                curr_df = df - aa + i
                curr_agreeing = (curr_df.subtract(curr_df.index.values, axis=0) == 0).any(axis=1).sum()
                if curr_agreeing >= max_agreeing:
                    max_agreeing = curr_agreeing
                    start_pos = aa - i
                if max_agreeing > N // 2:
                    flag = True
                    break
        all_dists = []
        for t in text_split[-N:]:
            curr_dist = [nltk.edit_distance(t, x) for x in t_margin_split]
            min_dist = min(curr_dist)
            curr_inds = [i for i, j in enumerate(curr_dist) if j == min_dist]
            all_dists.append(curr_inds)
        max_agreeing = 0
        df = pd.DataFrame(all_dists).ffill(axis=1)
        end_pos = 0
        flag = False
        for i, a in enumerate(all_dists):
            if flag:
                break
            for aa in a:
                curr_df = df - aa + i
                curr_agreeing = (curr_df.subtract(curr_df.index.values, axis=0) == 0).any(axis=1).sum()
                if curr_agreeing >= max_agreeing:
                    max_agreeing = curr_agreeing
                    end_pos = aa - i + N
                if max_agreeing > N // 2:
                    flag = True
                    break
        ret_text = t_margin_split[start_pos:end_pos]
        return ret_text
