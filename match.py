from detect_original_text import detect_original_text
import difflib
import pandas as pd
import re
import numpy as np


class match(detect_original_text):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.match_detected_to_real()

    def match_detected_to_real(self):
        detected_text = self.letters_df.groupby('group_right')['letter2write'].sum().str.cat(sep='').strip()
        text_diff = difflib.SequenceMatcher(None, a=self.detected_plain_text, b=detected_text, autojunk=False)
        matching_blocks = text_diff.get_matching_blocks()
        self.letters_df['len'] = self.letters_df['letter2write'].apply(len)
        self.letters_df['n_chars_up_to_here'] = self.letters_df['len'].cumsum()
        self.letters_df['n_chars_up_to_here'] = self.letters_df[['group_right', 'n_chars_up_to_here']].apply(
            lambda x: -1 if pd.isna(x[0]) else int(x[1]), axis=1)
        raw2ind = {}
        for m in matching_blocks:
            for i in range(m.size):
                raw2ind[m.b + i] = m.a + i
        self.letters_df['letter_loc'] = self.letters_df['letter2write'].apply(
            lambda x: [m.start() for m in re.finditer("[א-ת]", x)][0])
        self.letters_df['ind_in_real_text'] = (
                self.letters_df['n_chars_up_to_here'].shift().fillna(0) + self.letters_df['letter_loc']).map(raw2ind)
        self.letters_df['real_letter'] = self.letters_df['ind_in_real_text'].apply(
            lambda x: -1 if pd.isna(x) else self.detected_plain_text[int(x)])
        ind2word = {}
        for i in range(len(self.detected_plain_text)):
            curr_text = self.detected_plain_text[:i]
            ind2word[i] = curr_text.count(' ')
        words = self.detected_text[:]
        self.letters_df['real_word_ind'] = self.letters_df['ind_in_real_text'].map(ind2word)
        self.letters_df['real_word'] = self.letters_df['ind_in_real_text'].apply(
            lambda i: words[ind2word[i]] if i >= 0 else np.nan)
        self.letters_df['is_detected_end_word'] = self.letters_df['letter2write'].apply(lambda s: s
                                                                                        .endswith(' '))
        self.letters_df['detected_word_group'] = self.letters_df['is_detected_end_word'].shift(1).fillna(0).cumsum()
        self.letters_df['is_new_word_in'] = self.letters_df['letter2write'].apply(
            lambda x: 1 if '_' in str(x) and ' ' in str(x) else 0)
        self.letters_df['is_new_word_in'] = self.letters_df['is_new_word_in'].cumsum()
        self.letters_df['detected_word_group'] = self.letters_df['detected_word_group'] + self.letters_df[
            'is_new_word_in']
        # making sure each detected_word_group will be on the same line
        self.letters_df['detected_word_group'] = self.letters_df.groupby(['detected_word_group', 'group_right']).ngroup()
        self.letters_df['detected_word'] = self.letters_df.groupby('detected_word_group')['letter2write'].transform(sum)
        self.letters_df['real_word_ind_shift_down'] = self.letters_df['real_word_ind'].shift()
        self.letters_df['real_word_ind_shift_up'] = self.letters_df['real_word_ind'].shift(-1)
        self.letters_df['detected_word_group_shift_down'] = self.letters_df['detected_word_group'].shift()
        self.letters_df['detected_word_group_shift_up'] = self.letters_df['detected_word_group'].shift(-1)
        self.letters_df['letter2write_shift_down'] = self.letters_df['letter2write'].shift()

        def update_word_group(s):
            if not pd.isna(s['real_word_ind']):
                return s['real_word_ind']
            else:
                if s['detected_word_group_shift_up'] == s['detected_word_group']:
                    return s['real_word_ind_shift_up']
                elif s['detected_word_group_shift_down'] == s['detected_word_group']:
                    return s['real_word_ind_shift_down']
                else:
                    return np.nan

        self.letters_df['real_word_ind_copy'] = self.letters_df['real_word_ind']
        self.letters_df['prev_real_word_ind'] = self.letters_df['real_word_ind']
        for i in range(50):
            self.letters_df['real_word_ind'] = self.letters_df.apply(update_word_group, axis=1)
            if (self.letters_df['prev_real_word_ind'].fillna(-1) == self.letters_df['real_word_ind'].fillna(-1)).all():
                break
            self.letters_df['real_word_ind_shift_down'] = self.letters_df['real_word_ind'].shift()
            self.letters_df['real_word_ind_shift_up'] = self.letters_df['real_word_ind'].shift(-1)
            self.letters_df['prev_real_word_ind'] = self.letters_df['real_word_ind']
        self.letters_df['real_word_ind_curated'] = self.letters_df['real_word_ind']
        self.letters_df['real_word_ind'] = self.letters_df['real_word_ind_copy']
        cols2del = ['real_word_ind_shift_down', 'real_word_ind_shift_up',
                    'letter2write_shift_down', 'detected_word_group_shift_down',
                    'detected_word_group_shift_up', 'prev_real_word_ind', 'real_word_ind_copy']
        self.letters_df['is_one_letter'] = self.letters_df.groupby('detected_word_group')['real_word_ind'].apply(
            lambda x: x.isnull().all() and len(x) == 1)
        self.letters_df['is_additional_word'] = self.letters_df.groupby('detected_word_group')['real_word_ind'].apply(
            lambda x: x.isnull().all() and len(x) > 1)
        for c2d in cols2del:
            del self.letters_df[c2d]
        self.letters_df['is_end_word'] = (self.letters_df['real_word_ind_curated'] != self.letters_df[
            'real_word_ind_curated'].shift(-1)) & ~pd.isna(self.letters_df['real_word_ind_curated'])
