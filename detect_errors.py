from match import match
import pandas as pd
import cv2
import numpy as np


class detect_errors(match):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.missing_words_inds = self.detect_missing_words()
        self.missing_letters_inds = self.detect_missing_letters()
        self.additional_detected_words = self.detect_additional_words()
        self.additional_letters = self.detect_additional_letters()
        self.vertical_connected = self.detect_vertical_connected()
        self.dibukim_df, self.dibukim_inds = self.detect_dibukim()

    def detect_missing_words(self):
        missing_words_inds = set(range(len(self.detected_text))) - set(
            self.letters_df['real_word_ind_curated'].unique())
        return missing_words_inds

    def detect_missing_letters(self):
        space_text = ' '.join(self.detected_text)
        ind_space = [i for i, e in enumerate(space_text) if e == ' ']
        missing_letters_inds = set(range(len(''.join(self.detected_text)))) - set(
            self.letters_df['ind_in_real_text'].unique()) - set(
            ind_space)
        return missing_letters_inds

    def detect_additional_words(self):
        relevant_df = self.letters_df[~pd.isna(self.letters_df['group_right'])]
        s = (relevant_df['real_word_ind'].isnull().groupby(relevant_df['detected_word_group']).mean().sort_values(
            ascending=False) == 1)
        return s[s].index.values

    def detect_additional_letters(self):
        additional_inds = self.letters_df[
            pd.isna(self.letters_df['ind_in_real_text']) & ~pd.isna(self.letters_df['group_right'])].index.values
        return additional_inds

    def detect_vertical_connected(self):
        vertical_inds = self.letters_df.loc[pd.isna(self.letters_df['group_right'])].index.values
        return vertical_inds

    def detect_dibukim(self):
        padded = np.pad(self.img, ((1, 1), (1, 1)), mode='constant',
                        constant_values=255)  # I pad the image to avoid cases where letters that are near the border are detected as holes
        h, w = padded.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(padded, mask, (np.where(self.img_copy == 0)[0][0], np.where(self.img_copy == 0)[1][0]),
                      127)
        padded[np.where(padded == 255)] = 254
        padded = padded[1:-1, 1:-1]
        img_copy = np.ones(self.img.shape) * 255
        img_copy[np.where(self.img_copy > 0)] = 0
        img_copy[np.where(padded == 254)] = 0
        filled_df = self.my_connected_components(img_copy, return_stats=True)
        df_copy = self.letters_df[['left', 'right', 'top', 'bottom', 'area', 'letters']]
        df_copy['width'] = df_copy['right'] - df_copy['left']
        df_copy['height'] = df_copy['bottom'] - df_copy['top']
        merged = df_copy.merge(filled_df, on=['left', 'top', 'width', 'height'], how='left')
        merged['hole_area'] = merged['area_y'] - merged['area_x']
        inds = merged.loc[merged['hole_area'] > 0].index.values
        return merged, inds

    def detect_small_spaces_between_words(self):
        pass

    def detect_big_spaces_inside_word(self):
        pass


path = '../stam_old/sfaradi_efrat/7.jpg'
p = detect_errors(path=path, model_path='assets/cnn_model_v3.h5', train_mode=True, source="torah")
