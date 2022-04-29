from detect_errors import detect_errors
import numpy as np
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
from bidi.algorithm import get_display


class visualize_insights(detect_errors):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fontpath = "fonts/Shlomo.ttf"
        self.aligned_with_corrected_words = self.aligned.copy()
        self.already_colored = set()
        # self.color_problematic()
        self.draw_problematic_words()

    def color_inds(self, inds, color):
        c = self.letters_df.loc[inds, 'letter_colors']  # type: pd.Series
        if not c.empty:
            colors = set(np.hstack(c))
            relevant_colors = colors - self.already_colored
            self.already_colored = set.union(self.already_colored, colors)
            self.aligned[np.where(np.isin(self.img_copy, list(relevant_colors)))] = color
        else:
            return

    def color_missing_words(self):
        pass

    def color_missing_letters(self):
        pass

    def color_additional_words(self):
        pass

    def color_additional_letters(self):
        pass

    def color_vertical_connected(self):
        pass

    def color_dibukim(self):
        pass

    def color_small_spaces_between_words(self):
        pass

    def color_big_spaces_inside_word(self):
        pass

    def color_problematic(self):
        # self.color_inds(self.missing_words_inds, [0, 0, 255])
        # self.color_inds(self.missing_letters_inds, [255, 0, 0])
        # addditional_words_inds = self.letters_df.loc[
        #     np.isin(self.letters_df['detected_word_group'], self.additional_detected_words)].index.values
        # self.color_inds(addditional_words_inds, [0, 255, 0])
        self.color_missing_words()
        self.color_missing_letters()
        self.color_additional_words()
        self.color_additional_letters()
        self.color_vertical_connected()
        self.color_dibukim()
        self.color_small_spaces_between_words()
        self.color_big_spaces_inside_word()

    def draw_problematic_words(self):
        median_bottom = self.letters_df.groupby('line')['bottom'].apply(lambda x: np.median(x))
        median_top = self.letters_df.groupby('line')['top'].apply(lambda x: np.median(x))
        median_line_space = median_top.shift(-1) - median_bottom
        median_line_space = median_line_space[median_line_space > 0].median()
        size = int(((self.median_height * self.median_width) ** 0.5) // 3)

        font = ImageFont.truetype(self.fontpath, size)
        self.letters_df['detected_word_ns'] = self.letters_df['detected_word'].str.strip(
            ' ')  # ns - without spaces and underscores
        bad_df = self.letters_df[self.letters_df['real_word'] != self.letters_df['detected_word_ns']]
        bad_df['min_left'] = bad_df.groupby('detected_word_group')['left'].transform(min)
        # bad_df['adjusted_min_left'] = bad_df['min_left'] - bad_df[
        #     'missing_letters_in_end'] * 1.1 * self.median_width
        bad_df['max_right'] = bad_df.groupby('detected_word_group')['right'].transform(max)
        # bad_df['adjusted_max_right'] = bad_df['max_right'] + bad_df[
        #     'mising_letters_in_start'] * 1.1 * self.median_width
        bad_df['word_center'] = (bad_df['max_right'] + bad_df['min_left']) // 2
        bad_df['word_top'] = bad_df.groupby('detected_word_group')['top'].transform(lambda x: np.median(x))
        bad_df = bad_df[~pd.isna(bad_df['real_word'])]
        old = bad_df.drop_duplicates(subset=['detected_word_group'])[
            ['detected_word_group', 'real_word', 'word_center', 'word_top', 'detected_word_ns']].reset_index(drop=True)

        for i in range(len(old)):
            curr_row = old.iloc[i]
            if pd.isna(curr_row['real_word']):
                continue
            curr_word = "{0} ({1})".format(curr_row['real_word'], curr_row['detected_word_ns'])
            curr_loc = (int(curr_row['word_center']), int(curr_row['word_top'] - median_line_space / 2))
            img_pil = Image.fromarray(self.aligned_with_corrected_words.astype('uint8'))
            draw = ImageDraw.Draw(img_pil)
            draw.text(curr_loc,
                      get_display(curr_word), font=font, fill=(0, 255, 0, 255))
            self.aligned_with_corrected_words = np.array(img_pil)
            # mark the wrong letters in orange
            null_inds = self.letters_df.loc[pd.isna(self.letters_df['real_word'])].index
            c = self.letters_df.loc[null_inds, 'letter_colors']  # type: pd.Series
            if not c.empty:
                colors = set(np.hstack(c))
                self.already_colored = set.union(self.already_colored, colors)
                self.aligned_with_corrected_words[np.where(np.isin(self.img_copy, list(colors)))] = [0,165,255]


#path = '../stam_old/sfaradi_efrat/7.jpg'
path = '../stam_old/sfaradi_efrat/2.jpg'
p = visualize_insights(path=path, model_path='assets/cnn_model_v3.h5', train_mode=True, source="torah")
import cv2

cv2.imwrite('../stam_output/aligned_16_im2.jpg', p.aligned_with_corrected_words)
