from typing import List, Any

from improve_pic import improve_pic
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from collections import Counter
import pandas as pd
from tensorflow.keras.models import load_model


class detect_letters(improve_pic):

    def __init__(self, save_letters_to_s3=False, model_path='assets/cnn_model.pkl',train_mode=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_letters_to_s3 = save_letters_to_s3
        self.model = load_model(model_path)
        self.digit2letter = {1: 'א', 2: 'ב', 3: 'ג', 4: 'ד', 5: 'ה', 6: 'ו', 7: 'ז', 8: 'ח',
                             9: 'ט', 10: 'י', 11: 'כ', 12: 'ל', 13: 'מ', 14: 'נ', 15: 'ס',
                             16: 'ע', 17: 'פ', 18: 'צ', 19: 'ק', 20: 'ר', 21: 'ש', 22: 'ת',
                             23: 'ך', 24: 'ם', 25: 'ן', 26: 'ף', 27: 'ץ'}
        self.to_pass = self.mark_legs_to_pass()
        self.indexes, self.line_number = self.calculate_initial_line_number()
        self.train_mode = train_mode
        if self.train_mode:
            self.letters_df, self.letters2train = self.create_letters_stats()
        else:
            self.letters_df = self.create_letters_stats()

    def pad_letter(self, curr_img, const_val=255):
        curr_height = curr_img.shape[0]
        curr_width = curr_img.shape[1]
        if curr_height > self.to_pad_vertical or curr_width > self.to_pad_horizontal:
            if curr_height > self.to_pad_vertical:
                return 0
            output_stats = cv2.connectedComponentsWithStats(1 - curr_img.astype('uint8') // 255)
            n_colors = len(set(output_stats[1].flatten()) - set([0]))
            if n_colors > 2:
                return 0
            check_img = curr_img.copy()
            if n_colors == 2:
                maybe_leg_color = Counter(output_stats[1].flatten()).most_common(3)[-1][0]
                check_img[np.where(output_stats[1] == maybe_leg_color)] = 255
            h, w = check_img.shape
            if (255 - check_img)[h // 2:h, 0:w // 2].mean() / (
                    255 - check_img).mean() < 1 / 35:  # check that the bottom left square is empty
                to_pad_horizontal = w + self.median_width * 1.5
            else:
                return 0
        to_pad_vert = int((self.to_pad_vertical - curr_height) / 2)
        to_pad_horiz = int((self.to_pad_horizontal - curr_width) / 2)
        padded_img = np.pad(curr_img, ((to_pad_vert, to_pad_vert), (to_pad_horiz, to_pad_horiz)), mode='constant',
                            constant_values=const_val)
        return padded_img

    def decide_if_leg(self, curr_letter_bounds, c_bound, curr_img, relevants, const_val=255):
        if curr_letter_bounds[3] > self.median_height * 2:
            return False
        if c_bound[3] > self.median_height * 2:
            return False
        if (curr_letter_bounds[1] + curr_letter_bounds[1] + curr_letter_bounds[3]) / 2 > (
                c_bound[1] + c_bound[1] + c_bound[3]) / 2:
            return False
        if c_bound[2] > self.median_width:
            return False
        if c_bound[3] > self.median_height * 2:
            return False
        if c_bound[3] <= self.median_height / 10:
            return False
        if c_bound[2] <= self.median_width / 10:
            return False
        if c_bound[2] * 1.5 <= self.median_width and curr_letter_bounds[1] + curr_letter_bounds[3] / 3 <= c_bound[
            1] and curr_letter_bounds[1] + curr_letter_bounds[3] * 5 / 6 >= c_bound[1] and c_bound[1] + c_bound[
            3] >= curr_letter_bounds[1] + curr_letter_bounds[3] * 6 / 7:
            curr_img = ~np.isin(curr_img, relevants) * const_val
            curr_img1 = self.pad_letter(curr_img)
            if curr_img1 is 0:
                curr_img = np.array(Image.fromarray(curr_img).resize((60, 60)))
                if curr_img.mean() == 0:
                    return False
                if (255 - curr_img)[30:60, 0:30].mean() / (255 - curr_img).mean() < 1 / 35:
                    return True
                return False
            curr_img = curr_img1.copy()
            curr_img = np.array(Image.fromarray(curr_img.astype('uint8')).resize((60, 60)))
            predicted = self.model.predict(curr_img.reshape(1, curr_img.shape[0], curr_img.shape[1], 1)).argmax() + 1
            if predicted in [2, 4, 5, 6, 10, 19, 20, 23]:
                return True
            if predicted == 12:
                pass  # TODO: decide what to do when equals to ל
        return False

    def mark_legs_to_pass(self):
        to_pass = []
        letter_bounds = self.img_stats[2]
        for i in range(1, self.n_connected_components):
            if i in to_pass:
                continue
            curr_letter_bounds = letter_bounds[i]
            curr_img = self.img_copy[curr_letter_bounds[1]:curr_letter_bounds[1] + curr_letter_bounds[3], \
                       curr_letter_bounds[0]:curr_letter_bounds[0] + curr_letter_bounds[2]]

            distinct_vals = set(curr_img.flatten())
            relevants = [i]
            if len(distinct_vals) > 2:
                candidates_for_he_or_kuf_leg = distinct_vals - set([0, i])
                for c in candidates_for_he_or_kuf_leg:
                    c_bound = letter_bounds[c]
                    is_leg = self.decide_if_leg(curr_letter_bounds, c_bound, curr_img, relevants)
                    if is_leg:
                        relevants.append(c)
                        to_pass.append(c)
                        break
        return to_pass

    def calculate_initial_line_number(self):
        df = pd.DataFrame(self.img_stats[2], columns=['left', 'top', 'width', 'height', 'area'])
        df['bottom'] = df['top'] + df['height']
        df['right'] = df['left'] + df['width']
        avg_line_top_vec = [df.iloc[1]['top']]
        avg_line_bottom_vec = [df.iloc[1]['bottom']]
        curr_line = [0, 1]
        n_connected_components = self.img_stats[1].max() + 1
        for i in range(2, n_connected_components):
            if i in self.to_pass:
                curr_line.append(curr_line[-1])
                continue
            this_top = df.iloc[i]['top']
            this_bottom = df.iloc[i]['bottom']
            curr_min_top = min(avg_line_top_vec)
            # if (this_top - median_height / 2 <= sum(avg_line_top_vec) / len(avg_line_top_vec)) or (
            # this_bottom - median_height / 2 <= sum(avg_line_bottom_vec) / len(
            # avg_line_bottom_vec)):  # sum/len is the mean
            LAST_N = -5
            if ((np.abs(this_top - sum(avg_line_top_vec[LAST_N:]) / len(
                    avg_line_top_vec[LAST_N:])) <= self.median_height / 3) or (
                        np.abs(this_bottom - sum(avg_line_bottom_vec[LAST_N:]) / len(
                            avg_line_bottom_vec[LAST_N:])) <= self.median_height / 3)) and (
                    this_top - self.median_height <= curr_min_top):  # sum/len is the mean
                avg_line_top_vec.append(this_top)
                avg_line_bottom_vec.append(this_bottom)
                curr_line.append(curr_line[-1])
            else:
                avg_line_top_vec = [this_top]
                avg_line_bottom_vec = [this_bottom]
                curr_line.append(curr_line[-1] + 1)

        df['line_number'] = curr_line

        df.sort_values(by=['line_number', 'right'], ascending=[True, False], inplace=True)
        indexes = df.iloc[1:].index
        return indexes, df['line_number'].values

    def create_letters_stats(self):
        letters2train = []
        all_letters = ''
        probs = []
        lines = []
        all_max_x = []
        all_min_x = []
        all_max_y = []
        all_min_y = []
        all_relevants = []
        is_padded_list = []
        all_areas = []
        for i in self.indexes:
            try:
                # for i in indexes[168:]:
                is_he_or_kuf_flag = False
                if i in self.to_pass:
                    continue
                curr_letter_bounds = self.img_stats[2][i]
                curr_img = self.img_copy[curr_letter_bounds[1]:curr_letter_bounds[1] + curr_letter_bounds[3], \
                           curr_letter_bounds[0]:curr_letter_bounds[0] + curr_letter_bounds[2]]
                distinct_vals = set(curr_img.flatten())
                relevants = [i]
                if len(distinct_vals) > 2:
                    candidates_for_he_or_kuf_leg = distinct_vals - set([0, i])
                    for c in candidates_for_he_or_kuf_leg:
                        c_bound = self.img_stats[2][c]
                        is_leg = self.decide_if_leg(curr_letter_bounds, c_bound, curr_img, relevants)
                        if is_leg:
                            relevants.append(c)
                            # to_pass.append(c)
                            if len(relevants) == 2:
                                min_y_min = min(curr_letter_bounds[1], c_bound[1])
                                max_y_max = max(curr_letter_bounds[1] + curr_letter_bounds[3], c_bound[1] + c_bound[3])
                                min_x_min = min(curr_letter_bounds[0], c_bound[0])
                                max_x_max = max(curr_letter_bounds[0] + curr_letter_bounds[2], c_bound[0] + c_bound[2])
                                curr_img = self.img_copy[min_y_min:max_y_max, min_x_min:max_x_max]
                                is_he_or_kuf_flag = True
                            break
                const_val = 255
                curr_img = ~np.isin(curr_img, relevants) * const_val
                curr_img = self.pad_letter(curr_img, const_val)
                if curr_img is 0:
                    is_padded_list.append(False)
                    curr_img = self.img_copy[curr_letter_bounds[1]:curr_letter_bounds[1] + curr_letter_bounds[3], \
                               curr_letter_bounds[0]:curr_letter_bounds[0] + curr_letter_bounds[2]]
                    curr_img = ~np.isin(curr_img, relevants) * const_val
                else:
                    is_padded_list.append(True)
                curr_img = cv2.resize(curr_img.astype('uint8'), (60, 60))
                if self.train_mode:
                    letters2train.append([curr_img])  # for CNN
                lines.append(self.line_number[i])
                if not is_he_or_kuf_flag:
                    max_x_max = curr_letter_bounds[0] + curr_letter_bounds[2]
                    min_x_min = curr_letter_bounds[0]
                    max_y_max = curr_letter_bounds[1] + curr_letter_bounds[3]
                    min_y_min = curr_letter_bounds[1]
                all_max_x.append(max_x_max)
                all_min_x.append(min_x_min)
                all_max_y.append(max_y_max)
                all_min_y.append(min_y_min)
                all_relevants.append(relevants)
                all_areas.append(curr_letter_bounds[4])
            except Exception as ex:
                all_letters += 'X'
                probs.append(-1)
                lines.append(-1)
                all_max_x.append(-1)
                all_min_x.append(-1)
                all_max_y.append(-1)
                all_min_y.append(-1)
                all_areas.append(-1)
                all_relevants.append([])
                is_padded_list.append(False)
                print(f'ex: {ex}')

        letters = np.stack([letter[0] for letter in letters2train])
        predictions = self.model.predict(letters.reshape(letters.shape[0], letters.shape[1], letters.shape[2], 1))
        all_letters = [self.digit2letter[prediction.argmax() + 1] for prediction in predictions]
        if self.train_mode:
            letters2train = list(zip(letters2train, [prediction.argmax() + 1 for prediction in predictions]))
        probs = [prediction.max() for prediction in predictions]
        letters_df = pd.DataFrame(
            [list(all_letters), probs, lines, all_max_x, all_min_x, all_max_y, all_min_y, all_relevants, is_padded_list,
             all_areas]).T
        letters_df.columns = ['letters', 'probability', 'line', 'right', 'left', 'bottom', 'top', 'letter_colors',
                              'is_padded', 'area']
        letters_df['space'] = letters_df['left'] - letters_df['right'].shift(-1)
        if not self.train_mode:
            return letters_df
        else:
            return letters_df, letters2train

path = '../stam_old/sfaradi_efrat/7.jpg'
p = detect_letters(path=path, model_path='assets/cnn_model_v3.h5', train_mode=True)


