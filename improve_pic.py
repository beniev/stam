from preprocessing import preprocessing
import cv2
import pandas as pd
import numpy as np
from skimage.morphology import skeletonize
from scipy import sparse


class improve_pic(preprocessing):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img2 = self.img.copy()
        self.find_connected_tags()  # updates img to include also tags
        self.img = self.apply_threshold(self.img)
        self.img = self.img.astype('uint8')
        self.aligned = cv2.merge([self.img, self.img, self.img])
        self.connectivity = 8
        self.img_stats = cv2.connectedComponentsWithStats(self.img.max() - self.img, self.connectivity, cv2.CV_32S)
        self.median_width = np.median(self.img_stats[2][:, 2][1:])
        self.median_height = np.median(self.img_stats[2][:, 3][1:])
        self.median_area = np.median(self.img_stats[2][:, 4][1:])
        self.to_pad_vertical = self.median_height * 2.5
        self.to_pad_horizontal = self.median_width * 2.5
        self.n_connected_components = self.img_stats[1].max() + 1
        self.img_copy = self.img_stats[1].copy()

    def find_connected_tags(self):
        big = cv2.connectedComponentsWithStats(self.img.max() - self.img, connectivity=8)
        big_stats = pd.DataFrame(big[2], columns=['left', 'top', 'width', 'height', 'area'])[1:]

        # simple TH
        self.small_img[self.small_img <= 127] = 0
        self.small_img[self.small_img > 127] = 255
        small_img = self.small_img.astype('uint8')

        small = cv2.connectedComponentsWithStats(small_img.max() - small_img, connectivity=8)
        small_stats = pd.DataFrame(small[2], columns=['left', 'top', 'width', 'height', 'area'])[1:]
        small_stats['right'] = small_stats['left'] + small_stats['width']
        small_stats['bottom'] = small_stats['top'] + small_stats['height']

        merged_im = self.img & small_img
        merged_analysis = cv2.connectedComponentsWithStats(merged_im.max() - merged_im, connectivity=8)
        merged_stats = pd.DataFrame(merged_analysis[2], columns=['left', 'top', 'width', 'height', 'area'])[1:]
        # Create "is_big" column for merged_stats df to know if it is from big or from small
        bs = big_stats[['top', 'left']]
        bs['is_big'] = True
        is_big = merged_stats.merge(bs, on=['top', 'left'], how='left')['is_big']
        is_big = is_big.fillna(False)
        merged_stats.reset_index(inplace=True)
        merged_stats.index = merged_stats['index']
        is_big.index = merged_stats.index
        merged_stats['is_big'] = is_big
        # init merged "colors" column with np.nan
        merged_stats['colors'] = np.nan

        merged_stats['is_detached_tag'] = 1
        merged_stats.loc[~merged_stats['is_big'], 'is_detached_tag'] = 2
        # create DataFrame from the images using sparse library (as most pixels are 0)
        sparse_big = sparse.csr_matrix(big[1]).tocoo()
        df_big = pd.DataFrame(np.array([sparse_big.col, sparse_big.row, sparse_big.data]).T)
        df_big.columns = ['x', 'y_big', 'val_big']

        sparse_small = sparse.csr_matrix(small[1]).tocoo()
        df_small = pd.DataFrame(np.array([sparse_small.col, sparse_small.row, sparse_small.data]).T)
        df_small.columns = ['x', 'y_small', 'val_small']

        top_big = df_big.groupby(['val_big', 'x'])['y_big'].min().reset_index()
        bottom_small = df_small.groupby(['val_small', 'x'])['y_small'].max().reset_index()
        merged = top_big.merge(bottom_small, on=['x'], how='inner')

        merged['y_gap'] = merged['y_big'] - merged['y_small']
        merged = merged[(merged['y_gap'] >= 0) & (merged['y_gap'] <= self.median_height / 3)]

        sparse_original = sparse.csr_matrix(self.img_copy).tocoo()
        df_original = pd.DataFrame(np.array([sparse_original.col, sparse_original.row, sparse_original.data]).T)
        df_original.columns = ['x', 'y', 'val_original']

        for big_color in merged['val_big'].unique():
            curr_merged_colors_big = merged_stats.merge(big_stats.loc[[big_color]], on=['left', 'top'], how='left')
            curr_merged_colors_big = curr_merged_colors_big[
                ~pd.isna(curr_merged_colors_big['width_y'])].index.values.tolist()
            small_colors = merged.loc[merged['val_big'] == big_color, 'val_small'].unique()
            curr_small_stats = small_stats.loc[small_colors]
            curr_merged_colors_small = merged_stats.merge(small_stats.loc[small_colors], on=['left', 'top'], how='left')
            curr_merged_colors_small = curr_merged_colors_small[
                ~pd.isna(curr_merged_colors_small['width_y'])].index.values.tolist()
            curr_merged_colors = curr_merged_colors_big + curr_merged_colors_small
            merged_stats.loc[curr_merged_colors, 'colors'] = pd.Series([curr_merged_colors] * len(curr_merged_colors),
                                                                       index=curr_merged_colors)
            small_left = curr_small_stats['left'].min()
            big_left = big_stats.loc[big_color, 'left']
            small_right = curr_small_stats['right'].max()
            big_right = big_left + big_stats.loc[big_color, 'width']
            small_top = small_stats.loc[np.isin(small_stats.index, small_colors), 'top'].min()
            big_top = big_stats.loc[big_color, 'top']
            big_bottom = big_top + big_stats.loc[big_color, 'height']
            small_bottom = curr_small_stats['top'].max()
            curr_left = min(small_left, big_left)
            curr_right = max(small_right, big_right)
            curr_top = min(small_top, big_top)
            curr_bottom = max(big_bottom, small_bottom)
            curr_gray = self.gray_img[curr_top:curr_bottom, curr_left:curr_right]
            curr_merged_bin = merged_im[curr_top:curr_bottom, curr_left:curr_right]
            curr_big = big[1][curr_top:curr_bottom, curr_left:curr_right]
            curr_small = small[1][curr_top:curr_bottom, curr_left:curr_right]
            curr_mean_val = np.round(curr_gray[curr_merged_bin != 0].mean())
            curr_gray[curr_merged_bin == 0] = curr_mean_val
            _, binary_img = cv2.threshold(curr_gray.astype('uint8'), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_img[curr_merged_bin == 0] = 0

            temp = binary_img[0][0]
            binary_img[0][0] = binary_img.max()
            stats = cv2.connectedComponentsWithStats(binary_img.max() - binary_img, connectivity=8)
            binary_img[0][0] = temp
            curr_df = pd.DataFrame(stats[2], columns=['left', 'top', 'width', 'height', 'area'])
            curr_df = curr_df.iloc[1:, :]
            curr_ind = curr_df[curr_df['area'] == curr_df['area'].max()].index[0]
            im = ~np.isin(stats[1], curr_ind)
            im[0][0] = temp

            curr_merged = top_big.loc[top_big['val_big'] == big_color]
            curr_merged.loc[:, 'x'] -= curr_left
            curr_merged.loc[:, 'y_big'] -= curr_top

            skeleton = skeletonize(1 - im)
            patches = skeleton & curr_merged_bin
            patches[0][0] = 0
            patches_stats = cv2.connectedComponentsWithStats(patches)
            df_patches = pd.DataFrame(patches_stats[2], columns=['left', 'top', 'width', 'height', 'area'])[1:]

            small_patches_stats = cv2.connectedComponentsWithStats((curr_small > 0) | (patches))[2]
            df_small_patches = pd.DataFrame(small_patches_stats, columns=['left', 'top', 'width', 'height', 'area'])[1:]

            big_patches_stats = cv2.connectedComponentsWithStats((curr_big > 0) | (patches))[2]
            df_big_patches = pd.DataFrame(big_patches_stats, columns=['left', 'top', 'width', 'height', 'area'])[1:]

            patches_that_are_not_connected_to_big = df_patches[
                pd.concat([df_big_patches, df_patches]).duplicated().tail(len(df_patches))]
            patches_that_are_not_connected_to_small = df_patches[
                pd.concat([df_small_patches, df_patches]).duplicated().tail(len(df_patches))]
            not_connected_patches = pd.concat(
                [patches_that_are_not_connected_to_big, patches_that_are_not_connected_to_small]).drop_duplicates()
            connected_patches = df_patches[
                ~pd.concat([not_connected_patches, df_patches]).duplicated().tail(len(df_patches))]
            final_patches_im = 1 - (
                    np.isin(patches_stats[1], connected_patches.index) | (curr_big > 0) | (curr_small > 0))
            curr_stats = self.my_connected_components(final_patches_im)
            final_patches_im = (curr_stats[1] != 1) * 255
            patched = self.my_connected_components(final_patches_im)
            sparse_patched = sparse.csr_matrix(patched[1]).tocoo()
            df_patched = pd.DataFrame(np.array([sparse_patched.col, sparse_patched.row, sparse_patched.data]).T)
            df_patched.columns = ['x', 'y', 'val_patched']
            df_patched['x'] = df_patched['x'] + curr_left
            df_patched['y'] = df_patched['y'] + curr_top
            merged_patched_original = df_patched.merge(df_original, on=['x', 'y'], how='left')
            merged_patched_original['val_original'] = merged_patched_original['val_original'].fillna(0)
            if merged_patched_original['val_original'].nunique() == 2 and 0 in merged_patched_original[
                'val_original'].values:
                curr_inds = np.where(np.isin(self.img_copy[curr_top:curr_bottom, curr_left:curr_right],
                                             merged_patched_original['val_original'].unique()))
                self.img[curr_top: curr_bottom, curr_left: curr_right][curr_inds] = final_patches_im[curr_inds]

path = '../stam_old/sfaradi_efrat/7.jpg'
p = improve_pic(path=path)#, model_path='assets/cnn_model_v3.h5', train_mode=True)