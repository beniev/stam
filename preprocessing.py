import cv2
import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy.spatial import distance
from collections import Counter
import networkx as nx

pd.options.mode.chained_assignment = None
# from tools import return_graph
import os


def line(p1, p2):
    '''coefficients of a line, based on 2 points'''
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    '''intersection of 2 lines, where each line is defined by 2 points'''
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def return_graph(df, orientation, problematic, delete_problematic=True):
    df1 = df.reset_index()
    g = nx.Graph()
    g = nx.from_pandas_edgelist(df1, 'index', orientation)
    if delete_problematic:
        g.remove_nodes_from(problematic)
    connected_components = nx.connected_components(g)
    node2id = {}
    for cid, component in enumerate(connected_components):
        for node in component:
            node2id[node] = cid
    node2id = {k: v for k, v in node2id.items() if not pd.isna(k)}
    if 'right' in orientation:
        col_name = 'group_right_{}'.format(delete_problematic)
    elif 'left' in orientation:
        col_name = 'group_left_{}'.format(delete_problematic)
    return {col_name: df1['index'].map(node2id)}


class preprocessing:
    def __init__(self, path):
        np.random.seed(42)
        self.path = path
        self.raw_img = cv2.imread(self.path)
        self.binary_img = self.apply_threshold(self.raw_img)
        self.h, self.w = self.get_median_size_based_on_center()
        self.img, self.gray_img, self.small_img = self.transform_perspective()
        self.img = self.apply_threshold(self.img)
        self.img = self.img.astype('uint8')
        self.aligned = cv2.merge([self.img, self.img, self.img])
        self.connectivity = 8
        self.img_stats = cv2.connectedComponentsWithStats(self.img.max() - self.img, self.connectivity, cv2.CV_32S)
        self.median_width = np.median(self.img_stats[2][:, 2][1:])
        self.median_height = np.median(self.img_stats[2][:, 3][1:])
        self.median_area = np.median(self.img_stats[2][:, 4][1:])
        self.n_connected_components = self.img_stats[1].max() + 1
        self.img_copy = self.img_stats[1].copy()


    def my_connected_components(self, img, return_stats=False):
        if img[0][0] != 0:
            if img.dtype == 'bool':
                img = img.astype('uint8')
            img = img.max() - img
        try:
            res = cv2.connectedComponentsWithStats(img, connectivity=8)
        except:
            res = cv2.connectedComponentsWithStats(img.astype('uint8'), connectivity=8)
        if return_stats:
            res = pd.DataFrame(res[2], columns=['left', 'top', 'width', 'height', 'area'])[1:]
        return res
    def rgb2gray(self, img):
        if len(img.shape) == 3 & img.shape[-1] == 3:  # img is RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def apply_threshold(self, img, is_cropped=False):
        '''
        this function applies a threshold on the image,
        the first is Otsu TH on all the image, and afterwards an adaptive TH,
        based on the size of the image.
        I apply a logical OR between all the THs, becasue my assumption is that a letter will always be black,
        while the background can sometimes be black and sometimes white -
        thus I need to apply OR to have the background white.
        '''
        if len(np.unique(img)) == 2:  # img is already binary
            # return img
            gray_img = self.rgb2gray(img)
            _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary_img
        gray_img = self.rgb2gray(img)
        _, binary_img = cv2.threshold(gray_img.astype('uint8'), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        connectivity = 8
        output_stats = cv2.connectedComponentsWithStats(binary_img.max() - binary_img, connectivity, cv2.CV_32S)
        df = pd.DataFrame(output_stats[2], columns=['left', 'top', 'width', 'height', 'area'])[1:]
        if df['area'].max() / df['area'].sum() > 0.1 and is_cropped and False:
            binary_copy = gray_img.copy()
            gray_img_max = gray_img[np.where(output_stats[1] == df['area'].argmax())]
            TH1, _ = cv2.threshold(gray_img_max.astype('uint8'), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # curr_img = binary_copy[np.where(output_stats[1] == df['area'].argmax())]
            binary_copy[np.where((output_stats[1] == df['area'].argmax()) & (gray_img > TH1))] = 255
            binary_copy[np.where((output_stats[1] == df['area'].argmax()) & (gray_img <= TH1))] = 0

            gray_img_not_max = gray_img[np.where(output_stats[1] != df['area'].argmax())]
            TH2, _ = cv2.threshold(gray_img_not_max.astype('uint8'), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_copy[np.where((output_stats[1] != df['area'].argmax()) & (gray_img > TH2))] = 255
            binary_copy[np.where((output_stats[1] != df['area'].argmax()) & (gray_img <= TH2))] = 0
            binary_img = binary_copy.copy()
        # N = [3, 5, 7, 9, 11, 13,27, 45]  # sizes to divide the image shape in
        # N = [20,85]
        N = [3, 5, 25]
        min_dim = min(binary_img.shape)
        for n in N:
            block_size = int(min_dim / n)
            if block_size % 2 == 0:
                block_size += 1  # block_size needs to be odd
            binary_img = binary_img | cv2.adaptiveThreshold(gray_img.astype('uint8'), 255,
                                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                            cv2.THRESH_BINARY, block_size, 10)

        return binary_img

    def get_median_size_based_on_center(self):
        """
        this function assumes that the text is in the middle of the image,
        and it returns the median_width and the median_height
        of the black connected components that are in the center of the image
        """

        height_divide = 3  # should be odd, greater than 1
        width_divide = 3  # should be odd, greater than 1
        height = self.raw_img.shape[0]
        width = self.raw_img.shape[1]
        upper = int(height / height_divide * ((height_divide - 1) / 2))
        lower = int(height / height_divide * ((height_divide + 1) / 2))
        left = int(width / width_divide * ((width_divide - 1) / 2))
        right = int(width / width_divide * ((width_divide + 1) / 2))
        center = self.binary_img[upper:lower, left:right]
        connectivity = 8
        stats = cv2.connectedComponentsWithStats(center.max() - center, connectivity, cv2.CV_32S)
        df = pd.DataFrame(stats[2], columns=['left', 'top', 'width', 'height', 'area'])[1:]
        df['right'] = df['left'] + df['width'] - 1
        df['bottom'] = df['top'] + df['height'] - 1
        top_bound = 0
        left_bound = 0
        bottom_bound = center.shape[0] - 1
        right_bound = center.shape[1] - 1
        # filter potentialy cutted letters
        df = df.loc[
            (df.top > top_bound) & (df.left > left_bound) & (df.bottom < bottom_bound) & (df.right < right_bound)]
        return df.height.median(), df.width.median()

    def filter_small_blacks(self, img, bigs_as_well=False, is_cropped=False):  # filter big blacks as well
        """filter out small connected components"""
        # todo: think if to apply this filter a few times until convergence
        # todo: add area condition
        img = self.apply_threshold(img, is_cropped=is_cropped)
        connectivity = 8
        output_stats = cv2.connectedComponentsWithStats(img.max() - img, connectivity, cv2.CV_32S)
        df = pd.DataFrame(output_stats[2], columns=['left', 'top', 'width', 'height', 'area'])[1:]

        df = df[(df.height > (self.h / 6)) & (
                df.width > (self.w / 10))]  # & (df.width<(median_width*7)) & (df.height<(median_height*7)) ]
        df = df[df['area'] > df['area'].median() / 10]
        if bigs_as_well:
            df = df[(df['height'] < self.h * 4) & (df['width'] < self.w * 7)]
        img = (1 - pd.DataFrame(output_stats[1]).isin(df.index).values.astype(
            'uint8')) * 255  # big connected components (letters) are 0, the rest 255

        # disp just displays the small connected components
        disp1 = (1 - pd.DataFrame(output_stats[1]).isin(
            [0] + df.index.values.tolist()).values) * 255  # small connected components get 255
        disp2 = pd.DataFrame(output_stats[1]).isin([0]).values * 60  # background pixels get 60
        disp = disp1 + disp2  # small connected components get 255, background pixels get 60 , and big connected components get 0.

        return img, disp

    def crop_text_letter_erosion(self):
        binary_img_copy = self.binary_img.copy()
        binary_img = self.filter_small_blacks(self.binary_img)[0]
        ker = np.ones((int(self.h) * 2, int(self.w) * 2), np.uint8)
        # ker=np.ones((int(h),1),np.uint8)
        # filtered_img = extract_relevant_components(binary_img).astype('uint8')
        filtered_img = cv2.erode(self.binary_img, ker, iterations=1)
        connectivity = 8
        output_stats = cv2.connectedComponentsWithStats(filtered_img.max() - filtered_img, connectivity, cv2.CV_32S)
        df = pd.DataFrame(output_stats[2], columns=['left', 'top', 'width', 'height', 'area'])[1:]
        df['right'] = df['left'] + df['width'] - 1
        df['bottom'] = df['top'] + df['height'] - 1
        middle = self.binary_img.shape[1] / 2
        in_middle_inds = df.index[(df['left'] <= middle) & (df['right'] >= middle)].values
        components_img = output_stats[1]
        indexes = np.array(np.where(np.isin(components_img, in_middle_inds))).transpose()
        rc = cv2.boundingRect(indexes)
        cutted_components = binary_img_copy[rc[0]:rc[0] + rc[2], rc[1]:rc[1] + rc[3]]
        return cutted_components.astype('uint8'), self.raw_img[rc[0]:rc[0] + rc[2], rc[1]:rc[1] + rc[3]]

    def return_ransac_line(self, img):
        # apply a very rough estimation to number of lines, and to number of letters per line
        connectivity = 8
        output_filtered = cv2.connectedComponentsWithStats(img.max() - img, connectivity, cv2.CV_32S)
        df_filtered = pd.DataFrame(output_filtered[2], columns=['left', 'top', 'width', 'height', 'area'])[1:]

        df_filtered['right'] = df_filtered['left'] + df_filtered['width'] - 1
        df_filtered['bottom'] = df_filtered['top'] + df_filtered['height'] - 1
        df_filtered.reset_index(inplace=True, drop=True)
        est_n_letters = df_filtered.shape[0]
        est_n_lines = np.round(np.sqrt(est_n_letters / (img.shape[0] / img.shape[1]))).astype('int')
        est_letters_per_line = np.round(est_n_letters / est_n_lines).astype('int')

        # take the est_letters_per_line toppest connected components
        df_filtered['is_top'] = df_filtered.sort_values(by='top').index < est_letters_per_line
        df_top = df_filtered[df_filtered['is_top'] == 1]

        df_top.reset_index(inplace=True, drop=True)
        v = df_top.sort_values(['left', 'right'])
        # if the letters are different the right most coord in the lefter letter should be more small (lefter) than the left most coord in the next letter (in ה for example this won't happen)
        df_top['group'] = (v.right - v.left.shift(-1)).shift().lt(0).cumsum()
        # take from each group the top-most component
        df_top['top_in_group'] = df_top.groupby('group')['top'].transform('min')
        # live just the "toppest" componnents from each letter
        df_top = df_top[df_top['top'] == df_top['top_in_group']]

        # apply RANSAC on the letters average
        df_top['X'] = (df_top.right + df_top.left) / 2
        df_top['Y'] = (df_top.top + df_top.bottom) / 2
        ransac = linear_model.RANSACRegressor()
        ransac.fit(df_top.X.to_frame(), df_top.Y.to_frame())

        reshape = lambda x: np.array([x]).reshape(-1, 1)

        # calculate the line by the 2 farrest points
        y_0 = np.round(ransac.predict(reshape(0))).astype('int')
        y_n = np.round(ransac.predict(reshape(img.shape[1]))).astype('int')
        m = (y_0 - y_n) / (-img.shape[1])
        x = np.array(range(img.shape[1]))
        flag = 1
        n = y_0
        count = 0
        while flag:
            y = np.round(x * m + n).astype('int').flatten()
            xx = x[y >= 0]
            y = y[y >= 0]
            if 0 not in img[y, xx]:
                flag = 0
            else:
                n -= 1
                count += 1
        return ransac, count

    def return_ransac_line2(self, img):
        print('ransac!')
        connectivity = 8
        output_filtered = cv2.connectedComponentsWithStats(img.max() - img, connectivity, cv2.CV_32S)
        df = pd.DataFrame(output_filtered[2], columns=['left', 'top', 'width', 'height', 'area'])[1:]

        df = df[df['area'] > df['area'].mean() // 3]

        df['right'] = df['left'] + df['width'] - 1
        df['bottom'] = df['top'] + df['height'] - 1
        df.reset_index(inplace=True, drop=True)
        df['mean_coord'] = df[['top', 'bottom', 'right', 'left']].apply(
            lambda x: [(x[0] + x[1]) / 2, (x[2] + x[3]) / 2],
            axis=1)
        arr = np.array(df['mean_coord'].sum()).reshape(-1, 2)

        mat = distance.cdist(arr, arr)
        N_closest = 15
        closest_N = mat.argsort(axis=1)[:, 1:1 + N_closest]
        tops = df['top'].values
        max_tops = np.vstack([np.maximum(x, t) for x, t in zip(tops, tops[closest_N])])

        bottoms = df['bottom'].values
        min_bottoms = np.vstack([np.minimum(x, b) for x, b in zip(bottoms, bottoms[closest_N])])

        y_overlap = min_bottoms - max_tops + 1
        y_overlap[y_overlap < 0] = 0
        heights = (df['bottom'] - df['top'] + 1).values
        min_height = np.vstack([np.minimum(x, h) for x, h in zip(heights, heights[closest_N])])

        y_overlap_ratio = y_overlap / min_height

        lefts = df['left'].values
        max_lefts = np.vstack([np.maximum(x, l) for x, l in zip(lefts, lefts[closest_N])])

        rights = df['right'].values
        min_rights = np.vstack([np.minimum(x, r) for x, r in zip(rights, rights[closest_N])])

        x_overlap = min_rights - max_lefts + 1

        median_width = (df['right'] - df['left']).median()
        median_height = (df['bottom'] - df['top']).median()

        close_sides = [c[((o >= median_width * 0.7) & (ora >= 0.6)) |
                         ((xo < 0) & (ora >= 0.6)) |
                         (o >= median_width) |
                         (ora >= 0.7)] for
                       c, o, ora, xo in
                       zip(closest_N, y_overlap, y_overlap_ratio, x_overlap)]

        closest_right = [cs[arr[cs][:, 1] > a[1]] for cs, a in zip(close_sides, arr)]
        df['closest_right'] = [cs[arr[cs][:, 1].argmin()] if len(cs) > 0 else np.nan for cs in closest_right]
        closest_left = [cs[arr[cs][:, 1] < a[1]] for cs, a in zip(close_sides, arr)]
        df['closest_left'] = [cs[arr[cs][:, 1].argmax()] if len(cs) > 0 else np.nan for cs in closest_left]
        problematic_right = set({k for k, v in Counter(df['closest_right']).items() if v > 1})
        problematic_left = set({k for k, v in Counter(df['closest_left']).items() if v > 1})
        problematic = set.union(problematic_right, problematic_left)
        prob_df = df.loc[problematic]
        prob_df['height'] = prob_df['bottom'] - prob_df['top'] + 1
        correct_letters_in_prob = prob_df.loc[
            (prob_df['height'] < median_height * 1.4)].index
        problematic = problematic - set(correct_letters_in_prob)
        right_group = return_graph(df.copy(), 'closest_right', problematic)
        df = df.assign(**right_group)
        group2size = df.groupby('group_right_True').size().to_dict()
        df['group_size'] = df['group_right_True'].map(group2size)
        group2height = df.groupby('group_right_True')['top'].mean().to_dict()
        df['group_height'] = df['group_right_True'].map(group2height)
        N = 8
        df = df[df['group_size'] >= N]
        chosen_line = df[df['group_height'] == df['group_height'].min()]
        chosen_line = chosen_line[(chosen_line['height'] >= chosen_line['height'].quantile(0.1)) & (
                chosen_line['height'] <= chosen_line['height'].quantile(0.9))]
        coords = np.array(chosen_line['mean_coord'].sum()).reshape(-1, 2)
        ransac = linear_model.RANSACRegressor()
        ransac.fit(coords[:, 1].reshape(-1, 1), coords[:, 0])
        reshape = lambda x: np.array([x]).reshape(-1, 1)

        # calculate the line by the 2 farrest points
        y_0 = np.round(ransac.predict(reshape(0))).astype('int')
        y_n = np.round(ransac.predict(reshape(img.shape[1]))).astype('int')
        m = (y_0 - y_n) / (-img.shape[1])
        x = np.array(range(img.shape[1]))
        flag = 1
        n = y_0
        count = 0
        while flag:
            y = np.round(x * m + n).astype('int').flatten()
            xx = x[y >= 0]
            y = y[y >= 0]
            if 0 not in img[y, xx] and (ransac.predict(
                    df.loc[df['top'] == df['top'].min(), ['left', 'right']].head(1).mean(axis=1).values.reshape(1, -1))[
                                            0] - count) - (
                    df['top'].min()) < median_height // 5:
                flag = 0
            else:
                n -= 1
                count += 1
        return ransac, count

    def transform_perspective(self):
        reshape = lambda x: np.array([x]).reshape(-1, 1)
        binary_cropped, gray_cropped = self.crop_text_letter_erosion()
        # img = filter_small_blacks(cropped,is_cropped=True)[0]
        img = self.filter_small_blacks(binary_cropped, is_cropped=True)[0]
        gray_cropped = self.rgb2gray(gray_cropped)

        # Get the "small components" image in order to attach later the tags
        big = cv2.connectedComponentsWithStats(img.max() - img, connectivity=8)
        big_stats = pd.DataFrame(big[2], columns=['left', 'top', 'width', 'height', 'area'])[1:]
        median_area = big_stats['area'].median()
        temp_small_img = ((img == binary_cropped) * 255).astype('uint8')
        temp_small = cv2.connectedComponentsWithStats(temp_small_img.max() - temp_small_img, connectivity=8)
        temp_small_stats = pd.DataFrame(temp_small[2], columns=['left', 'top', 'width', 'height', 'area'])[1:]
        temp_small_stats = temp_small_stats[
            (temp_small_stats['area'] >= median_area // 100) & (temp_small_stats['area'] >= 4)]
        small_img = (~np.isin(temp_small[1], temp_small_stats.index) * 255).astype('uint8')

        # img=apply_threshold(cropped)
        # img=self.filter_img()
        shape0 = img.shape[0]
        shape1 = img.shape[1]
        f = lambda x: np.rot90(x)
        XY = []
        for i in range(4):
            if i % 2 == 0:
                ransac, count = self.return_ransac_line2(img)
            else:
                ransac, count = self.return_ransac_line(img)
            x1 = 0
            x2 = img.shape[1]
            y1 = (ransac.predict(reshape(0)) - count)[0]
            y2 = (ransac.predict(reshape(img.shape[1])) - count)[0]
            if type(y1) is list or type(y1) is np.ndarray:
                y1 = y1[0]
                y2 = y2[0]
            if i == 0:
                curr = ((x1, y1), (x2, y2))

            elif i == 2:  # (shape0-x,shape1-y)
                curr = ((shape1 - x1, shape0 - y1), (shape1 - x2, shape0 - y2))
            elif i == 1:  # (shape0-y,x)
                curr = ((shape1 - y1, x1), (shape1 - y2, x2))
            elif i == 3:  # (y, shape1-x)
                curr = ((y1, shape0 - x1), (y2, shape0 - x2))

            XY.append(curr)
            img = f(img).astype('uint8').copy()

        line1 = line(XY[0][0], XY[0][1])
        line2 = line(XY[1][0], XY[1][1])
        line3 = line(XY[2][0], XY[2][1])
        line4 = line(XY[3][0], XY[3][1])

        inter1 = tuple(reversed(intersection(line1, line2)))
        inter2 = tuple(reversed(intersection(line2, line3)))
        inter3 = tuple(reversed(intersection(line3, line4)))
        inter4 = tuple(reversed(intersection(line4, line1)))

        # by how much to pad the image in each side
        add_up = add_down = add_left = add_right = 0
        if min(inter1[0], inter4[0]) < 0:
            add_up = np.abs(np.floor(min(inter1[0], inter4[0]))).astype(int)
        if np.ceil(max(inter2[0], inter3[0])) > img.shape[0]:
            add_down = np.abs(img.shape[0] - np.ceil(max(inter2[0], inter3[0]))).astype(int)
        if min(inter3[1], inter4[1]) < 0:
            add_left = np.abs(np.floor(min(inter3[1], inter4[1]))).astype(int)
        if np.ceil(max(inter1[1], inter2[1])) > img.shape[1]:
            add_right = np.abs(img.shape[1] - np.ceil(max(inter1[1], inter2[1]))).astype(int)

        inter1 = (inter1[0] + add_up, inter1[1] + add_left)
        inter2 = (inter2[0] + add_up, inter2[1] + add_left)
        inter3 = (inter3[0] + add_up, inter3[1] + add_left)
        inter4 = (inter4[0] + add_up, inter4[1] + add_left)

        new_img = np.pad(img, ((add_up, add_down), (add_left, add_right)), mode='constant', constant_values=255)

        new_cropped = np.pad(gray_cropped, ((add_up, add_down), (add_left, add_right)), mode='constant',
                             constant_values=gray_cropped.max())

        new_small = np.pad(small_img, ((add_up, add_down), (add_left, add_right)), mode='constant',
                           constant_values=255)

        old_pts = np.float32([list(inter4), list(inter1), list(inter2), list(inter3)])

        old_pts = np.flip(old_pts, axis=1)

        shifted_old_pts = np.zeros(old_pts.shape)
        shifted_old_pts[:, 0] = old_pts[:, 0] - old_pts[:, 0].min()
        shifted_old_pts[:, 1] = old_pts[:, 1] - old_pts[:, 1].min()

        bor = cv2.boundingRect(np.float32(shifted_old_pts))  # bounding_rect
        ur = [bor[0], bor[1] + bor[2]]
        br = [bor[0] + bor[3], bor[1] + bor[2]]
        bl = [bor[0] + bor[3], bor[1]]
        ul = [bor[0], bor[1]]

        new_pts = np.float32([ul, ur, br, bl])
        new_pts = np.flip(new_pts, axis=1)

        M = cv2.getPerspectiveTransform(old_pts, new_pts)  # compute the transformation matrix
        img = cv2.warpPerspective(new_img, M, (bor[2], bor[3]))  # apply the transformation matrix on the image
        img = img.astype(int)
        #
        gray_img = cv2.warpPerspective(new_cropped, M, (bor[2], bor[3]))
        gray_img = gray_img.astype(int)

        small_img = cv2.warpPerspective(new_small, M, (bor[2], bor[3]))
        small_img = small_img.astype(int)
        return img, gray_img, small_img



