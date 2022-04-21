from detect_letters import detect_letters
import numpy as np
from scipy.spatial import distance
from collections import Counter


class improve_order(detect_letters):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.order_lines_simple()

    def order_lines_simple(self):
        self.letters_df['mean_coord'] = self.letters_df[['top', 'bottom', 'right', 'left']].apply(
            lambda x: [(x[0] + x[1]) / 2, (x[2] + x[3]) / 2],
            axis=1)
        arr = np.array(self.letters_df['mean_coord'].sum()).reshape(-1, 2)
        mat = distance.cdist(arr, arr)
        N_closest = 15
        closest_N = mat.argsort(axis=1)[:, 1:1 + N_closest]

        tops = self.letters_df['top'].values
        max_tops = np.vstack([np.maximum(x, t) for x, t in zip(tops, tops[closest_N])])

        bottoms = self.letters_df['bottom'].values
        min_bottoms = np.vstack([np.minimum(x, b) for x, b in zip(bottoms, bottoms[closest_N])])

        y_overlap = min_bottoms - max_tops + 1
        y_overlap[y_overlap < 0] = 0

        heights = (self.letters_df['bottom'] - self.letters_df['top'] + 1).values
        min_height = np.vstack([np.minimum(x, h) for x, h in zip(heights, heights[closest_N])])

        y_overlap_ratio = y_overlap / min_height

        neighbors_residual_heights = heights[closest_N] - min_height * y_overlap_ratio
        curr_residual_heights = heights[:, None] - min_height * y_overlap_ratio
        min_residual_heights = np.minimum(neighbors_residual_heights, curr_residual_heights)
        max_residual_heights = np.maximum(neighbors_residual_heights, curr_residual_heights)
        mean_residual_heights = np.mean([min_residual_heights, max_residual_heights], axis=0)

        lefts = self.letters_df['left'].values
        max_lefts = np.vstack([np.maximum(x, l) for x, l in zip(lefts, lefts[closest_N])])

        rights = self.letters_df['right'].values
        min_rights = np.vstack([np.minimum(x, r) for x, r in zip(rights, rights[closest_N])])

        x_overlap = min_rights - max_lefts + 1

        close_sides = [c[((o >= self.median_width * 0.7) & (ora >= 0.6)) |
                         ((xo < 0) & (ora >= 0.6)) |
                         (o >= self.median_width) |
                         (ora >= 0.7) |
                         ((xo < 0) & (ora >= 0.45) & (min_rh < self.median_height * 0.5)) |
                         ((xo < 0) & (ora >= 0.45) & (max_rh < self.median_height * 0.7)) |
                         ((xo < self.median_width // 5) & (ora >= 0.5) & (max_rh < self.median_height * 0.6))]
                       for
                       c, o, ora, xo, min_rh, max_rh in
                       zip(closest_N, y_overlap, y_overlap_ratio, x_overlap, min_residual_heights,
                           max_residual_heights)]

        closest_right = [cs[arr[cs][:, 1] > a[1]] for cs, a in zip(close_sides, arr)]
        self.letters_df['closest_right'] = [cs[arr[cs][:, 1].argmin()] if len(cs) > 0 else np.nan for cs in closest_right]
        closest_left = [cs[arr[cs][:, 1] < a[1]] for cs, a in zip(close_sides, arr)]
        self.letters_df['closest_left'] = [cs[arr[cs][:, 1].argmax()] if len(cs) > 0 else np.nan for cs in closest_left]

        problematic_right = set({k for k, v in Counter(self.letters_df['closest_right']).items() if v > 1})
        problematic_left = set({k for k, v in Counter(self.letters_df['closest_left']).items() if v > 1})
        problematic = set.union(problematic_right, problematic_left)
        maybe_not_problematic = set.union(problematic_right - problematic_left, problematic_left - problematic_right)
        not_problematic = set()
        if maybe_not_problematic:
            median_area = self.letters_df['area'].median()
        for mnp in maybe_not_problematic:
            curr_line = self.letters_df.loc[mnp]
            if curr_line['probability'] < 0.9:
                continue
            if curr_line['bottom'] - curr_line['top'] > self.median_height * 2.1:
                continue

            side_str = 'closest_right'
            other_side_str = 'closest_left'
            if mnp in problematic_left:
                side_str = 'closest_left'
                other_side_str = 'closest_right'
            curr_df = self.letters_df[self.letters_df[side_str] == mnp]
            if curr_df[other_side_str].nunique() == 1:
                not_problematic = set.union(not_problematic, set([mnp]))
                continue
            if "ל" in curr_df['letters'].values.tolist():
                lamed_line = curr_df[curr_df["letters"] == "ל"]
                lamed_line = lamed_line[lamed_line["bottom"] == lamed_line["bottom"].max()]
                if len(lamed_line) > 1:
                    lamed_line = lamed_line.iloc[0]
                l = lamed_line.index[0]
                if curr_residual_heights[l][np.where(closest_N[l] == mnp)[0][0]] > self.median_height * 1.2:
                    if (lamed_line['bottom'] > curr_line['bottom']).values[0]:
                        not_problematic = set.union(not_problematic, set([mnp]))
                        continue

            if curr_line['bottom'] - curr_line['top'] > self.median_height * 1.6:
                if not (curr_line['letters'] in list("ךלןץק") and curr_line['probability'] > 0.98):
                    continue

            # if (curr_df['area'] > median_area // 6).all():
            if curr_df['bottom'].max() - curr_df['top'].min() > self.median_height * 1.7:
                if (curr_df['area'] > median_area // 6).all():
                    continue
            not_problematic = set.union(not_problematic, set([mnp]))
        problematic = problematic - not_problematic
        prob_df = self.letters_df.loc[problematic]
        prob_df['height'] = prob_df['bottom'] - prob_df['top'] + 1
        correct_letters_in_prob = prob_df.loc[
            (prob_df['height'] < self.median_height * 1.4) & (prob_df['probability'] > 0.96)].index
        problematic = problematic - set(correct_letters_in_prob)

        right_group = self.return_graph(self.letters_df.copy(), 'closest_right',problematic)
        self.letters_df = self.letters_df.assign(**right_group)
        left_group = self.return_graph(self.letters_df.copy(), 'closest_left',problematic)
        self.letters_df = self.letters_df.assign(**left_group)
        full_right_group = self.return_graph(self.letters_df.copy(), 'closest_right',problematic, False)
        self.letters_df = self.letters_df.assign(**full_right_group)
        # df = df.sort_values(by=['group_right_True', 'right'], ascending=[True, False])
        s = self.letters_df.groupby('group_right_False')['group_right_True'].nunique() > 1
        broken_lines_inds = s[s].index

        if not broken_lines_inds.empty:
            d = self.letters_df[(self.letters_df['group_right_True'].isin(
                self.letters_df[self.letters_df['group_right_False'] == broken_lines_inds[0]]['group_right_True'].unique().tolist())) | \
                   (self.letters_df['group_left_True'].isin(
                       self.letters_df[self.letters_df['group_right_False'] == broken_lines_inds[0]]['group_left_True'].unique().tolist()))]
            curr_inds = d.index
            curr_problematic_inds = sorted(set(curr_inds) & set(problematic))
        else:
            curr_problematic_inds = []

        self.letters_df['mean_coord_problematic'] = self.letters_df['mean_coord']
        self.letters_df.loc[curr_problematic_inds, 'mean_coord'] = self.letters_df.loc[curr_problematic_inds, 'mean_coord'].apply(
            lambda x: [1e6, x[1]])

        self.letters_df['closest_right_with_problematic'] = self.letters_df['closest_right']
        close_sides = [np.array(sorted(set(cs) - set(curr_problematic_inds))) for cs in close_sides]
        closest_right = [cs[arr[cs][:, 1] > a[1]] if len(cs) > 0 else [] for cs, a in zip(close_sides, arr)]
        self.letters_df['closest_right'] = [cs[arr[cs][:, 1].argmin()] if len(cs) > 0 else np.nan for cs in closest_right]
        right_group = self.return_graph(self.letters_df.copy(), 'closest_right',problematic)
        self.letters_df = self.letters_df.assign(**right_group)

        self.letters_df['group_right'] = self.letters_df['group_right_True']

        line2squeeze = dict(
            zip(self.letters_df.groupby('group_right').size().index,
                np.array(range(len(self.letters_df.groupby('group_right').size().index))) + 1))

        self.letters_df['group_right'] = self.letters_df['group_right'].map(line2squeeze)
        r = np.array(range(len(self.letters_df.groupby('group_right').size().index))) + 1
        for _ in range(5):
            for g1, g2 in zip(r[:-1], r[1:]):
                group1 = self.letters_df.loc[self.letters_df['group_right'] == g1, ['left', 'right']].agg(['min', 'max']).values[[0, 1], [0, 1]]
                group2 = self.letters_df.loc[self.letters_df['group_right'] == g2, ['left', 'right']].agg(['min', 'max']).values[[0, 1], [0, 1]]
                max_left = max(group1[0], group2[0])
                min_right = min(group1[1], group2[1])
                if min_right < max_left:
                    continue
                try:
                    y1 = np.vstack(self.letters_df.loc[(self.letters_df['group_right'] == g1) & (self.letters_df['right'] < min_right) & (
                            self.letters_df['left'] > max_left), 'mean_coord'].values)[:, 0].mean()
                    y2 = np.vstack(self.letters_df.loc[(self.letters_df['group_right'] == g2) & (self.letters_df['right'] < min_right) & (
                            self.letters_df['left'] > max_left), 'mean_coord'].values)[:, 0].mean()
                    if y2 < y1:
                        replace_groups = {g1: g2, g2: g1}
                        self.letters_df['group_right'] = self.letters_df['group_right'].replace(replace_groups)
                except:
                    continue
        unit_distant_line_prev = self.letters_df.groupby('group_right')['right'].max() <= \
                                 self.letters_df.groupby('group_right')['left'].min().shift()
        unit_distant_line_next = self.letters_df.groupby('group_right')['right'].max() <= \
                                 self.letters_df.groupby('group_right')['left'].min().shift(-1)

        prev_dict = dict()
        next_dict = dict()
        if unit_distant_line_prev.any():
            vals = unit_distant_line_prev[unit_distant_line_prev].index.values
            prev_dict = dict(zip(vals, [a - 1 for a in vals]))
        if unit_distant_line_next.any():
            vals = unit_distant_line_next[unit_distant_line_next].index.values
            next_dict = dict(zip([a + 1 for a in vals], vals))

        replace_dict = {}
        replace_dict.update(prev_dict)
        replace_dict.update(next_dict)

        flag = True
        while flag:
            temp = {k: replace_dict.get(replace_dict.get(k, v), replace_dict.get(k, v)) for k, v in
                    replace_dict.items()}
            if temp == replace_dict:
                flag = False
            replace_dict = temp.copy()

        self.letters_df['group_right'] = self.letters_df['group_right'].replace(replace_dict)

        line2squeeze = dict(
            zip(self.letters_df.groupby('group_right').size().index,
                np.array(range(len(self.letters_df.groupby('group_right').size().index))) + 1))
        self.letters_df['group_right'] = self.letters_df['group_right'].map(line2squeeze)
        self.letters_df = self.letters_df.sort_values(by=['group_right', 'right'], ascending=[True, False])
        self.letters_df = self.letters_df.reset_index()
