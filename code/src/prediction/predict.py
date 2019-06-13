from __future__ import print_function

import os
import cv2
import sys
import wave
import pickle
import imutils
import audioread
import contextlib
import numpy as np
import pandas as pd
import soundfile as sf


class Decoder:
    def __init__(self):
        pass

    def decode(filename):
        filename = os.path.abspath(os.path.expanduser(filename))
        if not os.path.exists(filename):
            print("File not found.", file=sys.stderr)
            # sys.exit(1)

        try:
            with audioread.audio_open(filename) as f:
                # print('Input file: %i channels at %i Hz; %.1f seconds.' %
                #       (f.channels, f.samplerate, f.duration),
                #       file=sys.stderr)
                # print('Backend:', str(type(f).__module__).split('.')[1],
                #       file=sys.stderr)

                with contextlib.closing(wave.open(filename + '.wav', 'w')) as of:
                    of.setnchannels(f.channels)
                    of.setframerate(f.samplerate)
                    of.setsampwidth(2)

                    for buf in f:
                        of.writeframes(buf)

        except audioread.DecodeError:
            print("File could not be decoded.", file=sys.stderr)
            # sys.exit(1)


class Step1:
    def __init__(self):
        pass

    def get_frames(self, video_path):
        vs = cv2.VideoCapture(video_path)
        frames = []
        while True:
            success, image = vs.read()
            if not success:
                break
            frames.append(image)

        return frames


    def get_width(self, actual_contour, y_val):
        width = 0
        midpt = 0
        thresh = 0
        while True:
            middle = actual_contour[abs(actual_contour.y-y_val)<=thresh]
            if middle.shape[0] >= 2:
                width = middle.x.max() - middle.x.min()
                midpt = middle.x.min() + width/2
                break
            else:
                thresh += 1
        return [width, midpt]


    def get_features(self, actual_contour, plot=False, frame_no=None, j=None):
        widths = []
        for y in actual_contour.y.unique():
            width_y = self.get_width(actual_contour, y)
            width_y.append(y)
            widths.append(width_y)
        widths = np.array(widths)
        widths = widths[widths[:, 0] > 1]
        width_80 = np.percentile(widths[:, 0], 80)
        width_max = max(widths[:, 0])

        # if plot:
        #     plt.scatter(widths[:, 2], widths[:, 0])
        #     plt.show()
        #
        #     plt.scatter(actual_contour.x, actual_contour.y)
        #     plt.scatter(widths[:, 1], widths[:, 2])
        #     plt.show()

        tail = pd.DataFrame(widths[-1:, 1:])
        head = pd.DataFrame(widths[:1, 1:])
        leading_edge = head.iloc[0, :]
        diffs = head.mean() - tail.mean()
        dist = sum(diffs ** 2)
        if dist != 0:
            dist = dist ** .5

        direction = 0
        if diffs[0] != 0:
            direction = diffs[1] / diffs[0]

        if j is None:
            return (width_80, width_max, leading_edge.values[0],
                    leading_edge.values[1], direction, dist)
        else:
            return (width_80, width_max, leading_edge.values[0],
                    leading_edge.values[1], direction, dist,
                    frame_no, j)


    # requires initial ball position for now
    def step_1(self, video_path, ball_init, frames, predict_impact, classify_bbox):

        Decoder.decode(video_path)

        golf_shot, samplerate = sf.read(video_path + '.wav')
        golf_shot = golf_shot[800000:]
        max_800k = max(golf_shot)
        argmax_800k = np.argmax(golf_shot)

        est_impact = predict_impact.predict(np.array([argmax_800k, max_800k]).reshape(1, -1))[0]

        # frames = get_frames(video_path)
        ball_pos = ball_init

        threshold = 40
        num_since_found = 0
        impact = False
        counter = int(est_impact - 500)
        last_frame = frames[counter - 1]
        gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        last_frame = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_nos = []
        ball_contour_features = []
        ball_boxes = []

        while counter < len(frames):
            frame = frames[counter]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            frame_delta = cv2.absdiff(last_frame, gray)
            thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]

            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            found = False
            for contour in cnts:
                if cv2.contourArea(contour) < 200:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                if y < ball_pos[1]:
                    contour_df = pd.DataFrame(contour.reshape(-1, 2), columns=['x', 'y'])
                    contour_df.sort_values('y', inplace=True)
                    contour_features = self.get_features(contour_df, plot=False)
                    feat = [contour_features[1] / ball_init[2], contour_features[1] - ball_pos[2],
                            contour_features[2] - ball_pos[0], contour_features[3] - ball_pos[1],
                            contour_features[5] / ball_init[3], contour_features[5] - ball_pos[3]]
                    feat_pred = classify_bbox.predict_proba(np.array(feat).reshape(1, -1))[0][1]
                    if feat_pred > .7:
                        frame_nos.append(counter)
                        ball_contour_features.append(contour_features)
                        ball_boxes.append(cv2.boundingRect(contour))

                        ball_pos = [x, y, w, h]
                        num_since_found = 0
                        found = True
                        impact = True
                        #                     print(shot_no)
                        #                     print(counter)
                        break

            if not found:
                num_since_found += 1

            if impact and num_since_found > 2:
                threshold -= 5
                #             print(threshold)
                counter -= num_since_found - 1
                num_since_found = 0
                last_frame = frames[counter - 1]
                gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
                last_frame = cv2.GaussianBlur(gray, (21, 21), 0)
                if threshold < 10:
                    break
            else:
                last_frame = gray
                counter += 1

        return frame_nos, ball_contour_features, ball_boxes


class Step2:
    def __init__(self):
        pass

    def add_na_rows(self, shot_df, session_init):
        max_since_impact = shot_df.since_impact.max()
        num_rows = max_since_impact-shot_df.shape[0]+1
        impact_frame = shot_df.index.min()
        zerodf = pd.DataFrame([[-1 for j in range(shot_df.shape[1])]
                               for i in range(num_rows)], columns=shot_df.columns)
        zerodf.since_impact = [i for i in range(0, max_since_impact+1) if i not in shot_df.since_impact.values]
        zerodf.index = [i for i in range(impact_frame, impact_frame+max_since_impact+1)
                        if i not in shot_df.index.values]
        shot_df.width_max = shot_df.width_max/session_init[2]
        shot_df.leading_edge_x = shot_df.leading_edge_x/session_init[0]
        shot_df.leading_edge_y = shot_df.leading_edge_y/session_init[1]
        combined = pd.concat([shot_df, zerodf])
        return combined.sort_values('since_impact')


    def make_df(self, frame_nos, ball_contour_features, ball_init):
        contour_df = pd.DataFrame(ball_contour_features, index=frame_nos,
                                  columns=['width_80', 'width_max', 'leading_edge_x', 'leading_edge_y',
                                           'direction', 'length'])
        contour_df['since_impact'] = contour_df.index - contour_df.index.min()
        contour_df = self.add_na_rows(contour_df, ball_init)

        wide = pd.concat([contour_df.width_80, contour_df.width_max, contour_df.leading_edge_x,
                          contour_df.leading_edge_y, contour_df.direction, contour_df.length]).values
        col_names = []
        for col in contour_df.columns[:-1]:
            for i in range(contour_df.shape[0]):
                col_names.append(col + '_' + str(i))
        wide_df = pd.DataFrame(wide.reshape(1, -1), columns=col_names)

        return wide_df


    def make_predictions(self, wide_df, ball_init, azimuth_regr, back_spin_regr,
                         ball_speed_regr, launch_angle_regr, side_spin_dt, azimuth_high):
        back_spin_x = wide_df.leading_edge_y_3.values / ball_init[2]

        azimuth = azimuth_regr.predict(wide_df.leading_edge_x_3.values.reshape(1, -1))
        back_spin = back_spin_regr.predict(back_spin_x.reshape(1, -1))
        ball_speed = ball_speed_regr.predict(wide_df.width_max_3.values.reshape(1, -1))
        launch_angle = launch_angle_regr.predict(back_spin_x.reshape(1, -1))

        azimuth_pred_3 = azimuth_regr.predict(wide_df.leading_edge_x_3.values.reshape(-1, 1))
        azimuth_pred_high = azimuth_high.predict(wide_df.leading_edge_x_6.values.reshape(-1, 1))
        side_spin_x = azimuth_pred_high - azimuth_pred_3
        side_spin = side_spin_dt.predict(side_spin_x.reshape(-1, 1))

        return {'azimuth': azimuth[0], 'back_spin': back_spin[0],
                'ball_speed': ball_speed[0], 'launch_angle': launch_angle[0],
                'side_spin': side_spin[0]}


    def get_shot_factors(self, video_path, ball_init, predict_impact, classify_bbox,
                         azimuth_regr, back_spin_regr, ball_speed_regr, launch_angle_regr,
                         side_spin_dt, azimuth_high):
        s1 = Step1()
        frames = s1.get_frames(video_path)
        frame_nos, ball_contour_features, ball_boxes = s1.step_1(video_path, ball_init,
                                                              frames, predict_impact, classify_bbox)
        feature_df = self.make_df(frame_nos, ball_contour_features, ball_init)
        shot_factors = self.make_predictions(feature_df, ball_init, azimuth_regr, back_spin_regr,
                                        ball_speed_regr, launch_angle_regr,
                                             side_spin_dt, azimuth_high)

        return shot_factors

