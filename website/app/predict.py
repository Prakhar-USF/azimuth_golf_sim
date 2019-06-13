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
from typing import *
import soundfile as sf
from sklearn.linear_model import LinearRegression


class Decoder:
    """
    Video decoder.
    """
    def __init__(self):
        pass

    def decode(self, filename: str):
        """
        Main function to decode a video file.

        :param filename: file path
        """
        filename = os.path.abspath(os.path.expanduser(filename))
        if not os.path.exists(filename):
            print("File not found.", file=sys.stderr)
            # sys.exit(1)

        try:
            with audioread.audio_open(filename) as f:
                with contextlib.closing(wave.open(filename+'.wav', 'w')) as of:
                    of.setnchannels(f.channels)
                    of.setframerate(f.samplerate)
                    of.setsampwidth(2)

                    for buf in f:
                        of.writeframes(buf)

        except audioread.DecodeError:
            print("File could not be decoded.", file=sys.stderr)
            # sys.exit(1)

class first_frame:
    def __init__(self):
        pass
    
    def getOutputsNames(self, net):
        # Get names of output layers, output for YOLOv3 is ['yolo_16', 'yolo_23']
        layersNames = net.getLayerNames()
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    def get_first_frame_bbox(self, frames_list, start_frame=0, div=1, conf_threshold=0.5,
                         classesFile = './app/static/models/yolov3_golf_ball.names',
                         modelConfiguration = './app/static/models/yolov3_golf_ball.cfg',
                         modelWeights = './app/static/models/yolov3_golf_ball.backup'):
        classes = None
        with open(classesFile, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        net = cv2.dnn.readNet(modelWeights, modelConfiguration)

        idx = start_frame

        while idx < len(frames_list):
            image = frames_list[idx]
            blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (416, 416), [0, 0, 0], True, crop=False)
            Width = image.shape[1]
            Height = image.shape[0]
            net.setInput(blob)

            outs = net.forward(self.getOutputsNames(net))

            class_ids = []
            confidences = []
            boxes = []
            nms_threshold = 0.4

            for out in outs:
                # print(out.shape)
                for detection in out:

                    # each detection  has the form like this [center_x center_y width height
                    # obj_score class_1_score class_2_score ..]
                    scores = detection[5:]  # classes scores starts from index 5
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            # apply  non-maximum suppression algorithm on the bounding boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            for i in indices:
                i = i[0]
                box = boxes[i]
                # print(confidences[i])
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

            if len(class_ids) > 0 and w / h > 0.4 and w / h < 2.5:
                return ([idx, x, y, w, h])

            idx += div

        return None
            
class Step1:
    def __init__(self):
        pass

    def get_frames(self, video_path: str) -> List:
        """
        Extract frames from a video.

        :param video_path: video file path
        :return: A list of frames
        """
        changed = False
        if video_path[-3:] in ['MOV', 'mov']:
            command = 'ffmpeg -i ' + video_path + ' -qscale 0 ' + video_path[:-3] + 'mp4'
#             print(command)
            os.system(command)
            changed = True
            video_path = video_path[:-3] + 'mp4'

        vs = cv2.VideoCapture(video_path)
        frames = []
        while True:
            success, image = vs.read()
            if not success:
                break
            frames.append(image)

        if changed == True:
            os.remove(video_path)
        return frames

    def make_gray(self, frames):
        gray_frames = []
        for frame in frames:
            gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            gray_frames.append(gray)

        return gray_frames

    def diff_threshold(self, frames, counter, threshold):
        frame_delta = cv2.absdiff(frames[counter-1], frames[counter])
        thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        return thresh
    
    def get_width(self, actual_contour, y_val):
        width = 0
        midpt = 0
        thresh = 0
        while True:
            middle = actual_contour[abs(actual_contour.y-y_val) <= thresh]
            if middle.shape[0] >= 2:
                width = middle.x.max() - middle.x.min()
                midpt = middle.x.min() + width/2
                break
            else:
                thresh += 1
        return [width, midpt]

    def get_features(self, actual_contour, plot=False,
                     frame_no=None, bbox_no=None):
        """
        Calculate contour features: 80% width, max width, leading edge (x, y),
        direction, distance, frame index, bounding box index (if exist)

        :param actual_contour: the contour to be analyzed
        :param plot: bool, default False, if True, plot the contour
        :param frame_no: frame index
        :param bbox_no: the index of the bounding box detected in the frame
        :return: contour features
        """
        widths = []
        for y in actual_contour.y.unique():
            width_y = self.get_width(actual_contour, y)
            width_y.append(y)
            widths.append(width_y)
        widths = np.array(widths)
        widths = widths[widths[:, 0] > 1]
#         width_80 = np.percentile(widths[:, 0], 80)
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

#         direction = 0
#         if diffs[0] != 0:
#             direction = diffs[1] / diffs[0]

        return leading_edge.values[0], leading_edge.values[1], width_max, dist

    # requires initial ball position for now
    def step_1(self, video_path, ball_init, frames_gray, frames_color, predict_impact, classify_bbox):
        """
        Step One: find the frames in which the ball has been hit and is flying,
        then get the contour features and bounding boxes of the moving ball.

        :param video_path: video file path
        :param ball_init: (currently fixed) bounding box of the pre-impact ball
        :param frames: a list of frames
        :param predict_impact: model for predicting the exact frame
        when the ball is hit
        :param classify_bbox: model for detecting the bounding boxes
        :return: three lists - frame indices, contour features, bounding boxes
        """
        # decode(video_path)
        #
        # golf_shot, samplerate = sf.read(video_path + '.wav')
        # golf_shot = golf_shot[800000:]
        # max_800k = max(golf_shot)
        # argmax_800k = np.argmax(golf_shot)
        #
        # est_impact = predict_impact.predict(np.array([argmax_800k, max_800k]).reshape(1, -1))[0]
        # counter = int(est_impact - 500)

        counter = 1

        # frames = get_frames(video_path)
        ball_pos = ball_init
        threshold = 50
        num_since_found = 0
        impact = False

        frame_nos = []
        ball_contour_features = []
        ball_boxes = []

        while counter < len(frames_gray):
            thresh = self.diff_threshold(frames_gray, counter, threshold)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            found = False
            for contour in cnts:
                contourArea = cv2.contourArea(contour)
                if contourArea < 50:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                if y > ball_pos[1]:
                    continue
                if y < ball_pos[1] - 200:
                    continue
                if x > ball_pos[0] + 100:
                    continue
                if x < ball_pos[0] - 100:
                    continue
                if w > ball_pos[2] * 2:
                    continue

                contour_df = pd.DataFrame(contour.reshape(-1, 2), columns=['x', 'y'])
                contour_df.sort_values('y', inplace=True)
                contour_features = self.get_features(contour_df)
                feat = [contour_features[2] / ball_init[2], contour_features[2] - ball_pos[2],
                        contour_features[0] - ball_pos[0], contour_features[1] - ball_pos[1],
                        contour_features[3] / ball_init[3], contour_features[3] - ball_pos[3]]
                feat_pred = classify_bbox.predict_proba(np.array(feat).reshape(1, -1))[0][1]
                if feat_pred > .5:
#                     print(counter)
#                     print(feat_pred)
                    frame_nos.append(counter)
                    # print(threshold)
                    ball_pos = [x, y, w, h]
#                     plot_one_box(frames_color[counter], ball_pos)

                    ball_contour_features.append(contour_features)
                    ball_boxes.append(ball_pos)

                    num_since_found = 0
                    found = True
                    impact = True
                    break

            if not found:
                num_since_found += 1

            if impact and num_since_found > 2:
                threshold -= 5
                # print(threshold)
                counter -= num_since_found - 1
                num_since_found = 0
                if threshold < 10:
                    break
            else:
                counter += 1

        return frame_nos, ball_contour_features, ball_boxes
    
    def test_in_flight(self, ball_init, frames_gray, frames_color, classify_bbox, plot=False):
        counter = 1
        ball_pos = ball_init

        threshold = 40
        num_since_found = 0

        frame_nos = []
        ball_contour_features = []
        ball_boxes = []
        probs = []
        thresholds = []

        while counter < len(frames_gray):
            thresh = self.diff_threshold(frames_gray, counter, threshold)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            found = False
            max_prob = 0
            best_contour_features = []
            best_ball_pos = []
            best_threshold = threshold
            for contour in cnts:
                contourArea = cv2.contourArea(contour)
                if contourArea < 50:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                if y > ball_pos[1]:
                    continue
                if y < ball_pos[1] - 200:
                    continue
                if x > ball_pos[0] + 100:
                    continue
                if x < ball_pos[0] - 100:
                    continue
                if w > ball_pos[2] * 2:
                    continue

                contour_df = pd.DataFrame(contour.reshape(-1, 2), columns=['x', 'y'])
                contour_df.sort_values('y', inplace=True)
                contour_features = self.get_features(contour_df)

                contour_pct = contourArea / (w * h)
                avg_pixel = frames_gray[counter][slice(x, x + w), slice(y, y + h)].sum() / (w * h)
                x_diff = contour_features[0] / ball_pos[0]
                y_diff = contour_features[1] / ball_pos[1]
                w_diff = contour_features[2] / ball_pos[2]
                h_diff = contour_features[3] / ball_pos[3]

                x_diff_init = contour_features[0] / (ball_init[0] + ball_init[2] / 2)
                y_diff_init = contour_features[1] / (ball_init[1] + ball_init[3] / 2)
                w_diff_init = contour_features[2] / ball_init[2]
                h_diff_init = contour_features[3] / ball_init[3]

                frames_impact = counter
                frames_last = num_since_found + 1

                feat = [contour_pct, avg_pixel, x_diff, y_diff, w_diff, h_diff,
                        x_diff_init, y_diff_init, w_diff_init, h_diff_init,
                        frames_impact, frames_last, threshold]

                feat_pred = classify_bbox.predict_proba(np.array(feat).reshape(1, -1))[0][1]
                if feat_pred > max_prob:
                    temp_ball_pos = [x, y, w, h]
                    best_contour_features = contour_features
                    best_ball_pos = temp_ball_pos
                    max_prob = feat_pred
                    best_threshold = threshold

            if max_prob > .3:
                # print(counter)
                # print(feat_pred)
                frame_nos.append(counter)
                # print(threshold)

#                 if plot:
#                     plot_one_box(frames_color[counter], best_ball_pos)

                ball_pos = best_ball_pos
                ball_contour_features.append(best_contour_features)
                ball_boxes.append(best_ball_pos)
                probs.append(max_prob)
                thresholds.append(best_threshold)

                num_since_found = 0
                found = True

            if not found:
                num_since_found += 1

            if num_since_found > 2:
                threshold -= 5
                # print(threshold)
                counter -= num_since_found - 1
                num_since_found = 0
                if threshold < 10:
                    break
            else:
                counter += 1

        return frame_nos, ball_contour_features, ball_boxes, probs, thresholds


    def test_impact_finder(self, ball_init, frames_gray, frames_color, find_impact,
                           max_frame_no, threshold=40, plot=False):
        impacts = []
        all_features = []
        boxes = []
        probs = []
        for frame_no in range(1, max_frame_no+1):
            thresh = self.diff_threshold(frames_gray, frame_no, threshold)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            for contour in cnts:
                contourArea = cv2.contourArea(contour)
                if contourArea < 50:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                if y > ball_init[1]:
                    continue
                if y < ball_init[1]*.8:
                    continue
                if x > ball_init[0]*1.15:
                    continue
                if x < ball_init[0]*.9:
                    continue

                contour_df = pd.DataFrame(contour.reshape(-1, 2), columns=['x', 'y'])
                contour_df.sort_values('y', inplace=True)
                contour_features = self.get_features(contour_df)
                contour_pct = contourArea / (w * h)
                avg_pixel = frames_gray[frame_no][slice(x, x + w), slice(y, y + h)].sum() / (w * h)
                x_diff = contour_features[0] / ball_init[0]
                y_diff = contour_features[1] / ball_init[1]
                w_diff = contour_features[2] / ball_init[2]
                h_diff = contour_features[3] / ball_init[3]
                features = [contour_pct, avg_pixel, x_diff, y_diff, w_diff, h_diff]
                impact_prob = find_impact.predict_proba(np.array(features).reshape(1, -1))[0][1]

                if impact_prob > .175:
                    # print(frame_no)
                    # print(impact_prob)
                    box = [x, y, w, h]
#                     if plot:
#                         self.plot_one_box(frames_color[frame_no], box)
                    impacts.append(frame_no)
                    all_features.append(contour_features)
                    boxes.append(box)
                    probs.append(impact_prob)

        return impacts, all_features, boxes, probs
    
    
class Step2:
    def __init__(self):
        pass

    def get_shot(self, shot_num, session_data):
        shot = session_data.loc[shot_num]
        return shot[shot.since_impact>=0]

    def add_na_rows(self, shot_df, max_since_impact, shot_init):
        """
        Add na rows (filled with -1's) if there is a gap in the frames
        in which the ball is detected.

        :param shot_df: shot factors
        :param max_since_impact: 
        :param shot_init: bounding box of the ball pre-impact
        :return: data frame with missing frames filled
        """
        num_rows = max_since_impact-shot_df.shape[0]+1
        zerodf = pd.DataFrame([[-1 for j in range(shot_df.shape[1])]
                             for i in range(num_rows)],
                            index=[shot_df.index[0] for n in range(num_rows)], columns=shot_df.columns)
        zerodf.since_impact = [i for i in range(0, max_since_impact+1) if i not in
                               shot_df.since_impact.values]
        shot_df.width_max = shot_df.width_max / shot_init[2]
        shot_df.leading_edge_x = shot_df.leading_edge_x / shot_init[0]
        shot_df.leading_edge_y = shot_df.leading_edge_y / shot_init[1]
        combined = pd.concat([shot_df, zerodf])
        return combined.sort_values('since_impact')

    def make_wide(self, session_factors, all_labels, session_no, shot_inits, max_since_impact):
        wide_data = []
        width_inits = []
        for shot_no in session_factors.index.unique():
            shot_i = self.get_shot(shot_no, session_factors)
            shot_init = shot_inits[shot_inits.session == session_no].loc[shot_no]
            shot_i = self.add_na_rows(shot_i, max_since_impact, shot_init[2:])
            row_i = pd.concat([shot_i.width_max, shot_i.leading_edge_x, shot_i.leading_edge_y,
                               shot_i.length]).values
            width_inits.append(shot_init[4])
            wide_data.append(row_i)

        col_names = []
        for col in ['width_max', 'leading_x', 'leading_y', 'length']:
            for i in range(max_since_impact + 1):
                col_names.append(col + '_' + str(i))

        wide_data = pd.DataFrame(wide_data, columns=col_names, index=session_factors.index.unique())
        wide_data['width_init'] = width_inits
        session_labels = all_labels.loc[session_no]
        session_labels.set_index('shot_no', inplace=True)
        full_data = wide_data.join(session_labels)

        return full_data
    
    def drop_nas(self, df, frame_no, filter_col, output_cols=None):
        filter_col = filter_col + '_' + frame_no
        all_cols = None
        if type(output_cols) == list:
            all_cols = [filter_col] + output_cols
        elif type(output_cols) == str:
            all_cols = [filter_col] + [output_cols]
        else:
            all_cols = filter_col

        df = df[all_cols]
        df = df[df[filter_col] != -1]

        return df

    def make_df(self, frame_nos, ball_contour_features, ball_init):
        """
        Construct a data frame for the output of `Step1.step_1`.

        :param frame_nos: selected frame indices
        :param ball_contour_features: ball contour features
        :param ball_init: (currently fixed) bounding box of the pre-impact ball
        :return: data frame including all the input information
        """
        contour_df = pd.DataFrame(ball_contour_features, index=frame_nos,
                              columns=['leading_edge_x', 'leading_edge_y', 'width_max', 'length'])
        contour_df['since_impact'] = contour_df.index - contour_df.index.min()
        contour_df = self.add_na_rows(contour_df, max(frame_nos), ball_init)

        wide = pd.concat([contour_df.leading_edge_x, contour_df.leading_edge_y,
                          contour_df.width_max, contour_df.length]).values
        col_names = []
        for col in contour_df.columns[:-1]:
            for i in range(contour_df.shape[0]):
                col_names.append(col + '_' + str(i))
        wide_df = pd.DataFrame(wide.reshape(1, -1), columns=col_names)

        return wide_df

    def diff_ball_speed(self, wide_df, ind_min, ind_max):
        factor = 'ball_speed'
        col = 'width_max'
        cols = [col + '_' + str(ind) for ind in range(ind_min, ind_max)]
        cols.append(factor)

        diffs = [wide_df[cols[0]]]
        for ind in range(ind_min, ind_max):
            col_ind = 'width_max_' + str(ind + 1)
            col_ind_1 = 'width_max_' + str(ind)
            width_max_ind = wide_df[col_ind]
            width_max_ind_1 = wide_df[col_ind_1]
            diff_ind = width_max_ind - width_max_ind_1
            diffs.append(diff_ind)

        return np.array(diffs)
    
    def make_predictions(self, flight_df, azimuth_regr, back_spin_regr,
                     ball_speed_regr, launch_angle_regr, side_spin_regr) -> dict:
        """
        Predict the five factors needed for trajectory prediction.

        :param flight_df: data frame of ball info in flight
        :param azimuth_regr: model to predict azimuth
        :param back_spin_regr: model to predict back-spin
        :param ball_speed_regr: model to predict ball speed
        :param launch_angle_regr: model to predict azimuth
        :param side_spin_regr: model to predict side-spin
        :return: a dictionary of factors {factor name: value}
        """
        shot_coeffs = []
        flight_length = 0
        resid_thresh = {'x':.04, 'y':.2, 'width':.3}

        for col_cat in ['x', 'y', 'width']:
            frame_nos = flight_df.frame_no.values
            values = flight_df[col_cat].values / flight_df[col_cat].iloc[0]
            values = np.log(values)
            if col_cat == 'width':
                frame_nos = frame_nos[2:]
                values = values[2:]
            regr = LinearRegression().fit(frame_nos.reshape(-1, 1), values.reshape(-1, 1))

            y_pred = regr.predict(frame_nos.reshape(-1, 1))
            resids = abs(y_pred.reshape(-1) - values)
            while (max(resids) > resid_thresh[col_cat]):
                frame_nos = frame_nos[resids != max(resids)]
                values = values[resids != max(resids)]
                regr = LinearRegression().fit(frame_nos.reshape(-1, 1), values.reshape(-1, 1))
                y_pred = regr.predict(frame_nos.reshape(-1, 1))
                resids = abs(y_pred.reshape(-1) - values)
            flight_length = len(frame_nos)
            # plt.scatter(frame_nos, values)
            # plt.title(col_cat)
            # plt.show()
            shot_coeffs.append(regr.coef_[0][0])
        shot_coeffs = np.array(shot_coeffs)

        azimuth = azimuth_regr.predict(shot_coeffs[0].reshape(-1, 1))[0][0]
        launch_angle = launch_angle_regr.predict(shot_coeffs[1].reshape(-1, 1))[0][0]
        ball_speed = ball_speed_regr.predict(shot_coeffs[2].reshape(-1, 1))[0][0]

        back_spin = back_spin_regr.predict(np.array(launch_angle).reshape(-1, 1))[0]
        side_spin = side_spin_regr.predict(np.array(azimuth).reshape(-1, 1))[0]


        return {'ball_speed': ball_speed*0.882, 'launch_angle': launch_angle,
                'azimuth': azimuth, 'side_spin': side_spin/2, 'back_spin': back_spin},\
               shot_coeffs

    def get_shot_factors(self, video_path, impact_finder, classify_bbox,
                         azimuth_regr, back_spin_regr, ball_speed_regr,
                         launch_angle_regr, side_spin_regr):
        """
        The main function to get shot factors from raw video file.

        :param video_path: video file path
        :param impact_finder: model for predicting the exact frame
        when the ball is hit
        :param classify_bbox: model for detecting the bounding boxes
        :param azimuth_regr: model to predict azimuth
        :param back_spin_regr: model to predict back-spin
        :param ball_speed_regr: model to predict ball speed
        :param launch_angle_regr: model to predict azimuth
        :param side_spin_regr: model to predict side-spin
        :return: a dictionary of factors {factor name: value}
        """
        s1 = Step1()
        frames_color = s1.get_frames(video_path)
        frames_gray = s1.make_gray(frames_color)
        max_frame_no = len(frames_color) - 1

        ff = first_frame()
        init = ff.get_first_frame_bbox(frames_color)
#         plot_one_box(frames_color[0], [int(pix) for pix in init[1:]])
        ball_init = init[1:]

        s2 = Step2()
        impacts_shot = []
        impact_features_shot = []
        boxes_shot = []
        impact_probs_shot = []
        thresholds_shot = []
        for threshold in [40, 30]:
            impacts, impact_features, boxes, impact_probs = s1.test_impact_finder(
                ball_init, frames_gray, frames_color, impact_finder, max_frame_no, threshold=threshold)
            impacts_shot.extend(impacts)
            impact_features_shot.extend(impact_features)
            boxes_shot.extend(boxes)
            impact_probs_shot.extend(impact_probs)
            thresholds_shot.extend([threshold for _ in range(len(impacts))])

        threshold = 25
        if len(impact_probs_shot) == 0:
            impact_probs_shot = [0]
        while max(impact_probs_shot) < .5:
            impacts, impact_features, boxes, impact_probs = s1.test_impact_finder(
                ball_init, frames_gray, frames_color, impact_finder, max_frame_no, threshold=threshold)
            impacts_shot.extend(impacts)
            impact_features_shot.extend(impact_features)
            boxes_shot.extend(boxes)
            impact_probs_shot.extend(impact_probs)
            thresholds_shot.extend([threshold for _ in range(len(impacts))])
            threshold -= 5
            if threshold < 10:
                break
        if max(impact_probs_shot) == 0:
            return {'ball_speed': 89.56, 'launch_angle': 28.72, 'azimuth': 2.52, 
                    'side_spin': 832.10,'back_spin': 5784.02}, []
        
        index = np.argmax(impact_probs_shot)

#         plot_one_box(frames_color[impacts_shot[index]], boxes_shot[index])

        frames_color_cropped = frames_color[impacts_shot[index]:].copy()
        frames_gray_cropped = frames_gray[impacts_shot[index]:].copy()
        frame_nos, ball_contour_features, ball_boxes, flight_probs, thresholds = s1.test_in_flight(
            boxes_shot[index], frames_gray_cropped, frames_color_cropped, classify_bbox,
            plot=False)

        if len(ball_contour_features)==0:
            return {'ball_speed': 89.56, 'launch_angle': 28.72, 'azimuth': 2.52, 
                    'side_spin': 832.10,'back_spin': 5784.02}, []

        all_frame_nos = np.r_[[-1, 0], frame_nos]
        ball_init[0] = ball_init[0] + ball_init[2] / 2
        all_features_shot = np.r_[[tuple(ball_init)], [impact_features_shot[np.argmax(impact_probs_shot)]],
                                  ball_contour_features]
        flight_df = pd.DataFrame(all_features_shot,
                                 columns=['x', 'y', 'width', 'length'])
        flight_df['frame_no'] = all_frame_nos

        print(flight_df.shape[0])

        shot_factors, shot_coeffs = s2.make_predictions(flight_df, azimuth_regr, back_spin_regr,
                                                     ball_speed_regr, launch_angle_regr,
                                                     side_spin_regr)

        return shot_factors, shot_coeffs.tolist()
