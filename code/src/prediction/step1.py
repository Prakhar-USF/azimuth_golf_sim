import pandas as pd
import numpy as np
import imutils
import cv2
import soundfile as sf
import pickle
from convert_wav import *


def get_frames(video_path):
    vs = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, image = vs.read()
        if not success:
            break
        frames.append(image)

    return frames


def get_width(actual_contour, y_val):
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


def get_features(actual_contour, plot=True, frame_no=None, j=None):
    widths = []
    for y in actual_contour.y.unique():
        width_y = get_width(actual_contour, y)
        width_y.append(y)
        widths.append(width_y)
    widths = np.array(widths)
    widths = widths[widths[:, 0] > 1]
    width_80 = np.percentile(widths[:, 0], 80)
    width_max = max(widths[:, 0])

    if plot:
        plt.scatter(widths[:, 2], widths[:, 0])
        plt.show()

        plt.scatter(actual_contour.x, actual_contour.y)
        plt.scatter(widths[:, 1], widths[:, 2])
        plt.show()

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
        return (width_80, width_max, leading_edge.values[0], leading_edge.values[1], direction, dist)
    else:
        return (width_80, width_max, leading_edge.values[0], leading_edge.values[1], direction, dist,
                frame_no, j)


# requires initial ball position for now
def step_1(video_path, ball_init, frames, predict_impact, classify_bbox):
    decode(video_path)

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
                contour_features = get_features(contour_df, plot=False)
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
