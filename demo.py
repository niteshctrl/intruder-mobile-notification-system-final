import json
import requests
import cv2
import time
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


def send_alert(filename, alert_type, method='photo'):   # 'photo', 'document', animation
    with open('bot_cred.json','r') as json_file:
        bot_creds = json.load(json_file)
        
    files = {method:open(filename, 'rb')}
    
    resp = requests.post('https://api.telegram.org/bot' + bot_creds['bot_token'] + \
        '/send'+ method + '?chat_id=' + str(bot_creds['bot_chatID']) + '&caption=' + alert_type, files=files)

    return resp.status_code


def notifier(webcam_id=0, time_threshold=15):
    # importing model files
    print("Downloading Model files")
    detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
    print("Download finished.")
    labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
    labels = labels['OBJECT (2017 REL.)']

    # Defining width and height of input image
    width = 512
    height = 512

    # "/dev/video2" for external webcam on my system
    cap = cv2.VideoCapture(webcam_id)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == False:
            break

        # Resize to input_shape
        inp = cv2.resize(frame, (width, height))

        # Convert to RGB
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

        rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
        rgb_tensor = tf.expand_dims(rgb_tensor, 0)

        # predictions
        boxes, scores, classes, num_detections = detector(rgb_tensor)

        pred_labels = classes.numpy().astype('int')[0]
        pred_labels = [labels[i] for i in pred_labels]
        pred_boxes = boxes.numpy()[0].astype('int')
        pred_scores = scores.numpy()[0]

        # loop throughtout the detections and place a box around it
        for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
            if score < 0.5:
                continue
            score_txt = f'{100 * round(score,0)}'
            img_boxes = cv2.rectangle(rgb, (xmin, ymax), (xmax, ymin), (255,255,255), 1)
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(img_boxes,label,(xmin, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
            cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
            
            if label == 'person' and (time.time() % time_threshold) < 2:
                start = time.time()
                cv2.imwrite('alert.jpg', cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
                send_alert('alert.jpg', 'Person Presence')
        
        #Display the resulting frame
        cv2.imshow('black and white',cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    notifier()