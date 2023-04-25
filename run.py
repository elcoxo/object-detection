import cv2
import numpy as np
import yaml
import argparse
import os

def preprocess(img, net):
        blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        
        return outputs

def postprocess(img, outputs, object_idx):
        class_ids = []
        confidences = []
        boxes = []
        img_width, img_height = img.shape[1], img.shape[0]
        x_scale = img_width/640
        y_scale = img_height/640
        
        rows = outputs[0].shape[1]
        for i in range(rows):
            row = outputs[0][0][i]
            confidence = row[4]
            if confidence > 0.5:
                scores = row[5:]
                if scores[object_idx] > 0.5:
                    class_ids.append(object_idx)
                    confidences.append(confidence)
                    center_x, center_y, w, h = row[:4]
                    x = int((center_x- w/2)*x_scale)
                    y = int((center_y-h/2)*y_scale)
                    w = int(w * x_scale)
                    h = int(h * y_scale)

                    box = np.array([x, y, w, h])
                    boxes.append(box)

        indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

        for i in indices:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0), 3)
            cv2.rectangle(img,(0,y),(img_width,0),(0,0,0), -1)
            cv2.rectangle(img,(0,y),(x,img_height),(0,0,0), -1)
            cv2.rectangle(img,(0,y+h),(img_width,img_height),(0,0,0), -1)
            cv2.rectangle(img,(x+w,0),(img_width,img_height),(0,0,0), -1)

        return img

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Input video e.g. <name>.mp4", default='1.mp4')
    parser.add_argument("object", help="Coco class name to detect object e.g. cat", default='cat')
    args = parser.parse_args()
    filename = args.name
    objectname = args.object
    print('File name: ' + filename)
    print('Object name: ' + objectname)

    net = cv2.dnn.readNetFromONNX('yolov5/models/yolov5s.onnx')
    cap = cv2.VideoCapture('input/' + filename)
    
    
    
    classes = []
    with open('yolov5/data/coco.yaml', 'r') as file:
        dict = yaml.load(file, Loader=yaml.FullLoader)
        arr = list(dict['names'].items())
        for key in range(len(arr)):
            classes.append(arr[key][1]) # Coco 80 classes name
        if (classes.index(objectname)):
            print("Object name found!")
            object_idx = classes.index(objectname)
        else:
            print("Object name not found :(")

    if (cap.isOpened() == True):
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(5)
        print('Weight x Height FPS: ', width, 'x', height, fps)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video = cv2.VideoWriter('result/' + os.path.basename(filename).split(".")[0] + '_' + 
                                objectname + '.avi', fourcc, fps, (width,height))
    else:
        print('Can''t load the file, try another one :(')

    print('Processing...')
    while(cap.isOpened()):
        flag, img = cap.read()
        if flag==True:
            cap_blob = preprocess(img, net)
            output = postprocess(img, cap_blob, object_idx)
            video.write(output)
        else:
            break
    
    cap.release()
    video.release()
    cv2.destroyAllWindows()
    print('Done :)\nFile saved in: result/' + os.path.basename(filename).split(".")[0] + '_' + objectname + '.avi')