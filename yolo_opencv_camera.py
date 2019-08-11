#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
python3 yolo_opencv_camera.py
"""
import cv2
import argparse
import numpy as np
import time
import datetime
import os

import logging
logging.basicConfig(level=logging.INFO)

output_dir = '../out'

detectable_classes = {0, 15}


def arguments_parse():

	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', default=None,
					help = 'path to input image')
	ap.add_argument('-c', '--config', default='yolov3.cfg',
					help = 'path to yolo config file')
	ap.add_argument('-w', '--weights', default='../yolov3.weights',
					help = 'path to yolo pre-trained weights') # required=True
	ap.add_argument('-cl', '--classes', default='yolov3.txt',
					help = 'path to text file containing class names')
	args = ap.parse_args()
	return args


def make_script_dir_as_current():

	os.chdir(os.path.dirname(sys.argv[0]))
	logging.info('new current dir: {}'.format(os.getcwd()))


def get_output_layers(net):
	
	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

	label = str(classes[class_id])
	color = COLORS[class_id]
	cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
	cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_all_predictions(image, prediction):

	indices = prediction['indices']
	boxes = prediction['boxes']
	class_ids = prediction['class_ids']
	confidences = prediction['confidences']
	for i in indices:
		i = i[0]
		x, y, w, h = boxes[i]
		draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))


def print_all_predictions(prediction):

	indices = prediction['indices']
	boxes = prediction['boxes']
	class_ids = prediction['class_ids']
	confidences = prediction['confidences']
	for i in indices:
		i = i[0]
		x, y, w, h = boxes[i]		
		print('{}: {}, {:.4f}, box:{},{},{},{}'.format(i, class_ids[i], confidences[i], x, y, w, h))


def get_prediction(net, image):

	width = image.shape[1]
	height = image.shape[0]
	scale = 0.00392

	class_ids = []
	confidences = []
	boxes = []
	conf_threshold = 0.5
	nms_threshold = 0.4

	blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(get_output_layers(net))

	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)
				x = center_x - w / 2
				y = center_y - h / 2
				class_ids.append(class_id)
				confidences.append(float(confidence))
				boxes.append([x, y, w, h])

	indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
	prediction = {'indices': indices, 'boxes':boxes, 'class_ids':class_ids, 
					'confidences':confidences}
	return prediction


if __name__ == '__main__':

	make_script_dir_as_current()
	os.system('mkdir -p {}'.format(output_dir))	

	args = arguments_parse()
	with open(args.classes, 'r') as f:
		classes = [line.strip() for line in f.readlines()]
	COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

	if os.path.isfile(args.weights):
		path_weights = args.weights
	else:
		path_weights = '/mnt/lin2/ineru/yolov3.weights'
		assert os.path.isfile(path_weights)

	#if args.image:	
	#	image = cv2.imread(args.image)

	net = cv2.dnn.readNet(path_weights, args.config)
	capture = cv2.VideoCapture(0)

	count = 0
	prediction = None
	#indices = []

	while(True):

		return_value, image = capture.read()
			
		#cv2.imshow("object detection", image)
		#cv2.waitKey()				
		#cv2.imwrite("object-detection.jpg", image)

		if cv2.waitKey(1) == 27:
			break
		
		count += 1
		if count % 30 == 0:
			print(count)
			prediction = get_prediction(net, image)
			class_ids = list(prediction['class_ids'])
			class_ids.sort()
			print('class_ids:', class_ids)
			if len(prediction['indices']) > 0:
				if detectable_classes & set(class_ids):
					draw_all_predictions(image, prediction)
					str_class_ids = ','.join(map(str, class_ids))
					str_date = datetime.datetime.now().strftime('%y-%m-%d_%H:%M:%S')
					filename = '{}_{:05d}_[{}].jpg'.format(str_date, count, str_class_ids)
					cv2.imwrite(os.path.join(output_dir, filename), image)
					print('save in {}'.format(filename))
				
		if prediction:
			draw_all_predictions(image, prediction)

		cv2.imshow('video', image)

		if count % 30 == 0:
			print_all_predictions(prediction)			
				
	cv2.destroyAllWindows()
