from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from . import data_augment
import threading
import itertools
import skimage as ski

'''
Prepare the data to be fed to CNN.
1. resize the image to input_resize(input augment, default is 600).
	image shape: (1, depths, cols, rows)
2. generate the region proposals match groundtruth best:
	y_rpn_cls: shape=(1, anchor_nums*2, featuremap_depth, featuremap_height, featuremap_width)
				Contains each anchor box [is_valid_for_use, is_positive] on each point on the feature map.

	y_rpn_regr: shape=(1, 2*4*anchor_nums, featuremap_depth, featuremap_height, featuremap_width)
				Then second dimension related to: (is_positive,is_positive,is_positive,is_positive, tx1, tx2, tx3, tr) latter 4 are the offset between propsoal and groundtruth

Generator Outputs:
	x_img, [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug


'''

def union3d(au, bu, area_intersection):
	area_a = (au[3] - au[0]) * (au[4] - au[1]) * (au[5] - au[2])
	area_b = (bu[3] - bu[0]) * (bu[4] - bu[1]) * (bu[5] - bu[2])
	area_union = area_a + area_b - area_intersection
	return area_union


def intersection3d(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	z = max(ai[2], bi[2])
	w = min(ai[3], bi[3]) - x
	h = min(ai[4], bi[4]) - y
	d = min(ai[5], bi[5]) - z
	if w < 0 or h < 0 or d < 0:
		return 0
	return w*h*d


def iou_r(a, b):
	# a and b should be (x1,x2,x3,r)
	a_x1_min = a[0] - a[3]
	a_x1_max = a[0] + a[3]
	a_x2_min = a[1] - a[3]
	a_x2_max = a[1] + a[3]
	a_x3_min = a[2] - a[3]
	a_x3_max = a[2] + a[3]

	x = [a_x1_min, a_x2_min, a_x3_min, a_x1_max, a_x2_max, a_x3_max]

	b_x1_min = b[0] - b[3]
	b_x1_max = b[0] + b[3]
	b_x2_min = b[1] - b[3]
	b_x2_max = b[1] + b[3]
	b_x3_min = b[2] - b[3]
	b_x3_max = b[2] + b[3]

	y = [b_x1_min, b_x2_min, b_x3_min, b_x1_max, b_x2_max, b_x3_max]

	area_i = intersection3d(x, y)
	area_u = union3d(x, y, area_i)

	return float(area_i) / float(area_u + 1e-6)

def get_new_img_size(width, height, dense, img_min_side=600):
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
		resized_dense = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side
		resized_dense = img_min_side

	return resized_width, resized_height, resized_dense


class SampleSelector:
	def __init__(self, class_count):
		# ignore classes that have zero samples
		self.classes = [b for b in class_count.keys() if class_count[b] > 0]
		self.class_cycle = itertools.cycle(self.classes)
		self.curr_class = next(self.class_cycle)

	def skip_sample_for_balanced_class(self, img_data):

		class_in_img = False

		for bbox in img_data['bboxes']:

			cls_name = bbox['class']

			if cls_name == self.curr_class:
				class_in_img = True
				self.curr_class = next(self.class_cycle)
				break

		if class_in_img:
			return False
		else:
			return True


def calc_rpn(C, img_data, width, height, dense, resized_width, resized_height, resized_dense,img_length_calc_function):
	'''
	y_rpn_cls: shape=(1, anchor_nums*2, featuremap_depth, featuremap_height, featuremap_width)
				Contains each anchor box [is_valid_for_use, is_positive] on each point on the feature map.

	y_rpn_regr: shape=(1, 2*4*anchor_nums, featuremap_depth, featuremap_height, featuremap_width)
				Then second dimension related to: (is_positive,is_positive,is_positive,is_positive, tx1, tx2, tx3, tr) latter 4 are the offset between propsoal and groundtruth

	'''
	downscale = float(C.rpn_stride)
	anchor_sizes = C.anchor_box_scales
	anchor_ratios = C.anchor_box_ratios
	num_anchors = len(anchor_sizes) * len(anchor_ratios)	

	# calculate the output map size based on the network architecture

	(output_width, output_height, output_dense) = img_length_calc_function(resized_width, resized_height, resized_dense)

	n_anchratios = len(anchor_ratios)
	
	# initialise empty output objectives
	y_rpn_overlap = np.zeros((output_dense, output_height, output_width, num_anchors))
	y_is_box_valid = np.zeros((output_dense, output_height, output_width, num_anchors))
	y_rpn_regr = np.zeros((output_dense, output_height, output_width, num_anchors * 4))

	num_bboxes = len(img_data['bboxes'])

	num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
	best_anchor_for_bbox = -1*np.ones((num_bboxes, 5)).astype(int) # [x1,x2,x3, anchor_ratio_idx, anchor_size_idx]
	best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
	best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)        # [x1, x2, x3, r]
	best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)# [tx1, tx2, tx3, tr]

	# get the GT box coordinates, and resize to account for image resizing
	gta = np.zeros((num_bboxes, 4))
	for bbox_num, bbox in enumerate(img_data['bboxes']):
		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = bbox['x1'] * (resized_dense / float(dense))
		gta[bbox_num, 1] = bbox['x2'] * (resized_dense / float(dense))
		gta[bbox_num, 2] = bbox['x3'] * (resized_height / float(height))
		gta[bbox_num, 3] = bbox['r'] * (resized_height / float(height))
	
	# rpn ground truth

	for anchor_size_idx in range(len(anchor_sizes)):
		for anchor_ratio_idx in range(n_anchratios):
			anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
			anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]
			anchor_z = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][2]

			assert anchor_x == anchor_y
			assert anchor_y == anchor_z

			r_anc = anchor_x

			for ix in range(output_width):					
				# x-coordinates of the current anchor box	
				x1_anc = downscale * ix
				# ignore boxes that go across image boundaries					
				if x1_anc < 0 or x1_anc > resized_width:
					continue
					
				for jy in range(output_height):
					# y-coordinates of the current anchor box
					y1_anc = downscale * jy 
					# ignore boxes that go across image boundaries
					if y1_anc < 0 or y1_anc > resized_height:
						continue

					for kz in range(output_dense):
						# y-coordinates of the current anchor box
						z1_anc = downscale * kz
						# ignore boxes that go across image boundaries
						if z1_anc < 0 or z1_anc > resized_dense:
							continue

						# bbox_type indicates whether an anchor should be a target 
						bbox_type = 'neg'	

						# this is the best IOU for the (x,y) coord and the current anchor
						# note that this is different from the best IOU for a GT bbox
						best_iou_for_loc = 0.0	

						for bbox_num in range(num_bboxes):
							# get IOU of the current GT box and the current anchor box
							curr_iou = iou_r([gta[bbox_num, 0], gta[bbox_num, 1], gta[bbox_num, 2], gta[bbox_num, 3]], [z1_anc, y1_anc, x1_anc, r_anc])
							#curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
							# calculate the regression targets if they will be needed
							if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
								cx1 = gta[bbox_num, 0]
								cx2 = gta[bbox_num, 1]
								cx3 = gta[bbox_num, 2]

								cx1a = z1_anc
								cx2a = y1_anc
								cx3a = x1_anc

								tx1 = (cx1 - cx1a) / r_anc
								tx2 = (cx2 - cx2a) / r_anc
								tx3 = (cx3 - cx3a) / r_anc

								tr = np.log(gta[bbox_num,3] / r_anc)

							if img_data['bboxes'][bbox_num]['class'] != 'bg':	
								# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
								if curr_iou > best_iou_for_bbox[bbox_num]:
									best_anchor_for_bbox[bbox_num] = [kz, jy, ix, anchor_ratio_idx, anchor_size_idx]
									best_iou_for_bbox[bbox_num] = curr_iou
									best_x_for_bbox[bbox_num,:] = [z1_anc, y1_anc, x1_anc, r_anc]
									best_dx_for_bbox[bbox_num,:] = [tx1, tx2, tx3, tr]	

								# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
								if curr_iou > C.rpn_max_overlap:
									bbox_type = 'pos'
									num_anchors_for_bbox[bbox_num] += 1
									# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
									if curr_iou > best_iou_for_loc:
										best_iou_for_loc = curr_iou
										best_regr = (tx1, tx2, tx3, tr)	

								# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
								if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
									# gray zone between neg and pos
									if bbox_type != 'pos':
										bbox_type = 'neutral'	

						# turn on or off outputs depending on IOUs
						if bbox_type == 'neg':
							y_is_box_valid[kz, jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
							y_rpn_overlap[kz, jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						elif bbox_type == 'neutral':
							y_is_box_valid[kz, jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
							y_rpn_overlap[kz, jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						elif bbox_type == 'pos':
							y_is_box_valid[kz, jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
							y_rpn_overlap[kz, jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
							start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
							y_rpn_regr[kz, jy, ix, start:start+4] = best_regr

	# we ensure that every bbox has at least one positive RPN region

	for idx in range(num_anchors_for_bbox.shape[0]):
		if num_anchors_for_bbox[idx] == 0:
			# no box with an IOU greater than zero ...
			if best_anchor_for_bbox[idx, 0] == -1:
				continue
			y_is_box_valid[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2], best_anchor_for_bbox[idx,3] + n_anchratios *
				best_anchor_for_bbox[idx,4]] = 1
			y_rpn_overlap[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2], best_anchor_for_bbox[idx,3] + n_anchratios *
				best_anchor_for_bbox[idx,4]] = 1
			start = 4 * (best_anchor_for_bbox[idx,3] + n_anchratios * best_anchor_for_bbox[idx,4])
			y_rpn_regr[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2], start:start+4] = best_dx_for_bbox[idx, :]

	y_rpn_overlap = np.transpose(y_rpn_overlap, (3, 0, 1, 2))
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

	y_is_box_valid = np.transpose(y_is_box_valid, (3, 0, 1, 2))
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

	y_rpn_regr = np.transpose(y_rpn_regr, (3, 0, 1, 2))
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :y_rpn_overlap, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

	num_pos = len(pos_locs[0])


	# one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
	# regions. We also limit it to 256 regions.
	num_regions = 512
	print('calc_rpn: {} pos anchors, {} neg anchors'.format(len(pos_locs[0]), len(neg_locs[0])))

	if len(pos_locs[0]) > num_regions/2:
		val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs], pos_locs[3][val_locs]] = 0
		num_pos = num_regions/2

	if len(neg_locs[0]) + num_pos > num_regions:
		val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs], neg_locs[3][val_locs]] = 0

	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

	return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return next(self.it)		

	
def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g

def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):

	# The following line is not useful with Python 3.5, it is kept for the legacy
	# all_img_data = sorted(all_img_data)

	sample_selector = SampleSelector(class_count)

	while True:
		if mode == 'train':
			np.random.shuffle(all_img_data)

		for img_data in all_img_data:
			try:

				if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
					continue

				# read in image, and optionally add augmentation

				if mode == 'train':
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)
				else:
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

				(width, height, dense) = (img_data_aug['width'], img_data_aug['height'], img_data_aug['depth'])
				(dens,rows, cols) = x_img.shape

				assert cols == width
				assert rows == height
				assert dens == dense

				# get image dimensions for resizing
				(resized_width, resized_height, resized_dense) = get_new_img_size(width, height, dense, C.im_size)

				# resize the image so that smalles side is length = 600px
				#x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
				x_img = ski.transform.resize(x_img, (resized_dense, resized_height, resized_width))

				try:
					y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, dense,resized_width, resized_height, resized_dense, img_length_calc_function)
				except:
					continue

				# Zero-center by mean pixel, and preprocess image

				#x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
				x_img = x_img.astype(np.float32)
				#x_img[:, :, 0] -= C.img_channel_mean[0]
				#x_img[:, :, 1] -= C.img_channel_mean[1]
				#x_img[:, :, 2] -= C.img_channel_mean[2]
				x_img /= C.img_scaling_factor

				x_img = np.transpose(x_img, (2, 0, 1)) #For RGB image: (3, cols, rows)
				x_img = np.expand_dims(x_img, axis=0)

				y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

				if backend == 'tf':
					x_img = np.transpose(x_img, (0, 2, 3, 1)) #For RGB image: (1, cols, rows, 3)
					y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
					y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

				yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

			except Exception as e:
				print(e)
				continue
