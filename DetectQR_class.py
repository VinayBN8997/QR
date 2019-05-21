'''
QR codes: Quick read codes

Error correction level:
L (Low = 7%) = 11
M (Medium = 15%) = 10
Q (Quartile = 25%) = 01
H (High = 30%) = 10

Main resource:
https://www.thonky.com/qr-code-tutorial

'''

import cv2
import numpy as np
import copy
from sklearn.cluster import KMeans
from math import sqrt,floor
from numpy import linalg as LA

class QR_decode:

	def __init__(self,image,version=2):
		"""
		The code that runs when creating the object of the QR_generate class
		"""
		self.version = version
		self.ratio = 15
		self.modules = 4*self.version + 17
		self.dim = (self.modules*self.ratio,self.modules*self.ratio) #Size of resized image
		self.img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		self.img = cv2.resize(self.img, self.dim, interpolation = cv2.INTER_AREA)		
		retval,self.img = cv2.threshold(self.img,150,255,cv2.THRESH_BINARY) #Binaty threshold the image
		self.corners = self.get_corners() #Get 4 corners of QR image

		#pts = np.array(self.corners, np.int32)
		#pts = pts.reshape((-1,1,2))
		#cv2.polylines(self.img,[pts],True,(0,255,255),1)
		#cv2.imshow('Corner image',self.img)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()	

		self.get_transform() #Geometric transformation of image to get proper image
		self.img = self.__trim_white_spaces(self.img) #Trim off the blank lines
		self.img = cv2.medianBlur(self.img,3) #To remove salt and pepper noise
		retval,self.img = cv2.threshold(self.img,127,255,cv2.THRESH_BINARY) #Binaty threshold the image
		self.original_img = self.img

		res = self.probable_FPs()
		self.FIPs = self.get_centers(res)
		print("FIPs: ",self.FIPs)
		self.APs = self.probable_APs(h_check=1)
		print("APs: ",self.APs)
		print("\n")
		self.draw_patterns()
		self.get_scaled_matrix()
		#print("Corners: ",self.corners)	    
	    

	def __trim_white_spaces(self,input_data):
		"""
		To remove the non QR region
		"""
		img = []
		for i in input_data:
			if np.all(i == 255):
				continue
			else:
				img.append(i)
		img = np.array(img)
		img_2 = []
		for i in range(np.shape(img)[1]):
			if np.all(img[:,i] == 255):
				continue
			else:
				img_2.append(img[:,i])
		temp = np.array(img_2)
		return temp.transpose()

	def __get_corner_pixel(self,canvas_corner, vector, max_distance):
		"""
		To get the corner most black pixel from a specific direction
		"""
		for dist in range(max_distance):
			for x in range(dist + 1):
				coords = (canvas_corner[0] + vector[0] * x, canvas_corner[1] + vector[1] * (dist - x))
				if self.img[coords[1]][coords[0]] == 0:
					return coords


	def get_corners(self):
		"""
		To get the 4 corners of the QR image so that prespective transformation can be achieved 
		"""
		size = len(self.img)
		top_left = self.__get_corner_pixel((0, 0), (1, 1), size)
		bottom_left = self.__get_corner_pixel((size - 1, 0), (-1, 1), size)
		top_right = self.__get_corner_pixel((0, size - 1), (1, -1), size)
		bottom_right = self.__get_corner_pixel((size-1, size - 1), (-1, -1), size)
		return [top_left,top_right,bottom_left,bottom_right]		

	def __get_pattern(self,input_data):
		"""
		To get pattern of the image in terms of white and black successive pixels
		"""
		res = []
		temp = 1
		for i in range(1,len(input_data)):
			if  input_data[i] == input_data[i-1]:
				temp += 1
			else:
				res.append(temp)
				temp = 1
		res.append(temp)
		return res

	def __check_ratio_FP(self,input_data):
		"""
		To check foe 1:1:3:1:1 ratio of finder patterns
		"""
		temp = [i/j for (i,j) in zip(input_data,[1,1,3,1,1])]
		if np.all(np.array(temp) <= (temp[0]+5)) and np.all(np.array(temp) >= (temp[0] - 5)) and len(input_data) == 5 :
			return True
		else:
			return False

	def __check_value(self,pixel,value):
		"""
		Convert image into binary matriz by transforming a box to one value
		"""
		temp = self.img[pixel[0]:pixel[0]+1,pixel[1]:pixel[1]+1]
		size = (np.shape(temp)[0]*np.shape(temp)[1])
		if size > 0:
			val =  int(np.sum(np.sum(temp))/size)
			if value == 0:
				if val < 128:
					return 1
				else:
					return 0
			else:
				if val >= 128:
					return 1
				else:
					return 0 
		else:
			return 0

	def probable_FPs(self):
		"""
		Get all probable values of the Finder patterns of the image
		"""
		no_of_FP = 0
		location_FP=[0,0]
		[rows,columns]=np.shape(self.img)
		for row in range(rows):
			row_pattern = self.__get_pattern(self.img[row,:])
			col = 0	
			for i in range(len(row_pattern)-4):
				row_vector = row_pattern[i:i+5]
				isRowFP = self.__check_ratio_FP(row_vector)
				if isRowFP and self.__check_value([row,col],0):
					col_no = col+sum(row_vector)//2
					row_pixel = 0
					col_pattern = self.__get_pattern(self.img[:,col_no])
					for j in range(len(col_pattern)-4):
						col_vector = col_pattern[j:j+5]
						isColFP = self.__check_ratio_FP(col_vector)
						if isColFP and self.__check_value([row_pixel,col_no],0):
							row_multiple = row_pixel + sum(col_vector)/2
							if abs(row_multiple-row) <= 8: #Error orrection level
								no_of_FP = no_of_FP + 1
								location_FP += [[row,col_no]]
						row_pixel = row_pixel + col_pattern[j]
				col = col + row_pattern[i] 
		location_FP = location_FP[2:]
		return location_FP

	def get_centers(self,input_data):
		"""
		Converting all the probable finder patterns into 3 cluster points for K-means
		"""
		kmeans = KMeans(n_clusters=3, random_state=0).fit(np.array(input_data))
		centers = kmeans.cluster_centers_
		res = []
		for i in centers:
			temp = []
			for j in i:
				temp.append(int(j))
			res.append(temp)
		[A,B,C] = res
		'''
		A------B
		'
		'
		'
		C
		'''
		dis_dict={}
		dis_dict["AB"]=sum(abs(np.array(A)-np.array(B)))
		dis_dict["AC"]=sum(abs(np.array(A)-np.array(C)))
		dis_dict["BC"]=sum(abs(np.array(B)-np.array(C)))
		max_dis = max(list(dis_dict.values()))
		if max_dis == dis_dict["AB"]:
			A_point = C
			B_point = B #Arbitrary
			C_point = A #Arbitrary
		elif max_dis == dis_dict["AC"]:
			A_point = B
			B_point = C #Arbitrary
			C_point = A #Arbitrary		
		else:
			A_point = A
			B_point = B #Arbitrary
			C_point = C #Arbitrary

		AB = np.array(B)-np.array(A)
		AC = np.array(C)-np.array(A)
		k = AB[0]*AC[1] - AB[1]*AC[0]
		if k>0:
			(C_point,B_point) = (B_point,C_point)
		
		return [A_point,B_point,C_point]



	def probable_APs(self,h_check = 1):
		"""
		Finding all probable Allignment patterns
		"""
		location_AP = [0,0]
		no_of_AP = 0
		[A,B,C] = self.FIPs
		Rw = [A[0],B[0],C[0]]
		Cl = [A[1],B[1],C[1]]
		min_Rw = min(Rw)
		max_Rw = max(Rw)
		min_Cl = min(Cl)
		max_Cl = max(Cl)
		AB = np.array(B)-np.array(A)
		AC = np.array(C)-np.array(A)
		dis_1 = sqrt(AC[0]**2 + AC[1]**2)
		dis_2 = sqrt(AB[0]**2 + AB[1]**2)

		cell_width_1 = dis_1/(self.modules-7)
		cell_width_2 = dis_2/(self.modules-7)

		norm_AC = [float(i)/dis_1 for i in AC]
		norm_AB = [float(i)/dis_2 for i in AB]

		near_AP = np.array(norm_AC)*(self.modules-9)*cell_width_1 + np.array(norm_AB)*(self.modules-9)*cell_width_2
		[rows,cols]=np.shape(self.img)
		for row in range(((self.FIPs[1][0] - self.FIPs[0][0])//2 + self.FIPs[0][0]),rows):
			row_pattern = self.__get_pattern(self.img[row,:])
			min_col = (self.FIPs[2][1] - self.FIPs[1][1])//2 + self.FIPs[1][1]
			col = 0
			for i in range(len(row_pattern)-2):
				row_vector = row_pattern[i:i+3]
				isRowAP = self.__check_ratio_AP(row_vector)
				if isRowAP and self.__check_value([row,col],255):
					col_no = col+sum(row_vector)//2
					if col_no > min_col:
						row_pixel = 0
						col_pattern = self.__get_pattern(self.img[:,col_no])
						for j in range(len(col_pattern)-2):
							col_vector = col_pattern[j:j+3]
							isColAP = self.__check_ratio_AP(col_vector)
							if isColAP and self.__check_value([row_pixel,col_no],255):
								row_multiple = row_pixel + sum(col_vector)/2
								if abs(row_multiple-row) <= 4: #Error orrection level
									no_of_AP = no_of_AP + 1
									location_AP += [[row,col_no]]
							row_pixel = row_pixel + col_pattern[j]
				col = col + row_pattern[i] 
		location_AP = location_AP[2:]
		if h_check == 1: #Find the most likely AP point by calculating the distance between the found AP candidates and the calculated nearAP
			location_AP_new = []
			for i in range(len(location_AP)):
				AP = location_AP[i]
				isAP = self.__check_AP(AP)
				if isAP:
					location_AP_new += [AP]
			location_AP = location_AP_new		

		if len(location_AP) > 0:
			distances = np.array([self.__get_euclidian(i,near_AP) for i in location_AP])
			min_distance = min(distances)
			AP = np.array(location_AP[distances is min_distance])
			d2 = np.array([self.__get_euclidian(i,AP) for i in location_AP])
			I = d2[d2 < 1.5]
			B1=[]
			B2=[]
			if len(I) > 1:
				for i in range(len(I)):
					A = location_AP[int(I[i])]
					B1.append(A[0])
					B2.append(A[1])
				B1 = sum(B1)/len(I)
				B2 = sum(B2)/len(I)
				AP = [int(B1),int(B2)]
		else:
			AP = []

		return AP

	def __get_euclidian(self,x,y):
		"""
		Returns Eucidian distance of 2 vectors
		"""   
		return np.sqrt(np.sum((np.array(x)-np.array(y))**2))


	def __check_ratio_AP(self,input_data):
		"""
		To check the 1:1:1 ratio of ALiignment pattern
		"""
		temp = [i/j for (i,j) in zip(input_data,[1,1,1])]
		if np.all(np.array(temp) <= (temp[0]+5)) and np.all(np.array(temp) >= (temp[0] - 5)) and len(input_data) == 3 :
			return True
		else:
			return False	

	def __check_AP(self,AP):
		"""
		To check if the selected AP is with required ratio in diagonal direction as well
		"""
		input_shape = np.shape(self.img)
		initial = min([input_shape[1]-AP[1],input_shape[0]-AP[0],AP[0],AP[1]])
		ans = self.__check_diagonal(AP,floor(initial/2 -1))
		if ans :
			return 1
		else:
			return 0

	def __check_diagonal(self,AP,initial):
		"""
		To check for 1:1:1 in diagonal direction as well for allignment pattern
		"""
		vec = []
		for i in range(-initial,initial+1):
			vec.append(self.img[AP[0]+i,AP[1]+i])
		pattern = self.__get_pattern(vec)
		pos = []
		for k in range(1,len(pattern)-2):
			row = sum(pattern[0:k-1])+1
			vector = pattern[k-1:k+2]
			isAP = self.__check_ratio_AP(vector)
			if isAP and (self.img[AP[0]-initial+row-1][AP[1]-initial+row-1] == 255):
				pos.append(sum(pattern[0:k])+pattern[k]/2)
		m = 1 #error
		if len(pos)>0:
			pos = np.array(pos)
			pos = pos[pos <= (m+1+initial)]
			pos = pos[pos >= (-m+1+initial)]
			if len(pos) > 0:
				return 1
			else:
				return 0		
		else:
			return 0


	def __get_unit_vector(self,input_data):
		input_data = input_data / (input_data**2).sum()**0.5
		return input_data

	def draw_patterns(self):
		"""
		To visualize the Finder patterns and Allignment patterns
		"""
		self.copy_original_img = self.original_img
		self.original_img = cv2.line(self.original_img,tuple(self.FIPs[0]),tuple(self.FIPs[1]),(255,0,0),2)
		self.original_img = cv2.line(self.original_img,tuple(self.FIPs[1]),tuple(self.FIPs[2]),(255,0,0),2)
		self.original_img = cv2.line(self.original_img,tuple(self.FIPs[0]),tuple(self.FIPs[2]),(255,0,0),2)
		if len(self.APs) > 0:
			self.original_img = cv2.line(self.original_img,tuple(self.FIPs[0]),tuple(self.APs),(255,0,0),2)
		cv2.imshow('Pointed image',self.original_img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	
	def __get_norm(self,A,B):
		"""
		Returns the normalised vector of given 2 vector's difference
		"""
		res = (np.array(B)-np.array(A))/LA.norm(np.array(B)-np.array(A))
		return res

	def get_transform(self):
		"""
		Prespective transformation of QR image so that only QR code is cropped
		"""
		tform = cv2.getPerspectiveTransform(np.float32(self.corners),np.float32([[0,0],[0,self.dim[1]],[self.dim[0],0],[self.dim[0],self.dim[1]]]))
		self.img = cv2.warpPerspective(self.img,tform,self.dim)
	
		cv2.imshow('Transformed image',self.img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def get_scaled_matrix(self):
		"""
		To get data matrix from the QR image
		"""
		scaled_matrix = []
		[no_rows,no_cols] = np.shape(self.img)
		for i in range(0,no_rows,self.ratio):
			value = []
			rows = self.img[i:i+self.ratio]
			if len(rows) == self.ratio:
				rows = rows.sum(axis = 0)/self.ratio
				for j in range(0,no_cols,self.ratio):
					if len(rows[j:j+self.ratio]) == self.ratio:
						temp = rows[j:j+self.ratio]
						temp = sum(temp)/len(temp)
						value.append(int(temp > 127)*255)
				scaled_matrix.append(value)
		self.img = scaled_matrix 

	def __extract_mask_pattern(self):
		"""
		To identify the mask pattern used for QR code encoding
		"""
		mask_pattern = self.img[8][2:5]
		#print("Mask Pattern: ",mask_pattern)
		power = 1
		total = 0
		for i in mask_pattern:
			if i == 0:
				total += power
			power <<= 1
		#print("Total: ",total)
		mask_matrix = []
		j = 0
		for row in self.img:
			i = 0
			new_row = []
			for val in self.img[j]:
				if self.__extract_mask_boolean(total,i,j):
					new_row.append(0)
				else:
					new_row.append(1)
				i += 1
			j += 1
			mask_matrix.append(new_row)
		return mask_matrix

	def __extract_mask_boolean(self,number,j,i):
		"""
		Get the mask 
		"""	
		if number == 0:
			return (i * j) % 2 + (i * j) % 3 == 0
		elif number == 1:
			return i % 2 == 0
		elif number == 2:
			return ((i * j) % 3 + i + j) % 2 == 0
		elif number == 3:
			return (i + j) % 3 == 0
		elif number == 4:
			return (i / 2 + j / 3) % 2 == 0
		elif number == 5:
			return (i + j) % 2 == 0
		elif number == 6:
			return ((i * j) % 3 + i * j) % 2 == 0
		elif number == 7:
			return j % 3 == 0
		else:
			raise Exception("Unknown Mask Pattern")

	def __demask(self):
		"""
		Using the mask pattern, the datafields are demasked
		"""
		mask = self.__extract_mask_pattern()
		decoded_matrix = []
		y = 0
		while y < len(self.img):
			row = []
			x = 0
			while x < len(self.img[0]):
				modifyValue = self.img[y][x]
				if modifyValue == 255:
					modifyValue = 1
				row += [(~modifyValue + 2 ^ ~mask[y][x] + 2)]
				x += 1
			decoded_matrix += [row]
			y += 1
		return decoded_matrix	

	def __traverse_matrix(self):
		"""
		Returns the traverse matrix which helps in getting the directions for reading bits individually
		"""
		traversal = []
		x, y, direction = len(self.img) - 1, len(self.img) - 1, -1
		matrix = self.__demask()
		while True:
			if self.__out_of_bounds(x, y):
				direction, y, x = -direction, y - 2, x - direction
			if not self.__in_fixed_area(x, y):
				traversal += [matrix[x][y]]
			if y < 8:
				break
			elif y % 2 != 0:
				x, y = x + direction, y + 1
			else:
				y -= 1
		return traversal

	def __out_of_bounds(self, x, y):
		"""
		To check if a co-ordinate is out of bound which helps in traversing the matrix
		"""
		if x > len(self.img) - 1 or y > len(self.img) - 1:
			return True
		elif x < 0 or y < 0:
			return True
		elif x < 9 and (y < 9 or y >= len(self.img) - 8):
			return True
		elif x < 9 and y >= len(self.img) - 8:
			return True
		else:
			return False

	def __in_fixed_area(self,x,y):
		if self.__within_orientation_markers(x,y):
			return True
		elif x == 6 or y == 6:
			return True

	def __within_orientation_markers(self,x,y):
		return x in range(len(self.img) - 10 + 1, len(self.img) - 5 + 1) and y in range(len(self.img) - 10 + 1, len(self.img) - 5 + 1)

	def __decode_bits(self,traversal, start, number_of_bits=8):
		"""
		Decoding the bits into its numerical value
		"""
		factor = 2 << (number_of_bits - 2)
		character = 0
		for i in traversal[start:start + number_of_bits]:
			character += i * factor
			if factor == 1:
				return int(character)
			factor /= 2

	def decode(self):
		"""
		The main fuction for calling all the fucntions sequentially and decode the complete bits
		"""
		zig_zag_traversal = self.__traverse_matrix()
		word = ""
		enc = self.__decode_bits(zig_zag_traversal, 0,4) 
		length = self.__decode_bits(zig_zag_traversal, 4)
		print("Enc: ",enc)
		print("Len: ",length)
		byte = 8
		for i in range(length):
			temp = self.__decode_bits(zig_zag_traversal, 12 + i * byte)
			if temp:
				word += chr(temp)
		return word