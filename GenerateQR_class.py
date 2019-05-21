'''
QR codes: Quick read codes
We will be considering only till version 6 as its enoughfor having our information and helping in faster procesing as well.

0: White
1: Black

Version = 1 will have no Allignment pattern (Non preferable as it may not be used in future)
Version = 2 to 6 will have 1 allignment pattern

Error correction level:
L (Low = 7%) = 11
M (Medium = 15%) = 10
Q (Quartile = 25%) = 01
H (High = 30%) = 10

Low error correction will help in having maximum amount of data

'''

import math
import numpy as np
import cv2
import copy
from Lost_point import lost_point

class QR_generate():

	def __init__(self,text,name):
		"""
		The code that runs when creating the object of the QR_generate class
		"""
		self.text=text
		self.name=name
		self.length = len(self.text)
		self.version = self.__get_best_version()
		print("Version used: ",self.version)
		self.size = self.__get_size()
		self.__initialize() #Makes the formated matrix and encodes the data
		self.matrix,mask_pattern = self.__finalize(self.final_data) #Constructs the QR code with the encoded final data
		self.width = 10
		self.border = 30
		self.__generate_image()#Write a JPG image using OpenCV

	def __initialize(self):
		"""
		1. Constructs the blank matrix for required size
		2. Encodes the data for Error correction with "L" level
		3. Sets the Finder patterns, Allignment patterns, Timing patterns and saves space for Format information
		"""
		self.matrix = [None] * self.size
		self.__get_log_values()
		for row in range(self.size):
			self.matrix[row] = [None] * self.size
		max_len = self.__get_max_length()
		data = self.__get_data(self.text,max_len)
		mpoly = self.__get_mpoly(data)
		gpoly = self.__get_gploy()
		self.final_data = self.__get_final_data(mpoly,gpoly)
		self.__set_FIP(FP_num = 1)
		self.__set_FIP(FP_num = 2)
		self.__set_FIP(FP_num = 3)
		self.__set_AP()
		self.__fill_format_info_area()
		self.__set_TP()

	def __generate_image(self):
		"""
		To write the image from the matrix
		"""
		self.img = np.ones((self.size*self.width+self.border,self.size*self.width+self.border,1), np.uint8)*255
		for i in range(len(self.matrix)):
			for j in range(len(self.matrix)):
				if self.matrix[j][i] == 1:
					self.img = cv2.rectangle(self.img,(i*self.width+int(self.border/2),j*self.width+int(self.border/2))
						,(i*self.width+self.width+int(self.border/2),j*self.width+self.width+int(self.border/2)),(0,0,0),-1)
		if '.' in self.name:
			cv2.imwrite(self.name,self.img)
		else:
			cv2.imwrite(self.name+'.jpg',self.img)
		cv2.imshow("Image",self.img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		

	def __get_log_values(self):
		"""
		Log and Anti-log values used for error correction
		"""
		self.EXP_TABLE = list(range(256))
		self.LOG_TABLE = list(range(256))

		for i in range(8):
			self.EXP_TABLE[i] = 1 << i

		for i in range(8,256):
			self.EXP_TABLE[i] = (self.EXP_TABLE[i-4] ^ self.EXP_TABLE[i-5] ^ self.EXP_TABLE[i-6] ^ self.EXP_TABLE[i-8])

		for i in range(255):
			self.LOG_TABLE[self.EXP_TABLE[i]] = i

	def __get_best_version(self):
		"""
		Finiding the best version size for the given text
		"""
		if self.length < 32:
			return 2 # version 2
		elif self.length < 53:
			return 3 # version 3
		elif self.length < 78:
			return 4 # version 4
		elif self.length < 106:
			return 5 # version 5
		elif self.length < 134:
			return 6 # version 6
		else:
			return "Too long data"
		

	def __get_max_length(self):
		"""
		Returns the maximum size for the given version in number of characters. For bits, multiply by 8
		Ref: https://www.qrcode.com/en/about/version.html
		"""
		if self.version == 1:
			return 19
		elif self.version == 2:
			return 34
		elif self.version == 3:
			return 55
		elif self.version == 4:
			return 80
		elif self.version == 5:
			return 108
		elif self.version == 6:
			return 136
		else:
			return "Version number > 6 not supported"

	def __get_EC_numner(self):
		"""
		Returns the number of error correction codewords
		"""
		if self.version == 1:
			return 7
		elif self.version == 2:
			return 10
		elif self.version == 3:
			return 15
		elif self.version == 4:
			return 20
		elif self.version == 5:
			return 26
		elif self.version == 6:
			return 18
		else:
			return "Version number > 6 not supported"


	 
	def __get_AP_position(self):
		"""
		Return the position of AP for the given version
		"""
		AP_position_table = [
						None,#Version 1
						18,#Version 2
						22,#Version 3
						26,#Version 4
						30,#Version 5
						34,#Version 6
						] 
		return AP_position_table[self.version - 1]

	def __get_mask_func(self,pattern):
	    """
	    Return the mask function for the given mask pattern.
	    """
	    if pattern == 0:   # Mask pattern : 000
	        return lambda i, j: (i + j) % 2 == 0
	    if pattern == 1:   # Mask pattern : 001
	        return lambda i, j: i % 2 == 0
	    if pattern == 2:   # Mask pattern : 010
	        return lambda i, j: j % 3 == 0
	    if pattern == 3:   # Mask pattern : 011
	        return lambda i, j: (i + j) % 3 == 0
	    if pattern == 4:   # Mask pattern : 100
	        return lambda i, j: (math.floor(i / 2) + math.floor(j / 3)) % 2 == 0
	    if pattern == 5:  # Mask pattern : 101
	        return lambda i, j: (i * j) % 2 + (i * j) % 3 == 0
	    if pattern == 6:  # Mask pattern : 110
	        return lambda i, j: ((i * j) % 2 + (i * j) % 3) % 2 == 0
	    if pattern == 7:  # Mask pattern : 111
	        return lambda i, j: ((i * j) % 3 + (i + j) % 2) % 2 == 0

	def __get_size(self):
		"""
		Return the size of matrix depending on version
		"""
		return 4*self.version + 17

	def __set_FIP(self,FP_num):
		"""
		FP1--------FP2
		|
		|
		|
		|
		FP3
		
		Returns the matrix by setting the FIP in required position according to (row,col)
		"""
		size = len(self.matrix)
		if FP_num == 1:
			[row,col] = [0,0]
		elif FP_num == 2:
			[row,col] = [0,size-7]
		elif FP_num == 3:
			[row,col] = [size-7,0]

		for r in range(7):
			for c in range(7):
				if (0 <= r and r <= 6 and (c ==0 or c == 6) or (0 <= c and c <= 6 and (r == 0 or r == 6))
					or (2 <= r and r <= 4 and 2 <= c and c <= 4)):
					self.matrix[row+r][col+c] = 1
				else:
					self.matrix[row+r][col+c] = 0

		
		if FP_num == 1:
			self.matrix[7][0:8] = [0] * 8
			for i in range(0,8):
				self.matrix[i][7] = 0
		elif FP_num == 2:
			self.matrix[7][size-8:size] = [0] * 8
			for i in range(0,8):
				self.matrix[i][size-8] = 0
		elif FP_num == 3:
			self.matrix[size-8][0:8] = [0] * 8
			for i in range(size-8,size):
				self.matrix[i][7] = 0

	def __set_AP(self):
		"""
		Returns the matrix by setting the FIP in required position according to version
		"""
		if self.version == 1:
			return 0
		loc = self.__get_AP_position()
		for r in range(-2,3):
			for c in range(-2,3):
				if (r == -2 or r == 2 or c == -2 or c == 2 or (r == 0 and c == 0)):
					self.matrix[loc+r][loc+c] = 1
				else:
					self.matrix[loc+r][loc+c] = 0

	def __set_TP(self):
		"""
		Returns the matrix by setting the timing pattern in required positions
		"""	
		for r in range(8,self.size - 8):
			self.matrix[r][6] = int(r % 2 == 0)

		for c in range(8,self.size - 8):
			self.matrix[6][c] = int(c % 2 == 0)

		self.matrix[self.size-8][8] = 1

	def __get_data(self,text,max_len,mode = "byte"):
		"""
		Return the data in encoded format
		"""

		pad_bits = ["11101100", "00010001"]

		if mode == "byte":
			data = []
			data += "0100" #FOr Byte mode
			data += format(len(text),'08b')
			data += [format(ord(i),'08b') for i in text]# Byte encoding of data
			data = "".join(data)
			dist = max_len*8 - len(data)
			if dist >= 4:
				data += "0000" #For 4 white boxes suggesting the end of the end of data
			elif dist == 3:
				data += "000"
			elif dist == 2:
				data += "00"
			elif dist == 1:
				data += "0"
			else:
				data += ""
			data += ""*(8 - (len(data)%8))

			i = 0
			while(len(data) < (max_len*8)):
				data += pad_bits[i%2]
				i += 1

			return data


	def __get_mpoly(self,data):
		return [int(data[i:i+8],2) for i in range(0, len(data), 8)]

	def __get_gploy(self):
		"""
		Ref: https://www.thonky.com/qr-code-tutorial/generator-polynomial-tool?degree=15
		"""
		if self.version == 1:#No.of EC codewaords = 7
			return [0,87,229,146,149,238,102,21]
		elif self.version == 2:#No.of EC codewaords = 10
			return [0,251,67,46,61,118,70,64,94,32,45]	
		elif self.version == 3:#No.of EC codewaords = 15
			return [0,8,183,61,91,202,37,51,58,58,237,140,124,5,99,105]
		elif self.version == 4:#No.of EC codewaords = 20
			return [0,17,60,79,50,61,163,26,187,202,180,221,225,83,239,156,164,212,212,188,190]
		elif self.version == 5:#No.of EC codewaords = 26
			return [0,173,125,158,2,103,182,118,17,145,201,111,28,165,53,161,21,245,142,13,102,48,227,153,145,218,70]
		elif self.version == 6:#No.of EC codewaords = 18
			return [0,215,234,158,94,184,97,118,170,79,187,152,148,252,179,5,98,96,153]
		else:
			return "Only upto version 6 is supported"


	def __longdivision(self,steps,mpoly,gpoly):
		"""
		Long polynomial division
		"""
		for i in range(0,steps):
			gap=len(mpoly)-len(gpoly)
			m=mpoly[0]
			m=self.LOG_TABLE[m]
			if gap>0:
				newgpoly=[self.EXP_TABLE[(g+m)%255] for g in gpoly]+[0]*gap
			else:
				newgpoly=[self.EXP_TABLE[(g+m)%255] for g in gpoly]
			blank=[]
			if gap<0:
				mpoly=mpoly+[0]*abs(gap)
			for i in range(0,len(newgpoly)):
				b=[(mpoly[i]^newgpoly[i])]
				blank=blank+b
			mpoly=np.trim_zeros(blank,trim='f')
		return mpoly


	def __get_final_data(self,mpoly,gpoly):
		"""
		Finalising the data by combining all encoded format, length, data and error correction data.
		"""	
		steps = len(mpoly)
		ecwords = self.__longdivision(steps,mpoly,gpoly)
		message = mpoly + ecwords
		if self.version == 1:
			rem = 0
		else:
			rem = 7
		message = ['{0:08b}'.format(i) for i in message]+['0']*rem
		message = "".join(message)
		return message

	def __fill_format_info_area(self):
		"""
		For setting the format information area with non-Null values. Can be anything (Example: 5)
		"""
		size = len(self.matrix)
		for i in range(0,9):
			self.matrix[8][i] = 5
		for i in range(size-8,size):
			self.matrix[8][i] = 5
		for i in range(0,9):
			self.matrix[i][8] = 5
		for i in range(size-8,size):
			self.matrix[i][8] = 5
		self.matrix[size-8][8] = 1

	def __get_format_info_locations(self,size):
		"""
		The cooerdinate for format information. Totally 15 bits 
		"""
		loc = {
				0: [[8,0],[size-1,8]],
				1: [[8,1],[size-2,8]],
				2: [[8,2],[size-3,8]],
				3: [[8,3],[size-4,8]],
				4: [[8,4],[size-5,8]],
				5: [[8,5],[size-6,8]],
				6: [[8,7],[size-7,8]],
				7: [[8,8],[8,size-8]],
				8: [[7,8],[8,size-7]],
				9: [[5,8],[8,size-6]],
				10: [[4,8],[8,size-5]],
				11: [[3,8],[8,size-4]],
				12: [[2,8],[8,size-3]],
				13: [[1,8],[8,size-2]],
				14: [[0,8],[8,size-1]]
		}
		return loc

	def __set_format_info(self,input_data,mask_pattern):
		"""
		Error corrention level : L
		Ref: https://www.thonky.com/qr-code-tutorial/format-version-tables
		"""
		mask_info = {
					0:"111011111000100",
					1:"111001011110011",
					2:"111110110101010",
					3:"111100010011101",
					4:"110011000101111",
					5:"110001100011000",
					6:"110110001000001",
					7:"110100101110110"
		}
		size = len(input_data)
		locations = self.__get_format_info_locations(size)
		for i in range(15):
			input_data[locations[i][0][0]][locations[i][0][1]] = int(mask_info[mask_pattern][i])
			input_data[locations[i][1][0]][locations[i][1][1]] = int(mask_info[mask_pattern][i])
		return input_data

		
	def __toggle(self,x):
		"""
		Flip the binary value
		"""
		if x == 1:
			return 0
		else:
			return 1

	def __fill_data(self,input_data,data,mask_pattern):
		"""
		Fill the data values into the matrix
		"""
		size = len(input_data)
		up = False
		data_index = 0
		mask_func = self.__get_mask_func(mask_pattern) #Get the mask function based on mask pattern
		for col in range(size-1,-1,-2):
			up = not up
			if up:
				if col >= size-8:
					row = size-1
					while row >= 9:
						if input_data[row][col] is None:
							input_data[row][col] = int(data[data_index])
							data_index += 1
							if mask_func(row,col):
								input_data[row][col] = self.__toggle(input_data[row][col])
						if col % 2 == 0:
							col = col - 1
						else:
							col = col + 1
							row = row - 1
				elif col >= 8:
					row = size-1
					while row >= 0:
						if input_data[row][col] is None:
							input_data[row][col] = int(data[data_index])
							data_index += 1
							if mask_func(row,col):
								input_data[row][col] = self.__toggle(input_data[row][col])
						if col % 2 == 0:
							col = col - 1
						else:
							col = col + 1
							row = row - 1
				else:
					row = size-9
					while row >= 9:
						if input_data[row][col] is None:
							input_data[row][col] = int(data[data_index])
							data_index += 1
							if mask_func(row,col):
								input_data[row][col] = self.__toggle(input_data[row][col])
						if col % 2 == 0:
							col = col - 1
						else:
							col = col + 1
							row = row - 1


			else:
				if col >= size-8:
					row = 9
					while row <= size-1:
						if input_data[row][col] is None:
							input_data[row][col] = int(data[data_index])
							data_index += 1
							if mask_func(row,col):
								input_data[row][col] = self.__toggle(input_data[row][col])
						if col % 2 == 0:
							col = col - 1
						else:
							col = col + 1
							row = row + 1
				elif col >= 8:
					row = 0
					while row <= size-1:
						if input_data[row][col] is None:
							input_data[row][col] = int(data[data_index])
							data_index += 1
							if mask_func(row,col):
								input_data[row][col] = self.__toggle(input_data[row][col])
						if col % 2 == 0:
							col = col - 1
						else:
							col = col + 1
							row = row + 1			
				else:
					row = 9
					while row <= size-9:
						if input_data[row][col] is None:
							input_data[row][col] = int(data[data_index])
							data_index += 1
							if mask_func(row,col):
								input_data[row][col] = self.__toggle(input_data[row][col])
						if col % 2 == 0:
							col = col - 1
						else:
							col = col + 1
							row = row + 1
		return input_data

	def __finalize(self,final_data):
		"""
		Finding the best mask pattern by trying out all 8 possibilitiees and keeping the one with minimum penalty 
		"""
		copy_input_data = copy.deepcopy(self.matrix)
		best_matrix = self.__set_format_info(copy_input_data,0)
		best_matrix = self.__fill_data(best_matrix,final_data,0)
		min_penalty = lost_point(best_matrix)
		best_mask_pattern = 0
		for i in range(1,8):
			copy_input_data = copy.deepcopy(self.matrix)
			temp_matrix = self.__set_format_info(copy_input_data,i)
			temp_matrix = self.__fill_data(temp_matrix,final_data,i)
			penalty = lost_point(temp_matrix)

			if penalty < min_penalty:
				best_matrix = copy.deepcopy(temp_matrix)
				best_mask_pattern = i
				min_penalty = penalty

		return best_matrix,best_mask_pattern

