import cv2
from DetectQR_class import QR_decode


if __name__ == "__main__":
	name = str(input("Enter the file name: "))
	version = int(input("Enter the version used: "))
	image = cv2.imread(name)
	QR = QR_decode(image,version = version)
	raw = QR.decode()
	print("Info: ",raw)
	
