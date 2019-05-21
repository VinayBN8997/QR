from GenerateQR_class import QR_generate
if __name__ == "__main__":
	text = str(input("Enter the info to be embedded into the QR code: "))
	name = str(input("Enter the name of the image to be saved as: "))
	QR_generate(text,name)