# QR
QR code encoder and decoder

# QR codes: 
#### Quick read codes created by Denso Wave Incorporated.

## 0: White
## 1: Black

Resources:

1. [How to Decode a QR Code by Hand](https://youtu.be/KA8hDldvfv0)
2. [How to Decode a QR Code by Hand | A Step by Step Guide](https://youtu.be/kHTWTyV7VJQ)

The smallest QR codes are 21x21 pixels, and the largest are 177x177. The sizes are called versions. The 21x21 pixel size is version 1, 25x25 is version 2, and so on. The 177x177 size is version 40.
Version = 1 will have no Allignment pattern (Non preferable as it may not be used in future)
Version = 2 to 6 will have 1 allignment pattern

## Error correction level:
QR codes include error correction: when you encode the QR code, you also create some redundant data that will help a QR reader accurately read the code even if part of it is unreadable. There are four levels of error correction that you can choose from. The lowest is L, which allows the code to be read even if 7% of it is unreadable. After that is M, which provides 15% error correction, then Q, which provides 25%, and finally H, which provides 30%.

L (Low = 7%) = 11
M (Medium = 15%) = 10
Q (Quartile = 25%) = 01
H (High = 30%) = 10

Low error correction will help in having maximum amount of data

## Max data as per version using "L" i.e, low error correction level:
Version 2 : 32 characters
Version 3 : 55 characters
Version 4 : 80 characters
Version 5 : 108 characters
Version 6 : 136 characters

A QR code encodes a string of text. The QR code standard has four modes for encoding text: numeric, alphanumeric, byte, and Kanji. Each mode encodes the text as a string of bits (1s and 0s), but each mode uses a different method for converting the text into bits. 

1. Numeric mode is for decimal digits 0 through 9.

2. Alphanumeric mode is for the decimal digits 0 through 9, as well as uppercase letters (not lowercase!), and the symbols $, %, *, +, -, ., /, and : as well as a space. All of the supported characters for alphanumeric mode are listed in the left column of this alphanumeric table.

3. Byte mode, by default, is for characters from the ISO-8859-1 character set. 

4. Kanji mode is for double-byte characters from the Shift JIS character set. While UTF-8 can encode Kanji characters, it must use three or four bytes to do so. Shift JIS, on the other hand, uses just two bytes to encode each Kanji character, so Kanji mode compresses Kanji characters more efficiently. If the entire input string consists of characters in the double-byte range of Shift JIS, use Kanji mode. It is also possible to use multiple modes within the same QR code, as described later on this page.

#### Byte mode is used for the code. 

#### Reed–Solomon error correction for the generation of error correction code.

## Requirements:

Worked with Python 3.x

1. Numpy
2. OpenCV
3. sklearn - Cluster (K-Means)

# To run:
DetectQR.py is for decoding the QR code given.... This uses the class from DetectQR_class.py
Similiarly GenerateQR.py is for encoding the QR code given.... This uses the class from GenerateQR_class.py



