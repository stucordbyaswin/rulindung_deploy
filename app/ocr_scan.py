import cv2
import pytesseract
import numpy as np

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


def get_text(img_path):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    print('Tesseract version:', pytesseract.get_tesseract_version())   
    
    print('Processing...')
    # Read image
    img = cv2.imread(img_path)

    # Preprocessing image
    # img = get_grayscale(img)
    img = remove_noise(img)
    # img = thresholding(img)
    # img = dilate(img)

    # Tesseract OCR scanning  
    """
    'oem' (Optical Engine Method) argument options: 
        0    Legacy engine only.
        1    Neural nets LSTM engine only.
        2    Legacy + LSTM engines.
        3    Default, based on what is available.
    """

    """
    'psm' (Page Segmentation Mode) argument options:
        0    Orientation and script detection (OSD) only.
        1    Automatic page segmentation with OSD.
        2    Automatic page segmentation, but no OSD, or OCR.
        3    Fully automatic page segmentation, but no OSD. (Default)
        4    Assume a single column of text of variable sizes.
        5    Assume a single uniform block of vertically aligned text.
        6    Assume a single uniform block of text.
        7    Treat the image as a single text line.
        8    Treat the image as a single word.
        9    Treat the image as a single word in a circle.
        10    Treat the image as a single character.
        11    Sparse text. Find as much text as possible in no particular order.
        12    Sparse text with OSD.
        13    Raw line. Treat the image as a single text line,
    """

    # 'tessedit_char_blacklist' = mengabaikan karakter yang tidak diinginkan. 
    custom_config = r'-l ind+eng -c tessedit_char_blacklist=0123456789 --oem 3 --psm 6'
    text = pytesseract.image_to_string(img, config=custom_config)

    print('Done.')   
    print('Scan results:', text)

    return text 
    