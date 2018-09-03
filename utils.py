import qrcode
from pyzbar.pyzbar import decode

def get_qrcode(text):
	qr = qrcode.QRCode(
    version = 1,
    error_correction = qrcode.constants.ERROR_CORRECT_H,
    box_size = 10,
    border = 4,
	)

	# The data that you want to store
	data = "Nataly"

	# Add data
	qr.add_data(data)
	qr.make(fit=True)

	# Create an image from the QR Code instance
	qr_img = qr.make_image()
	return qr_img

def read_qrcodes(img):
	return decode(image)[0].data