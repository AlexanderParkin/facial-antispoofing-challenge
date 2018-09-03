import qrcode
from pyzbar.pyzbar import decode
from PIL import Image

def get_qrcode(text):
    qr = qrcode.QRCode(
    version = 1,
    error_correction = qrcode.constants.ERROR_CORRECT_H,
    box_size = 10,
    border = 4,
    )

    # The data that you want to store
    data = text

    # Add data
    qr.add_data(data)
    qr.make(fit=True)

    # Create an image from the QR Code instance
    qr_img = qr.make_image()
    return qr_img

def read_qrcodes(img):
    return decode(image)[0].data

def add_qrcode_in_image(img, qr_text):
    qr_img = get_qrcode(qr_text)

    new_img = Image.new('RGB', 
                        (img.size[0] + qr_img.size[0], img.size[1]), 
                        (255,255,255))

    new_img.paste(img, box=(qr_img.size[0],0))
    new_img.paste(qr_img, box=(0,0))

    return new_img