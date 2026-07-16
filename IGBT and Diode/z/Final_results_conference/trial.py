import qrcode

qr = qrcode.QRCode(box_size=20, border=2)  # box_size high = high-res
qr.add_data("https://github.com/AnirudhKatoch/PEARL")
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")
img.save("Paper_figures/pearl_qr.pdf")