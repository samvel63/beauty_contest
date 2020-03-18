# imgs_path = ['data_predict/unknow/1.jpg',
#              'data_predict/unknow/2.jpg',
#              'data_predict/unknow/3.jpg',
#              'data_predict/faces/face_0.jpg',
#              'data_predict/faces/face_1.jpg']
# imgs = []
#
# for img_path in imgs_path:
#     img_raw = tf.io.read_file(os.path.join(PATH, img_path))
#
#     img_tensor = tf.image.decode_image(img_raw)
#     img_tensor = tf.image.resize(img_tensor, [IMG_HEIGHT, IMG_WIDTH])
#     img_tensor /= 255.0
#
#     img = np.reshape(img_tensor, [1, IMG_HEIGHT, IMG_WIDTH, 3])
#     imgs.append(img_tensor)
#
#     classes = model.predict_classes(img)
#     print(img_path, classes[0])
