from PIL import Image

# print(np.arange(len(df['frame']) - 1).shape)
im = Image.open('simulate_rotation_3.gif')

im.save('simulate_rotation_3.tiff',save_all=True)  # or 'test.tif'