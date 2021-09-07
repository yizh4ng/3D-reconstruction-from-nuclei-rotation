from PIL import Image

# print(np.arange(len(df['frame']) - 1).shape)
im = Image.open('data.tif')

im.save('data.gif',save_all=True)  # or 'test.tif'