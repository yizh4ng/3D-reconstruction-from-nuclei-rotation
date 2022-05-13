import pickle
data = 'adam'
data = 'three_before_division'
result_path = f"C:/Users/Administrator/Desktop/lambai/11_NR/cell_class/{data}.pkl"
file_path = f"C:/Users/Administrator/Desktop/lambai/11_NR/data/{data}_bf.tif"
from lambo.gui.vinci.vinci import DaVinci
import pims
import os
from rotating_cell import Rotating_Cell

def crop_roi(frames, radius, center):
  new_frames = []
  for i, frame in enumerate(frames):
    c_y, c_x = center[i]
    new_frames.append(frame[int(c_x-radius):int(c_x+radius),
              int(c_y-radius):int(c_y+radius)])
  return new_frames


class ShowROI(DaVinci):
  def __init__(self, radius, center):
    super(ShowROI, self).__init__()
    self.radius = radius
    self.center = center

  def show_ROI(self, x, ax):
    cursor = self.object_cursor
    c_y, c_x = center[cursor]
    ax.imshow(x[int(c_x-self.radius):int(c_x+self.radius),
              int(c_y-self.radius):int(c_y+self.radius)])


if __name__ == '__main__':
  if not os.path.exists(file_path):
    raise FileNotFoundError('!! File `{}` not found.'.format(file_path))
  frames = pims.open(file_path)
  frames = frames[2:17]

  with open(result_path, 'rb') as f:
    rotating_cell = pickle.load(f)

  radius = rotating_cell.radius * 1.2
  # rotation = dict['r']
  center = rotating_cell.center

  cropped_frames = crop_roi(frames, radius, center)
  showroi = ShowROI(radius, center)
  showroi.objects = cropped_frames
  # showroi.add_plotter(showroi.show_ROI)
  showroi.add_plotter(showroi.imshow)
  showroi.show()