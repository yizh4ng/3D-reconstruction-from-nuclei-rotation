from lambo import DaVinci


da = DaVinci()

def show_text(ax, y=None):
  da.show_text(ax, text=f'{y}')

for i in [10, 20, 39]:
  def f(ax, yy=i):
    show_text(ax, y=yy)

  da.add_plotter(f)

da.show()