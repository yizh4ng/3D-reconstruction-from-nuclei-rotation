import pickle
import sys
import os
sys.path.insert(0, "../../lambai")
sys.path.insert(0, '../roma')

from roma import console
from optimize import Trainer
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == '__main__':
  steps = 5
  with open(f'./adam66.pkl', 'rb') as f:
    cell = pickle.load(f)
  console.show_status('Start training...')
  for i in range(steps):
    console.print_progress(i, total=steps)
    Trainer.train(cell)

  console.show_status('Finish training')