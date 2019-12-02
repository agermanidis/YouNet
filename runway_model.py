import matplotlib; matplotlib.use('TkAgg')
import torch
import numpy as np
import runway
import PIL
from fastai.vision import *
from skimage.transform import resize


@runway.setup(options={'checkpoint': runway.file(extension='.pkl')})
def setup(opts):
  return load_learner('.', opts['checkpoint'])


@runway.command('mask', inputs={'image': runway.image, 'threshold': runway.number(default=0.9, min=0, max=1, step=0.001)}, outputs={'image': runway.image})
def mask(learner, inputs):
  inp = inputs['image'].convert('RGB')
  original_size = inp.size
  inp_resized = inp.resize((512, 512))
  im = Image(pil2tensor(inp_resized, np.float32).div(255))
  result = learner.predict(im)
  np_img = result[2].data.numpy()[1]
  np_img[np_img > inputs['threshold']] = 1
  mask = np_img.astype(np.uint8)
  mask = (np_img > 0).reshape((512, 512, 1))
  mask = np.stack((np_img,)*3, axis=-1)
  mask = resize(mask, (original_size[1], original_size[0]), anti_aliasing=False).astype(np.uint8)
  masked = np.array(inp)
  masked[mask == 0] = 0
  return masked


if __name__ == '__main__':
  runway.run(port=7843, model_options={'checkpoint': './512x512_resnet34_2.pkl'})
