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


@runway.command('mask', inputs={'image': runway.image}, outputs={'segmented': runway.segmentation(label_to_id={'background': 0, 'person': 1}, label_to_color={'background': [0,0,0], 'person': [255, 255, 255]})})
def mask(learner, inputs):
  inp = inputs['image'].convert('RGB')
  original_size = inp.size
  inp_resized = inp.resize((512, 512))
  im = Image(pil2tensor(inp_resized, np.float32).div(255))
  result = learner.predict(im)
  np_img = image2np(result[1].data)
  mask = np_img.astype(np.uint8)
  # mask = (np_img > 0).reshape((512, 512, 1))
  # mask = np.stack((np_img,)*3, axis=-1)
  # mask = resize(mask, (original_size[1], original_size[0]), anti_aliasing=False).astype(np.uint8)
  # masked = np.array(inp)
  # masked[mask == 0] = 0

  return PIL.Image.fromarray(mask).resize(original_size)


if __name__ == '__main__':
  runway.run(port=7843, model_options={'checkpoint': './512x512_resnet34_2.pkl'})
