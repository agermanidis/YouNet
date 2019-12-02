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


def inference(learner, input_arr):
  im = Image(pil2tensor(input_arr, np.float32).div(255))
  result = learner.predict(im)
  np_img = result[2].data.numpy()[1]
  return np_img

@runway.command('mask', inputs={'image': runway.image, 'threshold': runway.number(default=0.5, min=0, max=1, step=0.001)}, outputs={'image': runway.image(channels=4)})
def mask(learner, inputs):
  inp = inputs['image'].convert('RGB')
  original_size = inp.size
  inp_resized = inp.resize((512, 512))
  mask1 = inference(learner, inp_resized)
  mask2 = np.fliplr(inference(learner, np.fliplr(inp_resized)))
  mask = (mask1 + mask2) / 2
  mask[mask > inputs['threshold']] = 255
  mask = resize(mask, (original_size[1], original_size[0]), anti_aliasing=False).astype(np.uint8)
  masked = np.concatenate((np.array(inp), np.expand_dims(mask, -1)), axis=2)
  return masked


if __name__ == '__main__':
  runway.run(port=7843, model_options={'checkpoint': './512x512_resnet34_2.pkl'})
