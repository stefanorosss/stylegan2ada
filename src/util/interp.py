import argparse
import numpy as np
import PIL.Image

import re
import sys
from io import BytesIO
import IPython.display
import numpy as np
from math import ceil
from PIL import Image, ImageDraw
import imageio
import os
import pickle
from google.colab import files

# Generates a list of images, based on a list of latent vectors (Z), and a list (or a single constant) of truncation_psi's.
def generate_images_in_w_space(dlatents, truncation_psi):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi
    dlatent_avg = Gs.get_var('dlatent_avg') # [component]

    imgs = []
    for row, dlatent in log_progress(enumerate(dlatents), name = "Generating images"):
        #row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(truncation_psi, [-1, 1, 1]) + dlatent_avg
        dl = (dlatent-dlatent_avg)*truncation_psi   + dlatent_avg
        row_images = Gs.components.synthesis.run(dlatent,  **Gs_kwargs)
        imgs.append(PIL.Image.fromarray(row_images[0], 'RGB'))
    return imgs       

def generate_images(zs, truncation_psi):
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if not isinstance(truncation_psi, list):
        truncation_psi = [truncation_psi] * len(zs)
        
    imgs = []
    for z_idx, z in log_progress(enumerate(zs), size = len(zs), name = "Generating images"):
        Gs_kwargs.truncation_psi = truncation_psi[z_idx]
        noise_rnd = np.random.RandomState(1) # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        imgs.append(PIL.Image.fromarray(images[0], 'RGB'))
    return imgs


# Generates a list of images, based on a list of seed for latent vectors (Z), and a list (or a single constant) of truncation_psi's.
def generate_images_from_seeds(seeds, truncation_psi):
    return generate_images(generate_zs_from_seeds(seeds), truncation_psi)

def saveImgs(imgs, location):
  for idx, img in log_progress(enumerate(imgs), size = len(imgs), name="Saving images"):
    file = location+ str(idx) + ".png"
    img.save(file)

def imshow(a, format='png', jpeg_fallback=True):
  a = np.asarray(a, dtype=np.uint8)
  str_file = BytesIO()
  PIL.Image.fromarray(a).save(str_file, format)
  im_data = str_file.getvalue()
  try:
    disp = IPython.display.display(IPython.display.Image(im_data))
  except IOError:
    if jpeg_fallback and format != 'jpeg':
      print ('Warning: image was too large to display in format "{}"; '
             'trying jpeg instead.').format(format)
      return imshow(a, format='jpeg')
    else:
      raise
  return disp

def showarray(a, fmt='png'):
    a = np.uint8(a)
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

        
def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))
    
def drawLatent(image,latents,x,y,x2,y2, color=(255,0,0,100)):
  buffer = PIL.Image.new('RGBA', image.size, (0,0,0,0))
   
  draw = ImageDraw.Draw(buffer)
  cy = (y+y2)/2
  draw.rectangle([x,y,x2,y2],fill=(255,255,255,180), outline=(0,0,0,180))
  for i in range(len(latents)):
    mx = x + (x2-x)*(float(i)/len(latents))
    h = (y2-y)*latents[i]*0.1
    h = clamp(h,cy-y2,y2-cy)
    draw.line((mx,cy,mx,cy+h),fill=color)
  return PIL.Image.alpha_composite(image,buffer)
             
  
def createImageGrid(images, scale=0.25, rows=1):
   w,h = images[0].size
   w = int(w*scale)
   h = int(h*scale)
   height = rows*h
   cols = ceil(len(images) / rows)
   width = cols*w
   canvas = PIL.Image.new('RGBA', (width,height), 'white')
   for i,img in enumerate(images):
     img = img.resize((w,h), PIL.Image.ANTIALIAS)
     canvas.paste(img, (w*(i % cols), h*(i // cols))) 
   return canvas

def convertZtoW(latent, truncation_psi=0.5, truncation_cutoff=18):
  dlatent = Gs.components.mapping.run(latent, None) # [seed, layer, component]
  dlatent_avg = Gs.get_var('dlatent_avg') # [component]
  for i in range(truncation_cutoff):
    dlatent[0][i] = (dlatent[0][i]-dlatent_avg)*truncation_psi + dlatent_avg
    
  return dlatent

def interpolate(zs, steps):
   out = []
   for i in range(len(zs)-1):
    for index in range(steps):
     fraction = index/float(steps) 
     out.append(zs[i+1]*fraction + zs[i]*(1-fraction))
   return out

# Taken from https://github.com/alexanderkuk/log-progress
def log_progress(sequence, every=1, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )


# added functions 2021

# extraction function : video to stills (helpful for larger interpolations that OOM)
# does not require the location content/out to exist 


# def generate_images_stills(zs, truncation_psi, location='out', prefix=''):

#     Gs_kwargs = dnnlib.EasyDict()
#     Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
#     Gs_kwargs.randomize_noise = False
#     if not isinstance(truncation_psi, list):
#         truncation_psi = [truncation_psi] * len(zs)
#     path = "/content/"+location
#     os.makedirs(path, exist_ok=True)
    
#     for z_idx, z in log_progress(enumerate(zs), size = len(zs), name = "Generating images"):
#         Gs_kwargs.truncation_psi = truncation_psi[z_idx]
#         noise_rnd = np.random.RandomState(1) # fix noise
#         tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
#         images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        
   
#         image_save = PIL.Image.fromarray(images[0], 'RGB')
        
#         file = path + '/' + prefix + str(z_idx).zfill(6) + ".jpg"
#         image_save.save(file)
        
#     return



import subprocess
from opensimplex import OpenSimplex
import os

def generate_zs_from_seeds(seeds, squeeze=True):
    zs = []
    for seed_idx, seed in enumerate(seeds):
        rnd = np.random.RandomState(seed)
        np.random.seed(seed)
        if squeeze:
            z = rnd.randn(1,512) # [minibatch, component]
        else:
            z = rnd.randn(1, 1,512) # [minibatch, component]
        zs.append(z)
    return zs


def truncation_walk(outdir='./', seed=[0],start=-1.0,stop=1.0,increment=0.02,framerate=24):

    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }

    count = 1
    trunc = start
    steps = int((stop-start)//increment)
    images = []
    rnd = np.random.RandomState(seed)
    np.random.seed(seed)
    z = rnd.randn(1, *Gs.input_shape[1:])
    for step in log_progress(range(steps), name='Generating truncated images'):
        Gs_kwargs['truncation_psi'] = trunc
        #print('Generating truncation %0.2f' % trunc)

        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        image = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        images.append(image[0])
        PIL.Image.fromarray(image[0], 'RGB').save(f'{outdir}/frame{count:05d}.png')

        trunc+=increment
        count+=1

    cmd=f"ffmpeg -y -r {framerate} -i {outdir}/frame%05d.png -vcodec libx264 -pix_fmt yuv420p {outdir}/truncation-walk-seed{seed}-start{start}-stop{stop}.mp4"
    subprocess.call(cmd, shell=True)

# OpenSimplexNoise Interpolator
# diameter <= 1, with one being probably the best value
# Z space only
class OSNI():

    def __init__(self,seed,diameter = 1, min = -1, max = 1):
        self.core = OpenSimplex(seed)
        self.seed = seed
        self.d = diameter
        self.x = 0
        self.y = 0
        self.min = min
        self.max = max

    def valmap(self, value, istart, istop, ostart, ostop):
        return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))

    def get_val(self,angle):
        self.xoff = self.valmap(np.cos(angle), self.min, self.max, self.x, self.x + self.d);
        self.yoff = self.valmap(np.sin(angle), self.min, self.max, self.y, self.y + self.d);
        return self.core.noise2d(self.xoff,self.yoff)

    def get_3Dval(self,angle, zdim):
        self.xoff = self.valmap(np.cos(angle), self.min, self.max, self.x, self.x + self.d);
        self.yoff = self.valmap(np.sin(angle), self.min, self.max, self.y, self.y + self.d);
        return self.core.noise3d(self.xoff,self.yoff,zdim)

    # Circular random walk 
    def get_noiseloop_seeds(self, nf):
        zs = []

        inc = (np.pi*2)/nf
        for f in range(nf):
          z = np.zeros((1, 512))
          for i in range(512):
            self.core = OpenSimplex(self.seed+i)
            z[0,i] = self.get_val(inc*f)
          zs.append(z)
        return zs
    # Random Walk across the 3rd dimension of the noise
    def get_noiseloop_3D(self, nf):
        zs = []

        inc = (np.pi*2)/nf
        for f in range(nf):
          z = np.zeros((1, 512))
          for i in range(512):
            z[0,i] = self.get_3Dval(inc*f,i)
          zs.append(z)

        return zs

    # Elliptical random
    def get_noiseloop_seeds3D(self, nf):
        zs = []

        inc = (np.pi*2)/nf
        for f in range(nf):
          z = np.zeros((1, 512))
          for i in range(512):
            self.core = OpenSimplex(self.seed+i)
            if i < 256:
              z[0,i] = self.get_3Dval(inc*f,i)
            else:
              z[0,i] = self.get_3Dval(inc*f,512-(i+1))
          zs.append(z)

        return zs

      
    # Elliptical random
    def get_noiseloop_seeds3D_loop(self, nf):
        zs = []

        inc = (np.pi*2)/nf
        for f in range(nf):
          z = np.zeros((1, 512))
          for i in range(512):
            
            if i < 256:
              self.core = OpenSimplex(self.seed+i)
              z[0,i] = self.get_3Dval(inc*f,i)
            else:
              self.core = OpenSimplex(512 -i +self.seed)
              z[0,i] = self.get_3Dval(inc*f,512-(i+1))
          zs.append(z)

        return zs
# ----------------------------------------------------------------------
def circular_interpolation(radius, latents_persistent, latents_interpolate):
    latents_a, latents_b, latents_c = latents_persistent

    latents_axis_x = (latents_a - latents_b).flatten() / np.linalg.norm(latents_a - latents_b)
    latents_axis_y = (latents_a - latents_c).flatten() / np.linalg.norm(latents_a - latents_c)

    latents_x = np.sin(np.pi * 2.0 * latents_interpolate) * radius
    latents_y = np.cos(np.pi * 2.0 * latents_interpolate) * radius

    latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
    return latents

# Diameter values need to be high (>80) or not much is going to happen
def get_circularloop(nf, d = 120, seed = 42):
    r = d/2
    if seed:
        np.random.RandomState(seed)
        np.random.seed(seed)
    zs = []

    rnd = np.random
    latents_a = rnd.randn(1, Gs.input_shape[1])
    latents_b = rnd.randn(1, Gs.input_shape[1])
    latents_c = rnd.randn(1, Gs.input_shape[1])
    latents = (latents_a, latents_b, latents_c)

    current_pos = 0.0
    step = 1./nf
    
    while(current_pos < 1.0):
        zs.append(circular_interpolation(r, latents, current_pos))
        current_pos += step
    return zs

# It doesn't always work but it's funny to try, better with low diameter (<20)
# Possibly interesting to use a double slerp instead
def get_circularloop_in_w(nf, d, seed, w_vectors=3):
    if w_vectors > 3:
        print("Only three vectors are being used for the loop...")
    r = d/2
    if seed:
        np.random.RandomState(seed)

    ws = []

    rnd = np.random
    if w_vectors >= 3:
      latents_a = convertZtoW(rnd.randn(1, Gs.input_shape[1]), truncation_psi=0.5)[:,0,:]
      latents_b = convertZtoW(rnd.randn(1, Gs.input_shape[1]), truncation_psi=0.5)[:,0,:]
      latents_c = convertZtoW(rnd.randn(1, Gs.input_shape[1]), truncation_psi=0.5)[:,0,:]
    elif w_vectors == 2:
      latents_a = convertZtoW(rnd.randn(1, Gs.input_shape[1]), truncation_psi=0.5)[:,0,:]
      latents_b = convertZtoW(rnd.randn(1, Gs.input_shape[1]), truncation_psi=0.5)[:,0,:]
      latents_c = rnd.randn(1, Gs.input_shape[1])
    elif w_vectors == 1:
      latents_a = convertZtoW(rnd.randn(1, Gs.input_shape[1]), truncation_psi=0.5)[:,0,:]
      latents_b = rnd.randn(1, Gs.input_shape[1])
      latents_c = rnd.randn(1, Gs.input_shape[1])
    else:
      latents_a = rnd.randn(1, Gs.input_shape[1])
      latents_b = rnd.randn(1, Gs.input_shape[1])
      latents_c = rnd.randn(1, Gs.input_shape[1]) 

    latents = (latents_a, latents_b, latents_c)

    current_pos = 0.0
    step = 1./nf
    
    while(current_pos < 1.0):
        zs = circular_interpolation(r, latents, current_pos)
        ws.append(np.repeat(zs,18,axis=0))
        current_pos += step
    return np.expand_dims(ws,1)



#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Adapted from https://github.com/torresjrjr
class Bezier():
    def TwoPoints(t, P1, P2):
        """
        Returns a point between P1 and P2, parametised by t.
        INPUTS:
            t     float/int; a parameterisation.
            P1    numpy array; a point.
            P2    numpy array; a point.
        OUTPUTS:
            Q1    numpy array; a point.
        """

        if not isinstance(P1, np.ndarray) or not isinstance(P2, np.ndarray):
            raise TypeError('Points must be an instance of the numpy.ndarray!')
        if not isinstance(t, (int, float)):
            raise TypeError('Parameter t must be an int or float!')

        Q1 = (1 - t) * P1 + t * P2
        return Q1

    def Points(t, points):
        """
        Returns a list of points interpolated by the Bezier process
        INPUTS:
            t            float/int; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            newpoints    list of numpy arrays; points.
        """
        newpoints = []
        #print("points =", points, "\n")
        for i1 in range(0, len(points) - 1):
            #print("i1 =", i1)
            #print("points[i1] =", points[i1])

            newpoints += [Bezier.TwoPoints(t, points[i1], points[i1 + 1])]
            #print("newpoints  =", newpoints, "\n")
        return newpoints

    def Point(t, points):
        """
        Returns a point interpolated by the Bezier process
        INPUTS:
            t            float/int; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            newpoint     numpy array; a point.
        """
        newpoints = points
        #print("newpoints = ", newpoints)
        while len(newpoints) > 1:
            newpoints = Bezier.Points(t, newpoints)
            #print("newpoints in loop = ", newpoints)

        #print("newpoints = ", newpoints)
        #print("newpoints[0] = ", newpoints[0])
        return newpoints[0]

    def Curve(t_values, points):
        """
        Returns a point interpolated by the Bezier process
        INPUTS:
            t_values     list of floats/ints; a parameterisation.
            points       list of numpy arrays; points.
        OUTPUTS:
            curve        list of numpy arrays; points.
        """

        if not hasattr(t_values, '__iter__'):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if len(t_values) < 1:
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if not isinstance(t_values[0], (int, float)):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")

        if len(np.shape(points)) >= 3:
            points = np.array(points).squeeze()
        curve = np.array([[0.0] * len(points[0])])
        for t in t_values:
            #print("curve                  \n", curve)
            #print("Bezier.Point(t, points) \n", Bezier.Point(t, points))

            curve = np.append(curve, [Bezier.Point(t, points)], axis=0)

            #print("curve after            \n", curve, "\n--- --- --- --- --- --- ")
        curve = np.delete(curve, 0, 0)
        #print("curve final            \n", curve, "\n--- --- --- --- --- --- ")
        return curve[:,np.newaxis,:]

def get_bezier_interp(latents, nf, in_w=False):
    if len(np.shape(latents)) >= 3:
            latents = np.array(latents).squeeze()
    step = 1./nf
    t_points = np.arange(0, 1, step) 
    if in_w:
        # ws = convertZtoW(latents)
        latents = latents[:,0,:]
    latents = Bezier.Curve(t_points, latents)
    if in_w:
        latents = np.expand_dims(np.repeat(latents,18,1),1)
    return latents
#-----------------------------------------------------------------------------
import scipy.interpolate as interpolate

# Interpolation in W space seems to be ok 
def get_latent_interpolation_bspline( nf, k=3, s=5, latents=None, seeds=None,circle= False ,in_w=False):
    if seeds:
        latents = generate_zs_from_seeds(seeds,squeeze=True)
    
    x = np.array(latents)
    if x.shape[1] == 1:
       x = np.reshape(x,(x.shape[0],x.shape[2]))
    if in_w:
        x = x[:,0,:]
    if circle:
        x = np.append(x, x[0,:].reshape(1, x.shape[1]), axis=0)

    nd = x.shape[1]
    latents = np.zeros((nd, nf))
    for i in range(nd-9):
        tck, u = interpolate.splprep([x[:,j] for j in range(i,i+10)], k=k, s=s)
        out = interpolate.splev(np.linspace(0, 1, num=nf, endpoint=True), tck)
        latents[i:i+10,:] = np.array(out)

    interpolated = latents.T[:, np.newaxis, :]
    if in_w:
      interpolated = np.expand_dims(np.repeat(interpolated,18,axis=1),1)
    return interpolated
#-----------------------------------------------------------------------------
def slerp(val, low, high):
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

def get_slerp_interp(nl, nf):
    low = np.random.randn(512)
    
    latent_interps = np.empty(shape=(0, 512), dtype=np.float32)
    for _ in range(nl):
        high = np.random.randn(512)
        
        interp_vals = np.linspace(1./nf, 1, num=nf)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                  dtype=np.float32)
        
        latent_interps = np.vstack((latent_interps, latent_interp))
        low = high

    return latent_interps[:, np.newaxis, :]

def get_slerp_interp_seeds(seeds, nf):
    latents = generate_zs_from_seeds(seeds,squeeze=True)
    low = latents[0]
    latent_interps = np.empty(shape=(0, 512), dtype=np.float32)
    for code in latents[1:]:
        high = code  

        interp_vals = np.linspace(1. / nf, 1, num=nf)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)

        latent_interps = np.vstack((latent_interps, latent_interp))
        low = high

    return latent_interps[:, np.newaxis, :]

def get_slerp_interp_latents(latents, nf):
    low = latents[0][0]
    latent_interps = np.empty(shape=(0, 512), dtype=np.float32)
    for code in latents[1:]:
        high = code[0]  

        interp_vals = np.linspace(1. / nf, 1, num=nf)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)

        latent_interps = np.vstack((latent_interps, latent_interp))
        low = high

    return latent_interps[:, np.newaxis, :]


# Spherical linear interpolation in W space starting from zs or seeds or ws
# Priority: ws, seeds, zs
def get_slerp_for_w(nf, zs=None, seeds = None, ws = None):
    assert seeds or zs or ws, 'Please provide one among zs, ws and seeds list'
    if zs:
        zs = np.array(zs)

    if seeds and not ws:
        zs = np.array(generate_zs_from_seeds(seeds, squeeze=True))

    if ws == None:
        if zs.shape[1] == 1:
            zs = np.reshape(zs,(zs.shape[0],zs.shape[2]))
        ws = convertZtoW(zs,truncation_psi= 1, truncation_cutoff=18)

    latents = []
    for latent in ws:
        latents.append(latent[0])
    low = latents[0]
    
    latent_interps = np.empty(shape=(0, 512), dtype=np.float32)
    for _ in range(1,len(latents)):
        high = latents[_]
        
        interp_vals = np.linspace(1./nf, 1, num=nf)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                  dtype=np.float32)
        #print('w', np.shape(latent_interp))
        latent_interps = np.vstack((latent_interps, latent_interp))
        low = high
    interpolated = latent_interps[:, np.newaxis, :]
    return  np.expand_dims(np.repeat(interpolated,18,axis=1),1)

#-----------------------------------
def generate_video_from_stills(stills, vid_name, fps=16):
    with imageio.get_writer(f'/content/{vid_name}.mp4', mode='I', fps=fps) as writer:
        for image in log_progress(list(stills), name = "Creating animation"):
            writer.append_data(np.array(image))
    return

def exctract_video_in_w_space(ws, truncation_psi,outdir,vidname,framerate,save_npy=False, prefix='img'):
    path = f'{outdir}/frames/'
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    

    for w_idx, w in enumerate(log_progress(list(ws), name= 'Generating images in W')):
        
        #print(f'Generating image for step {w_idx}/{len(ws)} ...') 
        noise_rnd = np.random.RandomState(1) # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.components.synthesis.run(w, **Gs_kwargs) # [minibatch, height, width, channel]
        os.makedirs(path, exist_ok=True)
        PIL.Image.fromarray(images[0], 'RGB').save(path+f'{prefix}{w_idx:05d}.png')
        if save_npy:
            np.save(f'{outdir}/vectors/{prefix}{w_idx:05d}.npz',w)
            # np.savetxt(f'{outdir}/vectors/{prefix}{w_idx:05d}.txt',w.reshape(w.shape[0], -1))
    os.makedirs(outdir, exist_ok=True)
    cmd=f"ffmpeg -y -r {framerate} -i {path}/{prefix}%05d.png -vcodec libx264 -pix_fmt yuv420p {outdir}/{vidname}-{framerate}fps.mp4"
    subprocess.call(cmd, shell=True)

def generate_neighbors_latents(seed, diameter=5, num_latents=15, clip=True, in_w=False):
 
    rnd = np.random.RandomState(seed)
    og_z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
    latents = [og_z]
    if in_w:
        og_w = convertZtoW(og_z,truncation_psi=1)
        latents = [og_w]
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]

    for s in log_progress(range(num_latents), name='Image'):
        if in_w:
            rndm_vec = np.random.uniform(-diameter,diameter,[1,1,512])
            rndm_vec = rndm_vec*diameter / np.linalg.norm(rndm_vec)
            nnw = np.repeat(rndm_vec,18,1)
            new_w = sum_latents(og_w,nnw, clip=clip)
            latents.append(new_w)
        else:
            rndm_vec = np.random.uniform(-diameter,diameter,[1,512])
            nnz = rndm_vec*diameter / np.linalg.norm(rndm_vec)
            new_z = sum_latents(og_z,nnz, clip=clip)
            latents.append(new_z)
    return latents


def sum_latents(v,w, clip=True, keep_len = False ):
    v = np.array(v)
    w = np.array(w)
    vw = np.add(v,w)
    if keep_len:
        vw = [vw_*np.linalg.norm(v[_]) / np.linalg.norm(vw_) for _, vw_ in enumerate(vw)]
    if clip:
        return np.clip(vw,-1,1)
    else:
        return vw

def add_noise(zs,zs_noise):
    if len(zs) != len(zs_noise):
        zs_noise.pop()
    modulate_noise = [np.sin(_*np.pi/(len(zs_noise)-1))*z_noise for _, z_noise in enumerate(zs_noise)]
    return sum_latents(zs,modulate_noise,keep_len=True)

def combined_walk(nf, slerp=False,spline=False, bezier=False, combine=None, comb_seed=42, seeds=None, latents= None, circle=False, circular_d = 120):
    assert (slerp or spline or bezier),  'Either slerp or spline or bezier has to be True'
    assert seeds or latents, 'Seeds or latents must be provided'
    combinations = ['random-walk-c','random-walk-e','random-walk-v','random-walk-s', 'circular-loop']
    if len(combine) > 1 and len(combine) <= len(combinations):
        for combination in combine:
            assert combination in combinations, f'Combine elements must be among {combinations}'
    else:
        assert combine in combinations, f'Combine elements must be among {combinations}'
        combine = [combine]
    if seeds:
        latents = generate_zs_from_seeds(seeds)
    
    if slerp:
        if circle:
            latents.append(latents[0])
        nf = int(nf//(len(latents)-1))
        zs = get_slerp_interp_latents(latents, nf)
    elif spline:
        condition = len(latents) > 3
        k = 2 + condition
        zs = get_latent_interpolation_bspline(latents=latents, nf = nf, circle=circle, k=k)
    else:
        zs = get_bezier_interp(latents, nf=nf)

    for combination in combine:
        if combination == 'random-walk-c':
            osni = OSNI(comb_seed)
            zs_noise = osni.get_noiseloop_seeds(len(zs))  
            zs = add_noise(zs, zs_noise)

        elif combination == 'random-walk-v':
            osni = OSNI(comb_seed)
            zs_noise = osni.get_noiseloop_3D(len(zs))  
            zs = add_noise(zs, zs_noise)
        
        elif combination == 'random-walk-e':
            osni = OSNI(comb_seed)
            zs_noise = osni.get_noiseloop_seeds3D_loop(len(zs))  
            zs = add_noise(zs, zs_noise)
        elif combination == 'random-walk-s':
            osni = OSNI(comb_seed)
            zs_noise = osni.get_noiseloop_seeds3D(len(zs))  
            zs = add_noise(zs, zs_noise)
        
        else:
            zs_noise = get_circularloop(nf = len(zs), seed = comb_seed, d = circular_d)
            zs = add_noise(zs, zs_noise)
    return zs

def forever_loop(latent_pair, nloops, frame_per_loop = 36, spline=False, slerp=False, bezier=False, circular_d=5):
    latent_pair.append(latent_pair[0])
    zs = []
    for loop in range (nloops):
        _zs = combined_walk(nf = frame_per_loop, slerp=slerp, spline=spline, bezier=bezier, combine='circular-loop',latents = latent_pair, circular_d = circular_d)
        zs.extend(_zs)
    return zs 

#Interface helper functions
 
def show_images(location, size = 30):
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import numpy as np
    from PIL import Image

    elements = os.listdir(f'/content/{location}')
    n = len(elements)
    im_arr = [None]*n
    for idx, imgpath in enumerate(elements):
        im = Image.open(f'/content/{location}/{imgpath}')
        im_arr[idx] = im.resize((256,256)) 

    nrows = int(np.ceil(n/5))
    fig = plt.figure(figsize=(size, size))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(nrows, 5),  # 
                 axes_pad=0.5,  # pad between axes in inch.
                 )

    for idx, (ax, im) in enumerate(zip(grid, im_arr)):
     # Iterating over the grid returns the Axes.
        ax.set_title("img%d.png" % idx)
        ax.imshow(im)

    plt.show()


def saveImgs(imgs, location,prefix=''):
    os.makedirs("/content/"+location, exist_ok=True)
    for idx, img in log_progress(enumerate(imgs), size = len(imgs), name="Saving images"):
        file = "/content/"+location+ '/img' +prefix + str(idx) + ".png"
        img.save(file)

def extract_number(f):
    s = re.search("(\d+)",f)
    return (int(s[0]) if s else -1,f)

def generate_images_stills(zs, truncation_psi, Gs, location='out', prefix=''):

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if not isinstance(truncation_psi, list):
        truncation_psi = [truncation_psi] * len(zs)
    path = "/content/"+location
    os.makedirs(path, exist_ok=True)
    try:
        last_idx = max(os.listdir(path),key=extract_number)
        p = re.compile("(\d+)")
        last_idx = int(re.match(p, last_idx)[0])
        last_idx = max(1,last_idx)
    except:
        last_idx = 1
    for z_idx, z in log_progress(enumerate(zs), size = len(zs), name = "Generating images"):
        Gs_kwargs.truncation_psi = truncation_psi[z_idx]
        noise_rnd = np.random.RandomState(1) # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        
        
        image_save = PIL.Image.fromarray(images[0], 'RGB')
        z_idx += last_idx
        file = path + '/' + prefix + str(z_idx).zfill(5) + ".jpg"
        image_save.save(file)
        
    return 

def generate_images_stills_in_w(ws, truncation_psi, Gs, location='out'):
    path = f'/content/{location}'
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi
    ws = ws.squeeze()
    ws = np.expand_dims(ws, axis=1)
    os.makedirs(path, exist_ok=True)
    try:
        last_idx = max(os.listdir(path),key=extract_number)
        p = re.compile("(\d+)")
        last_idx = int(re.match(p, last_idx)[0])
        last_idx = max(1,last_idx)
    except:
        last_idx = 1

    
    for w_idx, w in enumerate(log_progress(list(ws), name= 'Generating images in W')):
        
        #print(f'Generating image for step {w_idx}/{len(ws)} ...') 
        noise_rnd = np.random.RandomState(1) # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.components.synthesis.run(w, **Gs_kwargs) # [minibatch, height, width, channel]
        
        image_save = PIL.Image.fromarray(images[0], 'RGB')
        w_idx += last_idx
        file = path + '/'  + str(w_idx).zfill(5) + ".jpg"
        image_save.save(file)

def load_model(pkl_path):
  dnnlib.tflib.init_tf()
   
  print('Loading networks from "%s"...' % pkl_path)
  with dnnlib.util.open_url(pkl_path) as fp:
      _G, _D, Gs = pickle.load(fp)
  noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
  return Gs, noise_vars


def linear(latents, steps):
   out = []
   for i in range(len(latents)-1):
    for index in range(steps):
     fraction = index/float(steps) 
     out.append(latents[i+1]*fraction + latents[i]*(1-fraction))
   return out

def get_linear_interpolation(latents, frames, in_w=False):
    steps = int(frames//(len(latents)-1))
    if in_w:
        latents = [w[0,:] for w in latents]
    interpolated = np.array(linear(latents, steps))
    if in_w:
        interpolated = interpolated[:, np.newaxis, np.newaxis, :]
        interpolated = np.repeat(interpolated,18,axis=2)
        
    return interpolated

def interpolation_from_seeds(seeds = None, interpolation_type=None, interp_in_w = False, truncation = 0, frames = 240 ):
    latents = generate_zs_from_seeds(seeds=seeds)
    if interpolation_type == 'linear':
        if interp_in_w:
            latents = np.squeeze(latents)
            latents = convertZtoW(latent=latents, truncation_psi= truncation, truncation_cutoff=18)
            latents = list(latents)
        interpolated = get_linear_interpolation(latents=latents, frames=frames, in_w=interp_in_w)

    elif interpolation_type == 'slerp':
        nf = int(frames//(len(latents)-1))
        if interp_in_w:
            latents = np.squeeze(latents)
            latents = convertZtoW(latent=latents, truncation_psi= truncation, truncation_cutoff=18)
            latents = list(latents)
            interpolated = get_slerp_for_w(ws = latents, nf=nf)
        else:
            interpolated = get_slerp_interp_latents(latents=latents, nf=nf)

    elif interpolation_type == 'spline':
        # Set condition to avoid failure if less than 3 seeds are selected
        condition = len(latents) > 3
        k = 2 + condition
        interpolated = get_latent_interpolation_bspline(latents=latents, nf = frames, k=k, in_w=interp_in_w)

    elif interpolation_type == 'bezier':
        interpolated = get_bezier_interp(latents, nf=frames,  in_w=interp_in_w)

    else:
        print('Interpolation type provided not supported')
        return
    return interpolated

def create_video(video_path, framerate, input_path, vidname):
    os.makedirs(video_path, exist_ok=True)
    cmd=f"ffmpeg -y -r {framerate} -i /content/{input_path}/%05d.jpg -vcodec libx264 -pix_fmt yuv420p {video_path}/{vidname}.mp4"
    success = subprocess.call(cmd, shell=True)
    return success



def show_images(location, size = 30):
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    import numpy as np
    from PIL import Image

    elements = os.listdir(f'/content/{location}')
    n = len(elements)
    im_arr = [None]*n
    for idx, imgpath in enumerate(elements):
        im = Image.open(f'/content/{location}/{imgpath}')
        im_arr[idx] = im.resize((256,256)) 

    nrows = int(np.ceil(n/5))
    fig = plt.figure(figsize=(size, size))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(nrows, 5),  # 
                 axes_pad=0.5,  # pad between axes in inch.
                 )

    for idx, (ax, im) in enumerate(zip(grid, im_arr)):
     # Iterating over the grid returns the Axes.
        ax.set_title("img%d.png" % idx)
        ax.imshow(im)

    plt.show()


def saveImgs(imgs, location,prefix=''):
    os.makedirs("/content/"+location, exist_ok=True)
    for idx, img in log_progress(enumerate(imgs), size = len(imgs), name="Saving images"):
        file = "/content/"+location+ '/img' +prefix + str(idx) + ".png"
        img.save(file)

def generate_images_stills(zs, truncation_psi, location='out', prefix=''):

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if not isinstance(truncation_psi, list):
        truncation_psi = [truncation_psi] * len(zs)
    path = "/content/"+location
    os.makedirs(path, exist_ok=True)
    
    for z_idx, z in log_progress(enumerate(zs), size = len(zs), name = "Generating images"):
        Gs_kwargs.truncation_psi = truncation_psi[z_idx]
        noise_rnd = np.random.RandomState(1) # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        
   
        image_save = PIL.Image.fromarray(images[0], 'RGB')
        
        file = path + '/' + prefix + str(z_idx).zfill(5) + ".jpg"
        image_save.save(file)
        
    return 

def generate_images_stills_in_w(ws, truncation_psi,location):
    path = f'/content/{location}'
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi
    ws = ws.squeeze()
    ws = np.expand_dims(ws, axis=1)
    for w_idx, w in enumerate(log_progress(list(ws), name= 'Generating images in W')):
        
        #print(f'Generating image for step {w_idx}/{len(ws)} ...') 
        noise_rnd = np.random.RandomState(1) # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.components.synthesis.run(w, **Gs_kwargs) # [minibatch, height, width, channel]
        os.makedirs(path, exist_ok=True)
        image_save = PIL.Image.fromarray(images[0], 'RGB')
        
        file = path + '/'  + str(w_idx).zfill(5) + ".jpg"
        image_save.save(file)

def load_model(pkl_path):
  dnnlib.tflib.init_tf()
   
  print('Loading networks from "%s"...' % pkl_path)
  with dnnlib.util.open_url(pkl_path) as fp:
      _G, _D, Gs = pickle.load(fp)
  noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
  return Gs, noise_vars


def linear(latents, steps):
   out = []
   for i in range(len(latents)-1):
    for index in range(steps):
     fraction = index/float(steps) 
     out.append(latents[i+1]*fraction + latents[i]*(1-fraction))
   return out

def get_linear_interpolation(latents, frames, in_w=False):
    steps = int(frames//(len(latents)-1))
    if in_w:
        latents = [w[0,:] for w in latents]
    interpolated = np.array(linear(latents, steps))
    if in_w:
        interpolated = interpolated[:, np.newaxis, np.newaxis, :]
        interpolated = np.repeat(interpolated,18,axis=2)
        
    return interpolated

def interpolation_from_seeds(seeds = None, interpolation_type=None, interp_in_w = False, truncation = 0, frames = 240 ):
    latents = generate_zs_from_seeds(seeds=seeds)
        
    if interpolation_type == 'linear':
        if interp_in_w:
            latents = np.squeeze(latents)
            latents = convertZtoW(latent=latents, truncation_psi= truncation, truncation_cutoff=18)
            latents = list(latents)
        interpolated = get_linear_interpolation(latents=latents, frames=frames, in_w=interp_in_w)

    elif interpolation_type == 'slerp':
        nf = int(frames//(len(latents)-1))
        if interp_in_w:
            latents = np.squeeze(latents)
            latents = convertZtoW(latent=latents, truncation_psi= truncation, truncation_cutoff=18)
            latents = list(latents)
            interpolated = get_slerp_for_w(ws = latents, nf=nf)
        else:
            interpolated = get_slerp_interp_latents(latents=latents, nf=nf)

    elif interpolation_type == 'spline':
        # Set condition to avoid failure if less than 3 seeds are selected
        condition = len(latents) > 3
        k = 2 + condition
        interpolated = get_latent_interpolation_bspline(latents=latents, nf = frames, k=k, in_w=interp_in_w)

    elif interpolation_type == 'bezier':
        interpolated = get_bezier_interp(latents, nf=frames,  in_w=interp_in_w)

    else:
        print('Interpolation type provided not supported')
        return
    return interpolated

def create_video(video_path, framerate, input_path, vidname):
    os.makedirs(video_path, exist_ok=True)
    cmd=f"ffmpeg -y -r {framerate} -i /content/{input_path}/%05d.jpg -vcodec libx264 -pix_fmt yuv420p {video_path}/{vidname}.mp4"
    success = subprocess.call(cmd, shell=True)
    return success

