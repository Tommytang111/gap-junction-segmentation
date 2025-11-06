import pyvista as pv
import numpy as np 
import gc
from skimage.measure import block_reduce

import os
import panel as pn
pn.extension('vtk')

#Load volumes
pred = np.load("/home/tommytang111/gap-junction-segmentation/outputs/volumetric_results/unet_h1qrqboc/sem_dauer_1_s000-850/volume_block_downsampled.npy")
points = np.load("/home/tommytang111/gap-junction-segmentation/gj_point_annotations/sem_dauer_1_GJs_block_downsampled.npy")
point_entities = np.load("/home/tommytang111/gap-junction-segmentation/gj_point_annotations/sem_dauer_1_GJs_entities_downsampled.npy")

#Block reduce
pred_ = block_reduce(pred, block_size=(1,2,2), func=np.max)
points_ = block_reduce(points, block_size=(1,2,2), func=np.max)
point_entities_ = block_reduce(point_entities, block_size=(1,2,2), func=np.max)

#Experimenting with overlaying objects
print(pred_.shape)
print(points_.shape)
print(point_entities_.shape)

#Convert to list of points
points_list = np.argwhere(points_ == 255).astype(np.float32)

#Transform pred into isosurface
grid = pv.wrap(pred_)
contour = grid.contour(isosurfaces=[255])

#Transform point_entities into isosurface
grid2 = pv.wrap(point_entities_)
contour2 = grid2.contour(isosurfaces=[255])

#Transform points into glyphs/spheres
point_cloud = pv.PolyData(points_list)
lowpoly = pv.Sphere(radius=2.0, theta_resolution=8, phi_resolution=8)  # low triangle count
spheres = point_cloud.glyph(scale=False, geom=lowpoly, orient=False) #pv.Sphere(radius=2)
spheres.clear_data()  # drop data arrays to shrink fil

#Brighter parameters
bright = dict(ambient=0.2, diffuse=1.0, specular=0.9, specular_power=128, lighting=True) #Most of these are default values except ambient and specular_power

#Plotting
pv.global_theme.background = 'black'
pv.force_float=False

p = pv.Plotter(off_screen=True, shape=(2, 2))
# Predictions
# p.subplot(0, 0)  # first subplot is active by default
p.add_mesh(contour, color="#02EBFC", show_scalar_bar=False, **bright)

# Point Entities
p.subplot(0, 1)
p.add_mesh(contour2, color="#BF35FF", show_scalar_bar=False, **bright)

# Point + Point Entities Overlay
p.subplot(1, 0)
p.add_mesh(spheres, color="#FA9017", show_scalar_bar=False, **bright)
p.add_mesh(contour2, color="#BF35FF", opacity=0.7, show_scalar_bar=False, **bright)

# Predictions + Point Entities Overlay
p.subplot(1, 1)
p.add_mesh(contour, color="#02EBFC", opacity=0.3, show_scalar_bar=False)
p.add_mesh(contour2, color="#BF35FF", opacity=1, show_scalar_bar=False, **bright)

p.link_views()
p.show_axes()

#p.export_html("/home/tommytang111/gap-junction-segmentation/html_objects/Dauer 1/combination_overlay.html")
p.show()

# Ensure clean teardown to avoid __del__ errors at interpreter exit
pv.close_all()

