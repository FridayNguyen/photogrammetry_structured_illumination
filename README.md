# photogrammetry_structured_illumination
A photogrammetry program that generates a 3d mesh using input images with structured illumination

Using a collection of provided photographs as input. The implemented program 
would interpret these images and generate corresponding 3D meshes. The specific lighting technique applied within the provided input images, known
as structured illumination, allowed me the program to triangulate the object’s 3D geometry from
sets of flat 2D images.

## THE DATA
The program utilizes two sets of input
images to accomplish its task. The first set
of images, which hereafter will be referred
to as the calibration set, is a collection of
images of a chessboard placed at various
positions and orientations. Each image
has a resolution of 1920x1200 pixels, and
there must be a total of 21 images for each of
the two cameras. The calibration set would then be used for camera calibrations

![image](https://user-images.githubusercontent.com/44386004/179340998-f25892b1-433f-49a1-a3f8-78f10d50e6ba.png)

The second set of images, possessing the
same resolution as the calibration set,
are pictures of the actual object being
scanned. For each perspective of the
object, two color images are provided per
camera, one showing only the
background, and one with the objects.
These images will be later used to create
a foreground mask of the object of interest, as well as retrieving the object’s color
information.

![image](https://user-images.githubusercontent.com/44386004/179341009-4b58b034-bd9e-4efe-be9d-53a6ee4c2357.png)

As for the remaining images
in the input set, each camera
also captured 40 additional
photographs of the object.
Every image is uniquely
illuminated with specific
lighting patterns, an
approach also known as
structured illumination.
This process is repeated for each perspective, each producing their own set of 44 input
images. The number of input sets varies per object, with more complex ones requiring
more perspectives to fully capture the unique geometry. For this particular teapot, 7
perspectives were provided as input.

![image](https://user-images.githubusercontent.com/44386004/179341042-d3e1840e-5826-4b0d-b4fe-140d5ebb02a1.png)

## THE ALGORITHMS
In order to perform triangulation on the object,
the program must first know the cameras’
parameters. This is accomplished using the
calibration image set. These images show
multiple views of a known planar surface (the
chessboard) which allows for the use of
multiplane calibration to infer the unknown
intrinsic parameters of the cameras. Multiplane calibration is a popular method of
calibrating cameras by solving a particular homogeneous linear system that captures the
homographic relationships between multiple perspective views of the same plane. The
actual implementation of multiplane calibration was done by the provided calibrate.pyfile,
which uses various functions from the opencvlibrary.

![image](https://user-images.githubusercontent.com/44386004/179341056-8eae6839-c721-442b-96a3-2516cd75deea.png)

The next step was to infer the camera’s extrinsic parameters (i.e. translations and
rotations). Much of the code for this came from assignment 3, which focused on
calibration. Essentially the process revolves around the calibratePose function, which given
the known position of a point in both 3D space and a 2D image, would solve a nonlinear
least square problem to optimize an initial estimate of a camera’s extrinsic parameters
toward the correct values.
The optimization is solved using the leastsq function from the SciPy library. The code for
calibratePose is shown below:

```
def calibratePose(pts3,pts2,cam,params_init):
  """
  Calibrate the provided camera by updating R,t so that pts3 projects
  as close as possible to pts2
  """
  estimate = scipy.optimize.leastsq(wrapped_residuals, x0=params_init,
  args=(pts3,pts2,cam))
  best_params = estimate[0]
  cam.update_extrinsics(best_params)
  return cam
```

Next, we set out to decode the information that was encoded with structured illumination.
Structural illumination works by projecting a sequence of black and white patterns onto the
object in such a way that each pixel of interest has a unique sequence of 1 (bright) and 0
(dark) as its ID. Furthermore, rather than having to agonize over which color value is
considered bright or dark, which can drastically vary due to the color and distance of the
actual surface, each pattern projected is also subsequently accompanied by its inverse. The
program would then simply threshold the difference to determine whether a pixel is
considered bright, dark, or undecodable (which were commonly background pixels where
light was not projected). The follow code decode the pattern sequence into graycode:

```
for i in range(0, len(imgs), 2): # loop through (10) pairs of images
  first = imgs[i] # the image of the pattern
  second = imgs[i+1] # image of inverse pattern
  diff = first - second # calculate the difference
  # if the difference is positive, the pixel is bright (otherwise dark)
  bit = np.where(diff > 0, 1, 0)
  # the bits are later combined into the graycode
  binary_imgs.append(bit)
```

The graycode would then be converted to a decimal. This process is performed once for the
horizontal patterns, and once for the vertical ones, after which the resulting decimals were
combined into unique IDs to be assigned to the pixels of interest. Using a similar
difference thresholding method as before, a foreground mask was also generated using
the camera’s two color images (one with the object and one with only the background). The
foreground mask and the undecodable mask would then be applied to the pixel ID array,
thus filtering out pixels that are either undecodable or belonging to the background. Once
we have the two pixel ID arrays for the two cameras, it was trivial to generate pts2L and
pts2R—two matching 2xN arrays containing the 2D coordinate of each pixel, aligned by
indexes. Finally, the 3D positions of the pixels were generated with the following lines of
code:

```
# camutils was provided with the course files
from camutils import triangulate
# triangulation using much of the technique from Assignment #4
pts3 = triangulate(pts2L,camL, pts2R,camR)
```

The array faces of the mesh were then calculated using spatial.Delaunay from the
SciPy library
Additionally, the color information was also preserved by averaging the color value of the
left and right image, and storing the resulting pixel color values in a 3xN array, also aligned
by indexes.
Lastly, the program processed the triangulated points so that they would create a nicer
resulting mesh. The points were subjected to bounding-box pruning, triangle pruning, and
mesh smoothing. Bounding-box pruning simply takes into a specified 3D volume (defined
by its bound), and deletes any point that lies outside said volume. This prunes many outlier
points outside of the area of interest. Triangle pruning takes an edge length threshold, then
loops over each vertex of the mesh. If the vertex is part of an edge with length longer than
the threshold, it would be deleted, resulting in many outlier points being removed. The
cleaned data would then be written into a ply file with the writeply function from the
provided meshutils.py, ready to be imported into MeshLab. The process was repeated
again for every set of input images, each resulting in a partial mesh of the object being
scanned.

## THE RESULT
Each set of input images resulted in a partial colored mesh of
the object, which can be parsed into a ply file and imported into
MeshLab
![image](https://user-images.githubusercontent.com/44386004/179341161-cff6636b-6a44-484d-94c6-0b7e31fd97b9.png)

The meshes were then aligned together to form the
complete object. This process can be done by hand,
or assisted by MeshLab’s Align functionality.
Once successfully aligned
and merged, the mesh must
undergo Poisson surface reconstruction, which can also
accomplished within MeshLab. This will correct the missing
surfaces and create an airtight mesh
![image](https://user-images.githubusercontent.com/44386004/179341168-37c32ce0-15da-435c-8949-d4a6ee49abf1.png)

Finally, the final mesh can be exported into Blender and rendered, ready for viewing.
![image](https://user-images.githubusercontent.com/44386004/179341197-55cd652d-f7fa-4b0e-92bc-660340e7f398.png)

