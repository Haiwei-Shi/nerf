import numpy as np
import os, imageio


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

# resize images, basedir: images directory, factor : integer, resolution: [H,W] list
# so basically this function will take the images folder in the object
# then create the images_8 folder, and put resized images in it
# if the original images are not png, delete the original images in images_8 folder
def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        # create a directory, name images_{factor number}
        if not os.path.exists(imgdir):
            # if the folder does not exist, 
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
        # same as above
    if not needtoload:
        return
    # so this part check whether both folder exits, if anyone is missing, we need to load the images
    # if there are these two folders then quit the minify function
    # however this part only check the folder's existence,
    # not check whether the folders are empty or not
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    # find the images folder
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    # get all the images in the folder and sort them in order
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    # only take certain types of images
    imgdir_orig = imgdir
    #imgdir = data/object/images
    
    wd = os.getcwd()
    # get the current working directory

    for r in factors + resolutions:
        # concatenate the two lists, since factors is list of integer
        # and resolutions is list of list, then it will be a list like [integer1, [interger2, integer3]]
        # so basically this loop will iterate through the two elements of the list, an integer and a list
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        # create image folder name like images_8 or images_256x256
        imgdir = os.path.join(basedir, name)
        # create directory path name
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        # create the directory just named above, images_8
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        # copy images from the original folder to the new folder images_8
        
        ext = imgs[0].split('.')[-1]
        # take the file type of the image, like jpg or png
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        # create a string like 'mogrify -resize 12.5% -format png *.jpg'
        # this command will resize the images in the folder, and convert them to png format
        print(args)
        os.chdir(imgdir)
        # change from the basedir to the new folder iamges_8
        check_output(args, shell=True)
        # execute the command
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        # if the original image is not png, then remove the original images in the folder images_8
        print('Done')
            
        
        

# this function take the pose_bounds.npy file and images folder, and return the poses (3*5*N), bounds(2*N), and images(H*W*C*N)
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    # Step-by-step transformation:
    # N*17 -> N*15 -> N*3*5 -> 3*5*N
    # 1. poses_arr[:, :-2] → Extracts the first 15 columns (shape: (N, 15))
    # 2. .reshape([-1, 3, 5]) → Reshapes into (N, 3, 5), treating each pose as a 3×5 matrix
    # the -1 at first means the first axis number will be computed by numpy after the last two axis become 3*5
    # 3. .transpose([1, 2, 0]) → Rearranges axes to (3, 5, N)
    #    - `3` → Rotation matrix rows
    #    - `5` → Rotation + translation matrix columns
    #    - `N` → Number of samples

    # [
    #    [[a1, b1, c1, ...], [[a6,  b6,  c6, ...],  [[a11,  b11,  c11, ...],
    #     [a2, b2, c2, ...],  [a7,  b7,  c7, ...],   [a12,  b12,  c12, ...],
    #     [a3, b3, c3, ...],  [a8,  b8,  c8, ...],   [a13,  b13,  c13, ...],
    #     [a4, b4, c4, ...],  [a9,  b9,  c9, ...],   [a14,  b14,  c14, ...],
    #     [a5, b5, c5, ...]], [a10, b10, c10, ...]], [a15,  b15,  c15, ...],
    # ]
    bds = poses_arr[:, -2:].transpose([1,0])
    # Step-by-step transformation:
    # N*17 -> N*2 -> 2*N
    # 1. poses_arr[:, -2:] → Extracts the last two columns (shape: (N, 2))
    # 2. .transpose([1, 0]) → Swaps axes to (2, N), making:
    # [
    #    [a16, b16, c16, ...],  # Near bounds (row 1)
    #    [a17, b17, c17, ...]   # Far bounds (row 2)
    # ]
    
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    # find the first image in the folder, and get the path
    sh = imageio.imread(img0).shape
    #get the shpae of the image
    
    sfx = ''
    # this is the suffix of the folder name, like images_8 or images_256x256, so that it can find the folder later
    
    #this part prioritize the factor input, if there is factor input, then resize the images by the factor
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    # if there is factor input, the resize the images by the factor and put all images in the folder images_{factor}
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    # if there is resolution input, the resize the images into the resolution and put all images in the folder images_{width}x{height}
    # the new height will be the input height, and the width will be calculated by the factor by original_height / new_height
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    # find the images folder with the suffix, images_8
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    # find all the images directory
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        # if the poses and images numbers are not matched, then print the error message
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    # (H, W, C)
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    # Updates:
    # - Selects the last column (index 4) of the first two 3x5*N matrices**.
    # - Replaces it with the height (H) and width (W) of the image.
    # [
    #    [[a1, b1, c1, ...], [[a6,  b6,  c6, ...],  [[a11,  b11,  c11, ...],
    #     [a2, b2, c2, ...],  [a7,  b7,  c7, ...],   [a12,  b12,  c12, ...],
    #     [a3, b3, c3, ...],  [a8,  b8,  c8, ...],   [a13,  b13,  c13, ...],
    #     [a4, b4, c4, ...],  [a9,  b9,  c9, ...],   [a14,  b14,  c14, ...],
    #     [H,  H,  H, ...]],  [W,   W,   W, ...]],   [a15,  b15,  c15, ...],
    # ]
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    # - Selects the **last column (index 4) of the third 3x5*N matrix**.
    # - Divides its values by `factor`.

    # [
    #    [[a1, b1, c1, ...], [[a6,  b6,  c6, ...],  [[a11,  b11,  c11, ...],
    #     [a2, b2, c2, ...],  [a7,  b7,  c7, ...],   [a12,  b12,  c12, ...],
    #     [a3, b3, c3, ...],  [a8,  b8,  c8, ...],   [a13,  b13,  c13, ...],
    #     [a4, b4, c4, ...],  [a9,  b9,  c9, ...],   [a14,  b14,  c14, ...],
    #     [H,  H,  H, ...]],  [W,   W,   W, ...]],   [a15/factor,  b15/factor,  c15/factor, ...],
    # ]
    
    if not load_imgs:
        return poses, bds
    
    # if it is png, ignore the gamma
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
    
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    # read all the images in the folder, and only take the first three channels RGB, and normalize the image to [0,1]
    imgs = np.stack(imgs, -1)
    # stack all the images together, so the shape will be (H, W, C, N)
    # A big H × W matrix, where each “cell” (pixel) contains a 3 × N matrix.
    # This 3 × N matrix stores the RGB values of that pixel across all N images.
    # Picture style:
    # [
    #    [[R11_1, R11_2, R1_3, ... R11_N],  [[R12_1,  R12_2,  R12_3, ... R12_N],  [[R13_1,  R13_2,  R13_3, ... R13_N],  ... [[R1W_1,  R1W_2,  R1W_3, ... R1W_N],
    #     [G11_1, G11_2, G1_3, ... G11_N],   [G12_1,  G12_2,  G12_3, ... G12_N],   [G13_1,  G13_2,  G13_3, ... G13_N],  ...  [G1W_1,  G1W_2,  G1W_3, ... G1W_N],
    #     [B11_1, B11_2, B1_3, ... B11_N]],  [B12_1,  B12_2,  B12_3, ... B12_N]],  [B13_1,  B13_2,  B13_3, ... B13_N]], ...  [B1W_1,  B1W_2,  B1W_3, ... B1W_N]],  
    #
    #    [[R21_1, R21_2, R2_3, ... R21_N],  [[R22_1,  R22_2,  R22_3, ... R22_N],  [[R23_1,  R23_2,  R23_3, ... R23_N],  ... [[R2W_1,  R2W_2,  R2W_3, ... R2W_N],
    #     [G21_1, G21_2, G2_3, ... G21_N],   [G22_1,  G22_2,  G22_3, ... G22_N],   [G23_1,  G23_2,  G23_3, ... G23_N],  ...  [G2W_1,  G2W_2,  G2W_3, ... G2W_N],
    #     [B21_1, B21_2, B2_3, ... B21_N]],  [B22_1,  B22_2,  B22_3, ... B22_N]],  [B23_1,  B23_2,  B23_3, ... B23_N]], ...  [B2W_1,  B2W_2,  B2W_3, ... B2W_N]],
    #
    #    [[R31_1, R31_2, R3_3, ... R31_N],  [[R32_1,  R32_2,  R32_3, ... R32_N],  [[R33_1,  R33_2,  R33_3, ... R33_N],  ... [[R3W_1,  R3W_2,  R3W_3, ... R3W_N],
    #     [G31_1, G31_2, G3_3, ... G31_N],   [G32_1,  G32_2,  G32_3, ... G32_N],   [G33_1,  G33_2,  G33_3, ... G33_N],  ...  [G3W_1,  G3W_2,  G3W_3, ... G3W_N],
    #     [B31_1, B31_2, B3_3, ... B31_N]],  [B32_1,  B32_2,  B32_3, ... B32_N]],  [B33_1,  B33_2,  B33_3, ... B33_N]], ...  [B3W_1,  B3W_2,  B3W_3, ... B3W_N]],
    #    ...
    #    [[RH1_1, RH1_2, RH_3, ... RH1_N],  [[RH2_1,  RH2_2,  RH2_3, ... RH2_N],  [[RH3_1,  RH3_2,  RH3_3, ... RH3_N],  ... [[RHW_1,  RHW_2,  RHW_3, ... RHW_N],
    #     [GH1_1, GH1_2, GH_3, ... GH1_N],   [GH2_1,  GH2_2,  GH2_3, ... GH2_N],   [GH3_1,  GH3_2,  GH3_3, ... GH3_N],  ...  [GHW_1,  GHW_2,  GHW_3, ... GHW_N],
    #     [BH1_1, BH1_2, BH_3, ... BH1_N]],  [BH2_1,  BH2_2,  BH2_3, ... BH2_N]],  [BH3_1,  BH3_2,  BH3_3, ... BH3_N]], ...  [BHW_1,  BHW_2,  BHW_3, ... BHW_N]],
    # ]
    # OR
    # Numpy style:
    # [
    #   [ 
    #     [[R11_1, R11_2, ..., R11_N],  [G11_1, G11_2, ..., G11_N],  [B11_1, B11_2, ..., B11_N]],  # Pixel (0,0)
    #     [[R12_1, R12_2, ..., R12_N],  [G12_1, G12_2, ..., G12_N],  [B12_1, B12_2, ..., B12_N]],  # Pixel (0,1)
    #     [[R13_1, R13_2, ..., R13_N],  [G13_1, G13_2, ..., G13_N],  [B13_1, B13_2, ..., B13_N]],  # Pixel (0,2)
    #     ... 
    #     [[R1W_1, R1W_2, ..., R1W_N],  [G1W_1, G1W_2, ..., G1W_N],  [B1W_1, B1W_2, ..., B1W_N]]  # Pixel (0,W-1)
    #   ], # Row 0
    #
    #   [ 
    #     [[R21_1, R21_2, ..., R21_N],  [G21_1, G21_2, ..., G21_N],  [B21_1, B21_2, ..., B21_N]],  # Pixel (1,0)
    #     [[R22_1, R22_2, ..., R22_N],  [G22_1, G22_2, ..., G22_N],  [B22_1, B22_2, ..., B22_N]],  # Pixel (1,1)
    #     [[R23_1, R23_2, ..., R23_N],  [G23_1, G23_2, ..., G23_N],  [B23_1, B23_2, ..., B23_N]],  # Pixel (1,2)
    #     ... 
    #     [[R2W_1, R2W_2, ..., R2W_N],  [G2W_1, G2W_2, ..., G2W_N],  [B2W_1, B2W_2, ..., B2W_N]]  # Pixel (1,W-1)
    #   ], # Row 1
    #
    #   ...
    #
    #   [ 
    #     [[RH1_1, RH1_2, ..., RH1_N],  [GH1_1, GH1_2, ..., GH1_N],  [BH1_1, BH1_2, ..., BH1_N]],  # Pixel (H-1,0)
    #     [[RH2_1, RH2_2, ..., RH2_N],  [GH2_1, GH2_2, ..., GH2_N],  [BH2_1, BH2_2, ..., BH2_N]],  # Pixel (H-1,1)
    #     [[RH3_1, RH3_2, ..., RH3_N],  [GH3_1, GH3_2, ..., GH3_N],  [BH3_1, BH3_2, ..., BH3_N]],  # Pixel (H-1,2)
    #     ... 
    #     [[RHW_1, RHW_2, ..., RHW_N],  [GHW_1, GHW_2, ..., GHW_N],  [BHW_1, BHW_2, ..., BHW_N]]  # Pixel (H-1,W-1)
    #   ] # Row H-1
    # ]
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

    
            
            
    

def normalize(x):
    return x / np.linalg.norm(x)
# return the original matrix devided by the L2 Norm of array x (sqrt(sum(x**2)))

def viewmatrix(z, up, pos):
    """
    This is a tricky part, everything is in NeRF's camera coordinate system
    For up, since in COLMAP, the positive Y-axis is downward, and the world coordinate system is up-postive
    So the y value is negative, so it is negated in the load_llff_data(),
    so it is point up right now in NeRF's camera coordinate system
    For z, in COLMAP, the positive Z-axis is forward, in NeRF, the positive Z-axis is backward
    However, if we just put the positive value of Z into this function, it will directly point to the backward
    """
    # z: viewing direction, vec2, (3,)
    # up: up direction, Y, (3,)
    # pos: camera's location in world space, space, center, (3,)
    vec2 = normalize(z)
    # normalize the viewing direction to unit vector
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    """
    # compute the rightward X-axis direction of the camera, not very sure about whether right or left
    """
    # using cross product of up and viewing direction make sure this vector is perpendicular to both Y and Z
    # cross product: produce a perpendicular vector to the plane of the two input vectors
    vec1 = normalize(np.cross(vec2, vec0))
    # recompute the Y(up) direction of the camera, make sure this vector is perpendicular to both Z and X
    
    m = np.stack([vec0, vec1, vec2, pos], 1)
    # shape (3,4)
    # [[vec0_1, vec1_1, vec2_1, pos_1]
    #  [vec0_2, vec1_2, vec2_2, pos_2]
    #  [vec0_3, vec1_3, vec2_3, pos_3]]
    # so this one is actually not the homogeneous transformation matrix
    # it is only the camera axis' axis projection in world space
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):
    # poses: (N,3,5), all poses
    hwf = poses[0, :3, -1:]
    # shape: (3,1)
    # the forth column represent the camera translation value
    # Extracts the height, width, and focal length (hwf) from the first pose.
    # assume all camera have the same intrinsic parameters
    # take the last column of the first 3*5 matrix
    # [[[a2,  -a1,  a3,  a4*sc,  H]
    #   [a7,  -a6.  a8,  a9*sc,  W]
    #   [a12, -a11, a13, a14*sc, a15/factor]]
    center = poses[:, :3, 3].mean(0)
    # shape: (3,)
    # compute the average position/center of all poses
    # take all the 3*5 matrix's 4th column, and take the mean of them along the rows
    vec2 = normalize(poses[:, :3, 2].sum(0))
    # shape: (3,)
    # Computes the summation of Z(viewing direction) of all poses.
    # take the 3rd column of all 3*5 matrix, and sum them up along the rows, then normalize the sum
    up = poses[:, :3, 1].sum(0)
    # shape: (3,)
    # Computes the summation up direction Y(up) of all poses.
    # take the 2nd column of all 3*5 matrix, and sum them up along the rows
    
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    # this c2w matrix is the average camera position and its axis in the world space
    # shape (3,5)
    # [[vec0_1, vec1_1, vec2_1, pos_1, h]
    #  [vec0_2, vec1_2, vec2_2, pos_2, w]
    #  [vec0_3, vec1_3, vec2_3, pos_3, f]]
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):
    # poses: (N,3,5)
    # [[[a2,  -a1,  a3,  a4*sc,  H]
    #   [a7,  -a6.  a8,  a9*sc,  W]
    #   [a12, -a11, a13, a14*sc, a15/factor]]
    poses_ = poses+0
    # make a copy of poses
    bottom = np.reshape([0,0,0,1.], [1,4])
    # [[0,0,0,1]]
    c2w = poses_avg(poses)
    # c2w is the average camera position and its axis in the world space
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    # shape (4,4)
    # take the first 3*4 matrix of c2w, and concatenate the bottom to the last column
    # [[vec0_1, vec1_1, vec2_1, pos_1]
    #  [vec0_2, vec1_2, vec2_2, pos_2]
    #  [vec0_3, vec1_3, vec2_3, pos_3]
    #  [0,      0,      0,      1]]
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    # shape (N,1,4)
    # first reshape the bottom to (1,1,4), [[[0,0,0,1]]]
    # then tile (repeat) it to (N,1,4), [[[0,0,0,1]], [[0,0,0,1]], [[0,0,0,1]], ...]
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)
    # shape (N,4,4)
    # [[[a2,  -a1,  a3,  a4*sc]
    #   [a7,  -a6.  a8,  a9*sc]
    #   [a12, -a11, a13, a14*sc]
    #   [0,    0,    0,    1]]
    # take the first 3*4 matrix of poses, and concatenate all the new bottom to the last column
    poses = np.linalg.inv(c2w) @ poses
    # shape (N,4,4)
    # the inv change c2w to w2c, the do matrix multiplication with poses, 
    # the poses are transformed from world space to the average-centered camera space
    # [[[x1_c,  y1_c,  z1_c,  t1_c]
    #   [x2_c,  y2_c,  z2_c,  t2_c]
    #   [x3_c,  y3_c,  z3_c,  t3_c]
    #   [0,    0,    0,    1]]
    poses_[:,:3,:4] = poses[:,:3,:4]
    # shape (N,3,5)
    # [[[x1_c,  y1_c,  z1_c,  t1_c, H]
    #   [x2_c,  y2_c,  z2_c,  t2_c, W]
    #   [x3_c,  y3_c,  z3_c,  t3_c, f]]
    '''
    focal length is not changed
    '''
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    # poses: (N,3,5)
    # [[[a2,  -a1,  a3,  a4*sc,  H]
    #   [a7,  -a6.  a8,  a9*sc,  W]
    #   [a12, -a11, a13, a14*sc, a15/factor]]
    # bds: (2,N)
    # [[a16,a17], 
    #  [b16,b17], 
    #  [c16,c17], 
    #   .........] * 1 / (min(bds) * bd_factor)
    
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    # Converts a 3x4 pose matrix to a 4x4 homogeneous transformation matrix by adding a row [0, 0, 0, 1].
    
    rays_d = poses[:,:3,2:3]
    # shape (N,3,1)
    # ray direction
    # [[[a3]
    #   [a8]
    #   [a13]]
    rays_o = poses[:,:3,3:4]
    # ray origin
    # [[[a4*sc]
    #   [a9*sc]
    #   [a14*sc]]

    def min_line_dist(rays_o, rays_d):
        # find a point that is closest to all the camera rays, like a center
        # fancy math, not sure about the details
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        # shape (N,3,3)
        #[ 1  0  0 - [rd_x^2  rd_x*rd_y  rd_x*rd_z
        #  0  1  0    rd_y*rd_x  rd_y^2  rd_y*rd_z
        #  0  0  1]   rd_z*rd_x  rd_z*rd_y  rd_z^2]
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])
    # convert every camera pose to the new camera coordinate system, centered at the center

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    # get the radius of the sphere, the average distance from the center to all the camera positions
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    '''
    in here the hwf are all set to the first pose's hwf, need to be changed in later steps
    '''
    # shape (N,3,5)
    # take the first 3*4 matrix of poses_reset, and concatenate the first data's hwf tp the last column of all poses_reset
    # in here the hwf are all set to the first pose's hwf, need to be changed in later steps
    # [[[x1_c,  y1_c,  z1_c,  t1_c, H]
    #   [x2_c,  y2_c,  z2_c,  t2_c, W]
    #   [x3_c,  y3_c,  z3_c,  t3_c, f]]
    
    return poses_reset, new_poses, bds
    

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    

    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    """
    # In COLMAP, the coordinate is [Y(down), X(right), Z(foward)]
    # But in NeRF, the coordinate is [X(right), Y(up), Z(backward)]
    #
    # So we need to swap 1st and 2nd and multiply -1 to 1st
    #
    # However, the reason don't negate the 3rd is :
    # In COLMAP, the Z-axis represents depth and points forward from the camera.
	# In NeRF, the Z-axis still represents depth, but it is interpreted as negative along the viewing direction.
    #
    # Take the second row of each 5*N matrix, to the  first row
    # take the first two to second two, multiply by -1
    # and keep the reset the same
    """
    # [
    #    [[a2, b2, c2, ...],   [[a7,  b7,  c7, ...],   [[a12,  b12,  c12, ...],,
    #    [-a1,-b1,-c1, ...],  [-a6, -b6, -c6, ...],    [-a11, -b11, -c11, ...],
    #     [a3, b3, c3, ...],    [a8,  b8,  c8, ...],    [a13,  b13,  c13, ...],
    #     [a4, b4, c4, ...],    [a9,  b9,  c9, ...],    [a14,  b14,  c14, ...],
    #     [H,  H,  H, ...]],    [W,   W,   W, ...]],    [a15/factor,  b15/factor,  c15/factor, ...],
    # ]
    
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    # move the last axis to the first axis, so the shape will be (N, 3, 5)
    """
    # The first three columns are the [X(right), Y(up), Z(forward)] of camera's coordinate system in world space
    # The forth column is the translation vector of the camera (camera position) in world space
    """
    # [[[a2,  -a1,  a3,  a4,  H]
    #   [a7,  -a6.  a8,  a9,  W]
    #   [a12, -a11, a13, a14, a15/factor]]
    #
    # [[[b2,  -b1,  b3,  b4,  H]
    #   [b7,  -b6.  b8,  b9,  W]
    #   [b12, -b11, b13, b14, b15/factor]]
    #
    # [[[c2,  -c1,  c3,  c4,  H]
    #   [c7,  -c6.  c8,  c9,  W]
    #   [c12, -c11, c13, c14, c15/factor]]
    #    ...
    
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    # move the last axis to the first axis, so the shape will be (N, H, W, C)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    # move the last axis to the first axis, so the shape will be (N, 2)
    # [[a16,a17], 
    #  [b16,b17], 
    #  [c16,c17], 
    #   .........]
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    # if there is bd_factor input, then sc = 1 / (min(bds) * bd_factor)
    poses[:,:3,3] *= sc
    # This normalizes the scene scale so that the smallest bound is 1.0.
    # take each 3*5 Matrix's 4th column multiply by sc
    # the first column is the camera's x-axis projection in the world frame
    # the second column is the camera's y-axis projection in the world frame
    # the third column is the camera's z-axis projection in the world frame
    # the forth column is the camera's location in the world frame
    # the fifth column is the camera's intrinsic parameters, height, width, focal length
    # [[[a2,  -a1,  a3,  a4*sc,  H]
    #   [a7,  -a6.  a8,  a9*sc,  W]
    #   [a12, -a11, a13, a14*sc, a15/factor]]
    #
    # [[[b2,  -b1,  b3,  b4*sc,  H]
    #   [b7,  -b6.  b8,  b9*sc,  W]
    #   [b12, -b11, b13, b14*sc, b15/factor]]
    #
    # [[[c2,  -c1,  c3,  c4*sc,  H]
    #   [c7,  -c6.  c8,  c9*sc,  W]
    #   [c12, -c11, c13, c14*sc, c15/factor]]
    #    ...
    bds *= sc
    # multiply bds by sc
    # [[a16,a17], 
    #  [b16,b17], 
    #  [c16,c17], 
    #   .........] * 1 / (min(bds) * bd_factor)
    # so there will be an element value in bds is bd_factor
    
    if recenter:
        poses = recenter_poses(poses)
        # this is the poses in the average center camera's coordinate system
        # shape (N,3,5)
        # [[[x1_c,  y1_c,  z1_c,  t1_c, H]
        #   [x2_c,  y2_c,  z2_c,  t2_c, W]
        #   [x3_c,  y3_c,  z3_c,  t3_c, f]]
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        # recompute the average camera position and its axis after convert the camera location to the average center coordinate system
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))
        # recompute the average up direction of the camera
        

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    # recenter agrain
    # [[vec0_1, vec1_1, vec2_1, pos_1, h]
    #  [vec0_2, vec1_2, vec2_2, pos_2, w]
    #  [vec0_3, vec1_3, vec2_3, pos_3, f]]
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    # shape (N,)
    i_test = np.argmin(dists)
    #finds the index of the smallest distance between center and all cameras
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test



