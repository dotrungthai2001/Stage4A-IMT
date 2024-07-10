import torch
import numpy as np
from simulatevaisseau import generate_random_shape, simulate_circle, thicken_curve
from helper_functions import bilinear_interpolation,bilinear_interpolation_torch
import multiprocessing
from functools import partial
import random


dtype = torch.float32
torch.set_default_dtype(dtype )

def generate_image(N=256,seed = None, n_curve_min=1,n_curve_max=5,n_control_min=5,n_control_max=8,n_pts_curve=200,
                   line_width_min=1,line_width_max=3,
                   n_circle_min=2, n_circle_max=10, n_pts_circle=50,r_circle_min=2,r_circle_max=7):
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # Generate a clean image
    n = np.random.randint(n_control_min, n_control_max)
    x = []
    y = []
    nb_cou = np.random.randint(n_curve_min, n_curve_max)
    nb_cir = np.random.randint(n_circle_min, n_circle_max)
    for i in range(nb_cou):
        line_width = np.random.randint(line_width_min, line_width_max)
        xi, yi = generate_random_shape(n, n_pts_curve, 'remove')
        xi,yi = thicken_curve(xi, yi, line_width, N)
        x = np.concatenate((x, xi))
        y = np.concatenate((y, yi))
        
    for i in range(nb_cir):
        xi, yi = simulate_circle(n_pts_circle, n_pts_circle, r_min=r_circle_min/N, r_max=r_circle_max/N)
        x = np.concatenate((x, xi))
        y = np.concatenate((y, yi))
    
    # x = torch.tensor(x, dtype = dtype)
    # y = torch.tensor(y, dtype = dtype)
    # u = torch.ones(x.shape[0],dtype = dtype)
    u = np.ones(x.shape[0])
    interpolated_array = bilinear_interpolation(u, x, y, N, N)
    return interpolated_array 
    # clean_image = Image.fromarray((interpolated_array * 255).astype('uint8'))
    # X[j,0] = torch.tensor(interpolated_array,dtype = dtype)
    
def generate_sample(index, seed = None, N=256,n_curve_min=1,n_curve_max=5,n_control_min=5,n_control_max=8,n_pts_curve=200,
                   line_width_min=1,line_width_max=3,
                   n_circle_min=2, n_circle_max=10, n_pts_circle=50,r_circle_min=2,r_circle_max=7):
    
    sample = generate_image(N=N, seed = seed, n_curve_min=n_curve_min,n_curve_max=n_curve_max,n_control_min=n_control_min,
                            n_control_max=n_control_max,n_pts_curve=n_pts_curve,
                       line_width_min=line_width_min,line_width_max=line_width_max,
                       n_circle_min=n_circle_min, n_circle_max=n_circle_max, n_pts_circle=n_pts_circle,
                       r_circle_min=r_circle_min,r_circle_max=r_circle_max)
    
    return sample, index

def generate_batch(batch_size,N=256,n_curve_min=1,n_curve_max=5,n_control_min=5,n_control_max=8,n_pts_curve=200,
                    line_width_min=1,line_width_max=3,
                    n_circle_min=2, n_circle_max=10, n_pts_circle=50,r_circle_min=2,r_circle_max=7,pool=None):
    
    X = torch.zeros((batch_size,1,N,N))
    
    # Set random seed for the main process (optional)
    main_process_seed = 42  # You can change this value or set it to None
    torch.manual_seed(main_process_seed)
    np.random.seed(main_process_seed)
    random.seed(main_process_seed)
    
    # Use multiprocessing to parallelize sample generation
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    # generate_sample_partial = partial(generate_sample,N=N,n_curve_min=n_curve_min,n_curve_max=n_curve_max,n_control_min=n_control_min,
    #                         n_control_max=n_control_max,n_pts_curve=n_pts_curve,
    #                         line_width_min=line_width_min,line_width_max=line_width_max,
    #                         n_circle_min=n_circle_min, n_circle_max=n_circle_max, n_pts_circle=n_pts_circle,
    #                         r_circle_min=r_circle_min,r_circle_max=r_circle_max)
    
    seeds = [main_process_seed + i for i in range(batch_size)]
    # samples_and_indices = pool.starmap(generate_sample_partial, zip(range(batch_size), seeds))
    samples_and_indices = pool.starmap(generate_sample, [(i, seeds[i], N, n_curve_min,n_curve_max,n_control_min,n_control_max,n_pts_curve,
                       line_width_min,line_width_max, n_circle_min, n_circle_max, n_pts_circle, r_circle_min, r_circle_max)
                                                     for i in range(batch_size)])
    
    for sample, index in samples_and_indices:
        # print(index)
        # plt.imshow(sample)
        # plt.show()
        # plt.title('sample %d' % index)
        X[index,0] = torch.tensor(sample, dtype=dtype)

    return X 
