**Pywave**
======

## Description
**Pywave* is a open-source Python package for solving wave equations using various methods for educational purposes


## Reference
Chen, Y., O.M. Saad, M. Bai, X. Liu, and S. Fomel, 2021, A compact program for 3D passive seismic source-location imaging, Seismological Research Letters, 92, 3187–3201.

Chen et al., 202X, Pywave: a Python package for stuying wave equations, TBD

BibTeX:

	@Article{chen2021srl,
	  author={Yangkang Chen and Omar M. Saad and Min Bai and Xingye Liu and Sergey Fomel},
	  title = {A compact program for 3{D} passive seismic source-location imaging},
	  journal={Seismological Research Letters},
	  year=2021,
	  volume=92,
	  issue=5,
	  number=5,
	  pages={3187–3201},
	  doi={10.1785/0220210050},
	}
	
	@article{npre,
	  title={Pywave: a Python package for stuying wave equations},
	  author={Yangkang Chen and co-authors},
	  journal={TBD},
	  volume={TBD},
	  number={TBD},
	  issue={3},
	  pages={TBD},
	  doi={TBD},
	  year={TBD}
	}

-----------
## Copyright
    The pywave developing team, 2021-present
-----------

## License
    MIT License 

-----------

## Install
Using the latest version (Suggested)

    git clone https://github.com/chenyk1990/pywave
    cd pywave
    pip install -v -e .
    
or using Pypi (not available yet)

    pip install pywave

or (recommended, because we update very fast)

    pip install git+https://github.com/chenyk1990/pywave
    
-----------
## Examples
    The "demo" directory contains all runable scripts to demonstrate different applications of pywave. 

-----------
## Dependence Packages
* scipy 
* numpy 
* matplotlib

-----------
## Modules
    xxx.py  -> description
    
-----------
## Development
    The development team welcomes voluntary contributions from any open-source enthusiast. 
    If you want to make contribution to this project, feel free to contact the development team. 

-----------
## Contact
    Regarding any questions, bugs, developments, collaborations, please contact  
    Yangkang Chen
    chenyk2016@gmail.com

-----------
## Gallery
The gallery figures of the pywave package can be found at
    https://github.com/chenyk1990/gallery/tree/main/pywave
Each figure in the gallery directory corresponds to a DEMO script in the "demo" directory with the exactly the same file name.

These gallery figures are also presented below. 

# Below are several examples
Generated by [demos/test_first](https://github.com/chenyk1990/pywave/tree/main/demos/test_first.py)

<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/vel3d.png' alt='comp' width=960/>
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/data3d.png' alt='comp' width=640/>
</p>

Generated by [demos/test_second_wfd2ds](https://github.com/chenyk1990/pywave/tree/main/demos/test_second_wfd3ds.py)

<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/wfd3ds.gif' alt='comp' width=960/>
</p>

# Figures below are the reproduced results (using "Pywave") of the synthetic example presented in 
Chen, Y., O.M. Saad, M. Bai, X. Liu, and S. Fomel, 2021, A compact program for 3D passive seismic source-location imaging, Seismological Research Letters, 92, 3187–3201.

Grouped data: Generated by [demos/test_third_tri](https://github.com/chenyk1990/pywave/tree/main/demos/test_third_tri.py)
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/data3d-mask-0.png' alt='comp' width=960/>
</p>
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/data3d-mask-1.png' alt='comp' width=960/>
</p>
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/data3d-mask-2.png' alt='comp' width=960/>
</p>
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/data3d-mask-3.png' alt='comp' width=960/>
</p>

Time-reversed wavefields: Generated by [demos/test_third_tri](https://github.com/chenyk1990/pywave/tree/main/demos/test_third_tri.py)
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/wfd3ds-tri-0.gif' alt='comp' width=960/>
</p>
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/wfd3ds-tri-1.gif' alt='comp' width=960/>
</p>
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/wfd3ds-tri-2.gif' alt='comp' width=960/>
</p>
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/wfd3ds-tri-3.gif' alt='comp' width=960/>
</p>

Source-location image (three locations): Generated by [demos/test_third_tri](https://github.com/chenyk1990/pywave/tree/main/demos/test_third_tri.py)
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/imag3d-1.png' alt='comp' width=960/>
</p>
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/imag3d-2.png' alt='comp' width=960/>
</p>
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/imag3d-3.png' alt='comp' width=960/>
</p>




# Below are several 2D examples in case 3D computation is too expensive
Generated by [demos/test_mod2d](https://github.com/chenyk1990/pywave/tree/main/demos/test_mod2d.py)

<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/vel2d.png' alt='comp' width=960/>
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/data2d.png' alt='comp' width=640/>
</p>

Generated by [demos/test_mod2d_tri](https://github.com/chenyk1990/pywave/tree/main/demos/test_mod2d_tri.py)

<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/wfd2ds.gif' alt='comp' width=960/>
</p>

# Below is an example for 2D TRI imaging
Grouped data: Generated by [demos/test_mod2d_tri](https://github.com/chenyk1990/pywave/tree/main/demos/test_mod2d_tri.py)
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/data2d-1.png' alt='comp' width=960/>
</p>
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/data2d-2.png' alt='comp' width=960/>
</p>
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/data2d-3.png' alt='comp' width=960/>
</p>
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/data2d-4.png' alt='comp' width=960/>
</p>

Time-reversed wavefields: Generated by [demos/test_mod2d_tri](https://github.com/chenyk1990/pywave/tree/main/demos/test_mod2d_tri.py)
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/wfd2ds-0.gif' alt='comp' width=960/>
</p>
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/wfd2ds-1.gif' alt='comp' width=960/>
</p>
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/wfd2ds-2.gif' alt='comp' width=960/>
</p>
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/wfd2ds-3.gif' alt='comp' width=960/>
</p>

Source-location image (two locations in one image): Generated by [demos/test_mod2d_tri](https://github.com/chenyk1990/pywave/tree/main/demos/test_mod2d_tri.py)
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/location-new.png' alt='comp' width=960/>


# Below is an example for active-source simulation
Generated by [demos/test_mod2d_active](https://github.com/chenyk1990/pywave/tree/main/demos/test_mod2d_active.py)

<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/vel2d-active.png' alt='comp' width=960/>
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/data2d-active.png' alt='comp' width=640/>
</p>
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/wfd2ds-active.gif' alt='comp' width=960/>
</p>


# Below is an example for comparing the pseudo-spectral (PS) method and finite-difference (FD) method based on the same active-source experiment (as above)
Generated by [demos/test_mod2d_active_psVSfd](https://github.com/chenyk1990/pywave/tree/main/demos/test_mod2d_active_psVSfd.py)

The first is a data trace comparison, showing the stronger dispersion of FD
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/data2d-active-tracenew.png' alt='comp' width=960/>

The second is a wavefield trace comparison, showing the stronger dispersion of FD
<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/data2d-active-trace2new.png' alt='comp' width=960/>


# Below is an example for comparing the pseudo-spectral (PS) method and finite-difference (FD) method based on the same passive-source experiment (as the previous 2D one)
Generated by [demos/test_mod2d_tri_psVSfd](https://github.com/chenyk1990/pywave/tree/main/demos/test_mod2d_tri_psVSfd.py)

Here is a masked data comparison (with a better focus), showing the stronger dispersion of FD

<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/data2d-psfd-1.gif' alt='comp' width=960/>


# Below is an example of full waveform inversion of passive seismic data (the sources are indicated on the top velocity model)
Generated by [demos/test_pfwi_vel2d](https://github.com/chenyk1990/pywave/tree/main/demos/test_pfwi_vel2d.py)

<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/test_pfwi_vel2d_onenew.png' alt='comp' width=960/>

# Below is an example of generating shot gathers for full waveform inversion of active-source seismic data 
Generated by [demos/test_fwi_mod2d](https://github.com/chenyk1990/pywave/tree/main/demos/test_fwi_mod2d.py)

<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/test_pfwi_vel2d_data.png' alt='comp' width=960/>


# Below is an example of full waveform inversion of active-source seismic data 
Generated by [demos/test_fwi_vel2d](https://github.com/chenyk1990/pywave/tree/main/demos/test_fwi_vel2d.py)

<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/test_fwi_vel2d_vel.png' alt='comp' width=960/>


# Below is the same example as above but with more iterations
Generated by [demos/test_fwi_vel2d_niter100.py](https://github.com/chenyk1990/pywave/tree/main/demos/test_fwi_vel2d_niter100.py)

<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/test_fwi_vel2d_vel-100.png' alt='comp' width=960/>

# Below is the same example as above, with a comparison of data fitting between initial model and FWI inverted model
Generated by [demos/test_fwi_vel2d_niter100.py](https://github.com/chenyk1990/pywave/tree/main/demos/test_fwi_vel2d_niter100.py)

<p align="center">
<img src='https://github.com/chenyk1990/gallery/blob/main/pywave/test_fwi_vel2d_datacomp.png' alt='comp' width=960/>

