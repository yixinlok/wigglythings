# wigglythings

running:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

todos:
- implement stvk or neo hookean. these have volume preservation


updates:
- concerns about using chang yues paper:
  - how would i make this work for FEM? need to modify the loss function (how?)
  - need to find a way to sample and interpolate the discrete
  - what happens if i start with points on grids instead of random points?
  - interpolation step
  
- moved stuff to warp
- make face selector for obj files for displaying instances
- finite differences for velocity then acceleration
- adjusted parameters for dyrt until it worked
- eigenvalues smaller than 0, just removed them and changed eigenvalue solver to look around 3 instead of 0
  

  - make a 3D grid
  - 

source:

notes:

tetwild terminal command:
./FloatTetwild_bin --input /Users/yixinlok/Desktop/empty2/loosecoil.obj --coarsen --manifold-surface


- when tetwilding, make sure the obj file is triangulated
- 
to cite:
- dyrt
- simkit
- tetwild
- https://viterbi-web.usc.edu/~jbarbic/cuda-uUq/WangBarbic-CUDA-MIG-2020.pdf
- https://arxiv.org/pdf/2408.10099