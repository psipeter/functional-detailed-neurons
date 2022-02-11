# functional-detailed-neurons
Use an online learning rule to construct functional neural networks with biologically detailed neurons

Clone the repository
=======
  
  git clone https://github.com/psipeter/functional_detailed_neurons.git

  cd functional_detailed_neurons
    
Install virtual environment
=======

  pip3 install pipenv

  mkdir .venv

  pipenv --python=3.7

  pipenv shell


Install packages
=======
    
  pipenv install numpy scipy matplotlib seaborn pandas nengo nengolib neuron hyperopt


Compile NEURON channel mechanisms detailed_neurons
=======

  cd NEURON

  .venv/bin/nrnivmodl
