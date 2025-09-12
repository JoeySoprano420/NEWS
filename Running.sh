# Transpile NEWS → DGM
python transpiler.py examples/hello.news hello.dgm

# Run in VM
python vm.py hello.dgm

python transpiler.py examples/demo.news demo.dgm
python vm.py demo.dgm

