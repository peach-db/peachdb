```bash
conda create -y -n peachdb python=3.10
conda activate peachdb
pip install -r requirements.txt
```

# gRPC

```bash
python -m grpc_tools.protoc -I . --python_out=. --pyi_out=. --grpc_python_out=. api.proto
```

For Mac, had to additionally run,

```bash
pip uninstall grpcio
conda install grpcio
```
