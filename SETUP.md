```bash
conda create -y -n peachdb python=3.10
conda activate peachdb
pip install -r requirements.txt
```

# gRPC

```bash
cd peachdb_grpc
python -m grpc_tools.protoc -I . --python_out=. --pyi_out=. --grpc_python_out=. api.proto
```

For Mac, had to additionally run,

```bash
pip uninstall grpcio
conda install grpcio
```

# Deployment

```bash
ngrok tcp --region=us --remote-addr=1.tcp.ngrok.io:24448 50051
```
